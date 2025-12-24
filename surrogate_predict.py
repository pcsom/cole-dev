import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from scipy import stats
from scipy.stats import t
import os
from embedding_config import FORCE_RERUN_EXPERIMENTS

# Configuration
INPUT_PATH = '/storage/ice-shared/vip-vvk/data/AOT/psomu3/codenas/nasbench201_corpus_embedded_half.pkl'
OUTPUT_PATH = '/storage/ice-shared/vip-vvk/data/AOT/psomu3/codenas/mlp_prediction_results_cv.csv'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
RANDOM_STATE = 42

"""
STATISTICAL METHODOLOGY
========================================================

Previous (Invalid) Approach:
  - Averaged 5 folds → computed stats across seeds
  - This collapsed within-CV variance, measuring RNG variance instead of model variance
  
Current (Valid) Approach (Nadeau & Bengio, 2003):
  - Store ALL individual fold results (5 folds × 20 seeds = 100 data points)
  - Use Corrected Paired T-Test to account for non-independent training sets
  - The correction factor adjusts for overlapping data: (1/n + n_test/n_train)
  
Result Format:
  - Each row = 1 fold result (input_type, split_config, random_state, fold_idx, metrics)
  - Total rows per model: 5 folds × 20 seeds = 100 rows
  - Statistical tests operate on raw fold data, NOT aggregated means
"""

class ArchitecturePredictor(nn.Module):
    """
    3-Layer MLP to predict architecture metrics (Accuracy and Loss) from embeddings.
    """
    def __init__(self, input_dim, hidden_dims=[512, 256, 128], output_dim=2):
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        # Hidden layers
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            prev_dim = dim
            
        # Output layer (Predicts [Accuracy, Loss])
        layers.append(nn.Linear(prev_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def train_single_fold(X, y, train_idx, val_idx, epochs=100, batch_size=32, lr=1e-3):
    """
    Train and evaluate MLP for a specific set of indices.
    """
    # Select data based on indices
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    # Scale targets
    scaler = StandardScaler()
    y_train_scaled = scaler.fit_transform(y_train)
    y_val_scaled = scaler.transform(y_val)
    
    # Convert to Tensors
    X_train_t = torch.FloatTensor(X_train).to(DEVICE)
    y_train_t = torch.FloatTensor(y_train_scaled).to(DEVICE)
    X_val_t = torch.FloatTensor(X_val).to(DEVICE)
    y_val_t = torch.FloatTensor(y_val_scaled).to(DEVICE)
    
    # Setup Model
    model = ArchitecturePredictor(input_dim=X.shape[1]).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # Training Loop
    model.train()
    loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)
    
    for _ in range(epochs):
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            preds = model(batch_X)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            
    # Evaluation
    model.eval()
    with torch.no_grad():
        val_preds_scaled = model(X_val_t).cpu().numpy()
        
    # Inverse transform predictions
    val_preds = scaler.inverse_transform(val_preds_scaled)
    
    # Calculate Metrics (Validation only for CV aggregation)
    metrics = {
        'val_r2_acc': r2_score(y_val[:, 0], val_preds[:, 0]),
        'val_mse_acc': mean_squared_error(y_val[:, 0], val_preds[:, 0]),
        'val_r2_loss': r2_score(y_val[:, 1], val_preds[:, 1]),
        'val_mse_loss': mean_squared_error(y_val[:, 1], val_preds[:, 1])
    }
    
    return metrics

def corrected_paired_ttest(diffs, n_train, n_test, n_splits, n_repeats):
    """
    Computes the Corrected Paired t-test (Nadeau & Bengio, 2003)
    
    Args:
        diffs (array-like): Differences between Model A and Model B for every fold (length = n_splits * n_repeats)
        n_train (int): Size of training set in a single fold
        n_test (int): Size of test set in a single fold
        n_splits (int): Number of folds (K)
        n_repeats (int): Number of times CV was repeated (seeds)
    
    Returns:
        t_statistic, p_value (two-tailed)
    """
    # 1. Compute basic statistics
    mean_diff = np.mean(diffs)
    var_diff = np.var(diffs, ddof=1)  # Variance of the differences
    
    n_total_runs = len(diffs)  # Should be n_splits * n_repeats
    
    # 2. The Correction Factor
    # (1/n) accounts for the sample size variance
    # (n_test/n_train) accounts for the correlation due to overlapping training data
    correction = (1 / n_total_runs) + (n_test / n_train)
    
    # 3. Compute t-statistic and p-value
    # Standard Error = sqrt(Correction * Variance)
    se = np.sqrt(correction * var_diff)
    
    if se == 0:
        return 0.0, 1.0  # Avoid division by zero
        
    t_stat = mean_diff / se
    
    # Degrees of freedom = n - 1
    df = n_total_runs - 1
    p_value = t.sf(np.abs(t_stat), df) * 2  # Two-sided p-value
    
    return t_stat, p_value

def run_experiments(random_state=42, embedding_filter=None):
    """
    Main execution pipeline: Load data -> Loop inputs -> Loop splits (CV) -> Save results.
    
    Args:
        random_state: Random seed for reproducibility
        embedding_filter: List of embedding column names or model names to filter. 
                         If None, uses all embeddings. Can specify full column names or model names.
    
    Returns:
        DataFrame with results
    """
    # Load Corpus
    df = pd.read_pickle(INPUT_PATH)
    
    # Identify Embedding Columns
    all_embedding_cols = [col for col in df.columns if 'embedding' in col]
    
    # Filter embeddings if specified
    if embedding_filter is not None:
        filtered_cols = []
        for emb_filter in embedding_filter:
            # Check if it's a full column name
            if emb_filter in all_embedding_cols:
                filtered_cols.append(emb_filter)
            else:
                # Otherwise treat as model name and find matching columns
                matching = [col for col in all_embedding_cols if emb_filter in col]
                filtered_cols.extend(matching)
        embedding_cols = list(set(filtered_cols))  # Remove duplicates
        if not embedding_cols:
            raise ValueError(f"No embeddings found matching filter: {embedding_filter}")
    else:
        embedding_cols = all_embedding_cols
    
    # Prepare Targets
    target_acc = df['cifar10-valid_valid_accuracy'].values
    target_loss = df['cifar10-valid_valid_loss'].values
    y = np.column_stack((target_acc, target_loss))
    
    results = []
    
    # Initialize KFold with provided random state
    kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
    
    # Iterate over every embedding type
    for emb_col in embedding_cols:
        # Convert list-in-column to numpy array
        X = np.vstack(df[emb_col].values)
        
        # Iterate over split configurations
        # Config 1: 80% Train, 20% Val (Standard CV)
        # Config 2: 20% Train, 80% Val (Inverted CV)
        split_configs = [
            {'name': '80_20', 'train_on_fold': False}, 
            {'name': '20_80', 'train_on_fold': True}
        ]
        
        for config in split_configs:
            split_name = config['name']
            train_on_fold = config['train_on_fold']
            print(f"  > Running 5-Fold CV for split: {split_name}")
            
            fold_metrics = []
            
            # Run 5-Fold CV
            for fold_idx, (idx_a, idx_b) in enumerate(kf.split(X)):
                if train_on_fold:
                    # 20/80 Split: Train on small fold (idx_b), Val on large fold (idx_a)
                    train_idx, val_idx = idx_b, idx_a
                else:
                    # 80/20 Split: Train on large fold (idx_a), Val on small fold (idx_b)
                    train_idx, val_idx = idx_a, idx_b
                
                # Train and get metrics
                metrics = train_single_fold(X, y, train_idx, val_idx)
                
                # Store INDIVIDUAL fold result (not aggregated)
                result_row = {
                    'input_type': emb_col,
                    'split_config': split_name,
                    'random_state': random_state,
                    'fold_idx': fold_idx,  # Track which fold this was
                    **metrics  # Unpack the metrics directly (no _mean/_std suffixes)
                }
                results.append(result_row)
            
    # Create Results DataFrame
    results_df = pd.DataFrame(results)
    return results_df

def run_multiple_seeds_experiment(
    n_seeds=20,
    output_path='/storage/ice-shared/vip-vvk/data/AOT/psomu3/codenas/mlp_prediction_results_multi_seed.csv',
    stats_path='/storage/ice-shared/vip-vvk/data/AOT/psomu3/codenas/mlp_prediction_stats_comparison.csv',
    embedding_filter=None,
    compare_models=None,
    force=None
):
    """
    Run the experiment multiple times with different random seeds and perform statistical comparison.
    
    Args:
        n_seeds: Number of different random seeds to run
        output_path: Path to save all individual results
        stats_path: Path to save statistical comparison results
        embedding_filter: List of embedding column names or model names to filter.
                         If None, uses all embeddings.
        compare_models: List of two model names to compare (e.g., ['modernbert', 'deepseek_coder']).
                       If None and exactly 2 models in filter, compares those. Otherwise no comparison.
        force: If True, force rerun even if results exist. 
               If None, uses FORCE_RERUN_EXPERIMENTS from config.
    
    Returns:
        Tuple of (all_results_df, stats_df)
    """
    if force is None:
        force = FORCE_RERUN_EXPERIMENTS
    
    print(f"{'='*80}")
    print(f"Starting Multi-Seed Experiments (n={n_seeds})")
    print(f"{'='*80}\n")
    
    print(f"Loading corpus from {INPUT_PATH}...")
    df = pd.read_pickle(INPUT_PATH)
    print(f"Loaded {len(df)} architectures")
    
    # Identify embedding columns
    all_embedding_cols = [col for col in df.columns if 'embedding' in col]
    
    # Apply filter
    if embedding_filter is not None:
        filtered_cols = []
        for emb_filter in embedding_filter:
            if emb_filter in all_embedding_cols:
                filtered_cols.append(emb_filter)
            else:
                matching = [col for col in all_embedding_cols if emb_filter in col]
                filtered_cols.extend(matching)
        embedding_cols = list(set(filtered_cols))
        if not embedding_cols:
            raise ValueError(f"No embeddings found matching filter: {embedding_filter}")
    else:
        embedding_cols = all_embedding_cols
    
    print(f"Found {len(embedding_cols)} embedding types: {embedding_cols}\n")
    
    # Check if results already exist
    existing_results = None
    seeds_to_run = list(range(n_seeds))
    
    if os.path.exists(output_path) and not force:
        print(f"Found existing results at {output_path}")
        existing_results = pd.read_csv(output_path)
        
        # Check which embeddings already have results
        existing_embeddings = set(existing_results['input_type'].unique())
        requested_embeddings = set(embedding_cols)
        
        already_computed = requested_embeddings.intersection(existing_embeddings)
        need_to_compute = requested_embeddings - existing_embeddings
        
        # Initialize dict to track embeddings that need more seeds
        embeddings_need_more_seeds = {}
        
        if already_computed:
            print(f"\nAlready computed embeddings: {already_computed}")
            
            # For embeddings that exist, check how many seeds we have
            for emb in already_computed:
                emb_data = existing_results[existing_results['input_type'] == emb]
                existing_seeds = set(emb_data['random_state'].unique())
                n_existing = len(existing_seeds)
                
                if n_existing < n_seeds:
                    # Need more seeds for this embedding
                    needed_seeds = [s for s in range(n_seeds) if s not in existing_seeds]
                    embeddings_need_more_seeds[emb] = needed_seeds
                    print(f"  {emb}: Has {n_existing} seeds, needs {len(needed_seeds)} more to reach {n_seeds}")
                elif n_existing >= n_seeds:
                    print(f"  {emb}: Has {n_existing} seeds (>= {n_seeds} requested), will reuse")
            
            # Determine final strategy
            embeddings_with_enough_seeds = already_computed - set(embeddings_need_more_seeds.keys())
            
            if embeddings_with_enough_seeds:
                print(f"\nEmbeddings with sufficient seeds: {embeddings_with_enough_seeds}")
            
            if embeddings_need_more_seeds:
                print(f"\nEmbeddings needing more seeds: {set(embeddings_need_more_seeds.keys())}")
                # We'll run additional seeds for these embeddings
                need_to_compute = need_to_compute.union(set(embeddings_need_more_seeds.keys()))
        
        if not need_to_compute:
            print(f"\nAll requested embeddings already have sufficient results!")
            print(f"Using existing results (set force=True or FORCE_RERUN_EXPERIMENTS=True to recompute)")
            
            # Filter existing results to requested embeddings and first n_seeds
            all_results_df = existing_results[
                existing_results['input_type'].isin(embedding_cols)
            ]
            # Keep only first n_seeds for each embedding
            filtered_results = []
            for emb in embedding_cols:
                emb_data = all_results_df[all_results_df['input_type'] == emb]
                unique_seeds = sorted(emb_data['random_state'].unique())[:n_seeds]
                filtered_results.append(emb_data[emb_data['random_state'].isin(unique_seeds)])
            all_results_df = pd.concat(filtered_results, ignore_index=True)
        else:
            print(f"\nNeed to compute: {need_to_compute}")
            print(f"Will run experiments for new/incomplete embeddings")
            
            # Determine which seeds to run for each embedding
            seeds_per_embedding = {}
            for emb in need_to_compute:
                if emb in embeddings_need_more_seeds:
                    # Run only the missing seeds
                    seeds_per_embedding[emb] = embeddings_need_more_seeds[emb]
                else:
                    # New embedding, run all seeds
                    seeds_per_embedding[emb] = list(range(n_seeds))
            
            # Store for later use
            embedding_cols_to_compute = list(need_to_compute)
    elif os.path.exists(output_path) and force:
        print(f"Found existing results at {output_path}")
        print(f"WARNING: Force flag is True, will recompute all results")
        existing_results = None
        seeds_per_embedding = None
        embedding_cols_to_compute = embedding_cols  # Compute all requested embeddings
    else:
        seeds_per_embedding = None
        embedding_cols_to_compute = embedding_cols  # No existing results, compute all
    
    # Run experiments for each seed (only for embeddings that need computation)
    all_results = []
    
    if 'embedding_cols_to_compute' in locals() and embedding_cols_to_compute:
        # We have embeddings to compute
        print(f"\nRunning experiments for {len(embedding_cols_to_compute)} embedding type(s)...")
        
        if seeds_per_embedding is not None:
            # Run specific seeds for each embedding
            for emb in embedding_cols_to_compute:
                seeds_for_this_emb = seeds_per_embedding[emb]
                print(f"\n{emb}: Running {len(seeds_for_this_emb)} seed(s): {seeds_for_this_emb}")
                
                for seed in tqdm(seeds_for_this_emb, desc=f"Seeds for {emb.split('_')[0]}"):
                    results_df = run_experiments(random_state=seed, embedding_filter=[emb])
                    all_results.append(results_df)
        else:
            # Run all seeds for all embeddings
            for seed in tqdm(range(n_seeds), desc="Running experiments with different seeds"):
                results_df = run_experiments(random_state=seed, embedding_filter=embedding_cols_to_compute)
                all_results.append(results_df)
        
        # new_results_df = the NEW results we just computed
        new_results_df = pd.concat(all_results, ignore_index=True)
        
        # Merge with existing results - ONLY ADD, NEVER DELETE
        if existing_results is not None and not force:
            # Strategy: KEEP ALL OLD RESULTS, just avoid duplicates
            # (duplicates = results we just computed that are also in existing_results)
            keep_existing_parts = []
            
            for emb in existing_results['input_type'].unique():
                emb_existing = existing_results[existing_results['input_type'] == emb]
                
                if emb in seeds_per_embedding:
                    # This embedding: we just added some NEW seeds
                    # OLD seeds from CSV: KEEP THEM ALL
                    # NEW seeds (in new_results_df): skip them here to avoid duplicates
                    newly_added_seeds = set(seeds_per_embedding[emb])
                    old_seeds_to_preserve = emb_existing[~emb_existing['random_state'].isin(newly_added_seeds)]
                    if len(old_seeds_to_preserve) > 0:
                        keep_existing_parts.append(old_seeds_to_preserve)
                elif emb in embedding_cols_to_compute:
                    # This embedding: completely new, didn't exist in CSV before
                    # Nothing to keep from old CSV (there was nothing there)
                    pass
                else:
                    # This embedding: we didn't touch it at all this run
                    # KEEP EVERYTHING from old CSV
                    keep_existing_parts.append(emb_existing)
            
            # Final result = ALL old results we're preserving + ALL new results we computed
            if keep_existing_parts:
                keep_existing = pd.concat(keep_existing_parts, ignore_index=True)
                all_results_df = pd.concat([keep_existing, new_results_df], ignore_index=True)
                print(f"\nMerged {len(new_results_df)} new results with {len(keep_existing)} existing results")
            else:
                all_results_df = new_results_df
        else:
            all_results_df = new_results_df
    # else: all_results_df was already set from existing results
    
    print(f"\n{'='*80}")
    print("Aggregating Results and Computing Statistics")
    print(f"{'='*80}\n")
    
    # Save all results
    print(f"Saving all results to {output_path}")
    all_results_df.to_csv(output_path, index=False)
    
    # Get the embedding columns that were requested for this run
    # Use embedding_cols (requested) not final results (which includes all merged data)
    requested_embedding_cols = embedding_cols
    
    # Determine which models to compare
    if compare_models is None:
        # Try to auto-detect if exactly 2 different model prefixes from REQUESTED embeddings
        model_prefixes = set()
        for col in requested_embedding_cols:
            # Extract model name (before code type suffix)
            # Handle special cases with underscores in model names
            if 'deepseek_coder' in col:
                model_prefixes.add('deepseek_coder')
            elif 'codellama_python' in col:
                model_prefixes.add('codellama_python')
            elif 'modernbert' in col:
                model_prefixes.add('modernbert')
            elif 'codebert' in col:
                model_prefixes.add('codebert')
            else:
                # Generic: take everything before the code type
                # Remove common suffixes
                for suffix in ['_pytorch_code_embedding', '_onnx_code_embedding', '_grammar_code_embedding']:
                    if col.endswith(suffix):
                        model_prefixes.add(col[:-len(suffix)])
                        break
        
        if len(model_prefixes) == 2:
            compare_models = list(model_prefixes)
            print(f"\nAuto-detected models to compare: {compare_models}")
        else:
            print(f"\nSkipping statistical comparison (found {len(model_prefixes)} models, need exactly 2)")
            print(f"Models found: {model_prefixes}")
            print(f"Tip: Use embedding_filter to request exactly 2 models for comparison")
            return all_results_df, None
    
    # Perform statistical comparison between models
    print(f"\nPerforming statistical comparison between {compare_models[0]} and {compare_models[1]}")
    stats_results = []
    
    # Get code types (pytorch, onnx, grammar)
    code_types = ['pytorch_code', 'onnx_code', 'grammar_code']
    
    model1_name, model2_name = compare_models
    
    # For each split configuration and code type, compare models
    for split_config in ['80_20', '20_80']:
        for code_type in code_types:
            # Filter for this split and code type
            model1_col = f'{model1_name}_{code_type}_embedding'
            model2_col = f'{model2_name}_{code_type}_embedding'
            
            # Check if columns exist
            model1_data = all_results_df[
                (all_results_df['input_type'] == model1_col) & 
                (all_results_df['split_config'] == split_config)
            ]
            
            model2_data = all_results_df[
                (all_results_df['input_type'] == model2_col) & 
                (all_results_df['split_config'] == split_config)
            ]
            
            if len(model1_data) == 0 or len(model2_data) == 0:
                continue  # Skip if either model doesn't have this combination
            
            # For each metric, perform corrected t-test
            # Note: metric_cols no longer have _mean suffix since we're using raw fold data
            base_metrics = ['val_r2_acc', 'val_mse_acc', 'val_r2_loss', 'val_mse_loss']
            
            for metric_col in base_metrics:
                if metric_col not in model1_data.columns or metric_col not in model2_data.columns:
                    continue
                    
                model1_values = model1_data[metric_col].values
                model2_values = model2_data[metric_col].values
                
                # Calculate differences for corrected paired t-test
                diffs = model1_values - model2_values
                
                # Determine n_train/n_test based on split config
                total_n = 7812  # Total dataset size
                if split_config == '80_20':
                    n_train = int(total_n * 0.8)
                    n_test = int(total_n * 0.2)
                else:  # '20_80'
                    n_train = int(total_n * 0.2)
                    n_test = int(total_n * 0.8)
                
                # PERFORM CORRECTED T-TEST (Nadeau & Bengio, 2003)
                t_stat, p_value = corrected_paired_ttest(
                    diffs,
                    n_train=n_train,
                    n_test=n_test,
                    n_splits=5,
                    n_repeats=n_seeds
                )
                
                # Calculate means and std
                model1_mean = np.mean(model1_values)
                model1_std = np.std(model1_values)
                model2_mean = np.mean(model2_values)
                model2_std = np.std(model2_values)
                
                # Determine winner (higher is better for R2, lower is better for MSE)
                if 'r2' in metric_col:
                    winner = model1_name if model1_mean > model2_mean else model2_name
                elif 'mse' in metric_col:
                    winner = model1_name if model1_mean < model2_mean else model2_name
                else:
                    winner = 'N/A'
                
                # Check significance
                is_significant = p_value < 0.05
                
                stats_results.append({
                    'split_config': split_config,
                    'code_type': code_type,
                    'metric': metric_col,
                    f'{model1_name}_mean': model1_mean,
                    f'{model1_name}_std': model1_std,
                    f'{model2_name}_mean': model2_mean,
                    f'{model2_name}_std': model2_std,
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'significant': is_significant,
                    'winner': winner
                })
    
    stats_df = pd.DataFrame(stats_results)
    
    # Save statistics
    print(f"Saving statistical comparison to {stats_path}")
    stats_df.to_csv(stats_path, index=False)
    
    # Print summary
    print(f"\n{'='*80}")
    print("Statistical Comparison Summary")
    print(f"{'='*80}\n")
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.precision', 4)
    
    # Print significant differences
    sig_results = stats_df[stats_df['significant'] == True]
    print(f"Significant differences found: {len(sig_results)} out of {len(stats_df)}")
    print("\nSignificant Results:")
    display_cols = ['split_config', 'code_type', 'metric', 'p_value', 'winner']
    # Add mean columns dynamically
    mean_cols = [col for col in stats_df.columns if col.endswith('_mean')]
    display_cols = ['split_config', 'code_type', 'metric'] + mean_cols + ['p_value', 'winner']
    print(sig_results[display_cols])
    
    print(f"\n{'='*80}")
    print("Winner Summary (Significant Results Only)")
    print(f"{'='*80}\n")
    winner_counts = sig_results['winner'].value_counts()
    print(winner_counts)
    
    return all_results_df, stats_df

if __name__ == "__main__":
    # Check Device
    print(f"Using device: {DEVICE}")
    if DEVICE == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}\n")
    
    # Run multi-seed experiment with statistical comparison
    all_results_df, stats_df = run_multiple_seeds_experiment(n_seeds=20)