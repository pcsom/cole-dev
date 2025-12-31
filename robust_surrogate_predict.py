"""
Robust surrogate prediction with corrected statistical testing.
Implements subsampled repeated k-fold CV with corrected paired t-test.

Methodology:
  - "Split First, Subsample Later": K-fold creates train_pool/test (e.g., 6250/1562),
    then subsample exactly N (15, 50, 150, etc.) from train_pool for training
  - Constant test size ensures stable evaluation even with tiny training sets
  - Target scaling (mean=0, std=1) for balanced gradients, NO feature scaling
  - Primary Metric: Kendall's Tau (rank correlation)

Based on Nadeau & Bengio (2003): Inference for the Generalization Error
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import scipy.stats as stats
from scipy.stats import kendalltau
from typing import List, Dict, Optional, Tuple
import os
from results_io import save_per_embedding_results, save_comparison_results, split_results_for_saving, load_existing_trials
from stat_utils import corrected_paired_ttest, subsampled_repeated_kfold_comparison

class MLPSurrogate(nn.Module):
    """3-layer MLP for architecture performance prediction."""
    
    def __init__(self, input_dim, hidden_dims=[512, 256, 128], output_dim=2, dropout=0.1):
        super(MLPSurrogate, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LeakyReLU(negative_slope=0.01),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


def train_model_on_subsample(
    X_train_pool, y_train_pool, X_test, y_test,
    sample_size, random_state, epochs=100, lr=0.001, batch_size=16, device='cuda', patience=20
):
    """
    Train a model on a subsample of the training pool.
    Implements "Split First, Subsample Later" methodology:
      1. K-fold creates train_pool and test (done by caller)
      2. We subsample exactly sample_size from train_pool
      3. Train on the full subsample for fixed epochs
      4. Test on the full test fold for stable evaluation
    
    Args:
        X_train_pool: Full training pool features (from K-fold split)
        y_train_pool: Full training pool targets
        X_test: Test fold features (constant size for stable evaluation)
        y_test: Test fold targets
        sample_size: EXACT number of training samples (15, 50, 150, etc.)
        random_state: Random seed
        epochs: Training epochs (fixed, no early stopping)
        lr: Learning rate
        batch_size: Batch size
        device: Device to use
        patience: (Unused, kept for API compatibility)
    
    Preprocessing:
        - Features (embeddings): NOT scaled (sensitive distributions)
        - Targets (accuracy/loss): Scaled to mean=0, std=1 for balanced gradients
    
    Returns:
        Dictionary with performance metrics
    """
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    
    # Subsample from training pool
    n_pool = len(X_train_pool)
    if sample_size > n_pool:
        raise ValueError(f"sample_size ({sample_size}) > pool size ({n_pool})")
    
    indices = np.random.choice(n_pool, size=sample_size, replace=False)
    X_train = X_train_pool[indices]
    y_train = y_train_pool[indices]
    
    # CRITICAL: Split First, Subsample Later methodology
    # The subsample IS the full training set. We do NOT split it further.
    # This ensures:
    #   1. Training size is exactly what's specified (15, 50, 150, etc.)
    #   2. Test set is constant size (full fold) for stable evaluation
    #   3. We don't waste data - use all available test samples
    
    # Scale TARGETS (not features - embeddings have sensitive distributions)
    target_scaler = StandardScaler()
    y_train_scaled = target_scaler.fit_transform(y_train)
    y_test_scaled = target_scaler.transform(y_test)
    
    # Convert to tensors (features are NOT scaled)
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train_scaled).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)
    y_test_t = torch.FloatTensor(y_test_scaled).to(device)
    
    # Create model
    input_dim = X_train.shape[1]
    model = MLPSurrogate(input_dim=input_dim).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # Training loop (fixed epochs, no early stopping)
    model.train()
    for epoch in range(epochs):
        # Mini-batch training
        for i in range(0, len(X_train_t), batch_size):
            batch_X = X_train_t[i:i+batch_size]
            batch_y = y_train_t[i:i+batch_size]
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
    
    # Evaluation on test set
    model.eval()
    with torch.no_grad():
        # Test set predictions (in scaled space)
        test_pred_scaled = model(X_test_t).cpu().numpy()
        
        # Inverse transform to original scale
        test_pred = target_scaler.inverse_transform(test_pred_scaled)
        test_true = y_test
        
        # Calculate metrics (for accuracy prediction - 2nd column)
        acc_pred = test_pred[:, 1]
        acc_true = test_true[:, 1]
        
        # 1. Check for NaNs in output (Gradient explosion)
        if np.any(np.isnan(acc_pred)):
            print(f"  WARNING: NaNs in prediction. Assigning Tau=0.0.")
            return { 'kendall_tau': 0.0, 'mse': 999.0, 'n_train': sample_size, 'n_test': len(X_test) }
        
        pred_variance = np.var(acc_pred)
        true_variance = np.var(acc_true)

        # 2. Check for Soft Collapse (Variance is tiny but not zero)
        pred_variance = np.var(acc_pred)
        if pred_variance < 1e-7:  # Changed from == 0 to epsilon threshold
            print(f"  WARNING: Soft model collapse (Var={pred_variance:.8f}). Assigning Tau=0.0.") 
            ktau = 0.0
        
        # Kendall's Tau - PRIMARY METRIC for NAS
        # Measures rank correlation: does the surrogate order architectures correctly?
        # More robust than MSE for low-data regimes and non-linear relationships
        try:
            ktau, ktau_pvalue = kendalltau(acc_true, acc_pred)
        except Exception:
            ktau = np.nan # Catch internal scipy errors
        
        # 3. Handle NaN return from kendalltau (occurs with ties/constants)
        if np.isnan(ktau):
            print(f"  WARNING: kendalltau returned NaN (likely due to ties). Assigning Tau=0.0.")
            ktau = 0.0      # Penalize the model
        
        # MSE (mean squared error)
        mse = np.mean((acc_true - acc_pred) ** 2)
    
    return {
        'kendall_tau': ktau,  # Primary metric
        'mse': mse,
        'n_train': sample_size,
        'n_test': len(X_test)
    }



def run_comparison(
    embedding1_name,
    corpus1_name,
    embedding2_name,
    corpus2_name,
    corpus_path1,
    corpus_path2=None,
    comparison_label='comparison',
    sample_sizes=[15, 50, 150, 500, 1500, 5000],
    n_folds=5,
    n_repeats=5,
    benchmark_type='nasbench',
    comparison_output_path=None,
    per_embedding_output_dir=None,
    device='cuda',
    force=False
):
    """
    Compare two embeddings (same corpus or cross-corpus).
    
    Workflow:
      1. Load existing trials from Type 1 CSVs for both embeddings
      2. Determine which trials are missing and run them
      3. Save new trials to Type 1 CSVs
      4. Compute aggregated statistics and save to Type 2 CSV
    
    Args:
        embedding1_name: Name of first embedding
        corpus1_name: Name identifier for corpus 1 (used in filenames)
        embedding2_name: Name of second embedding
        corpus2_name: Name identifier for corpus 2 (used in filenames)
        corpus_path1: Path to corpus for embedding1 (and embedding2 if corpus_path2=None)
        corpus_path2: Path to corpus for embedding2 (if None, use corpus_path1 for both)
        comparison_label: Label for comparison (e.g., 'quant_vs_noquant')
        sample_sizes: List of training sizes
        n_folds: Number of CV folds
        n_repeats: Number of CV repeats
        benchmark_type: 'nasbench' or 'jahs'
        comparison_output_path: Path to global comparison CSV (Type 2)
        per_embedding_output_dir: Directory for per-embedding CSVs (Type 1)
        device: Device to use
        force: If True, ignore existing trials and recompute all
    
    Returns:
        DataFrame with comparison results
    """
    print(f"\n{'='*80}")
    print(f"Comparison: {comparison_label}")
    print(f"{'='*80}")
    
    # Determine if same corpus or cross-corpus
    if corpus_path2 is None:
        corpus_path2 = corpus_path1
        is_cross_corpus = False
    else:
        is_cross_corpus = (corpus_path1 != corpus_path2)
    
    # Load corpus/corpora
    if not is_cross_corpus:
        # Same corpus comparison
        print(f"Loading corpus from {corpus_path1}...")
        df = pd.read_pickle(corpus_path1)
        print(f"  Loaded {len(df)} architectures")
        
        # Verify both embeddings exist
        if embedding1_name not in df.columns:
            raise ValueError(f"Embedding '{embedding1_name}' not found in corpus")
        if embedding2_name not in df.columns:
            raise ValueError(f"Embedding '{embedding2_name}' not found in corpus")
        
        # Extract embeddings
        X1 = np.array(df[embedding1_name].tolist()).astype(np.float32)
        X2 = np.array(df[embedding2_name].tolist()).astype(np.float32)
        
        model1_name = embedding1_name
        model2_name = embedding2_name
        corpus1 = corpus_path1
        corpus2 = corpus_path1
        
    else:
        # Cross-corpus comparison
        print(f"Loading corpus 1 from {corpus_path1}...")
        df1 = pd.read_pickle(corpus_path1)
        print(f"  Loaded {len(df1)} architectures")
        
        print(f"Loading corpus 2 from {corpus_path2}...")
        df2 = pd.read_pickle(corpus_path2)
        print(f"  Loaded {len(df2)} architectures")
        
        if len(df1) != len(df2):
            raise ValueError(f"Corpus sizes don't match: {len(df1)} vs {len(df2)}")
        
        # Verify embeddings exist
        if embedding1_name not in df1.columns:
            raise ValueError(f"Embedding '{embedding1_name}' not found in corpus 1")
        if embedding2_name not in df2.columns:
            raise ValueError(f"Embedding '{embedding2_name}' not found in corpus 2")
        
        # Extract embeddings
        X1 = np.array(df1[embedding1_name].tolist()).astype(np.float32)
        X2 = np.array(df2[embedding2_name].tolist()).astype(np.float32)
        
        model1_name = f"{embedding1_name}_corpus1"
        model2_name = f"{embedding2_name}_corpus2"
        corpus1 = corpus_path1
        corpus2 = corpus_path2
        df = df1  # Use first dataframe for targets
    
    # Prepare targets
    if benchmark_type == 'nasbench':
        y = df[['cifar10-valid_valid_loss', 'cifar10-valid_valid_accuracy']].values.astype(np.float32)
    elif benchmark_type == 'jahs':
        y = df[['test_acc', 'valid_acc']].values.astype(np.float32)
    else:
        raise ValueError(f"Unknown benchmark_type: {benchmark_type}")
    
    print(f"Targets shape: {y.shape}, dtype: {y.dtype}")
    
    print(f"\nComparing: {model1_name} vs {model2_name}")
    print(f"  {model1_name}: {X1.shape}")
    print(f"  {model2_name}: {X2.shape}")
    print(f"{'='*80}\n")
    
    # Load existing trials from Type 1 CSVs
    trial_data_dict = {}
    if not force and per_embedding_output_dir:
        print(f"Loading existing trials from Type 1 CSVs...")
        trial_data_dict[model1_name] = load_existing_trials(
            per_embedding_output_dir, embedding1_name, corpus1_name
        )
        trial_data_dict[model2_name] = load_existing_trials(
            per_embedding_output_dir, embedding2_name, corpus2_name
        )
        
        total_m1 = sum(len(trials) for trials in trial_data_dict[model1_name].values())
        total_m2 = sum(len(trials) for trials in trial_data_dict[model2_name].values())
        print(f"  {model1_name}: {total_m1} existing trials")
        print(f"  {model2_name}: {total_m2} existing trials")
    
    # Prepare X dict
    X = {model1_name: X1, model2_name: X2}
    embedding_types = [model1_name, model2_name]
    
    # Run comparison
    result_df = subsampled_repeated_kfold_comparison(
        X, y, embedding_types,
        sample_sizes=sample_sizes,
        n_folds=n_folds,
        n_repeats=n_repeats,
        model1_idx=0,
        model2_idx=1,
        device=device,
        pairs_to_compute=None,
        trial_data_dict=trial_data_dict
    )
    
    # Add metadata
    result_df['comparison_type'] = comparison_label
    result_df['embedding1'] = embedding1_name
    result_df['embedding2'] = embedding2_name
    result_df['corpus1'] = corpus1_name
    result_df['corpus2'] = corpus2_name
    
    # Split and save results
    per_embedding_dict, comparison_df = split_results_for_saving(result_df, trial_data_dict)
    
    # Save Type 1: Per-embedding results (only NEW trials)
    if per_embedding_output_dir:
        print(f"\nSaving per-embedding results (Type 1)...")
        for embedding_name, emb_df in per_embedding_dict.items():
            if len(emb_df) > 0:  # Only save if there are new trials
                # Determine which corpus this embedding belongs to
                if embedding_name == model1_name:
                    corpus_name = corpus1_name
                    actual_embedding_name = embedding1_name
                else:
                    corpus_name = corpus2_name
                    actual_embedding_name = embedding2_name
                
                save_per_embedding_results(emb_df, per_embedding_output_dir, 
                                          actual_embedding_name, corpus_name)
    
    # Save Type 2: Comparison results (always save)
    if comparison_output_path:
        print(f"\nSaving comparison results (Type 2)...")
        save_comparison_results(comparison_df, comparison_output_path)
    
    return result_df



if __name__ == '__main__':
    # Example usage with simplified run_comparison
    results = run_comparison(
        embedding1_name='deepseek_coder_pytorch_code_embedding',
        embedding2_name='codellama_python_pytorch_code_embedding',
        corpus_path1='/storage/ice-shared/vip-vvk/data/AOT/psomu3/codenas/nasbench201_corpus_embedded_half.pkl',
        corpus_path2=None,  # Same corpus
        comparison_label='deepseek_vs_codellama',
        sample_sizes=[15, 50, 150, 500, 1500],
        n_folds=5,
        n_repeats=5,
        benchmark_type='nasbench',
        comparison_output_path='/storage/ice-shared/vip-vvk/data/AOT/psomu3/codenas/comparisons_global.csv',
        per_embedding_output_dir='/storage/ice-shared/vip-vvk/data/AOT/psomu3/codenas/per_embedding_results/',
        device='cuda'
    )
    
    print("\n" + "="*80)
    print("Final Results Summary")
    print("="*80)
    print(results)

