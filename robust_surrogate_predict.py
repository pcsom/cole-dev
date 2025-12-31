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


def corrected_paired_ttest(differences, n_train, n_test):
    """
    One-tailed corrected paired t-test accounting for non-independence in CV.
    Tests if model2 is significantly better than model1.
    
    From Nadeau & Bengio (2003):
    t = mean(d) / sqrt( (1/n + n_test/n_train) * var(d) )
    
    Args:
        differences: Array of performance differences (Model1_R² - Model2_R²)
        n_train: Training set size used
        n_test: Test set size
    
    Returns:
        t-statistic, p-value (one-tailed: tests if model2 > model1)
    """
    d_mean = np.mean(differences)
    d_var = np.var(differences, ddof=1)
    n = len(differences)
    
    # Corrected variance
    corrected_var = (1.0 / n + n_test / n_train) * d_var
    
    # t-statistic
    if corrected_var <= 0:
        return 0.0, 1.0
    
    t_stat = d_mean / np.sqrt(corrected_var)
    
    # Degrees of freedom
    df = n - 1
    
    # One-tailed p-value: tests H1: model2 > model1 (i.e., difference < 0)
    # If t_stat < 0, model2 is better (negative difference means model1 < model2)
    if t_stat < 0:
        # Model2 is better, report lower tail probability
        p_value = stats.t.cdf(t_stat, df)
    else:
        # Model1 is better or equal, report upper tail (1 - prob)
        p_value = 1 - stats.t.cdf(t_stat, df)
    
    return t_stat, p_value


def subsampled_repeated_kfold_comparison(
    X, y, embedding_types,
    sample_sizes=[15, 50, 150, 500, 1500, 5000],
    n_folds=5,
    n_repeats=5,
    model1_idx=0,
    model2_idx=1,
    epochs=100,
    lr=0.001,
    batch_size=32,
    device='cuda',
    random_state=42,
    pairs_to_compute=None,
    trial_data_dict=None
):
    """
    Perform subsampled repeated k-fold CV with corrected statistical testing.
    Implements "Split First, Subsample Later" methodology for stable evaluation.
    Only computes missing trials - adds incrementally to existing results.
    
    Args:
        X: Dictionary mapping embedding_type to feature arrays
        y: Target array (same for all embedding types)
        embedding_types: List of embedding type names
        sample_sizes: List of training sample sizes to test
        n_folds: Number of folds for CV
        n_repeats: Number of repeats for each fold
        model1_idx: Index of first model in embedding_types
        model2_idx: Index of second model in embedding_types
        epochs: Training epochs per model (fixed, no early stopping)
        lr: Learning rate
        batch_size: Batch size (default 32 to match surrogate_predict.py)
        device: Device to use
        random_state: Base random seed
        existing_results: DataFrame of existing results (optional)
        pairs_to_compute: List of (sample_size, existing_trials, needed_trials) tuples
    
    Methodology:
        - Split First, Subsample Later: K-fold creates train_pool/test, 
          then we subsample EXACTLY sample_size from train_pool
        - Target Scaling: Targets scaled to mean=0, std=1 (like surrogate_predict.py)
        - Feature Scaling: None (embeddings have sensitive distributions)
        - Training: Fixed epochs on full subsample, test on full test fold
    
    Returns:
        DataFrame with results for all sample sizes (only NEW trials)
    """
    model1_name = embedding_types[model1_idx]
    model2_name = embedding_types[model2_idx]
    
    print(f"\n{'='*80}")
    print(f"Computing additional trials for {model1_name} vs {model2_name}")
    if pairs_to_compute:
        print(f"Sample sizes to compute: {[p[0] for p in pairs_to_compute]}")
    else:
        print(f"Sample sizes: {sample_sizes}")
    print(f"Folds: {n_folds}, Repeats: {n_repeats}")
    print(f"{'='*80}\n")
    
    X1 = X[model1_name]
    X2 = X[model2_name]
    
    results = []
    
    # For stratification, bin the continuous accuracy into 5 bins
    y_acc = y[:, 1]  # Accuracy is the second column
    y_bins = pd.qcut(y_acc, q=5, labels=False, duplicates='drop')
    
    for sample_size in sample_sizes:
        print(f"\n--- Sample Size: {sample_size} ---")
        
        # Determine how many trials already exist from trial_data_dict (Type 1 CSVs)
        existing_trials_m1 = len(trial_data_dict.get(model1_name, {}).get(sample_size, []))
        existing_trials_m2 = len(trial_data_dict.get(model2_name, {}).get(sample_size, []))
        existing_trials = min(existing_trials_m1, existing_trials_m2)  # Both models must have same trials
        
        total_trials_needed = n_folds * n_repeats
        trials_to_compute = total_trials_needed - existing_trials
        
        if trials_to_compute <= 0:
            print(f"Already have {existing_trials} trials, skipping")
            continue
        
        print(f"Computing {trials_to_compute} new trials (have {existing_trials}, need {total_trials_needed})")
        
        differences = []  # Store Model1 - Model2 for ONLY NEW trials
        model1_ktau_scores = []  # Track individual model1 Kendall's Tau scores
        model2_ktau_scores = []  # Track individual model2 Kendall's Tau scores
        model1_mse_scores = []  # Track individual model1 MSE scores
        model2_mse_scores = []  # Track individual model2 MSE scores
        
        differences_ktau = []  # Store Model1 - Model2 Kendall's Tau differences
        differences_mse = []  # Store Model1 - Model2 MSE differences
        
        # Start from where we left off
        start_trial = existing_trials
        total_trials = 0
        
        # Outer loop: Repeat the CV procedure
        for repeat in tqdm(range(n_repeats), desc=f"Repeats (S={sample_size})"):
            # Create stratified k-fold with different seed per repeat
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, 
                                 random_state=random_state + repeat)
            
            # Inner loop: Folds
            for fold_idx, (pool_idx, test_idx) in enumerate(skf.split(X1, y_bins)):
                trial_number = repeat * n_folds + fold_idx
                
                # Skip trials we already computed
                if trial_number < start_trial:
                    continue
                
                if total_trials >= trials_to_compute:
                    break
                
                # Training pool (4 folds)
                X1_pool, y_pool = X1[pool_idx], y[pool_idx]
                X2_pool = X2[pool_idx]
                
                # Test set (1 fold)
                X1_test, y_test = X1[test_idx], y[test_idx]
                X2_test = X2[test_idx]
                
                # Train Model 1
                seed1 = random_state + repeat * 1000 + fold_idx * 100
                result1 = train_model_on_subsample(
                    X1_pool, y_pool, X1_test, y_test,
                    sample_size=min(sample_size, len(X1_pool)),
                    random_state=seed1,
                    epochs=epochs, lr=lr, batch_size=batch_size, device=device
                )
                
                # Train Model 2 on SAME subsample indices
                seed2 = random_state + repeat * 1000 + fold_idx * 100 + 1
                result2 = train_model_on_subsample(
                    X2_pool, y_pool, X2_test, y_test,
                    sample_size=min(sample_size, len(X2_pool)),
                    random_state=seed2,
                    epochs=epochs, lr=lr, batch_size=batch_size, device=device
                )
                
                # Store individual scores and difference (using Kendall's Tau)
                score1 = result1['kendall_tau']
                score2 = result2['kendall_tau']
                model1_ktau_scores.append(score1)
                model2_ktau_scores.append(score2)
                diff_ktau = score1 - score2
                differences.append(diff_ktau)
                differences_ktau.append(diff_ktau)
                
                # Also store MSE scores and differences
                model1_mse_scores.append(result1['mse'])
                model2_mse_scores.append(result2['mse'])
                diff_mse = result1['mse'] - result2['mse']
                differences_mse.append(diff_mse)
                
                # Track individual trial data for Type 1 CSV saving
                if trial_data_dict is not None:
                    # Initialize dicts for model1 and model2 if not present
                    if model1_name not in trial_data_dict:
                        trial_data_dict[model1_name] = {}
                    if model2_name not in trial_data_dict:
                        trial_data_dict[model2_name] = {}
                    
                    # Initialize sample_size list if not present
                    if sample_size not in trial_data_dict[model1_name]:
                        trial_data_dict[model1_name][sample_size] = []
                    if sample_size not in trial_data_dict[model2_name]:
                        trial_data_dict[model2_name][sample_size] = []
                    
                    # Append trial data: (fold, repeat, ktau, mse)
                    trial_data_dict[model1_name][sample_size].append(
                        (fold_idx, repeat, result1['kendall_tau'], result1['mse'])
                    )
                    trial_data_dict[model2_name][sample_size].append(
                        (fold_idx, repeat, result2['kendall_tau'], result2['mse'])
                    )
                
                total_trials += 1
            
            if total_trials >= trials_to_compute:
                break
        
        # Aggregate with existing trials
        n_train = len(pool_idx)
        n_test = len(test_idx)
        
        # Get ALL trials from trial_data_dict (existing + new)
        all_model1_trials = trial_data_dict.get(model1_name, {}).get(sample_size, [])
        all_model2_trials = trial_data_dict.get(model2_name, {}).get(sample_size, [])
        
        # Extract metrics from all trials: (fold_idx, repeat, ktau, mse)
        all_model1_ktau = [t[2] for t in all_model1_trials]  # index 2 is ktau
        all_model1_mse = [t[3] for t in all_model1_trials]   # index 3 is mse
        
        all_model2_ktau = [t[2] for t in all_model2_trials]  # index 2 is ktau
        all_model2_mse = [t[3] for t in all_model2_trials]   # index 3 is mse
        
        # Convert to arrays
        model1_ktau_array = np.array(all_model1_ktau)
        model2_ktau_array = np.array(all_model2_ktau)
        model1_mse_array = np.array(all_model1_mse)
        model2_mse_array = np.array(all_model2_mse)
        
        # Calculate differences for all trials
        all_differences_ktau = model1_ktau_array - model2_ktau_array
        all_differences_mse = model1_mse_array - model2_mse_array
        
        # Calculate statistics on Kendall's Tau
        model1_mean_ktau = np.mean(model1_ktau_array)
        model1_std_ktau = np.std(model1_ktau_array)
        model2_mean_ktau = np.mean(model2_ktau_array)
        model2_std_ktau = np.std(model2_ktau_array)
        mean_diff_ktau = np.mean(all_differences_ktau)
        std_diff_ktau = np.std(all_differences_ktau)
        
        # Calculate statistics on MSE
        model1_mean_mse = np.mean(model1_mse_array)
        model1_std_mse = np.std(model1_mse_array)
        model2_mean_mse = np.mean(model2_mse_array)
        model2_std_mse = np.std(model2_mse_array)
        mean_diff_mse = np.mean(all_differences_mse)
        std_diff_mse = np.std(all_differences_mse)
        
        # Perform corrected paired t-test on both metrics (using ALL trials)
        t_stat_ktau, p_value_ktau = corrected_paired_ttest(all_differences_ktau, n_train, n_test)
        t_stat_mse, p_value_mse = corrected_paired_ttest(all_differences_mse, n_train, n_test)
        
        results.append({
            'sample_size': sample_size,
            'model1': model1_name,
            'model2': model2_name,
            'metric': 'kendall_tau',
            'model1_mean_ktau': model1_mean_ktau,
            'model1_std_ktau': model1_std_ktau,
            'model2_mean_ktau': model2_mean_ktau,
            'model2_std_ktau': model2_std_ktau,
            'mean_diff_ktau': mean_diff_ktau,
            'std_diff_ktau': std_diff_ktau,
            't_statistic_ktau': t_stat_ktau,
            'p_value_ktau': p_value_ktau,
            'significant_ktau': p_value_ktau < 0.05,
            'n_trials': len(all_model1_trials),  # Total trials (existing + new)
            'n_train_actual': n_train,
            'n_test_actual': n_test,
            # MSE statistics and significance
            'model1_mean_mse': model1_mean_mse,
            'model1_std_mse': model1_std_mse,
            'model2_mean_mse': model2_mean_mse,
            'model2_std_mse': model2_std_mse,
            'mean_diff_mse': mean_diff_mse,
            'std_diff_mse': std_diff_mse,
            't_statistic_mse': t_stat_mse,
            'p_value_mse': p_value_mse,
            'significant_mse': p_value_mse < 0.05
        })
        
        # Detailed reporting
        print(f"\n{'='*70}")
        print(f"Results for Sample Size {sample_size}:")
        print(f"{'='*70}")
        print(f"  {model1_name}:")
        print(f"    Kendall's Tau = {model1_mean_ktau:.4f} ± {model1_std_ktau:.4f}")
        print(f"    MSE = {model1_mean_mse:.4f} ± {model1_std_mse:.4f}")
        print(f"  {model2_name}:")
        print(f"    Kendall's Tau = {model2_mean_ktau:.4f} ± {model2_std_ktau:.4f}")
        print(f"    MSE = {model2_mean_mse:.4f} ± {model2_std_mse:.4f}")
        print(f"  Difference (M1-M2): {mean_diff_ktau:.4f} ± {std_diff_ktau:.4f}")
        if model1_mean_ktau < 0 or model2_mean_ktau < 0:
            print(f"  ⚠ Negative Kendall's Tau: Model has negative rank correlation")
        
        # Determine which model is better based on observed difference
        if mean_diff_ktau > 0:
            # Model1 is better (positive difference)
            better_model_ktau = model1_name
            worse_model_ktau = model2_name
            print(f"\nOne-Tailed Hypothesis Test (Kendall's Tau):")
            print(f"  Observed: {model1_name} has higher Kendall's Tau")
            print(f"  H0: {model1_name}_Tau ≤ {model2_name}_Tau (no real difference)")
            print(f"  H1: {model1_name}_Tau > {model2_name}_Tau ({model1_name} is genuinely better)")
        else:
            # Model2 is better (negative difference)
            better_model_ktau = model2_name
            worse_model_ktau = model1_name
            print(f"\nOne-Tailed Hypothesis Test (Kendall's Tau):")
            print(f"  Observed: {model2_name} has higher Kendall's Tau")
            print(f"  H0: {model2_name}_Tau ≤ {model1_name}_Tau (no real difference)")
            print(f"  H1: {model2_name}_Tau > {model1_name}_Tau ({model2_name} is genuinely better)")
        
        print(f"  t-statistic: {t_stat_ktau:.3f}")
        print(f"  p-value (one-tailed): {p_value_ktau:.4f}")
        print(f"  → P-value tests if the observed advantage is statistically significant")
        
        if p_value_ktau < 0.05:
            print(f"  ✓ SIGNIFICANT (p<0.05): {better_model_ktau} is significantly better than {worse_model_ktau}")
        else:
            print(f"  ✗ NOT SIGNIFICANT (p≥0.05): Cannot conclude which is better")
        
        # MSE significance test
        print(f"\nOne-Tailed Hypothesis Test (MSE):")
        print(f"  Difference (M1-M2): {mean_diff_mse:.4f} ± {std_diff_mse:.4f}")
        print(f"  t-statistic: {t_stat_mse:.3f}, p-value: {p_value_mse:.4f}")
        if p_value_mse < 0.05:
            # For MSE, lower is better, so reverse the logic
            better_model_mse = model2_name if mean_diff_mse > 0 else model1_name
            worse_model_mse = model1_name if mean_diff_mse > 0 else model2_name
            print(f"  ✓ SIGNIFICANT (p<0.05): {better_model_mse} is significantly better than {worse_model_mse}")
        else:
            print(f"  ✗ NOT SIGNIFICANT (p≥0.05): Cannot conclude which is better")
        
        print(f"{'='*70}")
    
    return pd.DataFrame(results)


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

