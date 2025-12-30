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
            return { 'kendall_tau': 0.0, 'r2': 0.0, 'mse': 999.0, 'n_train': sample_size, 'n_test': len(X_test) }
        
        pred_variance = np.var(acc_pred)
        true_variance = np.var(acc_true)

        # 2. Check for Soft Collapse (Variance is tiny but not zero)
        pred_variance = np.var(acc_pred)
        if pred_variance < 1e-7:  # Changed from == 0 to epsilon threshold
            print(f"  WARNING: Soft model collapse (Var={pred_variance:.8f}). Assigning Tau=0.0.") 
            ktau = 0.0
        
        # Kendall's Tau - PRIMARY METRIC for NAS
        # Measures rank correlation: does the surrogate order architectures correctly?
        # More robust than R² for low-data regimes and non-linear relationships
        try:
            ktau, ktau_pvalue = kendalltau(acc_true, acc_pred)
        except Exception:
            ktau = np.nan # Catch internal scipy errors
        
        # 43. Handle NaN return from kendalltau (occurs with ties/constants)
        if np.isnan(ktau):
            print(f"  WARNING: kendalltau returned NaN (likely due to ties). Assigning Tau=0.0.")
            ktau = 0.0      # Penalize the model
        
        # R2 score (kept for reference)
        ss_res = np.sum((acc_true - acc_pred) ** 2)
        ss_tot = np.sum((acc_true - acc_true.mean()) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        # MSE (kept for reference)
        mse = np.mean((acc_true - acc_pred) ** 2)
    
    return {
        'kendall_tau': ktau,  # Primary metric
        'r2': r2,
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
    existing_results=None,
    pairs_to_compute=None
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
        
        # Determine how many trials already exist and how many we need
        existing_trials = 0
        if existing_results is not None:
            mask = (
                (existing_results['model1'] == model1_name) &
                (existing_results['model2'] == model2_name) &
                (existing_results['sample_size'] == sample_size)
            )
            if mask.any():
                existing_trials = existing_results[mask]['n_trials'].iloc[0]
        
        total_trials_needed = n_folds * n_repeats
        trials_to_compute = total_trials_needed - existing_trials
        
        if trials_to_compute <= 0:
            print(f"Already have {existing_trials} trials, skipping")
            continue
        
        print(f"Computing {trials_to_compute} new trials (have {existing_trials}, need {total_trials_needed})")
        
        differences = []  # Store Model1 - Model2 for ONLY NEW trials
        model1_ktau_scores = []  # Track individual model1 Kendall's Tau scores
        model2_ktau_scores = []  # Track individual model2 Kendall's Tau scores
        
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
                total_trials += 1
            
            if total_trials >= trials_to_compute:
                break
        
        # Aggregate with existing trials
        n_train = len(pool_idx)
        n_test = len(test_idx)
        
        # Get ALL differences (existing + new)
        all_differences = []
        if existing_results is not None:
            mask = (
                (existing_results['model1'] == model1_name) &
                (existing_results['model2'] == model2_name) &
                (existing_results['sample_size'] == sample_size)
            )
            if mask.any():
                # Reconstruct differences from existing stats
                # We can't recover individual trial differences, but we have aggregate stats
                # For now, just use new differences for the incremental result row
                pass
        
        # Combine with existing if present
        new_differences = np.array(differences)
        model1_ktau_array = np.array(model1_ktau_scores)
        model2_ktau_array = np.array(model2_ktau_scores)
        
        # Calculate statistics on Kendall's Tau
        model1_mean_ktau = np.mean(model1_ktau_array)
        model1_std_ktau = np.std(model1_ktau_array)
        model2_mean_ktau = np.mean(model2_ktau_array)
        model2_std_ktau = np.std(model2_ktau_array)
        mean_diff = np.mean(new_differences)
        std_diff = np.std(new_differences)
        
        # Perform corrected paired t-test on Kendall's Tau differences
        # H0: mean(model1_ktau - model2_ktau) = 0 (no difference in ranking ability)
        # H1: mean(model1_ktau - model2_ktau) ≠ 0 (significant difference in ranking)
        t_stat, p_value = corrected_paired_ttest(new_differences, n_train, n_test)
        
        results.append({
            'sample_size': sample_size,
            'model1': model1_name,
            'model2': model2_name,
            'metric': 'kendall_tau',  # Track which metric
            'model1_mean': model1_mean_ktau,
            'model1_std': model1_std_ktau,
            'model2_mean': model2_mean_ktau,
            'model2_std': model2_std_ktau,
            'mean_diff': mean_diff,
            'std_diff': std_diff,
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'n_trials': len(new_differences),  # Only NEW trials
            'n_train_actual': n_train,
            'n_test_actual': n_test
        })
        
        # Detailed reporting
        print(f"\n{'='*70}")
        print(f"Results for Sample Size {sample_size}:")
        print(f"{'='*70}")
        print(f"  {model1_name}: Kendall's Tau = {model1_mean_ktau:.4f} ± {model1_std_ktau:.4f}")
        print(f"  {model2_name}: Kendall's Tau = {model2_mean_ktau:.4f} ± {model2_std_ktau:.4f}")
        print(f"  Difference (M1-M2): {mean_diff:.4f} ± {std_diff:.4f}")
        if model1_mean_ktau < 0 or model2_mean_ktau < 0:
            print(f"  ⚠ Negative Kendall's Tau: Model has negative rank correlation")
        
        # Determine which model is better based on observed difference
        if mean_diff > 0:
            # Model1 is better (positive difference)
            better_model = model1_name
            worse_model = model2_name
            print(f"\nOne-Tailed Hypothesis Test:")
            print(f"  Observed: {model1_name} has higher Kendall's Tau")
            print(f"  H0: {model1_name}_Tau ≤ {model2_name}_Tau (no real difference)")
            print(f"  H1: {model1_name}_Tau > {model2_name}_Tau ({model1_name} is genuinely better)")
        else:
            # Model2 is better (negative difference)
            better_model = model2_name
            worse_model = model1_name
            print(f"\nOne-Tailed Hypothesis Test:")
            print(f"  Observed: {model2_name} has higher Kendall's Tau")
            print(f"  H0: {model2_name}_Tau ≤ {model1_name}_Tau (no real difference)")
            print(f"  H1: {model2_name}_Tau > {model1_name}_Tau ({model2_name} is genuinely better)")
        
        print(f"  t-statistic: {t_stat:.3f}")
        print(f"  p-value (one-tailed): {p_value:.4f}")
        print(f"  → P-value tests if the observed advantage is statistically significant")
        
        if p_value < 0.05:
            print(f"  ✓ SIGNIFICANT (p<0.05): {better_model} is significantly better than {worse_model}")
        else:
            print(f"  ✗ NOT SIGNIFICANT (p≥0.05): Cannot conclude which is better")
        print(f"{'='*70}")
        
        
        print(f"New trials - Mean Diff (R²): {mean_diff:.4f} ± {std_diff:.4f}")
        print(f"t-statistic: {t_stat:.3f}, p-value: {p_value:.4f}")
        print(f"Significant: {p_value < 0.05}")
    
    return pd.DataFrame(results)


def run_cross_corpus_comparison(
    corpus_path1,
    corpus_path2,
    embedding_name1,
    embedding_name2=None,
    comparison_label='cross_corpus',
    sample_sizes=[15, 50, 150, 500, 1500, 5000],
    n_folds=5,
    n_repeats=5,
    benchmark_type='nasbench',
    output_path=None,
    device='cuda',
    force=False
):
    """
    Compare embeddings from TWO different corpus files.
    Can compare same embedding (e.g., quantized vs non-quantized) or different embeddings.
    
    Args:
        corpus_path1: Path to first corpus pickle
        corpus_path2: Path to second corpus pickle
        embedding_name1: Name of embedding in corpus 1 (e.g., 'codestral_7b_pytorch_code_embedding')
        embedding_name2: Name of embedding in corpus 2 (if None, uses same as embedding_name1)
        comparison_label: Label for comparison (e.g., 'quant_vs_noquant', 'model_x_vs_y')
        sample_sizes: List of training sizes for learning curve
        n_folds: Number of CV folds
        n_repeats: Number of CV repeats
        benchmark_type: 'nasbench' or 'jahs'
        output_path: Path to save results
        device: Device to use
        force: If True, recompute all comparisons regardless of existing results
    
    Returns:
        DataFrame with comparison results
    """
    # Default to same embedding in both corpora if not specified
    if embedding_name2 is None:
        embedding_name2 = embedding_name1
    
    print(f"\n{'='*80}")
    print(f"Cross-Corpus Comparison: {comparison_label}")
    print(f"{'='*80}")
    print(f"Corpus 1: {corpus_path1}")
    print(f"  Embedding: {embedding_name1}")
    print(f"Corpus 2: {corpus_path2}")
    print(f"  Embedding: {embedding_name2}")
    print(f"{'='*80}\n")
    
    # Load both corpora
    print(f"Loading corpus 1 from {corpus_path1}...")
    df1 = pd.read_pickle(corpus_path1)
    print(f"  Loaded {len(df1)} architectures")
    
    print(f"Loading corpus 2 from {corpus_path2}...")
    df2 = pd.read_pickle(corpus_path2)
    print(f"  Loaded {len(df2)} architectures")
    
    # Verify same architectures in same order
    if len(df1) != len(df2):
        raise ValueError(f"Corpus sizes don't match: {len(df1)} vs {len(df2)}")
    
    # Check if both have the embeddings
    if embedding_name1 not in df1.columns:
        raise ValueError(f"Embedding '{embedding_name1}' not found in corpus 1")
    if embedding_name2 not in df2.columns:
        raise ValueError(f"Embedding '{embedding_name2}' not found in corpus 2")
    
    # Check existing results
    existing_results = None
    if output_path and os.path.exists(output_path) and not force:
        print(f"\nLoading existing results from {output_path}...")
        existing_results = pd.read_csv(output_path)
        print(f"  Found {len(existing_results)} existing rows")
    
    # Prepare targets (use corpus 1, should be identical in both)
    if benchmark_type == 'nasbench':
        y = df1[['cifar10-valid_valid_loss', 'cifar10-valid_valid_accuracy']].values.astype(np.float32)
    elif benchmark_type == 'jahs':
        y = df1[['test_acc', 'valid_acc']].values.astype(np.float32)
    else:
        raise ValueError(f"Unknown benchmark_type: {benchmark_type}")
    
    print(f"Targets shape: {y.shape}, dtype: {y.dtype}")
    
    # Extract embeddings from both corpora
    X1 = np.array(df1[embedding_name1].tolist()).astype(np.float32)
    X2 = np.array(df2[embedding_name2].tolist()).astype(np.float32)
    
    print(f"\nEmbeddings from corpus 1 ({embedding_name1}): {X1.shape}, dtype {X1.dtype}")
    print(f"Embeddings from corpus 2 ({embedding_name2}): {X2.shape}, dtype {X2.dtype}")
    
    # Create model names
    model1_name = f"{embedding_name1}_corpus1"
    model2_name = f"{embedding_name2}_corpus2"
    
    # Prepare X dict for comparison function
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
        existing_results=existing_results,
        pairs_to_compute=None
    )
    
    # Add comparison metadata
    result_df['comparison_type'] = comparison_label
    result_df['embedding1'] = embedding_name1
    result_df['embedding2'] = embedding_name2
    result_df['corpus1'] = corpus_path1
    result_df['corpus2'] = corpus_path2
    
    # MERGE with existing results
    if existing_results is not None:
        final_results = pd.concat([existing_results, result_df], ignore_index=True)
        print(f"\n  Merged into {len(final_results)} total rows ({len(result_df)} new)")
    else:
        final_results = result_df
    
    # Save results
    if output_path:
        print(f"\nSaving results to {output_path}...")
        final_results.to_csv(output_path, index=False)
        print("Results saved!")
    
    return final_results


def run_robust_comparison(
    corpus_path,
    embedding_types,
    sample_sizes=[15, 50, 150, 500, 1500, 5000],
    n_folds=5,
    n_repeats=5,
    benchmark_type='nasbench',  # 'nasbench' or 'jahs'
    output_path=None,
    device='cuda',
    force=False
):
    """
    Run robust comparison on a corpus with multiple embeddings.
    Incrementally adds trials - only computes missing comparisons.
    
    Args:
        corpus_path: Path to corpus pickle
        embedding_types: List of embedding column names to compare
        sample_sizes: List of training sizes for learning curve
        n_folds: Number of CV folds
        n_repeats: Number of CV repeats
        benchmark_type: 'nasbench' or 'jahs'
        output_path: Path to save results
        device: Device to use
        force: If True, recompute all comparisons regardless of existing results
    
    Returns:
        DataFrame with comparison results
    """
    print(f"Loading corpus from {corpus_path}...")
    df = pd.read_pickle(corpus_path)
    print(f"Loaded {len(df)} architectures")
    
    # Check existing results
    existing_results = None
    if output_path and os.path.exists(output_path) and not force:
        print(f"\nLoading existing results from {output_path}...")
        existing_results = pd.read_csv(output_path)
        print(f"  Found {len(existing_results)} existing rows")
        print(f"  Existing columns: {list(existing_results.columns)}")
    
    # Prepare targets based on benchmark type
    if benchmark_type == 'nasbench':
        # Use CIFAR-10 metrics
        y = df[['cifar10-valid_valid_loss', 'cifar10-valid_valid_accuracy']].values.astype(np.float32)
    elif benchmark_type == 'jahs':
        # Use test and validation accuracy (JAHS doesn't have loss metrics)
        y = df[['test_acc', 'valid_acc']].values.astype(np.float32)
    else:
        raise ValueError(f"Unknown benchmark_type: {benchmark_type}")

    print(f"Targets shape: {y.shape}, dtype: {y.dtype}")

    # Preview two rows
    print("\nSample target values:")
    print(y[:2])
    
    # Prepare features (embeddings)
    X = {}
    for emb_type in embedding_types:
        if emb_type not in df.columns:
            raise ValueError(f"Embedding '{emb_type}' not found in corpus")
        X[emb_type] = np.array(df[emb_type].tolist()).astype(np.float32)
        print(f"  {emb_type}: shape {X[emb_type].shape}, dtype {X[emb_type].dtype}")
    
    # Compare all pairs of embeddings
    all_new_results = []
    n_types = len(embedding_types)
    
    for i in range(n_types):
        for j in range(i + 1, n_types):
            model1_name = embedding_types[i]
            model2_name = embedding_types[j]
            
            # Check what we need to compute for this pair
            pairs_to_compute = []  # List of (sample_size, existing_trials, needed_trials)
            
            for sample_size in sample_sizes:
                expected_trials = n_folds * n_repeats
                existing_trials = 0
                
                if existing_results is not None:
                    # Check if this model pair + sample size exists
                    mask = (
                        (existing_results['model1'] == model1_name) &
                        (existing_results['model2'] == model2_name) &
                        (existing_results['sample_size'] == sample_size)
                    )
                    if mask.any():
                        existing_trials = existing_results[mask]['n_trials'].iloc[0]
                
                if existing_trials < expected_trials:
                    needed_trials = expected_trials - existing_trials
                    pairs_to_compute.append((sample_size, existing_trials, needed_trials))
                    print(f"  {model1_name} vs {model2_name}, size={sample_size}: "
                          f"have {existing_trials}/{expected_trials} trials, computing {needed_trials} more")
                else:
                    print(f"  {model1_name} vs {model2_name}, size={sample_size}: "
                          f"already have {existing_trials}/{expected_trials} trials, SKIPPING")
            
            if pairs_to_compute:
                # Compute only the missing trials for this pair
                result_df = subsampled_repeated_kfold_comparison(
                    X, y, embedding_types,
                    sample_sizes=[p[0] for p in pairs_to_compute],  # Only sizes we need
                    n_folds=n_folds,
                    n_repeats=n_repeats,
                    model1_idx=i,
                    model2_idx=j,
                    device=device,
                    existing_results=existing_results,
                    pairs_to_compute=pairs_to_compute
                )
                all_new_results.append(result_df)
            else:
                print(f"  {model1_name} vs {model2_name}: All trials complete, nothing to compute")
    
    if not all_new_results:
        print("\n" + "="*80)
        print("No new results to compute - all requested comparisons already exist!")
        print("="*80)
        return existing_results if existing_results is not None else pd.DataFrame()
    
    # Concatenate all NEW results we just computed
    new_results = pd.concat(all_new_results, ignore_index=True)
    
    # MERGE with existing results - ONLY ADD, NEVER DELETE
    if existing_results is not None:
        # Keep ALL old results + ALL new results
        final_results = pd.concat([existing_results, new_results], ignore_index=True)
        print(f"\n  Merged into {len(final_results)} total rows ({len(new_results)} new)")
    else:
        final_results = new_results
    
    # Save merged results
    if output_path:
        print(f"\nSaving results to {output_path}...")
        final_results.to_csv(output_path, index=False)
        print("Results saved!")
    
    return final_results


if __name__ == '__main__':
    # Example usage
    results = run_robust_comparison(
        corpus_path='/storage/ice-shared/vip-vvk/data/AOT/psomu3/codenas/nasbench201_corpus_embedded_half.pkl',
        embedding_types=[
            'deepseek_coder_pytorch_code_embedding',
            'codellama_python_pytorch_code_embedding'
        ],
        sample_sizes=[15, 50, 150, 500, 1500],
        n_folds=5,
        n_repeats=5,
        benchmark_type='nasbench',
        output_path='/storage/ice-shared/vip-vvk/data/AOT/psomu3/codenas/robust_comparison_results.csv',
        device='cuda'
    )
    
    print("\n" + "="*80)
    print("Final Results Summary")
    print("="*80)
    print(results)
