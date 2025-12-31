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
