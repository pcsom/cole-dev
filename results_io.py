"""
Results saving and loading utilities for robust surrogate prediction.

This module handles two types of CSV files:
  Type 1: Per-embedding CSVs with individual trial results (one row per trial)
  Type 2: Global comparison CSV with aggregated statistics and significance tests
"""

import pandas as pd
import os


def load_existing_trials(output_dir, embedding_name, corpus_name):
    """
    Load existing trial data from a Type 1 CSV file.
    
    Args:
        output_dir: Directory containing per-embedding CSV files
        embedding_name: Name of the embedding
        corpus_name: Name of the corpus
    
    Returns:
        Dictionary mapping sample_size to list of (trial_num, ktau, mse) tuples
        Returns empty dict if file doesn't exist
    """
    if output_dir is None:
        return {}
    
    csv_path = os.path.join(output_dir, f"{embedding_name}_{corpus_name}.csv")
    
    if not os.path.exists(csv_path):
        return {}
    
    try:
        df = pd.read_csv(csv_path)
        trials_dict = {}
        
        for _, row in df.iterrows():
            sample_size = int(row['sample_size'])
            fold = int(row['fold'])
            repeat = int(row['repeat'])
            ktau = float(row['kendall_tau'])
            mse = float(row['mse'])
            
            if sample_size not in trials_dict:
                trials_dict[sample_size] = []
            
            trials_dict[sample_size].append((fold, repeat, ktau, mse))
        
        return trials_dict
    except Exception as e:
        print(f"  Warning: Error loading {csv_path}: {e}")
        return {}


def save_per_embedding_results(results_df, output_dir, embedding_name, corpus_name):
    """
    Save per-embedding results to a CSV file.
    Type 1: One CSV per embedding+corpus containing individual trial results (one row per trial).
    Always appends to existing CSV (add, don't delete behavior).
    
    Args:
        results_df: DataFrame with per-embedding trial results for ONE embedding
        output_dir: Directory to save CSV files
        embedding_name: Name of the embedding (used in filename)
        corpus_name: Name of the corpus (used in filename)
    
    Returns:
        Path to the saved CSV file
    """
    os.makedirs(output_dir, exist_ok=True)
    base_filename = f"{embedding_name}_{corpus_name}.csv"
    output_path = os.path.join(output_dir, base_filename)
    
    if os.path.exists(output_path):
        # Always append to existing CSV
        existing = pd.read_csv(output_path)
        results_df = pd.concat([existing, results_df], ignore_index=True)
        print(f"  Appended to existing per-embedding CSV: {base_filename} ({len(existing)} -> {len(results_df)} rows)")
    else:
        print(f"  Created new per-embedding CSV: {base_filename}")
    
    results_df.to_csv(output_path, index=False)
    print(f"  Saved per-embedding results to: {output_path}")
    return output_path


def save_comparison_results(results_df, output_path):
    """
    Save comparison results to a global CSV file.
    Type 2: One global CSV containing all comparison results with aggregated statistics.
    Always appends to existing file (add, don't delete behavior).
    
    Args:
        results_df: DataFrame with comparison results
        output_path: Path to the global comparison CSV file
    
    Returns:
        Path to the saved CSV file
    """
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    if os.path.exists(output_path):
        # Always append new comparisons to existing
        existing = pd.read_csv(output_path)
        results_df = pd.concat([existing, results_df], ignore_index=True)
        print(f"  Appended to existing comparison CSV ({len(existing)} -> {len(results_df)} rows)")
    
    results_df.to_csv(output_path, index=False)
    print(f"  Saved comparison results to: {output_path}")
    return output_path


def split_results_for_saving(results_df, trial_data_dict):
    """
    Split results into per-embedding (individual trials) and comparison (aggregated) DataFrames.
    
    Args:
        results_df: DataFrame with aggregated results from subsampled_repeated_kfold_comparison
        trial_data_dict: Dict with structure {embedding_name: {sample_size: [(trial_num, ktau, r2, mse), ...]}}
    
    Returns:
        Tuple of (per_embedding_dict, comparison_df)
        - per_embedding_dict: {embedding_name: DataFrame with individual trial results}
        - comparison_df: DataFrame with comparison metrics AND aggregated intrinsics
    """
    # Columns for comparison CSV (Type 2) - now includes aggregated intrinsics and metadata
    comparison_cols = [
        'sample_size', 'model1', 'model2', 'metric',
        # Model1 intrinsics (aggregated)
        'model1_mean_ktau', 'model1_std_ktau',
        'model1_mean_mse', 'model1_std_mse',
        # Model2 intrinsics (aggregated)
        'model2_mean_ktau', 'model2_std_ktau',
        'model2_mean_mse', 'model2_std_mse',
        # Comparison statistics
        'mean_diff_ktau', 'std_diff_ktau', 't_statistic_ktau', 'p_value_ktau', 'significant_ktau',
        'mean_diff_mse', 'std_diff_mse', 't_statistic_mse', 'p_value_mse', 'significant_mse',
        'n_trials', 'n_train_actual', 'n_test_actual',
        # Metadata (optional, from cross-corpus comparisons)
        'comparison_type', 'embedding1', 'embedding2', 'corpus1', 'corpus2'
    ]
    
    # Build comparison DataFrame with renamed columns for clarity
    comparison_df = results_df.copy()
    comparison_df = comparison_df.rename(columns={
        'model1_mean': 'model1_mean_ktau',
        'model1_std': 'model1_std_ktau',
        'model2_mean': 'model2_mean_ktau',
        'model2_std': 'model2_std_ktau',
        'mean_diff': 'mean_diff_ktau',
        'std_diff': 'std_diff_ktau',
        't_statistic': 't_statistic_ktau',
        'p_value': 'p_value_ktau',
        'significant': 'significant_ktau'
    })
    
    # Only keep columns that exist in the DataFrame
    existing_cols = [col for col in comparison_cols if col in comparison_df.columns]
    comparison_df = comparison_df[existing_cols]
    
    # Build per-embedding DataFrames from trial data (Type 1)
    per_embedding_dict = {}
    
    for embedding_name, sample_dict in trial_data_dict.items():
        trials_list = []
        for sample_size, trials in sample_dict.items():
            for fold, repeat, ktau, mse in trials:
                trials_list.append({
                    'embedding_name': embedding_name,
                    'sample_size': sample_size,
                    'fold': fold,
                    'repeat': repeat,
                    'kendall_tau': ktau,
                    'mse': mse
                })
        
        if trials_list:
            per_embedding_dict[embedding_name] = pd.DataFrame(trials_list)
    
    return per_embedding_dict, comparison_df
