"""
Merge existing NASBench-201 corpus with true ONNX encodings by matching accuracy values.
Uses multi-precision matching (0.001 → 0.01 → 0.1) to maximize pairing.
"""

import pandas as pd
import numpy as np
from collections import defaultdict

# Paths
EXISTING_CORPUS_PATH = '/storage/ice-shared/vip-vvk/data/AOT/psomu3/codenas/nasbench201_corpus.pkl'
ONNX_CSV_PATH = '/storage/ice-shared/vip-vvk/data/AOT/psomu3/codenas/onnx/encodings/chain_slim_input/nasbench201.csv'
OUTPUT_PATH = '/storage/ice-shared/vip-vvk/data/AOT/psomu3/codenas/nasbench201_corpus_onnx.pkl'

def round_to_precision(values, precision):
    """Round values to specified precision."""
    return np.round(values / precision) * precision

def match_by_accuracy(corpus_accs, corpus_indices, onnx_accs, onnx_indices, precision):
    """
    Match architectures by rounded accuracy at given precision.
    Returns lists of matched (corpus_idx, onnx_idx) pairs and remaining unmatched indices.
    """
    # Round accuracies
    corpus_rounded = round_to_precision(corpus_accs, precision)
    onnx_rounded = round_to_precision(onnx_accs, precision)
    
    # Group by rounded accuracy
    corpus_bins = defaultdict(list)
    for i, idx in enumerate(corpus_indices):
        corpus_bins[corpus_rounded[i]].append(idx)
    
    onnx_bins = defaultdict(list)
    for i, idx in enumerate(onnx_indices):
        onnx_bins[onnx_rounded[i]].append(idx)
    
    # Match bins
    pairs = []
    used_corpus = set()
    used_onnx = set()
    
    # Find common accuracy bins
    common_bins = set(corpus_bins.keys()) & set(onnx_bins.keys())
    
    for bin_val in sorted(common_bins):
        corpus_list = corpus_bins[bin_val]
        onnx_list = onnx_bins[bin_val]
        
        # Pair up as many as possible
        n_pairs = min(len(corpus_list), len(onnx_list))
        for i in range(n_pairs):
            pairs.append((corpus_list[i], onnx_list[i]))
            used_corpus.add(corpus_list[i])
            used_onnx.add(onnx_list[i])
    
    # Find remaining unmatched
    remaining_corpus = [idx for idx in corpus_indices if idx not in used_corpus]
    remaining_onnx = [idx for idx in onnx_indices if idx not in used_onnx]
    
    return pairs, remaining_corpus, remaining_onnx

def main():
    print("="*80)
    print("Merging NASBench-201 corpus with true ONNX encodings")
    print("="*80)
    
    # Load existing corpus
    print(f"\nLoading existing corpus from {EXISTING_CORPUS_PATH}...")
    df_corpus = pd.read_pickle(EXISTING_CORPUS_PATH)
    print(f"  Loaded {len(df_corpus)} architectures")
    print(f"  Columns: {list(df_corpus.columns)}")
    
    # Check for accuracy column
    if 'cifar10-valid_test_accuracy' not in df_corpus.columns:
        print("\nERROR: 'cifar10-valid_test_accuracy' column not found!")
        print(f"Available columns: {list(df_corpus.columns)}")
        return
    
    # Load ONNX CSV
    print(f"\nLoading ONNX encodings from {ONNX_CSV_PATH}...")
    df_onnx = pd.read_csv(ONNX_CSV_PATH)
    print(f"  Loaded {len(df_onnx)} architectures")
    print(f"  Columns: {list(df_onnx.columns)}")
    
    # Verify required columns
    if 'accuracy' not in df_onnx.columns:
        print("\nERROR: 'accuracy' column not found in ONNX CSV!")
        print(f"Available columns: {list(df_onnx.columns)}")
        return
    
    if 'onnx_encoding' not in df_onnx.columns:
        print("\nERROR: 'onnx_encoding' column not found in ONNX CSV!")
        print(f"Available columns: {list(df_onnx.columns)}")
        return
    
    # Extract accuracies
    corpus_accs = df_corpus['cifar10-valid_test_accuracy'].values
    onnx_accs = df_onnx['accuracy'].values
    
    print(f"\nAccuracy statistics:")
    print(f"  Corpus: min={corpus_accs.min():.4f}, max={corpus_accs.max():.4f}, mean={corpus_accs.mean():.4f}")
    print(f"  ONNX:   min={onnx_accs.min():.4f}, max={onnx_accs.max():.4f}, mean={onnx_accs.mean():.4f}")
    
    # Multi-precision matching
    all_pairs = []
    remaining_corpus = list(range(len(df_corpus)))
    remaining_onnx = list(range(len(df_onnx)))
    
    precisions = [0.00001, 0.0001, 0.001, 0.01, 0.1]
    
    for precision in precisions:
        if len(remaining_corpus) == 0 or len(remaining_onnx) == 0:
            break
        
        print(f"\n--- Matching at precision {precision} ---")
        print(f"  Remaining to match: {len(remaining_corpus)} corpus, {len(remaining_onnx)} ONNX")
        
        # Get accuracies for remaining architectures
        corpus_accs_remaining = corpus_accs[remaining_corpus]
        onnx_accs_remaining = onnx_accs[remaining_onnx]
        
        # Match at this precision
        new_pairs, remaining_corpus, remaining_onnx = match_by_accuracy(
            corpus_accs_remaining, remaining_corpus,
            onnx_accs_remaining, remaining_onnx,
            precision
        )
        
        all_pairs.extend(new_pairs)
        print(f"  Matched {len(new_pairs)} pairs")
        print(f"  Still unmatched: {len(remaining_corpus)} corpus, {len(remaining_onnx)} ONNX")
    
    print(f"\n{'='*80}")
    print(f"Matching complete!")
    print(f"  Total pairs: {len(all_pairs)}")
    print(f"  Unmatched corpus architectures: {len(remaining_corpus)}")
    print(f"  Unmatched ONNX architectures: {len(remaining_onnx)}")
    print(f"{'='*80}")
    
    if len(all_pairs) == 0:
        print("\nERROR: No pairs matched! Check if accuracy ranges overlap.")
        return
    
    # Create aligned dataframes
    print(f"\nCreating aligned corpus...")
    corpus_indices = [p[0] for p in all_pairs]
    onnx_indices = [p[1] for p in all_pairs]
    
    # Create new corpus with matched rows
    df_aligned = df_corpus.iloc[corpus_indices].reset_index(drop=True).copy()
    
    # Add ONNX encoding column
    df_aligned['true_onnx_encoding'] = df_onnx.iloc[onnx_indices]['onnx_encoding'].values
    
    # Verify accuracy alignment
    corpus_matched_accs = df_aligned['cifar10-valid_test_accuracy'].values
    onnx_matched_accs = df_onnx.iloc[onnx_indices]['accuracy'].values
    acc_diff = np.abs(corpus_matched_accs - onnx_matched_accs)
    
    print(f"\nAccuracy alignment verification:")
    print(f"  Mean absolute difference: {acc_diff.mean():.6f}")
    print(f"  Max absolute difference: {acc_diff.max():.6f}")
    print(f"  Median absolute difference: {np.median(acc_diff):.6f}")
    
    # Show some examples
    print(f"\nExample pairs (first 5):")
    for i in range(min(5, len(all_pairs))):
        corpus_acc = corpus_matched_accs[i]
        onnx_acc = onnx_matched_accs[i]
        diff = acc_diff[i]
        print(f"  Pair {i+1}: Corpus={corpus_acc:.4f}, ONNX={onnx_acc:.4f}, Diff={diff:.6f}")
    
    # Save aligned corpus
    print(f"\nSaving aligned corpus to {OUTPUT_PATH}...")
    df_aligned.to_pickle(OUTPUT_PATH)
    
    # Also save CSV version
    csv_path = OUTPUT_PATH.replace('.pkl', '.csv')
    print(f"Saving CSV version to {csv_path}...")
    df_aligned.to_csv(csv_path, index=False)
    
    print(f"\n{'='*80}")
    print(f"Success!")
    print(f"  Aligned corpus size: {len(df_aligned)}")
    print(f"  Columns: {list(df_aligned.columns)}")
    print(f"  New column 'true_onnx_encoding' added")
    print(f"  Saved to: {OUTPUT_PATH}")
    print(f"{'='*80}")

if __name__ == '__main__':
    main()
