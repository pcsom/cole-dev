"""
JAHS-Bench robust comparison with CodeLlama and ModernBERT embeddings.
Safe to run multiple times - will only compute what's missing
"""

from embed_corpus import add_embeddings_to_corpus
from robust_surrogate_predict import run_robust_comparison

# Paths
JAHS_CORPUS_PATH = '/storage/ice-shared/vip-vvk/data/AOT/psomu3/codenas/jahsbench201_corpus.pkl'
OUTPUT_PATH = '/storage/ice-shared/vip-vvk/data/AOT/psomu3/codenas/jahs_robust_comparison_10_kt_lrelu.csv'

# Configuration
SAMPLE_SIZES = [15, 50, 150, 500, 1500, 5000]  # Training set sizes to test
N_FOLDS = 10       # 10-fold cross-validation
N_REPEATS = 20     # Repeat CV 5 times = 25 total trials per sample size
FORCE = False     # If True, recompute everything; if False, only add missing data

# Step 1: Add embeddings to JAHS corpus (skips if already present)
print("=" * 80)
print("STEP 1: Adding codestral_7b embeddings to JAHS corpus")
print("  (Will skip if embeddings already exist)")
print("=" * 80)

add_embeddings_to_corpus(
    corpus_path=JAHS_CORPUS_PATH,
    model_name='codestral_7b',
    output_path=JAHS_CORPUS_PATH,
    force=FORCE,  # Only recompute if explicitly requested
    pytorch_only=True  # Only use PyTorch code representation
)

print("=" * 80)
print("STEP 2: Adding modernbert_large embeddings to JAHS corpus")
print("  (Will skip if embeddings already exist)")
print("=" * 80)

add_embeddings_to_corpus(
    corpus_path=JAHS_CORPUS_PATH,
    model_name='modernbert_large',
    output_path=JAHS_CORPUS_PATH,
    force=FORCE,  # Only recompute if explicitly requested
    pytorch_only=True  # Only use PyTorch code representation
)

# Run robust comparison (only computes missing trials)
print("\n" + "=" * 80)
print("Running robust comparison with corrected paired t-test")
print(f"  Sample sizes: {SAMPLE_SIZES}")
print(f"  CV setup: {N_FOLDS}-fold × {N_REPEATS} repeats = {N_FOLDS * N_REPEATS} trials per size")
print("  (Will only compute missing trials, preserves existing results)")
print("=" * 80)

results_df = run_robust_comparison(
    corpus_path=JAHS_CORPUS_PATH,
    embedding_types=['modernbert_large_pytorch_code_embedding', 'codestral_7b_pytorch_code_embedding'],
    sample_sizes=SAMPLE_SIZES,
    n_folds=N_FOLDS,
    n_repeats=N_REPEATS,
    benchmark_type='jahs',
    output_path=OUTPUT_PATH,
    device='cuda',
    force=FORCE  # Only recompute if explicitly requested
)

print("\n" + "=" * 80)
print("FINAL RESULTS SUMMARY")
print("=" * 80)
if len(results_df) > 0:
    print(results_df[['sample_size', 'model1', 'model2', 'metric', 'model1_mean', 'model2_mean', 'mean_diff', 'p_value', 'significant', 'n_trials']])
    print(f"\nTotal result rows: {len(results_df)}")
    print(f"Results saved to {OUTPUT_PATH}")
else:
    print("No new results computed - all comparisons already complete!")
print("=" * 80)
