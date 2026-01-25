"""
NASBench-201 robust comparison with CodeLlama and ModernBERT embeddings.
Safe to run multiple times - will only compute what's missing
"""

from robust_surrogate_predict import run_comparison
from embed_corpus import add_embeddings_to_corpus

# Paths
NASBENCH_CORPUS_PATH = '/storage/ice-shared/vip-vvk/data/AOT/psomu3/codenas/nasbench201_corpus_onnx.pkl'
NASBENCH_CORPUS_OUTPUT_PATH = '/storage/ice-shared/vip-vvk/data/AOT/psomu3/codenas/nasbench201_corpus_embedded_pytorch.pkl'
OUTPUT_PATH = '/storage/ice-shared/vip-vvk/data/AOT/psomu3/codenas/refactor/nasbench_onnx_comparison.csv'
OUTPUT_PATH_CSV = '/storage/ice-shared/vip-vvk/data/AOT/psomu3/codenas/refactor'
COMPARISON_LABEL = 'nasbench201'

# Configuration
# SAMPLE_SIZES = [15, 50, 150, 500, 1500, 5000]  # Training set sizes to test
# SAMPLE_SIZES = [14, 56, 224, 896, 3584, 14062]  # Training set sizes to test
SAMPLE_SIZES = [8, 39, 78, 224, 896]#, 3584, 14062]  # Training set sizes to test
N_FOLDS = 10       # 10-fold cross-validation
N_REPEATS = 10    # If Repeat CV 50 times, then 500 total trials per sample size
FORCE = False     # If True, recompute everything; if False, only add missing data

df = add_embeddings_to_corpus(
    corpus_path=NASBENCH_CORPUS_PATH,
    model_name='modernbert_large',
    output_path=NASBENCH_CORPUS_PATH,
    onnx_only=True,
    device='cuda',
    max_length=2048
)
df = add_embeddings_to_corpus(
    corpus_path=NASBENCH_CORPUS_PATH,
    model_name='modernbert_large',
    output_path=NASBENCH_CORPUS_PATH,
    pytorch_only=True,
    device='cuda',
    max_length=512
)
df = add_embeddings_to_corpus(
    corpus_path=NASBENCH_CORPUS_PATH,
    model_name='codellama_python_7b',
    output_path=NASBENCH_CORPUS_PATH,
    onnx_only=True,
    device='cuda',
    max_length=2048
)
df = add_embeddings_to_corpus(
    corpus_path=NASBENCH_CORPUS_PATH,
    model_name='codellama_python_7b',
    output_path=NASBENCH_CORPUS_PATH,
    pytorch_only=True,
    device='cuda',
    max_length=512
)


# Run robust comparison (only computes missing trials)
print("\n" + "=" * 80)
print("NASBench-201 Robust Comparison: modernbert_large_true_onnx_encoding_embedding with pairwise loss, pca, single target vs codellama_python_7b_pytorch_code_embedding with pairwise loss, pca, single target")
print(f"  Sample sizes: {SAMPLE_SIZES}")
print(f"  CV setup: {N_FOLDS}-fold × {N_REPEATS} repeats = {N_FOLDS * N_REPEATS} trials per size")
print("  (Will only compute missing trials, preserves existing results)")
print("=" * 80)

results_df = run_comparison(
    embedding1_name='modernbert_large_true_onnx_encoding_embedding',
    corpus1_name='nasbench201',
    embedding2_name='codellama_python_7b_pytorch_code_embedding',
    corpus2_name='nasbench201',
    corpus_path1=NASBENCH_CORPUS_PATH,
    corpus_path2=NASBENCH_CORPUS_PATH,
    comparison_label=COMPARISON_LABEL,
    sample_sizes=SAMPLE_SIZES,
    n_folds=N_FOLDS,
    n_repeats=N_REPEATS,
    benchmark_type='nasbench',
    comparison_output_path=OUTPUT_PATH,
    per_embedding_output_dir=OUTPUT_PATH_CSV,
    device='cuda',
    force=FORCE,
    apply_pca_to_embedding1=True,
    pca_n_components_embedding1=128,
    apply_pca_to_embedding2=True,
    pca_n_components_embedding2=128,
    use_pairwise_loss_embedding1=True,
    use_pairwise_loss_embedding2=True,
    use_single_target_embedding1=True,
    use_single_target_embedding2=True
)


print("\n" + "=" * 80)
print("FINAL RESULTS SUMMARY")
print("=" * 80)
if len(results_df) > 0:
    print(results_df[['sample_size', 'model1', 'model2', 'model1_mean_ktau', 'model2_mean_ktau', 'mean_diff_ktau', 'p_value_ktau', 'significant_ktau', 'n_trials']])
    print(f"\nTotal result rows: {len(results_df)}")
    print(f"Results saved to {OUTPUT_PATH}")
else:
    print("No new results computed - all comparisons already complete!")
print("=" * 80)
