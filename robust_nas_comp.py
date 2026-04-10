"""
NASBench-201 robust comparison with CodeLlama and ModernBERT embeddings.
Safe to run multiple times - will only compute what's missing
"""

from robust_surrogate_predict import run_comparison
from embed_corpus import add_embeddings_to_corpus

# Paths
NASBENCH_CORPUS_PATH = '/storage/ice-shared/vip-vvk/data/AOT/psomu3/codenas/nasbench201_corpus_embedded.pkl'
NASBENCH_CORPUS_OUTPUT_PATH = '/storage/ice-shared/vip-vvk/data/AOT/psomu3/codenas/nasbench201_corpus_embedded_pytorch.pkl'
OUTPUT_PATH = '/storage/ice-shared/vip-vvk/data/AOT/psomu3/codenas/zca/nasbench_zca_comparison.csv'
OUTPUT_PATH_CSV = '/storage/ice-shared/vip-vvk/data/AOT/psomu3/codenas/zca'
COMPARISON_LABEL = 'nasbench201'

# Configuration
# SAMPLE_SIZES = [15, 50, 150, 500, 1500, 5000]  # Training set sizes to test
SAMPLE_SIZES = [14, 55, 220, 879, 3516, 14062]  # Training set sizes to test
# SAMPLE_SIZES = [8, 39, 78, 224, 896, 3584, 14062]  # Training set sizes to test
N_FOLDS = 10       # 10-fold cross-validation
N_REPEATS = 20    # If Repeat CV 50 times, then 500 total trials per sample size
FORCE = False     # If True, recompute everything; if False, only add missing data

# df = add_embeddings_to_corpus(
#     corpus_path=NASBENCH_CORPUS_PATH,
#     model_name='codellama_python_7b',
#     output_path=NASBENCH_CORPUS_PATH,
#     pytorch_only=True,
#     use_echo_embeddings=False,
#     device='cuda',
#     max_length=512
# )
# df = add_embeddings_to_corpus(
#     corpus_path=NASBENCH_CORPUS_PATH,
#     model_name='codellama_instruct_7b',
#     output_path=NASBENCH_CORPUS_PATH,
#     pytorch_only=True,
#     use_echo_embeddings=False,
#     device='cuda',
#     max_length=512
# )

# # Run robust comparison (only computes missing trials)
# print("\n" + "=" * 80)
# print("NASBench-201 Robust Comparison: codellama_python_7b_pytorch_code_with_network_embedding vs codellama_python_7b_pytorch_code_embedding")
# print(f"  Sample sizes: {SAMPLE_SIZES}")
# print(f"  CV setup: {N_FOLDS}-fold × {N_REPEATS} repeats = {N_FOLDS * N_REPEATS} trials per size")
# print("  (Will only compute missing trials, preserves existing results)")
# print("=" * 80)

# results_df = run_comparison(
#     embedding1_name='codellama_python_7b_pytorch_code_with_network_embedding',
#     corpus1_name='nasbench201',
#     embedding2_name='codellama_python_7b_pytorch_code_embedding',
#     corpus2_name='nasbench201',
#     corpus_path1=NASBENCH_CORPUS_OUTPUT_PATH,
#     corpus_path2=NASBENCH_CORPUS_PATH,
#     comparison_label=COMPARISON_LABEL,
#     sample_sizes=SAMPLE_SIZES,
#     n_folds=N_FOLDS,
#     n_repeats=N_REPEATS,
#     benchmark_type='nasbench',
#     comparison_output_path=OUTPUT_PATH,
#     per_embedding_output_dir=OUTPUT_PATH_CSV,
#     device='cuda',
#     force=FORCE
# )

# print("XGB config: previously lr 0.05, n_estimators=500. now lr 0.01, n_estimators 2000")
# print("XGB config: prev did not define min_child_weight or gamma. now min_child_weight=5, gamma=0.1")


# Run robust comparison (only computes missing trials)
print("\n" + "=" * 80)
print("NASBench-201 Robust Comparison: codellama_python_7b_pytorch_code_embedding with pairwise loss, single target, zca 0.8, pca 128 vs codellama_python_7b_pytorch_code_embedding with pairwise loss, single target, zca 0.8, pca 256")
print(f"  Sample sizes: {SAMPLE_SIZES}")
print(f"  CV setup: {N_FOLDS}-fold × {N_REPEATS} repeats = {N_FOLDS * N_REPEATS} trials per size")
print("  (Will only compute missing trials, preserves existing results)")
print("=" * 80)

results_df = run_comparison(
    embedding1_name='codellama_python_7b_pytorch_code_embedding',
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
    dim_reduction_method_embedding1='pca',
    dim_reduction_components_embedding1=128,
    dim_reduction_method_embedding2='pca',
    dim_reduction_components_embedding2=256,
    apply_zca_to_embedding1=True,
    zca_epsilon_embedding1=0.8,
    apply_zca_to_embedding2=True,
    zca_epsilon_embedding2=0.8,
    use_pairwise_loss_embedding1=True,
    use_pairwise_loss_embedding2=True,
    use_single_target_embedding1=True,
    use_single_target_embedding2=True,
    head_type_embedding1='mlp',
    head_type_embedding2='mlp'
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
