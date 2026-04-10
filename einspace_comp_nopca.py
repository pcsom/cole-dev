"""
NASBench-201 robust comparison with CodeLlama and ModernBERT embeddings.
Safe to run multiple times - will only compute what's missing
"""

from robust_surrogate_predict import run_comparison
from embed_corpus import add_embeddings_to_corpus

# Paths
CORPUS_PATH = '/storage/ice-shared/vip-vvk/data/AOT/psomu3/codenas/einspace/einspace_corpus_dedup.pkl'
OUTPUT_CORPUS_PATH = '/storage/ice-shared/vip-vvk/data/AOT/psomu3/codenas2/einspace/einspace_corpus_dedup_embedded.pkl'
OUTPUT_PATH = '/storage/ice-shared/vip-vvk/data/AOT/psomu3/codenas2/einspace_nopca/nasbench_einspace_comparison.csv'
OUTPUT_PATH_CSV = '/storage/ice-shared/vip-vvk/data/AOT/psomu3/codenas2/einspace_nopca'
COMPARISON_LABEL = 'einspace'

# Configuration
# SAMPLE_SIZES = [15, 50, 150, 500, 1500, 5000]  # Training set sizes to test
# SAMPLE_SIZES = [14, 56, 224, 896, 3584, 14062]  # Training set sizes to test
# SAMPLE_SIZES = [10, 40, 160, 638, 2553]  # Training set sizes to test
SAMPLE_SIZES = [14, 55, 220, 879, 2553]  # Training set sizes to test
N_FOLDS = 10       # 10-fold cross-validation
N_REPEATS = 10    # If Repeat CV 50 times, then 500 total trials per sample size
FORCE = False     # If True, recompute everything; if False, only add missing data

# df = add_embeddings_to_corpus(
#     corpus_path=CORPUS_PATH,
#     model_name='codellama_python_7b',
#     output_path=OUTPUT_CORPUS_PATH,
#     grammar_only=True,
#     use_echo_embeddings=False,
#     quantization='fp16',
#     device='cuda',
#     max_length=4096,
#     pooling_mode='mean'
# )
# df = add_embeddings_to_corpus(
#     corpus_path=CORPUS_PATH,
#     model_name='codellama_python_7b',
#     output_path=OUTPUT_CORPUS_PATH,
#     pytorch_all=True,
#     use_echo_embeddings=False,
#     quantization='fp16',
#     device='cuda',
#     max_length=4096,
#     pooling_mode='mean'
# )
# df = add_embeddings_to_corpus(
#     corpus_path=CORPUS_PATH,
#     model_name='modernbert_large',
#     output_path=OUTPUT_CORPUS_PATH,
#     grammar_only=True,
#     use_echo_embeddings=False,
#     quantization='fp16',
#     device='cuda',
#     max_length=4096,
#     pooling_mode='mean'
# )
# df = add_embeddings_to_corpus(
#     corpus_path=CORPUS_PATH,
#     model_name='modernbert_large',
#     output_path=OUTPUT_CORPUS_PATH,
#     pytorch_all=True,
#     use_echo_embeddings=False,
#     quantization='fp16',
#     device='cuda',
#     max_length=4096,
#     pooling_mode='mean'
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

results_df = run_comparison(
    embedding1_name=f'modernbert_large_grammar_code_fp16_embedding',
    corpus1_name='einspace',
    embedding2_name=f'modernbert_large_pytorch_code_fp16_embedding',
    corpus2_name='einspace',
    corpus_path1=OUTPUT_CORPUS_PATH,
    corpus_path2=None,
    comparison_label=COMPARISON_LABEL,
    sample_sizes=SAMPLE_SIZES,
    n_folds=N_FOLDS,
    n_repeats=N_REPEATS,
    benchmark_type='einspace',
    comparison_output_path=OUTPUT_PATH,
    per_embedding_output_dir=OUTPUT_PATH_CSV,
    device='cuda',
    force=FORCE,
    dim_reduction_method_embedding1=None,
    dim_reduction_components_embedding1=128,
    dim_reduction_method_embedding2=None,
    dim_reduction_components_embedding2=128,
    apply_zca_to_embedding1=False,
    zca_epsilon_embedding1=1,
    apply_zca_to_embedding2=False,
    zca_epsilon_embedding2=0.75,
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

results_df = run_comparison(
    embedding1_name=f'codellama_python_7b_grammar_code_fp16_embedding',
    corpus1_name='einspace',
    embedding2_name=f'codellama_python_7b_pytorch_code_fp16_embedding',
    corpus2_name='einspace',
    corpus_path1=OUTPUT_CORPUS_PATH,
    corpus_path2=None,
    comparison_label=COMPARISON_LABEL,
    sample_sizes=SAMPLE_SIZES,
    n_folds=N_FOLDS,
    n_repeats=N_REPEATS,
    benchmark_type='einspace',
    comparison_output_path=OUTPUT_PATH,
    per_embedding_output_dir=OUTPUT_PATH_CSV,
    device='cuda',
    force=FORCE,
    dim_reduction_method_embedding1=None,
    dim_reduction_components_embedding1=128,
    dim_reduction_method_embedding2=None,
    dim_reduction_components_embedding2=128,
    apply_zca_to_embedding1=False,
    zca_epsilon_embedding1=1,
    apply_zca_to_embedding2=False,
    zca_epsilon_embedding2=0.75,
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
