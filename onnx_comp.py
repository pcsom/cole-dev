"""
NASBench-201 robust comparison with CodeLlama and ModernBERT embeddings.
Safe to run multiple times - will only compute what's missing
"""

from robust_surrogate_predict import run_comparison
from embed_corpus import add_embeddings_to_corpus

# Paths
model = "modernbert_large"
NASBENCH_CORPUS_PATH = '/storage/ice-shared/vip-vvk/data/AOT/psomu3/codenas/nasbench201_corpus_onnx_paper.pkl'
NASBENCH_CORPUS_OUTPUT_PATH = '/storage/ice-shared/vip-vvk/data/AOT/psomu3/codenas2/nasbench201_corpus_onnx_paper_embedded.pkl'
OUTPUT_PATH = f'/storage/ice-shared/vip-vvk/data/AOT/psomu3/codenas2/onnx_paper_{model}/nasbench_onnx_comparison.csv'
OUTPUT_PATH_CSV = f'/storage/ice-shared/vip-vvk/data/AOT/psomu3/codenas2/onnx_paper_{model}'
COMPARISON_LABEL = 'nasbench201'

# Configuration
# SAMPLE_SIZES = [15, 50, 150, 500, 1500, 5000]  # Training set sizes to test
# SAMPLE_SIZES = [14, 56, 224, 896, 3584, 14062]  # Training set sizes to test
# SAMPLE_SIZES = [8, 39, 78, 224, 896]#, 3584, 14062]  # Training set sizes to test
SAMPLE_SIZES = [14, 55, 220, 879, 3516]#, 6660]  # Training set sizes to test
N_FOLDS = 10       # 10-fold cross-validation
N_REPEATS = 10    # If Repeat CV 50 times, then 500 total trials per sample size
FORCE = False     # If True, recompute everything; if False, only add missing data

# df = add_embeddings_to_corpus(
#     corpus_path=NASBENCH_CORPUS_PATH,
#     model_name=model,
#     output_path=NASBENCH_CORPUS_OUTPUT_PATH,
#     pytorch_all=True,
#     pytorch_context_mode='network',
#     use_echo_embeddings=False,
#     quantization='fp16',
#     device='cuda',
#     max_length=4096,
#     pooling_mode='mean'
# )
# df = add_embeddings_to_corpus(
#     corpus_path=NASBENCH_CORPUS_PATH,
#     model_name=model,
#     output_path=NASBENCH_CORPUS_OUTPUT_PATH,
#     onnx_only=True,
#     use_echo_embeddings=False,
#     quantization='fp16',
#     device='cuda',
#     max_length=4096,
#     pooling_mode='mean'
# )


# Run robust comparison (only computes missing trials)

results_df = run_comparison(
    embedding1_name=f'{model}_true_onnx_encoding_fp16_embedding',
    corpus1_name='onnx',
    embedding2_name=f'{model}_pytorch_code_exclude_helper_with_network_fp16_embedding',
    corpus2_name='onnx',
    corpus_path1=NASBENCH_CORPUS_OUTPUT_PATH,
    corpus_path2=NASBENCH_CORPUS_OUTPUT_PATH,
    comparison_label=COMPARISON_LABEL,
    sample_sizes=SAMPLE_SIZES,
    n_folds=N_FOLDS,
    n_repeats=N_REPEATS,
    benchmark_type='nasbench',
    comparison_output_path=OUTPUT_PATH,
    per_embedding_output_dir=OUTPUT_PATH_CSV,
    device='cuda',
    force=FORCE,
    dim_reduction_method_embedding1='softpca',
    dim_reduction_components_embedding1=128,
    dim_reduction_method_embedding2='softpca',
    dim_reduction_components_embedding2=128,
    apply_zca_to_embedding1=False,
    apply_zca_to_embedding2=False,
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
    embedding1_name=f'{model}_pytorch_code_inline_with_network_fp16_embedding',
    corpus1_name='onnx',
    embedding2_name=f'{model}_pytorch_code_helper_with_network_fp16_embedding',
    corpus2_name='onnx',
    corpus_path1=NASBENCH_CORPUS_OUTPUT_PATH,
    corpus_path2=NASBENCH_CORPUS_OUTPUT_PATH,
    comparison_label=COMPARISON_LABEL,
    sample_sizes=SAMPLE_SIZES,
    n_folds=N_FOLDS,
    n_repeats=N_REPEATS,
    benchmark_type='nasbench',
    comparison_output_path=OUTPUT_PATH,
    per_embedding_output_dir=OUTPUT_PATH_CSV,
    device='cuda',
    force=FORCE,
    dim_reduction_method_embedding1='softpca',
    dim_reduction_components_embedding1=128,
    dim_reduction_method_embedding2='softpca',
    dim_reduction_components_embedding2=128,
    apply_zca_to_embedding1=False,
    apply_zca_to_embedding2=False,
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
