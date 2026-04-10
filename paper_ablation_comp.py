"""
NASBench-201 robust comparison with CodeLlama and ModernBERT embeddings.
Safe to run multiple times - will only compute what's missing
"""

from robust_surrogate_predict import run_comparison
from embed_corpus import add_embeddings_to_corpus
import os
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Compare two LLM embeddings for NAS')
parser.add_argument('--mode', type=str, default='main', help='Choices: main, helper, context, quantization, pooling, head, loss, pca, pca2, loss-head')
# parser.add_argument('--model2', type=str, default='modernbert_large', help='Second model name')
args = parser.parse_args()

# grab user environment variable
user = os.getenv('USER')


model = "codellama_python_7b"
pca_components = 128
quant_arg = 'fp16'


# Paths
MODE = args.mode
COMPARISON_LABEL = f'{model}_{MODE}'
SUFFIX = '4'
# os makedir
os.makedirs(f'/storage/ice-shared/vip-vvk/data/AOT/psomu3/codenas{SUFFIX}', exist_ok=True)
NASBENCH_CORPUS_PATH = '/storage/ice-shared/vip-vvk/data/AOT/psomu3/codenas/nasbench201_corpus_pytorch_ablation.pkl'
if MODE in ['helper', 'context', 'quantization', 'pooling']:
    NASBENCH_CORPUS_OUTPUT_PATH = f'/storage/ice-shared/vip-vvk/data/AOT/psomu3/codenas{SUFFIX}/nasbench201_corpus_pytorch_embedded_ablation_{COMPARISON_LABEL}.pkl'
else:
    NASBENCH_CORPUS_OUTPUT_PATH = f'/storage/ice-shared/vip-vvk/data/AOT/psomu3/codenas{SUFFIX}/nasbench201_corpus_pytorch_embedded_ablation_{model}_main.pkl'
OUTPUT_PATH = f'/storage/ice-shared/vip-vvk/data/AOT/{user}/codenas{SUFFIX}/{COMPARISON_LABEL}/nasbench_ablation_comparison.csv'
OUTPUT_PATH_CSV = f'/storage/ice-shared/vip-vvk/data/AOT/{user}/codenas{SUFFIX}/{COMPARISON_LABEL}'

# Configuration
# SAMPLE_SIZES = [15, 50, 150, 500, 1500, 5000]  # Training set sizes to test
SAMPLE_SIZES = [14, 55, 220, 879, 3516]  # Training set sizes to test
# SAMPLE_SIZES = [8, 39, 78, 224, 896, 3584, 14062]  # Training set sizes to test
N_FOLDS = 10       # 10-fold cross-validation
N_REPEATS = 20    # If Repeat CV 50 times, then 500 total trials per sample size
FORCE = False     # If True, recompute everything; if False, only add missing data

# MAIN
if MODE == 'main' or MODE == 'pca':
    df = add_embeddings_to_corpus(
        corpus_path=NASBENCH_CORPUS_PATH,
        model_name=model,
        output_path=NASBENCH_CORPUS_OUTPUT_PATH,
        pytorch_only=True,
        use_echo_embeddings=False,
        quantization=quant_arg,
        device='cuda',
        max_length=512,
        pooling_mode='mean'
    )


# POOLING ABLATION
if MODE == 'pooling':
    df = add_embeddings_to_corpus(
        corpus_path=NASBENCH_CORPUS_PATH,
        model_name=model,
        output_path=NASBENCH_CORPUS_OUTPUT_PATH,
        pytorch_only=True,
        use_echo_embeddings=False,
        quantization=quant_arg,
        device='cuda',
        max_length=512,
        pooling_mode='avg_avg'
    )
    df = add_embeddings_to_corpus(
        corpus_path=NASBENCH_CORPUS_PATH,
        model_name=model,
        output_path=NASBENCH_CORPUS_OUTPUT_PATH,
        pytorch_only=True,
        use_echo_embeddings=True,
        quantization=quant_arg,
        device='cuda',
        max_length=512,
        pooling_mode='mean'
    )


# HELPER METHOD MODE ABLATION
if MODE == 'helper':
    df = add_embeddings_to_corpus(
        corpus_path=NASBENCH_CORPUS_PATH,
        model_name=model,
        output_path=NASBENCH_CORPUS_OUTPUT_PATH,
        pytorch_only=True,
        pytorch_all=True,
        use_echo_embeddings=False,
        quantization=quant_arg,
        device='cuda',
        max_length=1024,
        pooling_mode='mean'
    )


# CONTEXT MODE ABLATION
if MODE == 'context':
    df = add_embeddings_to_corpus(
        corpus_path=NASBENCH_CORPUS_PATH,
        model_name=model,
        output_path=NASBENCH_CORPUS_OUTPUT_PATH,
        pytorch_only=True,
        use_echo_embeddings=False,
        quantization=quant_arg,
        pytorch_context_mode='network',
        device='cuda',
        max_length=4096,
        pooling_mode='mean'
    )
    df = add_embeddings_to_corpus(
        corpus_path=NASBENCH_CORPUS_PATH,
        model_name=model,
        output_path=NASBENCH_CORPUS_OUTPUT_PATH,
        pytorch_only=True,
        use_echo_embeddings=False,
        quantization=quant_arg,
        pytorch_context_mode='comment',
        device='cuda',
        max_length=4096,
        pooling_mode='mean'
    )


# QUANTIZATION ABLATION
if MODE == 'quantization':
    df = add_embeddings_to_corpus(
        corpus_path=NASBENCH_CORPUS_PATH,
        model_name=model,
        output_path=NASBENCH_CORPUS_OUTPUT_PATH,
        pytorch_only=True,
        use_echo_embeddings=False,
        quantization='int8',
        device='cuda',
        max_length=512,
        pooling_mode='mean'
    )
    df = add_embeddings_to_corpus(
        corpus_path=NASBENCH_CORPUS_PATH,
        model_name=model,
        output_path=NASBENCH_CORPUS_OUTPUT_PATH,
        pytorch_only=True,
        use_echo_embeddings=False,
        quantization='fp16',
        device='cuda',
        max_length=512,
        pooling_mode='mean'
    )





# Run robust comparison (only computes missing trials)
print("\n" + "=" * 80)
print("NASBench-201 Robust Comparison:")
print(f"  Sample sizes: {SAMPLE_SIZES}")
print(f"  CV setup: {N_FOLDS}-fold × {N_REPEATS} repeats = {N_FOLDS * N_REPEATS} trials per size")
print("  (Will only compute missing trials, preserves existing results)")
print("=" * 80)

if MODE == 'loss':
    results_df = run_comparison(
        embedding1_name=f'{model}_pytorch_code_exclude_helper_{quant_arg}_embedding',
        corpus1_name='ablation',
        embedding2_name=f'{model}_pytorch_code_exclude_helper_{quant_arg}_embedding',
        corpus2_name='ablation',
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
        dim_reduction_components_embedding1=pca_components,
        dim_reduction_method_embedding2='softpca',
        dim_reduction_components_embedding2=pca_components,
        apply_zca_to_embedding1=False,
        apply_zca_to_embedding2=False,
        use_pairwise_loss_embedding1=False,
        use_pairwise_loss_embedding2=True,
        use_single_target_embedding1=True,
        use_single_target_embedding2=True,
        head_type_embedding1='mlp',
        head_type_embedding2='mlp'
    )
elif MODE == 'head':
    results_df = run_comparison(
        embedding1_name=f'{model}_pytorch_code_exclude_helper_{quant_arg}_embedding',
        corpus1_name='ablation',
        embedding2_name=f'{model}_pytorch_code_exclude_helper_{quant_arg}_embedding',
        corpus2_name='ablation',
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
        dim_reduction_method_embedding1=None,
        dim_reduction_components_embedding1=pca_components,
        dim_reduction_method_embedding2='softpca',
        dim_reduction_components_embedding2=pca_components,
        apply_zca_to_embedding1=False,
        apply_zca_to_embedding2=False,
        use_pairwise_loss_embedding1=False,
        use_pairwise_loss_embedding2=True,
        use_single_target_embedding1=True,
        use_single_target_embedding2=True,
        head_type_embedding1='xgboost',
        head_type_embedding2='mlp'
    )
elif MODE == 'loss-head':
    results_df = run_comparison(
        embedding1_name=f'{model}_pytorch_code_exclude_helper_{quant_arg}_embedding',
        corpus1_name='ablation',
        embedding2_name=f'{model}_pytorch_code_exclude_helper_{quant_arg}_embedding',
        corpus2_name='ablation',
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
        dim_reduction_method_embedding1=None,
        dim_reduction_components_embedding1=pca_components,
        dim_reduction_method_embedding2='softpca',
        dim_reduction_components_embedding2=pca_components,
        apply_zca_to_embedding1=False,
        apply_zca_to_embedding2=False,
        use_pairwise_loss_embedding1=False,
        use_pairwise_loss_embedding2=False,
        use_single_target_embedding1=True,
        use_single_target_embedding2=True,
        head_type_embedding1='xgboost',
        head_type_embedding2='mlp'
    )
elif MODE == 'pca':
    results_df = run_comparison(
        embedding1_name=f'{model}_pytorch_code_exclude_helper_{quant_arg}_embedding',
        corpus1_name='ablation',
        embedding2_name=f'{model}_pytorch_code_exclude_helper_{quant_arg}_embedding',
        corpus2_name='ablation',
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
        dim_reduction_method_embedding1=None,
        dim_reduction_components_embedding1=None,
        dim_reduction_method_embedding2='softpca',
        dim_reduction_components_embedding2=pca_components,
        apply_zca_to_embedding1=False,
        apply_zca_to_embedding2=False,
        use_pairwise_loss_embedding1=True,
        use_pairwise_loss_embedding2=True,
        use_single_target_embedding1=True,
        use_single_target_embedding2=True,
        head_type_embedding1='mlp',
        head_type_embedding2='mlp'
    )
elif MODE == 'helper':
    results_df = run_comparison(
        embedding1_name=f'{model}_pytorch_code_inline_{quant_arg}_embedding',
        corpus1_name='ablation',
        embedding2_name=f'{model}_pytorch_code_helper_{quant_arg}_embedding',
        corpus2_name='ablation',
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
        dim_reduction_components_embedding1=pca_components,
        dim_reduction_method_embedding2='softpca',
        dim_reduction_components_embedding2=pca_components,
        apply_zca_to_embedding1=False,
        apply_zca_to_embedding2=False,
        use_pairwise_loss_embedding1=True,
        use_pairwise_loss_embedding2=True,
        use_single_target_embedding1=True,
        use_single_target_embedding2=True,
        head_type_embedding1='mlp',
        head_type_embedding2='mlp'
    )
elif MODE == 'context':
    results_df = run_comparison(
        embedding1_name=f'{model}_pytorch_code_exclude_helper_with_network_{quant_arg}_embedding',
        corpus1_name='ablation',
        embedding2_name=f'{model}_pytorch_code_exclude_helper_with_comment_{quant_arg}_embedding',
        corpus2_name='ablation',
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
        dim_reduction_components_embedding1=pca_components,
        dim_reduction_method_embedding2='softpca',
        dim_reduction_components_embedding2=pca_components,
        apply_zca_to_embedding1=False,
        apply_zca_to_embedding2=False,
        use_pairwise_loss_embedding1=True,
        use_pairwise_loss_embedding2=True,
        use_single_target_embedding1=True,
        use_single_target_embedding2=True,
        head_type_embedding1='mlp',
        head_type_embedding2='mlp'
    )
elif MODE == 'pooling':
    results_df = run_comparison(
        embedding1_name=f'{model}_pytorch_code_exclude_helper_echo_{quant_arg}_embedding',
        corpus1_name='ablation',
        embedding2_name=f'{model}_pytorch_code_exclude_helper_{quant_arg}_avg_avg_embedding',
        corpus2_name='ablation',
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
        dim_reduction_components_embedding1=pca_components,
        dim_reduction_method_embedding2='softpca',
        dim_reduction_components_embedding2=pca_components,
        apply_zca_to_embedding1=False,
        apply_zca_to_embedding2=False,
        use_pairwise_loss_embedding1=True,
        use_pairwise_loss_embedding2=True,
        use_single_target_embedding1=True,
        use_single_target_embedding2=True,
        head_type_embedding1='mlp',
        head_type_embedding2='mlp'
    )
elif MODE == 'quantization':
    results_df = run_comparison(
        embedding1_name=f'{model}_pytorch_code_exclude_helper_int8_embedding',
        corpus1_name='ablation',
        embedding2_name=f'{model}_pytorch_code_exclude_helper_fp16_embedding',
        corpus2_name='ablation',
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
        dim_reduction_components_embedding1=pca_components,
        dim_reduction_method_embedding2='softpca',
        dim_reduction_components_embedding2=pca_components,
        apply_zca_to_embedding1=False,
        apply_zca_to_embedding2=False,
        use_pairwise_loss_embedding1=True,
        use_pairwise_loss_embedding2=True,
        use_single_target_embedding1=True,
        use_single_target_embedding2=True,
        head_type_embedding1='mlp',
        head_type_embedding2='mlp'
    )
elif MODE == 'pca2':
    results_df = run_comparison(
        embedding1_name=f'{model}_pytorch_code_exclude_helper_{quant_arg}_embedding',
        corpus1_name='ablation',
        embedding2_name=f'{model}_pytorch_code_exclude_helper_{quant_arg}_embedding',
        corpus2_name='ablation',
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
        dim_reduction_components_embedding1=512,
        dim_reduction_method_embedding2='softpca',
        dim_reduction_components_embedding2=32,
        apply_zca_to_embedding1=False,
        apply_zca_to_embedding2=False,
        use_pairwise_loss_embedding1=True,
        use_pairwise_loss_embedding2=True,
        use_single_target_embedding1=True,
        use_single_target_embedding2=True,
        head_type_embedding1='mlp',
        head_type_embedding2='mlp'
    )



if results_df is not None:
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
