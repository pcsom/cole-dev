"""
NASBench-201 robust comparison with CodeLlama and ModernBERT embeddings.
Safe to run multiple times - will only compute what's missing
"""

from robust_surrogate_predict import run_comparison
from embed_corpus import add_embeddings_to_corpus
from generate_corpus import generate_pytorch_corpus
import os

# Paths
NASBENCH_CORPUS_NEW_PATH = '/storage/ice-shared/vip-vvk/data/AOT/psomu3/codenas/nasbench101_corpus.pkl'
NASBENCH_CORPUS_OUTPUT_PATH = '/storage/ice-shared/vip-vvk/data/AOT/psomu3/codenas/nasbench101_corpus_embedded.pkl'
OUTPUT_PATH = '/storage/ice-shared/vip-vvk/data/AOT/psomu3/codenas/nb101/nasbench101_comparison.csv'
OUTPUT_PATH_CSV = '/storage/ice-shared/vip-vvk/data/AOT/psomu3/codenas/nb101'
COMPARISON_LABEL = 'nasbench101'

# Configuration
SAMPLE_SIZES = [14, 55, 220, 879]#, 3584, 14062]  # Training set sizes to test
N_FOLDS = 10       # 10-fold cross-validation
N_REPEATS = 10    # If Repeat CV 50 times, then 500 total trials per sample size
FORCE = False     # If True, recompute everything; if False, only add missing data

# Generate new corpus with all 3 primitives modes if it doesn't exist
if not os.path.exists(NASBENCH_CORPUS_NEW_PATH):
    print("\n" + "=" * 80)
    print("Generating new corpus with all primitives modes...")
    print("=" * 80)
    df_corpus = generate_pytorch_corpus(
        output_path=NASBENCH_CORPUS_NEW_PATH,
        context_mode=None
    )
else:
    print(f"\nCorpus already exists at {NASBENCH_CORPUS_NEW_PATH}")

# Add embeddings to the new corpus
print("\n" + "=" * 80)
print("Adding embeddings to corpus...")
print("=" * 80)
df = add_embeddings_to_corpus(
    corpus_path=NASBENCH_CORPUS_NEW_PATH,
    model_name='codellama_python_7b',
    output_path=NASBENCH_CORPUS_OUTPUT_PATH,
    pytorch_all=True,
    quantization='fp16',
    use_echo_embeddings=False,
    device='cuda',
    max_length=1024
)
# Add embeddings to the new corpus
print("\n" + "=" * 80)
print("Adding embeddings to corpus...")
print("=" * 80)
df = add_embeddings_to_corpus(
    corpus_path=NASBENCH_CORPUS_NEW_PATH,
    model_name='modernbert_large',
    output_path=NASBENCH_CORPUS_OUTPUT_PATH,
    pytorch_all=True,
    quantization='fp16',
    use_echo_embeddings=False,
    device='cuda',
    max_length=1024
)
# Run robust comparison (only computes missing trials)


results_df = run_comparison(
    embedding1_name=f'codellama_python_7b_pytorch_code_fp16_embedding',
    corpus1_name='nasbench101',
    embedding2_name=f'modernbert_large_pytorch_code_fp16_embedding',
    corpus2_name='nasbench101',
    corpus_path1=NASBENCH_CORPUS_OUTPUT_PATH,
    corpus_path2=NASBENCH_CORPUS_OUTPUT_PATH,
    comparison_label=COMPARISON_LABEL,
    sample_sizes=SAMPLE_SIZES,
    n_folds=N_FOLDS,
    n_repeats=N_REPEATS,
    benchmark_type='nasbench101',
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


