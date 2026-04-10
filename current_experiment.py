from embed_corpus import add_embeddings_to_corpus
from surrogate_predict import run_multiple_seeds_experiment

# This will add codebert embeddings ONLY if they don't already exist
# Existing model embeddings (deepseek, modernbert) are NOT recomputed
# df = add_embeddings_to_corpus(
#     corpus_path='/storage/ice-shared/vip-vvk/data/AOT/psomu3/codenas/nasbench201_corpus_embedded_half.pkl',
#     model_name='codebert',
#     device='cuda'
# )
# df = add_embeddings_to_corpus(
#     corpus_path='/storage/ice-shared/vip-vvk/data/AOT/psomu3/codenas/nasbench201_corpus_embedded_half.pkl',
#     model_name='codellama',
#     device='cuda'
# )
# df = add_embeddings_to_corpus(
#     corpus_path='/storage/ice-shared/vip-vvk/data/AOT/psomu3/codenas/nasbench201_corpus_embedded_half.pkl',
#     model_name='codellama_python',
#     device='cuda'
# )
# df = add_embeddings_to_corpus(
#     corpus_path='/storage/ice-shared/vip-vvk/data/AOT/psomu3/codenas/nasbench201_corpus_embedded_half.pkl',
#     model_name='coderankembed',
#     device='cuda'
# )

all_results, stats = run_multiple_seeds_experiment(
    n_seeds=20,
    output_path='/storage/ice-shared/vip-vvk/data/AOT/psomu3/codenas/results_multi_model.csv',
    embedding_filter=['codebert'],
)


print("Results for all models:")
print(all_results.groupby('input_type')['val_r2_acc_mean'].mean())


