from generate_corpus import generate_pytorch_corpus

# Paths
NASBENCH_CORPUS_PATH = '/storage/ice-shared/vip-vvk/data/AOT/psomu3/codenas/nasbench201_corpus_averaged.pkl'
EMBEDDED_CORPUS_PATH = '/storage/ice-shared/vip-vvk/data/AOT/psomu3/codenas/nasbench201_corpus_embedded.pkl'

df_corpus = generate_pytorch_corpus(
        output_path=NASBENCH_CORPUS_PATH,
        datasets=['cifar10', 'cifar100', 'ImageNet16-120'],
        context_mode=None
)