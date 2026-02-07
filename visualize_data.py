import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import argparse
import os


FILE_LOC = "/storage/ice-shared/vip-vvk/data/AOT/psomu3/codenas2/nasbench201_corpus_onnx_paper_embedded.csv"
OUTPUT_FILE = "/storage/ice-shared/vip-vvk/data/AOT/mgullapalli6/codenas2/tsne_output"
def createTSNEoutput(FILE_LOC , OUTPUT_FILE):
    df = pd.read_csv(FILE_LOC)

    #extracting the arch index, arch string, cifar10-valid_test_accuracy, pytorch_code_inline, and 'true_onnx_encoding'
    df = df[['arch_index', 'arch_string', 'cifar10-valid_test_accuracy', 'pytorch_code_inline', 'modernbert_large_true_onnx_encoding_fp16_embedding']]
    # clean up modernbert encoding by removing brackets and splitting by comma
    df['modernbert_large_true_onnx_encoding_fp16_embedding'] = df['modernbert_large_true_onnx_encoding_fp16_embedding'].apply(lambda x: x.strip('[]').replace(' ', ''))
    #run tsne on the 'true_onnx_encoding' column
    X = df['modernbert_large_true_onnx_encoding_fp16_embedding'].apply(lambda x: [float(i) for i in x.split(',')])
    X = np.array(X.tolist())
    X_scaled = StandardScaler().fit_transform(X)
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X_scaled)

    #add the tsne results to the dataframe
    df['tsne_1'] = X_tsne[:, 0]
    df['tsne_2'] = X_tsne[:, 1]
    #plot the tsne results
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='tsne_1', y='tsne_2', hue='cifar10-valid_test_accuracy', palette='viridis', data=df)
    plt.title('t-SNE Visualization of NASBench201 Architectures')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend(title='CIFAR-10 Valid Test Accuracy')
    plt.savefig(os.path.join(OUTPUT_FILE, 'tsne_plot.png'))
    print(f"image saved at {os.path.join(OUTPUT_FILE, 'tsne_plot.png')}")
    plt.show()