import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import argparse
import os


def createTSNE(df, encoding):
    #extracting the arch index, arch string, cifar10-valid_test_accuracy, pytorch_code_inline, and 'true_onnx_encoding'
    
    # clean up modernbert encoding by removing brackets and splitting by comma
    vector = df[encoding].apply(lambda x: x.strip('[]').replace(' ', ''))
    #run tsne on the 'true_onnx_encoding' column
    X = vector.apply(lambda x: [float(i) for i in x.split(',')])
    X = np.array(X.tolist())
    X_scaled = StandardScaler().fit_transform(X)
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X_scaled)

    #return results
    return X_tsne[:,0], X_tsne[:,1], df['cifar10-valid_test_accuracy']
def drawTSNE(tsne_1,tsne_2, FILE_NAME, OUTPUT_FOLDER, plotting_number,name, plot_type, range = [0,1]):
    OUTPUT_FILE = os.path.join(OUTPUT_FOLDER, FILE_NAME)
    plt.figure(figsize=(10, 8))
    #if plotting type is discreet : use a discret gradient or use continous gradient (deided by plotting number)
    paletteDiscrete = (
        list(plt.cm.tab20.colors) +
        list(plt.cm.tab20b.colors) +
        list(plt.cm.tab20c.colors)
    )
    #delete data points that are outside the range
    if(plot_type == "gradient"):
        mask = (plotting_number >= range[0]) & (plotting_number <= range[1])
        tsne_1 = tsne_1[mask]
        tsne_2 = tsne_2[mask]
        plotting_number = plotting_number[mask]

        sns.scatterplot(x=tsne_1, y=tsne_2, hue=plotting_number, palette='viridis')
        plt.legend(title='CIFAR-10 Valid Test Accuracy')
    if(plot_type == "discrete"):
        sns.stripplot(x=tsne_1, y=tsne_2, hue=plotting_number, palette=paletteDiscrete[:plotting_number.nunique()], jitter=True)
        #create an underbar matching colors to clusters
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.legend(handles, labels, title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title(f'tsne_plot_{name}_{FILE_NAME}')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    
    plt.savefig(os.path.join(OUTPUT_FOLDER, f'tsne_plot_{name}_{FILE_NAME}.png'))
    print(f"image saved at {os.path.join(OUTPUT_FOLDER, f'tsne_plot_{name}_{FILE_NAME}.png')}")
    plt.show()

def cluster(tsne_x, tsne_y, alpha):
    #run a HDBSCAN clustering on the tsne values and return a df containing the tsne values and theri cluster 
    from sklearn.cluster import DBSCAN
    df = pd.DataFrame({'tsne_x': tsne_x, 'tsne_y': tsne_y})
    clustering = DBSCAN(eps=alpha, min_samples=5).fit(df)
    df['cluster'] = clustering.labels_
    return df
    
def createReversemapping(df, cluster_df, code):
    #create a new df that unifies both dataframes
    merged_df = pd.concat([df.reset_index(drop=True), cluster_df.reset_index(drop=True)], axis=1)
    #create a dictionarty that maps clusetrs to arch strings

    reverse_mapping = {}
    for cluster in merged_df['cluster'].unique():
        cluster_archs = merged_df[merged_df['cluster'] == cluster][code].tolist()
        reverse_mapping[int(cluster)] = cluster_archs
    return reverse_mapping    
def main():
    parser = argparse.ArgumentParser(description='t-SNE Visualization of Genome Embeddings')
    parser.add_argument('--input', type=str, default=f'/storage/ice-shared/vip-vvk/data/AOT/psomu3/codenas2/nasbench201_corpus_onnx_paper_embedded.csv', help='Path to the input CSV file containing the data')
    parser.add_argument('--output_dir', type=str, default=f'/storage/ice-shared/vip-vvk/data/AOT/mgullapalli6/codenas2/tsne_output', help='Directory to save the output visualization')
    parser.add_argument('--output_prefix', type=str, default='onnx_main', help='Prefix for the output visualization file name')
    #make it input a list of encodings to be visualized and default to the modernbert_large_true_onnx_encoding_fp16_embedding encoding
    parser.add_argument('--encodings', type=str, nargs='+', default=['modernbert_large_true_onnx_encoding_fp16_embedding'], help='Column name for the encoding to be visualized')
    
    parser.add_argument('--test_field', type=str, default='cifar10-valid_test_accuracy', help='Column name for the field to be plotted as hue in the gradient plot')
    parser.add_argument('--code_field', type=str, default='pytorch_code_inline', help='Column name for the code field to be used in reverse mapping')
    parser.add_argument('--debug', type=bool, default=True, help='Whether to print debug information, such as the columns of the input dataframe')
    parser.add_argument('--range', type=int, nargs=2, default=[80,100], help='Range of values to be plotted in the gradient plot')
    parser.add_argument('--cluster', type=bool, default=True, help='Whether to perform clustering on the t-SNE values and plot the clustered results')
    args = parser.parse_args()
    print("starting t-SNE visualization with the following arguments:")
    df = pd.read_csv(args.input)
    if args.debug:
        print(df.columns)
    
    for encoding in args.encodings:
        tsne_1, tsne_2,accuracy = createTSNE(df, encoding)
        cluster_df = cluster(tsne_1, tsne_2, alpha=5.0)
        print(f"done with clustering for {encoding}, now drawing t-SNE plots")
        drawTSNE(tsne_1, tsne_2, f"{args.output_prefix}_{encoding}", args.output_dir, df[args.test_field],"curated","gradient", range=args.range)
        if(args.cluster):
            drawTSNE(tsne_1, tsne_2, f"{args.output_prefix}_{encoding}_clustered", args.output_dir, cluster_df['cluster'],"clustered","discrete")
    #save the reverse mapping of cluster to arch index and arch string as a json
    reverse_mapping = createReversemapping(df, cluster_df, args.code_field)
    with open(os.path.join(args.output_dir, f'{args.output_prefix}_reverse_mapping.txt'), 'w') as f:
        for cluster1, archs in reverse_mapping.items():
            f.write(f"Cluster {cluster1}:\n")
            for arch in archs:
                f.write(f"  {arch}\n")
    print(f"reverse mapping saved at {os.path.join(args.output_dir, f'{args.output_prefix}_reverse_mapping.txt')}")
main()