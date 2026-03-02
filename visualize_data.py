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
def drawTSNE(tsne_1,tsne_2, FILE_NAME, OUTPUT_FOLDER, legend_name, plotting_number,name, plot_type, vrange = [0,1], pallete = 'viridis'):
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
        
        mask_600 = (plotting_number == 600)
       # Copy so we don't modify original
        plotting_number_clipped = plotting_number.copy()

        # Clip only non-600 values
        plotting_number_clipped[~mask_600] = plotting_number_clipped[~mask_600].clip(
            lower=vrange[0],
            upper=vrange[1]
        )# Plot all other points
        sc = plt.scatter(
            tsne_1[~mask_600],
            tsne_2[~mask_600],
            c=plotting_number_clipped[~mask_600],
            cmap=pallete,
            alpha=0.3
        )

        plt.scatter(
            tsne_1[mask_600],
            tsne_2[mask_600],
            color="darkred",
            s=60,
            edgecolor="black"
        )

        plt.colorbar(sc, label=legend_name)
        #plt.legend( title=legend_name, bbox_to_anchor=(1, 1), loc='upper left')
    if(plot_type == "discrete"):
        sns.stripplot(x=tsne_1, y=tsne_2, hue=plotting_number, palette=paletteDiscrete[:plotting_number.nunique()], jitter=True)
        #create an underbar matching colors to clusters
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.legend(handles, labels, title=name, bbox_to_anchor=(1, 1), loc='upper left')
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
    parser.add_argument('--input', type=str, default=f'/storage/ice-shared/vip-vvk/data/AOT/mgullapalli6/codenas/collecteddata_unified.csv', help='Path to the input CSV file containing the data')
    parser.add_argument('--output_dir', type=str, default='/home/hice1/mgullapalli6/scratch/tsne', help='Directory to save the output visualization')
    parser.add_argument('--output_prefix', type=str, default='onnx_main', help='Prefix for the output visualization file name')
    parser.add_argument('--encoding', type=str,  default='modernbert_large_true_onnx_encoding_fp16_embedding', help='Column name for the encoding to be visualized')
    
    parser.add_argument('--test_fields', type=str, nargs='+', default=['cifar10-valid_test_accuracy', 'cifar100_valid_accuracy','predicted_accuracy_cifar10','predicted_accuracy_cifar100', 'generation_firstdiscovery_cifar10', 'generation_firstdiscovery_cifar100','average_predicted_accuracy_cifar10','average_predicted_accuracy_cifar100','average_generation_cifar10','average_generation_cifar100'], help='Column name for the field to be plotted as hue in the gradient plot')
    parser.add_argument('--code_field', type=str, default='pytorch_code_inline', help='Column name for the code field to be used in reverse mapping')
    parser.add_argument('--debug', type=bool, default=True, help='Whether to print debug information, such as the columns of the input dataframe')
    
    parser.add_argument('--ranges', type=int, nargs='+', default=[80,100,50,100, 80, 100, 50, 100, 0, 700, 0, 700, 80, 100, 50, 100, 0, 700, 0, 700], help='Range of values to be plotted in the gradient plot')
    parser.add_argument('--cluster', type=bool, default=True, help='Whether to perform clustering on the t-SNE values and plot the clustered results')
    args = parser.parse_args()
    print("starting t-SNE visualization with the following arguments: ")
    print(args)
    df = pd.read_csv(args.input)
    if args.debug:
        print(df.columns)
        print(df.head())
    tsne_1, tsne_2,accuracy = createTSNE(df, args.encoding)
    cluster_df = cluster(tsne_1, tsne_2, alpha=5.0)
    
    for i, test_field in enumerate(args.test_fields):
        drawTSNE(tsne_1, tsne_2, f"{args.output_prefix}_{test_field}", args.output_dir,test_field, df[test_field],"curated","gradient", vrange=[args.ranges[i*2], args.ranges[i*2+1]])
    drawTSNE(tsne_1, tsne_2, f"{args.output_prefix}_cifar10_discovery_curbed_grad", args.output_dir,"generation", df['generation_firstdiscovery_cifar10'],"generation","gradient", vrange=[0, 100], pallete='crest')
    drawTSNE(tsne_1, tsne_2, f"{args.output_prefix}_cifar100_discovery_curbed_grad", args.output_dir,"generation", df['generation_firstdiscovery_cifar100'],"generation","gradient", vrange=[0, 100], pallete='crest')
    if(args.cluster):
        drawTSNE(tsne_1, tsne_2, f"{args.output_prefix}_clustered", args.output_dir,"cluster", cluster_df['cluster'],"clustered","discrete")
    #save the reverse mapping of cluster to arch index and arch string as a json
    reverse_mapping = createReversemapping(df, cluster_df, args.code_field)
    with open(os.path.join(args.output_dir, f'{args.output_prefix}_reverse_mapping.txt'), 'w') as f:
        for cluster1, archs in reverse_mapping.items():
            f.write(f"Cluster {cluster1}:\n")
            for arch in archs:
                f.write(f"  {arch}\n")
    print(f"reverse mapping saved at {os.path.join(args.output_dir, f'{args.output_prefix}_reverse_mapping.txt')}")
main()