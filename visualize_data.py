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
def drawTSNE(tsne_1,tsne_2, FILE_NAME, OUTPUT_FOLDER, legend_name, title_name, plotting_number, plot_type = "gradient", vrange = [0,1], pallete = 'viridis', reject = np.nan):
    if title_name is None:
        title_name = legend_name
    #sort the tsne values by the plotting number so that the points with higher plotting number are plotted on top
    sorted_indices = np.argsort(plotting_number)
    #if legend_name is discovery_errorplot descending order, otherwise ascending order
    if legend_name == "discovery_error":
        sorted_indices = sorted_indices[::-1]
    tsne_1 = tsne_1[sorted_indices]
    tsne_2 = tsne_2[sorted_indices]
    plotting_number = plotting_number[sorted_indices]
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
        
        mask_reject = (plotting_number == reject)
       # Copy so we don't modify original
        plotting_number_clipped = plotting_number.copy()

        # Clip only non-600 values
        plotting_number_clipped[~mask_reject] = plotting_number_clipped[~mask_reject].clip(
            lower=vrange[0],
            upper=vrange[1]
        )# Plot all other points
        sc = plt.scatter(
            tsne_1[~mask_reject],
            tsne_2[~mask_reject],
            c=plotting_number_clipped[~mask_reject],
            cmap=pallete,
            alpha=0.3
        )

        plt.colorbar(sc, label=legend_name)
        #plt.legend( title=legend_name, bbox_to_anchor=(1, 1), loc='upper left')
        
    if(plot_type == "discrete"):
        sns.stripplot(x=tsne_1, y=tsne_2, hue=plotting_number, palette=paletteDiscrete[:plotting_number.nunique()], jitter=True)
        #create an underbar matching colors to clusters
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.legend(handles, labels, title=legend_name, bbox_to_anchor=(1, 1), loc='upper left')
    plt.title(f'tsne_plot_{title_name}_{FILE_NAME}')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    
    plt.savefig(os.path.join(OUTPUT_FOLDER, f'tsne_plot_{title_name}_{FILE_NAME}.png'))
    print(f"image saved at {os.path.join(OUTPUT_FOLDER, f'tsne_plot_{title_name}_{FILE_NAME}.png')}")
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
def createAnimation(tsne_1, tsne_2, FILE_NAME, OUTPUT_FOLDER,
                    legend_name, title_name, plotting_number,
                    plot_type="gradient", vrange=[0,1],
                    pallete='viridis', reject=np.nan):

    import matplotlib.animation as animation
    import numpy as np
    import os

    fig, ax = plt.subplots(figsize=(10,8))
    mask = (plotting_number <= vrange[1]) & (plotting_number != reject)
    sc = ax.scatter(tsne_1[mask], tsne_2[mask], c=plotting_number[mask], cmap=pallete, vmin=vrange[0], vmax=vrange[1], alpha=0.3)
    fig.colorbar(sc, ax=ax, label=legend_name)

    def update(frame):

        ax.clear()

        mask = (plotting_number <= frame) & (plotting_number != reject)

        sc = ax.scatter(
            tsne_1[mask],
            tsne_2[mask],
            c=plotting_number[mask],
            cmap=pallete,
            vmin=vrange[0],
            vmax=vrange[1],
            alpha=0.3
        )

        ax.set_title(f"{title_name} (generation ≤ {frame})")

    frames = np.arange(vrange[0], vrange[1] + 1, 10)

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=frames,
        repeat=False
    )

    output_path = os.path.join(OUTPUT_FOLDER, f"{FILE_NAME}.gif")

    ani.save(output_path, writer="pillow", fps=5)

    print(f"animation saved at {output_path}")
def main():
    parser = argparse.ArgumentParser(description='t-SNE Visualization of Genome Embeddings')
    parser.add_argument('--input', type=str, default=f'/storage/ice-shared/vip-vvk/data/AOT/mgullapalli6/codenas/collecteddata_unified80.csv', help='Path to the input CSV file containing the data')
    parser.add_argument('--output_dir', type=str, default='/home/hice1/mgullapalli6/scratch/tsne/10001', help='Directory to save the output visualization')
    parser.add_argument('--encoding', type=str,  default='modernbert_large_true_onnx_encoding_fp16_embedding', help='Column name for the encoding to be visualized')
    
    parser.add_argument('--code_field', type=str, default='pytorch_code_inline', help='Column name for the code field to be used in reverse mapping')
    parser.add_argument('--debug', type=bool, default=True, help='Whether to print debug information, such as the columns of the input dataframe')
    
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
    

    drawTSNE(tsne_1, tsne_2, f"tsne_average_generation_cifar10", args.output_dir,"generation", "average_generation_cifar10", df["average_generation_cifar10"],"gradient", vrange=[0,800], reject=-1000)
    drawTSNE(tsne_1, tsne_2, f"tsne_average_generation_cifar100", args.output_dir,"generation", "average_generation_cifar100", df["average_generation_cifar100"],"gradient", vrange=[0, 800], reject=-1000)
    drawTSNE(tsne_1, tsne_2, f"tsne_average_predicted_accuracy_cifar10", args.output_dir,"predicted accuracy", "average_predicted_accuracy_cifar10", df["average_predicted_accuracy_cifar10"],"gradient", vrange=[80,100], reject=-1000)
    drawTSNE(tsne_1, tsne_2, f"tsne_average_predicted_accuracy_cifar100", args.output_dir,"predicted accuracy", "average_predicted_accuracy_cifar100", df["average_predicted_accuracy_cifar100"],"gradient", vrange=[50,100], reject=-1000)
    drawTSNE(tsne_1, tsne_2, f"tsne_average_real_accuracy_cifar10", args.output_dir,"accuracy", "average_real_accuracy_cifar10", df["cifar10-valid_test_accuracy"],"gradient", vrange=[0,20])
    drawTSNE(tsne_1, tsne_2, f"tsne_average_real_accuracy_cifar100", args.output_dir,"accuracy", "average_real_accuracy_cifar100", df["cifar100_valid_accuracy"],"gradient", vrange=[0,20])
    #draw errors in discovery with a continous plot (between correct and predicted accuracy) and cluster with a discrete plot, also if predicted acciracy = -1000, error should be -1000
    df['discovery_error_cifar10'] = df.apply(lambda row: abs(row['cifar10-valid_test_accuracy'] - row['predicted_accuracy_cifar10']) if row['predicted_accuracy_cifar10'] != -1000 else -1000, axis=1)
    df['discovery_error_cifar100'] = df.apply(lambda row: abs(row['cifar100_valid_accuracy'] - row['predicted_accuracy_cifar100']) if row['predicted_accuracy_cifar100'] != -1000 else -1000, axis=1)
    drawTSNE(tsne_1, tsne_2, f"tsne_cifar10_discovery_error", args.output_dir,"discovery_error", "discovery_error_cifar10", df['discovery_error_cifar10'],"gradient", vrange=[0, 20], pallete = 'Reds', reject=-1000)
    drawTSNE(tsne_1, tsne_2, f"tsne_cifar100_discovery_error", args.output_dir,"discovery_error", "discovery_error_cifar100", df['discovery_error_cifar100'],"gradient", vrange=[0, 20], pallete = 'Reds', reject=-1000)
    #draw number of times showed up with a continous plot and cluster with a discrete plot, also if showed up = ０, then it should be colored differently (with a discrete color)
    drawTSNE(tsne_1, tsne_2, f"tsne_cifar10_showed_up", args.output_dir,"showed_up", "showed_up_cifar10", df['showed_up_cifar10'],"gradient", vrange=[0, 20], pallete='crest',reject=-1000)
    drawTSNE(tsne_1, tsne_2, f"tsne_cifar100_showed_up", args.output_dir,"showed_up", "showed_up_cifar100", df['showed_up_cifar100'],"gradient", vrange=[0, 20], pallete='crest', reject=-1000)
    #draw real accuracy if preedicted accuracy is not -１０００ else set real accuracy to -１０００
    df['real_accuracy_cifar10'] = df.apply(lambda row: row['cifar10-valid_test_accuracy'] if row['predicted_accuracy_cifar10'] != -1000 else -1000, axis=1)
    df['real_accuracy_cifar100'] = df.apply(lambda row: row['cifar100_valid_accuracy'] if row['predicted_accuracy_cifar100'] != -1000 else -1000, axis=1)
    drawTSNE(tsne_1, tsne_2, f"tsne_cifar10_real_accuracy", args.output_dir,"real_accuracy", "real_accuracy_cifar10", df['real_accuracy_cifar10'],"gradient", vrange=[80, 100], pallete='viridis', reject=-1000)
    drawTSNE(tsne_1, tsne_2, f"tsne_cifar100_real_accuracy", args.output_dir,"real_accuracy", "real_accuracy_cifar100", df['real_accuracy_cifar100'],"gradient", vrange=[50, 100], pallete='viridis', reject=-1000)
    if(args.cluster):
        drawTSNE(tsne_1, tsne_2, f"tsne_clustered", args.output_dir,"cluster", "clustered", cluster_df['cluster'],"discrete")
    #create an animation based on generations for cifar10
    createAnimation(tsne_1, tsne_2, f"tsne_animation_average_generation_cifar10", args.output_dir,"generation", "average_generation_cifar10", df["average_generation_cifar10"],"gradient", vrange=[0, 800], reject=-1000)
    #save the reverse mapping of cluster to arch index and arch string as a json
    reverse_mapping = createReversemapping(df, cluster_df, args.code_field)
    with open(os.path.join(args.output_dir, f'reverse_mapping.txt'), 'w') as f:
        for cluster1, archs in reverse_mapping.items():
            f.write(f"Cluster {cluster1}:\n")
            for arch in archs:
                f.write(f"  {arch}\n")
    print(f"reverse mapping saved at {os.path.join(args.output_dir, f'reverse_mapping.txt')}")
main()