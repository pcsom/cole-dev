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

def drawTSNE(tsne_1,tsne_2, FILE_NAME, OUTPUT_FOLDER, legend_name, plotting_number,name, plot_type, vrange = [0,1], pallete = 'viridis', alpha=0.3, xlim=None, ylim=None):
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
        
        # Filter out points with value 600 (not captured) or -1000 (missing/placeholder)
        mask_missing = (plotting_number == 600) | (plotting_number == -1000)
        
        # MINIMAL FIX: If using global dimensions, plot the background/common points in light gray
        if xlim is not None:
            plt.scatter(
                tsne_1[mask_missing],
                tsne_2[mask_missing],
                c='lightgray',
                alpha=0.2,
                label='Other Architectures'
            )

       # Copy so we don't modify original
        plotting_number_clipped = plotting_number.copy()

        # Clip only valid values
        plotting_number_clipped[~mask_missing] = plotting_number_clipped[~mask_missing].clip(
            lower=vrange[0],
            upper=vrange[1]
        )
        # Plot only valid points
        sc = plt.scatter(
            tsne_1[~mask_missing],
            tsne_2[~mask_missing],
            c=plotting_number_clipped[~mask_missing],
            cmap=pallete,
            alpha=alpha
        )

        plt.colorbar(sc, label=legend_name)
        #plt.legend( title=legend_name, bbox_to_anchor=(1, 1), loc='upper left')
        
    if(plot_type == "discrete"):
        import matplotlib.colors as mcolors
        n_clusters = plotting_number.nunique()
        cmap_discrete = mcolors.ListedColormap(paletteDiscrete[:n_clusters])
        
        # Use plt.scatter instead of sns.scatterplot, and add a colorbar explicitly.
        # Adding a colorbar ensures Matplotlib shrinks the main plot area by the EXACT
        # same proportion as the gradient plots, guaranteeing matching output dimensions.
        sc = plt.scatter(
            tsne_1, 
            tsne_2, 
            c=plotting_number, 
            cmap=cmap_discrete, 
            alpha=max(alpha, 0.6) # Ensure discrete points are sufficiently opaque
        )
        
        plt.colorbar(sc, label=name)
        
        # Add cluster number annotations at centroids
        # Create temp dataframe to compute centroids
        temp_df = pd.DataFrame({'x': tsne_1, 'y': tsne_2, 'c': plotting_number.values})
        centroids = temp_df.groupby('c').mean()
        
        for cluster_id, row in centroids.iterrows():
            plt.annotate(str(int(cluster_id)), 
                         (row['x'], row['y']),
                         horizontalalignment='center',
                         verticalalignment='center',
                         weight='bold',
                         fontsize=10,
                         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.7))
        
    plt.title(f'tsne_plot_{name}_{FILE_NAME}')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)

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
    
    parser.add_argument('--ranges', type=int, nargs='+', default=[65,74,50,100, 80, 100, 50, 100, 0, 700, 0, 700, 80, 100, 50, 100, 0, 700, 0, 700], help='Range of values to be plotted in the gradient plot')
    parser.add_argument('--cluster', type=bool, default=True, help='Whether to perform clustering on the t-SNE values and plot the clustered results')
    parser.add_argument('--traj_max_iter', type=int, default=None, help='If set, only plot trajectory individuals with generation_firstchosen <= this value')
    parser.add_argument('--override_bounds', type=bool, default=False)
    parser.add_argument('--global_dimensions', action='store_true', help='Use global t-SNE bounds for all plots to allow overlaying')
    args = parser.parse_args()
    print("starting t-SNE visualization with the following arguments: ")
    print(args)
    df = pd.read_csv(args.input)
    if args.debug:
        print(df.columns)
        print(df.head())
    tsne_1, tsne_2,accuracy = createTSNE(df, args.encoding)
    cluster_df = cluster(tsne_1, tsne_2, alpha=5.0)
    override_bounds = args.override_bounds

    # Calculate global limits if requested
    xlim, ylim = None, None
    if args.global_dimensions:
        margin_x = (tsne_1.max() - tsne_1.min()) * 0.05
        margin_y = (tsne_2.max() - tsne_2.min()) * 0.05
        xlim = (tsne_1.min() - margin_x, tsne_1.max() + margin_x)
        ylim = (tsne_2.min() - margin_y, tsne_2.max() + margin_y)
    
    for i, test_field in enumerate(args.test_fields):
        # Determine default range from arguments or fallback
        if i*2+1 < len(args.ranges):
            r_min = args.ranges[i*2]
            r_max = args.ranges[i*2+1]
        else:
            r_min = 0
            r_max = 100
        
        # Heuristic: If valid data lies entirely outside the [r_min, r_max] range,
        # override with the data's actual min/max.
        # This handles cases where user-specified fields don't match default ranges.
        if test_field in df.columns:
            field_data = df[test_field]
            # specific to this dataset: -1000 and 600 are special markers
            valid_mask = (field_data != -1000) & (field_data != 600)
            if valid_mask.any():
                d_min = field_data[valid_mask].min()
                d_max = field_data[valid_mask].max()
                # If range is completely disjoint from data
                if override_bounds and (d_max < r_min or d_min > r_max):
                    print(f"Auto-adjusting range for {test_field}: provided/default [{r_min}, {r_max}] mismatch with data [{d_min:.2f}, {d_max:.2f}]. Using data limits.")
                    r_min, r_max = d_min, d_max
        
        drawTSNE(tsne_1, tsne_2, f"{args.output_prefix}_{test_field}", args.output_dir,test_field, df[test_field],"curated","gradient", vrange=[r_min, r_max], pallete='Reds', alpha=0.8, xlim=xlim, ylim=ylim)
    drawTSNE(tsne_1, tsne_2, f"{args.output_prefix}_cifar10_discovery_curbed_grad", args.output_dir,"generation", df['generation_firstdiscovery_cifar10'],"generation","gradient", vrange=[0, 100], pallete='crest', xlim=xlim, ylim=ylim)
    drawTSNE(tsne_1, tsne_2, f"{args.output_prefix}_cifar100_discovery_curbed_grad", args.output_dir,"generation", df['generation_firstdiscovery_cifar100'],"generation","gradient", vrange=[0, 100], pallete='crest', xlim=xlim, ylim=ylim)

    # Trajectory-only plots: only the individuals actually chosen and evaluated during NAS
    traj_suffix = f"_first{args.traj_max_iter}" if args.traj_max_iter is not None else ""
    
    # MINIMAL FIX: Avoid array slicing, pass full arrays and let drawTSNE filter via -1000s
    if 'traj_generation_firstchosen_cifar10' in df.columns:
        traj_gen_10 = df['traj_generation_firstchosen_cifar10'].copy()
        if args.traj_max_iter is not None:
            traj_gen_10[traj_gen_10 > args.traj_max_iter] = -1000
            
        if (traj_gen_10 != -1000).any():
            drawTSNE(
                tsne_1, tsne_2,  # pass full arrays instead of slicing
                f"{args.output_prefix}_cifar10_trajectory_only{traj_suffix}", args.output_dir,
                "generation_chosen", traj_gen_10,
                "trajectory", "gradient", vrange=[0, int(traj_gen_10[traj_gen_10 != -1000].max())], pallete='Reds', alpha=0.9, xlim=xlim, ylim=ylim
            )
        else:
            print("Skipping cifar10 trajectory plot: no trajectory data found.")
            
    if 'traj_generation_firstchosen_cifar100' in df.columns:
        traj_gen_100 = df['traj_generation_firstchosen_cifar100'].copy()
        if args.traj_max_iter is not None:
            traj_gen_100[traj_gen_100 > args.traj_max_iter] = -1000
            
        if (traj_gen_100 != -1000).any():
            drawTSNE(
                tsne_1, tsne_2,  # pass full arrays instead of slicing
                f"{args.output_prefix}_cifar100_trajectory_only{traj_suffix}", args.output_dir,
                "generation_chosen", traj_gen_100,
                "trajectory", "gradient", vrange=[0, int(traj_gen_100[traj_gen_100 != -1000].max())], pallete='Reds', alpha=0.9, xlim=xlim, ylim=ylim
            )
        else:
            print("Skipping cifar100 trajectory plot: no trajectory data found.")

    if(args.cluster):
        drawTSNE(tsne_1, tsne_2, f"{args.output_prefix}_clustered", args.output_dir,"cluster", cluster_df['cluster'],"clustered","discrete", xlim=xlim, ylim=ylim)
    
    #save the reverse mapping of cluster to arch index and arch string as a json
    reverse_mapping = createReversemapping(df, cluster_df, args.code_field)
    with open(os.path.join(args.output_dir, f'{args.output_prefix}_reverse_mapping.txt'), 'w') as f:
        for cluster1, archs in reverse_mapping.items():
            f.write(f"Cluster {cluster1}:\n")
            for arch in archs:
                f.write(f"  {arch}\n")
    print(f"reverse mapping saved at {os.path.join(args.output_dir, f'{args.output_prefix}_reverse_mapping.txt')}")

if __name__ == "__main__":
    main()