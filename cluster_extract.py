import argparse
import os

def extract_clusters(input_file, clusters_to_extract):
    clusters = {}
    current_cluster = None
    
    with open(input_file, 'r') as f:
        for line in f:
            if line.startswith('Cluster '):
                current_cluster = int(line.strip().split(' ')[1].replace(':', ''))
                clusters[current_cluster] = []
            elif current_cluster is not None and line.strip():
                clusters[current_cluster].append(line.rstrip())
                
    os.makedirs('clusters', exist_ok=True)
    
    for cluster_id in clusters_to_extract:
        if cluster_id in clusters:
            output_file = os.path.join('clusters', f'cluster_{cluster_id}.txt')
            with open(output_file, 'w') as f:
                for item in clusters[cluster_id]:
                    f.write(f"{item}\n")
            print(f"Extracted Cluster {cluster_id} to {output_file}")
        else:
            print(f"Warning: Cluster {cluster_id} not found in input file.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract specific clusters from a cluster mapping file.")
    parser.add_argument("-i", "--input", required=True, help="Input file containing clusters (e.g., test_reverse_mapping.txt)")
    parser.add_argument("-c", "--clusters", type=int, nargs='+', required=True, help="List of cluster IDs to extract")
    
    args = parser.parse_args()
    
    extract_clusters(args.input, args.clusters)
    print("Extraction complete.")
