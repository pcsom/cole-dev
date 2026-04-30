import os
import re
import numpy as np
from collections import defaultdict

HIGH_ERROR_CLUSTERS = [4, 5, 11, 18, 2]
LOW_ERROR_CLUSTERS = [24, 25, 29, 26, 30]
CLUSTERS_DIR = 'clusters'

FEATURES_REGEX = {
    'conv': r'Conv[123]d',
    'pool': r'(MaxPool|AvgPool)[123]d',
    'norm': r'(BatchNorm|LayerNorm|GroupNorm)',
    'linear': r'Linear',
    'activation': r'(ReLU|GELU|SiLU|Tanh|Sigmoid)',
    'concat': r'torch\.cat',
    'add': r'\+',
}

def extract_features(arch_text):
    """Extract quantitative features from a single architecture string."""
    features = {
        'num_chars': len(arch_text),
        'num_lines': len(arch_text.strip().split('\n')),
    }
    
    for feature_name, pattern in FEATURES_REGEX.items():
        features[feature_name] = len(re.findall(pattern, arch_text))
        
    return features

def analyze_cluster_file(cluster_id):
    """Parse a cluster file and return features for all its architectures."""
    filepath = os.path.join(CLUSTERS_DIR, f'cluster_{cluster_id}.txt')
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found.")
        return []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
        
    # Attempt to split into discrete architectures. 
    # If architectures start with 'class ', we split by it.
    # Otherwise, fallback to line-by-line if they are single lines.
    if 'class ' in content:
        archs = ['class ' + a for a in content.split('class ') if a.strip()]
    else:
        archs = [line for line in content.split('\n') if line.strip()]
        
    cluster_features = []
    for arch in archs:
        cluster_features.append(extract_features(arch))
        
    return cluster_features

def aggregate_stats(cluster_features_list):
    """Compute mean and variance for a list of architecture features."""
    if not cluster_features_list:
        return {}
    
    aggregated = {}
    keys = cluster_features_list[0].keys()
    
    for key in keys:
        values = [f[key] for f in cluster_features_list]
        aggregated[key] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'sum': np.sum(values)
        }
    return aggregated

def print_group_comparison(high_error_stats, low_error_stats):
    col1_w = 15
    col2_w = 32
    col3_w = 34
    total_w = col1_w + col2_w + col3_w + 6
    
    print("=" * total_w)
    
    header_high = f"High Error ({str(HIGH_ERROR_CLUSTERS)[1:-1]})"
    header_low = f"Low Error ({str(LOW_ERROR_CLUSTERS)[1:-1]})"
    
    print(f"{'Feature':<{col1_w}} | {header_high:<{col2_w}} | {header_low:<{col3_w}}")
    print("-" * total_w)
    
    if not high_error_stats or not low_error_stats:
        print("Data missing for comparison.")
        return

    for key in high_error_stats.keys():
        high_val = high_error_stats[key]['mean']
        high_std = high_error_stats[key]['std']
        low_val = low_error_stats[key]['mean']
        low_std = low_error_stats[key]['std']
        
        high_str = f"{high_val:.2f} ± {high_std:.2f}"
        low_str = f"{low_val:.2f} ± {low_std:.2f}"
        
        print(f"{key:<{col1_w}} | {high_str:<{col2_w}} | {low_str:<{col3_w}}")
    print("=" * total_w)

def main():
    high_error_data = []
    low_error_data = []
    
    print("Analyzing Internal Cluster Variances...")
    for c_id in HIGH_ERROR_CLUSTERS + LOW_ERROR_CLUSTERS:
        arch_features = analyze_cluster_file(c_id)
        if not arch_features:
            continue
            
        stats = aggregate_stats(arch_features)
        
        # Determine group
        if c_id in HIGH_ERROR_CLUSTERS:
            high_error_data.extend(arch_features)
            group = "HIGH ERROR"
        else:
            low_error_data.extend(arch_features)
            group = "LOW ERROR"
            
        print(f"\nCluster {c_id} ({group}) - {len(arch_features)} architectures:")
        print(f"  Avg Length (chars): {stats['num_chars']['mean']:.1f} ± {stats['num_chars']['std']:.1f}")
        print(f"  Avg Convs: {stats['conv']['mean']:.1f} ± {stats['conv']['std']:.1f}")
        print(f"  Avg Skips/Add: {stats['add']['mean']:.1f} ± {stats['add']['std']:.1f}")

    print("\n\n" + "#"*60)
    print("MACRO GROUP COMPARISON")
    print("#"*60)
    
    high_stats = aggregate_stats(high_error_data)
    low_stats = aggregate_stats(low_error_data)
    
    print_group_comparison(high_stats, low_stats)

if __name__ == '__main__':
    main()
