"""
JAHS-Bench-201 corpus generation utilities.
Randomly samples architectures and saves them with performance metrics.
"""

import jahs_bench
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from jahs_stringify_utils import jahs_config_to_all_formats

# Cache the benchmark instance
_jahs_benchmark = None

def get_jahs_benchmark(task='cifar10', kind='surrogate'):
    """
    Get or create a singleton JAHS-Bench benchmark instance.
    
    Args:
        task: Task name (cifar10, colorectal_histology, fashion_mnist)
        kind: Query type ('surrogate', 'table', or 'live')
    
    Returns:
        JAHS-Bench benchmark instance
    """
    global _jahs_benchmark
    if _jahs_benchmark is None:
        print(f"Initializing JAHS-Bench-201 ({task}, {kind})...")
        _jahs_benchmark = jahs_bench.Benchmark(task=task, kind=kind, download=True)
        print("JAHS-Bench-201 initialized successfully!")
    return _jahs_benchmark


def sample_jahs_architectures(n_samples=7812, seed=42):
    """
    Sample random architectures from JAHS-Bench-201.
    
    Args:
        n_samples: Number of architectures to sample
        seed: Random seed for reproducibility
    
    Returns:
        List of config dictionaries
    """
    np.random.seed(seed)
    benchmark = get_jahs_benchmark()
    
    configs = []
    print(f"Sampling {n_samples} random JAHS-Bench-201 architectures...")
    
    for _ in tqdm(range(n_samples), desc="Sampling architectures"):
        config = benchmark.sample_config()
        configs.append(config)
    
    return configs


def get_jahs_arch_properties(config, benchmark=None, nepochs=200):
    """
    Get performance metrics for a JAHS-Bench architecture.
    
    Args:
        config: Architecture configuration dict
        benchmark: JAHS-Bench benchmark instance (optional)
        nepochs: Number of epochs to query
    
    Returns:
        Dictionary with performance metrics
    """
    if benchmark is None:
        benchmark = get_jahs_benchmark()
    
    # Query the benchmark at full epochs
    results = benchmark(config, nepochs=nepochs)
    
    # Extract metrics at final epoch
    final_results = results[nepochs]
    
    properties = {
        'config': config,
        'valid_acc': final_results.get('valid-acc', None),
        'test_acc': final_results.get('test-acc', None),
        'train_loss': final_results.get('train-loss', None),
        'valid_loss': final_results.get('valid-loss', None),
        'test_loss': final_results.get('test-loss', None),
        'runtime': final_results.get('runtime', None),
        'size_MB': final_results.get('size_MB', None),
        'n_params': final_results.get('n_params', None),
    }
    
    return properties


def generate_jahs_corpus(
    n_samples=7812,
    output_path='/storage/ice-shared/vip-vvk/data/AOT/psomu3/codenas/jahsbench201_corpus.pkl',
    task='cifar10',
    nepochs=200,
    seed=42
):
    """
    Generate a corpus of JAHS-Bench-201 architectures with performance metrics.
    
    Args:
        n_samples: Number of architectures to sample
        output_path: Path to save the corpus
        task: JAHS-Bench task name
        nepochs: Number of training epochs to query
        seed: Random seed
    
    Returns:
        DataFrame with architecture configs and metrics
    """
    print("="*80)
    print("JAHS-Bench-201 Corpus Generation")
    print("="*80)
    print(f"Task: {task}")
    print(f"Number of samples: {n_samples}")
    print(f"Epochs: {nepochs}")
    print(f"Output path: {output_path}\n")
    
    # Initialize benchmark
    benchmark = get_jahs_benchmark(task=task, kind='surrogate')
    
    # Sample architectures
    configs = sample_jahs_architectures(n_samples=n_samples, seed=seed)
    
    # Collect properties
    corpus_data = []
    print(f"\nQuerying performance metrics for {n_samples} architectures...")
    
    for i, config in enumerate(tqdm(configs, desc="Querying metrics")):
        properties = get_jahs_arch_properties(config, benchmark, nepochs)
        
        # Generate code representations
        code_formats = jahs_config_to_all_formats(config)
        
        # Flatten config into row
        row = {
            'arch_index': i,
            'config_str': str(config),
            # Code representations (only PyTorch for JAHS - includes hyperparameters)
            'pytorch_code': code_formats['pytorch_code'],
            # Architecture
            'Op1': config['Op1'],
            'Op2': config['Op2'],
            'Op3': config['Op3'],
            'Op4': config['Op4'],
            'Op5': config['Op5'],
            'Op6': config['Op6'],
            # Hyperparameters
            'Optimizer': config['Optimizer'],
            'LearningRate': config['LearningRate'],
            'WeightDecay': config['WeightDecay'],
            'Activation': config['Activation'],
            'TrivialAugment': config['TrivialAugment'],
            # Fidelity parameters
            'N': config['N'],
            'W': config['W'],
            'Resolution': config['Resolution'],
            # Performance metrics
            'valid_acc': properties['valid_acc'],
            'test_acc': properties['test_acc'],
            'train_loss': properties['train_loss'],
            'valid_loss': properties['valid_loss'],
            'test_loss': properties['test_loss'],
            'runtime': properties['runtime'],
            'size_MB': properties['size_MB'],
            'n_params': properties['n_params'],
        }
        corpus_data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(corpus_data)
    
    # Save corpus
    print(f"\nSaving corpus to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_pickle(output_path)
    
    # Also save as CSV for easy inspection
    csv_path = output_path.replace('.pkl', '.csv')
    df.to_csv(csv_path, index=False)
    print(f"Also saved CSV to {csv_path}")
    
    print(f"\nCorpus generation complete!")
    print(f"Total architectures: {len(df)}")
    print(f"\nSample statistics:")
    print(f"Valid Accuracy: {df['valid_acc'].mean():.4f} ± {df['valid_acc'].std():.4f}")
    print(f"Test Accuracy: {df['test_acc'].mean():.4f} ± {df['test_acc'].std():.4f}")
    print(f"Parameters: {df['n_params'].mean():.0f} ± {df['n_params'].std():.0f}")
    
    return df


def load_jahs_corpus(path='/storage/ice-shared/vip-vvk/data/AOT/psomu3/codenas/jahsbench201_corpus.pkl'):
    """
    Load a saved JAHS-Bench corpus.
    
    Args:
        path: Path to the corpus pickle file
    
    Returns:
        DataFrame with corpus data
    """
    return pd.read_pickle(path)


if __name__ == '__main__':
    # Generate corpus with 7812 samples (same as NAS-Bench-201 half corpus)
    df = generate_jahs_corpus(n_samples=7812)
    print("\nFirst few rows:")
    print(df.head())
