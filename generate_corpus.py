import torch
import pickle
import json
import pandas as pd
from tqdm import tqdm
from stringify_utils import arch_to_all_formats, arch_to_pytorch_code, get_api, get_architecture_string

def get_arch_properties(arch_index, dataset='cifar10-valid'):
    """
    Get regressable properties (fitness metrics) for an architecture.
    
    Args:
        arch_index: Architecture index in NASBench-201
        dataset: Dataset to query ('cifar10-valid', 'cifar100', 'ImageNet16-120')
    
    Returns:
        Dictionary of properties
    """
    # Get architecture metrics
    api = get_api()
    info = api.get_more_info(arch_index, dataset=dataset, hp='200', is_random=False)
    
    # Safely extract properties with defaults for missing keys
    properties = {
        'test_accuracy': info.get('test-accuracy', None),
        'valid_accuracy': info.get('valid-accuracy', None),
        'train_accuracy': info.get('train-accuracy', None),
        'test_loss': info.get('test-loss', None),
        'valid_loss': info.get('valid-loss', None),
        'train_loss': info.get('train-loss', None),
        'train_time': info.get('train-all-time', None),  # Training time in seconds
        'flops': info.get('flops', None),  # FLOPs
    }
    
    return properties

def generate_corpus(output_path='/storage/ice-shared/vip-vvk/data/AOT/psomu3/codenas/nasbench201_corpus.csv',
                    datasets=['cifar10-valid', 'cifar100', 'ImageNet16-120']):
    """
    Generate corpus of architecture code strings and their properties.
    
    Args:
        output_path: Path to save the corpus (CSV file)
        datasets: List of datasets to include metrics for
    
    Returns:
        DataFrame with all architectures and their representations
    """
    api = get_api()
    print("Generating NASBench-201 corpus...")
    print(f"Total architectures: {len(api)}")
    
    data_rows = []
    
    for arch_idx in tqdm(range(len(api)), desc="Processing architectures"):
        # Get all format representations
        formats = arch_to_all_formats(arch_idx)
        
        # Get architecture string
        arch_string = get_architecture_string(arch_idx)
        
        # Base entry with architecture info and all 3 formats
        entry = {
            'arch_index': arch_idx,
            'arch_string': arch_string,
            'pytorch_code': formats['pytorch'],
            'onnx_code': formats['onnx'],
            'grammar_code': formats['grammar']
        }
        
        # Get properties for each dataset and add to entry
        for dataset in datasets:
            props = get_arch_properties(arch_idx, dataset=dataset)
            # Prefix keys with dataset name
            for key, value in props.items():
                entry[f'{dataset}_{key}'] = value
        
        data_rows.append(entry)
    
    # Create DataFrame
    print(f"\nCreating DataFrame...")
    df = pd.DataFrame(data_rows)
    
    # Save to CSV
    print(f"Saving corpus to {output_path}")
    df.to_csv(output_path, index=False)
    
    # Also save as pickle for faster loading
    pkl_path = output_path.replace('.csv', '.pkl')
    print(f"Saving pickle version to {pkl_path}")
    df.to_pickle(pkl_path)
    
    # Save first 10 rows as JSON for inspection
    json_path = output_path.replace('.csv', '_sample.json')
    print(f"Saving sample JSON to {json_path}")
    df.head(10).to_json(json_path, orient='records', indent=2)
    
    print(f"\nCorpus generation complete!")
    print(f"Total entries: {len(df)}")
    print(f"DataFrame shape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")
    
    return df

def load_corpus(corpus_path='/storage/ice-shared/vip-vvk/data/AOT/psomu3/codenas/nasbench201_corpus.pkl'):
    """
    Load the generated corpus.
    
    Args:
        corpus_path: Path to corpus file (.pkl or .csv)
    
    Returns:
        DataFrame with corpus entries
    """
    if corpus_path.endswith('.pkl'):
        return pd.read_pickle(corpus_path)
    elif corpus_path.endswith('.csv'):
        return pd.read_csv(corpus_path)
    else:
        raise ValueError("Corpus path must be .pkl or .csv file")

def generate_pytorch_corpus(output_path='/storage/ice-shared/vip-vvk/data/AOT/psomu3/codenas/nasbench201_corpus_new.pkl',
                           datasets=['cifar10-valid', 'cifar100', 'ImageNet16-120'],
                           context_mode=None):
    """
    Generate corpus with PyTorch code representations for all 3 primitives modes.
    
    Args:
        output_path: Path to save the corpus (.pkl file)
        datasets: List of datasets to include metrics for
        context_mode: None (cell only), 'network' (full code), or 'comment' (docstring)
    
    Returns:
        DataFrame with all architectures and their PyTorch code in 3 primitives modes
    """
    api = get_api()
    context_str = f"with context_mode='{context_mode}'" if context_mode else "without context"
    print(f"Generating NASBench-201 PyTorch corpus for all primitives modes {context_str}...")
    print(f"Total architectures: {len(api)}")
    
    data_rows = []
    
    for arch_idx in tqdm(range(len(api)), desc="Processing architectures"):
        # Get architecture string
        arch_string = get_architecture_string(arch_idx)
        
        # Generate PyTorch code for all 3 primitives modes
        pytorch_code_inline = arch_to_pytorch_code(arch_idx, context_mode=context_mode, primitives_mode='inline')
        pytorch_code_helper = arch_to_pytorch_code(arch_idx, context_mode=context_mode, primitives_mode='helper')
        pytorch_code_exclude_helper = arch_to_pytorch_code(arch_idx, context_mode=context_mode, primitives_mode='exclude_helper')
        
        # Base entry with architecture info and all 3 pytorch code variants
        entry = {
            'arch_index': arch_idx,
            'arch_string': arch_string,
            'pytorch_code_inline': pytorch_code_inline,
            'pytorch_code_helper': pytorch_code_helper,
            'pytorch_code_exclude_helper': pytorch_code_exclude_helper
        }
        
        # Get properties for each dataset and add to entry
        for dataset in datasets:
            props = get_arch_properties(arch_idx, dataset=dataset)
            # Prefix keys with dataset name
            for key, value in props.items():
                entry[f'{dataset}_{key}'] = value
        
        data_rows.append(entry)
    
    # Create DataFrame
    print(f"\nCreating DataFrame...")
    df = pd.DataFrame(data_rows)
    
    # Save as pickle
    print(f"Saving corpus to {output_path}")
    df.to_pickle(output_path)
    
    # Also save as CSV
    csv_path = output_path.replace('.pkl', '.csv')
    print(f"Saving CSV version to {csv_path}")
    df.to_csv(csv_path, index=False)
    
    # Save first 10 rows as JSON for inspection
    json_path = output_path.replace('.pkl', '_sample.json')
    print(f"Saving sample JSON to {json_path}")
    df.head(10).to_json(json_path, orient='records', indent=2)
    
    print(f"\nCorpus generation complete!")
    print(f"Total entries: {len(df)}")
    print(f"DataFrame shape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nPyTorch code columns:")
    print(f"  - pytorch_code_inline")
    print(f"  - pytorch_code_helper")
    print(f"  - pytorch_code_exclude_helper")
    
    return df

if __name__ == "__main__":
    # Generate corpus for all three datasets
    df = generate_pytorch_corpus(output_path='/storage/ice-shared/vip-vvk/data/AOT/psomu3/codenas/nasbench201_corpus_pytorch_corrected.pkl')
    
    # Print example
    print("\n" + "="*80)
    print("Example corpus entries:")
    print("="*80)
    print("\nFirst 3 rows:")
    print(df.head(3))

    print("\nColumns:")
    print(list(df.columns))
    
    print("\n" + "="*80)
    print("Example architecture details:")
    print("="*80)
    example = df.iloc[0]
    print(f"\nArchitecture Index: {example['arch_index']}")
    print(f"Architecture String: {example['arch_string']}")
    print(f"\nSample Properties:")
    prop_cols = [col for col in df.columns if 'cifar10-valid' in col][:5]
    for col in prop_cols:
        print(f"  {col}: {example[col]}")
