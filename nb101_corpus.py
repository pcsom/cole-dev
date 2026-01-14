import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
import sys
import os

# -------------------------------------------------------------------------
# 1. NAS-Bench-101 Architecture to PyTorch String Converter
# -------------------------------------------------------------------------

OP_MAP = {
    'conv3x3-bn-relu': 'nn.Sequential(nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(16), nn.ReLU())',
    'conv1x1-bn-relu': 'nn.Sequential(nn.Conv2d(16, 16, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(16), nn.ReLU())',
    'maxpool3x3': 'nn.MaxPool2d(kernel_size=3, stride=1, padding=1)',
}

def get_op_string(op_label):
    return OP_MAP.get(op_label, None)

def spec_to_pytorch_code(matrix, ops):
    """
    Converts NAS-Bench-101 matrix and ops to a PyTorch code string.
    """
    num_vertices = len(ops)
    
    lines = []
    lines.append("class Cell(nn.Module):")
    lines.append("    def __init__(self, C_in, C_out, stride):")
    lines.append("        super().__init__()")
    lines.append("        self.input_projection = nn.Sequential(nn.Conv2d(C_in, 16, kernel_size=1, bias=False), nn.BatchNorm2d(16), nn.ReLU())")
    
    # Define operations for internal nodes
    for t in range(1, num_vertices - 1):
        op_label = ops[t]
        op_str = get_op_string(op_label)
        if op_str:
            lines.append(f"        self.op_{t} = {op_str}")

    lines.append("")
    lines.append("    def forward(self, x):")
    lines.append("        node_0 = self.input_projection(x)")
    
    # Internal nodes
    for t in range(1, num_vertices - 1):
        incoming_indices = [src for src in range(t) if matrix[src][t] == 1]
        
        if not incoming_indices:
            lines.append(f"        node_{t} = torch.zeros_like(node_0)")
        else:
            inputs = [f"node_{src}" for src in incoming_indices]
            sum_str = " + ".join(inputs)
            
            op_label = ops[t]
            op_str = get_op_string(op_label)
            
            if op_str:
                lines.append(f"        node_{t} = self.op_{t}({sum_str})")
            else:
                lines.append(f"        node_{t} = {sum_str}")
    
    # Output node (concatenation)
    output_idx = num_vertices - 1
    incoming_indices = [src for src in range(output_idx) if matrix[src][output_idx] == 1]
    
    if not incoming_indices:
         lines.append("        return torch.zeros_like(node_0)")
    else:
        cat_parts = [f"node_{src}" for src in incoming_indices]
        cat_str = ", ".join(cat_parts)
        lines.append(f"        return torch.cat([{cat_str}], dim=1)")

    return "\n".join(lines)

def nasbench101_arch_to_pytorch(matrix, ops):
    """Wrapper to handle direct matrix/ops input."""
    return spec_to_pytorch_code(matrix, ops)

# -------------------------------------------------------------------------
# 2. NAS-Bench-101 API Lazy Loader & Utils
# -------------------------------------------------------------------------

# NEED TO UPDATE
NASBENCH_TFRECORD_PATH = '/storage/ice-shared/vip-vvk/data/AOT/psomu3/codenas/nasbench_full.tfrecord' 

_nasbench_api = None

def get_nasbench_api():
    """Lazy load the NASBench API."""
    global _nasbench_api
    if _nasbench_api is None:
        try:
            from nasbench import api
            print(f"Loading NAS-Bench-101 from {NASBENCH_TFRECORD_PATH}...")
            _nasbench_api = api.NASBench(NASBENCH_TFRECORD_PATH)
        except ImportError:
            raise ImportError("Could not import nasbench. Please ensure google-research/nasbench is in your PYTHONPATH.")
        except Exception as e:
            raise RuntimeError(f"Failed to load NASBench file: {e}")
    return _nasbench_api

def get_arch_properties(api, module_hash, epochs=108):
    """
    Get fitness metrics for a specific architecture hash.
    Averages results across all repeats for the specified epoch count.
    """
    # fixed_stat contains adjacency, ops, parameters
    fixed_stat = api.fixed_statistics[module_hash]
    
    # computed_stat contains training metrics (epochs -> list of repeats)
    computed_stat = api.computed_statistics[module_hash]
    
    properties = {
        'trainable_parameters': fixed_stat['trainable_parameters']
    }
    
    if epochs in computed_stat:
        repeats = computed_stat[epochs]
        # Average metrics across repeats
        metrics_to_avg = [
            'final_test_accuracy', 'final_train_accuracy', 'final_validation_accuracy', 
            'final_training_time'
        ]
        
        for metric in metrics_to_avg:
            values = [r[metric] for r in repeats]
            properties[metric] = np.mean(values)
            # Optional: store std dev or individual runs if needed
            # properties[f'{metric}_std'] = np.std(values)
    else:
        # Fallback if 108 epochs not available (unlikely for full dataset)
        for metric in ['final_test_accuracy', 'final_train_accuracy', 'final_validation_accuracy', 'final_training_time']:
            properties[metric] = None
            
    return properties, fixed_stat

# -------------------------------------------------------------------------
# 3. Corpus Generation
# -------------------------------------------------------------------------

def generate_nasbench101_corpus(output_path='/storage/ice-shared/vip-vvk/data/AOT/psomu3/codenas/nasbench101_corpus.csv'):
    """
    Generate corpus of architecture code strings and their properties for NAS-Bench-101.
    """
    api = get_nasbench_api()
    print("Generating NASBench-101 corpus...")
    
    # Get all hashes
    all_hashes = list(api.hash_iterator())
    print(f"Total architectures: {len(all_hashes)}")
    
    data_rows = []
    
    for arch_hash in tqdm(all_hashes, desc="Processing architectures"):
        # 1. Get properties and raw spec
        props, fixed_stat = get_arch_properties(api, arch_hash, epochs=108)
        
        matrix = fixed_stat['module_adjacency']
        ops = fixed_stat['module_operations']
        
        # 2. Generate PyTorch Code string
        pytorch_code = nasbench101_arch_to_pytorch(matrix, ops)
        
        # 3. Create entry
        entry = {
            'arch_hash': arch_hash,
            # For 101, 'arch_string' isn't standard like 201, so we store the raw spec as a string repr
            'arch_string': str((matrix.tolist(), ops)), 
            'pytorch_code': pytorch_code,
            
            # Metrics (CIFAR-10 is the only dataset in NB101)
            'cifar10_test_accuracy': props['final_test_accuracy'],
            'cifar10_train_accuracy': props['final_train_accuracy'],
            'cifar10_valid_accuracy': props['final_validation_accuracy'],
            'cifar10_training_time': props['final_training_time'],
            'trainable_parameters': props['trainable_parameters']
        }
        
        data_rows.append(entry)
    
    # Create DataFrame
    print(f"\nCreating DataFrame...")
    df = pd.DataFrame(data_rows)
    
    # Save to CSV
    print(f"Saving corpus to {output_path}")
    df.to_csv(output_path, index=False)
    
    # Save pickle
    pkl_path = output_path.replace('.csv', '.pkl')
    print(f"Saving pickle version to {pkl_path}")
    df.to_pickle(pkl_path)
    
    print(f"\nCorpus generation complete!")
    print(f"Total entries: {len(df)}")
    print(f"DataFrame shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    return df

if __name__ == "__main__":
    # Ensure output directory exists
    output_dir = './data'
    os.makedirs(output_dir, exist_ok=True)
    
    save_path = os.path.join(output_dir, '/storage/ice-shared/vip-vvk/data/AOT/psomu3/codenas/nasbench101_corpus.csv')
    
    # Run generation
    try:
        df = generate_nasbench101_corpus(save_path)
        
        # Validation print
        print("\n" + "="*80)
        print("Example corpus entry:")
        print("="*80)
        example = df.iloc[0]
        print(f"Hash: {example['arch_hash']}")
        print(f"Test Acc: {example['cifar10_test_accuracy']}")
        print(f"\nPyTorch Code:\n{example['pytorch_code']}")
        
    except Exception as e:
        print(f"\nError: {e}")
        print("Please check that NASBENCH_TFRECORD_PATH is correct and the nasbench library is installed.")