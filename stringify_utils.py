
import torch
import numpy as np
from nas_201_api import NASBench201API as API

# Lazy initialization of API to avoid loading until needed
_api_instance = None
_api_path = '/storage/ice-shared/vip-vvk/data/AOT/psomu3/codenas/NAS-Bench-201-v1_1-096897.pth'

def get_api():
    """Get or create the NASBench-201 API instance (singleton pattern)."""
    global _api_instance
    if _api_instance is None:
        _api_instance = API(_api_path, verbose=False)
    return _api_instance

def get_architecture_string(arch_index_or_str):
    """
    Get architecture string from NASBench-201 API.
    
    Args:
        arch_index_or_str: Either an architecture index (int) or architecture string.
    
    Returns:
        Architecture string in NASBench-201 format.
    """
    if isinstance(arch_index_or_str, int):
        api = get_api()
        return api.arch(arch_index_or_str)
    else:
        return arch_index_or_str

# MAPPING: NASBench-201 Ops -> Real Ops
OP_MAP = {
    'nor_conv_1x1': 'Conv2d(16, 16, kernel_size=1)',
    'nor_conv_3x3': 'Conv2d(16, 16, kernel_size=3, padding=1)',
    'avg_pool_3x3': 'AvgPool2d(kernel_size=3, padding=1, stride=1)',
    'skip_connect': 'Identity()',
    'none': 'Zero()',
}

# MAPPING: NASBench-201 Ops -> ONNX-Net Style
ONNX_MAP = {
    'nor_conv_1x1': ('Conv', '(1x16x32x32, 16x1x1, 16) (k=1,p=0,s=1)'),
    'nor_conv_3x3': ('Conv', '(1x16x32x32, 16x3x3, 16) (k=3,p=1,s=1)'),
    'avg_pool_3x3': ('MaxPool', '(k=3,p=1,s=1)'), # ONNX usually calls it MaxPool or AveragePool
    'skip_connect': ('Identity', ''),
    'none': ('Constant', '(value=0)'),
}

# MAPPING: NASBench-201 Ops -> Einspace Grammar Style
# Based on the snippet "computation<o>" and "routing[M]"
GRAMMAR_MAP = {
    'nor_conv_1x1': 'computation<conv1x1>',
    'nor_conv_3x3': 'computation<conv3x3>',
    'avg_pool_3x3': 'computation<avgpool3x3>',
    'skip_connect': 'computation<identity>',
    'none': 'computation<zero>',
}

def parse_nb201_string(arch_str):
    """
    Parses '|op~0|+|op~0|op~1|+|op~0|op~1|op~2|' into a list of (src, dst, op_name).
    """
    nodes = arch_str.strip('|').split('+')
    edges = []
    # Node 0 is input.
    # Node 1 connects from [0]
    # Node 2 connects from [0, 1]
    # Node 3 connects from [0, 1, 2]
    
    # NB201 string structure:
    # nodes[0] defines edges to Node 1: "op~0" (from node 0)
    # nodes[1] defines edges to Node 2: "op~0|op~1" (from 0, then 1)
    # nodes[2] defines edges to Node 3: "op~0|op~1|op~2" (from 0, 1, 2)
    
    # Parse edges to Node 1
    ops = [e for e in nodes[0].split('|') if e]
    edges.append((0, 1, ops[0].split('~')[0]))
    
    # Parse edges to Node 2
    ops = [e for e in nodes[1].split('|') if e]
    edges.append((0, 2, ops[0].split('~')[0]))
    edges.append((1, 2, ops[1].split('~')[0]))
    
    # Parse edges to Node 3
    ops = [e for e in nodes[2].split('|') if e]
    edges.append((0, 3, ops[0].split('~')[0]))
    edges.append((1, 3, ops[1].split('~')[0]))
    edges.append((2, 3, ops[2].split('~')[0]))
    
    return edges

def to_pytorch_code(edges):
    """Format A: The Class Definition (Your Novelty)"""
    lines = ["class Cell(nn.Module):", "    def __init__(self):", "        super().__init__()"]
    
    # Init Ops
    for i, (src, dst, op) in enumerate(edges):
        if op == 'none': continue
        lines.append(f"        self.op_{src}_{dst} = nn.{OP_MAP[op]}")
    
    lines.append("")
    lines.append("    def forward(self, x):")
    lines.append("        node_0 = x")
    
    # Forward Logic
    for target in [1, 2, 3]:
        incoming = []
        for src, dst, op in edges:
            if dst == target:
                if op == 'none': continue
                if op == 'skip_connect':
                    incoming.append(f"node_{src}")
                else:
                    incoming.append(f"self.op_{src}_{dst}(node_{src})")
        
        if not incoming:
            lines.append(f"        node_{target} = torch.zeros_like(node_0)")
        else:
            sum_str = " + ".join(incoming)
            lines.append(f"        node_{target} = {sum_str}")
            
    lines.append("        return node_3")
    return "\n".join(lines)

def to_onnx_net(edges):
    """Format B: ONNX-Net Linearized String"""
    # Mimics: Conv (Val, Shape) -> Val_Out
    lines = []
    lines.append("Input (1x16x32x32) -> Value_0")
    
    val_map = {0: "Value_0"}
    
    # In NB201, we process nodes 1, 2, 3 in order
    for target in [1, 2, 3]:
        inputs_to_sum = []
        for src, dst, op in edges:
            if dst == target:
                # Generate intermediate value name
                input_val = val_map[src]
                output_val = f"Value_{src}_{dst}"
                
                onnx_op, params = ONNX_MAP[op]
                line = f"{onnx_op} ({input_val}) {params} -> {output_val}:1x16x32x32"
                lines.append(line)
                inputs_to_sum.append(output_val)
        
        # Aggregation (Sum)
        final_val = f"Value_{target}"
        if inputs_to_sum:
            inputs_str = ", ".join(inputs_to_sum)
            lines.append(f"Add ({inputs_str}) -> {final_val}:1x16x32x32")
        val_map[target] = final_val

    return "\n".join(lines)

def to_transferable_grammar(edges):
    """Format C: Transferable Surrogates Grammar Style"""
    # Mimics: routing[computation<conv>, computation<skip>...]
    # Since NB201 is a fixed DAG, we can represent it as a flat list of operations 
    # sorted topologically, wrapped in the grammar's routing block.
    
    ops_list = []
    for src, dst, op in edges:
        ops_list.append(GRAMMAR_MAP[op])
    
    # Wrap in the "routing" function seen in the paper
    # The structure is implicitly defined by the fixed NASBench-201 edges
    # so identifying the ops in order is often sufficient for the embedding.
    inner = ", ".join(ops_list)
    return f"routing[{inner}]"

# --- HIGH-LEVEL API FUNCTIONS ---

def arch_to_pytorch_code(arch_index_or_str):
    """
    Convert a NASBench-201 architecture to PyTorch code string.
    
    Args:
        arch_index_or_str: Either an architecture index (int) or architecture string.
    
    Returns:
        String containing the PyTorch class definition.
    """
    arch_str = get_architecture_string(arch_index_or_str)
    edges = parse_nb201_string(arch_str)
    return to_pytorch_code(edges)

def arch_to_onnx_net(arch_index_or_str):
    """
    Convert a NASBench-201 architecture to ONNX-Net linearized string.
    
    Args:
        arch_index_or_str: Either an architecture index (int) or architecture string.
    
    Returns:
        String containing the ONNX-Net representation.
    """
    arch_str = get_architecture_string(arch_index_or_str)
    edges = parse_nb201_string(arch_str)
    return to_onnx_net(edges)

def arch_to_grammar(arch_index_or_str):
    """
    Convert a NASBench-201 architecture to transferable grammar string.
    
    Args:
        arch_index_or_str: Either an architecture index (int) or architecture string.
    
    Returns:
        String containing the grammar representation.
    """
    arch_str = get_architecture_string(arch_index_or_str)
    edges = parse_nb201_string(arch_str)
    return to_transferable_grammar(edges)

def arch_to_all_formats(arch_index_or_str):
    """
    Convert a NASBench-201 architecture to all three formats.
    
    Args:
        arch_index_or_str: Either an architecture index (int) or architecture string.
    
    Returns:
        Dictionary with keys 'pytorch', 'onnx', 'grammar' containing the three formats.
    """
    arch_str = get_architecture_string(arch_index_or_str)
    edges = parse_nb201_string(arch_str)
    
    return {
        'pytorch': to_pytorch_code(edges),
        'onnx': to_onnx_net(edges),
        'grammar': to_transferable_grammar(edges)
    }

# --- EXAMPLE USAGE ---
if __name__ == "__main__":
    print("Example 1:")
    print("=" * 80)
    arch_idx = 0
    formats = arch_to_all_formats(arch_idx)
    
    print(f"\nArchitecture Index: {arch_idx}")
    print(f"Architecture String: {get_architecture_string(arch_idx)}")
    print("\n--- FORMAT A: PYTORCH CODE ---")
    print(formats['pytorch'])
    print("\n--- FORMAT B: ONNX-NET STRING (BASELINE 1) ---")
    print(formats['onnx'])
    print("\n--- FORMAT C: GRAMMAR STRING (BASELINE 2) ---")
    print(formats['grammar'])
    
    print("\n" + "=" * 80)
    print("\nExample 2:")
    print("=" * 80)
    arch_idx = 100
    print(f"\nArchitecture Index: {arch_idx}")
    print(f"Architecture String: {get_architecture_string(arch_idx)}")
    print("\nPyTorch Code:")
    print(arch_to_pytorch_code(arch_idx))
    
    print("\n" + "=" * 80)
    print("\nExample 3:")
    print("=" * 80)
    arch_idx = 200
    print(f"\nArchitecture Index: {arch_idx}")
    print(f"Architecture String: {get_architecture_string(arch_idx)}")
    print("\nPyTorch Code:")
    print(arch_to_pytorch_code(arch_idx))