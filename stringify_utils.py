
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
    Correctly parses source index from ~X suffix instead of relying on list position.
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
    
    # Parse edges to each target node
    for target_idx, node_group in enumerate(nodes):
        target = target_idx + 1  # Target nodes are 1, 2, 3
        ops = [e for e in node_group.split('|') if e]
        
        for op_str in ops:
            # Parse "op_name~source_idx"
            parts = op_str.split('~')
            op_name = parts[0]
            source = int(parts[1])
            edges.append((source, target, op_name))
    
    return edges

def generate_dependency_classes():
    """Generate ReLUConvBN and ResNetBasicblock classes needed by Network."""
    return """class ReLUConvBN(nn.Module):
    def __init__(self, channels, kernel_size, stride, padding, dilation, affine=True, track_running_stats=True):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(channels, channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=not affine),
            nn.BatchNorm2d(channels, affine=affine, track_running_stats=track_running_stats)
        )

    def forward(self, x):
        return self.op(x)

class ResNetBasicblock(nn.Module):
    def __init__(self, inplanes, planes, stride, affine=True, track_running_stats=True):
        super(ResNetBasicblock, self).__init__()
        assert stride == 1 or stride == 2
        self.conv_a = ReLUConvBN(
            inplanes, planes, 3, stride, 1, 1, affine, track_running_stats
        )
        self.conv_b = ReLUConvBN(
            planes, planes, 3, 1, 1, 1, affine, track_running_stats
        )
        if stride == 2:
            self.downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
                nn.Conv2d(
                    inplanes, planes, kernel_size=1, stride=1, padding=0, bias=False
                ),
            )
        elif inplanes != planes:
            self.downsample = ReLUConvBN(
                inplanes, planes, 1, 1, 0, 1, affine, track_running_stats
            )
        else:
            self.downsample = None
        self.in_dim = inplanes
        self.out_dim = planes
        self.stride = stride
        self.num_conv = 2

    def forward(self, inputs):
        basicblock = self.conv_a(inputs)
        basicblock = self.conv_b(basicblock)
        if self.downsample is not None:
            residual = self.downsample(inputs)
        else:
            residual = inputs
        return residual + basicblock"""

def generate_network_class():
    """Generate the Network class that wraps Cell for full NASBench-201 architecture."""
    return """class Network(nn.Module):
    def __init__(self, channels, N, genotype, num_classes):
        super(Network, self).__init__()
        self.C = channels
        self.N = N

        self.stem = nn.Sequential(
            nn.Conv2d(3, self.C, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(self.C)
        )

        layer_channels = [self.C] * N + [self.C * 2] + [self.C * 2] * N + [self.C * 4] + [self.C * 4] * N
        layer_reductions = [False] * N + [True] + [False] * N + [True] + [False] * N

        C_prev = self.C
        self.cells = nn.ModuleList()
        for index, (C_curr, reduction) in enumerate(
            zip(layer_channels, layer_reductions)
        ):
            if reduction:
                cell = ResNetBasicblock(C_prev, C_curr, 2, True)
            else:
                cell = Cell(C_curr)
            self.cells.append(cell)
            C_prev = C_curr
        self._Layer = len(self.cells)

        self.lastact = nn.Sequential(nn.BatchNorm2d(C_prev), nn.ReLU(inplace=True))
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

    def get_training_config(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=0.1,
            momentum=0.9,
            weight_decay=5e-4,
            nesterov=True
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=200,
            eta_min=0
        )
        config = {
            'optimizer': optimizer,
            'scheduler': scheduler,
            'batch_size': 256,
            'epochs': 200
        }
        return config

    def forward(self, inputs):
        feature = self.stem(inputs)
        for i, cell in enumerate(self.cells):
            feature = cell(feature)

        out = self.lastact(feature)
        out = self.global_pooling(out)
        out = out.view(out.size(0), -1)
        logits = self.classifier(out)

        return out, logits"""

def generate_context_docstring():
    """Generate a docstring describing the broader network context."""
    return '''"""
Task: CIFAR-10 image classification (10 classes, 32x32 RGB images).

This Cell is one building block within a larger neural network.
Full architecture:
- Stem layer: Conv2d(3 channels -> 16 channels, 3x3 kernel) + BatchNorm2d.
- Main head: stacks 15 copies of the Cell into a sequence. 1 ResNetBasicblock layer is inserted every 5 Cells (total 2).
- Final layers: BatchNorm2d + ReLU + Global Average Pooling + Linear layer to 10 classes.

Helpers:
- ReLUConvBN: Sequential ReLU → Conv → BatchNorm (pre-activation)
- ResNetBasicblock: Residual block with 2 ReLUConvBN plus 1 skip connection with input downsampling

Training Details: SGD optimizer with momentum=0.9, weight_decay=5e-4, initial learning_rate=0.1 
with cosine annealing schedule over 200 epochs, batch_size=256, plus standard data augmentation.
"""'''

def generate_helper_class():
    """
    Generate the ReLU_Conv2d_BatchNorm helper class definition.
    This helper wraps the pre-activation pattern: ReLU→Conv→BN used in NASBench-201.
    
    Returns:
        String containing the helper class definition
    """
    helper_code = [
        "class ReLU_Conv2d_BatchNorm(nn.Module):",
        "    def __init__(self, channels, kernel_size, stride, padding):",
        "        super().__init__()",
        "        self.op = nn.Sequential(",
        "            nn.ReLU(inplace=False),",
        "            nn.Conv2d(channels, channels, kernel_size, stride=stride, padding=padding, bias=False),",
        "            nn.BatchNorm2d(channels)",
        "        )",
        "    ",
        "    def forward(self, x):",
        "        return self.op(x)"
    ]
    return "\n".join(helper_code)

def to_pytorch_code(edges, context_mode=None, primitives_mode='inline'):
    """Format A: The Class Definition
    
    Args:
        edges: List of (src, dst, op) tuples representing the cell architecture
        context_mode: None (cell only), "network" (full code), or "comment" (docstring description)
        primitives_mode: 'inline' (default, use nn.Sequential), 'helper' (use Conv2d_BatchNorm_ReLU helper),
                        or 'exclude_helper' (reference helper but don't define it)
    
    Returns:
        String containing PyTorch code for Cell (and optionally Network or context docstring)
    """
    
    # Helper function to get op string based on primitives_mode
    def get_op_string(op):
        if op == 'skip_connect':
            return None  # Identity - don't create an op, just pass through
        elif op == 'avg_pool_3x3':
            return 'nn.AvgPool2d(kernel_size=3, stride=1, padding=1)'
        elif op == 'nor_conv_1x1':
            if primitives_mode == 'inline':
                return 'nn.Sequential(nn.ReLU(inplace=False), nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(channels))'
            else:  # helper or exclude_helper
                return 'ReLU_Conv2d_BatchNorm(channels, kernel_size=1, stride=1, padding=0)'
        elif op == 'nor_conv_3x3':
            if primitives_mode == 'inline':
                return 'nn.Sequential(nn.ReLU(inplace=False), nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(channels))'
            else:  # helper or exclude_helper
                return 'ReLU_Conv2d_BatchNorm(channels, kernel_size=3, stride=1, padding=1)'
        else:  # none or unknown
            return None
    lines = ["class Cell(nn.Module):", "    def __init__(self, channels):", "        super().__init__()"]
    
    # Init Ops
    for i, (src, dst, op) in enumerate(edges):
        if op == 'none': continue
        op_str = get_op_string(op)
        if op_str:
            lines.append(f"        self.op_{src}_{dst} = {op_str}")
    
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
    
    cell_code = "\n".join(lines)
    
    # Prepend helper class if primitives_mode is 'helper'
    if primitives_mode == 'helper':
        helper_code = generate_helper_class()
        cell_code = f"{helper_code}\n\n{cell_code}"
    
    # Apply context mode wrapping
    if context_mode == "network":
        dependency_code = generate_dependency_classes()
        network_code = generate_network_class()
        return f"{dependency_code}\n\n{cell_code}\n\n{network_code}"
    elif context_mode == "comment":
        docstring = generate_context_docstring()
        return f"{docstring}\n\n{cell_code}"
    else:
        return cell_code

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

def arch_to_pytorch_code(arch_index_or_str, context_mode=None, primitives_mode='inline'):
    """
    Convert a NASBench-201 architecture to PyTorch code string.
    
    Args:
        arch_index_or_str: Either an architecture index (int) or architecture string.
        context_mode: None (cell only), "network" (full code), or "comment" (docstring description).
        primitives_mode: 'inline' (default), 'helper' (define and use helper), or 'exclude_helper' (use but don't define).
    
    Returns:
        String containing the PyTorch class definition(s).
    """
    arch_str = get_architecture_string(arch_index_or_str)
    edges = parse_nb201_string(arch_str)
    return to_pytorch_code(edges, context_mode=context_mode, primitives_mode=primitives_mode)

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

def arch_to_all_formats(arch_index_or_str, context_mode=None, primitives_mode='inline'):
    """
    Convert a NASBench-201 architecture to all three formats.
    
    Args:
        arch_index_or_str: Either an architecture index (int) or architecture string.
        context_mode: None (cell only), "network" (full code), or "comment" (docstring) for PyTorch format.
        primitives_mode: 'inline' (default), 'helper' (define and use helper), or 'exclude_helper' (use but don't define).
    
    Returns:
        Dictionary with keys 'pytorch', 'onnx', 'grammar' containing the three formats.
    """
    arch_str = get_architecture_string(arch_index_or_str)
    edges = parse_nb201_string(arch_str)
    
    return {
        'pytorch': to_pytorch_code(edges, context_mode=context_mode, primitives_mode=primitives_mode),
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