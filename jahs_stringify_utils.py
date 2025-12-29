"""
JAHS-Bench-201 architecture stringification utilities.
Converts JAHS-Bench configurations to PyTorch code and ONNX-Net format.
Matches NAS-Bench-201 stringify_utils.py structure for fair comparison.
"""

import torch
import torch.nn as nn
import numpy as np


# JAHS-Bench uses same operations as NAS-Bench-201
# Op codes: 0=skip_connect, 1=none, 2=conv_1x1, 3=conv_3x3, 4=avg_pool_3x3
OP_NAMES = {
    0: 'skip_connect',
    1: 'none',  # zero operation
    2: 'nor_conv_1x1',  # Match NAS-Bench naming
    3: 'nor_conv_3x3',
    4: 'avg_pool_3x3'
}

# MAPPING: JAHS Ops -> Real Ops (matching NAS-Bench format)
OP_MAP = {
    'nor_conv_1x1': 'Conv2d(16, 16, kernel_size=1)',
    'nor_conv_3x3': 'Conv2d(16, 16, kernel_size=3, padding=1)',
    'avg_pool_3x3': 'AvgPool2d(kernel_size=3, padding=1, stride=1)',
    'skip_connect': 'Identity()',
    'none': 'Zero()',
}

# MAPPING: JAHS Ops -> ONNX-Net Style (matching NAS-Bench format)
ONNX_MAP = {
    'nor_conv_1x1': ('Conv', '(1x16x32x32, 16x1x1, 16) (k=1,p=0,s=1)'),
    'nor_conv_3x3': ('Conv', '(1x16x32x32, 16x3x3, 16) (k=3,p=1,s=1)'),
    'avg_pool_3x3': ('MaxPool', '(k=3,p=1,s=1)'),
    'skip_connect': ('Identity', ''),
    'none': ('Constant', '(value=0)'),
}


def parse_jahs_ops(config):
    """
    Parse JAHS-Bench config into edge list format (matching NAS-Bench structure).
    
    Args:
        config: JAHS-Bench configuration dict with Op1-Op6
    
    Returns:
        List of (src, dst, op_name) tuples representing edges
    """
    # JAHS-Bench has Op1-Op6 which map to the 6 edges in the 4-node DAG
    # Same structure as NAS-Bench-201:
    # Edge 0-1 (to node 1)
    # Edge 0-2, 1-2 (to node 2)
    # Edge 0-3, 1-3, 2-3 (to node 3)
    
    ops = [config[f'Op{i}'] for i in range(1, 7)]
    op_names = [OP_NAMES[op] for op in ops]
    
    edges = [
        (0, 1, op_names[0]),  # Op1: 0->1
        (0, 2, op_names[1]),  # Op2: 0->2
        (1, 2, op_names[2]),  # Op3: 1->2
        (0, 3, op_names[3]),  # Op4: 0->3
        (1, 3, op_names[4]),  # Op5: 1->3
        (2, 3, op_names[5]),  # Op6: 2->3
    ]
    
    return edges


def to_pytorch_code(edges, config):
    """
    Format A: The Class Definition (matching NAS-Bench format + JAHS hyperparameters).
    
    For JAHS-Bench, includes hyperparameters as actual code in the network definition.
    For fair comparison, Cell structure matches NAS-Bench exactly.
    """
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
    
    # Add JAHS-specific network wrapper with hyperparameters
    activation_cls = f"nn.{config['Activation']}()"
    N = config['N']
    W = config['W']
    lr = config['LearningRate']
    wd = config['WeightDecay']
    opt = config['Optimizer']
    
    lines.append("")
    lines.append("")
    lines.append("class Network(nn.Module):")
    lines.append("    def __init__(self, num_classes=10):")
    lines.append("        super().__init__()")
    lines.append(f"        self.learning_rate = {lr}")
    lines.append(f"        self.weight_decay = {wd}")
    lines.append(f"        # TrivialAugment: if True, use random augmentations; if False, use manual transforms")
    lines.append(f"        self.trivial_augment = {config['TrivialAugment']}")
    lines.append(f"        # Resolution: input image scale factor (e.g., 0.5 = 16x16 for 32x32 images)")
    lines.append(f"        self.resolution = {config['Resolution']}")
    lines.append("")
    lines.append(f"        self.activation = {activation_cls}")
    lines.append(f"        self.stem = nn.Sequential(")
    lines.append(f"            nn.Conv2d(3, {W}, kernel_size=3, padding=1, bias=False),")
    lines.append(f"            nn.BatchNorm2d({W}),")
    lines.append(f"            self.activation")
    lines.append(f"        )")
    lines.append(f"        self.cells = nn.ModuleList([Cell() for _ in range({N})])")
    lines.append(f"        self.global_pool = nn.AdaptiveAvgPool2d(1)")
    lines.append(f"        self.classifier = nn.Linear({W}, num_classes)")
    lines.append("")
    lines.append("    def forward(self, x):")
    lines.append("        x = self.stem(x)")
    lines.append("        for cell in self.cells:")
    lines.append("            x = cell(x)")
    lines.append("        x = self.global_pool(x)")
    lines.append("        x = x.view(x.size(0), -1)")
    lines.append("        x = self.classifier(x)")
    lines.append("        return x")
    lines.append("")
    lines.append("    def get_optimizer(self):")
    
    if opt == 'SGD':
        lines.append("        return torch.optim.SGD(self.parameters(), lr=self.learning_rate,")
        lines.append("                                weight_decay=self.weight_decay, momentum=0.9)")
    elif opt == 'AdamW':
        lines.append("        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate,")
        lines.append("                                 weight_decay=self.weight_decay, betas=(0.9, 0.999))")
    else:  # Adam
        lines.append("        return torch.optim.Adam(self.parameters(), lr=self.learning_rate,")
        lines.append("                                 weight_decay=self.weight_decay)")
    
    return "\n".join(lines)


# --- HIGH-LEVEL API FUNCTIONS (matching NAS-Bench interface) ---

def jahs_config_to_pytorch_code(config):
    """
    Convert a JAHS-Bench-201 configuration to PyTorch code string.
    Includes both Cell structure (matching NAS-Bench) and JAHS hyperparameters.
    
    Args:
        config: JAHS-Bench configuration dict
    
    Returns:
        String containing the PyTorch class definitions (Cell + JAHSNetwork)
    """
    edges = parse_jahs_ops(config)
    return to_pytorch_code(edges, config)


def jahs_config_to_all_formats(config):
    """
    Convert a JAHS-Bench configuration to all supported formats.
    
    For JAHS-Bench, only PyTorch code is generated (includes hyperparameters).
    ONNX format omitted since linearized graphs cannot capture hyperparameters.
    
    Args:
        config: JAHS-Bench configuration dict
    
    Returns:
        Dictionary with 'pytorch_code' key only
    """
    return {
        'pytorch_code': jahs_config_to_pytorch_code(config)
    }


if __name__ == '__main__':
    # Test with a sample config
    test_config = {
        'Optimizer': 'SGD',
        'LearningRate': 0.1,
        'WeightDecay': 5e-05,
        'Activation': 'Mish',
        'TrivialAugment': False,
        'Op1': 4,
        'Op2': 1,
        'Op3': 2,
        'Op4': 0,
        'Op5': 2,
        'Op6': 1,
        'N': 5,
        'W': 16,
        'Resolution': 1.0,
    }
    
    print("="*80)
    print("PyTorch Code (Cell + JAHSNetwork with hyperparameters):")
    print("="*80)
    print(jahs_config_to_pytorch_code(test_config))
    
    print("\n" + "="*80)
    print("Key Features:")
    print("="*80)
    print("✓ Cell class matches NAS-Bench-201 format exactly")
    print("✓ JAHSNetwork wrapper includes all hyperparameters as code:")
    print("  - Optimizer type (SGD/Adam)")
    print("  - Learning rate")
    print("  - Weight decay")
    print("  - Activation function (Mish/Hardswish/ReLU)")
    print("  - Trivial augmentation flag")
    print("  - Depth multiplier (N)")
    print("  - Width multiplier (W)")
    print("  - Resolution")
    print("✓ No ONNX format (can't capture hyperparameters)")
    print("="*80)

