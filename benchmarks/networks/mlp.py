#!/usr/bin/env python3
"""
Minimal test to see what Orion does with a multi-layer MLP - no translation, just print Orion's IR.
"""

import sys
from pathlib import Path
import torch

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import Orion (assuming it's available)
import orion
import orion.nn as on


class MLP(on.Module):
    
    def __init__(self, num_classes=10):
        super().__init__()
        self.flatten = on.Flatten()
        
        self.fc1 = on.Linear(784, 128)
        self.bn1 = on.BatchNorm1d(128)
        self.act1 = on.Quad()
        
        self.fc2 = on.Linear(128, 128)
        self.bn2 = on.BatchNorm1d(128)
        self.act2 = on.Quad()
        
        self.fc3 = on.Linear(128, num_classes)
    
    def forward(self, x):
        x = self.flatten(x)
        x = self.act1(self.bn1(self.fc1(x)))
        x = self.act2(self.bn2(self.fc2(x)))
        return self.fc3(x)


def main():
    print("🚀 MLP Orion Test - Show Orion's Actual IR")
    print("=" * 50)
    
    # Initialize Orion scheme with actual MLP config values
    orion.init_scheme({
        'ckks_params': {
            'LogN': 13,
            'LogQ': [29, 26, 26, 26, 26, 26],
            'LogP': [29, 29],
            'LogScale': 26,
            'H': 8192
        },
        'orion': {
            'backend': 'lattigo'
        }
    })
    print("✅ Orion scheme initialized")
    
    # Create the model
    model = MLP()
    print(f"✅ Model: MLP with layers (matching Orion examples):")
    print(f"   - flatten: Flatten()")
    print(f"   - fc1: Linear(784, 128)")
    print(f"   - bn1: BatchNorm1d(128)")
    print(f"   - act1: Quad()")
    print(f"   - fc2: Linear(128, 128)")
    print(f"   - bn2: BatchNorm1d(128)")
    print(f"   - act2: Quad()")
    print(f"   - fc3: Linear(128, 10)")
    
    # Create input (MNIST format that Orion expects - 4D tensor)
    x = torch.randn(1, 1, 28, 28)  # Batch, Channel, Height, Width
    print(f"✅ Input shape: {x.shape} (MNIST format: batch, channel, height, width)")
    
    # Test cleartext forward
    model.eval()
    with torch.no_grad():
        output = model(x)
    print(f"✅ Cleartext output shape: {output.shape}")
    
    # Fit and compile the model
    print(f"\n🔧 Orion fit and compile...")
    orion.fit(model, x)
    input_level = orion.compile(model)
    print(f"✅ Compiled successfully, input level: {input_level}")
    
    # Print what Orion created during compile
    print(f"\n📋 Orion Compilation Results:")
    print(f"=" * 40)
    
    for name, layer in model.named_modules():
        if hasattr(layer, 'diagonals') or hasattr(layer, 'transform_ids') or hasattr(layer, 'level'):
            print(f"\nLayer: {name}")
            print(f"  Type: {layer.__class__.__name__}")
            print(f"  Level: {getattr(layer, 'level', 'N/A')}")
            
            if hasattr(layer, 'weight'):
                print(f"  Weight shape: {layer.weight.shape}")
            if hasattr(layer, 'bias') and layer.bias is not None:
                print(f"  Bias shape: {layer.bias.shape}")
            
            if hasattr(layer, 'diagonals'):
                print(f"  Diagonals: {len(layer.diagonals) if layer.diagonals else 0} blocks")
                if layer.diagonals:
                    print(f"    Keys: {list(layer.diagonals.keys())[:3]}{'...' if len(layer.diagonals) > 3 else ''}")
            
            if hasattr(layer, 'transform_ids'):
                print(f"  Transform IDs: {len(layer.transform_ids) if layer.transform_ids else 0}")
                if layer.transform_ids:
                    print(f"    Keys: {list(layer.transform_ids.keys())[:3]}{'...' if len(layer.transform_ids) > 3 else ''}")
            
            if hasattr(layer, 'output_rotations'):
                print(f"  Output rotations: {layer.output_rotations}")
            
            if hasattr(layer, 'on_bias_ptxt'):
                print(f"  Bias plaintext: {'Created' if layer.on_bias_ptxt is not None else 'None'}")
            
            # For activation layers, show polynomial info
            if hasattr(layer, 'polynomial_coeffs'):
                print(f"  Polynomial coeffs: {layer.polynomial_coeffs}")
            
            # For batch norm layers, show parameters
            if hasattr(layer, 'running_mean') and hasattr(layer, 'running_var'):
                print(f"  BatchNorm mean shape: {layer.running_mean.shape}")
                print(f"  BatchNorm var shape: {layer.running_var.shape}")
                if hasattr(layer, 'weight') and layer.weight is not None:
                    print(f"  BatchNorm weight shape: {layer.weight.shape}")
                if hasattr(layer, 'bias') and layer.bias is not None:
                    print(f"  BatchNorm bias shape: {layer.bias.shape}")
            
            # For flatten layer
            if hasattr(layer, 'start_dim') and hasattr(layer, 'end_dim'):
                print(f"  Flatten dims: {layer.start_dim} to {layer.end_dim}")
    
    # Switch to HE mode and see what operations Orion would do
    print(f"\n🔄 Switching to HE mode...")
    model.he()
    
    # Create encrypted input
    vec_ptxt = orion.encode(x, input_level)
    vec_ctxt = orion.encrypt(vec_ptxt)
    print(f"✅ Created encrypted input")
    
    # Show the inference operations
    print(f"\n🎯 Orion HE Inference Operations:")
    print(f"=" * 40)
    
    print(f"Input ciphertext level: {vec_ctxt.level()}")
    print(f"Input ciphertext slots: {vec_ctxt.slots()}")
    
    # Count the complexity
    total_diagonals = 0
    total_rotations = 0
    for name, layer in model.named_modules():
        if hasattr(layer, 'diagonals') and layer.diagonals:
            total_diagonals += len(layer.diagonals)
        if hasattr(layer, 'output_rotations'):
            total_rotations += layer.output_rotations
    
    print(f"Total diagonals across all layers: {total_diagonals}")
    print(f"Total rotations across all layers: {total_rotations}")
    
    # Extract operations from the compiled model
    print(f"\n🎯 Extracting Orion Operations:")
    print(f"=" * 40)
    
    from orion_heir.frontends.orion.orion_frontend import OrionFrontend
    frontend = OrionFrontend()
    
    operations = frontend.extract_operations(model)
    print(f"✅ Extracted {len(operations)} operations:")
    
    for i, op in enumerate(operations):
        print(f"  {i+1:2d}. {op.op_type:15} -> {op.result_var}")
        if op.metadata:
            desc = op.metadata.get('operation', '')
            layer = op.metadata.get('layer', '')
            if desc:
                print(f"      {'':15}    ↳ {desc} ({layer})")
    
    # Show operations by layer
    print(f"\n📊 Operations by Layer:")
    print(f"=" * 40)
    
    layer_ops = {}
    for op in operations:
        layer = op.metadata.get('layer', 'unknown')
        if layer not in layer_ops:
            layer_ops[layer] = []
        layer_ops[layer].append(op.op_type)
    
    for layer, ops in layer_ops.items():
        print(f"{layer}:")
        op_counts = {}
        for op in ops:
            op_counts[op] = op_counts.get(op, 0) + 1
        for op_type, count in op_counts.items():
            print(f"  - {op_type}: {count}")
    
    # Now translate to MLIR
    print(f"\n🔄 Translating to HEIR MLIR...")
    from orion_heir import GenericTranslator
    
    from orion_heir.frontends.orion.scheme_params import OrionSchemeParameters
    
    scheme_params = OrionSchemeParameters(
        logN=13,
        logQ=[29, 26, 26, 26, 26, 26],
        logP=[29, 29],
        logScale=26,
        slots=4096,
        ring_degree=8192,
        backend='lattigo',
        require_orion=True
    )
    translator = GenericTranslator()
    module = translator.translate(operations, scheme_params, "mlp")
    
    # Generate MLIR output
    from xdsl.printer import Printer
    from io import StringIO
    from xdsl.utils.exceptions import VerifyException
    
    output_buffer = StringIO()
    printer = Printer(stream=output_buffer)
    printer.print(module)
    mlir_output = output_buffer.getvalue()
    
    Path("mlp_fhe.mlir").write_text(mlir_output)
    print("💾 Saved to mlp_fhe.mlir")
    try:
        # Verify the individual operation
        module.verify()
    except VerifyException as e:
        print(f"❌ Failed verification: {e}")
        raise
    # Show MLIR statistics
    print(f"\n📄 MLIR Statistics:")
    print("=" * 50)
    lines = mlir_output.split('\n')
    
    # Count different operation types
    op_counts = {}
    for line in lines:
        line = line.strip()
        if ' = ' in line and ('ckks.' in line or 'lwe.' in line or 'arith.' in line):
            op_type = line.split(' = ')[1].split(' ')[0]
            op_counts[op_type] = op_counts.get(op_type, 0) + 1
    
    print(f"MLIR Operation Counts:")
    for op_type, count in sorted(op_counts.items()):
        print(f"  {op_type:20}: {count}")
    
    # Show complexity comparison
    print(f"\nComplexity Analysis:")
    print(f"  Total operations: {len(operations)}")
    print(f"  Linear layers: {len([op for op in operations if 'linear_transform' in op.op_type])}")
    print(f"  Activations: {len([op for op in operations if 'quad' in op.op_type or 'mul' in op.op_type])}")
    print(f"  Batch norms: {len([op for op in operations if 'batch_norm' in op.op_type or 'bn' in op.result_var])}")
    print(f"  Encoding ops: {len([op for op in operations if 'encode' in op.op_type])}")
    print(f"  Bias additions: {len([op for op in operations if 'add_plain' in op.op_type])}")
    
    print(f"\nTotal MLIR lines: {len(lines)}")
    fhe_ops = [l for l in lines if ' = ' in l and ('ckks.' in l or 'lwe.' in l)]
    print(f"FHE operations: {len(fhe_ops)}")
    print(f"Constants: {len([l for l in lines if 'arith.constant' in l])}")
    
    # Show expected computation levels
    print(f"\nExpected FHE Computation Levels:")
    print(f"  Input level: {input_level}")
    print(f"  After fc1+bn1: level {input_level-1}")
    print(f"  After act1 (quad): level {input_level-2}")
    print(f"  After fc2+bn2: level {input_level-3}")
    print(f"  After act2 (quad): level {input_level-4}")
    print(f"  After fc3: level {input_level-5}")
    
    # Show a sample of the MLIR (first few operations)
    print(f"\n📄 MLIR Sample (first 10 operations):")
    print("-" * 40)
    op_count = 0
    for line in lines:
        if line.strip() and (' = ckks.' in line or ' = lwe.' in line):
            print(f"  {line.strip()}")
            op_count += 1
            if op_count >= 10:
                break
    if len(operations) > 10:
        print(f"  ... ({len(operations) - 10} more operations)")
    
    print(f"\n✅ Translation complete!")
    print(f"   Operations: {len(operations)} total")
    print(f"   Layers: {len(layer_ops)} layers")
    print(f"   Complexity: {total_diagonals} diagonals, {total_rotations} rotations")
    
    return 0


if __name__ == "__main__":
    exit(main())

