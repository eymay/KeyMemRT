#!/usr/bin/env python3
"""
LeNet-5 CNN Architecture - Fixed to use the working model pattern
"""

import sys
from pathlib import Path
import torch

# Add the src directory to the path  
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import Orion
import orion
import orion.nn as on
from orion_heir.frontends.orion.orion_frontend import OrionFrontend


class LeNet(on.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = on.Conv2d(1, 32, kernel_size=5, padding=2, stride=2)
        self.bn1 = on.BatchNorm2d(32)
        self.act1 = on.Quad()
        
        self.conv2 = on.Conv2d(32, 64, kernel_size=5, padding=2, stride=2) 
        self.bn2 = on.BatchNorm2d(64)
        self.act2 = on.Quad()    
        
        self.flatten = on.Flatten()
        self.fc1 = on.Linear(7*7*64, 512)
        self.bn3 = on.BatchNorm1d(512)
        self.act3 = on.Quad() 
        
        self.fc2 = on.Linear(512, num_classes)

    def forward(self, x): 
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        x = self.flatten(x)
        x = self.act3(self.bn3(self.fc1(x)))
        return self.fc2(x)


def main():
    print("🚀 LeNet Orion Test - Extracting FHE Operations")
    print("=" * 50)
    
    # Initialize Orion scheme with LeNet config
    orion.init_scheme({
        'ckks_params': {
            'LogN': 14,  # Slightly larger for CNN
            'LogQ': [55, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40],
            'LogP': [61, 61, 61],
            'LogScale': 40,
            'H': 192
        },
        'orion': {
            'backend': 'lattigo',
            'margin': 2,
            'embedding_method': 'hybrid',
            'fuse_modules': True,
            'debug': True
        }
    })
    print("✅ Orion scheme initialized")
    
    # Create the model
    model = LeNet()
    print(f"✅ Model: LeNet-5 CNN with layers:")
    print(f"   - conv1: Conv2d(1, 32, kernel_size=5, padding=2, stride=2)")
    print(f"   - bn1: BatchNorm2d(32)")
    print(f"   - act1: Quad()")
    print(f"   - conv2: Conv2d(32, 64, kernel_size=5, padding=2, stride=2)")
    print(f"   - bn2: BatchNorm2d(64)")
    print(f"   - act2: Quad()")
    print(f"   - flatten: Flatten()")
    print(f"   - fc1: Linear(7*7*64=3136, 512)")
    print(f"   - bn3: BatchNorm1d(512)")
    print(f"   - act3: Quad()")
    print(f"   - fc2: Linear(512, 10)")
    
    # Create input (MNIST format - 28x28 grayscale images)
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
                
            if hasattr(layer, 'transform_ids'):
                print(f"  Transform IDs: {len(layer.transform_ids) if layer.transform_ids else 0}")
    
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
    
    frontend = OrionFrontend()
    
    operations = frontend.extract_operations(model)
    print(f"✅ Extracted {len(operations)} operations:")
    
    # Print operations using working pattern (objects, not dictionaries)
    for i, op in enumerate(operations):
        print(f"  {i+1:2d}. {op.op_type:15} -> {op.result_var}")
        if hasattr(op, 'metadata') and op.metadata:
            desc = op.metadata.get('operation', '')
            layer = op.metadata.get('layer', '')
            if desc:
                print(f"      {'':15}    ↳ {desc} ({layer})")
    
    # Generate MLIR using working pattern (GenericTranslator)
    print(f"\n🔧 Translating to HEIR MLIR...")
    try:
        from orion_heir import GenericTranslator
        from orion_heir.frontends.orion.scheme_params import OrionSchemeParameters
        
        scheme_params = OrionSchemeParameters(
            logN=14,
            logQ=[55, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40],
            logP=[61, 61, 61],
            logScale=40,
            slots=8192,
            ring_degree=16384,
            backend='lattigo',
            require_orion=True
        )
        translator = GenericTranslator()
        module = translator.translate(operations, scheme_params, "lenet")
        
        # Generate MLIR output
        from xdsl.printer import Printer
        from io import StringIO
        
        output_buffer = StringIO()
        printer = Printer(stream=output_buffer)
        printer.print(module)
        mlir_output = output_buffer.getvalue()
        
        # Save MLIR to file
        output_file = Path(__file__).parent / "lenet.mlir"
        output_file.write_text(mlir_output)
        print(f"✅ MLIR generated and saved to: {output_file}")
        
        # Show MLIR statistics
        lines = mlir_output.split('\n')
        
        # Count different operation types
        op_counts = {}
        for line in lines:
            line = line.strip()
            if ' = ' in line and ('ckks.' in line or 'lwe.' in line or 'arith.' in line):
                op_type = line.split(' = ')[1].split(' ')[0]
                op_counts[op_type] = op_counts.get(op_type, 0) + 1
        
        print(f"\n📊 MLIR Statistics:")
        print(f"  Total lines: {len(lines)}")
        print(f"  FHE operations: {len([l for l in lines if ' = ' in l and ('ckks.' in l or 'lwe.' in l)])}")
        print(f"  Constants: {len([l for l in lines if 'arith.constant' in l])}")
        
        print(f"\nMLIR Operation Counts:")
        for op_type, count in sorted(op_counts.items()):
            print(f"  {op_type:20}: {count}")
        
        # Show complexity comparison
        print(f"\nComplexity Analysis:")
        print(f"  Total operations: {len(operations)}")
        print(f"  Linear layers: {len([op for op in operations if 'linear_transform' in op.op_type])}")
        print(f"  Activations: {len([op for op in operations if 'quad' in op.op_type or 'mul' in op.op_type])}")
        print(f"  Encoding ops: {len([op for op in operations if 'encode' in op.op_type])}")
        print(f"  Bias additions: {len([op for op in operations if 'add_plain' in op.op_type])}")
        
        # Show expected computation levels
        print(f"\nExpected FHE Computation Levels:")
        print(f"  Input level: {input_level}")
        print(f"  After conv1+bn1: level {input_level-1}")
        print(f"  After act1 (quad): level {input_level-2}")
        print(f"  After conv2+bn2: level {input_level-3}")
        print(f"  After act2 (quad): level {input_level-4}")
        print(f"  After fc1+bn3: level {input_level-5}")
        print(f"  After act3 (quad): level {input_level-6}")
        print(f"  After fc2: level {input_level-7}")
        
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
        print(f"   Total MLIR lines: {len(lines)}")
        print(f"   Complexity: {total_diagonals} diagonals, {total_rotations} rotations")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error during MLIR translation: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
