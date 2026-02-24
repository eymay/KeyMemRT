#!/usr/bin/env python3
"""
VGG CNN Architecture - Fixed to use the working model pattern
Supports VGG11, VGG13, VGG16, VGG19 variants.
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn

# Add the src directory to the path  
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import Orion
import orion
import orion.nn as on
from orion_heir.frontends.orion.orion_frontend import OrionFrontend


# VGG configurations
cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(on.Module):
    def __init__(self, vgg_name='VGG16', num_classes=10):
        super().__init__()
        self.vgg_name = vgg_name
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = on.Linear(512, num_classes)
        self.flatten = on.Flatten()

    def forward(self, x):
        out = self.features(x)
        out = self.flatten(out)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [on.AvgPool2d(kernel_size=2, stride=2)]
            else:
                layers += [on.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           on.BatchNorm2d(x),
                           on.ReLU(degrees=[15,15,27])]
                in_channels = x
        layers += [on.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def VGG11(num_classes=10):
    return VGG('VGG11', num_classes)

def VGG13(num_classes=10):
    return VGG('VGG13', num_classes)

def VGG16(num_classes=10):
    return VGG('VGG16', num_classes)

def VGG19(num_classes=10):
    return VGG('VGG19', num_classes)


def main():
    print("🚀 VGG Orion Test - Extracting FHE Operations")
    print("=" * 50)
    
    # Initialize Orion scheme with VGG config (needs larger parameters)
    orion.init_scheme({
        'ckks_params': {
            'LogN': 16,  # Larger for complex CNN
            'LogQ': [55, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40],
            'LogP': [61, 61, 61, 61],
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
    
    # Create the model (using VGG16 as default, but can be changed)
    model = VGG11()
    print(f"✅ Model: VGG11 CNN with layers:")
    
    # Count layers for display
    conv_count = 0
    pool_count = 0
    for name, layer in model.named_modules():
        if isinstance(layer, on.Conv2d):
            conv_count += 1
        elif isinstance(layer, on.AvgPool2d):
            pool_count += 1
    
    print(f"   - Features: {conv_count} Conv2d layers + {pool_count} AvgPool2d layers")
    print(f"   - Classifier: Linear(512, 10)")
    print(f"   - Configuration: {cfg['VGG16']}")
    
    # Create input (CIFAR-10 format - 32x32 RGB images)
    x = torch.randn(1, 3, 32, 32)  # Batch, Channel, Height, Width
    print(f"✅ Input shape: {x.shape} (CIFAR-10 format: batch, channel, height, width)")
    
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
    
    layer_count = 0
    for name, layer in model.named_modules():
        if hasattr(layer, 'diagonals') or hasattr(layer, 'transform_ids') or hasattr(layer, 'level'):
            layer_count += 1
            
            # Only show first 10 layers to avoid clutter
            if layer_count <= 10:
                print(f"\nLayer {layer_count}: {name}")
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
    
    if layer_count > 10:
        print(f"\n... and {layer_count - 10} more layers")
    
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
    
    # Show first 15 operations (VGG has many operations)
    for i, op in enumerate(operations[:15]):
        print(f"  {i+1:2d}. {op.op_type:15} -> {op.result_var}")
        if hasattr(op, 'metadata') and op.metadata:
            desc = op.metadata.get('operation', '')
            layer = op.metadata.get('layer', '')
            if desc:
                print(f"      {'':15}    ↳ {desc} ({layer})")
    
    if len(operations) > 15:
        print(f"  ... and {len(operations) - 15} more operations")
    
    # Generate MLIR using working pattern (GenericTranslator)
    print(f"\n🔧 Translating to HEIR MLIR...")
    try:
        from orion_heir import GenericTranslator
        from orion_heir.frontends.orion.scheme_params import OrionSchemeParameters
    
        scheme_params = OrionSchemeParameters(
            logN=16,
            logQ=[55, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40],
            logP=[61, 61, 61, 61],
            logScale=40,
            slots=32768,
            ring_degree=65536,
            backend='lattigo',
            require_orion=True
        )
        translator = GenericTranslator()
        module = translator.translate(operations, scheme_params, "vgg_computation")
        
        # Generate MLIR output
        from xdsl.printer import Printer
        from io import StringIO
        
        output_buffer = StringIO()
        printer = Printer(stream=output_buffer)
        printer.print(module)
        mlir_output = output_buffer.getvalue()
        
        # Save MLIR to file
        output_file = Path(__file__).parent / "vgg11.mlir"
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
        print(f"  Activations: {len([op for op in operations if any(act in op.op_type for act in ['relu', 'quad', 'mul'])])}")
        print(f"  Encoding ops: {len([op for op in operations if 'encode' in op.op_type])}")
        print(f"  Bias additions: {len([op for op in operations if 'add_plain' in op.op_type])}")
        
        # Show expected computation levels (VGG can be deep)
        expected_levels = min(10, input_level)  # Show up to 10 levels
        print(f"\nExpected FHE Computation Levels (first {expected_levels}):")
        print(f"  Input level: {input_level}")
        for i in range(1, expected_levels + 1):
            print(f"  After layer {i}: level {input_level - i}")
        if input_level > expected_levels:
            print(f"  ... (total depth may reach level {max(0, input_level - len(operations)//2)})")
        
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
        print(f"   VGG variant: {model.vgg_name}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error during MLIR translation: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
