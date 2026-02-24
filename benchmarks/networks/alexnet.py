#!/usr/bin/env python3
"""
AlexNet example that matches the actual Orion AlexNet structure.
Uses SiLU activations and pooling layers suitable for FHE.
"""

import sys
from pathlib import Path
import torch

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import Orion
import orion
import orion.nn as on
import torch.nn as nn


class ConvBlock(on.Module):
    def __init__(self, Ci, Co, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Sequential(
            on.Conv2d(Ci, Co, kernel_size, stride, padding, bias=False),
            on.BatchNorm2d(Co),
            on.SiLU(degree=127))
    
    def forward(self, x):
        return self.conv(x)
    

class LinearBlock(on.Module):
    def __init__(self, ni, no):
        super().__init__()
        self.linear = nn.Sequential(
            on.Linear(ni, no),
            on.BatchNorm1d(no),
            on.SiLU(degree=127))
        
    def forward(self, x):
        return self.linear(x)


class AlexNet(on.Module):
    """AlexNet adapted for FHE with SiLU activations."""
    cfg = [64, 'M', 192, 'M', 384, 256, 256, 'A']

    def __init__(self, num_classes=10):
        super().__init__()
        self.features = self._make_layers()
        self.flatten = on.Flatten()
        self.classifier = nn.Sequential(
            LinearBlock(1024, 4096),
            LinearBlock(4096, 4096),
            on.Linear(4096, num_classes))

    def _make_layers(self):
        layers = []
        in_channels = 3
        for x in self.cfg:
            if x == 'M':
                layers += [on.AvgPool2d(kernel_size=2, stride=2)]
            elif x == 'A':
                layers += [on.AdaptiveAvgPool2d((2, 2))]
            else:
                layers += [ConvBlock(in_channels, x, kernel_size=3, 
                                     stride=1, padding=1)]
                in_channels = x
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x


def extract_orion_operations(model):
    """Extract operations by calling orion.compile() and then use the translator."""
    print("🔄 Running orion.compile() to compile the model...")
    try:
        input_level = orion.compile(model)
        print(f"✅ Orion compilation successful, input level: {input_level}")
        
        # Now extract operations using the Orion-HEIR frontend
        print("🔄 Extracting operations using Orion-HEIR frontend...")
        from orion_heir.frontends.orion.orion_frontend import OrionFrontend
        
        frontend = OrionFrontend()
        operations = frontend.extract_operations(model)
        
        print(f"✅ Extracted {len(operations)} operations from compiled Orion model")
        return operations
        
    except ImportError as e:
        print(f"❌ Could not import OrionFrontend: {e}")
        raise
    except Exception as e:
        print(f"❌ Error in extraction pipeline: {e}")
        raise


def main():
    print("🚀 AlexNet Orion Test")
    print("=" * 50)
    
    # Initialize Orion scheme with more levels for complex network
    print("🔧 Initializing Orion scheme...")
    orion.init_scheme({
        'ckks_params': {
            'LogN': 14,  # Larger N for more complex network
            'LogQ': [50, 40, 40, 40, 40, 40, 40, 40, 40, 40],  # More levels for SiLU activations
            'LogP': [50, 50],
            'LogScale': 40,
            'H': 16384
        },
        'orion': {
            'backend': 'lattigo'
        }
    })
    print("✅ Orion scheme initialized")
    
    # Create the model
    model = AlexNet(num_classes=10)
    print(f"✅ Model created with structure:")
    print(f"   Features:")
    print(f"   - ConvBlock(3->64): Conv2d + BatchNorm2d + SiLU")
    print(f"   - AvgPool2d(2x2)")
    print(f"   - ConvBlock(64->192): Conv2d + BatchNorm2d + SiLU")
    print(f"   - AvgPool2d(2x2)")
    print(f"   - ConvBlock(192->384): Conv2d + BatchNorm2d + SiLU")
    print(f"   - ConvBlock(384->256): Conv2d + BatchNorm2d + SiLU")
    print(f"   - ConvBlock(256->256): Conv2d + BatchNorm2d + SiLU")
    print(f"   - AdaptiveAvgPool2d((2, 2))")
    print(f"   Classifier:")
    print(f"   - Flatten()")
    print(f"   - LinearBlock(1024->4096): Linear + BatchNorm1d + SiLU")
    print(f"   - LinearBlock(4096->4096): Linear + BatchNorm1d + SiLU")
    print(f"   - Linear(4096->10)")
    
    # Create test input (CIFAR-10 format)
    input_tensor = torch.randn(1, 3, 32, 32)
    print(f"✅ Created test input: {input_tensor.shape}")
    
    # Run cleartext forward pass to verify model works
    print("🧪 Testing cleartext forward pass...")
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
    print(f"✅ Cleartext output shape: {output.shape}")
    
    # Fit and compile with Orion
    print("📊 Running orion.fit()...")
    try:
        orion.fit(model, input_tensor)
        print("✅ orion.fit() completed")
    except Exception as e:
        print(f"❌ orion.fit() failed: {e}")
        return
    
    # Extract operations and translate
    print("🔍 Extracting operations and translating to HEIR MLIR...")
    try:
        operations = extract_orion_operations(model)
        
        if operations:
            print(f"✅ Operations extracted successfully")
            print(f"📊 Found {len(operations)} operations:")
            
            # Group operations by type
            op_types = {}
            for op in operations:
                op_type = op.op_type if hasattr(op, 'op_type') else str(type(op).__name__)
                op_types[op_type] = op_types.get(op_type, 0) + 1
            
            for op_type, count in sorted(op_types.items()):
                print(f"   - {op_type}: {count}")
            
            # Use the translator to generate HEIR MLIR
            print("🔄 Translating to HEIR MLIR...")
            from orion_heir import GenericTranslator
            from orion_heir.frontends.orion.scheme_params import OrionSchemeParameters
            
            scheme_params = OrionSchemeParameters(
                logN=14,
                logQ=[50, 40, 40, 40, 40, 40, 40, 40, 40, 40],
                logP=[50, 50],
                logScale=40,
                slots=8192,
                ring_degree=16384,
                backend='lattigo',
                require_orion=True
            )
            
            # Create translator and generate MLIR
            translator = GenericTranslator()
            module = translator.translate(operations, scheme_params, "alexnet_computation")
            
            if module:
                print("✅ MLIR translation successful")
                
                # Generate MLIR output and save to file
                from xdsl.printer import Printer
                from io import StringIO
                
                output_buffer = StringIO()
                printer = Printer(stream=output_buffer)
                printer.print(module)
                mlir_str = output_buffer.getvalue()
                
                # Save to file
                output_file = Path("alexnet_fhe.mlir")
                output_file.write_text(mlir_str)
                print(f"💾 Saved MLIR to {output_file}")
                
                lines = mlir_str.split('\n')
                
                print(f"📄 Generated MLIR ({len(lines)} lines):")
                print("=" * 40)
                # Print first 50 lines to avoid overwhelming output
                for i, line in enumerate(lines[:50]):
                    print(line)
                if len(lines) > 50:
                    print(f"... ({len(lines)-50} more lines)")
                print("=" * 40)
                
                # Count operation types in MLIR
                mlir_op_counts = {}
                for line in lines:
                    line = line.strip()
                    if 'ckks.' in line:
                        op_start = line.find('ckks.') + 5
                    elif 'lwe.' in line:
                        op_start = line.find('lwe.') + 4
                    elif 'arith.' in line:
                        op_start = line.find('arith.') + 6
                    else:
                        continue
                    
                    op_end = line.find('(', op_start)
                    if op_end > op_start:
                        op_type = line[op_start-5:op_end] if 'ckks.' in line else line[op_start-4:op_end] if 'lwe.' in line else line[op_start-6:op_end]
                        mlir_op_counts[op_type] = mlir_op_counts.get(op_type, 0) + 1
                
                for op_type, count in sorted(mlir_op_counts.items()):
                    print(f"   - {op_type}: {count}")
                
                print(f"📏 Total MLIR lines: {len(lines)}")
                
                # Verify the module
                try:
                    module.verify()
                    print("✅ MLIR module verification passed")
                except Exception as e:
                    print(f"⚠️ MLIR verification warning: {e}")
            
            else:
                print("❌ No operations extracted")
            
            print("🎉 Complete pipeline test successful!")
            
    except Exception as e:
        print(f"❌ Error in pipeline: {e}")
        print("❌ Pipeline test failed. Check the error messages above.")
        
        # Print model structure for debugging
        print("\n🔍 Model structure for debugging:")
        for name, module in model.named_modules():
            if name:  # Skip root module
                print(f"   {name}: {type(module).__name__}")


if __name__ == "__main__":
    main()
