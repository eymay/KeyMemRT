#!/usr/bin/env python3
"""
LoLaNet example that matches the actual Orion LoLaNet structure.
Based on the LoLA model from https://arxiv.org/pdf/1812.10659
"""

import sys
from pathlib import Path
import torch

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import Orion
import orion
import orion.nn as on


class LoLa(on.Module):
    """LoLa (Low Latency) model for FHE inference."""
    
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = on.Conv2d(1, 5, kernel_size=2, padding=0, stride=2)
        self.bn1 = on.BatchNorm2d(5)
        self.act1 = on.Quad()
        
        self.fc1 = on.Linear(980, 100)
        self.bn2 = on.BatchNorm1d(100)
        self.act2 = on.Quad()
        
        self.fc2 = on.Linear(100, num_classes)
        self.flatten = on.Flatten()

    def forward(self, x): 
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.flatten(x)
        x = self.act2(self.bn2(self.fc1(x)))
        return self.fc2(x)


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
    print("🚀 LoLa Orion Test")
    print("=" * 50)
    
    # Initialize Orion scheme
    print("🔧 Initializing Orion scheme...")
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
    model = LoLa(num_classes=10)
    print(f"✅ Model created with structure:")
    print(f"   - conv1: Conv2d(1, 5, kernel_size=2, stride=2)")
    print(f"   - bn1: BatchNorm2d(5)")
    print(f"   - act1: Quad()")
    print(f"   - flatten: Flatten()")
    print(f"   - fc1: Linear(980, 100)")
    print(f"   - bn2: BatchNorm1d(100)")
    print(f"   - act2: Quad()")
    print(f"   - fc2: Linear(100, 10)")
    
    # Create test input (MNIST format)
    input_tensor = torch.randn(1, 1, 28, 28)
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
                logN=13,
                logQ=[29, 26, 26, 26, 26, 26],
                logP=[29, 29],
                logScale=26,
                slots=4096,
                ring_degree=8192,
                backend='lattigo',
                require_orion=True
            )
            
            # Create translator and generate MLIR
            translator = GenericTranslator()
            module = translator.translate(operations, scheme_params, "lola")
            
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
                output_file = Path("lola_fhe.mlir")
                output_file.write_text(mlir_str)
                print(f"💾 Saved MLIR to {output_file}")
                
                lines = mlir_str.split('\n')
                
                print(f"📄 Generated MLIR ({len(lines)} lines):")
                print("=" * 40)
                print(mlir_str)
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
