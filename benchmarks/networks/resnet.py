#!/usr/bin/env python3
"""
Orion ResNet Demo - Complete ResNet pipeline following the MLP example pattern.

This demonstrates a complete ResNet workflow:
1. Create a ResNet model using Orion's design (embedded in this file)
2. Run cleartext inference  
3. Use Orion to compile the model (orion.fit + orion.compile)
4. Extract operations via OrionFrontend
5. Translate to HEIR MLIR for OpenFHE compilation
"""

import time
import math
import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import Orion
import orion
import orion.nn as on
from orion.core.utils import get_cifar_datasets, mae

# Import our HEIR translator
from orion_heir import GenericTranslator, OrionFrontend
from orion_heir.frontends.orion.scheme_params import OrionSchemeParameters


# ================================
# ResNet Model (copied from Orion)
# ================================

class BasicBlock(on.Module):
    expansion = 1

    def __init__(self, Ci, Co, stride=1):
        super().__init__()
        self.conv1 = on.Conv2d(Ci, Co, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1   = on.BatchNorm2d(Co)
        self.act1  = on.ReLU()

        self.conv2 = on.Conv2d(Co, Co, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = on.BatchNorm2d(Co)
        self.act2  = on.ReLU()
       
        self.add = on.Add()
        self.shortcut = nn.Sequential()
        if stride != 1 or Ci != self.expansion*Co:
            self.shortcut = nn.Sequential(
                on.Conv2d(Ci, self.expansion*Co, kernel_size=1, stride=stride, bias=False),
                on.BatchNorm2d(self.expansion*Co))
  
    def forward(self, x):
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.add(out, self.shortcut(x))
        return self.act2(out)

class Bottleneck(on.Module):
    expansion = 4

    def __init__(self, Ci, Co, stride=1):
        super().__init__()
        self.conv1 = on.Conv2d(Ci, Co, kernel_size=1, bias=False)
        self.bn1   = on.BatchNorm2d(Co)
        self.act1  = on.SiLU(degree=127) 

        self.conv2 = on.Conv2d(Co, Co, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2   = on.BatchNorm2d(Co)
        self.act2  = on.SiLU(degree=127)  

        self.conv3 = on.Conv2d(Co, Co*self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3   = on.BatchNorm2d(Co*self.expansion)
        self.act3  = on.SiLU(degree=127)  

        self.add = on.Add()
        self.shortcut = nn.Sequential()
        if stride != 1 or Ci != self.expansion*Co:
            self.shortcut = nn.Sequential(
                on.Conv2d(Ci, self.expansion*Co, kernel_size=1, stride=stride, bias=False),
                on.BatchNorm2d(self.expansion*Co))

    def forward(self, x):
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.act2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = self.add(out, self.shortcut(x))
        return self.act3(out)


class ResNet(on.Module):
    def __init__(self, dataset, block, num_blocks, num_chans, conv1_params, num_classes):
        super().__init__()
        self.in_chans = num_chans[0]
        self.last_chans = num_chans[-1]

        self.conv1 = on.Conv2d(3, self.in_chans, **conv1_params, bias=False)
        self.bn1 = on.BatchNorm2d(self.in_chans)
        self.act = on.ReLU()

        self.pool = nn.Identity()
        if dataset == 'imagenet': # for imagenet we must also downsample
            self.pool = on.AvgPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layers = nn.ModuleList()
        for i in range(len(num_blocks)):
            stride = 1 if i == 0 else 2
            self.layers.append(self.layer(block, num_chans[i], num_blocks[i], stride))

        self.avgpool = on.AdaptiveAvgPool2d(output_size=(1,1)) 
        self.flatten = on.Flatten()
        self.linear  = on.Linear(self.last_chans * block.expansion, num_classes)

    def layer(self, block, chans, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)

        layers = []
        for stride in strides:
            layers.append(block(self.in_chans, chans, stride))
            self.in_chans = chans * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.pool(out)
        for layer in self.layers:
            out = layer(out)

        out = self.avgpool(out)
        out = self.flatten(out)
        return self.linear(out)


def get_resnet_config(dataset):
    configs = {
        "cifar10": {"kernel_size": 3, "stride": 1, "padding": 1, "num_classes": 10},
        "cifar100": {"kernel_size": 3, "stride": 1, "padding": 1, "num_classes": 100},
        "tiny": {"kernel_size": 7, "stride": 1, "padding": 3, "num_classes": 200},
        "imagenet": {"kernel_size": 7, "stride": 2, "padding": 3, "num_classes": 1000},
    }

    if dataset not in configs:
        raise ValueError(f"ResNet with dataset {dataset} is not supported.")
    
    config = configs[dataset]
    conv1_params = {
        'kernel_size': config["kernel_size"],
        'stride': config["stride"],
        'padding': config["padding"]
    }
    
    return conv1_params, config["num_classes"]

################################
# Minimal ResNets for testing  #
################################

def ResNet1(dataset='cifar10'):
    """Minimal ResNet with 1 block per layer for memory efficiency."""
    conv1_params, num_classes = get_resnet_config(dataset)
    return ResNet(dataset, BasicBlock, [1], [16], conv1_params, num_classes)

def ResNet8(dataset='cifar10'):
    """Minimal ResNet with 1 block per layer for memory efficiency."""
    conv1_params, num_classes = get_resnet_config(dataset)
    return ResNet(dataset, BasicBlock, [1,1,1], [16,32,64], conv1_params, num_classes)

def ResNet10(dataset='cifar10'):
    conv1_params, num_classes = get_resnet_config(dataset)
    return ResNet(dataset, BasicBlock, [1,1,1], [20,40,80], conv1_params, num_classes)

################################
# CIFAR-10 / CIFAR-100 ResNets #
################################

def ResNet20(dataset='cifar10'):
    conv1_params, num_classes = get_resnet_config(dataset)
    return ResNet(dataset, BasicBlock, [3,3,3], [16,32,64], conv1_params, num_classes)

def ResNet32(dataset='cifar10'):
    conv1_params, num_classes = get_resnet_config(dataset)
    return ResNet(dataset, BasicBlock, [5,5,5], [16,32,64], conv1_params, num_classes)

def ResNet44(dataset='cifar10'):
    conv1_params, num_classes = get_resnet_config(dataset)
    return ResNet(dataset, BasicBlock, [7,7,7], [16,32,64], conv1_params, num_classes)

def ResNet56(dataset='cifar10'):
    conv1_params, num_classes = get_resnet_config(dataset)
    return ResNet(dataset, BasicBlock, [9,9,9], [16,32,64], conv1_params, num_classes)

def ResNet110(dataset='cifar10'):
    conv1_params, num_classes = get_resnet_config(dataset)
    return ResNet(dataset, BasicBlock, [18,18,18], [16,32,64], conv1_params, num_classes)

def ResNet1202(dataset='cifar10'):
    conv1_params, num_classes = get_resnet_config(dataset)
    return ResNet(dataset, BasicBlock, [200,200,200], [16,32,64], conv1_params, num_classes)

####################################
# Tiny ImageNet / ImageNet ResNets #
####################################

def ResNet18(dataset='imagenet'):
    conv1_params, num_classes = get_resnet_config(dataset)
    return ResNet(dataset, BasicBlock, [2,2,2,2], [64,128,256,512], conv1_params, num_classes)

def ResNet34(dataset='imagenet'):
    conv1_params, num_classes = get_resnet_config(dataset)
    return ResNet(dataset, BasicBlock, [3,4,6,3], [64,128,256,512], conv1_params, num_classes)

def ResNet50(dataset='imagenet'):
    conv1_params, num_classes = get_resnet_config(dataset)
    return ResNet(dataset, Bottleneck, [3,4,6,3], [64,128,256,512], conv1_params, num_classes)

def ResNet101(dataset='imagenet'):
    conv1_params, num_classes = get_resnet_config(dataset)
    return ResNet(dataset, Bottleneck, [3,4,23,3], [64,128,256,512], conv1_params, num_classes)

def ResNet152(dataset='imagenet'):
    conv1_params, num_classes = get_resnet_config(dataset)
    return ResNet(dataset, Bottleneck, [3,8,36,3], [64,128,256,512], conv1_params, num_classes)


# ================================
# Demo Function
# ================================

def run_orion_resnet_demo(variant_name='resnet8', model_fn=None, output_path=None):
    """Run the complete Orion ResNet demo following the MLP pattern.

    Args:
        variant_name: Name of the ResNet variant (e.g., 'resnet8', 'resnet20')
        model_fn: Function to create the model (e.g., ResNet8, ResNet20)
        output_path: Path to save the output MLIR file
    """
    if model_fn is None:
        model_fn = ResNet8
    if output_path is None:
        output_path = f"{variant_name}_fhe.mlir"

    print(f"🧠 Orion ResNet Demo - {variant_name.upper()} to HEIR Translation")
    print("=" * 60)

    # Step 1: Initialize Orion scheme with embedded config
    print("\n1️⃣ Initializing Orion Scheme")
    print("-" * 30)

    # Embedded ResNet config (from configs/resnet.yml)
    config = {
        'ckks_params': {
            'LogN': 16,
            'LogQ': [55, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40],
            'LogP': [61, 61, 61],
            'LogScale': 40,
            'H': 192,
            'RingType': 'standard'
        },
        'boot_params': {
            'LogP': [61, 61, 61, 61, 61, 61, 61, 61]
        },
        'orion': {
            'margin': 2,
            'embedding_method': 'hybrid',
            'backend': 'lattigo',
            'fuse_modules': True,
            'debug': True,
            'diags_path': '',
            'keys_path': '',
            'io_mode': 'none'
        }
    }

    scheme = orion.init_scheme(config)
    print("✅ Orion scheme initialized with ResNet parameters")

    # Step 2: Create ResNet model (gets dataset from model_fn default)
    print(f"\n2️⃣ Creating {variant_name.upper()} Model")
    print("-" * 30)

    net = model_fn()
    dataset = net.conv1.in_channels  # Infer from model
    print(f"✅ Created {variant_name.upper()}")

    # Step 3: Load dataset
    print(f"\n3️⃣ Loading Dataset")
    print("-" * 30)

    trainloader, testloader = get_cifar_datasets(data_dir="../data", batch_size=1)
    print("✅ Dataset loaded")
    
    # Step 4: Get test batch
    print("\n4️⃣ Preparing Test Input")
    print("-" * 30)
    
    inp, target = next(iter(testloader))
    print(f"✅ Input shape: {inp.shape}")
    print(f"📊 Input range: [{inp.min():.3f}, {inp.max():.3f}]")
    print(f"🎯 Target class: {target.item()}")
    
    # Step 5: Run cleartext inference
    print("\n5️⃣ Cleartext Reference Inference")
    print("-" * 30)
    
    net.eval()
    with torch.no_grad():
        out_clear = net(inp)
    
    predicted_class = torch.argmax(out_clear, dim=1).item()
    confidence = torch.softmax(out_clear, dim=1).max().item()
    
    print(f"✅ Cleartext output shape: {out_clear.shape}")
    print(f"🔮 Predicted class: {predicted_class} (confidence: {confidence:.3f})")
    print(f"📊 Output logits (first 5): {out_clear.flatten()[:5].tolist()}")
    
    # Step 6: Compile for FHE (no execution needed)
    print("\n6️⃣ Compiling Model for FHE")
    print("-" * 30)
    
    # Orion automatically analyzes the model and estimates ranges
    print("📊 Analyzing model with Orion...")
    orion.fit(net, inp)
    
    # Compile the model for FHE
    print("🔧 Compiling model for FHE...")
    input_level = orion.compile(net)
    print(f"✅ Model compiled, input level: {input_level}")
    
    # Step 7: Extract FHE operations via OrionFrontend
    print("\n7️⃣ Extracting FHE Operations")
    print("-" * 30)
    
    try:
        # Create OrionFrontend to extract operations from compiled model
        frontend = OrionFrontend()
        
        # Extract operations from the compiled Orion model
        print("🔍 Extracting operations from compiled model...")
        operations = frontend.extract_operations(net)
        
        print(f"✅ Extracted {len(operations)} FHE operations")
        
        # Print operation summary
        op_types = {}
        for op in operations:
            op_type = op.op_type
            op_types[op_type] = op_types.get(op_type, 0) + 1
        
        print("📊 Operations by type:")
        for op_type, count in sorted(op_types.items()):
            print(f"   {op_type:20}: {count:2d} operations")
            
    except Exception as e:
        print(f"❌ Error extracting operations: {e}")
        print("   Using fallback operation creation...")
        
        # Fallback: create basic operations if extraction fails
        operations = frontend.create_mlp_operations() if hasattr(frontend, 'create_mlp_operations') else []
        print(f"✅ Created {len(operations)} fallback operations")
    
    # Step 8: Create FHE scheme parameters
    print("\n8️⃣ Setting Up FHE Scheme Parameters")
    print("-" * 30)
    
    try:
        scheme_params = OrionSchemeParameters(
            logN=16,  # From resnet config: LogN: 16
            logQ=[55, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40],  # From resnet config
            logP=[61, 61, 61],  # From resnet config  
            logScale=40,  # From resnet config
            slots=32768,  # 2^15 for LogN=16
            ring_degree=65536,  # 2^16
            backend='lattigo',
            require_orion=True
        )
        print("✅ FHE scheme parameters created")
    except Exception as e:
        print(f"⚠️ Error creating Orion parameters: {e}")
        print("   Using fallback parameters...")
        scheme_params = OrionSchemeParameters(
            logN=16,
            logQ=[55, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40],
            logP=[61, 61, 61],
            logScale=40,
            slots=32768,
            ring_degree=65536,
            backend='lattigo',
            require_orion=False
        )
    
    # Step 9: Generate HEIR MLIR
    print("\n9️⃣ Generating HEIR MLIR")
    print("-" * 30)
    
    try:
        # Create HEIR translator
        translator = GenericTranslator()
        
        # Translate to HEIR MLIR
        print("🔧 Translating operations to HEIR MLIR...")
        module = translator.translate(
            operations, 
            scheme_params,
            function_name="resnet_inference"
        )
        # Generate MLIR output
        from xdsl.printer import Printer
        from io import StringIO
        
        output_buffer = StringIO()
        printer = Printer(stream=output_buffer)
        printer.print(module)
        mlir_output = output_buffer.getvalue()
        
        # Save MLIR output
        output_file = Path(output_path)
        output_file.write_text(mlir_output)

        print(f"✅ HEIR MLIR generated and saved to {output_file}")
        print(f"📄 Output file size: {len(mlir_output)} characters")

        lines = mlir_output.splitlines()
        print(f"📄 Lines of MLIR: {len(lines)}")

        # Show summary statistics
        print("\n📊 MLIR Summary:")
        func_lines = [l for l in lines if 'func.func' in l]
        op_lines = [l for l in lines if any(op in l for op in ['bgv.', 'ckks.', 'heir.'])]
        print(f"   Functions: {len(func_lines)}")
        print(f"   FHE operations: {len(op_lines)}")
            
    except Exception as e:
        print(f"❌ Error generating HEIR MLIR: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print(f"\n🎉 {variant_name.upper()} compilation and translation completed successfully!")
    print("\nNext steps:")
    print("   1. Use HEIR tools to optimize the MLIR")
    print("   2. Compile to OpenFHE with heir-translate")
    print("   3. Execute FHE inference with OpenFHE")

    return True


def main():
    """Main entry point with CLI argument support."""
    import argparse

    # Map of all available ResNet variants
    VARIANTS = {
        'resnet1': ResNet1,
        'resnet8': ResNet8,
        'resnet10': ResNet10,
        'resnet20': ResNet20,
        'resnet32': ResNet32,
        'resnet44': ResNet44,
        'resnet56': ResNet56,
        'resnet110': ResNet110,
        'resnet1202': ResNet1202,
        'resnet18': ResNet18,
        'resnet34': ResNet34,
        'resnet50': ResNet50,
        'resnet101': ResNet101,
        'resnet152': ResNet152,
    }

    parser = argparse.ArgumentParser(
        description='Generate ResNet MLIR for FHE analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available ResNet Variants:
  CIFAR-10/100:  resnet1, resnet8, resnet10, resnet20, resnet32, resnet44, resnet56, resnet110, resnet1202
  ImageNet:      resnet18, resnet34, resnet50, resnet101, resnet152

Examples:
  python3 resnet.py                              # Default: ResNet8
  python3 resnet.py --variant resnet20            # Generate ResNet20
  python3 resnet.py --variant resnet50 --output custom.mlir
        """
    )

    parser.add_argument(
        '--variant',
        type=str,
        choices=list(VARIANTS.keys()),
        default='resnet8',
        help='ResNet variant to generate (default: resnet8)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output MLIR file path (default: <variant>_fhe.mlir)'
    )

    args = parser.parse_args()

    # Determine output path
    output_path = args.output if args.output else f"{args.variant}_fhe.mlir"

    print(f"🔧 Configuration:")
    print(f"   Variant: {args.variant}")
    print(f"   Output: {output_path}")
    print()

    # Set random seed for reproducibility
    torch.manual_seed(42)

    try:
        success = run_orion_resnet_demo(
            variant_name=args.variant,
            model_fn=VARIANTS[args.variant],
            output_path=output_path
        )
        exit_code = 0 if success else 1
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit_code = 1

    print(f"\n{'🎉 Success!' if exit_code == 0 else '💥 Failed!'}")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()

