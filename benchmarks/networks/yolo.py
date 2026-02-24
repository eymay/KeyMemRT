#!/usr/bin/env python3
"""
YOLOv1 Object Detection Architecture - Simple fix for dimension issue
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


# Complete ResNet components for YOLO backbone
class BasicBlock(on.Module):
    expansion = 1
    
    def __init__(self, in_chans, chans, stride=1):
        super().__init__()
        self.conv1 = on.Conv2d(in_chans, chans, 3, stride=stride, padding=1, bias=False)
        self.bn1 = on.BatchNorm2d(chans)
        self.act1 = on.SiLU(degree=127)
        
        self.conv2 = on.Conv2d(chans, chans, 3, stride=1, padding=1, bias=False)
        self.bn2 = on.BatchNorm2d(chans)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_chans != chans * self.expansion:
            self.shortcut = nn.Sequential(
                on.Conv2d(in_chans, chans * self.expansion, 1, stride=stride, bias=False),
                on.BatchNorm2d(chans * self.expansion)
            )
        
        self.add = on.Add()
        self.act2 = on.SiLU(degree=127)
    
    def forward(self, x):
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.add(out, self.shortcut(x))
        return self.act2(out)


class ResNet34(on.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.in_chans = 64
        
        self.conv1 = on.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = on.BatchNorm2d(64)
        self.act = on.SiLU(degree=127)
        self.pool = on.AvgPool2d(3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(BasicBlock, 64, 3, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 4, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 6, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 3, stride=2)
        
        self.avgpool = on.AdaptiveAvgPool2d((1, 1))
        self.flatten = on.Flatten()
        self.linear = on.Linear(512 * BasicBlock.expansion, num_classes)
    
    def _make_layer(self, block, chans, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_chans, chans, stride))
            self.in_chans = chans * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.pool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = self.flatten(out)
        return self.linear(out)


class YOLOv1(on.Module):
    def __init__(self, backbone=None, num_bboxes=2, num_classes=20, model_path=None):
        super().__init__()

        self.feature_size = 7
        self.num_bboxes = num_bboxes
        self.num_classes = num_classes
        self.model_path = model_path

        # Use provided backbone or create ResNet34
        if backbone is None:
            backbone = ResNet34()
        self.backbone = backbone
        
        self.conv_layers = self._make_conv_layers()
        
        # Remove last layers of backbone to use as feature extractor
        self.backbone.avgpool = nn.Identity()
        self.backbone.flatten = nn.Identity()
        self.backbone.linear = nn.Identity()
        
        # We'll set fc_layers after calculating size
        self.fc_layers = None
        
        self._init_weights()

    def _init_weights(self):
        if self.model_path:
            try:
                state_dict = torch.load(self.model_path, map_location='cpu', weights_only=False)
                self.load_state_dict(state_dict, strict=False)
                print(f"✅ Loaded weights from {self.model_path}")
            except Exception as e:
                print(f"⚠️ Could not load weights from {self.model_path}: {e}")

    def _make_conv_layers(self):
        net = nn.Sequential(
            on.Conv2d(512, 512, 3, padding=1),
            on.SiLU(degree=127),
            on.Conv2d(512, 512, 3, stride=2, padding=1),
            on.SiLU(degree=127),

            on.Conv2d(512, 512, 3, padding=1),
            on.SiLU(degree=127),
            on.Conv2d(512, 512, 3, padding=1),
            on.SiLU(degree=127)
        )
        return net

    def _make_fc_layers(self, fc_input_size):
        S, B, C = self.feature_size, self.num_bboxes, self.num_classes
        
        net = nn.Sequential(
            on.Flatten(),
            on.Linear(fc_input_size, 4096),
            on.SiLU(degree=127),
            on.Linear(4096, S * S * (5 * B + C)),  # 5*B + C = 5*2 + 20 = 30 for PASCAL VOC
        )
        return net
    
    def setup_fc_layers(self, input_tensor):
        """Calculate FC layer size and create the layers"""
        with torch.no_grad():
            # Pass through backbone
            x = self.backbone(input_tensor)
            
            # Pass through conv layers  
            x = self.conv_layers(x)
            
            # Get the flattened size
            fc_input_size = x.view(x.size(0), -1).size(1)
            print(f"📐 Calculated FC input size: {fc_input_size}")
            
            # Now create the FC layers with correct size
            self.fc_layers = self._make_fc_layers(fc_input_size)

    def forward(self, x):
        # If fc_layers not set up yet, set it up
        if self.fc_layers is None:
            self.setup_fc_layers(x)
        
        x = self.backbone(x)
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


def YOLOv1_ResNet34(model_path=None):
    """Create YOLOv1 with ResNet34 backbone."""
    backbone = ResNet34()
    net = YOLOv1(backbone, num_bboxes=2, num_classes=20, model_path=model_path)
    return net


def main():
    print("🚀 YOLOv1 Orion Test - Full Scale Object Detection")
    print("=" * 50)
    
    # Initialize Orion scheme with YOLO config (large parameters for full model)
    orion.init_scheme({
        'ckks_params': {
            'LogN': 16,  # Large for complex detection network
            'LogQ': [55, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40],
            'LogP': [61, 61, 61, 61, 61],
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
    
    # Create the full YOLO model
    model = YOLOv1_ResNet34()
    print(f"✅ Model: YOLOv1 with ResNet34 backbone:")
    print(f"   - Backbone: ResNet34 (3+4+6+3=16 BasicBlocks = 32 conv layers + initial = 34 total)")
    print(f"   - Conv layers: 4 additional Conv2d layers (512->512)")
    print(f"   - FC layers: Will be calculated dynamically")
    print(f"   - Output: {7}x{7} grid, {2} bboxes/cell, {20} classes")
    
    # Create input (YOLO uses 448x448 images, but start smaller for testing)
    input_sizes = [
        (1, 3, 224, 224),  # Start smaller for FHE
        # (1, 3, 448, 448),  # Full YOLO size - try if 224 works
    ]
    
    for input_shape in input_sizes:
        print(f"\n🧪 Testing with input size: {input_shape}")
        
        try:
            x = torch.randn(*input_shape)
            print(f"✅ Input shape: {x.shape}")
            
            # Test cleartext forward (this will set up FC layers)
            model.eval()
            with torch.no_grad():
                output = model(x)
            print(f"✅ Cleartext output shape: {output.shape}")
            
            # Fit and compile the model
            print(f"\n🔧 Orion fit and compile...")
            print("⚠️  Note: YOLO is a very large model - this may take a while...")
            
            orion.fit(model, x)
            input_level = orion.compile(model)
            print(f"✅ Compiled successfully, input level: {input_level}")
            
            # Extract operations
            print(f"\n🎯 Extracting Orion Operations:")
            print(f"=" * 40)
            
            frontend = OrionFrontend()
            operations = frontend.extract_operations(model)
            print(f"✅ Extracted {len(operations)} operations")
            
            # Show first few operations
            print(f"\nFirst 10 operations:")
            for i, op in enumerate(operations[:10]):
                print(f"  {i+1:2d}. {op.op_type:15} -> {op.result_var}")
                if hasattr(op, 'metadata') and op.metadata:
                    desc = op.metadata.get('operation', '')
                    layer = op.metadata.get('layer', '')
                    if desc:
                        print(f"      {'':15}    ↳ {desc} ({layer})")
            
            if len(operations) > 10:
                print(f"  ... and {len(operations) - 10} more operations")
            
            # Generate MLIR using working pattern
            print(f"\n🔧 Translating to HEIR MLIR...")
            from orion_heir import GenericTranslator
            from orion_heir.frontends.orion.scheme_params import OrionSchemeParameters
        
            scheme_params = OrionSchemeParameters(
                logN=16,
                logQ=[55, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40],
                logP=[61, 61, 61, 61, 61],
                logScale=40,
                slots=32768,
                ring_degree=65536,
                backend='lattigo',
                require_orion=True
            )
            translator = GenericTranslator()
            module = translator.translate(operations, scheme_params, "yolo_computation")
            
            # Generate MLIR output
            from xdsl.printer import Printer
            from io import StringIO
            
            output_buffer = StringIO()
            printer = Printer(stream=output_buffer)
            printer.print(module)
            mlir_output = output_buffer.getvalue()
            
            # Save MLIR to file
            output_file = Path(__file__).parent / "yolo.mlir"
            output_file.write_text(mlir_output)
            print(f"✅ MLIR generated and saved to: {output_file}")
            
            # Show statistics
            lines = mlir_output.split('\n')
            print(f"\n📊 MLIR Statistics:")
            print(f"  Total lines: {len(lines)}")
            print(f"  Total operations: {len(operations)}")
            
            # Count operation types
            op_counts = {}
            for op in operations:
                op_counts[op.op_type] = op_counts.get(op.op_type, 0) + 1
            
            print(f"\nOperation type breakdown:")
            for op_type, count in sorted(op_counts.items()):
                print(f"  {op_type:20}: {count}")
            
            print(f"\n🎉 SUCCESS! Full YOLO model compiled and translated to MLIR")
            print(f"   Input size: {input_shape}")
            print(f"   Operations: {len(operations)}")
            print(f"   MLIR lines: {len(lines)}")
            
            return 0
            
        except Exception as e:
            print(f"❌ Failed with input size {input_shape}: {e}")
            print(f"   Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n❌ All input sizes failed")
    print("💡 YOLO may still be too complex even with proper implementation")
    return 1


if __name__ == "__main__":
    exit(main())
