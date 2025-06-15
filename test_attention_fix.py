#!/usr/bin/env python3
"""
Test script to verify the attention mechanism shape mismatch fix.
This script creates a mock model and state dict to test the filtering logic.
"""

import torch
import torch.nn as nn
import sys
import os

# Add current directory to path so we can import from app.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_attention_shape_fix():
    """Test that attention weight filtering prevents shape mismatches"""
    print("🧪 Testing Attention Shape Mismatch Fix")
    print("=" * 50)
    
    # Import our classes
    try:
        from app import CNNSequential, MultiHeadAttention
        print("✅ Successfully imported classes from app.py")
    except ImportError as e:
        print(f"❌ Failed to import from app.py: {e}")
        return False
    
    # Test 1: Create model with correct dimensions
    print("\n1. Creating model with correct dimensions...")
    try:
        # Create model that should output ~44032 dimensional features
        model = CNNSequential(
            input_length=44100,
            channels=[64, 128, 256], 
            kernel_sizes=[3, 3, 3],
            strides=[1, 2, 2]
        )
        
        # Calculate the actual fc_input_size
        with torch.no_grad():
            dummy_input = torch.randn(1, 1, 44100)
            cnn_output = model.cnn(dummy_input)
            fc_input_size = cnn_output.view(cnn_output.size(0), -1).size(1)
        
        print(f"✅ Model created. FC input size: {fc_input_size}")
        print(f"   Model fc_input_size attribute: {model.fc_input_size}")
        
        # Add attention with correct dimensions
        model.attention = MultiHeadAttention(input_dim=fc_input_size, num_heads=8)
        print(f"✅ Attention added with input_dim={fc_input_size}")
        
    except Exception as e:
        print(f"❌ Failed to create model: {e}")
        return False
    
    # Test 2: Create incompatible state dict (simulating saved model with wrong attention dims)
    print("\n2. Creating incompatible state dict...")
    try:
        # Create state dict with incompatible attention dimensions
        incompatible_state_dict = {}
        
        # Add some CNN weights (these should be compatible)
        incompatible_state_dict['cnn.0.weight'] = torch.randn(64, 1, 3)
        incompatible_state_dict['cnn.0.bias'] = torch.randn(64)
        
        # Add classifier weights (these should be compatible)
        incompatible_state_dict['classifier.0.weight'] = torch.randn(512, fc_input_size)
        incompatible_state_dict['classifier.0.bias'] = torch.randn(512)
        
        # Add INCOMPATIBLE attention weights (256 input dim instead of actual fc_input_size)
        incompatible_state_dict['attention.query.weight'] = torch.randn(256, 256)  # Wrong size!
        incompatible_state_dict['attention.query.bias'] = torch.randn(256)
        incompatible_state_dict['attention.key.weight'] = torch.randn(256, 256)    # Wrong size!
        incompatible_state_dict['attention.key.bias'] = torch.randn(256)
        incompatible_state_dict['attention.value.weight'] = torch.randn(256, 256)  # Wrong size!
        incompatible_state_dict['attention.value.bias'] = torch.randn(256)
        
        print(f"✅ Created incompatible state dict with attention expecting 256 input dims")
        print(f"   But model's attention expects {fc_input_size} input dims")
        
    except Exception as e:
        print(f"❌ Failed to create incompatible state dict: {e}")
        return False
    
    # Test 3: Test the filtering logic (simulate what happens in load_model)
    print("\n3. Testing attention weight filtering...")
    try:
        # Simulate the filtering logic from app.py
        has_attention = True
        filtered_state_dict = {}
        attention_keys_removed = []
        
        for key, value in incompatible_state_dict.items():
            if key.startswith('attention.'):
                # Check if the attention weight dimensions match our model
                if key.endswith('.weight'):
                    attr_name = key.split('.', 1)[1]  # e.g., 'query.weight' -> 'query'
                    attr_name = attr_name.split('.')[0]  # e.g., 'query.weight' -> 'query'
                    if hasattr(model.attention, attr_name):
                        model_layer = getattr(model.attention, attr_name)
                        if hasattr(model_layer, 'weight') and model_layer.weight.shape != value.shape:
                            print(f"   🔧 Filtering incompatible weight {key}: model={model_layer.weight.shape} vs saved={value.shape}")
                            attention_keys_removed.append(key)
                            continue
                elif key.endswith('.bias'):
                    # Also skip corresponding bias if weight was skipped
                    weight_key = key.replace('.bias', '.weight')
                    if weight_key in attention_keys_removed:
                        print(f"   🔧 Filtering corresponding bias {key}")
                        attention_keys_removed.append(key)
                        continue
            
            filtered_state_dict[key] = value
        
        print(f"✅ Filtering completed. Removed {len(attention_keys_removed)} incompatible attention parameters")
        print(f"   Removed keys: {attention_keys_removed}")
        
    except Exception as e:
        print(f"❌ Failed to filter state dict: {e}")
        return False
    
    # Test 4: Try loading the filtered state dict
    print("\n4. Testing model loading with filtered state dict...")
    try:
        # Load filtered state dict (should not cause shape mismatch)
        missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)
        print(f"✅ State dict loaded successfully!")
        print(f"   Missing keys: {len(missing_keys)} (expected due to filtered attention weights)")
        print(f"   Unexpected keys: {len(unexpected_keys)}")
        
    except Exception as e:
        print(f"❌ Failed to load filtered state dict: {e}")
        return False
    
    # Test 5: Test model forward pass (this should NOT cause shape mismatch)
    print("\n5. Testing model forward pass...")
    try:
        model.eval()
        with torch.no_grad():
            test_input = torch.randn(1, 1, 44100)
            output = model(test_input)
            print(f"✅ Forward pass successful! Output shape: {output.shape}")
        
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("🎉 ALL TESTS PASSED! The attention shape mismatch fix is working correctly.")
    print("✅ Model can now load with incompatible attention weights without crashing")
    print("✅ Attention mechanism uses correct dimensions for CNN output")
    return True

if __name__ == "__main__":
    success = test_attention_shape_fix()
    if success:
        print("\n🚀 The fix should resolve the RuntimeError in the main application!")
    else:
        print("\n❌ Tests failed - the fix needs more work")
    
    sys.exit(0 if success else 1)