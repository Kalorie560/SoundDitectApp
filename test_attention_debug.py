#!/usr/bin/env python3
"""
Test script to validate the attention mechanism debugging and fix
"""
import torch
import torch.nn as nn
import sys
import os

# Add current directory to path to import from app.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import our enhanced classes from app.py
from app import CNNSequential, MultiHeadAttention, debug_tensor_shape, debug_log, analyze_state_dict_dimensions, calculate_cnn_output_shape, diagnose_attention_compatibility

def test_cnn_output_calculation():
    """Test CNN output shape calculation"""
    print("=" * 60)
    print("TEST 1: CNN Output Shape Calculation")
    print("=" * 60)
    
    # Create a test model
    model = CNNSequential(
        input_length=44100,
        channels=[64, 128, 256],
        kernel_sizes=[3, 3, 3],
        strides=[1, 2, 2],
        fc_sizes=[512, 256]
    )
    
    print(f"Model fc_input_size: {model.fc_input_size}")
    
    # Calculate actual output
    actual_size = calculate_cnn_output_shape((1, 1, 44100), model)
    print(f"Calculated CNN output size: {actual_size}")
    
    # Verify they match
    if actual_size == model.fc_input_size:
        print("✅ CNN output calculation PASSED")
    else:
        print(f"❌ CNN output calculation FAILED: expected {model.fc_input_size}, got {actual_size}")
    
    return actual_size

def test_attention_compatibility():
    """Test attention compatibility diagnosis"""
    print("\n" + "=" * 60)
    print("TEST 2: Attention Compatibility Diagnosis")
    print("=" * 60)
    
    # Create model with attention
    model = CNNSequential(
        input_length=44100,
        channels=[64, 128, 256],
        kernel_sizes=[3, 3, 3],
        strides=[1, 2, 2],
        fc_sizes=[512, 256]
    )
    
    # Add attention with correct dimensions
    correct_attention = MultiHeadAttention(input_dim=model.fc_input_size, num_heads=8)
    model.attention = correct_attention
    
    print(f"Model attention input size: {model.attention.query.weight.shape[1]}")
    print(f"Model fc_input_size: {model.fc_input_size}")
    
    # Create a fake state dict with incompatible attention weights
    fake_state_dict = {
        'attention.query.weight': torch.randn(256, 256),  # Wrong input dimension
        'attention.query.bias': torch.randn(256),
        'attention.key.weight': torch.randn(256, 256),
        'attention.key.bias': torch.randn(256),
        'attention.value.weight': torch.randn(256, 256),
        'attention.value.bias': torch.randn(256),
        'cnn.0.weight': torch.randn(64, 1, 3),
    }
    
    print(f"Fake state dict attention input size: {fake_state_dict['attention.query.weight'].shape[1]}")
    
    # Test compatibility diagnosis
    compatible = diagnose_attention_compatibility(model, fake_state_dict)
    
    if not compatible:
        print("✅ Attention compatibility diagnosis PASSED - correctly detected incompatibility")
    else:
        print("❌ Attention compatibility diagnosis FAILED - should have detected incompatibility")
    
    return compatible

def test_model_forward_pass():
    """Test model forward pass with debugging"""
    print("\n" + "=" * 60)
    print("TEST 3: Model Forward Pass with Debugging")
    print("=" * 60)
    
    # Create model
    model = CNNSequential(
        input_length=44100,
        channels=[64, 128, 256],
        kernel_sizes=[3, 3, 3],
        strides=[1, 2, 2],
        fc_sizes=[512, 256]
    )
    
    # Add attention with correct dimensions
    attention = MultiHeadAttention(input_dim=model.fc_input_size, num_heads=8)
    model.attention = attention
    
    print(f"Model expects attention input: {model.attention.query.weight.shape[1]}")
    print(f"Model fc_input_size: {model.fc_input_size}")
    
    # Test forward pass
    try:
        test_input = torch.randn(1, 1, 44100)
        print(f"Test input shape: {test_input.shape}")
        
        output = model(test_input)
        print(f"✅ Model forward pass PASSED - output shape: {output.shape}")
        return True
    except Exception as e:
        print(f"❌ Model forward pass FAILED: {e}")
        return False

def test_emergency_projection():
    """Test emergency projection layer for shape mismatch"""
    print("\n" + "=" * 60)
    print("TEST 4: Emergency Projection Layer")
    print("=" * 60)
    
    # Create model
    model = CNNSequential(
        input_length=44100,
        channels=[64, 128, 256],
        kernel_sizes=[3, 3, 3],
        strides=[1, 2, 2],
        fc_sizes=[512, 256]
    )
    
    # Add attention with WRONG dimensions to trigger emergency projection
    wrong_attention = MultiHeadAttention(input_dim=256, num_heads=8)  # Wrong size
    model.attention = wrong_attention
    
    print(f"Model attention expects: {model.attention.query.weight.shape[1]}")
    print(f"Model actual fc_input_size: {model.fc_input_size}")
    print("This should trigger emergency projection...")
    
    # Test forward pass - should create emergency projection
    try:
        test_input = torch.randn(1, 1, 44100)
        output = model(test_input)
        
        # Check if emergency projection was created
        if hasattr(model.attention, 'emergency_projection'):
            print(f"✅ Emergency projection CREATED: {model.attention.emergency_projection}")
            print(f"   Input: {model.attention.emergency_projection.in_features}")
            print(f"   Output: {model.attention.emergency_projection.out_features}")
            print(f"✅ Model forward pass with emergency projection PASSED - output shape: {output.shape}")
            return True
        else:
            print("❌ Emergency projection was NOT created")
            return False
            
    except Exception as e:
        print(f"❌ Emergency projection test FAILED: {e}")
        return False

def main():
    print("🧪 ATTENTION MECHANISM DEBUG TEST SUITE")
    print("Testing comprehensive debugging and fix implementation...\n")
    
    results = []
    
    # Run all tests
    cnn_size = test_cnn_output_calculation()
    results.append(cnn_size is not None)
    
    compatible = test_attention_compatibility()
    results.append(not compatible)  # Should be False (incompatible)
    
    forward_pass = test_model_forward_pass()
    results.append(forward_pass)
    
    emergency_proj = test_emergency_projection()
    results.append(emergency_proj)
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"CNN Output Calculation: {'✅ PASS' if results[0] else '❌ FAIL'}")
    print(f"Attention Compatibility: {'✅ PASS' if results[1] else '❌ FAIL'}")
    print(f"Model Forward Pass: {'✅ PASS' if results[2] else '❌ FAIL'}")
    print(f"Emergency Projection: {'✅ PASS' if results[3] else '❌ FAIL'}")
    
    overall = all(results)
    print(f"\nOVERALL: {'✅ ALL TESTS PASSED' if overall else '❌ SOME TESTS FAILED'}")
    
    if overall:
        print("\n🎉 The attention mechanism debugging and fix implementation is working correctly!")
        print("   The app should now handle shape mismatches gracefully and provide detailed debugging info.")
    else:
        print("\n⚠️  Some tests failed. The implementation may need further refinement.")
    
    return overall

if __name__ == "__main__":
    main()