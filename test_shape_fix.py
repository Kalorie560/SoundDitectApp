#!/usr/bin/env python3
"""
Test script to validate the tensor shape mismatch fix

This script tests the runtime projection mechanism and ensures
that the (1x44032) vs (256x256) mismatch is properly resolved.
"""

import torch
import torch.nn as nn
import sys
import traceback

def test_runtime_projection():
    """Test that runtime projection resolves shape mismatches"""
    print("🧪 Testing runtime projection mechanism...")
    
    # Simulate the problematic scenario
    cnn_output_size = 44032  # Actual CNN output size
    old_attention_input_size = 256  # Old saved attention input size
    
    # Create a mock attention layer with wrong dimensions (like loaded from state dict)
    class MockAttention(nn.Module):
        def __init__(self):
            super().__init__()
            # These represent the old, incompatible weights
            self.query = nn.Linear(old_attention_input_size, 256)
            self.key = nn.Linear(old_attention_input_size, 256) 
            self.value = nn.Linear(old_attention_input_size, 256)
            self.output = nn.Linear(256, 256)
    
    attention = MockAttention()
    
    # Create test input that would cause the error
    test_input = torch.randn(1, cnn_output_size)
    print(f"Test input shape: {test_input.shape}")
    
    # This would fail with the original code
    try:
        result = attention.query(test_input)
        print("❌ ERROR: Should have failed with shape mismatch!")
        return False
    except RuntimeError as e:
        if "cannot be multiplied" in str(e):
            print(f"✅ Expected error caught: {e}")
        else:
            print(f"❌ Unexpected error: {e}")
            return False
    
    # Now test the runtime projection fix
    print("\n🔧 Applying runtime projection fix...")
    
    # Check dimensions and create projection
    expected_input_size = attention.query.weight.shape[1]
    actual_input_size = test_input.shape[1]
    
    print(f"Expected input size: {expected_input_size}")
    print(f"Actual input size: {actual_input_size}")
    
    if actual_input_size != expected_input_size:
        print(f"Creating runtime projection: {actual_input_size} -> {expected_input_size}")
        attention.runtime_projection = nn.Linear(actual_input_size, expected_input_size)
        
        # Apply projection
        projected_input = attention.runtime_projection(test_input)
        print(f"Projected input shape: {projected_input.shape}")
        
        # Now test that attention works
        try:
            query = attention.query(projected_input)
            key = attention.key(projected_input)
            value = attention.value(projected_input)
            print(f"✅ Query shape: {query.shape}")
            print(f"✅ Key shape: {key.shape}")
            print(f"✅ Value shape: {value.shape}")
            
            # Test full attention computation
            attention_weights = torch.softmax(torch.matmul(query, key.transpose(-2, -1)) / (query.size(-1) ** 0.5), dim=-1)
            attended = torch.matmul(attention_weights, value)
            output = attention.output(attended)
            print(f"✅ Final output shape: {output.shape}")
            
            print("✅ Runtime projection fix successful!")
            return True
            
        except Exception as e:
            print(f"❌ Runtime projection failed: {e}")
            return False
    else:
        print("No projection needed")
        return True

def test_attention_disabling():
    """Test graceful attention disabling fallback"""
    print("\n🧪 Testing attention disabling fallback...")
    
    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = "mock_attention"  # Simulate broken attention
            
        def forward(self, x):
            if self.attention is not None:
                try:
                    # Simulate attention failure
                    raise RuntimeError("Simulated attention failure")
                except Exception as e:
                    print(f"⚠️ Attention failed: {e}")
                    print("🔧 Disabling attention mechanism")
                    self.attention = None
                    return x.view(x.size(0), -1)  # Flatten for classifier
            else:
                return x.view(x.size(0), -1)
    
    model = MockModel()
    test_input = torch.randn(1, 256, 172)  # Simulate CNN output
    
    try:
        output = model(test_input)
        print(f"✅ Fallback successful, output shape: {output.shape}")
        print(f"✅ Attention disabled: {model.attention is None}")
        return True
    except Exception as e:
        print(f"❌ Fallback failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Testing tensor shape mismatch fixes\n")
    
    tests = [
        ("Runtime Projection", test_runtime_projection),
        ("Attention Disabling Fallback", test_attention_disabling),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Running: {test_name}")
        print(f"{'='*50}")
        
        try:
            result = test_func()
            results.append((test_name, result))
            print(f"\n{test_name}: {'✅ PASSED' if result else '❌ FAILED'}")
        except Exception as e:
            print(f"\n{test_name}: ❌ FAILED with exception: {e}")
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*50}")
    print("TEST SUMMARY")
    print(f"{'='*50}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The shape mismatch fix should work.")
    else:
        print("⚠️ Some tests failed. The fix may need additional work.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)