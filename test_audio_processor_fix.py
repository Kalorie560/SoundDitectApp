#!/usr/bin/env python3
"""
Test script to validate AudioProcessor creation fixes
"""

import torch
import torch.nn as nn
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_model_creation():
    """Test creating a simple model for AudioProcessor testing"""
    print("🧪 Testing model creation...")
    
    class SimpleTestModel(nn.Module):
        def __init__(self):
            super(SimpleTestModel, self).__init__()
            self.conv1 = nn.Conv1d(1, 32, kernel_size=128, stride=4)
            self.bn1 = nn.BatchNorm1d(32)
            self.relu1 = nn.ReLU()
            self.pool1 = nn.MaxPool1d(4)
            
            self.conv2 = nn.Conv1d(32, 64, kernel_size=64, stride=2)
            self.bn2 = nn.BatchNorm1d(64)
            self.relu2 = nn.ReLU()
            self.pool2 = nn.MaxPool1d(4)
            
            # Calculate output size for 44100 input
            # After conv1 + pool1: ((44100 - 128) // 4 + 1) // 4 = 2757
            # After conv2 + pool2: ((2757 - 64) // 2 + 1) // 4 = 336
            self.fc_input_size = 336 * 64  # 21504
            
            self.classifier = nn.Sequential(
                nn.Linear(self.fc_input_size, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, 2)
            )
        
        def forward(self, x):
            x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
            x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x
    
    model = SimpleTestModel()
    model.eval()
    
    # Test with dummy input
    test_input = torch.randn(1, 1, 44100)
    with torch.no_grad():
        output = model(test_input)
    
    print(f"✅ Model created successfully: input {test_input.shape} -> output {output.shape}")
    return model

def test_audio_processor_import():
    """Test importing AudioProcessor from app.py"""
    print("🧪 Testing AudioProcessor import...")
    
    try:
        from app import AudioProcessor
        print("✅ AudioProcessor imported successfully")
        return AudioProcessor
    except Exception as e:
        print(f"❌ Failed to import AudioProcessor: {e}")
        return None

def test_audio_processor_creation(model, AudioProcessor):
    """Test creating AudioProcessor with the test model"""
    print("🧪 Testing AudioProcessor creation...")
    
    try:
        processor = AudioProcessor(model, target_sr=22050, chunk_length=44100)
        print("✅ AudioProcessor created successfully")
        
        # Test basic functionality
        predictions, chunks = processor.get_results()
        print(f"✅ AudioProcessor functionality test: {len(predictions)} predictions, {len(chunks)} chunks")
        
        return processor
    except Exception as e:
        print(f"❌ AudioProcessor creation failed: {e}")
        import traceback
        print(f"Full error: {traceback.format_exc()}")
        return None

def main():
    """Main test function"""
    print("🚀 Starting AudioProcessor fix validation tests\n")
    
    # Test 1: Model creation
    model = test_model_creation()
    if model is None:
        print("❌ Cannot proceed - model creation failed")
        return False
    
    print()
    
    # Test 2: AudioProcessor import
    AudioProcessor = test_audio_processor_import()
    if AudioProcessor is None:
        print("❌ Cannot proceed - AudioProcessor import failed")
        return False
    
    print()
    
    # Test 3: AudioProcessor creation
    processor = test_audio_processor_creation(model, AudioProcessor)
    if processor is None:
        print("❌ AudioProcessor creation test failed")
        return False
    
    print()
    print("🎉 All tests passed! AudioProcessor fixes are working correctly.")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)