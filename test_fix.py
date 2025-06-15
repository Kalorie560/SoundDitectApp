#!/usr/bin/env python3
"""
Test script to validate the attention mechanism fix
"""
import torch
import torch.nn as nn

# Simplified test versions of the classes
class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, num_heads=8):
        super(MultiHeadAttention, self).__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        
        # Make sure input_dim is divisible by num_heads for multi-head attention
        if input_dim % num_heads != 0:
            adjusted_dim = ((input_dim // num_heads) + 1) * num_heads
            self.hidden_dim = adjusted_dim
            # Add a projection layer to map input_dim to hidden_dim
            self.input_projection = nn.Linear(input_dim, adjusted_dim)
        else:
            self.hidden_dim = input_dim
            self.input_projection = None
            
        self.head_dim = self.hidden_dim // num_heads
        
        self.query = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.key = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.value = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.output = nn.Linear(self.hidden_dim, self.hidden_dim)

class TestCNNSequential(nn.Module):
    def __init__(self, input_length=44100):
        super(TestCNNSequential, self).__init__()
        
        # Simplified CNN layers to match the structure
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 64, 3, 1, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4, 4),
            nn.Conv1d(64, 128, 3, 2, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(4, 4),
            nn.Conv1d(128, 256, 3, 2, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(4, 4)
        )
        
        # Calculate fc_input_size
        self._calculate_fc_input_size(input_length)
        
        # Add attention
        self.attention = MultiHeadAttention(input_dim=self.fc_input_size, num_heads=8)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.fc_input_size, 512),
            nn.Dropout(0.3),
            nn.ReLU(), 
            nn.Linear(512, 256),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(256, 2)
        )
        
    def _calculate_fc_input_size(self, input_length):
        # Calculate based on architecture
        x = input_length
        # Conv1 + MaxPool1
        x = (x + 2 * 1 - 3) // 1 + 1  # Conv
        x = x // 4  # MaxPool
        # Conv2 + MaxPool2  
        x = (x + 2 * 1 - 3) // 2 + 1  # Conv
        x = x // 4  # MaxPool
        # Conv3 + MaxPool3
        x = (x + 2 * 1 - 3) // 2 + 1  # Conv
        x = x // 4  # MaxPool
        self.fc_input_size = x * 256
        print(f"Calculated fc_input_size: {self.fc_input_size}")
        
    def forward(self, x):
        x = self.cnn(x)
        
        # Apply attention 
        if self.attention is not None:
            batch_size, channels, length = x.size()
            x_flat = x.view(batch_size, channels * length)
            print(f"x_flat shape before attention: {x_flat.shape}")
            print(f"attention.query weight shape: {self.attention.query.weight.shape}")
            
            # Apply input projection if needed
            if hasattr(self.attention, 'input_projection') and self.attention.input_projection is not None:
                x_flat = self.attention.input_projection(x_flat)
                print(f"x_flat shape after projection: {x_flat.shape}")
            
            # Attention weights
            query = self.attention.query(x_flat)
            key = self.attention.key(x_flat)
            value = self.attention.value(x_flat)
            
            # Scaled dot-product attention
            attention_weights = torch.softmax(torch.matmul(query, key.transpose(-2, -1)) / (query.size(-1) ** 0.5), dim=-1)
            x_attended = torch.matmul(attention_weights, value)
            x = self.attention.output(x_attended)
        else:
            # Flatten for classifier
            x = x.view(x.size(0), -1)
            
        return self.classifier(x)

def test_model():
    print("Testing attention mechanism fix...")
    
    # Create model
    model = TestCNNSequential(input_length=44100)
    model.eval()
    
    # Test with dummy input
    test_input = torch.randn(1, 1, 44100)
    print(f"Test input shape: {test_input.shape}")
    
    try:
        with torch.no_grad():
            output = model(test_input)
        print(f"✅ SUCCESS! Output shape: {output.shape}")
        return True
    except Exception as e:
        print(f"❌ FAILED! Error: {e}")
        return False

if __name__ == "__main__":
    success = test_model()
    if success:
        print("\n🎉 The fix works! The attention mechanism should now work correctly.")
    else:
        print("\n💥 The fix needs more work.")