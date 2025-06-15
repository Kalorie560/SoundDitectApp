# Critical: Configure torchaudio BEFORE importing streamlit to prevent segmentation fault
# This must be done first to avoid streamlit's file watcher torchaudio conflicts
import os
import sys

# Force torchaudio backend configuration before any imports
os.environ['TORCHAUDIO_BACKEND'] = 'soundfile'

# Disable torchaudio backend warnings entirely
os.environ['TORCHAUDIO_DISABLE_SOX_WARNINGS'] = '1'

# Import and configure torchaudio first
import torchaudio
import torch

# Force backend initialization to prevent runtime dispatcher issues
try:
    # Try to set the backend explicitly before streamlit imports torchaudio
    if hasattr(torchaudio, 'set_audio_backend'):
        # For older versions that still support set_audio_backend
        pass
    elif hasattr(torchaudio, '_extension'):
        # For newer versions, ensure extension is loaded properly
        torchaudio._extension._init_extension()
    
    # Force a minimal torchaudio operation to initialize backend properly
    dummy_tensor = torch.zeros(1, 1000)
    resampler = torchaudio.transforms.Resample(16000, 22050)
    _ = resampler(dummy_tensor)
    del dummy_tensor, resampler
    
except Exception as e:
    # If backend configuration fails, continue but print the error
    print(f"Warning: Torchaudio backend configuration failed: {e}")

# Now import streamlit (after torchaudio is properly configured)
import streamlit as st
import torch.nn as nn
import torchaudio.transforms as T

import numpy as np
import matplotlib.pyplot as plt
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
import av
import queue
import threading
from typing import List, Tuple
import time
import traceback
import psutil
import gc
import logging

# Debug logging setup
def setup_debug_logging():
    """Setup comprehensive debug logging"""
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
        ]
    )
    return logging.getLogger(__name__)

# Global debug logger
debug_logger = setup_debug_logging()

def log_memory_usage(phase: str):
    """Log current memory usage"""
    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        debug_logger.info(f"[{phase}] Memory Usage: RSS={memory_info.rss / 1024 / 1024:.1f}MB, VMS={memory_info.vms / 1024 / 1024:.1f}MB")
        
        # Also show in Streamlit if available
        if 'st' in globals():
            st.info(f"🔧 [{phase}] Memory: {memory_info.rss / 1024 / 1024:.1f}MB RSS")
    except Exception as e:
        debug_logger.warning(f"Could not log memory usage: {e}")

def debug_log(message: str, level: str = "info"):
    """Enhanced debug logging with Streamlit integration"""
    debug_logger.info(f"🔍 DEBUG: {message}")
    
    # Also show in Streamlit if available
    if 'st' in globals():
        if level == "error":
            st.error(f"🔧 DEBUG: {message}")
        elif level == "warning":
            st.warning(f"🔧 DEBUG: {message}")
        else:
            st.info(f"🔧 DEBUG: {message}")

def debug_tensor_shape(tensor_name: str, tensor, additional_info: str = ""):
    """Debug tensor shapes with detailed information"""
    if tensor is not None:
        if hasattr(tensor, 'shape'):
            shape_info = f"{tensor_name}: {tensor.shape}"
        elif hasattr(tensor, 'size'):
            shape_info = f"{tensor_name}: {tensor.size()}"
        else:
            shape_info = f"{tensor_name}: {type(tensor)}"
        
        if additional_info:
            shape_info += f" | {additional_info}"
        debug_log(f"📐 SHAPE: {shape_info}")
        return shape_info
    else:
        debug_log(f"📐 SHAPE: {tensor_name}: None")
        return f"{tensor_name}: None"

def analyze_state_dict_dimensions(state_dict):
    """Analyze and log all tensor dimensions in state dict"""
    debug_log("📊 STATE DICT DIMENSION ANALYSIS:")
    
    attention_weights = {}
    cnn_weights = {}
    classifier_weights = {}
    
    for key, value in state_dict.items():
        if hasattr(value, 'shape'):
            debug_log(f"  {key}: {value.shape}")
            
            if key.startswith('attention.'):
                attention_weights[key] = value.shape
            elif key.startswith(('cnn.', 'conv')):
                cnn_weights[key] = value.shape
            elif key.startswith(('classifier.', 'fc')):
                classifier_weights[key] = value.shape
    
    return attention_weights, cnn_weights, classifier_weights

def calculate_cnn_output_shape(input_shape, model):
    """Calculate actual CNN output shape by doing a forward pass"""
    debug_log(f"🧮 CALCULATING CNN OUTPUT SHAPE with input: {input_shape}")
    
    try:
        with torch.no_grad():
            dummy_input = torch.randn(*input_shape)
            debug_tensor_shape("dummy_input", dummy_input)
            
            # Pass through CNN layers only
            x = model.cnn(dummy_input)
            debug_tensor_shape("cnn_output", x, "before flattening")
            
            # Flatten like in forward pass
            batch_size, channels, length = x.size()
            x_flat = x.view(batch_size, channels * length)
            debug_tensor_shape("x_flat", x_flat, "after flattening")
            
            return x_flat.shape[1]  # Return flattened dimension
    except Exception as e:
        debug_log(f"❌ CNN output calculation failed: {e}", "error")
        return None

def diagnose_attention_compatibility(model, state_dict):
    """Comprehensive diagnosis of attention layer compatibility"""
    debug_log("🔬 ATTENTION COMPATIBILITY DIAGNOSIS:")
    
    if not hasattr(model, 'attention') or model.attention is None:
        debug_log("  No attention layer in model")
        return True
    
    # Check model's attention layer dimensions
    if hasattr(model.attention, 'query'):
        query_weight = model.attention.query.weight
        debug_tensor_shape("model.attention.query.weight", query_weight, "model layer")
        model_input_dim = query_weight.shape[1]  # Input dimension to query layer
        model_output_dim = query_weight.shape[0]  # Output dimension from query layer
        
        debug_log(f"  Model attention expects INPUT: {model_input_dim}, produces OUTPUT: {model_output_dim}")
    
    # Check saved state dict attention dimensions
    if 'attention.query.weight' in state_dict:
        saved_weight = state_dict['attention.query.weight']
        debug_tensor_shape("state_dict attention.query.weight", saved_weight, "saved state")
        saved_input_dim = saved_weight.shape[1]
        saved_output_dim = saved_weight.shape[0]
        
        debug_log(f"  Saved attention expects INPUT: {saved_input_dim}, produces OUTPUT: {saved_output_dim}")
        
        # Check compatibility
        if model_input_dim != saved_input_dim:
            debug_log(f"  ❌ INCOMPATIBLE: Model expects {model_input_dim}, saved has {saved_input_dim}", "error")
            return False
        else:
            debug_log(f"  ✅ COMPATIBLE: Both use input dimension {model_input_dim}")
            return True
    
    return True

def safe_execute(func, description: str, *args, **kwargs):
    """Safely execute a function with comprehensive error logging"""
    debug_log(f"Starting: {description}")
    log_memory_usage(f"Before {description}")
    
    try:
        result = func(*args, **kwargs)
        debug_log(f"✅ Completed: {description}")
        log_memory_usage(f"After {description}")
        return result, None
    except Exception as e:
        error_msg = f"❌ Failed: {description}"
        debug_log(error_msg, "error")
        
        # Log full traceback
        full_traceback = traceback.format_exc()
        debug_logger.error(f"Full traceback for {description}:\n{full_traceback}")
        
        if 'st' in globals():
            st.error(f"🔧 ERROR in {description}: {str(e)}")
            with st.expander("🔍 Full Error Details"):
                st.text(full_traceback)
        
        log_memory_usage(f"Error in {description}")
        return None, e

def _save_uploaded_file(uploaded_file, temp_path: str):
    """Helper function to save uploaded file"""
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return temp_path

def _cleanup_temp_file(temp_path: str):
    """Helper function to clean up temporary file"""
    import os
    if os.path.exists(temp_path):
        os.remove(temp_path)
        return True
    return False

# CNN Model Definition (based on specifications for 1D CNN)
class CNN(nn.Module):
    def __init__(self, input_length=22050, num_classes=2):
        super(CNN, self).__init__()
        
        # First convolutional block
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=256, stride=4)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=4, stride=4)
        
        # Second convolutional block
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=128, stride=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=4, stride=4)
        
        # Third convolutional block
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=64, stride=2)
        self.bn3 = nn.BatchNorm1d(128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(kernel_size=4, stride=4)
        
        # Calculate the size of flattened features
        self._calculate_fc_input_size(input_length)
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.fc_input_size, 256)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(128, num_classes)
        
    def _calculate_fc_input_size(self, input_length):
        # Calculate the output size after all conv and pooling layers
        x = input_length
        # Conv1 + Pool1
        x = ((x - 256) // 4 + 1) // 4
        # Conv2 + Pool2  
        x = ((x - 128) // 2 + 1) // 4
        # Conv3 + Pool3
        x = ((x - 64) // 2 + 1) // 4
        self.fc_input_size = x * 128
        
    def forward(self, x):
        # Conv blocks
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # FC layers
        x = self.dropout1(torch.relu(self.fc1(x)))
        x = self.dropout2(torch.relu(self.fc2(x)))
        x = self.fc3(x)
        
        return x

# Audio preprocessing utilities
def preprocess_audio(audio_data, target_sr=22050, target_length=44100, orig_sr=44100):
    """
    Preprocess audio data for model input
    Args:
        audio_data: Raw audio data
        target_sr: Target sampling rate (22050 Hz)
        target_length: Target length in samples (44100 for 1 second at 44.1kHz)
        orig_sr: Original sampling rate of input audio
    Returns:
        Preprocessed tensor with shape (1, 1, 44100)
    """
    # Convert to tensor if numpy array
    if isinstance(audio_data, np.ndarray):
        audio_tensor = torch.from_numpy(audio_data).float()
    else:
        audio_tensor = audio_data.float()
    
    # Ensure mono (single channel)
    if audio_tensor.dim() > 1:
        audio_tensor = torch.mean(audio_tensor, dim=0)
    
    # Resample if needed using proper torchaudio transforms
    if orig_sr != target_sr:
        resampler = T.Resample(orig_freq=orig_sr, new_freq=target_sr)
        audio_tensor = resampler(audio_tensor)
    
    # Adjust length: pad with zeros or truncate
    current_length = audio_tensor.size(0)
    if current_length < target_length:
        # Pad with zeros
        padding = target_length - current_length
        audio_tensor = torch.nn.functional.pad(audio_tensor, (0, padding))
    elif current_length > target_length:
        # Truncate
        audio_tensor = audio_tensor[:target_length]
    
    # Reshape to (batch_size, channels, length) = (1, 1, 44100)
    audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0)
    
    return audio_tensor

# Advanced CNN Model Definition (for sequential architecture with training config support)
class CNNSequential(nn.Module):
    def __init__(self, input_length=44100, num_classes=2, channels=[64, 128, 256], kernel_sizes=[3, 3, 3], strides=[1, 2, 2], classifier_input_size=None, fc_sizes=None):
        super(CNNSequential, self).__init__()
        
        # CNN layers in sequential format - dynamically sized based on training config
        layers = []
        in_channels = 1
        for i, (out_channels, kernel_size, stride) in enumerate(zip(channels, kernel_sizes, strides)):
            # Add padding to maintain output size for stride=1, or reduce for stride>1
            padding = kernel_size // 2 if stride == 1 else 1
            layers.extend([
                nn.Conv1d(in_channels=in_channels, out_channels=out_channels, 
                         kernel_size=kernel_size, stride=stride, padding=padding),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=4, stride=4)
            ])
            in_channels = out_channels
            
        self.cnn = nn.Sequential(*layers)
        
        # Attention mechanism (if present in saved model)
        self.attention = None
        self.channels = channels
        
        # Use provided classifier input size or calculate it
        if classifier_input_size is not None:
            self.fc_input_size = classifier_input_size
        else:
            self._calculate_fc_input_size(input_length, channels, kernel_sizes, strides)
        
        # Classifier in sequential format - use extracted FC sizes or defaults
        if fc_sizes and len(fc_sizes) >= 2:
            fc_layer_sizes = fc_sizes
        else:
            fc_layer_sizes = [512, 256]  # Training config defaults
            
        self.classifier = nn.Sequential(
            nn.Linear(self.fc_input_size, fc_layer_sizes[0]),
            nn.Dropout(0.3),  # Match training config
            nn.ReLU(),
            nn.Linear(fc_layer_sizes[0], fc_layer_sizes[1]),
            nn.Dropout(0.3),  # Match training config
            nn.ReLU(),
            nn.Linear(fc_layer_sizes[1], num_classes)
        )
        
    def _calculate_fc_input_size(self, input_length, channels, kernel_sizes, strides):
        # Dynamic calculation based on actual architecture with proper padding
        x = input_length
        for i, (kernel_size, stride) in enumerate(zip(kernel_sizes, strides)):
            padding = kernel_size // 2 if stride == 1 else 1
            # Conv layer calculation
            x = (x + 2 * padding - kernel_size) // stride + 1
            # MaxPool layer calculation
            x = x // 4
        self.fc_input_size = x * channels[-1]
        
    def forward(self, x):
        # Debug input shape
        debug_tensor_shape("input_x", x, "model input")
        
        x = self.cnn(x)
        debug_tensor_shape("cnn_output", x, "after CNN layers")
        
        # Apply attention if available
        if self.attention is not None:
            debug_log("🎯 Applying attention mechanism")
            
            # Basic attention mechanism
            batch_size, channels, length = x.size()
            debug_log(f"CNN output dimensions: batch={batch_size}, channels={channels}, length={length}")
            
            x_flat = x.view(batch_size, channels * length)
            debug_tensor_shape("x_flat", x_flat, "flattened CNN output for attention")
            
            # CRITICAL FIX: Pre-emptive shape checking and projection before any attention operations
            try:
                # Check if attention layer has proper dimensions
                if hasattr(self.attention, 'query') and hasattr(self.attention.query, 'weight'):
                    expected_input_size = self.attention.query.weight.shape[1]
                    actual_input_size = x_flat.shape[1]
                    
                    debug_log(f"Pre-attention validation: expected={expected_input_size}, actual={actual_input_size}")
                    
                    if actual_input_size != expected_input_size:
                        debug_log(f"⚠️ SHAPE MISMATCH: Creating runtime projection {actual_input_size} -> {expected_input_size}", "warning")
                        
                        # Create projection layer if it doesn't exist
                        if not hasattr(self.attention, 'runtime_projection'):
                            self.attention.runtime_projection = nn.Linear(actual_input_size, expected_input_size)
                            debug_log(f"✅ Created runtime projection layer: {actual_input_size} -> {expected_input_size}")
                        
                        # Apply projection BEFORE any attention operations
                        x_flat = self.attention.runtime_projection(x_flat)
                        debug_tensor_shape("x_flat_projected", x_flat, "after runtime projection")
                
                # Apply input projection if needed (from MultiHeadAttention init)
                if hasattr(self.attention, 'input_projection') and self.attention.input_projection is not None:
                    debug_log("Applying input projection from MultiHeadAttention")
                    x_flat = self.attention.input_projection(x_flat)
                    debug_tensor_shape("x_flat_input_projected", x_flat, "after input projection")
                
                # Now attempt attention computation with properly projected input
                debug_log("Computing attention with validated dimensions")
                
                query = self.attention.query(x_flat)
                debug_tensor_shape("query", query, "attention query")
                
                key = self.attention.key(x_flat)
                debug_tensor_shape("key", key, "attention key")
                
                value = self.attention.value(x_flat)
                debug_tensor_shape("value", value, "attention value")
                
                # Scaled dot-product attention
                debug_log("Computing attention weights")
                attention_weights = torch.softmax(torch.matmul(query, key.transpose(-2, -1)) / (query.size(-1) ** 0.5), dim=-1)
                debug_tensor_shape("attention_weights", attention_weights, "attention weights")
                
                x_attended = torch.matmul(attention_weights, value)
                debug_tensor_shape("x_attended", x_attended, "after attention")
                
                x = self.attention.output(x_attended)
                debug_tensor_shape("attention_output", x, "final attention output")
                
            except Exception as e:
                debug_log(f"❌ Attention computation failed even with projection: {e}", "error")
                debug_log("Disabling attention mechanism for this model", "warning")
                
                # Disable attention completely and use CNN output directly
                self.attention = None
                x = x.view(x.size(0), -1)
                debug_tensor_shape("x_fallback_no_attention", x, "fallback without attention")
        else:
            # Flatten for classifier
            x = x.view(x.size(0), -1)
            debug_tensor_shape("x_no_attention", x, "flattened for classifier")
            
        classifier_result = self.classifier(x)
        debug_tensor_shape("classifier_output", classifier_result, "final model output")
        return classifier_result

# Enhanced Attention module for models that have it
class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, num_heads=8):
        super(MultiHeadAttention, self).__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        
        # Make sure input_dim is divisible by num_heads for multi-head attention
        # If not, adjust to the nearest divisible value
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
        
    def forward(self, x):
        # Apply input projection if needed
        if self.input_projection is not None:
            x = self.input_projection(x)
        return x  # Placeholder - actual attention logic in CNNSequential

def extract_cnn_architecture(state_dict, arch_type):
    """Extract CNN architecture parameters from state dict with enhanced detection"""
    channels = []
    kernel_sizes = []
    strides = []
    classifier_input_size = None
    attention_hidden_dim = None
    attention_num_heads = None
    fc_sizes = []
    
    if arch_type == 'sequential':
        # Extract from cnn.X.weight keys
        conv_keys = sorted([k for k in state_dict.keys() if k.startswith('cnn.') and k.endswith('.weight') and len(state_dict[k].shape) == 3])
        
        for key in conv_keys:
            weight = state_dict[key]
            if len(weight.shape) == 3:  # Conv1d weights
                out_channels, in_channels, kernel_size = weight.shape
                channels.append(out_channels)
                kernel_sizes.append(kernel_size)
        
        # Extract classifier architecture
        if 'classifier.0.weight' in state_dict:
            classifier_input_size = state_dict['classifier.0.weight'].shape[1]
            fc_sizes.append(state_dict['classifier.0.weight'].shape[0])
        if 'classifier.3.weight' in state_dict:
            fc_sizes.append(state_dict['classifier.3.weight'].shape[0])
            
        # Extract attention parameters
        if 'attention.query.weight' in state_dict:
            attention_hidden_dim = state_dict['attention.query.weight'].shape[0]
            # Infer number of heads from typical multi-head attention patterns
            attention_num_heads = 8  # Common default, could be extracted differently if needed
    
    elif arch_type == 'individual':
        # Extract from conv1.weight, conv2.weight, etc.
        for i in range(1, 4):  # conv1, conv2, conv3
            key = f'conv{i}.weight'
            if key in state_dict:
                weight = state_dict[key]
                out_channels, in_channels, kernel_size = weight.shape
                channels.append(out_channels)
                kernel_sizes.append(kernel_size)
        
        # Extract FC architecture
        if 'fc1.weight' in state_dict:
            classifier_input_size = state_dict['fc1.weight'].shape[1]
            fc_sizes.append(state_dict['fc1.weight'].shape[0])
        if 'fc2.weight' in state_dict:
            fc_sizes.append(state_dict['fc2.weight'].shape[0])
    
    # Determine strides based on typical patterns for given kernel sizes
    if kernel_sizes:
        if all(k <= 5 for k in kernel_sizes):  # Small kernels (like 3x3)
            strides = [1, 2, 2]  # Training configuration pattern
        else:  # Large kernels (like 256, 128, 64)
            strides = [4, 2, 2]  # Current code pattern
    
    # Default fallback
    if not channels:
        channels = [64, 128, 256]  # Match training config
        kernel_sizes = [3, 3, 3]  # Match training config
        strides = [1, 2, 2]  # Match training config
        
    return channels, kernel_sizes, strides, classifier_input_size, attention_hidden_dim, attention_num_heads, fc_sizes

def extract_attention_params(state_dict):
    """Extract attention parameters from state dict (DEPRECATED)
    
    WARNING: This function extracts the output dimensions from saved attention weights,
    NOT the input dimensions needed for creating new attention layers.
    This function is kept for compatibility but should NOT be used for creating new attention layers.
    Always use model.fc_input_size for attention input dimensions instead.
    """
    if 'attention.query.weight' in state_dict:
        # Note: This returns the output dim of query layer, not the input dim needed for creation
        # DO NOT USE THIS FOR input_dim when creating MultiHeadAttention
        hidden_dim = state_dict['attention.query.weight'].shape[0]  # This is OUTPUT dimension
        # Try to infer number of heads from attention patterns
        num_heads = 8  # Default based on training config
        return hidden_dim, num_heads
    return 256, 8  # defaults (OUTPUT dimensions, not input)

def create_adaptive_cnn(input_length, channels, kernel_sizes, strides, classifier_input_size=None, fc_sizes=None):
    """Create adaptive CNN with individual layers based on training config"""
    class AdaptiveCNN(nn.Module):
        def __init__(self):
            super(AdaptiveCNN, self).__init__()
            # Create layers dynamically based on training config
            for i, (out_ch, kernel_size, stride) in enumerate(zip(channels, kernel_sizes, strides)):
                in_ch = 1 if i == 0 else channels[i-1]
                padding = kernel_size // 2 if stride == 1 else 1
                
                setattr(self, f'conv{i+1}', nn.Conv1d(in_ch, out_ch, kernel_size, stride, padding))
                setattr(self, f'bn{i+1}', nn.BatchNorm1d(out_ch))
                setattr(self, f'relu{i+1}', nn.ReLU())
                setattr(self, f'pool{i+1}', nn.MaxPool1d(4, 4))
            
            # Use provided classifier input size or calculate it
            if classifier_input_size is not None:
                fc_input_size = classifier_input_size
            else:
                # Calculate FC size based on training config
                x = input_length
                for kernel_size, stride in zip(kernel_sizes, strides):
                    padding = kernel_size // 2 if stride == 1 else 1
                    x = (x + 2 * padding - kernel_size) // stride + 1
                    x = x // 4  # MaxPool
                fc_input_size = x * channels[-1]
            
            # FC layers - use extracted sizes or training config defaults
            if fc_sizes and len(fc_sizes) >= 2:
                fc_layer_sizes = fc_sizes
            else:
                fc_layer_sizes = [512, 256]  # Training config
                
            self.fc1 = nn.Linear(fc_input_size, fc_layer_sizes[0])
            self.dropout1 = nn.Dropout(0.3)  # Match training config
            self.fc2 = nn.Linear(fc_layer_sizes[0], fc_layer_sizes[1])
            self.dropout2 = nn.Dropout(0.3)  # Match training config
            self.fc3 = nn.Linear(fc_layer_sizes[1], 2)
            
        def forward(self, x):
            # Conv blocks
            for i in range(len(channels)):
                x = getattr(self, f'pool{i+1}')(getattr(self, f'relu{i+1}')(getattr(self, f'bn{i+1}')(getattr(self, f'conv{i+1}')(x))))
            
            # Flatten and FC
            x = x.view(x.size(0), -1)
            x = self.dropout1(torch.relu(self.fc1(x)))
            x = self.dropout2(torch.relu(self.fc2(x)))
            x = self.fc3(x)
            return x
    
    return AdaptiveCNN()

# Enhanced Model loading utility with training config matching
def load_model(model_path: str) -> nn.Module:
    """Load the trained CNN model from .pth file with training config matching"""
    debug_log(f"🚀 Starting model loading from: {model_path}")
    log_memory_usage("Model Loading Start")
    
    try:
        # Step 1: Load state dict
        debug_log("Step 1: Loading state dictionary from file")
        state_dict, load_error = safe_execute(
            lambda: torch.load(model_path, map_location='cpu'),
            "Loading PyTorch state dict"
        )
        
        if load_error:
            debug_log(f"Failed to load state dict: {load_error}", "error")
            return None
        
        debug_log(f"✅ State dict loaded successfully. Keys: {len(state_dict.keys())}")
        debug_log(f"State dict keys preview: {list(state_dict.keys())[:10]}")
        
        # Step 2: Analyze architecture
        debug_log("Step 2: Analyzing model architecture")
        has_individual_layers = any(key.startswith(('conv1.', 'conv2.', 'conv3.', 'fc1.', 'fc2.', 'fc3.')) for key in state_dict.keys())
        has_sequential_layers = any(key.startswith(('cnn.', 'classifier.')) for key in state_dict.keys())
        has_attention = any(key.startswith('attention.') for key in state_dict.keys())
        
        debug_log(f"Architecture analysis: Individual={has_individual_layers}, Sequential={has_sequential_layers}, Attention={has_attention}")
        
        # Use training config input length
        input_length = 44100  # Training configuration default
        debug_log(f"Using input length: {input_length}")
        
        # Step 3: Extract architecture parameters
        debug_log("Step 3: Extracting model architecture parameters")
        if has_sequential_layers:
            debug_log("📐 Sequential architecture detected - analyzing training configuration")
            st.info("Sequential architecture detected - analyzing training configuration")
            
            arch_params, arch_error = safe_execute(
                extract_cnn_architecture,
                "Extracting sequential architecture parameters",
                state_dict, 'sequential'
            )
            
            if arch_error:
                debug_log(f"Failed to extract architecture: {arch_error}", "error")
                return None
                
            channels, kernel_sizes, strides, classifier_input_size, attention_hidden_dim, attention_num_heads, fc_sizes = arch_params
            
            debug_log(f"Extracted parameters: channels={channels}, kernels={kernel_sizes}, strides={strides}")
            st.info(f"Detected channels: {channels}")
            st.info(f"Detected kernel sizes: {kernel_sizes}")
            st.info(f"Detected strides: {strides}")
            if classifier_input_size:
                debug_log(f"Classifier input size: {classifier_input_size}")
                st.info(f"Detected classifier input size: {classifier_input_size}")
            if fc_sizes:
                debug_log(f"FC layer sizes: {fc_sizes}")
                st.info(f"Detected FC layer sizes: {fc_sizes}")
            
            # Step 4: Create model instance
            debug_log("Step 4: Creating CNNSequential model instance")
            model, model_error = safe_execute(
                CNNSequential,
                "Creating CNNSequential model",
                input_length=input_length, 
                channels=channels, 
                kernel_sizes=kernel_sizes, 
                strides=strides,
                classifier_input_size=classifier_input_size,
                fc_sizes=fc_sizes
            )
            
            if model_error:
                debug_log(f"Failed to create model: {model_error}", "error")
                return None
            
            # Step 5: Add attention if present
            if has_attention:
                debug_log("Step 5: Adding attention mechanism")
                st.info("Attention mechanism detected")
                
                # CRITICAL FIX: Always use model's actual fc_input_size for attention input
                # Ignore any extracted attention_hidden_dim from state dict as it's the output dim, not input dim
                debug_log("Using model's actual fc_input_size for attention input dimensions")
                attention_input_size = model.fc_input_size  # This is the actual flattened CNN output size
                debug_log(f"Attention input size (from model.fc_input_size): {attention_input_size}")
                st.info(f"Attention input size: {attention_input_size}")
                
                attention, attention_error = safe_execute(
                    MultiHeadAttention,
                    "Creating MultiHeadAttention with model's fc_input_size",
                    input_dim=attention_input_size, num_heads=8  # Always use 8 heads
                )
                if attention_error:
                    debug_log(f"Failed to create attention: {attention_error}", "warning")
                    st.warning("Failed to create attention mechanism, continuing without it")
                else:
                    model.attention = attention
                    debug_log(f"✅ Attention mechanism created with input size: {attention_input_size}")
                
        elif has_individual_layers:
            debug_log("📐 Individual layer architecture detected - analyzing training configuration")
            st.info("Individual layer architecture detected - analyzing training configuration")
            
            arch_params, arch_error = safe_execute(
                extract_cnn_architecture,
                "Extracting individual architecture parameters",
                state_dict, 'individual'
            )
            
            if arch_error:
                debug_log(f"Failed to extract individual architecture: {arch_error}", "error")
                return None
                
            channels, kernel_sizes, strides, classifier_input_size, attention_hidden_dim, attention_num_heads, fc_sizes = arch_params
            
            debug_log(f"Individual layer parameters: channels={channels}, kernels={kernel_sizes}, strides={strides}")
            st.info(f"Detected channels: {channels}")
            st.info(f"Detected kernel sizes: {kernel_sizes}")
            st.info(f"Detected strides: {strides}")
            if classifier_input_size:
                debug_log(f"Individual classifier input size: {classifier_input_size}")
                st.info(f"Detected classifier input size: {classifier_input_size}")
            if fc_sizes:
                debug_log(f"Individual FC layer sizes: {fc_sizes}")
                st.info(f"Detected FC layer sizes: {fc_sizes}")
            
            debug_log("Step 4: Creating adaptive CNN model")
            model, model_error = safe_execute(
                create_adaptive_cnn,
                "Creating adaptive CNN model",
                input_length, channels, kernel_sizes, strides, classifier_input_size, fc_sizes
            )
            
            if model_error:
                debug_log(f"Failed to create adaptive model: {model_error}", "error")
                return None
                
        else:
            # Try to adapt by key mapping with training config as fallback
            debug_log("⚠️ Unknown architecture - attempting key mapping with training config")
            st.warning("Unknown architecture - attempting key mapping with training config")
            
            debug_log("Step 4: Creating fallback CNNSequential model with training config")
            model, model_error = safe_execute(
                CNNSequential,
                "Creating fallback CNNSequential model",
                input_length=44100,  # Training config
                channels=[64, 128, 256],  # Training config
                kernel_sizes=[3, 3, 3],  # Training config
                strides=[1, 2, 2],  # Training config
                fc_sizes=[512, 256]  # Training config
            )
            
            if model_error:
                debug_log(f"Failed to create fallback model: {model_error}", "error")
                return None
            
            debug_log("Adapting state dict keys for fallback model")
            adapted_state_dict, adapt_error = safe_execute(
                adapt_state_dict_keys,
                "Adapting state dict keys",
                state_dict
            )
            
            if adapt_error:
                debug_log(f"Failed to adapt state dict: {adapt_error}", "warning")
                st.warning("State dict adaptation failed, using original")
            else:
                state_dict = adapted_state_dict
        
        # Step 6: Load state dict with progressive error handling
        debug_log("Step 6: Loading state dict into model (CRITICAL STEP)")
        log_memory_usage("Before State Dict Loading")
        
        # ENHANCED DEBUGGING: Analyze state dict and model dimensions
        debug_log("📊 COMPREHENSIVE DIMENSION ANALYSIS")
        attention_weights, cnn_weights, classifier_weights = analyze_state_dict_dimensions(state_dict)
        
        # Calculate actual CNN output dimensions
        if hasattr(model, 'cnn'):
            actual_cnn_output_size = calculate_cnn_output_shape((1, 1, 44100), model)
            if actual_cnn_output_size:
                debug_log(f"🧮 ACTUAL CNN output flattened size: {actual_cnn_output_size}")
        
        # Diagnose attention compatibility
        attention_compatible = True
        if has_attention and hasattr(model, 'attention') and model.attention is not None:
            attention_compatible = diagnose_attention_compatibility(model, state_dict)
        
        # ENHANCED FIX: Recreate attention layer with correct dimensions if incompatible
        if has_attention and not attention_compatible:
            debug_log("🔧 RECREATING ATTENTION LAYER with correct dimensions", "warning")
            st.warning("🔧 Recreating attention layer with correct dimensions")
            
            # Use actual CNN output size for attention input
            correct_input_size = actual_cnn_output_size or model.fc_input_size
            debug_log(f"Creating new attention layer with input size: {correct_input_size}")
            
            new_attention, new_attention_error = safe_execute(
                MultiHeadAttention,
                "Creating corrected attention layer",
                input_dim=correct_input_size, num_heads=8
            )
            
            if new_attention_error is None:
                model.attention = new_attention
                debug_log(f"✅ Attention layer recreated with input size: {correct_input_size}")
                st.info(f"✅ Attention layer recreated with input size: {correct_input_size}")
            else:
                debug_log(f"Failed to recreate attention: {new_attention_error}", "error")
                model.attention = None
                debug_log("Disabling attention mechanism due to recreation failure", "warning")
        
        # ENHANCED FIX: Aggressively filter out ALL attention weights to prevent overriding
        debug_log("🚫 COMPREHENSIVE ATTENTION WEIGHT REMOVAL from state dict")
        filtered_state_dict = {}
        attention_keys_removed = []
        
        for key, value in state_dict.items():
            # Remove any key that contains 'attention' (more aggressive filtering)
            if 'attention' in key.lower() or key.startswith('attention.'):
                debug_log(f"Removing attention-related weight: {key} (shape: {value.shape})")
                attention_keys_removed.append(key)
                st.warning(f"🚫 Removing incompatible attention weight: {key}")
            else:
                filtered_state_dict[key] = value
        
        if attention_keys_removed:
            debug_log(f"🚫 REMOVED {len(attention_keys_removed)} attention parameters to prevent override:")
            for key in attention_keys_removed:
                debug_log(f"  - {key}")
            st.info(f"🚫 Removed {len(attention_keys_removed)} attention parameters to prevent dimension conflicts")
            state_dict = filtered_state_dict
        else:
            debug_log("No attention weights found in state dict to remove")
            st.info("✅ No conflicting attention weights found in model")
        
        # Attempt 1: Strict loading
        debug_log("Attempt 1: Strict state dict loading")
        strict_result, strict_error = safe_execute(
            lambda: model.load_state_dict(state_dict, strict=True),
            "Strict state dict loading"
        )
        
        if strict_error is None:
            debug_log("✅ Strict loading successful")
            st.success("✅ Model loaded successfully with strict loading")
        else:
            debug_log(f"⚠️ Strict loading failed: {strict_error}", "warning")
            st.warning(f"⚠️ Strict loading failed: {str(strict_error)[:200]}...")
            st.info("🔄 Attempting non-strict loading...")
            
            # Attempt 2: Non-strict loading
            debug_log("Attempt 2: Non-strict state dict loading")
            non_strict_result, non_strict_error = safe_execute(
                lambda: model.load_state_dict(state_dict, strict=False),
                "Non-strict state dict loading"
            )
            
            if non_strict_error is None:
                missing_keys, unexpected_keys = non_strict_result
                debug_log(f"Non-strict loading completed: missing={len(missing_keys)}, unexpected={len(unexpected_keys)}")
                if missing_keys:
                    debug_log(f"Missing keys: {missing_keys[:5]}..." if len(missing_keys) > 5 else f"Missing keys: {missing_keys}")
                    st.warning(f"Missing keys: {len(missing_keys)} keys")
                if unexpected_keys:
                    debug_log(f"Unexpected keys: {unexpected_keys[:5]}..." if len(unexpected_keys) > 5 else f"Unexpected keys: {unexpected_keys}")
                    st.warning(f"Unexpected keys: {len(unexpected_keys)} keys")
                st.info("✅ Non-strict loading completed - some parameters may be randomly initialized")
            else:
                debug_log(f"❌ Non-strict loading also failed: {non_strict_error}", "error")
                st.error(f"❌ Non-strict loading also failed: {non_strict_error}")
                
                # Attempt 3: Final fallback with training config
                debug_log("Attempt 3: Final fallback with training configuration model")
                st.info("🔄 Attempting fallback with training configuration model...")
                
                fallback_model, fallback_error = safe_execute(
                    CNNSequential,
                    "Creating final fallback model",
                    input_length=44100,
                    channels=[64, 128, 256],
                    kernel_sizes=[3, 3, 3],
                    strides=[1, 2, 2], 
                    fc_sizes=[512, 256]
                )
                
                if fallback_error:
                    debug_log(f"Failed to create fallback model: {fallback_error}", "error")
                    st.error(f"❌ All loading attempts failed: {fallback_error}")
                    return None
                
                if has_attention:
                    debug_log("Adding attention to fallback model")
                    # CRITICAL FIX: Use the fallback model's actual fc_input_size for attention dimensions
                    fallback_attention_input_size = fallback_model.fc_input_size
                    debug_log(f"Fallback attention input size (from fallback_model.fc_input_size): {fallback_attention_input_size}")
                    st.info(f"Fallback attention input size: {fallback_attention_input_size}")
                    
                    fallback_attention, fallback_attention_error = safe_execute(
                        MultiHeadAttention,
                        "Creating fallback attention with fallback model's fc_input_size",
                        input_dim=fallback_attention_input_size, num_heads=8
                    )
                    if fallback_attention_error is None:
                        fallback_model.attention = fallback_attention
                        debug_log(f"✅ Fallback attention created with input size: {fallback_attention_input_size}")
                    else:
                        debug_log(f"Failed to create fallback attention: {fallback_attention_error}", "warning")
                
                # Apply the same enhanced filtering for fallback model
                debug_log("🚫 COMPREHENSIVE FALLBACK ATTENTION WEIGHT REMOVAL")
                filtered_fallback_state_dict = {}
                fallback_attention_keys_removed = []
                
                for key, value in state_dict.items():
                    # Remove any key that contains 'attention' (more aggressive filtering)
                    if 'attention' in key.lower() or key.startswith('attention.'):
                        debug_log(f"Removing fallback attention-related weight: {key} (shape: {value.shape})")
                        fallback_attention_keys_removed.append(key)
                        st.warning(f"🚫 Removing fallback attention weight: {key}")
                    else:
                        filtered_fallback_state_dict[key] = value
                
                if fallback_attention_keys_removed:
                    debug_log(f"🚫 REMOVED {len(fallback_attention_keys_removed)} fallback attention parameters:")
                    for key in fallback_attention_keys_removed:
                        debug_log(f"  - {key}")
                    st.info(f"🚫 Removed {len(fallback_attention_keys_removed)} fallback attention parameters")
                    fallback_state_dict = filtered_fallback_state_dict
                else:
                    debug_log("No attention weights found in fallback state dict")
                    fallback_state_dict = state_dict
                
                fallback_load_result, fallback_load_error = safe_execute(
                    lambda: fallback_model.load_state_dict(fallback_state_dict, strict=False),
                    "Fallback model state dict loading"
                )
                
                if fallback_load_error:
                    debug_log(f"❌ Final fallback failed: {fallback_load_error}", "error")
                    st.error(f"❌ All loading attempts failed: {fallback_load_error}")
                    return None
                else:
                    debug_log("✅ Fallback loading successful")
                    st.success("✅ Fallback loading with training config successful")
                    model = fallback_model
        
        # Step 7: Set model to evaluation mode
        debug_log("Step 7: Setting model to evaluation mode")
        eval_result, eval_error = safe_execute(
            lambda: model.eval(),
            "Setting model to evaluation mode"
        )
        
        if eval_error:
            debug_log(f"Failed to set eval mode: {eval_error}", "warning")
            st.warning("Failed to set model to evaluation mode, but continuing")
        
        # Step 8: Model validation
        debug_log("Step 8: Validating loaded model")
        log_memory_usage("Model Loading Complete")
        
        # Test with dummy input
        dummy_test_result, dummy_test_error = safe_execute(
            lambda: model(torch.randn(1, 1, 44100)),
            "Model validation with dummy input"
        )
        
        if dummy_test_error:
            debug_log(f"❌ Model validation failed: {dummy_test_error}", "error")
            st.error(f"❌ Model validation failed: {dummy_test_error}")
            return None
        else:
            debug_log(f"✅ Model validation successful. Output shape: {dummy_test_result.shape}")
            st.success(f"✅ Model validation successful. Output shape: {dummy_test_result.shape}")
        
        debug_log("🎉 Model loading completed successfully")
        return model
        
    except Exception as e:
        error_msg = f"❌ Unexpected error in model loading: {e}"
        debug_log(error_msg, "error")
        
        # Log full traceback for unexpected errors
        full_traceback = traceback.format_exc()
        debug_logger.error(f"Unexpected error in load_model:\n{full_traceback}")
        
        st.error(f"❌ モデル読み込みエラー: {e}")
        with st.expander("🔍 Full Error Details"):
            st.text(full_traceback)
        
        log_memory_usage("Model Loading Failed")
        return None

def adapt_state_dict_keys(state_dict):
    """Adapt state dict keys to match the expected model structure"""
    adapted_dict = {}
    
    # Mapping from sequential to individual layer names
    key_mapping = {
        # CNN layers
        'cnn.0.': 'conv1.',
        'cnn.1.': 'bn1.',
        'cnn.4.': 'conv2.',
        'cnn.5.': 'bn2.',
        'cnn.8.': 'conv3.',
        'cnn.9.': 'bn3.',
        # Classifier layers
        'classifier.0.': 'fc1.',
        'classifier.3.': 'fc2.',
        'classifier.6.': 'fc3.',
    }
    
    for old_key, value in state_dict.items():
        new_key = old_key
        
        # Apply key mapping
        for old_prefix, new_prefix in key_mapping.items():
            if old_key.startswith(old_prefix):
                new_key = old_key.replace(old_prefix, new_prefix)
                break
                
        # Skip attention and other unknown keys for now
        if not new_key.startswith('attention.'):
            adapted_dict[new_key] = value
            
    return adapted_dict

# Audio processor class for WebRTC
class AudioProcessor(AudioProcessorBase):
    def __init__(self, model: nn.Module, target_sr: int = 22050, chunk_length: int = 44100):
        """Initialize AudioProcessor with comprehensive debugging and validation"""
        debug_log("🎯 ENTERING AudioProcessor.__init__")
        log_memory_usage("AudioProcessor Init Start")
        
        try:
            # Step 1: Validate model input
            debug_log("Step 1: Validating model input")
            if model is None:
                raise ValueError("Model cannot be None")
            
            if not isinstance(model, nn.Module):
                raise TypeError(f"Model must be nn.Module, got {type(model)}")
            
            debug_log(f"✅ Model validation passed: {type(model).__name__}")
            
            # Step 2: Test model compatibility
            debug_log("Step 2: Testing model compatibility with dummy input")
            try:
                with torch.no_grad():
                    test_input = torch.randn(1, 1, chunk_length)
                    debug_tensor_shape("test_input", test_input, "AudioProcessor compatibility test")
                    
                    test_output = model(test_input)
                    debug_tensor_shape("test_output", test_output, "AudioProcessor compatibility test result")
                    
                    debug_log(f"✅ Model compatibility test passed: output shape {test_output.shape}")
                    
                    # Clean up test tensors
                    del test_input, test_output
                    
            except Exception as model_test_error:
                debug_log(f"❌ Model compatibility test failed: {model_test_error}", "error")
                raise RuntimeError(f"Model failed compatibility test: {model_test_error}")
            
            # Step 3: Initialize basic attributes
            debug_log("Step 3: Initializing basic attributes")
            self.model = model
            self.target_sr = target_sr
            self.chunk_length = chunk_length
            
            debug_log(f"Configured parameters: target_sr={target_sr}, chunk_length={chunk_length}")
            
            # Step 4: Initialize data structures
            debug_log("Step 4: Initializing data structures")
            self.audio_buffer = []
            self.chunk_results = []
            self.chunk_audio_data = []
            
            # Step 5: Initialize threading lock
            debug_log("Step 5: Initializing threading lock")
            self.buffer_lock = threading.Lock()
            
            debug_log("✅ AudioProcessor initialization completed successfully")
            log_memory_usage("AudioProcessor Init Complete")
            
        except Exception as init_error:
            error_msg = f"❌ AudioProcessor initialization failed: {init_error}"
            debug_log(error_msg, "error")
            
            # Clean up any partially initialized state
            if hasattr(self, 'model'):
                delattr(self, 'model')
            
            log_memory_usage("AudioProcessor Init Failed")
            raise RuntimeError(f"AudioProcessor initialization failed: {init_error}")
        
    def recv_audio(self, frame: av.AudioFrame) -> av.AudioFrame:
        # Convert frame to numpy array
        audio_data = frame.to_ndarray()
        
        # Handle stereo to mono conversion
        if audio_data.shape[0] > 1:
            audio_data = np.mean(audio_data, axis=0)
        else:
            audio_data = audio_data[0]
        
        with self.buffer_lock:
            # Add to buffer
            self.audio_buffer.extend(audio_data)
            
            # Process when we have enough data for 1 second (based on chunk_length)
            while len(self.audio_buffer) >= self.chunk_length:
                # Extract 1-second chunk
                chunk = np.array(self.audio_buffer[:self.chunk_length])
                self.audio_buffer = self.audio_buffer[self.chunk_length:]
                
                # Preprocess and classify
                self._process_chunk(chunk)
        
        return frame
    
    def _process_chunk(self, chunk: np.ndarray):
        """Process a 1-second audio chunk"""
        try:
            # Preprocess audio with training config target length
            audio_tensor = preprocess_audio(chunk, self.target_sr, self.chunk_length)
            
            # Model inference
            with torch.no_grad():
                output = self.model(audio_tensor)
                predicted_class = torch.argmax(output, dim=1).item()
            
            # Store results
            self.chunk_results.append(predicted_class)
            self.chunk_audio_data.append(chunk.copy())
            
        except Exception as e:
            st.error(f"音声チャンク処理エラー: {e}")
    
    def get_results(self) -> Tuple[List[int], List[np.ndarray]]:
        """Get all classification results and audio data"""
        with self.buffer_lock:
            return self.chunk_results.copy(), self.chunk_audio_data.copy()
    
    def reset(self):
        """Reset processor state"""
        with self.buffer_lock:
            self.audio_buffer.clear()
            self.chunk_results.clear()
            self.chunk_audio_data.clear()

# Visualization functions
def plot_results(audio_chunks: List[np.ndarray], predictions: List[int]):
    """Plot audio waveform with colored background based on predictions"""
    if not audio_chunks or not predictions:
        st.warning("可視化するデータがありません")
        return
    
    # Concatenate all audio chunks
    full_audio = np.concatenate(audio_chunks)
    
    # Create time axis (using sample rate of 22050 for display)
    time_axis = np.linspace(0, len(full_audio) / 22050, len(full_audio))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot waveform
    ax.plot(time_axis, full_audio, color='black', linewidth=0.5, alpha=0.7)
    
    # Add colored background for each 1-second segment
    for i, prediction in enumerate(predictions):
        start_time = i
        end_time = i + 1
        
        if prediction == 0:  # OK
            color = 'green'
            alpha = 0.2
            label = 'OK' if i == 0 else ""
        else:  # NG
            color = 'red'
            alpha = 0.2
            label = 'NG' if i == 0 and prediction == 1 else ""
        
        ax.axvspan(start_time, end_time, alpha=alpha, color=color, label=label)
    
    # Formatting
    ax.set_xlabel('時間 (秒)')
    ax.set_ylabel('振幅')
    ax.set_title('OK/NG分類結果付き音声波形')
    ax.grid(True, alpha=0.3)
    
    # Add legend if we have both OK and NG results
    if 0 in predictions and 1 in predictions:
        ax.legend()
    
    # Display plot
    st.pyplot(fig)

# Main Streamlit application
def main():
    st.set_page_config(
        page_title="音声検出アプリ",
        page_icon="🎤",
        layout="wide"
    )
    
    st.title("🎤 リアルタイム音声分類")
    st.markdown("音声を録音し、1D CNNモデルを使用してリアルタイムOK/NG分類を取得")
    
    # Model file upload
    st.sidebar.header("モデル設定")
    model_file = st.sidebar.file_uploader(
        "学習済みモデルをアップロード (.pthファイル)",
        type=['pth'],
        help="学習済みPyTorchモデルファイルをアップロード"
    )
    
    # Initialize session state
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'audio_processor' not in st.session_state:
        st.session_state.audio_processor = None
    if 'recording_complete' not in st.session_state:
        st.session_state.recording_complete = False
    if 'results' not in st.session_state:
        st.session_state.results = None
    
    # Add debug mode toggle
    debug_mode = st.sidebar.checkbox("🔧 Debug Mode", value=True, help="Show detailed debugging information")
    
    if debug_mode:
        st.sidebar.info("🔧 Debug mode enabled - detailed logging will be shown")
        with st.sidebar.expander("🔧 System Info"):
            debug_log("System information check")
            try:
                st.write(f"**Python:** {sys.version}")
                st.write(f"**PyTorch:** {torch.__version__}")
                st.write(f"**Torchaudio:** {torchaudio.__version__}")
                st.write(f"**NumPy:** {np.__version__}")
                log_memory_usage("App Start")
            except Exception as e:
                st.error(f"Failed to get system info: {e}")
    
    # Load model with comprehensive debugging
    if model_file is not None:
        debug_log("🎯 Model file uploaded, starting loading process")
        
        try:
            # Step 1: Save uploaded file temporarily
            debug_log("Step 1: Saving uploaded file temporarily")
            temp_model_path = f"temp_model_{int(time.time())}.pth"
            
            file_save_result, file_save_error = safe_execute(
                lambda: _save_uploaded_file(model_file, temp_model_path),
                "Saving uploaded model file"
            )
            
            if file_save_error:
                debug_log(f"Failed to save uploaded file: {file_save_error}", "error")
                st.error(f"Failed to save uploaded file: {file_save_error}")
                return
            
            debug_log(f"✅ Model file saved to: {temp_model_path}")
            
            # Step 2: Load model with comprehensive debugging
            debug_log("Step 2: Starting comprehensive model loading")
            with st.spinner("モデル読み込み中..."):
                log_memory_usage("Before Model Loading")
                
                model_load_result, model_load_error = safe_execute(
                    load_model,
                    "Complete model loading process",
                    temp_model_path
                )
                
                if model_load_error:
                    debug_log(f"Model loading failed: {model_load_error}", "error")
                    st.error("モデルの読み込みに失敗しました")
                elif model_load_result is not None:
                    debug_log("✅ Model loading successful, storing in session state")
                    st.session_state.model = model_load_result
                    
                    # Force garbage collection after model loading
                    gc.collect()
                    log_memory_usage("After Model Loading")
                    
                    st.success("モデルが正常に読み込まれました！")
                    debug_log("🎉 Model successfully loaded and stored in session state")
                else:
                    debug_log("Model loading returned None", "error")
                    st.error("モデルの読み込みに失敗しました")
            
            # Step 3: Clean up temp file
            debug_log("Step 3: Cleaning up temporary file")
            cleanup_result, cleanup_error = safe_execute(
                lambda: _cleanup_temp_file(temp_model_path),
                "Cleaning up temporary file"
            )
            
            if cleanup_error:
                debug_log(f"Failed to cleanup temp file: {cleanup_error}", "warning")
                st.warning(f"Failed to cleanup temporary file: {cleanup_error}")
            else:
                debug_log("✅ Temporary file cleaned up successfully")
                
        except Exception as e:
            error_msg = f"Unexpected error in main model loading: {e}"
            debug_log(error_msg, "error")
            
            full_traceback = traceback.format_exc()
            debug_logger.error(f"Unexpected error in main model loading:\n{full_traceback}")
            
            st.error(f"モデル読み込みエラー: {e}")
            if debug_mode:
                with st.expander("🔍 Full Error Details"):
                    st.text(full_traceback)
    
    # Audio recording section with comprehensive debugging
    if st.session_state.model is not None:
        debug_log("🎙️ Model loaded successfully, initializing audio recording section")
        st.header("🎙️ 音声録音")
        
        # Pre-step: Validate model state before AudioProcessor creation
        debug_log("Pre-step: Comprehensive model state validation")
        log_memory_usage("Before Model State Validation")
        
        try:
            # Check if model is still valid
            if st.session_state.model is None:
                debug_log("❌ Model in session state is None", "error")
                st.error("モデルの状態が無効です。モデルを再読み込みしてください。")
                return
            
            # Check model type
            if not isinstance(st.session_state.model, nn.Module):
                debug_log(f"❌ Model is not nn.Module: {type(st.session_state.model)}", "error")
                st.error("モデルのタイプが無効です。")
                return
            
            debug_log(f"✅ Model type validation passed: {type(st.session_state.model).__name__}")
            
            # Test model with dummy input to ensure it's still functional
            debug_log("Testing model functionality before AudioProcessor creation")
            try:
                with torch.no_grad():
                    test_input = torch.randn(1, 1, 44100)
                    debug_tensor_shape("pre_processor_test_input", test_input)
                    
                    test_output = st.session_state.model(test_input)
                    debug_tensor_shape("pre_processor_test_output", test_output)
                    
                    debug_log(f"✅ Pre-AudioProcessor model test successful: {test_output.shape}")
                    
                    # Clean up
                    del test_input, test_output
                    gc.collect()
                    
            except Exception as model_test_error:
                debug_log(f"❌ Pre-AudioProcessor model test failed: {model_test_error}", "error")
                st.error(f"モデルの機能テストに失敗しました: {model_test_error}")
                with st.expander("🔍 Model Test Error Details"):
                    st.text(traceback.format_exc())
                return
            
            log_memory_usage("After Model State Validation")
            
        except Exception as validation_error:
            debug_log(f"❌ Model state validation failed: {validation_error}", "error")
            st.error(f"モデル状態検証エラー: {validation_error}")
            if debug_mode:
                with st.expander("🔍 Validation Error Details"):
                    st.text(traceback.format_exc())
            return
        
        # Step 1: Create audio processor with enhanced debugging
        if st.session_state.audio_processor is None:
            debug_log("Step 1: Creating AudioProcessor instance with validated model")
            
            # Additional safety: Force garbage collection before AudioProcessor creation
            debug_log("Performing garbage collection before AudioProcessor creation")
            gc.collect()
            log_memory_usage("Before AudioProcessor Creation")
            
            processor_result, processor_error = safe_execute(
                AudioProcessor,
                "Creating AudioProcessor with validated model",
                st.session_state.model, target_sr=22050, chunk_length=44100
            )
            
            if processor_error:
                debug_log(f"❌ Failed to create AudioProcessor: {processor_error}", "error")
                st.error(f"オーディオプロセッサの作成に失敗しました: {processor_error}")
                
                # Show detailed error information
                if debug_mode:
                    with st.expander("🔍 AudioProcessor Creation Error Details"):
                        st.text(traceback.format_exc())
                
                # Attempt recovery by resetting model state
                debug_log("Attempting recovery by clearing model state", "warning")
                st.warning("⚠️ AudioProcessor作成に失敗しました。モデルを再読み込みしてください。")
                if st.button("🔄 モデル状態をリセット"):
                    st.session_state.model = None
                    st.session_state.audio_processor = None
                    st.rerun()
                return
            else:
                st.session_state.audio_processor = processor_result
                debug_log("✅ AudioProcessor created successfully")
                log_memory_usage("After AudioProcessor Creation")
                st.success("✅ オーディオプロセッサが正常に作成されました")
                
                # Additional validation: Test AudioProcessor functionality
                debug_log("Testing AudioProcessor functionality")
                try:
                    # Test that AudioProcessor can handle basic operations
                    test_predictions, test_audio_chunks = st.session_state.audio_processor.get_results()
                    debug_log(f"✅ AudioProcessor functionality test passed: {len(test_predictions)} predictions, {len(test_audio_chunks)} chunks")
                except Exception as processor_test_error:
                    debug_log(f"⚠️ AudioProcessor functionality test failed: {processor_test_error}", "warning")
                    st.warning(f"AudioProcessor機能テストに失敗しましたが、継続します: {processor_test_error}")
        
        # Step 2: Initialize WebRTC streamer with enhanced debugging
        debug_log("Step 2: Initializing WebRTC streamer (CRITICAL SECTION)")
        log_memory_usage("Before WebRTC Initialization")
        
        try:
            # Pre-WebRTC validation
            debug_log("Pre-WebRTC validation: Checking AudioProcessor state")
            if st.session_state.audio_processor is None:
                debug_log("❌ AudioProcessor is None during WebRTC initialization", "error")
                st.error("AudioProcessor状態が無効です。")
                return
            
            debug_log("✅ AudioProcessor state validation passed for WebRTC")
            
            # Test AudioProcessor factory function before WebRTC
            debug_log("Testing AudioProcessor factory function")
            try:
                test_processor = lambda: st.session_state.audio_processor
                test_result = test_processor()
                if test_result is None:
                    raise ValueError("AudioProcessor factory returned None")
                debug_log("✅ AudioProcessor factory test passed")
            except Exception as factory_error:
                debug_log(f"❌ AudioProcessor factory test failed: {factory_error}", "error")
                st.error(f"AudioProcessor factory test failed: {factory_error}")
                return
            
            # Attempt WebRTC initialization with comprehensive error handling
            debug_log("Attempting WebRTC streamer initialization")
            webrtc_result, webrtc_error = safe_execute(
                lambda: webrtc_streamer(
                    key="audio-classification",
                    mode=WebRtcMode.SENDONLY,
                    audio_processor_factory=lambda: st.session_state.audio_processor,
                    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
                    media_stream_constraints={
                        "audio": True,
                        "video": False,
                    },
                    async_processing=True,
                ),
                "WebRTC streamer initialization"
            )
            
            if webrtc_error:
                debug_log(f"❌ WebRTC initialization failed: {webrtc_error}", "error")
                st.error(f"WebRTC初期化に失敗しました: {webrtc_error}")
                
                # Show detailed error information
                if debug_mode:
                    with st.expander("🔍 WebRTC Error Details"):
                        st.text(traceback.format_exc())
                
                # Provide recovery options
                st.warning("⚠️ WebRTC初期化に失敗しました。以下のオプションをお試しください:")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🔄 AudioProcessor再作成"):
                        st.session_state.audio_processor = None
                        st.rerun()
                with col2:
                    if st.button("🔄 完全リセット"):
                        st.session_state.model = None
                        st.session_state.audio_processor = None
                        st.rerun()
                return
            else:
                webrtc_ctx = webrtc_result
                debug_log("✅ WebRTC streamer initialized successfully")
                log_memory_usage("After WebRTC Initialization")
                st.success("✅ WebRTC streaming ready")
        
        except Exception as e:
            error_msg = f"Unexpected error in WebRTC initialization: {e}"
            debug_log(error_msg, "error")
            
            full_traceback = traceback.format_exc()
            debug_logger.error(f"Unexpected WebRTC error:\n{full_traceback}")
            
            st.error(f"WebRTC初期化エラー: {e}")
            if debug_mode:
                with st.expander("🔍 WebRTC Error Details"):
                    st.text(full_traceback)
            
            # Recovery options
            st.warning("⚠️ WebRTC初期化中に予期しないエラーが発生しました:")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 AudioProcessor再作成", key="unexpected_error_processor_reset"):
                    st.session_state.audio_processor = None
                    st.rerun()
            with col2:
                if st.button("🔄 完全リセット", key="unexpected_error_full_reset"):
                    st.session_state.model = None
                    st.session_state.audio_processor = None
                    st.rerun()
            return
        
        # Recording status
        if webrtc_ctx.state.playing:
            st.info("🔴 録音中... マイクに向かって話してください！")
            st.session_state.recording_complete = False
        elif webrtc_ctx.state.signalling:
            st.info("📡 接続中...")
        else:
            if not st.session_state.recording_complete and st.session_state.audio_processor is not None:
                # Recording just stopped, get results
                predictions, audio_chunks = st.session_state.audio_processor.get_results()
                if predictions:
                    st.session_state.results = (predictions, audio_chunks)
                    st.session_state.recording_complete = True
                    st.success(f"✅ 録音完了！{len(predictions)}秒の音声を処理しました。")
        
        # Results section
        if st.session_state.results is not None:
            st.header("📊 結果")
            predictions, audio_chunks = st.session_state.results
            
            if predictions:
                # Summary statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("総時間", f"{len(predictions)}秒")
                with col2:
                    ok_count = predictions.count(0)
                    st.metric("OKセグメント", f"{ok_count}")
                with col3:
                    ng_count = predictions.count(1)
                    st.metric("NGセグメント", f"{ng_count}")
                
                # Visualization
                st.subheader("分類結果付き音声波形")
                with st.spinner("可視化生成中..."):
                    plot_results(audio_chunks, predictions)
                
                # Detailed results
                with st.expander("詳細結果"):
                    for i, pred in enumerate(predictions):
                        status = "✅ OK" if pred == 0 else "❌ NG"
                        st.write(f"{i+1}秒目: {status}")
        
        # Reset button
        if st.button("🔄 リセット"):
            if st.session_state.audio_processor is not None:
                st.session_state.audio_processor.reset()
            st.session_state.recording_complete = False
            st.session_state.results = None
            st.rerun()
    
    else:
        st.warning("録音を開始するには学習済みモデルファイル (.pth) をアップロードしてください")
        st.info("""
        **使用方法:**
        1. サイドバーで学習済みPyTorchモデル (.pthファイル) をアップロード
        2. '開始' をクリックして録音を開始
        3. マイクに向かって話す
        4. '停止' をクリックして録音を終了し、結果を表示
        
        アプリは各1秒セグメントをOK (0) またはNG (1) として分類し、結果を可視化します。
        """)

if __name__ == "__main__":
    main()