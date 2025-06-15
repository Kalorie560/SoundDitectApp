import streamlit as st
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T

# Configure torchaudio backend to prevent deprecation warnings
try:
    # Try setting backend if the method exists (older versions)
    if hasattr(torchaudio, 'set_audio_backend'):
        torchaudio.set_audio_backend("soundfile")
except Exception:
    # For newer versions, backend is handled automatically
    pass
import numpy as np
import matplotlib.pyplot as plt
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
import av
import queue
import threading
from typing import List, Tuple
import time

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
        x = self.cnn(x)
        
        # Apply attention if available
        if self.attention is not None:
            # Basic attention mechanism
            batch_size, channels, length = x.size()
            x_flat = x.view(batch_size, channels * length)
            
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

# Enhanced Attention module for models that have it
class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim=256, num_heads=8):
        super(MultiHeadAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x):
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
    """Extract attention parameters from state dict"""
    if 'attention.query.weight' in state_dict:
        hidden_dim = state_dict['attention.query.weight'].shape[0]
        # Try to infer number of heads from attention patterns
        num_heads = 8  # Default based on training config
        return hidden_dim, num_heads
    return 256, 8  # defaults

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
    try:
        # Load state dict first to inspect keys and shapes
        state_dict = torch.load(model_path, map_location='cpu')
        
        # Analyze keys to determine architecture
        has_individual_layers = any(key.startswith(('conv1.', 'conv2.', 'conv3.', 'fc1.', 'fc2.', 'fc3.')) for key in state_dict.keys())
        has_sequential_layers = any(key.startswith(('cnn.', 'classifier.')) for key in state_dict.keys())
        has_attention = any(key.startswith('attention.') for key in state_dict.keys())
        
        # Use training config input length
        input_length = 44100  # Training configuration default
        
        # Extract comprehensive architecture parameters from checkpoint
        if has_sequential_layers:
            st.info("Sequential architecture detected - analyzing training configuration")
            channels, kernel_sizes, strides, classifier_input_size, attention_hidden_dim, attention_num_heads, fc_sizes = extract_cnn_architecture(state_dict, 'sequential')
            st.info(f"Detected channels: {channels}")
            st.info(f"Detected kernel sizes: {kernel_sizes}")
            st.info(f"Detected strides: {strides}")
            if classifier_input_size:
                st.info(f"Detected classifier input size: {classifier_input_size}")
            if fc_sizes:
                st.info(f"Detected FC layer sizes: {fc_sizes}")
            
            model = CNNSequential(
                input_length=input_length, 
                channels=channels, 
                kernel_sizes=kernel_sizes, 
                strides=strides,
                classifier_input_size=classifier_input_size,
                fc_sizes=fc_sizes
            )
            
            # Add attention if present
            if has_attention:
                st.info("Attention mechanism detected")
                if attention_hidden_dim:
                    st.info(f"Attention hidden dim: {attention_hidden_dim}, heads: {attention_num_heads}")
                    model.attention = MultiHeadAttention(attention_hidden_dim, attention_num_heads)
                else:
                    model.attention = MultiHeadAttention(256, 8)  # Training config defaults
                
        elif has_individual_layers:
            st.info("Individual layer architecture detected - analyzing training configuration")
            channels, kernel_sizes, strides, classifier_input_size, attention_hidden_dim, attention_num_heads, fc_sizes = extract_cnn_architecture(state_dict, 'individual')
            st.info(f"Detected channels: {channels}")
            st.info(f"Detected kernel sizes: {kernel_sizes}")
            st.info(f"Detected strides: {strides}")
            if classifier_input_size:
                st.info(f"Detected classifier input size: {classifier_input_size}")
            if fc_sizes:
                st.info(f"Detected FC layer sizes: {fc_sizes}")
                
            model = create_adaptive_cnn(input_length, channels, kernel_sizes, strides, classifier_input_size, fc_sizes)
        else:
            # Try to adapt by key mapping with training config as fallback
            st.warning("Unknown architecture - attempting key mapping with training config")
            # Create model with training config parameters
            model = CNNSequential(
                input_length=44100,  # Training config
                channels=[64, 128, 256],  # Training config
                kernel_sizes=[3, 3, 3],  # Training config
                strides=[1, 2, 2],  # Training config
                fc_sizes=[512, 256]  # Training config
            )
            state_dict = adapt_state_dict_keys(state_dict)
        
        # Load state dict with progressive error handling
        try:
            model.load_state_dict(state_dict, strict=True)
            st.success("✅ Model loaded successfully with strict loading")
        except RuntimeError as e:
            st.warning(f"⚠️ Strict loading failed: {str(e)[:200]}...")
            st.info("🔄 Attempting non-strict loading...")
            try:
                missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
                if missing_keys:
                    st.warning(f"Missing keys: {len(missing_keys)} keys")
                if unexpected_keys:
                    st.warning(f"Unexpected keys: {len(unexpected_keys)} keys")
                st.info("✅ Non-strict loading completed - some parameters may be randomly initialized")
            except Exception as e2:
                st.error(f"❌ Non-strict loading also failed: {e2}")
                # Try one more fallback with training config
                st.info("🔄 Attempting fallback with training configuration model...")
                try:
                    fallback_model = CNNSequential(
                        input_length=44100,
                        channels=[64, 128, 256],
                        kernel_sizes=[3, 3, 3],
                        strides=[1, 2, 2], 
                        fc_sizes=[512, 256]
                    )
                    if has_attention:
                        fallback_model.attention = MultiHeadAttention(256, 8)
                    fallback_model.load_state_dict(state_dict, strict=False)
                    st.success("✅ Fallback loading with training config successful")
                    fallback_model.eval()
                    return fallback_model
                except Exception as e3:
                    st.error(f"❌ All loading attempts failed: {e3}")
                    return None
                
        model.eval()
        return model
        
    except Exception as e:
        st.error(f"❌ モデル読み込みエラー: {e}")
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
        self.model = model
        self.target_sr = target_sr
        self.chunk_length = chunk_length  # Training config: 44100 samples for 1 second at 44.1kHz
        self.audio_buffer = []
        self.chunk_results = []
        self.chunk_audio_data = []
        self.buffer_lock = threading.Lock()
        
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
    
    # Load model
    if model_file is not None:
        try:
            # Save uploaded file temporarily
            temp_model_path = f"temp_model_{int(time.time())}.pth"
            with open(temp_model_path, "wb") as f:
                f.write(model_file.getbuffer())
            
            # Load model
            with st.spinner("モデル読み込み中..."):
                model = load_model(temp_model_path)
                if model is not None:
                    st.session_state.model = model
                    st.success("モデルが正常に読み込まれました！")
                else:
                    st.error("モデルの読み込みに失敗しました")
            
            # Clean up temp file
            import os
            if os.path.exists(temp_model_path):
                os.remove(temp_model_path)
                
        except Exception as e:
            st.error(f"モデル読み込みエラー: {e}")
    
    # Audio recording section
    if st.session_state.model is not None:
        st.header("🎙️ 音声録音")
        
        # Create audio processor with training config chunk length
        if st.session_state.audio_processor is None:
            st.session_state.audio_processor = AudioProcessor(st.session_state.model, chunk_length=44100)
        
        # WebRTC streamer
        webrtc_ctx = webrtc_streamer(
            key="audio-classification",
            mode=WebRtcMode.SENDONLY,
            audio_processor_factory=lambda: st.session_state.audio_processor,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={
                "audio": True,
                "video": False,
            },
            async_processing=True,
        )
        
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