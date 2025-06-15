import streamlit as st
import torch
import torch.nn as nn
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, ClientSettings
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
def preprocess_audio(audio_data, target_sr=22050, target_length=22050):
    """
    Preprocess audio data for model input
    Args:
        audio_data: Raw audio data
        target_sr: Target sampling rate (22050 Hz)
        target_length: Target length in samples (22050 for 1 second)
    Returns:
        Preprocessed tensor with shape (1, 1, 22050)
    """
    # Convert to tensor if numpy array
    if isinstance(audio_data, np.ndarray):
        audio_tensor = torch.from_numpy(audio_data).float()
    else:
        audio_tensor = audio_data.float()
    
    # Ensure mono (single channel)
    if audio_tensor.dim() > 1:
        audio_tensor = torch.mean(audio_tensor, dim=0)
    
    # Resample if needed (assuming input might be different sample rate)
    # For simplicity, we'll assume input is already at correct sample rate
    # In production, you might need: torchaudio.transforms.Resample(orig_freq, target_sr)
    
    # Adjust length: pad with zeros or truncate
    current_length = audio_tensor.size(0)
    if current_length < target_length:
        # Pad with zeros
        padding = target_length - current_length
        audio_tensor = torch.nn.functional.pad(audio_tensor, (0, padding))
    elif current_length > target_length:
        # Truncate
        audio_tensor = audio_tensor[:target_length]
    
    # Reshape to (batch_size, channels, length) = (1, 1, 22050)
    audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0)
    
    return audio_tensor

# Model loading utility
def load_model(model_path: str) -> CNN:
    """Load the trained CNN model from .pth file"""
    model = CNN()
    try:
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Audio processor class for WebRTC
class AudioProcessor(AudioProcessorBase):
    def __init__(self, model: CNN, target_sr: int = 22050):
        self.model = model
        self.target_sr = target_sr
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
            
            # Process when we have enough data for 1 second (22050 samples)
            while len(self.audio_buffer) >= self.target_sr:
                # Extract 1-second chunk
                chunk = np.array(self.audio_buffer[:self.target_sr])
                self.audio_buffer = self.audio_buffer[self.target_sr:]
                
                # Preprocess and classify
                self._process_chunk(chunk)
        
        return frame
    
    def _process_chunk(self, chunk: np.ndarray):
        """Process a 1-second audio chunk"""
        try:
            # Preprocess audio
            audio_tensor = preprocess_audio(chunk, self.target_sr, self.target_sr)
            
            # Model inference
            with torch.no_grad():
                output = self.model(audio_tensor)
                predicted_class = torch.argmax(output, dim=1).item()
            
            # Store results
            self.chunk_results.append(predicted_class)
            self.chunk_audio_data.append(chunk.copy())
            
        except Exception as e:
            st.error(f"Error processing audio chunk: {e}")
    
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
        st.warning("No data to visualize")
        return
    
    # Concatenate all audio chunks
    full_audio = np.concatenate(audio_chunks)
    
    # Create time axis
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
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Amplitude')
    ax.set_title('Audio Waveform with OK/NG Classification Results')
    ax.grid(True, alpha=0.3)
    
    # Add legend if we have both OK and NG results
    if 0 in predictions and 1 in predictions:
        ax.legend()
    
    # Display plot
    st.pyplot(fig)

# Main Streamlit application
def main():
    st.set_page_config(
        page_title="Sound Detection App",
        page_icon="🎤",
        layout="wide"
    )
    
    st.title("🎤 Real-time Audio Classification")
    st.markdown("Record audio and get real-time OK/NG classification using a 1D CNN model")
    
    # Model file upload
    st.sidebar.header("Model Configuration")
    model_file = st.sidebar.file_uploader(
        "Upload trained model (.pth file)",
        type=['pth'],
        help="Upload your trained PyTorch model file"
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
            with st.spinner("Loading model..."):
                model = load_model(temp_model_path)
                if model is not None:
                    st.session_state.model = model
                    st.success("Model loaded successfully!")
                else:
                    st.error("Failed to load model")
            
            # Clean up temp file
            import os
            if os.path.exists(temp_model_path):
                os.remove(temp_model_path)
                
        except Exception as e:
            st.error(f"Error loading model: {e}")
    
    # Audio recording section
    if st.session_state.model is not None:
        st.header("🎙️ Audio Recording")
        
        # Create audio processor
        if st.session_state.audio_processor is None:
            st.session_state.audio_processor = AudioProcessor(st.session_state.model)
        
        # WebRTC streamer
        webrtc_ctx = webrtc_streamer(
            key="audio-classification",
            mode=ClientSettings.MediaStreamConstraints.Mode.SENDONLY,
            audio_processor_factory=lambda: st.session_state.audio_processor,
            client_settings=ClientSettings(
                rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
                media_stream_constraints={
                    "audio": True,
                    "video": False,
                },
            ),
            async_processing=True,
        )
        
        # Recording status
        if webrtc_ctx.state.playing:
            st.info("🔴 Recording... Speak into your microphone!")
            st.session_state.recording_complete = False
        elif webrtc_ctx.state.signalling:
            st.info("📡 Connecting...")
        else:
            if not st.session_state.recording_complete and st.session_state.audio_processor is not None:
                # Recording just stopped, get results
                predictions, audio_chunks = st.session_state.audio_processor.get_results()
                if predictions:
                    st.session_state.results = (predictions, audio_chunks)
                    st.session_state.recording_complete = True
                    st.success(f"✅ Recording complete! Processed {len(predictions)} seconds of audio.")
        
        # Results section
        if st.session_state.results is not None:
            st.header("📊 Results")
            predictions, audio_chunks = st.session_state.results
            
            if predictions:
                # Summary statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Duration", f"{len(predictions)} seconds")
                with col2:
                    ok_count = predictions.count(0)
                    st.metric("OK Segments", f"{ok_count}")
                with col3:
                    ng_count = predictions.count(1)
                    st.metric("NG Segments", f"{ng_count}")
                
                # Visualization
                st.subheader("Audio Waveform with Classification")
                with st.spinner("Generating visualization..."):
                    plot_results(audio_chunks, predictions)
                
                # Detailed results
                with st.expander("Detailed Results"):
                    for i, pred in enumerate(predictions):
                        status = "✅ OK" if pred == 0 else "❌ NG"
                        st.write(f"Second {i+1}: {status}")
        
        # Reset button
        if st.button("🔄 Reset"):
            if st.session_state.audio_processor is not None:
                st.session_state.audio_processor.reset()
            st.session_state.recording_complete = False
            st.session_state.results = None
            st.rerun()
    
    else:
        st.warning("Please upload a trained model file (.pth) to start recording")
        st.info("""
        **Instructions:**
        1. Upload your trained PyTorch model (.pth file) in the sidebar
        2. Click 'Start' to begin recording
        3. Speak into your microphone
        4. Click 'Stop' to end recording and view results
        
        The app will classify each 1-second segment as OK (0) or NG (1) and visualize the results.
        """)

if __name__ == "__main__":
    main()