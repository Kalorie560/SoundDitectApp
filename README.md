# SoundDitectApp

A Streamlit web application for real-time audio recording and classification using a 1D CNN model.

## Features

- **Real-time Audio Recording**: Uses `streamlit-webrtc` to capture microphone audio from the browser
- **1D CNN Classification**: Classifies audio segments every 1 second as OK (0) or NG (1)
- **Audio Processing**: Automatically handles audio preprocessing (22050 Hz sampling, mono conversion, length normalization)
- **Visualization**: Displays results on a time-domain waveform graph with color-coded segments
- **Model Support**: Loads pre-trained PyTorch models (.pth files)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Kalorie560/SoundDitectApp.git
cd SoundDitectApp
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit app:
```bash
streamlit run app.py
```

2. Upload your trained model (.pth file) using the sidebar

3. Click "Start" to begin recording audio

4. Speak into your microphone - the app will process 1-second chunks in real-time

5. Click "Stop" to end recording and view results

## Model Requirements

The application expects a PyTorch model with the following specifications:

- **Input Shape**: `(batch_size, channels, length) = (1, 1, 22050)`
- **Output**: Binary classification (2 classes: OK=0, NG=1)
- **Architecture**: 1D CNN using `nn.Conv1d` layers
- **Audio Format**: 22050 Hz sampling rate, mono channel, 1-second segments

## Audio Processing

The app automatically handles:
- Resampling to 22050 Hz
- Mono conversion (if stereo input)
- Length normalization (padding or truncation to 22050 samples)
- Real-time chunking into 1-second segments

## Results Visualization

After recording, the app displays:
- Audio waveform plot with color-coded background
- Green segments: OK classification
- Red segments: NG classification
- Summary statistics (total duration, OK/NG counts)
- Detailed second-by-second results

## Technical Details

- Built with Streamlit and streamlit-webrtc
- Uses PyTorch for model inference
- Audio processing with torchaudio and numpy
- Visualization with matplotlib
- Thread-safe audio buffering and processing