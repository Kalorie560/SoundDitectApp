"""
Audio Processing Module for SoundDitect

This module handles audio preprocessing, feature extraction,
and real-time audio stream processing for anomaly detection.
"""

import numpy as np
import librosa
import scipy.signal
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class AudioProcessor:
    """
    Audio processor for real-time sound anomaly detection.
    
    Handles preprocessing, filtering, normalization, and feature extraction
    from raw audio data.
    """
    
    def __init__(self, config: dict):
        """
        Initialize the audio processor.
        
        Args:
            config: Configuration dictionary containing audio processing parameters
        """
        self.config = config
        self.sample_rate = config['audio']['sample_rate']
        self.chunk_size = config['audio']['chunk_size']
        self.target_length = config['model']['input_length']
        
        # Preprocessing settings
        self.normalize = config['preprocessing']['normalize']
        self.apply_filter = config['preprocessing']['apply_filter']
        
        if self.apply_filter:
            self.filter_type = config['preprocessing']['filter_type']
            self.low_freq = config['preprocessing']['filter_params']['low_freq']
            self.high_freq = config['preprocessing']['filter_params']['high_freq']
            self.filter_order = config['preprocessing']['filter_params']['order']
            
            # Design the filter coefficients
            self._design_filter()
    
    def _design_filter(self):
        """Design butterworth bandpass filter coefficients."""
        try:
            nyquist = self.sample_rate / 2
            low = self.low_freq / nyquist
            high = self.high_freq / nyquist
            
            self.filter_b, self.filter_a = scipy.signal.butter(
                self.filter_order, 
                [low, high], 
                btype='band'
            )
            logger.info(f"Filter designed: {self.low_freq}-{self.high_freq} Hz")
        except Exception as e:
            logger.error(f"Filter design failed: {e}")
            self.apply_filter = False
    
    def preprocess(self, audio_data: np.ndarray, sample_rate: Optional[int] = None) -> np.ndarray:
        """
        Preprocess raw audio data for model inference.
        
        Args:
            audio_data: Raw audio data as numpy array
            sample_rate: Sample rate of the audio data
            
        Returns:
            Preprocessed audio data ready for model input
        """
        try:
            # Ensure we're working with float32
            audio = audio_data.astype(np.float32)
            
            # Resample if necessary
            if sample_rate and sample_rate != self.sample_rate:
                audio = librosa.resample(
                    audio, 
                    orig_sr=sample_rate, 
                    target_sr=self.sample_rate
                )
            
            # Apply bandpass filter if enabled
            if self.apply_filter:
                audio = self._apply_filter(audio)
            
            # Normalize if enabled
            if self.normalize:
                audio = self._normalize_audio(audio)
            
            # Pad or truncate to target length
            audio = self._adjust_length(audio)
            
            return audio
            
        except Exception as e:
            logger.error(f"Preprocessing error: {e}")
            raise
    
    def _apply_filter(self, audio: np.ndarray) -> np.ndarray:
        """Apply bandpass filter to audio data."""
        try:
            filtered_audio = scipy.signal.filtfilt(
                self.filter_b, 
                self.filter_a, 
                audio
            )
            return filtered_audio.astype(np.float32)
        except Exception as e:
            logger.warning(f"Filtering failed: {e}")
            return audio
    
    def _normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio data to [-1, 1] range."""
        try:
            # Remove DC offset
            audio = audio - np.mean(audio)
            
            # Normalize to [-1, 1]
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                audio = audio / max_val
            
            return audio.astype(np.float32)
        except Exception as e:
            logger.warning(f"Normalization failed: {e}")
            return audio
    
    def _adjust_length(self, audio: np.ndarray) -> np.ndarray:
        """Adjust audio length to match model input requirements."""
        current_length = len(audio)
        
        if current_length > self.target_length:
            # Truncate to target length (take the most recent part)
            audio = audio[-self.target_length:]
        elif current_length < self.target_length:
            # Pad with zeros
            padding = self.target_length - current_length
            audio = np.pad(audio, (0, padding), mode='constant', constant_values=0)
        
        return audio
    
    def extract_features(self, audio: np.ndarray) -> dict:
        """
        Extract audio features for analysis and debugging.
        
        Args:
            audio: Preprocessed audio data
            
        Returns:
            Dictionary containing extracted features
        """
        try:
            features = {}
            
            # Time domain features
            features['rms'] = np.sqrt(np.mean(audio ** 2))
            features['zero_crossing_rate'] = librosa.feature.zero_crossing_rate(audio)[0].mean()
            features['spectral_centroid'] = librosa.feature.spectral_centroid(
                y=audio, sr=self.sample_rate
            )[0].mean()
            features['spectral_rolloff'] = librosa.feature.spectral_rolloff(
                y=audio, sr=self.sample_rate
            )[0].mean()
            
            # MFCC features
            mfccs = librosa.feature.mfcc(
                y=audio, sr=self.sample_rate, n_mfcc=13
            )
            features['mfcc_mean'] = np.mean(mfccs, axis=1)
            features['mfcc_std'] = np.std(mfccs, axis=1)
            
            return features
            
        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            return {}
    
    def detect_silence(self, audio: np.ndarray, threshold: float = 0.01) -> bool:
        """
        Detect if audio segment is mostly silence.
        
        Args:
            audio: Audio data
            threshold: RMS threshold for silence detection
            
        Returns:
            True if audio is considered silence
        """
        rms = np.sqrt(np.mean(audio ** 2))
        return rms < threshold
    
    def apply_augmentation(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply data augmentation for training data.
        
        Args:
            audio: Original audio data
            
        Returns:
            Augmented audio data
        """
        try:
            augmented = audio.copy()
            
            # Add noise
            noise_factor = self.config['training']['augmentation']['noise_factor']
            noise = np.random.normal(0, noise_factor, len(augmented))
            augmented += noise
            
            # Time shift
            time_shift_max = self.config['training']['augmentation']['time_shift_max']
            shift_amount = int(np.random.uniform(-time_shift_max, time_shift_max) * len(augmented))
            if shift_amount != 0:
                augmented = np.roll(augmented, shift_amount)
            
            # Pitch shift (using librosa)
            pitch_shift_range = self.config['training']['augmentation']['pitch_shift_range']
            pitch_shift = np.random.uniform(*pitch_shift_range)
            if abs(pitch_shift) > 0.1:
                augmented = librosa.effects.pitch_shift(
                    augmented, sr=self.sample_rate, n_steps=pitch_shift
                )
            
            return augmented.astype(np.float32)
            
        except Exception as e:
            logger.warning(f"Augmentation failed: {e}")
            return audio
    
    def create_sliding_windows(self, audio: np.ndarray, window_size: int, hop_size: int) -> np.ndarray:
        """
        Create sliding windows from audio data for batch processing.
        
        Args:
            audio: Input audio data
            window_size: Size of each window
            hop_size: Step size between windows
            
        Returns:
            Array of windowed audio segments
        """
        num_windows = (len(audio) - window_size) // hop_size + 1
        windows = np.zeros((num_windows, window_size), dtype=np.float32)
        
        for i in range(num_windows):
            start_idx = i * hop_size
            windows[i] = audio[start_idx:start_idx + window_size]
        
        return windows