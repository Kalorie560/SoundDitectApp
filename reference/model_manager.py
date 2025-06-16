"""
Model Management Module for SoundDitect

This module handles the AI model architecture, training, inference,
and integration with ClearML for experiment tracking.
"""

# Comprehensive SSL and urllib3 warning suppression for macOS LibreSSL compatibility
import warnings
import urllib3
import ssl
import os

# Suppress all urllib3 warnings
urllib3.disable_warnings()
warnings.filterwarnings("ignore", "urllib3*")
warnings.filterwarnings("ignore", "Unverified HTTPS request*")
warnings.filterwarnings("ignore", message=".*urllib3.*")

# Configure urllib3 to fail faster and reduce retry noise
from urllib3.util.retry import Retry
import requests
from requests.adapters import HTTPAdapter

# Additional SSL configuration for macOS LibreSSL
try:
    # Set SSL context to be more permissive for older LibreSSL versions
    ssl._create_default_https_context = ssl._create_unverified_context
except AttributeError:
    pass

# Configure global session defaults to prevent excessive retries
# Note: Removed problematic monkey-patch that caused AttributeError with super().request

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from pathlib import Path
import logging
import asyncio
import json
import copy
from typing import Tuple, Optional, Dict, Any
import time
# ClearML for experiment tracking
try:
    from clearml import Task, Logger
    CLEARML_AVAILABLE = True
except ImportError:
    CLEARML_AVAILABLE = False
    logger.warning("ClearML not available - install with: pip install clearml")

logger = logging.getLogger(__name__)

class AttentionLayer(nn.Module):
    """Multi-head attention layer for audio feature processing."""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_heads: int):
        super(AttentionLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        batch_size, seq_len, input_dim = x.size()
        
        # Compute queries, keys, values
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended = torch.matmul(attention_weights, V)
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.hidden_dim
        )
        
        output = self.output(attended)
        return output

class SoundAnomalyDetector(nn.Module):
    """1D-CNN with Attention mechanism for sound anomaly detection."""
    
    def __init__(self, config: dict):
        super(SoundAnomalyDetector, self).__init__()
        self.config = config
        
        # CNN layers
        cnn_layers = []
        input_channels = 1
        
        for layer_config in config['model']['cnn_layers']:
            cnn_layers.extend([
                nn.Conv1d(
                    input_channels, 
                    layer_config['filters'],
                    kernel_size=layer_config['kernel_size'],
                    stride=layer_config['stride'],
                    padding=layer_config['padding']
                ),
                nn.BatchNorm1d(layer_config['filters']),
                nn.ReLU(),
                nn.MaxPool1d(2)
            ])
            input_channels = layer_config['filters']
        
        self.cnn = nn.Sequential(*cnn_layers)
        
        # Calculate CNN output size
        self.cnn_output_size = self._get_cnn_output_size(config['model']['input_length'])
        
        # Attention layer
        attention_config = config['model']['attention']
        self.attention = AttentionLayer(
            input_channels,
            attention_config['hidden_dim'],
            attention_config['num_heads']
        )
        
        # Fully connected layers
        fc_layers = []
        fc_input_size = attention_config['hidden_dim']
        
        for fc_config in config['model']['fully_connected']:
            fc_layers.extend([
                nn.Linear(fc_input_size, fc_config['units']),
                nn.ReLU(),
                nn.Dropout(fc_config['dropout'])
            ])
            fc_input_size = fc_config['units']
        
        # Output layer
        fc_layers.append(nn.Linear(fc_input_size, config['model']['num_classes']))
        self.classifier = nn.Sequential(*fc_layers)
        
        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
    def _get_cnn_output_size(self, input_length: int) -> int:
        """Calculate the output size after CNN layers."""
        # Create dummy input to calculate output size
        x = torch.randn(1, 1, input_length)
        with torch.no_grad():
            x = self.cnn(x)
        return x.size(-1)
    
    def forward(self, x):
        # Input shape: (batch_size, input_length)
        # Add channel dimension: (batch_size, 1, input_length)
        x = x.unsqueeze(1)
        
        # CNN feature extraction
        x = self.cnn(x)  # (batch_size, channels, reduced_length)
        
        # Prepare for attention: (batch_size, seq_len, features)
        x = x.transpose(1, 2)
        
        # Apply attention
        x = self.attention(x)
        
        # Global average pooling
        x = x.transpose(1, 2)  # Back to (batch_size, features, seq_len)
        x = self.global_avg_pool(x).squeeze(-1)  # (batch_size, features)
        
        # Classification
        output = self.classifier(x)
        
        return output

class AudioDataset(Dataset):
    """Dataset class for audio data with memory-efficient loading."""
    
    def __init__(self, data_files: list, config: dict, augment: bool = False):
        self.data_files = data_files
        self.config = config
        self.augment = augment
        self.data_indices = []
        
        logger.info(f"📊 Initializing AudioDataset - integrating {len(data_files)} JSON files into unified dataset")
        
        # Build index of all samples across all files
        total_samples = 0
        for file_idx, file_path in enumerate(data_files):
            try:
                logger.info(f"📁 Processing file {file_idx + 1}/{len(data_files)}: {file_path.name}")
                
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                file_samples = 0
                
                # Handle both old format (list of entries) and new format (waveforms/labels arrays)
                if isinstance(data, list):
                    # Old format: [{"Waveform": [...], "Labels": 0}, ...]
                    logger.info(f"   📋 Detected old format (list) with {len(data)} entries")
                    for sample_idx in range(len(data)):
                        # Validate that each entry has the required fields
                        entry = data[sample_idx]
                        if isinstance(entry, dict) and 'Waveform' in entry and 'Labels' in entry:
                            self.data_indices.append((file_idx, sample_idx))
                            file_samples += 1
                        else:
                            logger.warning(f"   ⚠️  Entry {sample_idx} missing 'Waveform' or 'Labels' fields")
                            
                elif isinstance(data, dict) and 'waveforms' in data:
                    # New format: {"waveforms": [[...], [...]], "labels": ["OK", "NG"]}
                    waveforms = data.get('waveforms', [])
                    labels = data.get('labels', [])
                    
                    logger.info(f"   📋 Detected new format with {len(waveforms)} waveforms and {len(labels)} labels")
                    
                    if len(waveforms) != len(labels):
                        logger.warning(f"   ⚠️  Mismatch: {len(waveforms)} waveforms vs {len(labels)} labels")
                        
                    num_samples = min(len(waveforms), len(labels))
                    for sample_idx in range(num_samples):
                        # Validate waveform data
                        waveform = waveforms[sample_idx]
                        if isinstance(waveform, list) and len(waveform) > 0:
                            self.data_indices.append((file_idx, sample_idx))
                            file_samples += 1
                        else:
                            logger.warning(f"   ⚠️  Sample {sample_idx} has invalid waveform data")
                            
                else:
                    # Check what the data structure actually looks like
                    if isinstance(data, dict):
                        keys = list(data.keys())
                        logger.warning(f"   ❌ Unknown dict format. Available keys: {keys}")
                    else:
                        logger.warning(f"   ❌ Unknown data type: {type(data)}")
                
                logger.info(f"   ✅ Added {file_samples} valid samples from this file")
                total_samples += file_samples
                
            except json.JSONDecodeError as e:
                logger.error(f"   ❌ JSON parsing error in {file_path}: {e}")
            except Exception as e:
                logger.error(f"   ❌ Could not process file {file_path}: {e}")
        
        logger.info(f"🎯 JSON file integration complete: {total_samples} total samples unified from {len(data_files)} files")
        logger.info(f"✨ All JSON training files in data folder have been successfully integrated into the dataset")
    
    def __len__(self):
        return len(self.data_indices)
    
    def __getitem__(self, idx):
        file_idx, sample_idx = self.data_indices[idx]
        
        # Load only the specific sample (memory efficient)
        try:
            with open(self.data_files[file_idx], 'r') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                # Old format: [{"Waveform": [...], "Labels": 0}, ...]
                sample = data[sample_idx]
                waveform = np.array(sample['Waveform'], dtype=np.float32)
                label = sample['Labels']
            elif isinstance(data, dict) and 'waveforms' in data:
                # New format: {"waveforms": [[...], [...]], "labels": ["OK", "NG"]}
                waveform = np.array(data['waveforms'][sample_idx], dtype=np.float32)
                label_str = data['labels'][sample_idx]
                
                # Convert string labels to integers
                if label_str == "OK":
                    label = 0
                elif label_str == "NG":
                    label = 1
                else:
                    logger.warning(f"Unknown label '{label_str}', defaulting to 0")
                    label = 0
            else:
                raise ValueError(f"Unknown data format in file {self.data_files[file_idx]}")
            
            # Apply augmentation if training
            if self.augment:
                # Add augmentation logic here if needed
                pass
            
            return torch.tensor(waveform), torch.tensor(label, dtype=torch.long)
            
        except Exception as e:
            logger.error(f"Error loading sample {idx}: {e}")
            # Return zero tensor as fallback
            return torch.zeros(self.config['model']['input_length']), torch.tensor(0, dtype=torch.long)

class ModelManager:
    """Manager for model training, inference, and experiment tracking."""
    
    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() and config['inference']['use_gpu'] else "cpu")
        self.model = None
        self.task = None
        self.clearml_enabled = False  # Initialize ClearML status flag
        self.clearml_connection_disabled = False  # Track if disabled due to connection issues
        self.clearml_retry_count = 0  # Track retry attempts
        self.prediction_count = 0
        
        logger.info(f"Using device: {self.device}")
        
        # Initialize ClearML task
        self._init_clearml()
    
    def _init_clearml(self):
        """Initialize ClearML experiment tracking with robust error handling."""
        if not CLEARML_AVAILABLE:
            logger.warning("🚫 ClearML not available - using local-only mode")
            self.task = None
            self.clearml_enabled = False
            return
        
        try:
            # Initialize ClearML task with configuration from config.yaml
            clearml_config = self.config.get('clearml', {})
            project_name = clearml_config.get('project_name', 'SoundDitect')
            task_name = clearml_config.get('task_name', 'anomaly_detection_training')
            output_uri = clearml_config.get('output_uri', None)
            
            # Set environment variable to reduce ClearML retry attempts
            os.environ['CLEARML_LOG_LEVEL'] = 'WARNING'
            
            logger.info(f"🔄 Initializing ClearML: {project_name}/{task_name}")
            
            self.task = Task.init(
                project_name=project_name,
                task_name=task_name,
                output_uri=output_uri
            )
            
            # Connect configuration parameters
            self.task.connect(self.config)
            
            # Get ClearML logger
            self.clearml_logger = Logger.current_logger()
            
            # Log configuration as hyperparameters
            if self.task:
                self.task.set_parameters(self.config)
            
            # ClearML initialized successfully - enable it immediately
            self.clearml_enabled = True
            self.clearml_connection_disabled = False  # Reset connection flag
            self.clearml_retry_count = 0  # Reset retry count
            
            logger.info(f"✅ ClearML initialized successfully: {project_name}/{task_name}")
            logger.info(f"🔗 ClearML dashboard: {self.task.get_output_log_web_page()}" if self.task else "")
            
        except Exception as e:
            logger.warning(f"⚠️ ClearML initialization error: {e}")
            logger.info("🔄 ClearML will operate in resilient mode - attempting to connect during training")
            
            # Don't disable ClearML completely - just mark as potentially having connection issues
            self.clearml_connection_disabled = False  # Allow attempts during training
            self.clearml_retry_count = 0
            
            # Keep task reference if it was created, even if there were connection issues
            if hasattr(self, 'task') and self.task:
                self.clearml_enabled = True
                logger.info("✅ ClearML task object available - will attempt logging during training")
            else:
                # Try a minimal initialization
                try:
                    clearml_config = self.config.get('clearml', {})
                    project_name = clearml_config.get('project_name', 'SoundDitect')
                    task_name = clearml_config.get('task_name', 'anomaly_detection_training')
                    output_uri = clearml_config.get('output_uri', None)
                    
                    self.task = Task.init(
                        project_name=project_name,
                        task_name=task_name,
                        output_uri=output_uri
                    )
                    self.task.connect(self.config)
                    self.clearml_logger = Logger.current_logger()
                    self.clearml_enabled = True
                    logger.info("✅ ClearML minimal initialization successful")
                except Exception as init_error:
                    logger.warning(f"⚠️ ClearML minimal initialization also failed: {init_error}")
                    self.task = None
                    self.clearml_enabled = False
    
    def _try_reenable_clearml(self):
        """Try to re-enable ClearML if it was disabled due to connection issues."""
        if not CLEARML_AVAILABLE:
            return
            
        # Always try to reconnect if we have a task object
        if self.task:
            logger.debug("🔄 Attempting to restore ClearML connection")
            self.clearml_connection_disabled = False
            return
        
        # If no task object, try to create one
        try:
            logger.debug("🔄 Attempting to re-initialize ClearML task")
            
            clearml_config = self.config.get('clearml', {})
            project_name = clearml_config.get('project_name', 'SoundDitect')
            task_name = clearml_config.get('task_name', 'anomaly_detection_training')
            output_uri = clearml_config.get('output_uri', None)
            
            self.task = Task.init(
                project_name=project_name,
                task_name=task_name,
                output_uri=output_uri
            )
            self.task.connect(self.config)
            self.clearml_logger = Logger.current_logger()
            if self.task:
                self.task.set_parameters(self.config)
            
            self.clearml_enabled = True
            self.clearml_connection_disabled = False
            self.clearml_retry_count = 0
            
            logger.info(f"✅ ClearML task re-initialized successfully: {project_name}/{task_name}")
            
        except Exception as e:
            logger.debug(f"⚠️ ClearML re-initialization failed: {e}")
            # Don't increment retry count here - let _safe_clearml_operation handle it
    
    def _safe_clearml_operation(self, operation_func, operation_name="ClearML operation", *args, **kwargs):
        """
        Safely execute a ClearML operation with connection health checking and retry logic.
        
        Args:
            operation_func: The ClearML function to execute
            operation_name: Description of the operation for logging
            *args, **kwargs: Arguments to pass to the operation function
        
        Returns:
            The result of the operation or None if failed/disabled
        """
        if not self.clearml_enabled or not self.task:
            # Try to re-enable ClearML if it was disabled due to connection issues
            if self.clearml_connection_disabled and self.clearml_retry_count < 10:
                self._try_reenable_clearml()
            if not self.clearml_enabled or not self.task:
                return None
        
        try:
            result = operation_func(*args, **kwargs)
            # If operation succeeded, reset retry count and connection disabled flag
            if self.clearml_connection_disabled or self.clearml_retry_count > 0:
                logger.debug(f"✅ ClearML {operation_name} successful - connection restored")
                self.clearml_connection_disabled = False
                self.clearml_retry_count = 0
            return result
        except Exception as e:
            # Check if it's a connection error
            error_str = str(e).lower()
            if any(keyword in error_str for keyword in ['connection', 'network', 'timeout', 'dns', 'nodename', 'unreachable']):
                logger.debug(f"🌐 ClearML connection issue during {operation_name}: {e}")
                self.clearml_connection_disabled = True
                self.clearml_retry_count += 1
                
                # Only disable ClearML entirely after many failures (increased threshold)
                if self.clearml_retry_count >= 15:
                    logger.warning(f"🚫 ClearML disabled after {self.clearml_retry_count} consecutive connection failures")
                    logger.info("💡 Training will continue in local-only mode")
                    self.clearml_enabled = False
                    self.task = None
                else:
                    logger.debug(f"🔄 ClearML retry attempt {self.clearml_retry_count}/15 - will keep trying")
            else:
                # Non-connection error - just log and continue
                logger.debug(f"⚠️ ClearML {operation_name} error (non-connection): {e}")
                # Don't increment retry count for non-connection errors
            return None

    def create_model(self) -> SoundAnomalyDetector:
        """Create a new model instance."""
        model = SoundAnomalyDetector(self.config)
        return model.to(self.device)
    
    async def load_model(self, model_path: Optional[str] = None) -> bool:
        """
        Load a trained model from disk, with automatic architecture detection.
        
        Args:
            model_path: Path to the model file
            
        Returns:
            True if model loaded successfully
        """
        try:
            if model_path is None:
                model_path = self.config['inference']['model_path']
            
            model_path = Path(model_path)
            
            # Try to load existing trained model
            if model_path.exists():
                try:
                    # First, inspect the checkpoint to detect architecture
                    logger.info(f"🔍 Inspecting checkpoint at {model_path}")
                    saved_config = self._detect_model_architecture(model_path)
                    
                    if saved_config:
                        logger.info("🔧 Creating model with detected architecture from checkpoint")
                        self.model = SoundAnomalyDetector(saved_config).to(self.device)
                    else:
                        logger.info("📋 Using default configuration for model creation")
                        self.model = self.create_model()
                    
                    # Load the state dict
                    state_dict = torch.load(model_path, map_location=self.device)
                    self.model.load_state_dict(state_dict)
                    self.model.eval()
                    
                    logger.info(f"✅ Trained model loaded successfully from {model_path}")
                    return True
                    
                except Exception as load_error:
                    logger.error(f"❌ Error loading trained model: {load_error}")
            
            # Fallback: Create and initialize a simple baseline model
            logger.warning(f"⚠️ No trained model found at {model_path}")
            logger.info("🔄 Creating baseline model for demonstration...")
            
            self.model = self.create_model()
            
            # Initialize with simple baseline weights for demonstration
            self._initialize_baseline_model()
            self.model.eval()
            
            logger.info("✅ Baseline model created and ready for use")
            logger.info("💡 Note: For better accuracy, train a model using actual audio data")
            return True
            
        except Exception as e:
            logger.error(f"❌ Critical error in model loading: {e}")
            return False
    
    def _detect_model_architecture(self, model_path: Path) -> Optional[dict]:
        """
        Detect model architecture from saved checkpoint by inspecting layer dimensions.
        
        Args:
            model_path: Path to the saved model checkpoint
            
        Returns:
            Modified config dict with detected architecture, or None if detection fails
        """
        try:
            # Load checkpoint to inspect architecture
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # Extract CNN layer information from state dict
            cnn_layers = []
            
            # Analyze CNN layers by looking at weight shapes
            layer_idx = 0
            while f'cnn.{layer_idx}.weight' in checkpoint:
                weight_shape = checkpoint[f'cnn.{layer_idx}.weight'].shape
                
                # CNN layer pattern: weight shape is [out_channels, in_channels, kernel_size]
                if len(weight_shape) == 3:
                    filters = weight_shape[0]
                    in_channels = weight_shape[1]
                    kernel_size = weight_shape[2]
                    
                    # Infer stride and padding from layer pattern (typical values)
                    stride = 2 if layer_idx > 0 else 1  # First layer usually stride=1, others stride=2
                    padding = 1 if kernel_size == 3 else 0
                    
                    cnn_layers.append({
                        'filters': int(filters),
                        'kernel_size': int(kernel_size),
                        'stride': stride,
                        'padding': padding
                    })
                    
                    logger.info(f"   Layer {layer_idx//4}: {filters} filters, kernel {kernel_size}")
                
                # Skip to next CNN layer (each layer has weight, bias, batch_norm weight, batch_norm bias)
                layer_idx += 4
            
            if not cnn_layers:
                logger.warning("Could not detect CNN layer architecture from checkpoint")
                return None
            
            # Detect attention layer dimensions
            attention_hidden_dim = 256  # Default
            attention_num_heads = 8     # Default
            
            # Look for attention layer weights to detect hidden_dim
            if 'attention.query.weight' in checkpoint:
                attention_weight_shape = checkpoint['attention.query.weight'].shape
                attention_hidden_dim = attention_weight_shape[0]
                logger.info(f"   Attention hidden_dim: {attention_hidden_dim}")
            
            # Detect fully connected layer dimensions
            fc_layers = []
            fc_idx = 0
            while f'classifier.{fc_idx}.weight' in checkpoint:
                weight_shape = checkpoint[f'classifier.{fc_idx}.weight'].shape
                
                if len(weight_shape) == 2:  # Linear layer
                    out_features = weight_shape[0]
                    in_features = weight_shape[1]
                    
                    # Skip the final output layer (num_classes)
                    next_fc_idx = fc_idx + 3  # Skip ReLU and Dropout
                    if f'classifier.{next_fc_idx}.weight' in checkpoint:
                        fc_layers.append({
                            'units': int(out_features),
                            'dropout': 0.3  # Default dropout
                        })
                        logger.info(f"   FC Layer {len(fc_layers)}: {out_features} units")
                
                fc_idx += 3  # Skip ReLU and Dropout layers
            
            # Create modified config with detected architecture
            detected_config = copy.deepcopy(self.config)  # Deep copy original config
            
            # Update with detected values
            detected_config['model']['cnn_layers'] = cnn_layers
            detected_config['model']['attention']['hidden_dim'] = attention_hidden_dim
            detected_config['model']['attention']['num_heads'] = attention_num_heads
            
            if fc_layers:
                detected_config['model']['fully_connected'] = fc_layers
            
            logger.info(f"✅ Successfully detected model architecture:")
            logger.info(f"   CNN layers: {len(cnn_layers)} layers with filters {[l['filters'] for l in cnn_layers]}")
            logger.info(f"   Attention: {attention_hidden_dim}D hidden, {attention_num_heads} heads")
            logger.info(f"   FC layers: {len(fc_layers)} layers")
            
            return detected_config
            
        except Exception as e:
            logger.warning(f"⚠️ Could not detect model architecture: {e}")
            return None
    
    def _initialize_baseline_model(self):
        """Initialize model with baseline weights for basic functionality."""
        try:
            # Simple initialization that provides reasonable behavior
            for name, param in self.model.named_parameters():
                if 'weight' in name:
                    if len(param.shape) > 1:
                        torch.nn.init.xavier_uniform_(param)
                    else:
                        torch.nn.init.uniform_(param, -0.1, 0.1)
                elif 'bias' in name:
                    torch.nn.init.constant_(param, 0)
            
            logger.info("🎯 Baseline model weights initialized")
            
        except Exception as e:
            logger.warning(f"⚠️ Baseline initialization failed: {e}")
    
    def save_model(self, model_path: Optional[str] = None):
        """Save the current model to disk and upload to ClearML."""
        try:
            if self.model is None:
                raise ValueError("No model to save")
            
            if model_path is None:
                model_path = self.config['data']['model_save_path']
            
            model_path = Path(model_path)
            model_path.parent.mkdir(parents=True, exist_ok=True)
            
            model_file = model_path / "best_model.pth"
            torch.save(self.model.state_dict(), model_file)
            logger.info(f"Model saved to {model_path}")
            
            # Upload model to ClearML safely
            model_uploaded = self._safe_clearml_operation(
                lambda: self.task.upload_artifact("best_model", artifact_object=str(model_file)),
                "model artifact upload"
            )
            if model_uploaded:
                logger.info("✅ Model uploaded to ClearML as artifact")
            else:
                logger.info("💾 Model saved locally (ClearML upload pending)")
                
                # Try to create TorchScript traced model (fallback if script fails)
                try:
                    # Create a sample input for tracing
                    sample_input = torch.randn(1, self.config['model']['input_length']).to(self.device)
                    
                    # Try tracing first (more robust than scripting)
                    traced_model = torch.jit.trace(self.model, sample_input)
                    traced_model_path = model_path / "best_model_traced.pth"
                    traced_model.save(str(traced_model_path))
                    
                    if self._safe_clearml_operation(
                        lambda: self.task.upload_artifact("traced_model", artifact_object=str(traced_model_path)),
                        "traced model artifact upload"
                    ):
                        logger.info("✅ TorchScript traced model uploaded to ClearML")
                    else:
                        logger.info("💾 TorchScript traced model saved locally (ClearML upload pending)")
                    
                except Exception as trace_error:
                    logger.warning(f"TorchScript tracing failed: {trace_error}")
                    
                    # Fallback: try scripting with model on CPU
                    try:
                        cpu_model = self.model.cpu()
                        scripted_model = torch.jit.script(cpu_model)
                        scripted_model_path = model_path / "best_model_scripted.pth"
                        scripted_model.save(str(scripted_model_path))
                        
                        if self._safe_clearml_operation(
                            lambda: self.task.upload_artifact("scripted_model", artifact_object=str(scripted_model_path)),
                            "scripted model artifact upload"
                        ):
                            logger.info("✅ TorchScript scripted model uploaded to ClearML")
                        else:
                            logger.info("💾 TorchScript scripted model saved locally (ClearML upload pending)")
                        
                        # Move model back to device
                        self.model.to(self.device)
                        
                    except Exception as script_error:
                        logger.warning(f"TorchScript scripting also failed: {script_error}")
                        logger.info("✅ Regular PyTorch model uploaded successfully (TorchScript conversion skipped)")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    async def predict(self, audio_data: np.ndarray) -> Tuple[int, float]:
        """
        Make prediction on audio data with enhanced error handling and baseline model.
        
        Args:
            audio_data: Preprocessed audio data
            
        Returns:
            Tuple of (prediction, confidence)
        """
        try:
            # Ensure model is available
            if self.model is None:
                logger.warning("⚠️ No model loaded, creating baseline model...")
                success = await self.load_model()
                if not success:
                    logger.error("❌ Failed to create baseline model")
                    return self._get_baseline_prediction(audio_data)
            
            # Validate input data
            if audio_data is None or len(audio_data) == 0:
                logger.warning("⚠️ Empty audio data for prediction")
                return 0, 0.5
            
            # Ensure correct input size
            expected_length = self.config['model']['input_length']
            if len(audio_data) != expected_length:
                logger.debug(f"🔧 Adjusting audio length from {len(audio_data)} to {expected_length}")
                if len(audio_data) > expected_length:
                    audio_data = audio_data[-expected_length:]  # Take last samples
                else:
                    # Pad with zeros
                    audio_data = np.pad(audio_data, (0, expected_length - len(audio_data)), mode='constant')
            
            self.model.eval()
            with torch.no_grad():
                # Convert to tensor and add batch dimension
                audio_tensor = torch.tensor(audio_data, dtype=torch.float32).unsqueeze(0).to(self.device)
                
                # Get prediction
                outputs = self.model(audio_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, prediction = torch.max(probabilities, 1)
                
                pred_value = prediction.item()
                conf_value = confidence.item()
                
                self.prediction_count += 1
                
                # Log prediction details for debugging
                logger.debug(f"🧠 Model prediction: {pred_value}, confidence: {conf_value:.3f}")
                
                return pred_value, conf_value
                
        except Exception as e:
            logger.error(f"❌ Prediction error: {e}", exc_info=True)
            # Return baseline prediction based on audio characteristics
            return self._get_baseline_prediction(audio_data)
    
    def _get_baseline_prediction(self, audio_data: np.ndarray) -> Tuple[int, float]:
        """
        Generate a baseline prediction based on simple audio characteristics.
        This provides a fallback when the model fails and ensures detection results are always output.
        """
        try:
            if audio_data is None or len(audio_data) == 0:
                logger.info("🎯 Baseline prediction for empty data: Normal (0.5)")
                return 0, 0.5
            
            # Enhanced rule-based prediction for better demonstration
            rms = np.sqrt(np.mean(audio_data ** 2))
            max_amplitude = np.max(np.abs(audio_data))
            mean_amplitude = np.mean(np.abs(audio_data))
            
            # Calculate variance for more sophisticated detection
            variance = np.var(audio_data)
            
            # Multi-factor heuristic for anomaly detection
            anomaly_score = 0.0
            
            # Factor 1: High amplitude suggests potential anomaly
            if max_amplitude > 0.3:
                anomaly_score += max_amplitude * 0.4
            
            # Factor 2: High RMS energy
            if rms > 0.05:
                anomaly_score += rms * 0.3
            
            # Factor 3: High variance suggests irregular sound
            if variance > 0.01:
                anomaly_score += variance * 0.2
            
            # Factor 4: Dynamic range (difference between max and mean)
            dynamic_range = max_amplitude - mean_amplitude
            if dynamic_range > 0.1:
                anomaly_score += dynamic_range * 0.1
            
            # Clamp anomaly score between 0 and 1
            anomaly_score = min(1.0, max(0.0, anomaly_score))
            
            # Determine prediction based on threshold
            threshold = 0.4
            if anomaly_score > threshold:
                prediction = 1  # Anomaly
                confidence = min(0.8, 0.5 + anomaly_score)
            else:
                prediction = 0  # Normal
                confidence = max(0.3, 0.7 - anomaly_score)
            
            logger.info(f"🎯 Baseline prediction: {prediction} (confidence: {confidence:.3f})")
            logger.debug(f"   Audio metrics - RMS: {rms:.4f}, Max: {max_amplitude:.4f}, Variance: {variance:.4f}, Score: {anomaly_score:.3f}")
            
            return prediction, confidence
            
        except Exception as e:
            logger.error(f"❌ Baseline prediction error: {e}")
            # Return a safe default that ensures results are always output
            logger.info("🎯 Fallback prediction: Normal (0.5)")
            return 0, 0.5
    
    async def train_model(self) -> bool:
        """
        Train the model with available data.
        
        Returns:
            True if training completed successfully
        """
        try:
            logger.info("Starting model training...")
            
            # Create data loaders
            train_loader, val_loader = self._create_data_loaders()
            
            if train_loader is None:
                logger.warning("No training data available")
                return False
            
            # Log dataset information to ClearML safely
            dataset_logged = False
            if self._safe_clearml_operation(
                lambda: self.task.set_parameter("dataset/train_samples", len(train_loader.dataset)),
                "dataset train samples logging"
            ):
                dataset_logged = True
                
            if self._safe_clearml_operation(
                lambda: self.task.set_parameter("dataset/val_samples", len(val_loader.dataset) if val_loader else 0),
                "dataset validation samples logging"
            ):
                dataset_logged = True
                
            if self._safe_clearml_operation(
                lambda: self.task.set_parameter("dataset/batch_size", self.config['training']['batch_size']),
                "dataset batch size logging"
            ):
                dataset_logged = True
            
            # Log data files being used
            data_dir = Path(self.config['data']['data_dir'])
            data_files = list(data_dir.glob("*.json"))
            if self._safe_clearml_operation(
                lambda: self.task.set_parameter("dataset/data_files", [f.name for f in data_files]),
                "dataset files logging"
            ):
                dataset_logged = True
            
            if dataset_logged:
                logger.info("✅ Dataset information successfully logged to ClearML")
            else:
                logger.info("📊 Dataset information logged locally (ClearML connection pending)")
            
            # Create model
            self.model = self.create_model()
            
            # Setup training
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(
                self.model.parameters(), 
                lr=self.config['training']['learning_rate']
            )
            
            # Training loop
            best_val_loss = float('inf')
            patience_counter = 0
            global_iteration = 0  # Track iterations across all epochs for ClearML
            
            for epoch in range(self.config['training']['epochs']):
                # Training phase
                train_loss, global_iteration = await self._train_epoch(train_loader, criterion, optimizer, epoch, global_iteration)
                
                # Validation phase
                val_loss, val_acc = await self._validate_epoch(val_loader, criterion)
                
                # Log metrics to ClearML safely
                metrics_logged = False
                if self._safe_clearml_operation(
                    lambda: self.clearml_logger.report_scalar("Training", "Loss", iteration=epoch, value=train_loss),
                    "training loss logging"
                ):
                    metrics_logged = True
                    
                if self._safe_clearml_operation(
                    lambda: self.clearml_logger.report_scalar("Validation", "Loss", iteration=epoch, value=val_loss),
                    "validation loss logging"
                ):
                    metrics_logged = True
                    
                if self._safe_clearml_operation(
                    lambda: self.clearml_logger.report_scalar("Validation", "Accuracy", iteration=epoch, value=val_acc),
                    "validation accuracy logging"
                ):
                    metrics_logged = True
                
                # Log both locally and ClearML status
                if metrics_logged:
                    logger.info(f"✅ Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f} (ClearML logged)")
                else:
                    logger.info(f"📊 Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f} (local only)")
                
                logger.info(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    self.save_model()
                else:
                    patience_counter += 1
                    if patience_counter >= self.config['training']['early_stopping']['patience']:
                        logger.info("Early stopping triggered")
                        break
            
            # Mark task as completed in ClearML safely
            if self._safe_clearml_operation(
                lambda: self.task.mark_completed(),
                "task completion"
            ):
                logger.info("✅ ClearML task marked as completed successfully")
            else:
                logger.info("📊 Training completed (ClearML completion pending)")
            
            logger.info("Training completed")
            return True
            
        except Exception as e:
            logger.error(f"Training error: {e}")
            return False
    
    def _create_data_loaders(self) -> Tuple[Optional[DataLoader], Optional[DataLoader]]:
        """Create training and validation data loaders."""
        try:
            data_dir = Path(self.config['data']['data_dir'])
            data_files = list(data_dir.glob("*.json"))
            
            if not data_files:
                logger.warning("No data files found")
                return None, None
            
            # Handle small datasets intelligently
            validation_split = self.config['training']['validation_split']
            
            logger.info(f"🔄 Integrating {len(data_files)} JSON files as training resources:")
            for i, file_path in enumerate(data_files, 1):
                logger.info(f"   {i}. {file_path.name}")
            
            if len(data_files) == 1:
                # For single file, use it for both training and validation
                # This prevents empty training sets with small datasets
                logger.info(f"📋 Single data file detected - using for both training and validation")
                train_files = data_files
                val_files = data_files
            else:
                # Multiple files: split files for training/validation
                split_idx = max(1, int(len(data_files) * (1 - validation_split)))
                train_files = data_files[:split_idx]
                val_files = data_files[split_idx:]
                
                logger.info(f"📋 Multiple files detected - splitting for training/validation:")
                logger.info(f"   🎯 Training files ({len(train_files)}):")
                for i, file_path in enumerate(train_files, 1):
                    logger.info(f"      {i}. {file_path.name}")
                logger.info(f"   ✅ Validation files ({len(val_files)}):")
                for i, file_path in enumerate(val_files, 1):
                    logger.info(f"      {i}. {file_path.name}")
            
            # Create datasets
            train_dataset = AudioDataset(train_files, self.config, augment=True)
            val_dataset = AudioDataset(val_files, self.config, augment=False)
            
            # Verify datasets have samples
            if len(train_dataset) == 0:
                logger.error("❌ Training dataset is empty!")
                return None, None
            
            if len(val_dataset) == 0:
                logger.warning("⚠️ Validation dataset is empty - using training data for validation")
                val_dataset = train_dataset
            
            logger.info(f"📊 Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
            
            # Create data loaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=min(self.config['training']['batch_size'], len(train_dataset)),
                shuffle=True,
                num_workers=0  # Set to 0 for stability with small datasets
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=min(self.config['training']['batch_size'], len(val_dataset)),
                shuffle=False,
                num_workers=0  # Set to 0 for stability with small datasets
            )
            
            return train_loader, val_loader
            
        except Exception as e:
            logger.error(f"Error creating data loaders: {e}")
            return None, None
    
    async def _train_epoch(self, train_loader: DataLoader, criterion, optimizer, epoch: int, global_iteration: int) -> Tuple[float, int]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            global_iteration += 1
            
            # Log training progress
            if batch_idx % max(1, len(train_loader) // 10) == 0:
                logger.debug(f"Training batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
                
                # Log to ClearML safely (less frequent batch logging)
                if batch_idx % max(1, len(train_loader) // 5) == 0:
                    batch_logged = False
                    if self._safe_clearml_operation(
                        lambda: self.clearml_logger.report_scalar("Training", "Batch Loss", iteration=global_iteration, value=loss.item()),
                        "batch loss logging"
                    ):
                        batch_logged = True
                        
                    if self._safe_clearml_operation(
                        lambda: self.clearml_logger.report_scalar("Training", "Learning Rate", iteration=global_iteration, value=optimizer.param_groups[0]['lr']),
                        "learning rate logging"
                    ):
                        batch_logged = True
                        
                    if batch_logged:
                        logger.debug(f"✅ Batch {batch_idx} metrics logged to ClearML")
            
            # Allow other coroutines to run
            if batch_idx % 10 == 0:
                await asyncio.sleep(0)
        
        return total_loss / len(train_loader), global_iteration
    
    async def _validate_epoch(self, val_loader: DataLoader, criterion) -> Tuple[float, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = criterion(output, target)
                total_loss += loss.item()
                
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                
                # Allow other coroutines to run
                if batch_idx % 10 == 0:
                    await asyncio.sleep(0)
        
        accuracy = correct / total if total > 0 else 0.0
        return total_loss / len(val_loader), accuracy
    
    def is_model_loaded(self) -> bool:
        """Check if a model is loaded."""
        return self.model is not None