#!/usr/bin/env python3
"""
Quick test to verify model loading from reference/best_model.pth
"""

import torch
import numpy as np
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_model_loading():
    """Test if the reference model can be loaded and its structure"""
    
    # Check if model file exists
    model_path = Path('reference/best_model.pth')
    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        return False
    
    try:
        # Load the model state dict
        logger.info(f"Loading model from {model_path}")
        state_dict = torch.load(model_path, map_location='cpu')
        
        # Print model structure
        logger.info("Model structure:")
        for key in sorted(state_dict.keys()):
            shape = state_dict[key].shape if hasattr(state_dict[key], 'shape') else 'scalar'
            logger.info(f"  {key}: {shape}")
        
        # Check for expected keys
        expected_patterns = ['cnn.', 'attention.', 'classifier.']
        found_patterns = {pattern: False for pattern in expected_patterns}
        
        for key in state_dict.keys():
            for pattern in expected_patterns:
                if pattern in key:
                    found_patterns[pattern] = True
        
        logger.info("\nPattern analysis:")
        for pattern, found in found_patterns.items():
            status = "✅" if found else "❌"
            logger.info(f"  {pattern}: {status}")
        
        return all(found_patterns.values())
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False

if __name__ == "__main__":
    success = test_model_loading()
    if success:
        logger.info("✅ Model structure analysis completed")
    else:
        logger.error("❌ Model loading failed")