"""
Khmer Digits OCR Inference Engine

Provides inference capabilities for trained Khmer digits OCR models,
including checkpoint loading, image preprocessing, and prediction generation.
"""

import torch
import torch.nn.functional as F
from torch import nn
from PIL import Image
import numpy as np
from typing import List, Dict, Union, Optional, Tuple
import os
import yaml
import json
from pathlib import Path
import logging

# Import model components
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.ocr_model import KhmerDigitsOCR
from models.model_factory import ModelFactory
from modules.data_utils.preprocessing import ImagePreprocessor


class KhmerOCRInference:
    """
    Inference engine for Khmer digits OCR model.
    
    Handles model loading, image preprocessing, and prediction generation
    for both single images and batches.
    """
    
    def __init__(self, 
                 checkpoint_path: str,
                 config_path: Optional[str] = None,
                 device: Optional[str] = None,
                 model_preset: str = "small"):
        """
        Initialize the inference engine.
        
        Args:
            checkpoint_path: Path to the model checkpoint (.pth file)
            config_path: Path to model configuration file (optional)
            device: Device to run inference on ('cpu', 'cuda', or None for auto)
            model_preset: Model preset to use if config_path is not provided
        """
        self.checkpoint_path = checkpoint_path
        self.config_path = config_path
        self.model_preset = model_preset
        
        # Setup device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Initialize model and preprocessor
        self.model = None
        self.preprocessor = None
        self.model_info = None
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Load model
        self._load_model()
        
        # Initialize preprocessor
        self._setup_preprocessor()
    
    def _load_model(self):
        """Load the model from checkpoint."""
        try:
            # Load checkpoint
            checkpoint = torch.load(
                self.checkpoint_path, 
                map_location=self.device,
                weights_only=False
            )
            
            # Extract model configuration from checkpoint if available
            if 'model_config' in checkpoint:
                model_config = checkpoint['model_config']
                self.model = KhmerDigitsOCR(**model_config)
            elif self.config_path and os.path.exists(self.config_path):
                # Load from config file
                self.model = KhmerDigitsOCR.from_config(self.config_path)
            else:
                # Use preset configuration
                self.model = ModelFactory.create_model(preset=self.model_preset)
            
            # Load model weights
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            
            # Store model info
            if 'epoch' in checkpoint:
                self.model_info = {
                    'epoch': checkpoint['epoch'],
                    'train_loss': checkpoint.get('train_loss', 'N/A'),
                    'val_loss': checkpoint.get('val_loss', 'N/A'),
                    'train_char_accuracy': checkpoint.get('train_char_accuracy', 'N/A'),
                    'val_char_accuracy': checkpoint.get('val_char_accuracy', 'N/A'),
                    'train_seq_accuracy': checkpoint.get('train_seq_accuracy', 'N/A'),
                    'val_seq_accuracy': checkpoint.get('val_seq_accuracy', 'N/A')
                }
            
            self.logger.info(f"Model loaded successfully from {self.checkpoint_path}")
            self.logger.info(f"Model moved to device: {self.device}")
            
            if self.model_info:
                self.logger.info(f"Model training info: {self.model_info}")
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")
            raise RuntimeError(f"Could not load model from {self.checkpoint_path}: {str(e)}")
    
    def _setup_preprocessor(self):
        """Setup image preprocessor."""
        # Default image size for Khmer OCR
        image_size = (128, 64)  # (width, height)
        
        self.preprocessor = ImagePreprocessor(
            image_size=image_size,
            normalize=True,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )
    
    def predict_single(self, 
                      image: Union[str, Image.Image, np.ndarray],
                      return_confidence: bool = False) -> Union[str, Tuple[str, float]]:
        """
        Predict text from a single image.
        
        Args:
            image: Input image (file path, PIL Image, or numpy array)
            return_confidence: Whether to return confidence score
            
        Returns:
            Predicted text string, optionally with confidence score
        """
        # Load and preprocess image
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert('RGB')
        
        # Preprocess
        image_tensor = self.preprocessor.preprocess_image(image)
        image_tensor = image_tensor.unsqueeze(0).to(self.device)  # Add batch dimension
        
        # Predict
        with torch.no_grad():
            predictions = self.model.predict(image_tensor)
            predicted_text = predictions[0]
            
            if return_confidence:
                # Calculate confidence as average max probability
                logits = self.model(image_tensor)
                probs = F.softmax(logits, dim=-1)
                max_probs = torch.max(probs, dim=-1)[0]
                confidence = torch.mean(max_probs).item()
                return predicted_text, confidence
            
            return predicted_text
    
    def predict_batch(self, 
                     images: List[Union[str, Image.Image, np.ndarray]],
                     batch_size: int = 8,
                     return_confidence: bool = False) -> List[Union[str, Tuple[str, float]]]:
        """
        Predict text from a batch of images.
        
        Args:
            images: List of input images
            batch_size: Batch size for processing
            return_confidence: Whether to return confidence scores
            
        Returns:
            List of predicted text strings, optionally with confidence scores
        """
        results = []
        
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            
            # Preprocess batch
            batch_tensors = []
            for img in batch_images:
                if isinstance(img, str):
                    img = Image.open(img).convert('RGB')
                elif isinstance(img, np.ndarray):
                    img = Image.fromarray(img).convert('RGB')
                
                img_tensor = self.preprocessor.preprocess_image(img)
                batch_tensors.append(img_tensor)
            
            # Stack into batch
            batch_tensor = torch.stack(batch_tensors).to(self.device)
            
            # Predict
            with torch.no_grad():
                predictions = self.model.predict(batch_tensor)
                
                if return_confidence:
                    logits = self.model(batch_tensor)
                    probs = F.softmax(logits, dim=-1)
                    max_probs = torch.max(probs, dim=-1)[0]
                    confidences = torch.mean(max_probs, dim=-1).cpu().numpy()
                    
                    batch_results = [(pred, conf) for pred, conf in zip(predictions, confidences)]
                else:
                    batch_results = predictions
                
                results.extend(batch_results)
        
        return results
    
    def get_model_info(self) -> Dict:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        info = self.model.get_model_info()
        
        if self.model_info:
            info.update(self.model_info)
        
        info.update({
            'checkpoint_path': self.checkpoint_path,
            'device': str(self.device),
            'config_path': self.config_path,
            'model_preset': self.model_preset
        })
        
        return info
    
    def visualize_prediction(self, 
                           image: Union[str, Image.Image, np.ndarray],
                           save_path: Optional[str] = None) -> Tuple[str, Image.Image]:
        """
        Visualize prediction on an image.
        
        Args:
            image: Input image
            save_path: Path to save visualization (optional)
            
        Returns:
            Tuple of (predicted_text, annotated_image)
        """
        # Load image
        if isinstance(image, str):
            original_image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            original_image = Image.fromarray(image).convert('RGB')
        else:
            original_image = image.convert('RGB')
        
        # Get prediction
        prediction, confidence = self.predict_single(original_image, return_confidence=True)
        
        # Create visualization (simple text overlay)
        from PIL import ImageDraw, ImageFont
        
        vis_image = original_image.copy()
        draw = ImageDraw.Draw(vis_image)
        
        # Try to use a default font, fallback to built-in font
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        # Add prediction text
        text = f"Prediction: {prediction}\nConfidence: {confidence:.3f}"
        draw.text((10, 10), text, fill=(255, 0, 0), font=font)
        
        # Save if requested
        if save_path:
            vis_image.save(save_path)
        
        return prediction, vis_image


def setup_logging(level: int = logging.INFO):
    """Setup logging configuration."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
