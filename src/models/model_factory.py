"""
Model factory and utilities for creating OCR models.

Provides factory methods for creating models from configuration files
and utility functions for model management.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union
import yaml
import os

from .ocr_model import KhmerDigitsOCR, KhmerTextOCR
from .backbone import create_backbone
from .encoder import create_encoder
from .decoder import create_decoder


class ModelFactory:
    """
    Factory class for creating OCR models.
    
    Supports creating models from configuration files, presets,
    and custom parameters for both digit and full text recognition.
    """
    
    # Predefined model configurations for digits
    DIGIT_MODEL_PRESETS = {
        'small': {
            'cnn_type': 'resnet18',
            'encoder_type': 'bilstm',
            'decoder_type': 'attention',
            'feature_size': 256,
            'encoder_hidden_size': 128,
            'decoder_hidden_size': 128,
            'attention_size': 128,
            'num_encoder_layers': 1,
            'num_decoder_layers': 1,
            'dropout': 0.1
        },
        'medium': {
            'cnn_type': 'resnet18',
            'encoder_type': 'bilstm', 
            'decoder_type': 'attention',
            'feature_size': 512,
            'encoder_hidden_size': 256,
            'decoder_hidden_size': 256,
            'attention_size': 256,
            'num_encoder_layers': 2,
            'num_decoder_layers': 1,
            'dropout': 0.1
        },
        'large': {
            'cnn_type': 'efficientnet-b0',
            'encoder_type': 'bilstm',
            'decoder_type': 'attention', 
            'feature_size': 512,
            'encoder_hidden_size': 512,
            'decoder_hidden_size': 512,
            'attention_size': 512,
            'num_encoder_layers': 3,
            'num_decoder_layers': 2,
            'dropout': 0.1
        },
        'ctc_small': {
            'cnn_type': 'resnet18',
            'encoder_type': 'bilstm',
            'decoder_type': 'ctc',
            'feature_size': 256,
            'encoder_hidden_size': 128,
            'num_encoder_layers': 1,
            'dropout': 0.1
        },
        'ctc_medium': {
            'cnn_type': 'resnet18', 
            'encoder_type': 'bilstm',
            'decoder_type': 'ctc',
            'feature_size': 512,
            'encoder_hidden_size': 256,
            'num_encoder_layers': 2,
            'dropout': 0.1
        }
    }
    
    # Predefined model configurations for full Khmer text
    TEXT_MODEL_PRESETS = {
        'text_small': {
            'use_full_khmer': True,
            'max_sequence_length': 30,
            'cnn_type': 'resnet18',
            'encoder_type': 'bilstm',
            'decoder_type': 'attention',
            'feature_size': 512,
            'encoder_hidden_size': 256,
            'decoder_hidden_size': 256,
            'attention_size': 256,
            'num_attention_heads': 4,
            'num_encoder_layers': 2,
            'num_decoder_layers': 1,
            'dropout': 0.1,
            'enable_hierarchical': False,
            'enable_confidence_scoring': True
        },
        'text_medium': {
            'use_full_khmer': True,
            'max_sequence_length': 50,
            'cnn_type': 'resnet18',
            'encoder_type': 'bilstm',
            'decoder_type': 'attention',
            'feature_size': 512,
            'encoder_hidden_size': 512,
            'decoder_hidden_size': 512,
            'attention_size': 512,
            'num_attention_heads': 8,
            'num_encoder_layers': 3,
            'num_decoder_layers': 2,
            'dropout': 0.1,
            'enable_hierarchical': True,
            'enable_confidence_scoring': True
        },
        'text_large': {
            'use_full_khmer': True,
            'max_sequence_length': 100,
            'cnn_type': 'efficientnet-b0',
            'encoder_type': 'bilstm',
            'decoder_type': 'attention',
            'feature_size': 768,
            'encoder_hidden_size': 768,
            'decoder_hidden_size': 768,
            'attention_size': 768,
            'num_attention_heads': 12,
            'num_encoder_layers': 4,
            'num_decoder_layers': 3,
            'dropout': 0.1,
            'enable_hierarchical': True,
            'enable_confidence_scoring': True
        },
        'text_hierarchical': {
            'use_full_khmer': True,
            'max_sequence_length': 50,
            'cnn_type': 'resnet18',
            'encoder_type': 'bilstm',
            'decoder_type': 'attention',
            'feature_size': 512,
            'encoder_hidden_size': 512,
            'decoder_hidden_size': 512,
            'attention_size': 512,
            'num_attention_heads': 8,
            'num_encoder_layers': 3,
            'num_decoder_layers': 2,
            'dropout': 0.1,
            'enable_hierarchical': True,
            'enable_confidence_scoring': True
        },
        'text_fast': {
            'use_full_khmer': True,
            'max_sequence_length': 30,
            'cnn_type': 'resnet18',
            'encoder_type': 'conv',
            'decoder_type': 'ctc',
            'feature_size': 256,
            'encoder_hidden_size': 256,
            'num_encoder_layers': 2,
            'dropout': 0.1,
            'enable_hierarchical': False,
            'enable_confidence_scoring': False
        }
    }
    
    # Legacy presets for backward compatibility
    MODEL_PRESETS = DIGIT_MODEL_PRESETS
    
    @classmethod
    def create_model(cls,
                    config: Optional[Union[str, Dict[str, Any]]] = None,
                    preset: Optional[str] = None,
                    model_type: str = 'digits',
                    **kwargs) -> Union[KhmerDigitsOCR, KhmerTextOCR]:
        """
        Create OCR model from configuration or preset.
        
        Args:
            config: Configuration dict or path to config file
            preset: Name of predefined model preset
            model_type: Type of model ('digits' or 'text')
            **kwargs: Additional model parameters to override
            
        Returns:
            KhmerDigitsOCR or KhmerTextOCR model instance
            
        Raises:
            ValueError: If both config and preset are provided or neither is provided
        """
        if config is not None and preset is not None:
            raise ValueError("Cannot specify both config and preset")
        
        if config is None and preset is None:
            # Use default preset based on model type
            preset = 'medium' if model_type == 'digits' else 'text_medium'
        
        # Get base parameters
        if preset is not None:
            model_params = cls._get_preset_params(preset, model_type)
        else:
            # Load from config
            if isinstance(config, str):
                model_params = cls._load_config_file(config)
            else:
                model_params = cls._extract_model_params(config)
        
        # Override with kwargs
        model_params.update(kwargs)
        
        # Determine model class
        if model_type == 'text' or model_params.get('use_full_khmer', False):
            return KhmerTextOCR(**model_params)
        else:
            return KhmerDigitsOCR(**model_params)
    
    @classmethod
    def create_text_model(cls,
                         preset: str = 'text_medium',
                         **kwargs) -> KhmerTextOCR:
        """
        Create a Khmer text OCR model with enhanced features.
        
        Args:
            preset: Name of text model preset
            **kwargs: Additional model parameters to override
            
        Returns:
            KhmerTextOCR model instance
        """
        return cls.create_model(preset=preset, model_type='text', **kwargs)
    
    @classmethod
    def create_digit_model(cls,
                          preset: str = 'medium',
                          **kwargs) -> KhmerDigitsOCR:
        """
        Create a Khmer digits OCR model.
        
        Args:
            preset: Name of digit model preset
            **kwargs: Additional model parameters to override
            
        Returns:
            KhmerDigitsOCR model instance
        """
        return cls.create_model(preset=preset, model_type='digits', **kwargs)
    
    @classmethod
    def _get_preset_params(cls, preset: str, model_type: str) -> Dict[str, Any]:
        """Get parameters for a specific preset."""
        if model_type == 'text':
            if preset not in cls.TEXT_MODEL_PRESETS:
                raise ValueError(f"Unknown text preset: {preset}. Available: {list(cls.TEXT_MODEL_PRESETS.keys())}")
            return cls.TEXT_MODEL_PRESETS[preset].copy()
        else:
            if preset not in cls.DIGIT_MODEL_PRESETS:
                raise ValueError(f"Unknown digit preset: {preset}. Available: {list(cls.DIGIT_MODEL_PRESETS.keys())}")
            return cls.DIGIT_MODEL_PRESETS[preset].copy()
    
    @classmethod
    def _load_config_file(cls, config_path: str) -> Dict[str, Any]:
        """Load model parameters from configuration file."""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        return cls._extract_model_params(config)
    
    @classmethod  
    def _extract_model_params(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract model parameters from configuration dictionary."""
        model_config = config.get('model', {})
        
        # Determine if this is a text model
        use_full_khmer = model_config.get('use_full_khmer', False)
        vocab_size = model_config.get('characters', {}).get('total_classes', 13)
        
        # If vocab size > 13, assume it's a text model
        if vocab_size > 13:
            use_full_khmer = True
        
        # Base parameters
        model_params = {
            'vocab_size': vocab_size,
            'max_sequence_length': model_config.get('characters', {}).get('max_sequence_length', 8),
            'cnn_type': model_config.get('cnn', {}).get('type', 'resnet18'),
            'encoder_type': model_config.get('rnn', {}).get('encoder', {}).get('type', 'bilstm'),
            'decoder_type': 'attention',  # Default to attention decoder
            'feature_size': model_config.get('cnn', {}).get('feature_size', 512),
            'encoder_hidden_size': model_config.get('rnn', {}).get('encoder', {}).get('hidden_size', 256),
            'decoder_hidden_size': model_config.get('rnn', {}).get('decoder', {}).get('hidden_size', 256),
            'attention_size': model_config.get('rnn', {}).get('attention', {}).get('hidden_size', 256),
            'num_encoder_layers': model_config.get('rnn', {}).get('encoder', {}).get('num_layers', 2),
            'num_decoder_layers': model_config.get('rnn', {}).get('decoder', {}).get('num_layers', 1),
            'dropout': model_config.get('rnn', {}).get('encoder', {}).get('dropout', 0.1),
            'pretrained_cnn': model_config.get('cnn', {}).get('pretrained', True)
        }
        
        # Add text model specific parameters
        if use_full_khmer:
            model_params.update({
                'use_full_khmer': True,
                'num_attention_heads': model_config.get('attention', {}).get('num_heads', 8),
                'enable_hierarchical': model_config.get('hierarchical', {}).get('enabled', True),
                'enable_confidence_scoring': model_config.get('confidence', {}).get('enabled', True)
            })
        
        return model_params
    
    @classmethod
    def list_presets(cls, model_type: str = 'all') -> Dict[str, Dict[str, Any]]:
        """
        List available model presets.
        
        Args:
            model_type: Type of presets to list ('digits', 'text', or 'all')
        
        Returns:
            Dictionary of preset names and their configurations
        """
        if model_type == 'digits':
            return cls.DIGIT_MODEL_PRESETS.copy()
        elif model_type == 'text':
            return cls.TEXT_MODEL_PRESETS.copy()
        else:  # 'all'
            presets = cls.DIGIT_MODEL_PRESETS.copy()
            presets.update(cls.TEXT_MODEL_PRESETS)
            return presets
    
    @classmethod
    def get_preset_info(cls, preset: str) -> Dict[str, Any]:
        """
        Get information about a specific preset.
        
        Args:
            preset: Name of the preset
            
        Returns:
            Preset configuration and estimated parameters
        """
        # Check both digit and text presets
        config = None
        model_type = None
        
        if preset in cls.DIGIT_MODEL_PRESETS:
            config = cls.DIGIT_MODEL_PRESETS[preset].copy()
            model_type = 'digits'
        elif preset in cls.TEXT_MODEL_PRESETS:
            config = cls.TEXT_MODEL_PRESETS[preset].copy()
            model_type = 'text'
        else:
            raise ValueError(f"Unknown preset: {preset}")
        
        # Estimate parameters (rough approximation)
        if config['cnn_type'] == 'resnet18':
            cnn_params = 11_000_000
        elif config['cnn_type'] == 'efficientnet-b0':
            cnn_params = 5_000_000
        else:
            cnn_params = 10_000_000  # Default estimate
        
        encoder_params = config['encoder_hidden_size'] * config['feature_size'] * 4 * config['num_encoder_layers']
        
        if config.get('decoder_type', 'attention') == 'attention':
            decoder_params = config['decoder_hidden_size'] * config['encoder_hidden_size'] * 4
        else:  # CTC
            vocab_size = 13 if model_type == 'digits' else 115
            decoder_params = config['encoder_hidden_size'] * vocab_size
        
        # Add hierarchical and confidence scoring parameters for text models
        additional_params = 0
        if model_type == 'text':
            if config.get('enable_hierarchical', False):
                additional_params += config['decoder_hidden_size'] * 6  # Approximate
            if config.get('enable_confidence_scoring', False):
                additional_params += config['decoder_hidden_size'] * 2  # Approximate
        
        total_params = cnn_params + encoder_params + decoder_params + additional_params
        
        return {
            'config': config,
            'model_type': model_type,
            'estimated_parameters': {
                'cnn': cnn_params,
                'encoder': encoder_params, 
                'decoder': decoder_params,
                'additional': additional_params,
                'total': total_params
            },
            'memory_estimate_mb': total_params * 4 / (1024 * 1024)  # Float32
        }
    
    @classmethod
    def compare_presets(cls, presets: list) -> Dict[str, Dict[str, Any]]:
        """
        Compare multiple presets side by side.
        
        Args:
            presets: List of preset names to compare
            
        Returns:
            Dictionary with comparison information
        """
        comparison = {}
        for preset in presets:
            try:
                comparison[preset] = cls.get_preset_info(preset)
            except ValueError as e:
                comparison[preset] = {'error': str(e)}
        
        return comparison


# Convenience functions for backward compatibility and ease of use
def create_model(config_path: Optional[str] = None,
                preset: Optional[str] = None,
                model_type: str = 'digits',
                **kwargs) -> Union[KhmerDigitsOCR, KhmerTextOCR]:
    """
    Create OCR model using ModelFactory.
    
    Args:
        config_path: Path to configuration file
        preset: Name of model preset
        model_type: Type of model ('digits' or 'text')
        **kwargs: Additional model parameters
        
    Returns:
        OCR model instance
    """
    return ModelFactory.create_model(
        config=config_path,
        preset=preset,
        model_type=model_type,
        **kwargs
    )


def create_text_model(preset: str = 'text_medium', **kwargs) -> KhmerTextOCR:
    """Create a Khmer text OCR model."""
    return ModelFactory.create_text_model(preset=preset, **kwargs)


def create_digit_model(preset: str = 'medium', **kwargs) -> KhmerDigitsOCR:
    """Create a Khmer digits OCR model."""
    return ModelFactory.create_digit_model(preset=preset, **kwargs)


def load_model(checkpoint_path: str,
              map_location: Optional[str] = None) -> Union[KhmerDigitsOCR, KhmerTextOCR]:
    """
    Load OCR model from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        map_location: Device to load model to
        
    Returns:
        Loaded OCR model
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Load checkpoint
    if map_location is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        map_location = device
    
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    
    # Determine model type from checkpoint
    model_config = checkpoint.get('model_config', {})
    use_full_khmer = model_config.get('use_full_khmer', False)
    vocab_size = model_config.get('vocab_size', 13)
    
    if use_full_khmer or vocab_size > 13:
        model = KhmerTextOCR(**model_config)
    else:
        model = KhmerDigitsOCR(**model_config)
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model


def save_model(model: Union[KhmerDigitsOCR, KhmerTextOCR],
              checkpoint_path: str,
              optimizer: Optional[torch.optim.Optimizer] = None,
              epoch: Optional[int] = None,
              loss: Optional[float] = None,
              metrics: Optional[Dict[str, float]] = None):
    """
    Save OCR model to checkpoint.
    
    Args:
        model: OCR model to save
        checkpoint_path: Path to save checkpoint
        optimizer: Optimizer state to save (optional)
        epoch: Current epoch number (optional)
        loss: Current loss value (optional)
        metrics: Additional metrics to save (optional)
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_config': model.get_model_info()
    }
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    if epoch is not None:
        checkpoint['epoch'] = epoch
    
    if loss is not None:
        checkpoint['loss'] = loss
    
    if metrics is not None:
        checkpoint['metrics'] = metrics
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    torch.save(checkpoint, checkpoint_path) 