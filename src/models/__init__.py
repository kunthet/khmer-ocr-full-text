"""
Khmer OCR Model Components

This module contains the complete model architecture for Khmer OCR,
including CNN backbones, RNN encoders/decoders, attention mechanisms,
and both digit and full text OCR models.
"""

from .backbone import CNNBackbone, ResNetBackbone, EfficientNetBackbone
from .attention import (
    BahdanauAttention, 
    EnhancedBahdanauAttention,
    MultiHeadAttention,
    HierarchicalAttention,
    PositionalEncoding
)
from .encoder import RNNEncoder, BiLSTMEncoder, ConvEncoder
from .decoder import RNNDecoder, AttentionDecoder
from .ocr_model import KhmerDigitsOCR, KhmerTextOCR
from .model_factory import (
    ModelFactory, 
    create_model, 
    create_text_model,
    create_digit_model,
    load_model,
    save_model
)
from .utils import ModelSummary, count_parameters, get_model_info

__all__ = [
    # Backbone components
    'CNNBackbone',
    'ResNetBackbone', 
    'EfficientNetBackbone',
    
    # Sequence components
    'RNNEncoder',
    'BiLSTMEncoder',
    'ConvEncoder',
    'RNNDecoder', 
    'AttentionDecoder',
    
    # Attention mechanisms
    'BahdanauAttention',
    'EnhancedBahdanauAttention',
    'MultiHeadAttention',
    'HierarchicalAttention',
    'PositionalEncoding',
    
    # Complete models
    'KhmerDigitsOCR',
    'KhmerTextOCR',
    
    # Factory and utilities
    'ModelFactory',
    'create_model',
    'create_text_model',
    'create_digit_model',
    'load_model',
    'save_model',
    'ModelSummary',
    'count_parameters',
    'get_model_info'
] 