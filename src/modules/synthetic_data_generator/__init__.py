"""
Synthetic Data Generator Module for Khmer Digits OCR

This module provides utilities for generating synthetic training data
for Khmer digit recognition, including:
- Image generation with various fonts and backgrounds
- Data augmentation pipeline
- Unicode normalization utilities
- Dataset creation and validation
"""

from .generator import SyntheticDataGenerator
from .augmentation import ImageAugmentor
from .backgrounds import BackgroundGenerator
from .utils import normalize_khmer_text, validate_dataset

__all__ = [
    'SyntheticDataGenerator',
    'ImageAugmentor', 
    'BackgroundGenerator',
    'normalize_khmer_text',
    'validate_dataset'
]

__version__ = '1.0.0' 