"""
Data Pipeline and Utilities Module for Khmer Digits OCR

This module provides utilities for data loading, preprocessing, visualization,
and analysis for the Khmer digits OCR training pipeline, including:
- PyTorch Dataset and DataLoader utilities
- Image preprocessing and augmentation pipelines
- Data visualization and analysis tools
- Dataset statistics and validation utilities
"""

from .dataset import KhmerDigitsDataset, create_data_loaders
from .preprocessing import ImagePreprocessor, get_train_transforms, get_val_transforms
from .visualization import DataVisualizer, plot_samples, plot_dataset_stats
from .analysis import DatasetAnalyzer, calculate_dataset_metrics, validate_dataset_quality
from .font_utils import KhmerFontManager, safe_khmer_text, setup_khmer_fonts, print_font_status

__all__ = [
    'KhmerDigitsDataset',
    'create_data_loaders',
    'ImagePreprocessor',
    'get_train_transforms',
    'get_val_transforms',
    'DataVisualizer',
    'plot_samples',
    'plot_dataset_stats',
    'DatasetAnalyzer',
    'calculate_dataset_metrics',
    'validate_dataset_quality',
    'KhmerFontManager',
    'safe_khmer_text',
    'setup_khmer_fonts',
    'print_font_status'
]

__version__ = '1.0.1' 