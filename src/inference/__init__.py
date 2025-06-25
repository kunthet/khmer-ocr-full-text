"""
Khmer Digits OCR Inference Module

Provides inference capabilities for trained Khmer digits OCR models.
"""

from .inference_engine import KhmerOCRInference, setup_logging

__all__ = [
    'KhmerOCRInference',
    'setup_logging'
] 