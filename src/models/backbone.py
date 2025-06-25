"""
CNN Backbone architectures for feature extraction from Khmer digit images.

Supports ResNet-18 and EfficientNet-B0 backbones with pretrained weights
for robust feature extraction from OCR images.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Tuple, Optional, Dict, Any
from abc import ABC, abstractmethod

try:
    from efficientnet_pytorch import EfficientNet
    EFFICIENTNET_AVAILABLE = True
except ImportError:
    EFFICIENTNET_AVAILABLE = False


class CNNBackbone(nn.Module, ABC):
    """
    Abstract base class for CNN backbone architectures.
    
    Defines the interface for feature extraction from images.
    """
    
    def __init__(self, feature_size: int, pretrained: bool = True):
        """
        Initialize CNN backbone.
        
        Args:
            feature_size: Output feature vector size
            pretrained: Whether to use pretrained weights
        """
        super().__init__()
        self.feature_size = feature_size
        self.pretrained = pretrained
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from input images.
        
        Args:
            x: Input tensor [batch_size, channels, height, width]
            
        Returns:
            Feature tensor [batch_size, sequence_length, feature_size]
        """
        pass
    
    @property
    @abstractmethod
    def output_shape(self) -> Tuple[int, int]:
        """Return the output shape (sequence_length, feature_size)."""
        pass


class ResNetBackbone(CNNBackbone):
    """
    ResNet-18 backbone for feature extraction.
    
    Uses pretrained ResNet-18 and adapts it for sequence feature extraction
    by treating spatial dimensions as sequence positions.
    """
    
    def __init__(self, 
                 feature_size: int = 512,
                 pretrained: bool = True,
                 dropout: float = 0.1):
        """
        Initialize ResNet-18 backbone.
        
        Args:
            feature_size: Output feature vector size
            pretrained: Whether to use pretrained weights
            dropout: Dropout rate for regularization
        """
        super().__init__(feature_size, pretrained)
        
        # Load pretrained ResNet-18
        if pretrained:
            self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        else:
            self.resnet = models.resnet18(weights=None)
        
        # Remove the final classification layer
        self.features = nn.Sequential(*list(self.resnet.children())[:-2])
        
        # Adaptive pooling to get fixed spatial dimensions
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 8))  # Height=1, Width=8 for sequence
        
        # Feature projection layer
        resnet_feature_size = 512  # ResNet-18 final layer size
        self.feature_projection = nn.Sequential(
            nn.Linear(resnet_feature_size, feature_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # Store output dimensions
        self._output_shape = (8, feature_size)  # 8 sequence positions
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from input images.
        
        Args:
            x: Input tensor [batch_size, 3, height, width]
            
        Returns:
            Feature tensor [batch_size, 8, feature_size]
        """
        # Extract CNN features
        features = self.features(x)  # [batch_size, 512, h', w']
        
        # Adaptive pooling to fixed spatial size
        features = self.adaptive_pool(features)  # [batch_size, 512, 1, 8]
        
        # Reshape to sequence format
        batch_size = features.size(0)
        features = features.view(batch_size, 512, -1)  # [batch_size, 512, 8]
        features = features.permute(0, 2, 1)  # [batch_size, 8, 512]
        
        # Project to target feature size
        features = self.feature_projection(features)  # [batch_size, 8, feature_size]
        
        return features
    
    @property
    def output_shape(self) -> Tuple[int, int]:
        """Return the output shape (sequence_length, feature_size)."""
        return self._output_shape


class EfficientNetBackbone(CNNBackbone):
    """
    EfficientNet-B0 backbone for feature extraction.
    
    Uses pretrained EfficientNet-B0 for efficient feature extraction
    with better parameter efficiency than ResNet.
    """
    
    def __init__(self, 
                 feature_size: int = 512,
                 pretrained: bool = True,
                 dropout: float = 0.1):
        """
        Initialize EfficientNet-B0 backbone.
        
        Args:
            feature_size: Output feature vector size
            pretrained: Whether to use pretrained weights
            dropout: Dropout rate for regularization
        """
        super().__init__(feature_size, pretrained)
        
        if not EFFICIENTNET_AVAILABLE:
            raise ImportError(
                "EfficientNet not available. Install with: "
                "pip install efficientnet-pytorch"
            )
        
        # Load pretrained EfficientNet-B0
        if pretrained:
            self.efficientnet = EfficientNet.from_pretrained('efficientnet-b0')
        else:
            self.efficientnet = EfficientNet.from_name('efficientnet-b0')
        
        # Remove the final classification layer
        self.features = self.efficientnet.extract_features
        
        # Adaptive pooling to get fixed spatial dimensions
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 8))  # Height=1, Width=8 for sequence
        
        # Feature projection layer
        efficientnet_feature_size = 1280  # EfficientNet-B0 final layer size
        self.feature_projection = nn.Sequential(
            nn.Linear(efficientnet_feature_size, feature_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # Store output dimensions
        self._output_shape = (8, feature_size)  # 8 sequence positions
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from input images.
        
        Args:
            x: Input tensor [batch_size, 3, height, width]
            
        Returns:
            Feature tensor [batch_size, 8, feature_size]
        """
        # Extract CNN features
        features = self.features(x)  # [batch_size, 1280, h', w']
        
        # Adaptive pooling to fixed spatial size
        features = self.adaptive_pool(features)  # [batch_size, 1280, 1, 8]
        
        # Reshape to sequence format
        batch_size = features.size(0)
        features = features.view(batch_size, 1280, -1)  # [batch_size, 1280, 8]
        features = features.permute(0, 2, 1)  # [batch_size, 8, 1280]
        
        # Project to target feature size
        features = self.feature_projection(features)  # [batch_size, 8, feature_size]
        
        return features
    
    @property
    def output_shape(self) -> Tuple[int, int]:
        """Return the output shape (sequence_length, feature_size)."""
        return self._output_shape


def create_backbone(backbone_type: str, 
                   feature_size: int = 512,
                   pretrained: bool = True,
                   **kwargs) -> CNNBackbone:
    """
    Factory function to create CNN backbone.
    
    Args:
        backbone_type: Type of backbone ('resnet18' or 'efficientnet-b0')
        feature_size: Output feature vector size
        pretrained: Whether to use pretrained weights
        **kwargs: Additional arguments for the backbone
        
    Returns:
        CNN backbone instance
        
    Raises:
        ValueError: If backbone_type is not supported
    """
    if backbone_type.lower() in ['resnet18', 'resnet-18']:
        return ResNetBackbone(feature_size, pretrained, **kwargs)
    elif backbone_type.lower() in ['efficientnet-b0', 'efficientnet_b0']:
        return EfficientNetBackbone(feature_size, pretrained, **kwargs)
    else:
        raise ValueError(
            f"Unsupported backbone type: {backbone_type}. "
            f"Supported types: ['resnet18', 'efficientnet-b0']"
        ) 