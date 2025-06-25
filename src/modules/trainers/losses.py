"""
Loss Functions for Khmer Digits OCR Training

This module provides loss functions optimized for OCR sequence-to-sequence tasks,
including standard CrossEntropy and CTC (Connectionist Temporal Classification) alternatives.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class CrossEntropyLoss(nn.Module):
    """
    CrossEntropy loss with masking for variable-length sequences.
    Ignores PAD tokens and provides sequence-level loss normalization.
    """
    
    def __init__(self, 
                 pad_token_id: int = 11,
                 label_smoothing: float = 0.0,
                 reduction: str = 'mean'):
        """
        Initialize CrossEntropy loss.
        
        Args:
            pad_token_id: Token ID to ignore in loss calculation
            label_smoothing: Label smoothing factor (0.0 = no smoothing)
            reduction: Loss reduction method ('mean', 'sum', 'none')
        """
        super().__init__()
        self.pad_token_id = pad_token_id
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=pad_token_id,
            label_smoothing=label_smoothing,
            reduction=reduction
        )
    
    def forward(self, 
                predictions: torch.Tensor, 
                targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate CrossEntropy loss.
        
        Args:
            predictions: Model predictions [batch_size, seq_len, num_classes]
            targets: Target labels [batch_size, seq_len]
            
        Returns:
            Loss tensor
        """
        # Reshape for CrossEntropy: [batch_size * seq_len, num_classes]
        batch_size, seq_len, num_classes = predictions.shape
        predictions_flat = predictions.view(-1, num_classes)
        targets_flat = targets.view(-1)
        
        loss = self.criterion(predictions_flat, targets_flat)
        
        return loss


class CTCLoss(nn.Module):
    """
    CTC (Connectionist Temporal Classification) loss for alignment-free training.
    Useful for variable-length sequences without explicit alignment.
    """
    
    def __init__(self, 
                 blank_token_id: int = 0,
                 reduction: str = 'mean',
                 zero_infinity: bool = True):
        """
        Initialize CTC loss.
        
        Args:
            blank_token_id: Blank token ID for CTC
            reduction: Loss reduction method
            zero_infinity: Whether to zero infinite losses
        """
        super().__init__()
        self.blank_token_id = blank_token_id
        self.reduction = reduction
        self.zero_infinity = zero_infinity
        
        self.criterion = nn.CTCLoss(
            blank=blank_token_id,
            reduction=reduction,
            zero_infinity=zero_infinity
        )
    
    def forward(self,
                log_probs: torch.Tensor,
                targets: torch.Tensor,
                input_lengths: torch.Tensor,
                target_lengths: torch.Tensor) -> torch.Tensor:
        """
        Calculate CTC loss.
        
        Args:
            log_probs: Log probabilities [seq_len, batch_size, num_classes]
            targets: Target sequences (concatenated)
            input_lengths: Length of each sequence in the batch
            target_lengths: Length of each target in the batch
            
        Returns:
            Loss tensor
        """
        return self.criterion(log_probs, targets, input_lengths, target_lengths)


class OCRLoss(nn.Module):
    """
    Combined loss function for OCR tasks.
    Supports both CrossEntropy and CTC losses with optional weighting.
    """
    
    def __init__(self,
                 loss_type: str = 'crossentropy',
                 pad_token_id: int = 11,
                 blank_token_id: int = 0,
                 label_smoothing: float = 0.0,
                 focal_alpha: float = 1.0,
                 focal_gamma: float = 0.0,
                 reduction: str = 'mean'):
        """
        Initialize OCR loss.
        
        Args:
            loss_type: Type of loss ('crossentropy', 'ctc', 'focal')
            pad_token_id: PAD token ID for masking
            blank_token_id: BLANK token ID for CTC
            label_smoothing: Label smoothing factor
            focal_alpha: Focal loss alpha parameter
            focal_gamma: Focal loss gamma parameter (0.0 = standard CrossEntropy)
            reduction: Loss reduction method
        """
        super().__init__()
        self.loss_type = loss_type.lower()
        self.pad_token_id = pad_token_id
        self.blank_token_id = blank_token_id
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        
        if self.loss_type == 'crossentropy':
            self.loss_fn = CrossEntropyLoss(
                pad_token_id=pad_token_id,
                label_smoothing=label_smoothing,
                reduction=reduction
            )
        elif self.loss_type == 'ctc':
            self.loss_fn = CTCLoss(
                blank_token_id=blank_token_id,
                reduction=reduction
            )
        elif self.loss_type == 'focal':
            self.loss_fn = FocalLoss(
                alpha=focal_alpha,
                gamma=focal_gamma,
                ignore_index=pad_token_id,
                reduction=reduction
            )
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
    
    def forward(self, 
                predictions: torch.Tensor,
                targets: torch.Tensor,
                **kwargs) -> Dict[str, torch.Tensor]:
        """
        Calculate OCR loss.
        
        Args:
            predictions: Model predictions
            targets: Target labels
            **kwargs: Additional arguments for specific loss types
            
        Returns:
            Dictionary containing loss components
        """
        if self.loss_type in ['crossentropy', 'focal']:
            loss = self.loss_fn(predictions, targets)
            return {
                'loss': loss,
                'ce_loss': loss
            }
        
        elif self.loss_type == 'ctc':
            # CTC requires additional inputs
            log_probs = kwargs.get('log_probs', F.log_softmax(predictions, dim=-1))
            input_lengths = kwargs.get('input_lengths')
            target_lengths = kwargs.get('target_lengths')
            
            if input_lengths is None or target_lengths is None:
                raise ValueError("CTC loss requires input_lengths and target_lengths")
            
            # CTC expects [seq_len, batch_size, num_classes]
            log_probs = log_probs.permute(1, 0, 2)
            
            loss = self.loss_fn(log_probs, targets, input_lengths, target_lengths)
            return {
                'loss': loss,
                'ctc_loss': loss
            }


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in OCR tasks.
    """
    
    def __init__(self,
                 alpha: float = 1.0,
                 gamma: float = 2.0,
                 ignore_index: int = -100,
                 reduction: str = 'mean'):
        """
        Initialize Focal Loss.
        
        Args:
            alpha: Weighting factor for rare class
            gamma: Focusing parameter (0 = standard CrossEntropy)
            ignore_index: Index to ignore in loss calculation
            reduction: Loss reduction method
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction
    
    def forward(self, 
                predictions: torch.Tensor, 
                targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate Focal Loss.
        
        Args:
            predictions: Model predictions [batch_size, seq_len, num_classes]
            targets: Target labels [batch_size, seq_len]
            
        Returns:
            Loss tensor
        """
        # Reshape predictions and targets
        batch_size, seq_len, num_classes = predictions.shape
        predictions_flat = predictions.view(-1, num_classes)
        targets_flat = targets.view(-1)
        
        # Calculate standard cross entropy
        ce_loss = F.cross_entropy(
            predictions_flat, 
            targets_flat, 
            ignore_index=self.ignore_index, 
            reduction='none'
        )
        
        # Calculate probabilities and focal weight
        pt = torch.exp(-ce_loss)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        focal_loss = focal_weight * ce_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def create_loss_function(config: Dict[str, Any]) -> OCRLoss:
    """
    Factory function to create loss function from configuration.
    
    Args:
        config: Loss configuration dictionary
        
    Returns:
        Configured loss function
    """
    loss_config = config.get('loss', {})
    
    return OCRLoss(
        loss_type=loss_config.get('type', 'crossentropy'),
        pad_token_id=loss_config.get('pad_token_id', 11),
        blank_token_id=loss_config.get('blank_token_id', 0),
        label_smoothing=loss_config.get('label_smoothing', 0.0),
        focal_alpha=loss_config.get('focal_alpha', 1.0),
        focal_gamma=loss_config.get('focal_gamma', 0.0),
        reduction=loss_config.get('reduction', 'mean')
    ) 