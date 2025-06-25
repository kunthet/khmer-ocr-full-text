"""
Enhanced Loss Functions for Khmer OCR Training

This module provides advanced loss functions optimized for hierarchical character
structure and complex Khmer text recognition.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple
import logging
import math

logger = logging.getLogger(__name__)


class HierarchicalLoss(nn.Module):
    """
    Hierarchical loss for Khmer character structure.
    
    This loss function considers the hierarchical nature of Khmer characters,
    where characters can be composed of base characters and modifiers.
    """
    
    def __init__(self,
                 base_char_weight: float = 1.0,
                 modifier_weight: float = 0.5,
                 combination_weight: float = 0.3,
                 pad_token_id: int = 11):
        """
        Initialize hierarchical loss.
        
        Args:
            base_char_weight: Weight for base character classification
            modifier_weight: Weight for modifier classification
            combination_weight: Weight for character combination correctness
            pad_token_id: Token ID for padding
        """
        super().__init__()
        self.base_char_weight = base_char_weight
        self.modifier_weight = modifier_weight
        self.combination_weight = combination_weight
        self.pad_token_id = pad_token_id
        
        # Base loss functions
        self.base_loss = nn.CrossEntropyLoss(ignore_index=pad_token_id, reduction='mean')
        self.modifier_loss = nn.CrossEntropyLoss(ignore_index=pad_token_id, reduction='mean')
        self.combination_loss = nn.MSELoss(reduction='mean')
    
    def forward(self,
                predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Calculate hierarchical loss.
        
        Args:
            predictions: Dictionary containing different prediction types
            targets: Dictionary containing different target types
            
        Returns:
            Dictionary containing loss components
        """
        total_loss = 0.0
        loss_components = {}
        
        # Base character loss
        if 'base_chars' in predictions and 'base_chars' in targets:
            base_loss = self.base_loss(predictions['base_chars'], targets['base_chars'])
            loss_components['base_loss'] = base_loss
            total_loss += self.base_char_weight * base_loss
        
        # Modifier loss
        if 'modifiers' in predictions and 'modifiers' in targets:
            modifier_loss = self.modifier_loss(predictions['modifiers'], targets['modifiers'])
            loss_components['modifier_loss'] = modifier_loss
            total_loss += self.modifier_weight * modifier_loss
        
        # Character combination loss
        if 'combinations' in predictions and 'combinations' in targets:
            combo_loss = self.combination_loss(predictions['combinations'], targets['combinations'])
            loss_components['combination_loss'] = combo_loss
            total_loss += self.combination_weight * combo_loss
        
        loss_components['total_loss'] = total_loss
        return loss_components


class ConfidenceAwareLoss(nn.Module):
    """
    Confidence-aware loss that incorporates model uncertainty.
    
    This loss function adjusts the training based on the model's confidence
    in its predictions, focusing more on uncertain predictions.
    """
    
    def __init__(self,
                 base_loss_type: str = "crossentropy",
                 confidence_weight: float = 0.2,
                 uncertainty_threshold: float = 0.5,
                 pad_token_id: int = 11):
        """
        Initialize confidence-aware loss.
        
        Args:
            base_loss_type: Base loss function type
            confidence_weight: Weight for confidence regularization
            uncertainty_threshold: Threshold for high uncertainty
            pad_token_id: Token ID for padding
        """
        super().__init__()
        self.confidence_weight = confidence_weight
        self.uncertainty_threshold = uncertainty_threshold
        
        # Base loss function
        if base_loss_type == "crossentropy":
            self.base_loss = nn.CrossEntropyLoss(ignore_index=pad_token_id, reduction='none')
        else:
            raise ValueError(f"Unsupported base loss type: {base_loss_type}")
    
    def forward(self,
                predictions: torch.Tensor,
                targets: torch.Tensor,
                confidence_scores: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Calculate confidence-aware loss.
        
        Args:
            predictions: Model predictions [batch_size, seq_len, num_classes]
            targets: Target labels [batch_size, seq_len]
            confidence_scores: Confidence scores [batch_size, seq_len] (optional)
            
        Returns:
            Dictionary containing loss components
        """
        # Calculate base loss (per sample)
        batch_size, seq_len, num_classes = predictions.shape
        predictions_flat = predictions.view(-1, num_classes)
        targets_flat = targets.view(-1)
        
        base_losses = self.base_loss(predictions_flat, targets_flat)
        base_losses = base_losses.view(batch_size, seq_len)
        
        # Calculate confidence if not provided
        if confidence_scores is None:
            with torch.no_grad():
                probs = F.softmax(predictions, dim=-1)
                confidence_scores = torch.max(probs, dim=-1)[0]
        
        # Confidence-weighted loss
        uncertainty = 1.0 - confidence_scores
        confidence_weights = 1.0 + self.confidence_weight * uncertainty
        
        # Apply weights to base loss
        weighted_losses = base_losses * confidence_weights
        
        # Mask padding tokens
        mask = (targets != 11).float()  # Assuming pad_token_id = 11
        masked_losses = weighted_losses * mask
        
        # Calculate final loss
        total_loss = masked_losses.sum() / mask.sum().clamp(min=1)
        
        # Confidence regularization (encourage calibrated confidence)
        confidence_reg = torch.mean(torch.abs(confidence_scores - torch.max(F.softmax(predictions, dim=-1), dim=-1)[0]))
        
        return {
            'base_loss': base_losses.mean(),
            'confidence_loss': confidence_reg,
            'total_loss': total_loss + self.confidence_weight * confidence_reg
        }


class CurriculumLoss(nn.Module):
    """
    Curriculum-aware loss that adjusts difficulty based on training stage.
    
    This loss function adapts the training objective based on the current
    curriculum learning stage.
    """
    
    def __init__(self,
                 base_loss_type: str = "crossentropy",
                 difficulty_weight: float = 0.3,
                 pad_token_id: int = 11):
        """
        Initialize curriculum loss.
        
        Args:
            base_loss_type: Base loss function type
            difficulty_weight: Weight for difficulty adjustment
            pad_token_id: Token ID for padding
        """
        super().__init__()
        self.difficulty_weight = difficulty_weight
        
        if base_loss_type == "crossentropy":
            self.base_loss = nn.CrossEntropyLoss(ignore_index=pad_token_id, reduction='none')
        else:
            raise ValueError(f"Unsupported base loss type: {base_loss_type}")
    
    def forward(self,
                predictions: torch.Tensor,
                targets: torch.Tensor,
                difficulty_scores: Optional[torch.Tensor] = None,
                curriculum_stage: int = 1) -> Dict[str, torch.Tensor]:
        """
        Calculate curriculum-aware loss.
        
        Args:
            predictions: Model predictions [batch_size, seq_len, num_classes]
            targets: Target labels [batch_size, seq_len]
            difficulty_scores: Sample difficulty scores [batch_size] (optional)
            curriculum_stage: Current curriculum stage (1-5)
            
        Returns:
            Dictionary containing loss components
        """
        # Calculate base loss
        batch_size, seq_len, num_classes = predictions.shape
        predictions_flat = predictions.view(-1, num_classes)
        targets_flat = targets.view(-1)
        
        base_losses = self.base_loss(predictions_flat, targets_flat)
        base_losses = base_losses.view(batch_size, seq_len)
        
        # Curriculum weighting based on stage
        stage_weight = min(1.0, curriculum_stage / 5.0)  # Progressive weighting
        
        # Difficulty-based weighting
        if difficulty_scores is not None:
            # Normalize difficulty scores
            normalized_difficulty = torch.sigmoid(difficulty_scores)
            
            # Adjust weights based on curriculum stage
            if curriculum_stage <= 2:
                # Early stages: focus on easier samples
                difficulty_weights = 2.0 - normalized_difficulty
            else:
                # Later stages: include harder samples
                difficulty_weights = 0.5 + normalized_difficulty
            
            # Apply difficulty weights
            difficulty_weights = difficulty_weights.unsqueeze(1).expand(-1, seq_len)
            weighted_losses = base_losses * difficulty_weights
        else:
            weighted_losses = base_losses
        
        # Apply stage weight
        final_losses = weighted_losses * stage_weight
        
        # Mask padding tokens
        mask = (targets != 11).float()
        masked_losses = final_losses * mask
        
        total_loss = masked_losses.sum() / mask.sum().clamp(min=1)
        
        return {
            'base_loss': base_losses.mean(),
            'curriculum_loss': total_loss,
            'stage_weight': stage_weight,
            'total_loss': total_loss
        }


class OnlineHardExampleMining(nn.Module):
    """
    Online Hard Example Mining (OHEM) loss.
    
    This loss function focuses training on the hardest examples by selecting
    the samples with highest loss values.
    """
    
    def __init__(self,
                 base_loss_type: str = "crossentropy",
                 keep_ratio: float = 0.7,
                 min_kept: int = 1,
                 pad_token_id: int = 11):
        """
        Initialize OHEM loss.
        
        Args:
            base_loss_type: Base loss function type
            keep_ratio: Ratio of samples to keep for training
            min_kept: Minimum number of samples to keep
            pad_token_id: Token ID for padding
        """
        super().__init__()
        self.keep_ratio = keep_ratio
        self.min_kept = min_kept
        
        if base_loss_type == "crossentropy":
            self.base_loss = nn.CrossEntropyLoss(ignore_index=pad_token_id, reduction='none')
        else:
            raise ValueError(f"Unsupported base loss type: {base_loss_type}")
    
    def forward(self,
                predictions: torch.Tensor,
                targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Calculate OHEM loss.
        
        Args:
            predictions: Model predictions [batch_size, seq_len, num_classes]
            targets: Target labels [batch_size, seq_len]
            
        Returns:
            Dictionary containing loss components
        """
        batch_size, seq_len, num_classes = predictions.shape
        predictions_flat = predictions.view(-1, num_classes)
        targets_flat = targets.view(-1)
        
        # Calculate per-sample losses
        losses = self.base_loss(predictions_flat, targets_flat)
        losses = losses.view(batch_size, seq_len)
        
        # Mask padding tokens
        mask = (targets != 11).float()
        masked_losses = losses * mask
        
        # Calculate sequence-level losses (mean loss per sequence)
        seq_losses = masked_losses.sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        
        # Select hard examples
        num_kept = max(self.min_kept, int(batch_size * self.keep_ratio))
        
        if num_kept < batch_size:
            # Sort by loss (descending) and keep hardest examples
            _, indices = torch.topk(seq_losses, num_kept, largest=True)
            hard_losses = seq_losses[indices]
            final_loss = hard_losses.mean()
            
            # Statistics
            kept_ratio = num_kept / batch_size
            avg_hard_loss = hard_losses.mean().item()
            avg_total_loss = seq_losses.mean().item()
        else:
            final_loss = seq_losses.mean()
            kept_ratio = 1.0
            avg_hard_loss = final_loss.item()
            avg_total_loss = final_loss.item()
        
        return {
            'base_loss': losses.mean(),
            'hard_loss': final_loss,
            'kept_ratio': kept_ratio,
            'avg_hard_loss': avg_hard_loss,
            'avg_total_loss': avg_total_loss,
            'total_loss': final_loss
        }


class DistillationLoss(nn.Module):
    """
    Knowledge distillation loss for transferring knowledge from teacher model.
    """
    
    def __init__(self,
                 temperature: float = 4.0,
                 alpha: float = 0.3,
                 student_weight: float = 0.7):
        """
        Initialize distillation loss.
        
        Args:
            temperature: Temperature for softmax distillation
            alpha: Weight for distillation loss
            student_weight: Weight for student loss
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.student_weight = student_weight
        
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self,
                student_logits: torch.Tensor,
                teacher_logits: torch.Tensor,
                targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Calculate distillation loss.
        
        Args:
            student_logits: Student model predictions
            teacher_logits: Teacher model predictions
            targets: True labels
            
        Returns:
            Dictionary containing loss components
        """
        # Student loss (standard cross-entropy)
        student_loss = self.ce_loss(student_logits, targets)
        
        # Distillation loss (KL divergence between soft targets)
        student_soft = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=-1)
        
        distillation_loss = self.kl_div(student_soft, teacher_soft) * (self.temperature ** 2)
        
        # Combined loss
        total_loss = (self.student_weight * student_loss + 
                     self.alpha * distillation_loss)
        
        return {
            'student_loss': student_loss,
            'distillation_loss': distillation_loss,
            'total_loss': total_loss
        } 