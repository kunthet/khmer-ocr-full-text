"""
Multi-Task Learning for Khmer OCR Training

This module implements multi-task learning capabilities for training models
on multiple related objectives simultaneously, including character recognition,
word-level recognition, and confidence prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass
from enum import Enum
import logging
import numpy as np

logger = logging.getLogger(__name__)


class TaskType(Enum):
    """Types of tasks for multi-task learning."""
    CHARACTER_RECOGNITION = "char_recognition"
    WORD_RECOGNITION = "word_recognition"
    CONFIDENCE_PREDICTION = "confidence_prediction"
    SEQUENCE_SEGMENTATION = "sequence_segmentation"
    HIERARCHICAL_CLASSIFICATION = "hierarchical_classification"
    LANGUAGE_MODELING = "language_modeling"


@dataclass
class TaskConfig:
    """Configuration for a specific task in multi-task learning."""
    task_type: TaskType
    name: str
    weight: float = 1.0
    loss_type: str = "crossentropy"
    metrics: List[str] = None
    output_size: int = None
    label_smoothing: float = 0.0
    class_weights: Optional[torch.Tensor] = None
    enabled: bool = True
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = []


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss function that combines losses from multiple tasks.
    
    Supports various loss combination strategies including fixed weighting,
    adaptive weighting, and uncertainty-based weighting.
    """
    
    def __init__(self,
                 task_configs: List[TaskConfig],
                 weighting_strategy: str = "fixed",
                 uncertainty_weighting: bool = False,
                 gradient_normalization: bool = False):
        """
        Initialize multi-task loss.
        
        Args:
            task_configs: List of task configurations
            weighting_strategy: Strategy for combining losses ("fixed", "adaptive", "uncertainty")
            uncertainty_weighting: Whether to use uncertainty-based weighting
            gradient_normalization: Whether to normalize gradients across tasks
        """
        super().__init__()
        self.task_configs = {config.name: config for config in task_configs}
        self.weighting_strategy = weighting_strategy
        self.uncertainty_weighting = uncertainty_weighting
        self.gradient_normalization = gradient_normalization
        
        # Create loss functions for each task
        self.task_losses = nn.ModuleDict()
        for config in task_configs:
            if config.enabled:
                self.task_losses[config.name] = self._create_task_loss(config)
        
        # Initialize adaptive weights if needed
        if weighting_strategy == "adaptive":
            self.task_weights = nn.Parameter(torch.ones(len(task_configs)))
        elif uncertainty_weighting:
            # Learnable uncertainty parameters (log variance)
            self.log_vars = nn.Parameter(torch.zeros(len(task_configs)))
        
        # Track loss history for adaptive weighting
        self.loss_history = {name: [] for name in self.task_losses.keys()}
        
    def _create_task_loss(self, config: TaskConfig) -> nn.Module:
        """Create loss function for a specific task."""
        if config.loss_type == "crossentropy":
            return nn.CrossEntropyLoss(
                weight=config.class_weights,
                label_smoothing=config.label_smoothing,
                reduction='mean'
            )
        elif config.loss_type == "mse":
            return nn.MSELoss(reduction='mean')
        elif config.loss_type == "bce":
            return nn.BCEWithLogitsLoss(reduction='mean')
        elif config.loss_type == "focal":
            return FocalLoss(alpha=1.0, gamma=2.0)
        else:
            raise ValueError(f"Unsupported loss type: {config.loss_type}")
    
    def forward(self,
                predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Calculate multi-task loss.
        
        Args:
            predictions: Dictionary of task predictions
            targets: Dictionary of task targets
            
        Returns:
            Dictionary containing individual and combined losses
        """
        task_losses = {}
        total_loss = 0.0
        
        # Calculate individual task losses
        for task_name, loss_fn in self.task_losses.items():
            if task_name in predictions and task_name in targets:
                pred = predictions[task_name]
                target = targets[task_name]
                
                # Calculate task-specific loss
                task_loss = loss_fn(pred, target)
                task_losses[task_name] = task_loss
                
                # Update loss history
                self.loss_history[task_name].append(task_loss.item())
                if len(self.loss_history[task_name]) > 100:  # Keep only recent history
                    self.loss_history[task_name] = self.loss_history[task_name][-100:]
        
        # Combine losses based on weighting strategy
        if self.weighting_strategy == "fixed":
            total_loss = self._fixed_weighting(task_losses)
        elif self.weighting_strategy == "adaptive":
            total_loss = self._adaptive_weighting(task_losses)
        elif self.uncertainty_weighting:
            total_loss = self._uncertainty_weighting(task_losses)
        else:
            total_loss = sum(task_losses.values())
        
        # Add total loss to results
        task_losses['total_loss'] = total_loss
        
        return task_losses
    
    def _fixed_weighting(self, task_losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Combine losses using fixed weights."""
        total_loss = 0.0
        for task_name, loss in task_losses.items():
            config = self.task_configs[task_name]
            total_loss += config.weight * loss
        return total_loss
    
    def _adaptive_weighting(self, task_losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Combine losses using adaptive weights."""
        # Softmax normalization of learnable weights
        weights = F.softmax(self.task_weights, dim=0)
        
        total_loss = 0.0
        for i, (task_name, loss) in enumerate(task_losses.items()):
            total_loss += weights[i] * loss
        
        return total_loss
    
    def _uncertainty_weighting(self, task_losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Combine losses using uncertainty-based weighting."""
        total_loss = 0.0
        
        for i, (task_name, loss) in enumerate(task_losses.items()):
            # Uncertainty weighting: L = (1/2σ²)L_task + log(σ)
            precision = torch.exp(-self.log_vars[i])
            total_loss += precision * loss + self.log_vars[i]
        
        return total_loss
    
    def get_task_weights(self) -> Dict[str, float]:
        """Get current task weights."""
        if self.weighting_strategy == "adaptive":
            weights = F.softmax(self.task_weights, dim=0)
            return {name: weights[i].item() for i, name in enumerate(self.task_losses.keys())}
        elif self.uncertainty_weighting:
            precisions = torch.exp(-self.log_vars)
            total_precision = precisions.sum()
            weights = precisions / total_precision
            return {name: weights[i].item() for i, name in enumerate(self.task_losses.keys())}
        else:
            return {name: self.task_configs[name].weight for name in self.task_losses.keys()}


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance."""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class MultiTaskTrainer:
    """
    Trainer extension for multi-task learning.
    
    This class provides additional functionality for training models
    with multiple objectives simultaneously.
    """
    
    def __init__(self,
                 task_configs: List[TaskConfig],
                 base_trainer,
                 gradient_clipping: bool = True,
                 gradient_clip_norm: float = 1.0):
        """
        Initialize multi-task trainer.
        
        Args:
            task_configs: List of task configurations
            base_trainer: Base trainer instance
            gradient_clipping: Whether to apply gradient clipping
            gradient_clip_norm: Gradient clipping norm
        """
        self.task_configs = {config.name: config for config in task_configs}
        self.base_trainer = base_trainer
        self.gradient_clipping = gradient_clipping
        self.gradient_clip_norm = gradient_clip_norm
        
        # Create multi-task loss
        self.multi_task_loss = MultiTaskLoss(task_configs)
        
        # Task-specific metrics
        self.task_metrics = {}
        for config in task_configs:
            if config.enabled:
                self.task_metrics[config.name] = self._create_task_metrics(config)
        
        # Training statistics
        self.task_performance_history = {name: [] for name in self.task_metrics.keys()}
        
        logger.info(f"Multi-task trainer initialized with {len(task_configs)} tasks")
        enabled_tasks = [c.name for c in task_configs if c.enabled]
        logger.info(f"Enabled tasks: {enabled_tasks}")
    
    def _create_task_metrics(self, config: TaskConfig) -> Dict[str, Callable]:
        """Create metrics for a specific task."""
        metrics = {}
        
        for metric_name in config.metrics:
            if metric_name == "accuracy":
                metrics[metric_name] = self._accuracy_metric
            elif metric_name == "f1_score":
                metrics[metric_name] = self._f1_score_metric
            elif metric_name == "mae":
                metrics[metric_name] = self._mae_metric
            elif metric_name == "rmse":
                metrics[metric_name] = self._rmse_metric
        
        return metrics
    
    def _accuracy_metric(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Calculate accuracy metric."""
        pred_classes = torch.argmax(predictions, dim=-1)
        correct = (pred_classes == targets).float()
        return correct.mean().item()
    
    def _f1_score_metric(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Calculate F1 score metric."""
        pred_classes = torch.argmax(predictions, dim=-1)
        # Simplified F1 calculation for multi-class
        tp = ((pred_classes == targets) & (targets > 0)).sum().float()
        fp = ((pred_classes != targets) & (pred_classes > 0)).sum().float()
        fn = ((pred_classes != targets) & (targets > 0)).sum().float()
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        return f1.item()
    
    def _mae_metric(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Calculate Mean Absolute Error."""
        return F.l1_loss(predictions, targets).item()
    
    def _rmse_metric(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Calculate Root Mean Square Error."""
        return torch.sqrt(F.mse_loss(predictions, targets)).item()
    
    def calculate_task_metrics(self,
                             predictions: Dict[str, torch.Tensor],
                             targets: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, float]]:
        """Calculate metrics for all tasks."""
        task_results = {}
        
        for task_name, metrics in self.task_metrics.items():
            if task_name in predictions and task_name in targets:
                task_results[task_name] = {}
                pred = predictions[task_name]
                target = targets[task_name]
                
                for metric_name, metric_fn in metrics.items():
                    try:
                        value = metric_fn(pred, target)
                        task_results[task_name][metric_name] = value
                    except Exception as e:
                        logger.warning(f"Error calculating {metric_name} for {task_name}: {e}")
                        task_results[task_name][metric_name] = 0.0
        
        return task_results
    
    def update_task_performance(self, task_metrics: Dict[str, Dict[str, float]]):
        """Update task performance history."""
        for task_name, metrics in task_metrics.items():
            if task_name in self.task_performance_history:
                self.task_performance_history[task_name].append(metrics)
                # Keep only recent history
                if len(self.task_performance_history[task_name]) > 1000:
                    self.task_performance_history[task_name] = self.task_performance_history[task_name][-1000:]
    
    def get_task_balance_report(self) -> Dict[str, Any]:
        """Generate task balance report."""
        report = {
            'task_weights': self.multi_task_loss.get_task_weights(),
            'task_performance': {},
            'task_trends': {}
        }
        
        # Calculate recent performance for each task
        for task_name, history in self.task_performance_history.items():
            if history:
                recent_metrics = history[-10:]  # Last 10 epochs
                if recent_metrics and 'accuracy' in recent_metrics[0]:
                    accuracies = [m['accuracy'] for m in recent_metrics]
                    report['task_performance'][task_name] = {
                        'mean_accuracy': np.mean(accuracies),
                        'std_accuracy': np.std(accuracies),
                        'recent_accuracy': accuracies[-1] if accuracies else 0.0
                    }
                    
                    # Calculate trend (improvement over last epochs)
                    if len(accuracies) >= 5:
                        early_acc = np.mean(accuracies[:3])
                        late_acc = np.mean(accuracies[-3:])
                        report['task_trends'][task_name] = late_acc - early_acc
        
        return report
    
    def should_rebalance_tasks(self, 
                             task_metrics: Dict[str, Dict[str, float]],
                             rebalance_threshold: float = 0.1) -> bool:
        """
        Check if task weights should be rebalanced.
        
        Args:
            task_metrics: Current task metrics
            rebalance_threshold: Threshold for performance imbalance
            
        Returns:
            True if tasks should be rebalanced
        """
        if len(task_metrics) < 2:
            return False
        
        # Get accuracy scores for comparison
        accuracies = []
        for task_name, metrics in task_metrics.items():
            if 'accuracy' in metrics:
                accuracies.append(metrics['accuracy'])
        
        if len(accuracies) < 2:
            return False
        
        # Check for significant imbalance
        max_acc = max(accuracies)
        min_acc = min(accuracies)
        imbalance = max_acc - min_acc
        
        return imbalance > rebalance_threshold
    
    def get_multi_task_summary(self) -> Dict[str, Any]:
        """Get comprehensive multi-task learning summary."""
        return {
            'total_tasks': len(self.task_configs),
            'enabled_tasks': [name for name, config in self.task_configs.items() if config.enabled],
            'task_weights': self.multi_task_loss.get_task_weights(),
            'weighting_strategy': self.multi_task_loss.weighting_strategy,
            'gradient_clipping': self.gradient_clipping,
            'task_configs': {name: {
                'type': config.task_type.value,
                'weight': config.weight,
                'loss_type': config.loss_type,
                'enabled': config.enabled
            } for name, config in self.task_configs.items()}
        } 