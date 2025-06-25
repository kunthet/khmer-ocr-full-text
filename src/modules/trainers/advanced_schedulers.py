"""
Advanced Learning Rate Schedulers for Khmer OCR Training

This module provides advanced learning rate scheduling strategies optimized
for longer training runs and curriculum learning.
"""

import torch
import math
from typing import Dict, List, Any, Optional, Union
from torch.optim.lr_scheduler import _LRScheduler
import logging

logger = logging.getLogger(__name__)


class WarmupCosineAnnealingLR(_LRScheduler):
    """
    Learning rate scheduler with warmup and cosine annealing.
    
    This scheduler combines:
    1. Linear warmup phase
    2. Cosine annealing decay
    3. Optional restart capability
    """
    
    def __init__(self,
                 optimizer,
                 warmup_epochs: int,
                 max_epochs: int,
                 min_lr_ratio: float = 0.01,
                 restart_epochs: Optional[int] = None,
                 last_epoch: int = -1):
        """
        Initialize warmup cosine annealing scheduler.
        
        Args:
            optimizer: Optimizer to schedule
            warmup_epochs: Number of warmup epochs
            max_epochs: Total number of training epochs
            min_lr_ratio: Minimum learning rate as ratio of base LR
            restart_epochs: Epochs for cosine restart (optional)
            last_epoch: Last epoch number
        """
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr_ratio = min_lr_ratio
        self.restart_epochs = restart_epochs
        
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        """Calculate learning rate for current epoch."""
        if self.last_epoch < self.warmup_epochs:
            # Warmup phase: linear increase
            lr_scale = (self.last_epoch + 1) / self.warmup_epochs
            return [base_lr * lr_scale for base_lr in self.base_lrs]
        
        else:
            # Cosine annealing phase
            cosine_epochs = self.max_epochs - self.warmup_epochs
            current_cosine_epoch = self.last_epoch - self.warmup_epochs
            
            if self.restart_epochs and current_cosine_epoch > 0:
                # Cosine restart
                current_cosine_epoch = current_cosine_epoch % self.restart_epochs
                cosine_epochs = self.restart_epochs
            
            # Cosine annealing formula
            cosine_factor = 0.5 * (1 + math.cos(math.pi * current_cosine_epoch / cosine_epochs))
            lr_scale = self.min_lr_ratio + (1 - self.min_lr_ratio) * cosine_factor
            
            return [base_lr * lr_scale for base_lr in self.base_lrs]


class CurriculumAwareLR(_LRScheduler):
    """
    Curriculum-aware learning rate scheduler.
    
    This scheduler adjusts learning rate based on curriculum learning stages,
    typically reducing LR as complexity increases.
    """
    
    def __init__(self,
                 optimizer,
                 stage_lr_multipliers: List[float] = None,
                 stage_transitions: List[int] = None,
                 base_decay: float = 0.1,
                 last_epoch: int = -1):
        """
        Initialize curriculum-aware scheduler.
        
        Args:
            optimizer: Optimizer to schedule
            stage_lr_multipliers: LR multipliers for each curriculum stage
            stage_transitions: Epoch numbers for stage transitions
            base_decay: Base decay factor between stages
            last_epoch: Last epoch number
        """
        self.stage_lr_multipliers = stage_lr_multipliers or [1.0, 0.8, 0.6, 0.4, 0.2]
        self.stage_transitions = stage_transitions or [10, 25, 45, 70, 100]
        self.base_decay = base_decay
        self.current_stage = 0
        
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        """Calculate learning rate based on curriculum stage."""
        # Determine current curriculum stage
        for i, transition_epoch in enumerate(self.stage_transitions):
            if self.last_epoch < transition_epoch:
                self.current_stage = i
                break
        else:
            self.current_stage = len(self.stage_transitions)
        
        # Get stage multiplier
        if self.current_stage < len(self.stage_lr_multipliers):
            stage_multiplier = self.stage_lr_multipliers[self.current_stage]
        else:
            # Beyond defined stages, use minimum multiplier
            stage_multiplier = self.stage_lr_multipliers[-1]
        
        return [base_lr * stage_multiplier for base_lr in self.base_lrs]
    
    def update_stage_transition(self, new_transition_epoch: int):
        """Update stage transition dynamically."""
        if self.current_stage < len(self.stage_transitions):
            self.stage_transitions[self.current_stage] = new_transition_epoch
            logger.info(f"Updated stage {self.current_stage} transition to epoch {new_transition_epoch}")


class AdaptiveLR(_LRScheduler):
    """
    Adaptive learning rate scheduler based on performance metrics.
    
    This scheduler adjusts learning rate based on training/validation performance,
    reducing LR when performance plateaus.
    """
    
    def __init__(self,
                 optimizer,
                 metric_name: str = 'val_loss',
                 mode: str = 'min',
                 patience: int = 5,
                 factor: float = 0.5,
                 min_lr: float = 1e-6,
                 threshold: float = 1e-4,
                 last_epoch: int = -1):
        """
        Initialize adaptive scheduler.
        
        Args:
            optimizer: Optimizer to schedule
            metric_name: Name of metric to monitor
            mode: 'min' for loss, 'max' for accuracy
            patience: Epochs to wait before reducing LR
            factor: Factor to multiply LR by
            min_lr: Minimum learning rate
            threshold: Minimum change threshold
            last_epoch: Last epoch number
        """
        self.metric_name = metric_name
        self.mode = mode
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.threshold = threshold
        
        # Tracking variables
        self.best_metric = float('inf') if mode == 'min' else float('-inf')
        self.epochs_without_improvement = 0
        self.metric_history = []
        
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        """Get current learning rates."""
        return [max(lr, self.min_lr) for lr in self.optimizer.param_groups[0]['lr']]
    
    def step(self, metrics: Dict[str, float] = None, epoch=None):
        """
        Step scheduler with performance metrics.
        
        Args:
            metrics: Dictionary containing performance metrics
            epoch: Current epoch (for compatibility)
        """
        if metrics is None:
            logger.warning("No metrics provided to AdaptiveLR.step()")
            return
        if self.metric_name not in metrics:
            logger.warning(f"Metric '{self.metric_name}' not found in metrics")
            return
        
        current_metric = metrics[self.metric_name]
        self.metric_history.append(current_metric)
        
        # Check for improvement
        is_improvement = False
        if self.mode == 'min':
            is_improvement = current_metric < self.best_metric - self.threshold
        else:
            is_improvement = current_metric > self.best_metric + self.threshold
        
        if is_improvement:
            self.best_metric = current_metric
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
        
        # Reduce LR if no improvement for patience epochs
        if self.epochs_without_improvement >= self.patience:
            for param_group in self.optimizer.param_groups:
                new_lr = max(param_group['lr'] * self.factor, self.min_lr)
                param_group['lr'] = new_lr
            
            logger.info(f"Reduced learning rate to {new_lr:.2e} after {self.patience} epochs without improvement")
            self.epochs_without_improvement = 0
        
        self.last_epoch += 1


class GradualWarmupScheduler(_LRScheduler):
    """
    Gradual warmup scheduler that works with any base scheduler.
    
    This scheduler provides a warmup phase followed by the base scheduler.
    """
    
    def __init__(self,
                 optimizer,
                 multiplier: float,
                 total_warmup_epochs: int,
                 after_scheduler: _LRScheduler = None,
                 last_epoch: int = -1):
        """
        Initialize gradual warmup scheduler.
        
        Args:
            optimizer: Optimizer to schedule
            multiplier: Target multiplier for warmup
            total_warmup_epochs: Total warmup epochs
            after_scheduler: Scheduler to use after warmup
            last_epoch: Last epoch number
        """
        self.multiplier = multiplier
        self.total_warmup_epochs = total_warmup_epochs
        self.after_scheduler = after_scheduler
        self.finished_warmup = False
        
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        """Calculate learning rate with warmup."""
        if self.last_epoch < self.total_warmup_epochs:
            # Warmup phase
            warmup_factor = (self.last_epoch + 1) / self.total_warmup_epochs
            warmup_factor = warmup_factor * (self.multiplier - 1) + 1
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        
        else:
            # After warmup
            if not self.finished_warmup:
                self.finished_warmup = True
                if self.after_scheduler:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
            
            if self.after_scheduler:
                return self.after_scheduler.get_lr()
            else:
                return [base_lr * self.multiplier for base_lr in self.base_lrs]
    
    def step(self, epoch=None, metrics=None):
        """Step the scheduler."""
        if self.finished_warmup and self.after_scheduler:
            if hasattr(self.after_scheduler, 'step'):
                if isinstance(self.after_scheduler, AdaptiveLR):
                    self.after_scheduler.step(metrics, epoch)
                else:
                    self.after_scheduler.step(epoch)
        
        super().step(epoch)


class SchedulerFactory:
    """Factory for creating learning rate schedulers."""
    
    @staticmethod
    def create_scheduler(optimizer,
                        scheduler_config: Dict[str, Any]) -> _LRScheduler:
        """
        Create learning rate scheduler from configuration.
        
        Args:
            optimizer: Optimizer instance
            scheduler_config: Scheduler configuration
            
        Returns:
            Configured scheduler
        """
        scheduler_type = scheduler_config.get('type', 'cosine_warmup')
        
        if scheduler_type == 'cosine_warmup':
            return WarmupCosineAnnealingLR(
                optimizer,
                warmup_epochs=scheduler_config.get('warmup_epochs', 5),
                max_epochs=scheduler_config.get('max_epochs', 100),
                min_lr_ratio=scheduler_config.get('min_lr_ratio', 0.01),
                restart_epochs=scheduler_config.get('restart_epochs', None)
            )
        
        elif scheduler_type == 'curriculum':
            return CurriculumAwareLR(
                optimizer,
                stage_lr_multipliers=scheduler_config.get('stage_multipliers', [1.0, 0.8, 0.6, 0.4, 0.2]),
                stage_transitions=scheduler_config.get('stage_transitions', [10, 25, 45, 70, 100]),
                base_decay=scheduler_config.get('base_decay', 0.1)
            )
        
        elif scheduler_type == 'adaptive':
            return AdaptiveLR(
                optimizer,
                metric_name=scheduler_config.get('metric_name', 'val_loss'),
                mode=scheduler_config.get('mode', 'min'),
                patience=scheduler_config.get('patience', 5),
                factor=scheduler_config.get('factor', 0.5),
                min_lr=scheduler_config.get('min_lr', 1e-6)
            )
        
        elif scheduler_type == 'warmup':
            base_scheduler_config = scheduler_config.get('base_scheduler', {})
            base_scheduler = None
            
            if base_scheduler_config:
                base_scheduler = SchedulerFactory.create_scheduler(optimizer, base_scheduler_config)
            
            return GradualWarmupScheduler(
                optimizer,
                multiplier=scheduler_config.get('multiplier', 1.0),
                total_warmup_epochs=scheduler_config.get('warmup_epochs', 5),
                after_scheduler=base_scheduler
            )
        
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
    
    @staticmethod
    def get_scheduler_info(scheduler: _LRScheduler) -> Dict[str, Any]:
        """Get information about scheduler configuration."""
        scheduler_info = {
            'type': type(scheduler).__name__,
            'current_lr': [group['lr'] for group in scheduler.optimizer.param_groups],
            'last_epoch': scheduler.last_epoch
        }
        
        # Add scheduler-specific information
        if hasattr(scheduler, 'warmup_epochs'):
            scheduler_info['warmup_epochs'] = scheduler.warmup_epochs
        
        if hasattr(scheduler, 'max_epochs'):
            scheduler_info['max_epochs'] = scheduler.max_epochs
        
        if hasattr(scheduler, 'current_stage'):
            scheduler_info['current_stage'] = scheduler.current_stage
        
        if hasattr(scheduler, 'best_metric'):
            scheduler_info['best_metric'] = scheduler.best_metric
        
        return scheduler_info 