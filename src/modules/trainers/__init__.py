"""
Training Infrastructure for Khmer OCR

This module provides comprehensive training infrastructure including:
- Base trainer classes with common functionality
- Specialized OCR trainer for sequence-to-sequence tasks
- Loss functions optimized for OCR tasks
- Evaluation metrics (character accuracy, sequence accuracy, edit distance)
- TensorBoard logging and model checkpointing
- Learning rate scheduling and early stopping
- Curriculum learning for progressive training
- Multi-task learning capabilities
- Enhanced loss functions for hierarchical character structure
- Advanced learning rate schedulers
- Gradient accumulation and mixed precision training
"""

from .base_trainer import BaseTrainer
from .ocr_trainer import OCRTrainer
from .losses import OCRLoss, CTCLoss, CrossEntropyLoss
from .metrics import OCRMetrics, calculate_character_accuracy, calculate_sequence_accuracy
from .utils import (
    TrainingConfig,
    CheckpointManager,
    EarlyStopping,
    setup_training_environment,
    save_training_config
)

# Advanced training infrastructure
from .curriculum_learning import (
    CurriculumLearningManager,
    CurriculumStage,
    DifficultyLevel
)
from .multi_task_learning import (
    MultiTaskTrainer,
    MultiTaskLoss,
    TaskConfig,
    TaskType,
    FocalLoss
)
from .enhanced_losses import (
    HierarchicalLoss,
    ConfidenceAwareLoss,
    CurriculumLoss,
    OnlineHardExampleMining,
    DistillationLoss
)
from .advanced_schedulers import (
    WarmupCosineAnnealingLR,
    CurriculumAwareLR,
    AdaptiveLR,
    GradualWarmupScheduler,
    SchedulerFactory
)
from .advanced_training_utils import (
    GradientAccumulator,
    MixedPrecisionManager,
    EnhancedCheckpointManager
)

__all__ = [
    # Base trainers
    'BaseTrainer',
    'OCRTrainer',
    
    # Loss functions
    'OCRLoss',
    'CTCLoss', 
    'CrossEntropyLoss',
    
    # Enhanced loss functions
    'HierarchicalLoss',
    'ConfidenceAwareLoss',
    'CurriculumLoss',
    'OnlineHardExampleMining',
    'DistillationLoss',
    
    # Metrics
    'OCRMetrics',
    'calculate_character_accuracy',
    'calculate_sequence_accuracy',
    
    # Basic utilities
    'TrainingConfig',
    'CheckpointManager',
    'EarlyStopping',
    'setup_training_environment',
    'save_training_config',
    
    # Curriculum learning
    'CurriculumLearningManager',
    'CurriculumStage',
    'DifficultyLevel',
    
    # Multi-task learning
    'MultiTaskTrainer',
    'MultiTaskLoss',
    'TaskConfig',
    'TaskType',
    'FocalLoss',
    
    # Advanced schedulers
    'WarmupCosineAnnealingLR',
    'CurriculumAwareLR',
    'AdaptiveLR',
    'GradualWarmupScheduler',
    'SchedulerFactory',
    
    # Advanced training utilities
    'GradientAccumulator',
    'MixedPrecisionManager',
    'EnhancedCheckpointManager'
]

__version__ = "2.0.0" 