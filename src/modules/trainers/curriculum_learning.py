"""
Curriculum Learning for Khmer OCR Training

This module implements curriculum learning strategies for progressive complexity training,
advancing from simple single characters to complex multi-character sequences and full text.
"""

import torch
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import logging
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


class DifficultyLevel(Enum):
    """Difficulty levels for curriculum learning."""
    SINGLE_CHAR = 1       # Single character recognition
    SIMPLE_COMBO = 2      # Simple character combinations (consonant + vowel)
    COMPLEX_COMBO = 3     # Complex combinations (stacked consonants, diacritics)
    WORD_LEVEL = 4        # Word-level recognition
    MULTI_WORD = 5        # Multi-word and sentence recognition


@dataclass
class CurriculumStage:
    """Configuration for a curriculum learning stage."""
    level: DifficultyLevel
    name: str
    description: str
    min_epochs: int = 5
    max_epochs: int = 20
    success_threshold: float = 0.85
    sample_filters: List[str] = None
    data_augmentation_strength: float = 0.3
    batch_size_multiplier: float = 1.0
    learning_rate_multiplier: float = 1.0
    
    def __post_init__(self):
        if self.sample_filters is None:
            self.sample_filters = []


class CurriculumLearningManager:
    """
    Manages curriculum learning progression through difficulty stages.
    
    This class implements progressive training strategies that start with simple
    recognition tasks and gradually increase complexity as the model improves.
    """
    
    def __init__(self,
                 stages: List[CurriculumStage] = None,
                 auto_progression: bool = True,
                 min_improvement_epochs: int = 3,
                 progression_patience: int = 5):
        """
        Initialize curriculum learning manager.
        
        Args:
            stages: List of curriculum stages (uses default if None)
            auto_progression: Whether to automatically progress between stages
            min_improvement_epochs: Minimum epochs before considering progression
            progression_patience: Patience for stage progression
        """
        self.stages = stages or self._create_default_stages()
        self.auto_progression = auto_progression
        self.min_improvement_epochs = min_improvement_epochs
        self.progression_patience = progression_patience
        
        # Current state
        self.current_stage_idx = 0
        self.stage_epochs = 0
        self.stage_history = []
        self.best_stage_performance = 0.0
        self.epochs_without_improvement = 0
        
        logger.info(f"Curriculum learning initialized with {len(self.stages)} stages")
        logger.info(f"Auto progression: {auto_progression}")
    
    def _create_default_stages(self) -> List[CurriculumStage]:
        """Create default curriculum stages for Khmer OCR."""
        return [
            CurriculumStage(
                level=DifficultyLevel.SINGLE_CHAR,
                name="Single Character",
                description="Single character recognition (digits and simple characters)",
                min_epochs=3,
                max_epochs=15,
                success_threshold=0.90,
                sample_filters=["sequence_length=1"],
                data_augmentation_strength=0.2,
                learning_rate_multiplier=1.0
            ),
            CurriculumStage(
                level=DifficultyLevel.SIMPLE_COMBO,
                name="Simple Combinations",
                description="Simple character combinations (consonant + vowel)",
                min_epochs=5,
                max_epochs=20,
                success_threshold=0.85,
                sample_filters=["sequence_length<=3", "complexity_level<=2"],
                data_augmentation_strength=0.3,
                learning_rate_multiplier=0.8
            ),
            CurriculumStage(
                level=DifficultyLevel.COMPLEX_COMBO,
                name="Complex Combinations",
                description="Complex combinations with stacked consonants and diacritics",
                min_epochs=8,
                max_epochs=25,
                success_threshold=0.80,
                sample_filters=["sequence_length<=5", "complexity_level<=4"],
                data_augmentation_strength=0.4,
                learning_rate_multiplier=0.6
            ),
            CurriculumStage(
                level=DifficultyLevel.WORD_LEVEL,
                name="Word Level",
                description="Complete word recognition with proper spacing",
                min_epochs=10,
                max_epochs=30,
                success_threshold=0.75,
                sample_filters=["sequence_length<=8"],
                data_augmentation_strength=0.5,
                learning_rate_multiplier=0.5
            ),
            CurriculumStage(
                level=DifficultyLevel.MULTI_WORD,
                name="Multi-word",
                description="Multi-word and sentence recognition",
                min_epochs=15,
                max_epochs=40,
                success_threshold=0.70,
                sample_filters=[],  # No filters - use all data
                data_augmentation_strength=0.6,
                learning_rate_multiplier=0.4
            )
        ]
    
    @property
    def current_stage(self) -> CurriculumStage:
        """Get current curriculum stage."""
        return self.stages[self.current_stage_idx]
    
    @property
    def is_final_stage(self) -> bool:
        """Check if current stage is the final stage."""
        return self.current_stage_idx >= len(self.stages) - 1
    
    def should_progress(self, performance: float, epoch: int) -> bool:
        """
        Check if we should progress to the next stage.
        
        Args:
            performance: Current performance metric (accuracy)
            epoch: Current epoch within stage
            
        Returns:
            True if should progress to next stage
        """
        if self.is_final_stage:
            return False
        
        stage = self.current_stage
        
        # Must meet minimum epochs
        if self.stage_epochs < stage.min_epochs:
            return False
        
        # Check if performance threshold is met
        threshold_met = performance >= stage.success_threshold
        
        # Check maximum epochs
        max_epochs_reached = self.stage_epochs >= stage.max_epochs
        
        # Auto progression logic
        if self.auto_progression:
            # Track improvement
            if performance > self.best_stage_performance:
                self.best_stage_performance = performance
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
            
            # Progress if threshold met and minimum improvement time passed
            if threshold_met and self.stage_epochs >= self.min_improvement_epochs:
                return True
            
            # Progress if no improvement for too long
            if self.epochs_without_improvement >= self.progression_patience:
                logger.info(f"Progressing due to lack of improvement ({self.epochs_without_improvement} epochs)")
                return True
            
            # Force progression at max epochs
            if max_epochs_reached:
                logger.info(f"Progressing due to max epochs reached ({stage.max_epochs})")
                return True
        
        return False
    
    def progress_to_next_stage(self) -> bool:
        """
        Progress to the next curriculum stage.
        
        Returns:
            True if progressed successfully, False if already at final stage
        """
        if self.is_final_stage:
            logger.warning("Already at final curriculum stage")
            return False
        
        # Record current stage completion
        stage_info = {
            'stage_idx': self.current_stage_idx,
            'stage_name': self.current_stage.name,
            'epochs_completed': self.stage_epochs,
            'best_performance': self.best_stage_performance,
            'final_performance': self.best_stage_performance
        }
        self.stage_history.append(stage_info)
        
        # Progress to next stage
        self.current_stage_idx += 1
        self.stage_epochs = 0
        self.best_stage_performance = 0.0
        self.epochs_without_improvement = 0
        
        new_stage = self.current_stage
        logger.info(f"Progressed to curriculum stage {self.current_stage_idx + 1}: {new_stage.name}")
        logger.info(f"Stage description: {new_stage.description}")
        logger.info(f"Target success threshold: {new_stage.success_threshold:.1%}")
        
        return True
    
    def update_epoch(self, performance: float) -> Dict[str, Any]:
        """
        Update curriculum state for new epoch.
        
        Args:
            performance: Current epoch performance
            
        Returns:
            Curriculum update information
        """
        self.stage_epochs += 1
        
        # Check for progression
        should_progress = self.should_progress(performance, self.stage_epochs)
        
        update_info = {
            'current_stage_idx': self.current_stage_idx,
            'current_stage_name': self.current_stage.name,
            'stage_epochs': self.stage_epochs,
            'stage_performance': performance,
            'best_stage_performance': max(self.best_stage_performance, performance),
            'should_progress': should_progress,
            'is_final_stage': self.is_final_stage,
            'progress_reason': None
        }
        
        # Handle progression
        if should_progress:
            if performance >= self.current_stage.success_threshold:
                update_info['progress_reason'] = 'threshold_met'
            elif self.stage_epochs >= self.current_stage.max_epochs:
                update_info['progress_reason'] = 'max_epochs'
            elif self.epochs_without_improvement >= self.progression_patience:
                update_info['progress_reason'] = 'no_improvement'
            
            self.progress_to_next_stage()
        
        return update_info
    
    def get_data_loader_config(self) -> Dict[str, Any]:
        """
        Get data loader configuration for current stage.
        
        Returns:
            Configuration for filtering and loading data
        """
        stage = self.current_stage
        
        return {
            'sample_filters': stage.sample_filters,
            'batch_size_multiplier': stage.batch_size_multiplier,
            'augmentation_strength': stage.data_augmentation_strength,
            'stage_name': stage.name,
            'difficulty_level': stage.level.value
        }
    
    def get_training_config_adjustments(self) -> Dict[str, Any]:
        """
        Get training configuration adjustments for current stage.
        
        Returns:
            Configuration adjustments for current stage
        """
        stage = self.current_stage
        
        return {
            'learning_rate_multiplier': stage.learning_rate_multiplier,
            'batch_size_multiplier': stage.batch_size_multiplier,
            'augmentation_strength': stage.data_augmentation_strength,
            'min_epochs': stage.min_epochs,
            'max_epochs': stage.max_epochs,
            'success_threshold': stage.success_threshold
        }
    
    def get_curriculum_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive curriculum learning summary.
        
        Returns:
            Summary of curriculum state and history
        """
        return {
            'total_stages': len(self.stages),
            'current_stage_idx': self.current_stage_idx,
            'current_stage': {
                'name': self.current_stage.name,
                'description': self.current_stage.description,
                'level': self.current_stage.level.value,
                'epochs_completed': self.stage_epochs,
                'success_threshold': self.current_stage.success_threshold
            },
            'stage_history': self.stage_history,
            'is_final_stage': self.is_final_stage,
            'auto_progression': self.auto_progression
        }
    
    def save_curriculum_state(self, save_path: str):
        """Save curriculum state to file."""
        state = {
            'current_stage_idx': self.current_stage_idx,
            'stage_epochs': self.stage_epochs,
            'stage_history': self.stage_history,
            'best_stage_performance': self.best_stage_performance,
            'epochs_without_improvement': self.epochs_without_improvement,
            'auto_progression': self.auto_progression
        }
        
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save(state, save_path)
        logger.info(f"Curriculum state saved to {save_path}")
    
    def load_curriculum_state(self, load_path: str):
        """Load curriculum state from file."""
        state = torch.load(load_path, map_location='cpu')
        
        self.current_stage_idx = state['current_stage_idx']
        self.stage_epochs = state['stage_epochs']
        self.stage_history = state['stage_history']
        self.best_stage_performance = state['best_stage_performance']
        self.epochs_without_improvement = state['epochs_without_improvement']
        self.auto_progression = state['auto_progression']
        
        logger.info(f"Curriculum state loaded from {load_path}")
        logger.info(f"Resumed at stage {self.current_stage_idx + 1}: {self.current_stage.name}") 