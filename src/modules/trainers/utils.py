"""
Training Utilities for Khmer Digits OCR

This module provides utility classes and functions for training infrastructure,
including configuration management, checkpointing, early stopping, and environment setup.
"""

import os
import json
import yaml
import torch
import logging
from typing import Dict, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
import shutil

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration class for training parameters."""
    
    # Model configuration
    model_name: str = "medium"
    model_config_path: str = "config/model_config.yaml"
    
    # Data configuration
    metadata_path: str = "generated_data/metadata.yaml"
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    
    # Training configuration
    num_epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0
    
    # Loss configuration
    loss_type: str = "crossentropy"
    label_smoothing: float = 0.0
    
    # Scheduler configuration
    scheduler_type: str = "steplr"
    step_size: int = 10
    gamma: float = 0.5
    
    # Early stopping
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-4
    
    # Checkpointing
    save_every_n_epochs: int = 5
    keep_n_checkpoints: int = 3
    
    # Logging
    log_every_n_steps: int = 50
    use_tensorboard: bool = True
    
    # Paths
    output_dir: str = "training_output"
    experiment_name: str = "khmer_ocr_experiment"
    
    # Device
    device: str = "auto"
    mixed_precision: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        """Create from dictionary."""
        return cls(**config_dict)
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'TrainingConfig':
        """Load from YAML file."""
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
    
    def save_yaml(self, yaml_path: str):
        """Save to YAML file."""
        os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)


class CheckpointManager:
    """Manages model checkpoints and training state."""
    
    def __init__(self,
                 checkpoint_dir: str,
                 keep_n_checkpoints: int = 3,
                 save_best: bool = True):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            keep_n_checkpoints: Number of checkpoints to keep
            save_best: Whether to save best model separately
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_n_checkpoints = keep_n_checkpoints
        self.save_best = save_best
        
        self.best_metric = None
        self.best_epoch = 0
        self.checkpoint_history = []
    
    def save_checkpoint(self,
                       model: torch.nn.Module,
                       optimizer: torch.optim.Optimizer,
                       scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
                       epoch: int,
                       metrics: Dict[str, float],
                       is_best: bool = False,
                       extra_state: Optional[Dict[str, Any]] = None) -> str:
        """
        Save model checkpoint.
        
        Args:
            model: Model to save
            optimizer: Optimizer state
            scheduler: Learning rate scheduler state
            epoch: Current epoch
            metrics: Training metrics
            is_best: Whether this is the best model so far
            extra_state: Additional state to save
            
        Returns:
            Path to saved checkpoint
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        if extra_state:
            checkpoint.update(extra_state)
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pth"
        torch.save(checkpoint, checkpoint_path)
        self.checkpoint_history.append(checkpoint_path)
        
        # Save best model if specified
        if is_best and self.save_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            shutil.copy2(checkpoint_path, best_path)
            self.best_metric = metrics.get('val_loss', float('inf'))
            self.best_epoch = epoch
            logger.info(f"New best model saved at epoch {epoch}")
        
        # Save latest model
        latest_path = self.checkpoint_dir / "latest_model.pth"
        shutil.copy2(checkpoint_path, latest_path)
        
        # Clean up old checkpoints
        self._cleanup_checkpoints()
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")
        return str(checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Load checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Checkpoint dictionary
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        logger.info(f"Checkpoint loaded: {checkpoint_path}")
        return checkpoint
    
    def load_best_model(self) -> Optional[Dict[str, Any]]:
        """Load best model checkpoint."""
        best_path = self.checkpoint_dir / "best_model.pth"
        if best_path.exists():
            return self.load_checkpoint(str(best_path))
        return None
    
    def load_latest_model(self) -> Optional[Dict[str, Any]]:
        """Load latest model checkpoint."""
        latest_path = self.checkpoint_dir / "latest_model.pth"
        if latest_path.exists():
            return self.load_checkpoint(str(latest_path))
        return None
    
    def _cleanup_checkpoints(self):
        """Remove old checkpoints to keep only the specified number."""
        if len(self.checkpoint_history) > self.keep_n_checkpoints:
            # Sort by epoch number (extracted from filename)
            self.checkpoint_history.sort(key=lambda x: int(x.stem.split('_')[-1]))
            
            # Remove oldest checkpoints
            to_remove = self.checkpoint_history[:-self.keep_n_checkpoints]
            for path in to_remove:
                if path.exists():
                    path.unlink()
                    logger.debug(f"Removed old checkpoint: {path}")
            
            self.checkpoint_history = self.checkpoint_history[-self.keep_n_checkpoints:]


class EarlyStopping:
    """Early stopping utility to prevent overfitting."""
    
    def __init__(self,
                 patience: int = 10,
                 min_delta: float = 1e-4,
                 mode: str = 'min',
                 restore_best_weights: bool = True):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss (lower is better), 'max' for accuracy
            restore_best_weights: Whether to restore best weights when stopping
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        
        self.best_score = None
        self.counter = 0
        self.best_epoch = 0
        self.should_stop = False
        
        if mode == 'min':
            self.is_better = lambda score, best: score < best - min_delta
            self.best_score = float('inf')
        else:
            self.is_better = lambda score, best: score > best + min_delta
            self.best_score = float('-inf')
    
    def __call__(self, score: float, epoch: int) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current metric score
            epoch: Current epoch
            
        Returns:
            True if training should stop
        """
        if self.is_better(score, self.best_score):
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            self.should_stop = True
            logger.info(f"Early stopping triggered after {self.patience} epochs without improvement")
            logger.info(f"Best score: {self.best_score:.6f} at epoch {self.best_epoch}")
            
        return self.should_stop


def setup_training_environment(config: TrainingConfig) -> Dict[str, Any]:
    """
    Setup training environment and directories.
    
    Args:
        config: Training configuration
        
    Returns:
        Environment setup information
    """
    # Create output directories
    output_dir = Path(config.output_dir)
    experiment_dir = output_dir / config.experiment_name
    
    # Create directory structure
    dirs = {
        'experiment_dir': experiment_dir,
        'checkpoints_dir': experiment_dir / 'checkpoints',
        'logs_dir': experiment_dir / 'logs',
        'tensorboard_dir': experiment_dir / 'tensorboard',
        'configs_dir': experiment_dir / 'configs'
    }
    
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Setup device
    if config.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(config.device)
    
    # Setup logging
    log_file = dirs['logs_dir'] / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    # Save configuration
    config_path = dirs['configs_dir'] / 'training_config.yaml'
    config.save_yaml(str(config_path))
    
    logger.info(f"Training environment setup complete")
    logger.info(f"Experiment directory: {experiment_dir}")
    logger.info(f"Device: {device}")
    logger.info(f"Mixed precision: {config.mixed_precision}")
    
    return {
        'device': device,
        'dirs': dirs,
        'config_path': config_path
    }


def save_training_config(config: TrainingConfig, save_path: str):
    """
    Save training configuration to file.
    
    Args:
        config: Training configuration
        save_path: Path to save configuration
    """
    config.save_yaml(save_path)
    logger.info(f"Training configuration saved: {save_path}")


def load_training_config(config_path: str) -> TrainingConfig:
    """
    Load training configuration from file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Training configuration
    """
    config = TrainingConfig.from_yaml(config_path)
    logger.info(f"Training configuration loaded: {config_path}")
    return config


def get_parameter_count(model: torch.nn.Module) -> Dict[str, int]:
    """
    Get model parameter counts.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params
    }


def setup_optimizer_and_scheduler(model: torch.nn.Module,
                                config: TrainingConfig) -> tuple:
    """
    Setup optimizer and learning rate scheduler.
    
    Args:
        model: Model to optimize
        config: Training configuration
        
    Returns:
        Tuple of (optimizer, scheduler)
    """
    # Setup optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Setup scheduler
    if config.scheduler_type.lower() == 'steplr':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.step_size,
            gamma=config.gamma
        )
    elif config.scheduler_type.lower() == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.num_epochs
        )
    elif config.scheduler_type.lower() == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config.gamma,
            patience=config.step_size
        )
    else:
        scheduler = None
        logger.warning(f"Unknown scheduler type: {config.scheduler_type}")
    
    return optimizer, scheduler 