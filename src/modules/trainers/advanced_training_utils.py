"""
Advanced Training Utilities for Khmer OCR

This module provides advanced training utilities including gradient accumulation,
mixed precision training, and enhanced model checkpointing for long training runs.
"""

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from typing import Dict, List, Any, Optional, Union, Callable
import logging
import os
import json
import time
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)


class GradientAccumulator:
    """
    Gradient accumulation manager for training with larger effective batch sizes.
    
    This class handles gradient accumulation to simulate larger batch sizes
    when memory is limited.
    """
    
    def __init__(self,
                 accumulation_steps: int = 1,
                 max_grad_norm: float = 1.0,
                 scale_grad_by_freq: bool = False):
        """
        Initialize gradient accumulator.
        
        Args:
            accumulation_steps: Number of steps to accumulate gradients
            max_grad_norm: Maximum gradient norm for clipping
            scale_grad_by_freq: Whether to scale gradients by accumulation frequency
        """
        self.accumulation_steps = accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.scale_grad_by_freq = scale_grad_by_freq
        
        self.current_step = 0
        self.accumulated_loss = 0.0
        self.gradient_norms = []
        
    def should_update(self) -> bool:
        """Check if we should update parameters."""
        return (self.current_step + 1) % self.accumulation_steps == 0
    
    def accumulate_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """
        Accumulate loss and scale for gradient accumulation.
        
        Args:
            loss: Current batch loss
            
        Returns:
            Scaled loss for backward pass
        """
        # Scale loss by accumulation steps
        scaled_loss = loss / self.accumulation_steps
        self.accumulated_loss += scaled_loss.item()
        
        return scaled_loss
    
    def step_optimizer(self, 
                      optimizer: torch.optim.Optimizer,
                      model: nn.Module,
                      scaler: Optional[GradScaler] = None) -> Dict[str, float]:
        """
        Step optimizer with gradient clipping and accumulation.
        
        Args:
            optimizer: Optimizer to step
            model: Model being trained
            scaler: Mixed precision scaler (optional)
            
        Returns:
            Dictionary with gradient information
        """
        self.current_step += 1
        
        if self.should_update():
            # Calculate gradient norm before clipping
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            
            self.gradient_norms.append(total_norm)
            
            # Apply gradient clipping
            if scaler is not None:
                # Mixed precision training
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                # Regular training
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
                optimizer.step()
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Calculate average accumulated loss
            avg_loss = self.accumulated_loss
            self.accumulated_loss = 0.0
            
            return {
                'avg_accumulated_loss': avg_loss,
                'gradient_norm': total_norm,
                'effective_batch_size': self.accumulation_steps,
                'gradient_clip_applied': total_norm > self.max_grad_norm
            }
        
        return {
            'avg_accumulated_loss': self.accumulated_loss,
            'gradient_norm': 0.0,
            'effective_batch_size': self.accumulation_steps,
            'gradient_clip_applied': False
        }
    
    def get_gradient_stats(self) -> Dict[str, float]:
        """Get gradient statistics."""
        if not self.gradient_norms:
            return {'mean_grad_norm': 0.0, 'max_grad_norm': 0.0, 'min_grad_norm': 0.0}
        
        return {
            'mean_grad_norm': np.mean(self.gradient_norms[-100:]),  # Last 100 steps
            'max_grad_norm': np.max(self.gradient_norms[-100:]),
            'min_grad_norm': np.min(self.gradient_norms[-100:]),
            'current_grad_norm': self.gradient_norms[-1] if self.gradient_norms else 0.0
        }


class MixedPrecisionManager:
    """
    Manager for mixed precision training with automatic loss scaling.
    """
    
    def __init__(self,
                 enabled: bool = True,
                 init_scale: float = 2.**16,
                 growth_factor: float = 2.0,
                 backoff_factor: float = 0.5,
                 growth_interval: int = 2000):
        """
        Initialize mixed precision manager.
        
        Args:
            enabled: Whether to enable mixed precision
            init_scale: Initial loss scale
            growth_factor: Factor to grow scale by
            backoff_factor: Factor to reduce scale by
            growth_interval: Steps between scale growth attempts
        """
        self.enabled = enabled and torch.cuda.is_available()
        
        if self.enabled:
            self.scaler = GradScaler(
                init_scale=init_scale,
                growth_factor=growth_factor,
                backoff_factor=backoff_factor,
                growth_interval=growth_interval
            )
        else:
            self.scaler = None
        
        self.scale_history = []
    
    def autocast_context(self):
        """Get autocast context manager."""
        if self.enabled:
            return autocast()
        else:
            # Return a dummy context manager
            return torch.no_grad().__class__()
    
    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss for mixed precision."""
        if self.enabled:
            return self.scaler.scale(loss)
        return loss
    
    def backward(self, loss: torch.Tensor):
        """Backward pass with mixed precision."""
        if self.enabled:
            scaled_loss = self.scaler.scale(loss)
            scaled_loss.backward()
        else:
            loss.backward()
    
    def get_scale_info(self) -> Dict[str, Any]:
        """Get information about loss scaling."""
        if not self.enabled:
            return {'enabled': False}
        
        current_scale = self.scaler.get_scale()
        self.scale_history.append(current_scale)
        
        return {
            'enabled': True,
            'current_scale': current_scale,
            'scale_history_length': len(self.scale_history),
            'recent_scale_changes': len(set(self.scale_history[-10:])) if len(self.scale_history) >= 10 else 0
        }


class EnhancedCheckpointManager:
    """
    Enhanced checkpoint manager with comprehensive state saving and recovery.
    """
    
    def __init__(self,
                 checkpoint_dir: str,
                 keep_n_checkpoints: int = 5,
                 save_optimizer_state: bool = True,
                 save_scheduler_state: bool = True,
                 save_training_state: bool = True,
                 compression: bool = False):
        """
        Initialize enhanced checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            keep_n_checkpoints: Number of checkpoints to keep
            save_optimizer_state: Whether to save optimizer state
            save_scheduler_state: Whether to save scheduler state
            save_training_state: Whether to save training state
            compression: Whether to compress checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_n_checkpoints = keep_n_checkpoints
        self.save_optimizer_state = save_optimizer_state
        self.save_scheduler_state = save_scheduler_state
        self.save_training_state = save_training_state
        self.compression = compression
        
        self.checkpoint_history = []
        self.best_metrics = {}
        
    def save_checkpoint(self,
                       model: nn.Module,
                       epoch: int,
                       optimizer: Optional[torch.optim.Optimizer] = None,
                       scheduler: Optional[Any] = None,
                       metrics: Optional[Dict[str, float]] = None,
                       training_state: Optional[Dict[str, Any]] = None,
                       is_best: bool = False,
                       tag: str = "") -> str:
        """
        Save comprehensive checkpoint.
        
        Args:
            model: Model to save
            epoch: Current epoch
            optimizer: Optimizer state
            scheduler: Scheduler state
            metrics: Current metrics
            training_state: Additional training state
            is_best: Whether this is the best checkpoint
            tag: Additional tag for checkpoint name
            
        Returns:
            Path to saved checkpoint
        """
        # Prepare checkpoint data
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'timestamp': time.time(),
            'metrics': metrics or {},
            'model_info': {
                'total_params': sum(p.numel() for p in model.parameters()),
                'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad),
                'model_class': model.__class__.__name__
            }
        }
        
        # Add optimizer state
        if self.save_optimizer_state and optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        # Add scheduler state
        if self.save_scheduler_state and scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        # Add training state
        if self.save_training_state and training_state is not None:
            checkpoint['training_state'] = training_state
        
        # Determine checkpoint filename
        tag_suffix = f"_{tag}" if tag else ""
        checkpoint_name = f"checkpoint_epoch_{epoch:04d}{tag_suffix}.pth"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        # Save checkpoint
        if self.compression:
            torch.save(checkpoint, checkpoint_path, _use_new_zipfile_serialization=True)
        else:
            torch.save(checkpoint, checkpoint_path)
        
        # Update checkpoint history
        self.checkpoint_history.append({
            'path': str(checkpoint_path),
            'epoch': epoch,
            'timestamp': checkpoint['timestamp'],
            'metrics': metrics or {},
            'is_best': is_best
        })
        
        # Save best checkpoint separately
        if is_best:
            best_path = self.checkpoint_dir / "best_checkpoint.pth"
            if self.compression:
                torch.save(checkpoint, best_path, _use_new_zipfile_serialization=True)
            else:
                torch.save(checkpoint, best_path)
            
            # Update best metrics
            if metrics:
                self.best_metrics.update(metrics)
        
        # Save latest checkpoint
        latest_path = self.checkpoint_dir / "latest_checkpoint.pth"
        if self.compression:
            torch.save(checkpoint, latest_path, _use_new_zipfile_serialization=True)
        else:
            torch.save(checkpoint, latest_path)
        
        # Save checkpoint metadata
        self._save_checkpoint_metadata()
        
        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")
        if is_best:
            logger.info(f"New best checkpoint saved with metrics: {metrics}")
        
        return str(checkpoint_path)
    
    def load_checkpoint(self, 
                       checkpoint_path: str,
                       model: nn.Module,
                       optimizer: Optional[torch.optim.Optimizer] = None,
                       scheduler: Optional[Any] = None,
                       device: str = 'cpu') -> Dict[str, Any]:
        """
        Load checkpoint and restore training state.
        
        Args:
            checkpoint_path: Path to checkpoint file
            model: Model to load state into
            optimizer: Optimizer to load state into
            scheduler: Scheduler to load state into
            device: Device to load checkpoint on
            
        Returns:
            Checkpoint information
        """
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Extract checkpoint info
        checkpoint_info = {
            'epoch': checkpoint.get('epoch', 0),
            'metrics': checkpoint.get('metrics', {}),
            'training_state': checkpoint.get('training_state', {}),
            'model_info': checkpoint.get('model_info', {}),
            'timestamp': checkpoint.get('timestamp', 0)
        }
        
        logger.info(f"Checkpoint loaded: {checkpoint_path}")
        logger.info(f"Resumed from epoch: {checkpoint_info['epoch']}")
        
        return checkpoint_info
    
    def find_latest_checkpoint(self) -> Optional[str]:
        """Find the most recent checkpoint."""
        latest_path = self.checkpoint_dir / "latest_checkpoint.pth"
        if latest_path.exists():
            return str(latest_path)
        
        # Fall back to finding latest by timestamp
        if self.checkpoint_history:
            latest = max(self.checkpoint_history, key=lambda x: x['timestamp'])
            return latest['path']
        
        return None
    
    def find_best_checkpoint(self) -> Optional[str]:
        """Find the best checkpoint."""
        best_path = self.checkpoint_dir / "best_checkpoint.pth"
        if best_path.exists():
            return str(best_path)
        
        return None
    
    def _save_checkpoint_metadata(self):
        """Save checkpoint metadata."""
        metadata = {
            'checkpoint_history': self.checkpoint_history,
            'best_metrics': self.best_metrics,
            'last_updated': time.time()
        }
        
        metadata_path = self.checkpoint_dir / "checkpoint_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints beyond keep limit."""
        if len(self.checkpoint_history) <= self.keep_n_checkpoints:
            return
        
        # Sort by timestamp and remove oldest
        sorted_checkpoints = sorted(self.checkpoint_history, key=lambda x: x['timestamp'])
        checkpoints_to_remove = sorted_checkpoints[:-self.keep_n_checkpoints]
        
        for checkpoint_info in checkpoints_to_remove:
            checkpoint_path = Path(checkpoint_info['path'])
            if checkpoint_path.exists() and not checkpoint_info.get('is_best', False):
                checkpoint_path.unlink()
                logger.debug(f"Removed old checkpoint: {checkpoint_path}")
        
        # Update history
        self.checkpoint_history = sorted_checkpoints[-self.keep_n_checkpoints:]
    
    def get_checkpoint_summary(self) -> Dict[str, Any]:
        """Get summary of checkpoints."""
        return {
            'total_checkpoints': len(self.checkpoint_history),
            'best_metrics': self.best_metrics,
            'latest_epoch': max([cp['epoch'] for cp in self.checkpoint_history]) if self.checkpoint_history else 0,
            'checkpoint_dir': str(self.checkpoint_dir),
            'disk_usage_mb': sum(Path(cp['path']).stat().st_size for cp in self.checkpoint_history if Path(cp['path']).exists()) / (1024 * 1024)
        } 