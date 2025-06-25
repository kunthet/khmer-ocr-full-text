"""
Base Trainer Class for Khmer Digits OCR

This module provides an abstract base trainer class with common functionality
for training OCR models, including training loops, validation, and logging.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, Callable
import logging
from pathlib import Path
import time
import os

from .utils import (
    TrainingConfig, 
    CheckpointManager, 
    EarlyStopping,
    get_parameter_count,
    setup_optimizer_and_scheduler
)
from .metrics import OCRMetrics
from .losses import OCRLoss

logger = logging.getLogger(__name__)


class BaseTrainer(ABC):
    """
    Abstract base trainer class for OCR models.
    
    This class provides common training functionality including:
    - Training and validation loops
    - Loss calculation and metrics tracking
    - Model checkpointing and early stopping
    - TensorBoard logging
    - Learning rate scheduling
    """
    
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 config: TrainingConfig,
                 device: torch.device):
        """
        Initialize base trainer.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration
            device: Device to use for training
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Setup optimizer and scheduler
        self.optimizer, self.scheduler = setup_optimizer_and_scheduler(model, config)
        
        # Setup loss function
        self.criterion = self._create_loss_function()
        
        # Setup metrics
        self.metrics_calculator = self._create_metrics_calculator()
        
        # Setup checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=os.path.join(config.output_dir, config.experiment_name, 'checkpoints'),
            keep_n_checkpoints=config.keep_n_checkpoints,
            save_best=True
        )
        
        # Setup early stopping
        self.early_stopping = EarlyStopping(
            patience=config.early_stopping_patience,
            min_delta=config.early_stopping_min_delta,
            mode='min',  # Assuming we're monitoring validation loss
            restore_best_weights=True
        )
        
        # Setup TensorBoard logging
        self.writer = None
        if config.use_tensorboard:
            tensorboard_dir = os.path.join(config.output_dir, config.experiment_name, 'tensorboard')
            self.writer = SummaryWriter(tensorboard_dir)
        
        # Setup mixed precision training
        self.scaler = torch.cuda.amp.GradScaler() if config.mixed_precision and device.type == 'cuda' else None
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': []
        }
        
        # Log model information
        param_info = get_parameter_count(self.model)
        logger.info(f"Model parameters: {param_info}")
        
    @abstractmethod
    def _create_loss_function(self) -> OCRLoss:
        """Create loss function. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def _create_metrics_calculator(self) -> OCRMetrics:
        """Create metrics calculator. Must be implemented by subclasses."""
        pass
    
    def train(self) -> Dict[str, Any]:
        """
        Main training loop.
        
        Returns:
            Training results dictionary
        """
        logger.info(f"Starting training for {self.config.num_epochs} epochs")
        logger.info(f"Training samples: {len(self.train_loader.dataset)}")
        logger.info(f"Validation samples: {len(self.val_loader.dataset)}")
        
        start_time = time.time()
        
        for epoch in range(1, self.config.num_epochs + 1):
            self.current_epoch = epoch
            
            # Training phase
            train_results = self._train_epoch()
            
            # Validation phase
            val_results = self._validate_epoch()
            
            # Update learning rate
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_results['loss'])
                else:
                    self.scheduler.step()
            
            # Log results
            self._log_epoch_results(train_results, val_results)
            
            # Save checkpoint
            is_best = val_results['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_results['loss']
            
            if epoch % self.config.save_every_n_epochs == 0 or is_best:
                self.checkpoint_manager.save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch,
                    metrics=val_results,
                    is_best=is_best,
                    extra_state={
                        'config': self.config.to_dict(),
                        'training_history': self.training_history
                    }
                )
            
            # Check early stopping
            if self.early_stopping(val_results['loss'], epoch):
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break
        
        # Training completed
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.2f} seconds")
        
        # Load best model
        best_checkpoint = self.checkpoint_manager.load_best_model()
        if best_checkpoint:
            self.model.load_state_dict(best_checkpoint['model_state_dict'])
            logger.info(f"Loaded best model from epoch {best_checkpoint['epoch']}")
        
        # Close TensorBoard writer
        if self.writer:
            self.writer.close()
        
        return {
            'best_val_loss': self.best_val_loss,
            'total_epochs': self.current_epoch,
            'total_time': total_time,
            'training_history': self.training_history
        }
    
    def _train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Training results for the epoch
        """
        self.model.train()
        self.metrics_calculator.reset()
        
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        for batch_idx, (images, labels, metadata) in enumerate(self.train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass with mixed precision if enabled
            if self.scaler:
                with torch.cuda.amp.autocast():
                    predictions = self.model(images, target_sequences=labels)
                    loss_dict = self.criterion(predictions, labels)
                    loss = loss_dict['loss']
            else:
                predictions = self.model(images, target_sequences=labels)
                # Debug tensor shapes on first few batches
                if self.current_epoch == 1 and batch_idx <= 2:
                    logger.info(f"Epoch {self.current_epoch} Batch {batch_idx} - Images: {images.shape}, Labels: {labels.shape}, Predictions: {predictions.shape}")
                
                try:
                    loss_dict = self.criterion(predictions, labels)
                    loss = loss_dict['loss']
                except Exception as e:
                    logger.error(f"Loss calculation failed at batch {batch_idx}")
                    logger.error(f"Predictions shape: {predictions.shape}")
                    logger.error(f"Labels shape: {labels.shape}")
                    logger.error(f"Error: {e}")
                    raise e
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
                self.optimizer.step()
            
            # Update metrics
            self.metrics_calculator.update(predictions, labels, metadata)
            total_loss += loss.item()
            self.global_step += 1
            
            # Log batch results
            if batch_idx % self.config.log_every_n_steps == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                logger.info(
                    f"Epoch {self.current_epoch:3d} | "
                    f"Batch {batch_idx:4d}/{num_batches} | "
                    f"Loss: {loss.item():.4f} | "
                    f"LR: {current_lr:.6f}"
                )
                
                if self.writer:
                    self.writer.add_scalar('Train/BatchLoss', loss.item(), self.global_step)
                    self.writer.add_scalar('Train/LearningRate', current_lr, self.global_step)
        
        # Calculate epoch metrics
        avg_loss = total_loss / num_batches
        metrics = self.metrics_calculator.compute()
        
        results = {
            'loss': avg_loss,
            **metrics
        }
        
        self.training_history['train_loss'].append(avg_loss)
        self.training_history['train_metrics'].append(metrics)
        
        return results
    
    def _validate_epoch(self) -> Dict[str, float]:
        """
        Validate for one epoch.
        
        Returns:
            Validation results for the epoch
        """
        self.model.eval()
        self.metrics_calculator.reset()
        
        total_loss = 0.0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for batch_idx, (images, labels, metadata) in enumerate(self.val_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Debug validation batch shapes (only first few batches)
                if batch_idx <= 2:
                    logger.info(f"Validation Batch {batch_idx} - Images: {images.shape}, Labels: {labels.shape}")
                
                # Forward pass (no target_sequences for validation, but disable early stopping for consistent shapes)
                if self.scaler:
                    with torch.cuda.amp.autocast():
                        predictions = self.model(images, allow_early_stopping=False)
                        if batch_idx <= 2:
                            logger.info(f"Validation Batch {batch_idx} - Predictions: {predictions.shape}")
                        loss_dict = self.criterion(predictions, labels)
                        loss = loss_dict['loss']
                else:
                    predictions = self.model(images, allow_early_stopping=False)
                    if batch_idx <= 2:
                        logger.info(f"Validation Batch {batch_idx} - Predictions: {predictions.shape}")
                    
                    try:
                        loss_dict = self.criterion(predictions, labels)
                        loss = loss_dict['loss']
                    except Exception as e:
                        logger.error(f"Validation loss calculation failed at batch {batch_idx}")
                        logger.error(f"Predictions shape: {predictions.shape}")
                        logger.error(f"Labels shape: {labels.shape}")
                        logger.error(f"Error: {e}")
                        raise e
                
                # Update metrics
                self.metrics_calculator.update(predictions, labels, metadata)
                total_loss += loss.item()
        
        # Calculate epoch metrics
        avg_loss = total_loss / num_batches
        metrics = self.metrics_calculator.compute()
        
        results = {
            'loss': avg_loss,
            **metrics
        }
        
        self.training_history['val_loss'].append(avg_loss)
        self.training_history['val_metrics'].append(metrics)
        
        return results
    
    def _log_epoch_results(self,
                          train_results: Dict[str, float],
                          val_results: Dict[str, float]):
        """
        Log epoch results.
        
        Args:
            train_results: Training results
            val_results: Validation results
        """
        # Console logging
        logger.info(f"Epoch {self.current_epoch:3d} Results:")
        logger.info(f"  Train Loss: {train_results['loss']:.4f}")
        logger.info(f"  Val Loss:   {val_results['loss']:.4f}")
        
        if 'char_accuracy' in train_results:
            logger.info(f"  Train Char Acc: {train_results['char_accuracy']:.1%}")
            logger.info(f"  Val Char Acc:   {val_results['char_accuracy']:.1%}")
        
        if 'seq_accuracy' in train_results:
            logger.info(f"  Train Seq Acc:  {train_results['seq_accuracy']:.1%}")
            logger.info(f"  Val Seq Acc:    {val_results['seq_accuracy']:.1%}")
        
        # TensorBoard logging
        if self.writer:
            # Loss
            self.writer.add_scalars('Loss', {
                'Train': train_results['loss'],
                'Validation': val_results['loss']
            }, self.current_epoch)
            
            # Accuracy metrics
            if 'char_accuracy' in train_results:
                self.writer.add_scalars('CharAccuracy', {
                    'Train': train_results['char_accuracy'],
                    'Validation': val_results['char_accuracy']
                }, self.current_epoch)
            
            if 'seq_accuracy' in train_results:
                self.writer.add_scalars('SeqAccuracy', {
                    'Train': train_results['seq_accuracy'],
                    'Validation': val_results['seq_accuracy']
                }, self.current_epoch)
            
            # Edit distance
            if 'edit_distance' in train_results:
                self.writer.add_scalars('EditDistance', {
                    'Train': train_results['edit_distance'],
                    'Validation': val_results['edit_distance']
                }, self.current_epoch)
    
    def save_model(self, save_path: str):
        """Save model state dict."""
        torch.save(self.model.state_dict(), save_path)
        logger.info(f"Model saved: {save_path}")
    
    def load_model(self, load_path: str):
        """Load model state dict."""
        state_dict = torch.load(load_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        logger.info(f"Model loaded: {load_path}")
    
    def get_model_summary(self) -> str:
        """Get model summary string."""
        param_info = get_parameter_count(self.model)
        
        summary = f"""
Model Summary:
- Total parameters: {param_info['total_parameters']:,}
- Trainable parameters: {param_info['trainable_parameters']:,}
- Non-trainable parameters: {param_info['non_trainable_parameters']:,}
- Model size: {param_info['total_parameters'] * 4 / 1024 / 1024:.1f} MB (float32)
"""
        return summary 