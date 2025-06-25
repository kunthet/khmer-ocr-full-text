"""
OCR Trainer for Khmer Digits

This module provides a specialized trainer for OCR tasks, extending the base trainer
with OCR-specific functionality including character mapping and evaluation.
"""

import torch
from typing import Dict, Any
import logging

from .base_trainer import BaseTrainer
from .losses import OCRLoss, create_loss_function
from .metrics import OCRMetrics, create_metrics_calculator

logger = logging.getLogger(__name__)


class OCRTrainer(BaseTrainer):
    """
    Specialized trainer for OCR tasks.
    
    This trainer extends BaseTrainer with OCR-specific functionality:
    - Character mapping and vocabulary management
    - OCR-specific loss functions and metrics
    - Sequence prediction evaluation
    """
    
    def __init__(self,
                 model,
                 train_loader,
                 val_loader,
                 config,
                 device,
                 char_to_idx=None,
                 idx_to_char=None):
        """
        Initialize OCR trainer.
        
        Args:
            model: OCR model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration
            device: Device to use for training
            char_to_idx: Character to index mapping
            idx_to_char: Index to character mapping
        """
        # Setup character mappings
        if char_to_idx is None:
            char_to_idx = {
                '០': 0, '១': 1, '២': 2, '៣': 3, '៤': 4,
                '៥': 5, '៦': 6, '៧': 7, '៨': 8, '៩': 9,
                '<EOS>': 10, '<PAD>': 11, '<BLANK>': 12
            }
        
        if idx_to_char is None:
            idx_to_char = {v: k for k, v in char_to_idx.items()}
        
        self.char_to_idx = char_to_idx
        self.idx_to_char = idx_to_char
        self.vocab_size = len(char_to_idx)
        
        # Initialize base trainer
        super().__init__(model, train_loader, val_loader, config, device)
        
        logger.info(f"OCR Trainer initialized")
        logger.info(f"Vocabulary size: {self.vocab_size}")
        
        # Safe logging of characters to avoid Unicode encoding issues on Windows
        # Convert to ASCII-safe representation for logging
        char_list = []
        for char in char_to_idx.keys():
            if char.startswith('<') and char.endswith('>'):
                # Special tokens - safe to log as-is
                char_list.append(char)
            else:
                # Unicode characters - use safe representation
                char_list.append(f"U+{ord(char):04X}")
        
        logger.info(f"Characters: {char_list}")
    
    def _create_loss_function(self) -> OCRLoss:
        """Create OCR-specific loss function."""
        loss_config = {
            'loss': {
                'type': getattr(self.config, 'loss_type', 'crossentropy'),
                'pad_token_id': self.char_to_idx.get('<PAD>', 11),
                'blank_token_id': self.char_to_idx.get('<BLANK>', 12),
                'label_smoothing': getattr(self.config, 'label_smoothing', 0.0),
                'reduction': 'mean'
            }
        }
        return create_loss_function(loss_config)
    
    def _create_metrics_calculator(self) -> OCRMetrics:
        """Create OCR-specific metrics calculator."""
        metrics_config = {
            'idx_to_char': self.idx_to_char,
            'pad_token_id': self.char_to_idx.get('<PAD>', 11),
            'eos_token_id': self.char_to_idx.get('<EOS>', 10),
            'blank_token_id': self.char_to_idx.get('<BLANK>', 12)
        }
        return create_metrics_calculator(metrics_config)
    
    def predict_sequence(self, images: torch.Tensor) -> list:
        """
        Predict sequences for a batch of images.
        
        Args:
            images: Batch of images [batch_size, channels, height, width]
            
        Returns:
            List of predicted sequences as strings
        """
        self.model.eval()
        
        with torch.no_grad():
            images = images.to(self.device)
            predictions = self.model(images)
            pred_classes = torch.argmax(predictions, dim=-1)
            
            sequences = []
            for seq in pred_classes:
                sequence_str = ""
                for idx in seq:
                    idx_val = idx.item()
                    if idx_val == self.char_to_idx.get('<EOS>', 10):
                        break
                    if idx_val != self.char_to_idx.get('<PAD>', 11) and idx_val in self.idx_to_char:
                        sequence_str += self.idx_to_char[idx_val]
                sequences.append(sequence_str)
        
        return sequences
    
    def evaluate_samples(self, num_samples: int = 10) -> Dict[str, Any]:
        """
        Evaluate a few samples and show predictions vs targets.
        
        Args:
            num_samples: Number of samples to evaluate
            
        Returns:
            Evaluation results with examples
        """
        self.model.eval()
        
        # Get a batch from validation loader
        data_iter = iter(self.val_loader)
        images, labels, metadata = next(data_iter)
        
        # Limit to requested number of samples
        if images.size(0) > num_samples:
            images = images[:num_samples]
            labels = labels[:num_samples]
            metadata = metadata[:num_samples]
        
        # Get predictions
        predicted_sequences = self.predict_sequence(images)
        
        # Convert targets to strings
        target_sequences = []
        for seq in labels:
            target_str = ""
            for idx in seq:
                idx_val = idx.item()
                if idx_val == self.char_to_idx.get('<EOS>', 10):
                    break
                if idx_val != self.char_to_idx.get('<PAD>', 11) and idx_val in self.idx_to_char:
                    target_str += self.idx_to_char[idx_val]
            target_sequences.append(target_str)
        
        # Create evaluation results
        examples = []
        correct = 0
        
        for i, (pred, target, meta) in enumerate(zip(predicted_sequences, target_sequences, metadata)):
            is_correct = pred == target
            if is_correct:
                correct += 1
            
            example = {
                'sample_id': i,
                'predicted': pred,
                'target': target,
                'correct': is_correct,
                'original_label': meta.get('original_label', ''),
                'font_name': meta.get('font_name', ''),
                'sequence_length': meta.get('sequence_length', 0)
            }
            examples.append(example)
        
        accuracy = correct / len(examples) if examples else 0.0
        
        return {
            'accuracy': accuracy,
            'total_samples': len(examples),
            'correct_samples': correct,
            'examples': examples
        }
    
    def get_confusion_matrix(self) -> Dict[str, Dict[str, int]]:
        """
        Get character-level confusion matrix from validation set.
        
        Returns:
            Confusion matrix with character labels
        """
        self.model.eval()
        self.metrics_calculator.reset()
        
        with torch.no_grad():
            for images, labels, metadata in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                predictions = self.model(images)
                self.metrics_calculator.update(predictions, labels, metadata)
        
        return self.metrics_calculator.get_confusion_matrix()
    
    def analyze_errors(self, num_samples: int = 50) -> Dict[str, Any]:
        """
        Analyze prediction errors to identify common failure patterns.
        
        Args:
            num_samples: Number of samples to analyze
            
        Returns:
            Error analysis results
        """
        self.model.eval()
        
        errors = []
        total_samples = 0
        
        with torch.no_grad():
            for images, labels, metadata in self.val_loader:
                if total_samples >= num_samples:
                    break
                
                images = images.to(self.device)
                predictions = self.predict_sequence(images)
                
                # Convert targets to strings
                targets = []
                for seq in labels:
                    target_str = ""
                    for idx in seq:
                        idx_val = idx.item()
                        if idx_val == self.char_to_idx.get('<EOS>', 10):
                            break
                        if idx_val != self.char_to_idx.get('<PAD>', 11) and idx_val in self.idx_to_char:
                            target_str += self.idx_to_char[idx_val]
                    targets.append(target_str)
                
                # Find errors
                for i, (pred, target, meta) in enumerate(zip(predictions, targets, metadata)):
                    if total_samples >= num_samples:
                        break
                    
                    total_samples += 1
                    
                    if pred != target:
                        error = {
                            'predicted': pred,
                            'target': target,
                            'sequence_length': len(target),
                            'font_name': meta.get('font_name', ''),
                            'error_type': self._classify_error(pred, target)
                        }
                        errors.append(error)
        
        # Analyze error patterns
        error_types = {}
        length_errors = {}
        font_errors = {}
        
        for error in errors:
            # Error type analysis
            error_type = error['error_type']
            error_types[error_type] = error_types.get(error_type, 0) + 1
            
            # Sequence length analysis
            seq_len = error['sequence_length']
            length_errors[seq_len] = length_errors.get(seq_len, 0) + 1
            
            # Font analysis
            font = error['font_name']
            font_errors[font] = font_errors.get(font, 0) + 1
        
        return {
            'total_errors': len(errors),
            'total_samples': total_samples,
            'error_rate': len(errors) / total_samples if total_samples > 0 else 0.0,
            'error_types': error_types,
            'length_errors': length_errors,
            'font_errors': font_errors,
            'examples': errors[:10]  # First 10 examples
        }
    
    def _classify_error(self, predicted: str, target: str) -> str:
        """
        Classify the type of prediction error.
        
        Args:
            predicted: Predicted sequence
            target: Target sequence
            
        Returns:
            Error type classification
        """
        if len(predicted) == 0:
            return "empty_prediction"
        elif len(target) == 0:
            return "false_positive"
        elif len(predicted) < len(target):
            return "too_short"
        elif len(predicted) > len(target):
            return "too_long"
        elif len(predicted) == len(target):
            return "substitution"
        else:
            return "unknown" 