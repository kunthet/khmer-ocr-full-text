"""
Evaluation Metrics for Khmer Digits OCR

This module provides comprehensive metrics for evaluating OCR model performance,
including character-level accuracy, sequence-level accuracy, and edit distance.
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


def calculate_character_accuracy(predictions: torch.Tensor,
                               targets: torch.Tensor,
                               pad_token_id: int = 11) -> float:
    """
    Calculate character-level accuracy ignoring PAD tokens.
    
    Args:
        predictions: Model predictions [batch_size, seq_len, num_classes]
        targets: Target labels [batch_size, seq_len]
        pad_token_id: PAD token ID to ignore
        
    Returns:
        Character accuracy as float
    """
    # Get predicted classes
    pred_classes = torch.argmax(predictions, dim=-1)
    
    # Create mask to ignore PAD tokens
    mask = targets != pad_token_id
    
    # Calculate accuracy only for non-PAD tokens
    if mask.sum() == 0:
        return 0.0
    
    correct = (pred_classes == targets)[mask].sum().item()
    total = mask.sum().item()
    
    return correct / total


def calculate_sequence_accuracy(predictions: torch.Tensor,
                              targets: torch.Tensor,
                              pad_token_id: int = 11,
                              eos_token_id: int = 10) -> float:
    """
    Calculate sequence-level accuracy (exact match).
    
    Args:
        predictions: Model predictions [batch_size, seq_len, num_classes]
        targets: Target labels [batch_size, seq_len]
        pad_token_id: PAD token ID to ignore
        eos_token_id: EOS token ID for sequence termination
        
    Returns:
        Sequence accuracy as float
    """
    batch_size = predictions.size(0)
    pred_classes = torch.argmax(predictions, dim=-1)
    
    correct_sequences = 0
    
    for i in range(batch_size):
        pred_seq = pred_classes[i]
        target_seq = targets[i]
        
        # Find EOS positions or use full sequence
        pred_eos = torch.where(pred_seq == eos_token_id)[0]
        target_eos = torch.where(target_seq == eos_token_id)[0]
        
        pred_end = pred_eos[0].item() if len(pred_eos) > 0 else len(pred_seq)
        target_end = target_eos[0].item() if len(target_eos) > 0 else len(target_seq)
        
        # Compare sequences up to EOS (excluding EOS itself)
        pred_clean = pred_seq[:pred_end]
        target_clean = target_seq[:target_end]
        
        if torch.equal(pred_clean, target_clean):
            correct_sequences += 1
    
    return correct_sequences / batch_size


def calculate_edit_distance(str1: str, str2: str) -> int:
    """
    Calculate Levenshtein (edit) distance between two strings.
    
    Args:
        str1: First string
        str2: Second string
        
    Returns:
        Edit distance as integer
    """
    m, n = len(str1), len(str2)
    
    # Create DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Initialize base cases
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i-1] == str2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],    # deletion
                    dp[i][j-1],    # insertion
                    dp[i-1][j-1]   # substitution
                )
    
    return dp[m][n]


def calculate_normalized_edit_distance(predictions: torch.Tensor,
                                     targets: torch.Tensor,
                                     idx_to_char: Dict[int, str],
                                     pad_token_id: int = 11,
                                     eos_token_id: int = 10) -> float:
    """
    Calculate normalized edit distance (0-1 scale).
    
    Args:
        predictions: Model predictions [batch_size, seq_len, num_classes]
        targets: Target labels [batch_size, seq_len]
        idx_to_char: Index to character mapping
        pad_token_id: PAD token ID to ignore
        eos_token_id: EOS token ID for sequence termination
        
    Returns:
        Normalized edit distance as float (0=perfect, 1=completely wrong)
    """
    batch_size = predictions.size(0)
    pred_classes = torch.argmax(predictions, dim=-1)
    
    total_distance = 0
    total_length = 0
    
    for i in range(batch_size):
        pred_seq = pred_classes[i]
        target_seq = targets[i]
        
        # Convert to strings
        pred_str = ""
        target_str = ""
        
        # Build predicted string
        for idx in pred_seq:
            idx_val = idx.item()
            if idx_val == eos_token_id:
                break
            if idx_val != pad_token_id and idx_val in idx_to_char:
                pred_str += idx_to_char[idx_val]
        
        # Build target string
        for idx in target_seq:
            idx_val = idx.item()
            if idx_val == eos_token_id:
                break
            if idx_val != pad_token_id and idx_val in idx_to_char:
                target_str += idx_to_char[idx_val]
        
        # Calculate edit distance
        distance = calculate_edit_distance(pred_str, target_str)
        max_length = max(len(pred_str), len(target_str), 1)  # Avoid division by zero
        
        total_distance += distance
        total_length += max_length
    
    return total_distance / total_length if total_length > 0 else 0.0


class OCRMetrics:
    """
    Comprehensive metrics calculator for OCR evaluation.
    """
    
    def __init__(self,
                 idx_to_char: Dict[int, str],
                 pad_token_id: int = 11,
                 eos_token_id: int = 10,
                 blank_token_id: int = 0):
        """
        Initialize metrics calculator.
        
        Args:
            idx_to_char: Index to character mapping
            pad_token_id: PAD token ID
            eos_token_id: EOS token ID
            blank_token_id: BLANK token ID
        """
        self.idx_to_char = idx_to_char
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.blank_token_id = blank_token_id
        
        self.reset()
    
    def reset(self):
        """Reset all accumulated metrics."""
        self.total_char_correct = 0
        self.total_chars = 0
        self.total_seq_correct = 0
        self.total_sequences = 0
        self.total_edit_distance = 0.0
        self.total_edit_length = 0
        self.per_class_correct = defaultdict(int)
        self.per_class_total = defaultdict(int)
        self.confusion_matrix = defaultdict(lambda: defaultdict(int))
    
    def update(self,
               predictions: torch.Tensor,
               targets: torch.Tensor,
               metadata: Optional[List[Dict]] = None):
        """
        Update metrics with a batch of predictions.
        
        Args:
            predictions: Model predictions [batch_size, seq_len, num_classes]
            targets: Target labels [batch_size, seq_len]
            metadata: Optional batch metadata
        """
        batch_size = predictions.size(0)
        pred_classes = torch.argmax(predictions, dim=-1)
        
        # Character-level accuracy
        mask = targets != self.pad_token_id
        char_correct = (pred_classes == targets)[mask].sum().item()
        char_total = mask.sum().item()
        
        self.total_char_correct += char_correct
        self.total_chars += char_total
        
        # Sequence-level accuracy and edit distance
        seq_correct = 0
        edit_distance = 0.0
        edit_length = 0
        
        for i in range(batch_size):
            pred_seq = pred_classes[i]
            target_seq = targets[i]
            
            # Sequence accuracy
            pred_str, target_str = self._sequences_to_strings(pred_seq, target_seq)
            
            if pred_str == target_str:
                seq_correct += 1
            
            # Edit distance
            distance = calculate_edit_distance(pred_str, target_str)
            max_length = max(len(pred_str), len(target_str), 1)
            
            edit_distance += distance
            edit_length += max_length
            
            # Per-class accuracy
            for pred_idx, target_idx in zip(pred_seq, target_seq):
                pred_val = pred_idx.item()
                target_val = target_idx.item()
                
                if target_val != self.pad_token_id:
                    self.per_class_total[target_val] += 1
                    if pred_val == target_val:
                        self.per_class_correct[target_val] += 1
                    
                    # Confusion matrix
                    self.confusion_matrix[target_val][pred_val] += 1
        
        self.total_seq_correct += seq_correct
        self.total_sequences += batch_size
        self.total_edit_distance += edit_distance
        self.total_edit_length += edit_length
    
    def compute(self) -> Dict[str, float]:
        """
        Compute all metrics.
        
        Returns:
            Dictionary of computed metrics
        """
        metrics = {}
        
        # Character accuracy
        if self.total_chars > 0:
            metrics['char_accuracy'] = self.total_char_correct / self.total_chars
        else:
            metrics['char_accuracy'] = 0.0
        
        # Sequence accuracy
        if self.total_sequences > 0:
            metrics['seq_accuracy'] = self.total_seq_correct / self.total_sequences
        else:
            metrics['seq_accuracy'] = 0.0
        
        # Normalized edit distance
        if self.total_edit_length > 0:
            metrics['edit_distance'] = self.total_edit_distance / self.total_edit_length
        else:
            metrics['edit_distance'] = 0.0
        
        # Per-class accuracy
        per_class_acc = {}
        for class_id in self.per_class_total:
            if self.per_class_total[class_id] > 0:
                acc = self.per_class_correct[class_id] / self.per_class_total[class_id]
                char = self.idx_to_char.get(class_id, f'class_{class_id}')
                per_class_acc[char] = acc
        
        metrics['per_class_accuracy'] = per_class_acc
        
        # Overall metrics
        metrics['total_samples'] = self.total_sequences
        metrics['total_characters'] = self.total_chars
        
        return metrics
    
    def _sequences_to_strings(self,
                            pred_seq: torch.Tensor,
                            target_seq: torch.Tensor) -> Tuple[str, str]:
        """
        Convert tensor sequences to strings.
        
        Args:
            pred_seq: Predicted sequence tensor
            target_seq: Target sequence tensor
            
        Returns:
            Tuple of (predicted_string, target_string)
        """
        pred_str = ""
        target_str = ""
        
        # Build predicted string
        for idx in pred_seq:
            idx_val = idx.item()
            if idx_val == self.eos_token_id:
                break
            if idx_val != self.pad_token_id and idx_val in self.idx_to_char:
                pred_str += self.idx_to_char[idx_val]
        
        # Build target string
        for idx in target_seq:
            idx_val = idx.item()
            if idx_val == self.eos_token_id:
                break
            if idx_val != self.pad_token_id and idx_val in self.idx_to_char:
                target_str += self.idx_to_char[idx_val]
        
        return pred_str, target_str
    
    def get_confusion_matrix(self) -> Dict[str, Dict[str, int]]:
        """
        Get confusion matrix with character labels.
        
        Returns:
            Confusion matrix with character labels
        """
        char_confusion = {}
        
        for true_class, predictions in self.confusion_matrix.items():
            true_char = self.idx_to_char.get(true_class, f'class_{true_class}')
            char_confusion[true_char] = {}
            
            for pred_class, count in predictions.items():
                pred_char = self.idx_to_char.get(pred_class, f'class_{pred_class}')
                char_confusion[true_char][pred_char] = count
        
        return char_confusion


def create_metrics_calculator(config: Dict[str, Any]) -> OCRMetrics:
    """
    Factory function to create metrics calculator from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured metrics calculator
    """
    # Default character mapping (can be overridden by config)
    default_char_mapping = {
        0: '០', 1: '១', 2: '២', 3: '៣', 4: '៤',
        5: '៥', 6: '៦', 7: '៧', 8: '៨', 9: '៩',
        10: '<EOS>', 11: '<PAD>', 12: '<BLANK>'
    }
    
    idx_to_char = config.get('idx_to_char', default_char_mapping)
    pad_token_id = config.get('pad_token_id', 11)
    eos_token_id = config.get('eos_token_id', 10)
    blank_token_id = config.get('blank_token_id', 12)
    
    return OCRMetrics(
        idx_to_char=idx_to_char,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        blank_token_id=blank_token_id
    ) 