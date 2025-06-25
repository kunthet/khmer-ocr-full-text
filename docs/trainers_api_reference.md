# Trainers Module API Reference

## Module: `src.modules.trainers`

### Classes

#### `OCRLoss`
Loss function wrapper for OCR training.

```python
class OCRLoss(nn.Module):
    def __init__(self, loss_type: str = 'crossentropy', 
                 label_smoothing: float = 0.0,
                 focal_gamma: float = 2.0,
                 focal_alpha: float = 0.25,
                 pad_token_id: int = 11)
```

**Parameters:**
- `loss_type`: Type of loss ('crossentropy', 'ctc', 'focal')
- `label_smoothing`: Label smoothing factor (0.0-1.0)
- `focal_gamma`: Focal loss gamma parameter
- `focal_alpha`: Focal loss alpha parameter  
- `pad_token_id`: ID of padding token to ignore

**Methods:**
- `forward(predictions, targets, **kwargs) -> Dict[str, torch.Tensor]`

---

#### `CrossEntropyLoss`
Cross-entropy loss with masking support.

```python
class CrossEntropyLoss(nn.Module):
    def __init__(self, label_smoothing: float = 0.0, 
                 pad_token_id: int = 11)
```

**Methods:**
- `forward(predictions, targets) -> torch.Tensor`

---

#### `CTCLoss` 
CTC loss for alignment-free training.

```python
class CTCLoss(nn.Module):
    def __init__(self, blank_token_id: int = 12)
```

**Methods:**
- `forward(predictions, targets, input_lengths, target_lengths) -> torch.Tensor`

---

#### `FocalLoss`
Focal loss for handling class imbalance.

```python
class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, 
                 alpha: float = 0.25,
                 pad_token_id: int = 11)
```

**Methods:**
- `forward(predictions, targets) -> torch.Tensor`

---

#### `OCRMetrics`
Comprehensive metrics calculation for OCR tasks.

```python
class OCRMetrics:
    def __init__(self, idx_to_char: Dict[int, str],
                 pad_token_id: int = 11,
                 eos_token_id: int = 10)
```

**Parameters:**
- `idx_to_char`: Mapping from indices to characters
- `pad_token_id`: Padding token ID
- `eos_token_id`: End-of-sequence token ID

**Methods:**
- `update(predictions: torch.Tensor, targets: torch.Tensor) -> None`
- `compute() -> Dict[str, float]`
- `reset() -> None`
- `get_confusion_matrix() -> np.ndarray`
- `get_per_class_accuracy() -> Dict[str, float]`

---

#### `TrainingConfig`
Training configuration management.

```python
@dataclass
class TrainingConfig:
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
```

**Methods:**
- `to_dict() -> Dict[str, Any]`
- `from_dict(config_dict: Dict[str, Any]) -> TrainingConfig`
- `save_yaml(path: str) -> None`
- `from_yaml(path: str) -> TrainingConfig`

---

#### `CheckpointManager`
Automatic model checkpointing.

```python
class CheckpointManager:
    def __init__(self, checkpoint_dir: str,
                 keep_n_checkpoints: int = 3,
                 save_best: bool = True)
```

**Parameters:**
- `checkpoint_dir`: Directory to save checkpoints
- `keep_n_checkpoints`: Number of checkpoints to retain
- `save_best`: Whether to save best model separately

**Methods:**
- `save_checkpoint(model, optimizer, scheduler, epoch, metrics, is_best=False, extra_state=None) -> str`
- `load_checkpoint(checkpoint_path: str) -> Dict[str, Any]`
- `load_best_model() -> Optional[Dict[str, Any]]`
- `list_checkpoints() -> List[Path]`
- `cleanup_old_checkpoints() -> None`

---

#### `EarlyStopping`
Early stopping mechanism.

```python
class EarlyStopping:
    def __init__(self, patience: int = 10,
                 min_delta: float = 1e-4,
                 mode: str = 'min',
                 restore_best_weights: bool = True)
```

**Parameters:**
- `patience`: Number of epochs to wait for improvement
- `min_delta`: Minimum change to qualify as improvement
- `mode`: 'min' for loss, 'max' for accuracy
- `restore_best_weights`: Whether to restore best weights

**Methods:**
- `__call__(score: float, epoch: int) -> bool`
- `reset() -> None`

**Properties:**
- `should_stop: bool`
- `best_score: float`
- `best_epoch: int`

---

#### `BaseTrainer`
Abstract base trainer class.

```python
class BaseTrainer(ABC):
    def __init__(self, model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 config: TrainingConfig,
                 device: torch.device)
```

**Parameters:**
- `model`: PyTorch model to train
- `train_loader`: Training data loader
- `val_loader`: Validation data loader
- `config`: Training configuration
- `device`: Device for training

**Abstract Methods:**
- `_create_loss_function() -> nn.Module`
- `_create_metrics_calculator() -> Any`

**Methods:**
- `train() -> Dict[str, Any]`
- `_train_epoch() -> Dict[str, float]`
- `_validate_epoch() -> Dict[str, float]`
- `_log_epoch_results(train_results, val_results) -> None`

**Properties:**
- `current_epoch: int`
- `best_val_loss: float`
- `training_history: List[Dict[str, float]]`

---

#### `OCRTrainer`
Specialized trainer for OCR tasks.

```python
class OCRTrainer(BaseTrainer):
    def __init__(self, model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 config: TrainingConfig,
                 device: torch.device,
                 char_to_idx: Optional[Dict[str, int]] = None)
```

**Additional Parameters:**
- `char_to_idx`: Character to index mapping

**Additional Methods:**
- `_decode_predictions(predictions: torch.Tensor) -> List[str]`
- `_calculate_metrics(predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]`
- `_analyze_errors(predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, Any]`
- `generate_confusion_matrix() -> np.ndarray`

---

### Functions

#### `calculate_character_accuracy`
Calculate character-level accuracy.

```python
def calculate_character_accuracy(predictions: torch.Tensor,
                                targets: torch.Tensor,
                                pad_token_id: int = 11) -> float
```

**Parameters:**
- `predictions`: Model predictions [batch_size, seq_len, num_classes]
- `targets`: Target sequences [batch_size, seq_len]
- `pad_token_id`: ID of padding token to ignore

**Returns:** Character accuracy as float

---

#### `calculate_sequence_accuracy`
Calculate sequence-level accuracy.

```python
def calculate_sequence_accuracy(predictions: torch.Tensor,
                               targets: torch.Tensor,
                               pad_token_id: int = 11,
                               eos_token_id: int = 10) -> float
```

**Parameters:**
- `predictions`: Model predictions [batch_size, seq_len, num_classes]
- `targets`: Target sequences [batch_size, seq_len]
- `pad_token_id`: ID of padding token
- `eos_token_id`: ID of end-of-sequence token

**Returns:** Sequence accuracy as float

---

#### `calculate_edit_distance`
Calculate normalized edit distance.

```python
def calculate_edit_distance(pred_sequences: List[str],
                           target_sequences: List[str]) -> float
```

**Parameters:**
- `pred_sequences`: List of predicted sequences
- `target_sequences`: List of target sequences

**Returns:** Normalized edit distance [0, 1]

---

#### `setup_training_environment`
Set up training environment.

```python
def setup_training_environment(config: TrainingConfig) -> str
```

**Parameters:**
- `config`: Training configuration

**Returns:** Output directory path

**Side Effects:**
- Creates output directories
- Sets up logging
- Configures device

---

#### `setup_optimizer_and_scheduler`
Create optimizer and learning rate scheduler.

```python
def setup_optimizer_and_scheduler(model: nn.Module,
                                 config: TrainingConfig) -> Tuple[torch.optim.Optimizer, Optional[Any]]
```

**Parameters:**
- `model`: PyTorch model
- `config`: Training configuration

**Returns:** Tuple of (optimizer, scheduler)

---

### Factory Functions

#### `create_loss_function`
Factory function for loss functions.

```python
def create_loss_function(loss_type: str, **kwargs) -> nn.Module
```

**Parameters:**
- `loss_type`: Type of loss function
- `**kwargs`: Additional arguments for loss function

**Returns:** Loss function instance

---

#### `create_metrics_calculator`
Factory function for metrics calculators.

```python
def create_metrics_calculator(idx_to_char: Dict[int, str],
                             **kwargs) -> OCRMetrics
```

**Parameters:**
- `idx_to_char`: Character mapping
- `**kwargs`: Additional arguments

**Returns:** Metrics calculator instance

---

### Constants

```python
# Default character mappings
DEFAULT_CHAR_TO_IDX = {
    '០': 0, '១': 1, '២': 2, '៣': 3, '៤': 4, '៥': 5, '៦': 6, '៧': 7, '៨': 8, '៩': 9,
    '<EOS>': 10, '<PAD>': 11, '<BLANK>': 12
}

DEFAULT_IDX_TO_CHAR = {v: k for k, v in DEFAULT_CHAR_TO_IDX.items()}

# Special token IDs
PAD_TOKEN_ID = 11
EOS_TOKEN_ID = 10
BLANK_TOKEN_ID = 12
```

---

### Type Hints

```python
from typing import Dict, List, Optional, Tuple, Any, Union
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
```

---

### Usage Examples

#### Basic Usage
```python
from src.modules.trainers import OCRTrainer, TrainingConfig

config = TrainingConfig()
trainer = OCRTrainer(model, train_loader, val_loader, config, device)
results = trainer.train()
```

#### Custom Loss Function
```python
from src.modules.trainers import OCRLoss

loss_fn = OCRLoss(loss_type='focal', focal_gamma=2.0)
```

#### Metrics Calculation
```python
from src.modules.trainers import OCRMetrics

metrics = OCRMetrics(idx_to_char)
metrics.update(predictions, targets)
results = metrics.compute()
```

#### Configuration Management
```python
from src.modules.trainers import TrainingConfig

config = TrainingConfig.from_yaml('config/training_config.yaml')
config.num_epochs = 100
config.save_yaml('config/custom_config.yaml')
```

---

## Error Handling

### Common Exceptions

- `ValueError`: Invalid configuration parameters
- `FileNotFoundError`: Missing configuration or checkpoint files
- `RuntimeError`: CUDA/training related errors
- `KeyError`: Missing required configuration keys

### Exception Handling Patterns

```python
try:
    trainer = OCRTrainer(model, train_loader, val_loader, config, device)
    results = trainer.train()
except RuntimeError as e:
    logger.error(f"Training failed: {e}")
    # Handle CUDA out of memory, etc.
except ValueError as e:
    logger.error(f"Configuration error: {e}")
    # Handle invalid parameters
``` 