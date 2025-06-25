# Training Infrastructure Documentation

## Overview

The `src/modules/trainers` module provides comprehensive training infrastructure for the Khmer Digits OCR project. It implements a flexible, scalable training system with support for multiple loss functions, evaluation metrics, checkpointing, early stopping, and TensorBoard logging.

## Architecture

### Design Principles

The training infrastructure follows these key design principles:

1. **Modularity**: Each component is self-contained and can be used independently
2. **Extensibility**: Easy to add new loss functions, metrics, and trainers
3. **Configuration-Driven**: All training parameters configurable via YAML files
4. **Production-Ready**: Robust error handling, logging, and monitoring
5. **OCR-Optimized**: Specialized for sequence-to-sequence OCR tasks

### Component Architecture

```
src/modules/trainers/
├── __init__.py           # Module exports and factory functions
├── losses.py             # Loss functions for OCR training
├── metrics.py            # Evaluation metrics and accuracy calculation
├── utils.py              # Training utilities and configuration
├── base_trainer.py       # Abstract base trainer class
└── ocr_trainer.py        # Specialized OCR trainer implementation
```

## Components

### 1. Loss Functions (`losses.py`)

The module provides multiple loss functions optimized for OCR tasks:

#### CrossEntropyLoss
- **Purpose**: Standard classification loss with sequence masking
- **Features**: 
  - PAD token masking to ignore padding positions
  - Label smoothing for regularization
  - Support for variable-length sequences
- **Use Case**: Standard training for attention-based models

#### CTCLoss
- **Purpose**: Connectionist Temporal Classification for alignment-free training
- **Features**:
  - No need for character-level alignment
  - Handles variable-length input/output sequences
  - Built-in blank token handling
- **Use Case**: Training without explicit alignment information

#### FocalLoss
- **Purpose**: Addresses class imbalance by focusing on hard examples
- **Features**:
  - Configurable focusing parameter (gamma)
  - Class weighting support (alpha)
  - Reduces impact of easy examples
- **Use Case**: When dealing with imbalanced character distributions
- **Use Case**: Imbalanced character distributions
- 
#### OCRLoss (Unified Wrapper)
- **Purpose**: Factory and wrapper for all loss functions
- **Features**:
  - Automatic loss function selection
  - Consistent interface across all loss types
  - Configuration-driven instantiation
- **Use Case**: Simplified loss function management

### 2. Evaluation Metrics (`metrics.py`)

Comprehensive metrics for OCR evaluation:

#### Character Accuracy
- **Calculation**: Correct characters / total non-PAD characters
- **Features**: 
  - Ignores special tokens (PAD, EOS)
  - Per-batch and accumulated calculation
  - Handles variable-length sequences
- Correct characters / total non-PAD characters
- Ignores special tokens, handles variable-length sequences
- 
#### Sequence Accuracy
- **Calculation**: Exact sequence matches / total sequences
- **Features**:
  - Strict exact match evaluation
  - Proper sequence termination handling
  - Binary accuracy metric
- Exact sequence matches / total sequences
- Strict exact match evaluation
- 
#### Edit Distance
- **Implementation**: Levenshtein distance algorithm
- **Features**:
  - Normalized to [0, 1] range
  - Character-level string comparison
  - Handles sequence length differences

#### OCRMetrics Class
- **Purpose**: Comprehensive metrics tracking and reporting
- **Features**:
  - Batch accumulation
  - Confusion matrix generation
  - Per-class accuracy tracking
  - Statistics computation

### 3. Training Utilities (`utils.py`)

Essential utilities for training management:

#### TrainingConfig
- **Purpose**: Centralized configuration management
- **Features**:
  - Dataclass-based with type hints
  - YAML serialization/deserialization
  - Validation and default values
  - Environment variable support

#### CheckpointManager
- **Purpose**: Automatic model checkpointing
- **Features**:
  - Configurable checkpoint retention
  - Best model preservation
  - Automatic cleanup of old checkpoints
  - Metadata storage

#### EarlyStopping
- **Purpose**: Prevents overfitting through early termination
- **Features**:
  - Configurable patience and minimum delta
  - Support for both min/max metrics
  - Best weights restoration
  - Training history tracking

### 4. Base Trainer (`base_trainer.py`)

Abstract foundation for all trainers:

#### Core Features
- **Training Loop**: Complete train/validation cycle implementation
- **Mixed Precision**: Automatic mixed precision training support
- **Gradient Management**: Clipping and accumulation
- **Learning Rate Scheduling**: Multiple scheduler support
- **Logging**: TensorBoard integration with metrics tracking
- **Error Handling**: Robust exception handling and recovery

#### Architecture
```python
class BaseTrainer(ABC):
    def __init__(self, model, train_loader, val_loader, config, device)
    def train(self) -> Dict[str, Any]
    def _train_epoch(self) -> Dict[str, float]
    def _validate_epoch(self) -> Dict[str, float]
    def _create_loss_function(self) -> nn.Module
    def _create_metrics_calculator(self) -> Any
```

### 5. OCR Trainer (`ocr_trainer.py`)

Specialized trainer for OCR tasks:

#### OCR-Specific Features
- **Character Mapping**: Automatic vocabulary management
- **Sequence Processing**: Variable-length sequence handling
- **Error Analysis**: Character-level error categorization
- **Confusion Matrix**: Visual error pattern analysis

#### Specialized Methods
```python
class OCRTrainer(BaseTrainer):
    def _decode_predictions(self, predictions) -> List[str]
    def _calculate_metrics(self, predictions, targets) -> Dict[str, float]
    def _analyze_errors(self, predictions, targets) -> Dict[str, Any]
    def _generate_confusion_matrix(self, predictions, targets) -> np.ndarray
```

## Configuration System

### Training Configuration Schema

```yaml
# Model Configuration
model_name: "medium"  # small, medium, large, ctc_small, ctc_medium
model_config_path: "config/model_config.yaml"

# Data Configuration
metadata_path: "generated_data/metadata.yaml"
batch_size: 32
num_workers: 4

# Training Configuration
num_epochs: 50
learning_rate: 0.001
weight_decay: 0.0001
gradient_clip_norm: 1.0

# Loss Configuration
loss_type: "crossentropy"  # crossentropy, ctc, focal
label_smoothing: 0.0

# Scheduler Configuration
scheduler_type: "steplr"  # steplr, cosine, plateau
step_size: 10
gamma: 0.5

# Early Stopping
early_stopping_patience: 10
early_stopping_min_delta: 0.0001

# Checkpointing
save_every_n_epochs: 5
keep_n_checkpoints: 3

# Logging
log_every_n_steps: 50
use_tensorboard: true

# Paths
output_dir: "training_output"
experiment_name: "khmer_ocr_experiment"

# Device and Performance
device: "auto"  # auto, cuda, cpu
mixed_precision: true
```

### Configuration Loading

```python
from src.modules.trainers import TrainingConfig

# Load from YAML
config = TrainingConfig.from_yaml("config/training_config.yaml")

# Load from dictionary
config_dict = {...}
config = TrainingConfig.from_dict(config_dict)

# Default configuration
config = TrainingConfig()
```

## Usage Examples

### Basic Training Setup

```python
import torch
from src.modules.trainers import OCRTrainer, TrainingConfig
from src.modules.data_utils import create_data_loaders
from src.models import create_model

# Load configuration
config = TrainingConfig.from_yaml("config/training_config.yaml")

# Create data loaders
train_loader, val_loader = create_data_loaders(
    metadata_path=config.metadata_path,
    batch_size=config.batch_size,
    num_workers=config.num_workers
)

# Create model
model = create_model(config.model_name)

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create trainer
trainer = OCRTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    config=config,
    device=device
)

# Start training
results = trainer.train()
print(f"Best validation loss: {results['best_val_loss']:.4f}")
```

### Custom Loss Function Training

```python
# Create custom configuration
config = TrainingConfig()
config.model_name = "large"
config.loss_type = "focal"
config.focal_gamma = 2.0
config.learning_rate = 0.0005
config.mixed_precision = True

# Save and use
config.save_yaml("config/custom_config.yaml")
```

### Metrics Analysis

```python
from src.modules.trainers import OCRMetrics

idx_to_char = {0: '០', 1: '១', 2: '២', ...}
metrics = OCRMetrics(idx_to_char)

# Update with predictions
metrics.update(predictions, targets)

# Get results
results = metrics.compute()
print(f"Character Accuracy: {results['char_accuracy']:.3f}")
print(f"Sequence Accuracy: {results['seq_accuracy']:.3f}")
```

## Performance Optimization

### Memory Optimization
- Enable mixed precision: `config.mixed_precision = True`
- Optimize data loading: `config.pin_memory = True`
- Gradient accumulation: `config.gradient_accumulation_steps = 4`

### Training Speed
- Efficient data loading: `config.num_workers = 4`
- Optimized scheduling: `config.scheduler_type = "cosine"`
- Reduce logging frequency: `config.log_every_n_steps = 100`

## Error Handling

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size
   - Enable gradient checkpointing
   - Use mixed precision training

2. **Slow Training**
   - Increase num_workers
   - Enable pin_memory
   - Check data loading bottlenecks

3. **Poor Convergence**
   - Try different loss functions
   - Adjust learning rate schedules
   - Enable gradient clipping

### Debugging

Enable debug mode for detailed logging:
```python
config.debug_mode = True
config.log_every_n_steps = 1
```

## Best Practices

### Training Configuration
1. Start with defaults, adjust incrementally
2. Use mixed precision for speed
3. Monitor GPU memory usage
4. Enable early stopping

### Experimental Workflow
1. Run short experiments first
2. Monitor training curves
3. Save checkpoints frequently
4. Version control configurations

### Production Deployment
1. Use frozen configurations
2. Implement proper logging
3. Set up automated cleanup
4. Document all changes

## Integration

### With Data Pipeline
```python
from src.modules.data_utils import KhmerDigitsDataset
from src.modules.trainers import setup_training_environment

output_dir = setup_training_environment(config)
```

### With Model Factory
```python
from src.models import create_model

model = create_model(config.model_name)
trainer = OCRTrainer(model, train_loader, val_loader, config, device)
```

## Testing

Run comprehensive tests:
```bash
python src/sample_scripts/test_training_infrastructure.py
```

## Documentation Files

- `trainers_documentation.md`: This comprehensive guide
- `trainers_api_reference.md`: Detailed API documentation
- `trainers_quick_reference.md`: Quick usage patterns
- `trainers_examples.md`: Practical examples

## Quick Reference

### Key Classes
- `OCRTrainer`: Main trainer for OCR tasks
- `OCRLoss`: Unified loss function wrapper
- `OCRMetrics`: Comprehensive metrics calculation
- `TrainingConfig`: Configuration management
- `CheckpointManager`: Model checkpointing
- `EarlyStopping`: Training termination control

### Main Functions
- `setup_training_environment()`: Environment initialization
- `calculate_character_accuracy()`: Character-level accuracy
- `calculate_sequence_accuracy()`: Sequence-level accuracy

### Configuration Files
- `config/training_config.yaml`: Main training configuration
- Custom configurations can be created by copying and modifying

This documentation provides comprehensive coverage of the training infrastructure. For specific implementation details, refer to the source code and API reference. 