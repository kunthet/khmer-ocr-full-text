# Hyperparameter Tuning API Reference

## Overview

This document provides detailed API reference for the Khmer OCR hyperparameter tuning system implemented in Phase 3.1.

## Main Script: `phase3_hyperparameter_tuning.py`

### Command Line Interface

```bash
python src/sample_scripts/phase3_hyperparameter_tuning.py [OPTIONS]
```

#### Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--config` | str | `config/phase3_simple_configs.yaml` | Path to configuration file |
| `--experiments` | str[] | All experiments | List of experiment names to run |
| `--output-dir` | str | `training_output` | Output directory for results |
| `--results-file` | str | Auto-generated | Custom results file name |

#### Examples

```bash
# Run all experiments
python src/sample_scripts/phase3_hyperparameter_tuning.py

# Run specific experiments
python src/sample_scripts/phase3_hyperparameter_tuning.py --experiments conservative_small baseline_optimized

# Custom configuration
python src/sample_scripts/phase3_hyperparameter_tuning.py --config my_config.yaml --output-dir my_output
```

## Core Classes

### HyperparameterTuner

```python
class HyperparameterTuner:
    """Main class for managing hyperparameter tuning experiments."""
    
    def __init__(self, config_file: str = "config/phase3_simple_configs.yaml"):
        """
        Initialize the hyperparameter tuner.
        
        Args:
            config_file: Path to YAML configuration file
            
        Raises:
            FileNotFoundError: If configuration file doesn't exist
            yaml.YAMLError: If configuration file is invalid
        """
        
    def load_config(self, config_file: str) -> Dict:
        """
        Load and validate configuration from YAML file.
        
        Args:
            config_file: Path to configuration file
            
        Returns:
            Dictionary containing configuration
            
        Raises:
            yaml.YAMLError: If YAML syntax is invalid
        """
        
    def validate_config(self, config: Dict) -> None:
        """
        Validate configuration structure and required fields.
        
        Args:
            config: Configuration dictionary
            
        Raises:
            ValueError: If required fields are missing or invalid
        """
        
    def run_experiments(self, experiment_names: List[str] = None) -> Dict:
        """
        Run all or specified experiments.
        
        Args:
            experiment_names: List of experiment names to run. If None, runs all.
            
        Returns:
            Dictionary containing results from all experiments
        """
        
    def run_single_experiment(self, experiment_name: str, experiment_config: Dict) -> Dict:
        """
        Run a single hyperparameter experiment.
        
        Args:
            experiment_name: Name of the experiment
            experiment_config: Configuration dictionary for the experiment
            
        Returns:
            Dictionary containing experiment results
        """
        
    def save_results(self) -> str:
        """
        Save tuning results to JSON file.
        
        Returns:
            Path to the saved results file
        """
```

### TrainingConfig

```python
@dataclass
class TrainingConfig:
    """Configuration class for training parameters."""
    
    # Model configuration
    model_name: str = "medium"
    model_config_path: str = "config/model_config.yaml"
    vocab_size: int = 13
    max_sequence_length: int = 9
    
    # Data configuration
    metadata_path: str = "generated_data/metadata.yaml"
    train_split: str = "train"
    val_split: str = "val"
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = False
    augmentation: bool = True
    
    # Training configuration
    device: str = "auto"
    mixed_precision: bool = False
    num_epochs: int = 40
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0
    log_every_n_steps: int = 25
    save_every_n_epochs: int = 5
    keep_n_checkpoints: int = 3
    use_tensorboard: bool = True
    
    # Loss configuration
    loss_type: str = "crossentropy"
    label_smoothing: float = 0.0
    
    # Optimizer configuration
    optimizer_type: str = "adam"
    optimizer_betas: List[float] = field(default_factory=lambda: [0.9, 0.999])
    
    # Scheduler configuration
    scheduler_type: str = "steplr"
    warmup_epochs: int = 0
    min_lr: float = 1e-6
    step_size: int = 10
    gamma: float = 0.5
    patience: int = 5
    factor: float = 0.5
    
    # Early stopping
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-4
    early_stopping_monitor: str = "val_char_accuracy"
    early_stopping_mode: str = "max"
    
    # Output configuration
    experiment_name: str = "hyperparameter_experiment"
    output_dir: str = "training_output"
```

## Utility Functions

### Configuration Creation

```python
def create_training_config(experiment_config: Dict) -> TrainingConfig:
    """
    Create TrainingConfig object from experiment configuration dictionary.
    
    Args:
        experiment_config: Dictionary containing experiment configuration
        
    Returns:
        TrainingConfig object with specified parameters
        
    Example:
        config_dict = {
            'training': {'batch_size': 64, 'learning_rate': 0.002},
            'model': {'name': 'medium'},
            'optimizer': {'type': 'adamw'}
        }
        training_config = create_training_config(config_dict)
    """
```

### Environment Setup

```python
def setup_training_environment(config: TrainingConfig) -> Tuple[torch.device, bool]:
    """
    Setup training environment including device and precision settings.
    
    Args:
        config: TrainingConfig object
        
    Returns:
        Tuple of (device, use_amp) where device is torch.device and use_amp is boolean
        
    Example:
        device, use_amp = setup_training_environment(config)
        print(f"Training on: {device}")
    """
```

### Data Loading

```python
def create_data_loaders(config: TrainingConfig) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation data loaders.
    
    Args:
        config: TrainingConfig object
        
    Returns:
        Tuple of (train_loader, val_loader)
        
    Example:
        train_loader, val_loader = create_data_loaders(config)
        print(f"Training batches: {len(train_loader)}")
    """
```

### Model Creation

```python
def create_model_from_config(config: TrainingConfig) -> torch.nn.Module:
    """
    Create model from configuration.
    
    Args:
        config: TrainingConfig object
        
    Returns:
        PyTorch model instance
        
    Example:
        model = create_model_from_config(config)
        total_params = sum(p.numel() for p in model.parameters())
    """
```

### Training Components

```python
def create_optimizer(model: torch.nn.Module, config: TrainingConfig) -> torch.optim.Optimizer:
    """
    Create optimizer from configuration.
    
    Args:
        model: PyTorch model
        config: TrainingConfig object
        
    Returns:
        PyTorch optimizer instance
    """

def create_scheduler(optimizer: torch.optim.Optimizer, config: TrainingConfig) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Create learning rate scheduler from configuration.
    
    Args:
        optimizer: PyTorch optimizer
        config: TrainingConfig object
        
    Returns:
        PyTorch scheduler instance
    """

def create_loss_function(config: TrainingConfig) -> torch.nn.Module:
    """
    Create loss function from configuration.
    
    Args:
        config: TrainingConfig object
        
    Returns:
        PyTorch loss function instance
    """
```

## Configuration Schema

### YAML Configuration Structure

```yaml
experiments:
  experiment_name:
    experiment_name: str                    # Unique experiment identifier
    
    model:
      name: str                            # Model preset: "small"|"medium"|"large"|"ctc"
      config_path: str                     # Path to model configuration file
    
    data:
      metadata_path: str                   # Path to dataset metadata
      train_split: str                     # Training split name
      val_split: str                       # Validation split name
      num_workers: int                     # DataLoader workers
      pin_memory: bool                     # Pin memory for DataLoader
      augmentation: bool                   # Enable data augmentation
    
    training:
      device: str                          # Device: "auto"|"cpu"|"cuda"
      mixed_precision: bool                # Enable mixed precision training
      gradient_clip_norm: float            # Gradient clipping norm
      log_every_n_steps: int               # Logging frequency
      save_every_n_epochs: int             # Checkpoint saving frequency
      keep_n_checkpoints: int              # Number of checkpoints to keep
      use_tensorboard: bool                # Enable TensorBoard logging
      batch_size: int                      # Training batch size
      learning_rate: float                 # Initial learning rate
      weight_decay: float                  # L2 regularization strength
      num_epochs: int                      # Maximum number of epochs
      loss_type: str                       # Loss function type
      label_smoothing: float               # Label smoothing factor
    
    optimizer:
      type: str                            # Optimizer type: "adam"|"adamw"
      learning_rate: float                 # Learning rate (overrides training.learning_rate)
      weight_decay: float                  # Weight decay (overrides training.weight_decay)
      betas: [float, float]                # Adam beta parameters
    
    scheduler:
      type: str                            # Scheduler type: "cosine"|"steplr"|"plateau"
      warmup_epochs: int                   # Warmup epochs for cosine scheduler
      min_lr: float                        # Minimum learning rate
      step_size: int                       # Step size for StepLR
      gamma: float                         # Decay factor for StepLR
      patience: int                        # Patience for ReduceLROnPlateau
      factor: float                        # Factor for ReduceLROnPlateau
    
    early_stopping:
      patience: int                        # Early stopping patience
      min_delta: float                     # Minimum improvement threshold
      monitor: str                         # Metric to monitor
      mode: str                            # Mode: "max"|"min"

targets:
  character_accuracy: float                # Target character accuracy
  sequence_accuracy: float                 # Target sequence accuracy
  training_time_per_epoch: int             # Target time per epoch (seconds)
  convergence_epochs: int                  # Target convergence epochs
```

### Experiment Presets

#### conservative_small
```yaml
model: {name: "small"}
training: {batch_size: 32, learning_rate: 0.001, num_epochs: 40}
optimizer: {type: "adam"}
scheduler: {type: "plateau", patience: 5}
```

#### baseline_optimized
```yaml
model: {name: "medium"}
training: {batch_size: 64, learning_rate: 0.002, num_epochs: 30}
optimizer: {type: "adamw"}
scheduler: {type: "cosine", warmup_epochs: 3}
```

#### high_learning_rate
```yaml
model: {name: "medium"}
training: {batch_size: 96, learning_rate: 0.003, num_epochs: 25}
optimizer: {type: "adam"}
scheduler: {type: "steplr", step_size: 8}
```

## Results Schema

### JSON Results Structure

```json
{
  "timestamp": "YYYYMMDD_HHMMSS",
  "experiments_completed": int,
  "best_result": {
    "experiment_name": str,
    "status": "completed"|"failed",
    "training_time": float,
    "best_val_char_accuracy": float,
    "best_val_seq_accuracy": float,
    "final_train_loss": float,
    "total_epochs": int,
    "hyperparameters": {
      "model_name": str,
      "batch_size": int,
      "learning_rate": float,
      "weight_decay": float,
      "loss_type": str,
      "scheduler_type": str,
      "optimizer_type": str
    }
  },
  "all_results": [
    {
      "experiment_name": str,
      "status": str,
      "training_time": float,
      "best_val_char_accuracy": float,
      "best_val_seq_accuracy": float,
      "final_train_loss": float,
      "total_epochs": int,
      "error_message": str,
      "hyperparameters": {...}
    }
  ]
}
```

## Error Handling

### Exception Types

```python
class ConfigurationError(Exception):
    """Raised when configuration is invalid or missing required fields."""
    pass

class ExperimentFailedError(Exception):
    """Raised when an experiment fails during execution."""
    pass

class ModelCreationError(Exception):
    """Raised when model creation fails."""
    pass

class DataLoaderError(Exception):
    """Raised when data loading fails."""
    pass
```

### Error Recovery

The hyperparameter tuning system implements graceful error recovery:

1. **Configuration Errors**: Validation before experiment execution
2. **Experiment Failures**: Continue with remaining experiments
3. **Resource Errors**: Automatic fallback configurations
4. **Logging**: Comprehensive error logging for debugging

## Performance Monitoring

### Metrics Tracked

- **Training Metrics**:
  - Loss (training and validation)
  - Character accuracy (training and validation)
  - Sequence accuracy (training and validation)
  - Learning rate
  - Training time per epoch

- **System Metrics**:
  - Memory usage
  - CPU utilization
  - Training time
  - Model parameters

### TensorBoard Integration

```python
# TensorBoard logs are automatically created at:
# training_output/{experiment_name}/tensorboard/

# View with:
tensorboard --logdir training_output/{experiment_name}/tensorboard
```

## Integration Examples

### Custom Experiment Script

```python
#!/usr/bin/env python3
"""Custom hyperparameter tuning script."""

import yaml
from src.sample_scripts.phase3_hyperparameter_tuning import HyperparameterTuner

def main():
    # Create custom configuration
    custom_config = {
        'experiments': {
            'my_experiment': {
                'experiment_name': 'my_experiment',
                'model': {'name': 'medium'},
                'training': {
                    'batch_size': 48,
                    'learning_rate': 0.0015,
                    'num_epochs': 35
                },
                'optimizer': {'type': 'adamw'},
                'scheduler': {'type': 'cosine', 'warmup_epochs': 2}
            }
        },
        'targets': {
            'character_accuracy': 0.85,
            'sequence_accuracy': 0.70
        }
    }
    
    # Save configuration
    with open('custom_config.yaml', 'w') as f:
        yaml.dump(custom_config, f)
    
    # Run experiments
    tuner = HyperparameterTuner('custom_config.yaml')
    results = tuner.run_experiments(['my_experiment'])
    
    print(f"Experiment completed: {results['best_result']['best_val_char_accuracy']:.4f}")

if __name__ == "__main__":
    main()
```

### Results Analysis Integration

```python
import json
from pathlib import Path

def load_latest_results():
    """Load the most recent hyperparameter tuning results."""
    results_dir = Path('.')
    result_files = list(results_dir.glob('hyperparameter_tuning_results_*.json'))
    
    if not result_files:
        return None
        
    latest_file = max(result_files, key=lambda f: f.stat().st_mtime)
    
    with open(latest_file) as f:
        return json.load(f)

def get_best_hyperparameters():
    """Get hyperparameters from the best performing experiment."""
    results = load_latest_results()
    if results and results['best_result']:
        return results['best_result']['hyperparameters']
    return None
```

---

*This API reference provides complete documentation for the hyperparameter tuning system. For usage examples and troubleshooting, see the main [Hyperparameter Tuning Guide](hyperparameter_tuning_guide.md).* 