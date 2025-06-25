# Hyperparameter Tuning Documentation

## Overview

The Khmer OCR hyperparameter tuning system provides automated optimization for the Phase 3.1 training pipeline. This system was designed to improve beyond the baseline 24% character accuracy achieved in Phase 2.

## Quick Start

### Basic Usage

```bash
# Run all predefined experiments
python src/sample_scripts/phase3_hyperparameter_tuning.py

# Run specific experiments
python src/sample_scripts/phase3_hyperparameter_tuning.py --experiments conservative_small baseline_optimized

# Use custom configuration
python src/sample_scripts/phase3_hyperparameter_tuning.py --config my_config.yaml
```

### Monitor Progress

```bash
# Check training logs
tail -f training_output/*/logs/training_*.log

# Launch TensorBoard
tensorboard --logdir training_output/
```

## Configuration

### Main Configuration File: `config/phase3_simple_configs.yaml`

```yaml
experiments:
  experiment_name:
    experiment_name: "unique_name"
    
    model:
      name: "small|medium|large|ctc"
      config_path: "config/model_config.yaml"
    
    training:
      batch_size: 32
      learning_rate: 0.001
      num_epochs: 40
      loss_type: "crossentropy"
      label_smoothing: 0.05
    
    optimizer:
      type: "adam|adamw"
      learning_rate: 0.001
      weight_decay: 0.0001
    
    scheduler:
      type: "cosine|steplr|plateau"
      warmup_epochs: 3
      step_size: 10
      gamma: 0.5
    
    early_stopping:
      patience: 8
      monitor: "val_char_accuracy"
      mode: "max"

targets:
  character_accuracy: 0.85
  sequence_accuracy: 0.70
```

## Predefined Experiments

### 1. Conservative Small
- **Purpose**: Stable baseline with small model
- **Model**: Small (12.5M parameters)
- **Batch Size**: 32
- **Learning Rate**: 0.001
- **Scheduler**: Plateau (patient)
- **Best For**: Reliable convergence, limited resources

### 2. Baseline Optimized
- **Purpose**: Balanced performance optimization
- **Model**: Medium (16.2M parameters)
- **Batch Size**: 64
- **Learning Rate**: 0.002
- **Scheduler**: Cosine with warmup
- **Best For**: Standard optimization target

### 3. High Learning Rate
- **Purpose**: Aggressive training for fast convergence
- **Model**: Medium (16.2M parameters)
- **Batch Size**: 96
- **Learning Rate**: 0.003
- **Scheduler**: StepLR
- **Best For**: Quick experimentation

## Results Analysis

### Results File Format

```json
{
  "timestamp": "20250623_214947",
  "experiments_completed": 1,
  "best_result": {
    "experiment_name": "conservative_small",
    "status": "completed",
    "training_time": 3421.5,
    "best_val_char_accuracy": 0.4521,
    "best_val_seq_accuracy": 0.1834,
    "hyperparameters": {
      "model_name": "small",
      "batch_size": 32,
      "learning_rate": 0.001
    }
  }
}
```

### Key Metrics

- **best_val_char_accuracy**: Best validation character accuracy
- **best_val_seq_accuracy**: Best validation sequence accuracy
- **training_time**: Total training time in seconds
- **status**: Experiment status (`completed`, `failed`)

## Performance Guidelines

### CPU-Optimized Settings

| Model Size | Batch Size | Learning Rate | Workers |
|------------|------------|---------------|---------|
| Small      | 32-64      | 0.001-0.003   | 2-4     |
| Medium     | 32-96      | 0.0008-0.003  | 2-4     |
| Large      | 16-32      | 0.0005-0.002  | 2-4     |

### Memory Management

```yaml
# For limited memory
batch_size: 16
num_workers: 2
pin_memory: false
mixed_precision: false
```

## Troubleshooting

### Common Issues

1. **YAML Configuration Errors**
   ```bash
   # Validate YAML syntax
   python -c "import yaml; yaml.safe_load(open('config/phase3_simple_configs.yaml'))"
   ```

2. **Dataset Loading Issues**
   ```python
   # Check dataset parameters
   train_dataset = KhmerDigitsDataset(
       metadata_path="generated_data/metadata.yaml",
       split='train',
       transform=get_train_transforms()  # Not apply_transforms=True
   )
   ```

3. **Memory Issues**
   ```yaml
   # Reduce memory usage
   batch_size: 16
   num_workers: 2
   ```

4. **Slow Training**
   ```yaml
   # Speed optimizations
   num_epochs: 20
   batch_size: 64
   log_every_n_steps: 50
   ```

### Debug Commands

```bash
# Check environment
python -c "import torch; print(f'PyTorch: {torch.__version__}')"

# Test dataset loading
python -c "
from modules.data_utils import KhmerDigitsDataset
dataset = KhmerDigitsDataset('generated_data/metadata.yaml', split='train')
print(f'Dataset: {len(dataset)} samples')"

# Test model creation
python -c "
from models import create_model
model = create_model(preset='small', vocab_size=13, max_sequence_length=9)
print('Model created successfully')"
```

## API Reference

### Main Class: HyperparameterTuner

```python
class HyperparameterTuner:
    def __init__(self, config_file: str):
        """Initialize the tuner with configuration file."""
        
    def run_experiments(self, experiment_names: List[str] = None):
        """Run all or specified experiments."""
        
    def save_results(self):
        """Save results to JSON file."""
```

### Configuration Functions

```python
def create_training_config(experiment_config: Dict) -> TrainingConfig:
    """Create TrainingConfig from experiment configuration."""

def run_single_experiment(experiment_name: str, experiment_config: Dict) -> Dict:
    """Run a single hyperparameter experiment."""
```

## Best Practices

### 1. Systematic Approach
- Start with `conservative_small` for baseline
- Use `baseline_optimized` for standard performance
- Try `high_learning_rate` for quick validation

### 2. Resource Management
- Monitor system resources during training
- Use appropriate batch sizes for available RAM
- Consider training time vs. accuracy trade-offs

### 3. Iterative Improvement
- Analyze failed experiments for insights
- Build on best-performing combinations
- Document results and reasoning

## Examples

### Custom Experiment Configuration

```yaml
experiments:
  my_experiment:
    experiment_name: "my_experiment"
    model:
      name: "medium"
    training:
      batch_size: 48
      learning_rate: 0.0015
      num_epochs: 35
      loss_type: "crossentropy"
      label_smoothing: 0.08
    optimizer:
      type: "adamw"
    scheduler:
      type: "cosine"
      warmup_epochs: 2
    early_stopping:
      patience: 6
      monitor: "val_char_accuracy"
      mode: "max"
```

### Results Analysis Script

```python
import json
import glob

def analyze_results():
    # Load latest results
    result_files = glob.glob("hyperparameter_tuning_results_*.json")
    latest_file = max(result_files)
    
    with open(latest_file, 'r') as f:
        results = json.load(f)
    
    # Print best result
    if results['best_result']:
        best = results['best_result']
        print(f"Best: {best['experiment_name']}")
        print(f"Accuracy: {best['best_val_char_accuracy']:.4f}")
        print(f"Time: {best['training_time']:.1f}s")

if __name__ == "__main__":
    analyze_results()
```

## Integration with Existing System

The hyperparameter tuning system integrates seamlessly with:

- **Data Pipeline**: Uses existing `KhmerDigitsDataset` and transforms
- **Model Architecture**: Works with all model presets (small, medium, large, ctc)
- **Training Infrastructure**: Leverages existing trainers and loss functions
- **Logging**: Integrates with TensorBoard and text logging
- **Checkpointing**: Uses existing checkpoint management

## File Structure

```
hyperparameter_tuning/
├── config/
│   ├── phase3_simple_configs.yaml      # Main configuration
│   └── phase3_training_configs.yaml    # Advanced (legacy)
├── src/sample_scripts/
│   └── phase3_hyperparameter_tuning.py # Main script
├── training_output/
│   └── [experiment_name]/              # Results
│       ├── checkpoints/
│       ├── logs/
│       ├── tensorboard/
│       └── configs/
└── hyperparameter_tuning_results_*.json # Summary results
```

## Performance Targets

- **Baseline**: 24% character accuracy (Phase 2 achievement)
- **Target**: 85% character accuracy, 70% sequence accuracy
- **Training Time**: <300 seconds per epoch on CPU
- **Convergence**: Within 20 epochs

---

*This documentation covers the complete hyperparameter tuning system for Khmer OCR Phase 3.1. For additional details, see the implementation in `src/sample_scripts/phase3_hyperparameter_tuning.py`.* 