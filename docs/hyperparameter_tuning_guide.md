# Hyperparameter Tuning Guide for Khmer OCR

## Overview

The Khmer OCR hyperparameter tuning system provides a comprehensive framework for systematically optimizing model performance through automated experimentation. This system was designed specifically for Phase 3.1 to improve beyond the baseline 24% character accuracy achieved in Phase 2.

## 📋 Table of Contents

1. [System Architecture](#system-architecture)
2. [Quick Start](#quick-start)
3. [Configuration Files](#configuration-files)
4. [Experiment Types](#experiment-types)
5. [Usage Examples](#usage-examples)
6. [Results Analysis](#results-analysis)
7. [Performance Optimization](#performance-optimization)
8. [Troubleshooting](#troubleshooting)
9. [API Reference](#api-reference)

## System Architecture

### Components

```
hyperparameter_tuning/
├── config/
│   ├── phase3_simple_configs.yaml      # Main experiment configurations
│   └── phase3_training_configs.yaml    # Advanced configurations (complex YAML)
├── src/sample_scripts/
│   └── phase3_hyperparameter_tuning.py # Main tuning script
├── training_output/
│   └── [experiment_name]/              # Individual experiment results
│       ├── checkpoints/                # Model checkpoints
│       ├── logs/                       # Training logs
│       ├── tensorboard/                # TensorBoard events
│       └── configs/                    # Saved configurations
└── results/
    └── hyperparameter_tuning_results_*.json  # Results summaries
```

### Key Features

- **Systematic Experimentation**: Automated execution of multiple hyperparameter combinations
- **CPU Optimization**: Configurations specifically tuned for CPU training environments
- **Results Tracking**: Comprehensive logging and JSON export of all experiment results
- **Error Recovery**: Robust error handling with graceful failure recovery
- **Integration**: Seamless integration with existing data pipeline and training infrastructure

## Quick Start

### 1. Basic Usage

```bash
# Run all configured experiments
python src/sample_scripts/phase3_hyperparameter_tuning.py

# Run specific experiments
python src/sample_scripts/phase3_hyperparameter_tuning.py --experiments conservative_small baseline_optimized

# Use custom configuration file
python src/sample_scripts/phase3_hyperparameter_tuning.py --config config/my_custom_configs.yaml
```

### 2. Monitor Progress

```bash
# Check experiment status
ls training_output/

# View logs in real-time
tail -f training_output/[experiment_name]/logs/training_*.log

# Launch TensorBoard
tensorboard --logdir training_output/[experiment_name]/tensorboard
```

### 3. Analyze Results

```bash
# Check generated results files
ls hyperparameter_tuning_results_*.json

# View best results
python -c "
import json
with open('hyperparameter_tuning_results_*.json', 'r') as f:
    results = json.load(f)
    print('Best Result:', results['best_result'])
"
```

## Configuration Files

### Main Configuration: `config/phase3_simple_configs.yaml`

```yaml
experiments:
  # Experiment definition
  experiment_name:
    experiment_name: "unique_name"
    
    # Model configuration
    model:
      name: "small|medium|large|ctc"
      config_path: "config/model_config.yaml"
    
    # Data configuration
    data:
      metadata_path: "generated_data/metadata.yaml"
      train_split: "train"
      val_split: "val"
      num_workers: 4
      pin_memory: false
      augmentation: true
    
    # Training configuration
    training:
      device: "auto"
      mixed_precision: false
      gradient_clip_norm: 1.0
      log_every_n_steps: 25
      save_every_n_epochs: 5
      keep_n_checkpoints: 3
      use_tensorboard: true
      batch_size: 32
      learning_rate: 0.001
      weight_decay: 0.0001
      num_epochs: 40
      loss_type: "crossentropy"
      label_smoothing: 0.05
    
    # Optimizer configuration
    optimizer:
      type: "adam|adamw"
      learning_rate: 0.001
      weight_decay: 0.0001
      betas: [0.9, 0.999]
    
    # Scheduler configuration
    scheduler:
      type: "cosine|steplr|plateau"
      warmup_epochs: 3
      min_lr: 1e-6
      step_size: 10
      gamma: 0.5
      patience: 5
      factor: 0.5
    
    # Early stopping
    early_stopping:
      patience: 8
      min_delta: 0.001
      monitor: "val_char_accuracy"
      mode: "max"

# Performance targets
targets:
  character_accuracy: 0.85
  sequence_accuracy: 0.70
  training_time_per_epoch: 300
  convergence_epochs: 20
```

### Configuration Parameters

#### Model Parameters
- **name**: Model preset (`small`, `medium`, `large`, `ctc`)
  - `small`: 12.5M parameters, ResNet-18 + BiLSTM(128)
  - `medium`: 16.2M parameters, ResNet-18 + BiLSTM(256)
  - `large`: 30M+ parameters, EfficientNet-B0 + BiLSTM(512)
  - `ctc`: 12.3M parameters, CTC decoder variant

#### Training Parameters
- **batch_size**: Training batch size (16-128, CPU: 32-96 recommended)
- **learning_rate**: Initial learning rate (0.0001-0.005)
- **weight_decay**: L2 regularization strength (0.00001-0.001)
- **num_epochs**: Maximum training epochs (20-50)
- **loss_type**: Loss function (`crossentropy`, `focal`, `ctc`)
- **label_smoothing**: Label smoothing factor (0.0-0.2)

#### Optimizer Parameters
- **type**: Optimizer type (`adam`, `adamw`)
- **betas**: Adam beta parameters [β1, β2]

#### Scheduler Parameters
- **type**: Learning rate scheduler
  - `cosine`: Cosine annealing with warmup
  - `steplr`: Step decay at fixed intervals
  - `plateau`: Reduce on validation plateau

## Experiment Types

### 1. Conservative Small (`conservative_small`)

**Purpose**: Stable baseline with small model for reliable convergence

**Configuration**:
- Model: Small (12.5M parameters)
- Learning Rate: 0.001 (conservative)
- Batch Size: 32
- Epochs: 40
- Scheduler: Plateau (patient convergence)
- Label Smoothing: 0.05

**Best For**: 
- Establishing reliable baseline
- Limited computational resources
- Stable, patient training

### 2. Baseline Optimized (`baseline_optimized`)

**Purpose**: Balanced approach with optimized hyperparameters

**Configuration**:
- Model: Medium (16.2M parameters)
- Learning Rate: 0.002 (optimized)
- Batch Size: 64
- Epochs: 30
- Scheduler: Cosine with warmup
- Label Smoothing: 0.1

**Best For**:
- Standard optimization target
- Good balance of speed and accuracy
- Recommended starting point

### 3. High Learning Rate (`high_learning_rate`)

**Purpose**: Aggressive training for fast convergence

**Configuration**:
- Model: Medium (16.2M parameters)
- Learning Rate: 0.003 (aggressive)
- Batch Size: 96
- Epochs: 25
- Scheduler: StepLR
- Label Smoothing: 0.15

**Best For**:
- Fast experimentation
- When computational time is limited
- High-confidence in data quality

## Usage Examples

### Running Single Experiment

```bash
# Conservative approach for stable results
python src/sample_scripts/phase3_hyperparameter_tuning.py \
    --experiments conservative_small

# Quick optimization test
python src/sample_scripts/phase3_hyperparameter_tuning.py \
    --experiments high_learning_rate
```

### Running Multiple Experiments

```bash
# Run all experiments sequentially
python src/sample_scripts/phase3_hyperparameter_tuning.py

# Run specific subset
python src/sample_scripts/phase3_hyperparameter_tuning.py \
    --experiments conservative_small baseline_optimized

# Custom configuration file
python src/sample_scripts/phase3_hyperparameter_tuning.py \
    --config config/my_experiments.yaml \
    --experiments my_custom_experiment
```

### Creating Custom Experiments

```yaml
# config/custom_experiments.yaml
experiments:
  my_experiment:
    experiment_name: "my_experiment"
    model:
      name: "medium"
      config_path: "config/model_config.yaml"
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
    # ... other configurations
```

### Background Execution

```bash
# Run experiments in background (Linux/Mac)
nohup python src/sample_scripts/phase3_hyperparameter_tuning.py > tuning.log 2>&1 &

# Windows PowerShell background
Start-Job -ScriptBlock {
    python src/sample_scripts/phase3_hyperparameter_tuning.py
}
```

## Results Analysis

### Results File Structure

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
    "final_train_loss": 1.8234,
    "hyperparameters": {
      "model_name": "small",
      "batch_size": 32,
      "learning_rate": 0.001,
      "weight_decay": 0.0001,
      "loss_type": "crossentropy",
      "scheduler_type": "plateau"
    }
  },
  "all_results": [...]
}
```

### Key Metrics

- **best_val_char_accuracy**: Best validation character accuracy achieved
- **best_val_seq_accuracy**: Best validation sequence accuracy achieved
- **training_time**: Total training time in seconds
- **final_train_loss**: Final training loss value
- **status**: Experiment status (`completed`, `failed`)

### Analysis Scripts

```python
# analyze_results.py
import json
import glob

def analyze_tuning_results():
    """Analyze hyperparameter tuning results."""
    
    # Load latest results
    result_files = glob.glob("hyperparameter_tuning_results_*.json")
    latest_file = max(result_files)
    
    with open(latest_file, 'r') as f:
        results = json.load(f)
    
    # Extract completed experiments
    completed = [r for r in results['all_results'] if r['status'] == 'completed']
    
    if not completed:
        print("No completed experiments found.")
        return
    
    # Sort by character accuracy
    completed.sort(key=lambda x: x['best_val_char_accuracy'], reverse=True)
    
    print("=== HYPERPARAMETER TUNING RESULTS ===")
    print(f"Total experiments: {len(results['all_results'])}")
    print(f"Completed: {len(completed)}")
    print(f"Failed: {len(results['all_results']) - len(completed)}")
    
    print("\n=== TOP 3 EXPERIMENTS ===")
    for i, exp in enumerate(completed[:3]):
        print(f"\n{i+1}. {exp['experiment_name']}")
        print(f"   Character Accuracy: {exp['best_val_char_accuracy']:.4f}")
        print(f"   Sequence Accuracy: {exp['best_val_seq_accuracy']:.4f}")
        print(f"   Training Time: {exp['training_time']:.1f}s")
        print(f"   Hyperparameters:")
        for key, value in exp['hyperparameters'].items():
            print(f"     {key}: {value}")
    
    # Performance analysis
    accuracies = [exp['best_val_char_accuracy'] for exp in completed]
    times = [exp['training_time'] for exp in completed]
    
    print(f"\n=== PERFORMANCE STATISTICS ===")
    print(f"Best Character Accuracy: {max(accuracies):.4f}")
    print(f"Average Character Accuracy: {sum(accuracies)/len(accuracies):.4f}")
    print(f"Fastest Training: {min(times):.1f}s")
    print(f"Average Training Time: {sum(times)/len(times):.1f}s")
    
    # Improvement over baseline
    baseline_acc = 0.24  # Phase 2 baseline
    best_acc = max(accuracies)
    improvement = best_acc - baseline_acc
    
    print(f"\n=== IMPROVEMENT ANALYSIS ===")
    print(f"Phase 2 Baseline: {baseline_acc:.4f}")
    print(f"Best Phase 3.1 Result: {best_acc:.4f}")
    print(f"Absolute Improvement: {improvement:.4f}")
    print(f"Relative Improvement: {improvement/baseline_acc*100:.1f}%")

if __name__ == "__main__":
    analyze_tuning_results()
```

## Performance Optimization

### CPU-Specific Optimizations

1. **Batch Size Tuning**
   ```yaml
   # Small models: 32-64
   # Medium models: 32-96
   # Large models: 16-32
   batch_size: 64
   ```

2. **Worker Configuration**
   ```yaml
   # CPU cores - 1, max 4
   num_workers: 4
   pin_memory: false  # Disabled for CPU
   ```

3. **Memory Management**
   ```yaml
   mixed_precision: false  # Not beneficial for CPU
   gradient_accumulation_steps: 1  # Keep simple for CPU
   ```

### Learning Rate Guidelines

| Model Size | Conservative | Balanced | Aggressive |
|------------|-------------|----------|------------|
| Small      | 0.001       | 0.002    | 0.003      |
| Medium     | 0.0008      | 0.0015   | 0.003      |
| Large      | 0.0005      | 0.001    | 0.002      |

### Scheduler Recommendations

- **Cosine**: Best for smooth convergence, requires warmup
- **StepLR**: Good for aggressive training, simple setup
- **Plateau**: Conservative, waits for validation plateau

## Troubleshooting

### Common Issues

#### 1. YAML Configuration Errors

**Error**: `yaml.scanner.ScannerError: while scanning an alias`
```bash
# Solution: Check for invalid YAML syntax
python -c "import yaml; yaml.safe_load(open('config/phase3_simple_configs.yaml'))"
```

#### 2. Dataset Loading Issues

**Error**: `KhmerDigitsDataset.__init__() got an unexpected keyword argument`
```python
# Solution: Check dataset initialization parameters
train_dataset = KhmerDigitsDataset(
    metadata_path="generated_data/metadata.yaml",
    split='train',
    transform=get_train_transforms()  # Not apply_transforms=True
)
```

#### 3. Model Creation Errors

**Error**: `KhmerDigitsOCR.__init__() got an unexpected keyword argument 'model_name'`
```python
# Solution: Use preset parameter
model = create_model(
    preset="medium",  # Not model_name="medium"
    vocab_size=13,
    max_sequence_length=9
)
```

#### 4. Memory Issues

**Error**: `RuntimeError: [enforce fail at alloc_cpu.cpp]`
```yaml
# Solution: Reduce batch size
batch_size: 16  # Reduce from 32
num_workers: 2  # Reduce from 4
```

#### 5. Slow Training

**Issue**: Training taking too long
```yaml
# Solutions:
num_epochs: 20     # Reduce from 40
batch_size: 64     # Increase for efficiency
log_every_n_steps: 50  # Reduce logging frequency
```

### Debugging Commands

```bash
# Check environment
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# Validate configuration
python -c "
import yaml
with open('config/phase3_simple_configs.yaml') as f:
    config = yaml.safe_load(f)
    print('Configuration loaded successfully')
    print(f'Experiments: {list(config[\"experiments\"].keys())}')"

# Test dataset loading
python -c "
from modules.data_utils import KhmerDigitsDataset
dataset = KhmerDigitsDataset('generated_data/metadata.yaml', split='train')
print(f'Dataset loaded: {len(dataset)} samples')
print(f'Sample: {dataset[0][1].shape}')"

# Test model creation
python -c "
from models import create_model
model = create_model(preset='small', vocab_size=13, max_sequence_length=9)
print('Model created successfully')
print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')"
```

### Log Analysis

```bash
# Check training progress
grep "Epoch.*Results:" training_output/*/logs/training_*.log

# Find errors
grep "ERROR" training_output/*/logs/training_*.log

# Check convergence
grep "best model saved" training_output/*/logs/training_*.log

# Monitor memory usage (Linux/Mac)
grep "memory" training_output/*/logs/training_*.log
```

## API Reference

### HyperparameterTuner Class

```python
class HyperparameterTuner:
    """Systematic hyperparameter tuning for Khmer OCR model."""
    
    def __init__(self, config_file: str):
        """Initialize the hyperparameter tuner."""
        
    def run_experiments(self, experiment_names: List[str] = None):
        """Run all or specified experiments."""
        
    def save_results(self):
        """Save tuning results to file."""
```

### Main Functions

```python
def create_training_config(experiment_config: Dict) -> TrainingConfig:
    """Create TrainingConfig object from experiment configuration."""

def run_single_experiment(experiment_name: str, experiment_config: Dict) -> Dict:
    """Run a single hyperparameter experiment."""
```

### Configuration Classes

```python
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
    pin_memory: bool = False
    
    # Training configuration
    num_epochs: int = 40
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
```

## Best Practices

### 1. Systematic Approach
- Start with `conservative_small` to establish baseline
- Run `baseline_optimized` for standard performance
- Use `high_learning_rate` for quick validation

### 2. Resource Management
- Monitor system resources during training
- Use appropriate batch sizes for available RAM
- Consider training time vs. accuracy trade-offs

### 3. Result Documentation
- Save all experiment configurations
- Document reasoning for hyperparameter choices
- Track improvements over previous baselines

### 4. Iterative Improvement
- Analyze failed experiments for insights
- Adjust configurations based on successful experiments
- Build on best-performing hyperparameter combinations

---

*This guide covers the complete hyperparameter tuning system for Khmer OCR Phase 3.1. For additional support, check the troubleshooting section or review the implementation code in `src/sample_scripts/phase3_hyperparameter_tuning.py`.* 