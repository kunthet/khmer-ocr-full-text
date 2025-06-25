# Progressive Training Strategy API Reference

## Overview

This document provides a comprehensive API reference for the Progressive Training Strategy implementation in Phase 3.1. It covers all classes, methods, and functions available for developers.

## Table of Contents

1. [Core Classes](#core-classes)
2. [Methods and Functions](#methods-and-functions)
3. [Configuration Parameters](#configuration-parameters)
4. [Return Types and Data Structures](#return-types-and-data-structures)
5. [Integration Points](#integration-points)
6. [Usage Examples](#usage-examples)

## Core Classes

### `ProgressiveTrainingStrategy`

Main class that orchestrates the progressive training pipeline.

```python
class ProgressiveTrainingStrategy:
    """
    Implements progressive training strategy for Khmer OCR models.
    
    This class orchestrates the entire training pipeline through 5 stages:
    1. Single character recognition (with digit model transfer)
    2. Simple character combinations
    3. Complex combinations with diacritics
    4. Word-level recognition
    5. Multi-word and sentence recognition
    """
```

#### Constructor

```python
def __init__(self, 
             config_path: str,
             output_dir: str = "progressive_training_output") -> None
```

**Parameters:**
- `config_path` (str): Path to model configuration YAML file
- `output_dir` (str, optional): Output directory for training results

**Example:**
```python
strategy = ProgressiveTrainingStrategy(
    config_path="config/model_config.yaml",
    output_dir="my_training_results"
)
```

## Methods and Functions

### Core Training Methods

#### `create_curriculum_manager()`

Creates and configures the curriculum learning manager with 5 progressive stages.

```python
def create_curriculum_manager(self) -> CurriculumLearningManager
```

**Returns:**
- `CurriculumLearningManager`: Configured curriculum manager with 5 stages

**Stage Configuration:**
| Stage | Difficulty Level | Success Threshold | Epoch Range |
|-------|------------------|-------------------|-------------|
| 1 | `SINGLE_CHAR` | 92% | 5-15 |
| 2 | `SIMPLE_COMBO` | 88% | 8-20 |
| 3 | `COMPLEX_COMBO` | 82% | 12-25 |
| 4 | `WORD_LEVEL` | 78% | 15-30 |
| 5 | `MULTI_WORD` | 75% | 20-40 |

#### `initialize_model()`

Initializes the KhmerTextOCR model using the model factory.

```python
def initialize_model(self) -> None
```

**Side Effects:**
- Sets `self.model` to initialized KhmerTextOCR instance
- Moves model to appropriate device (GPU/CPU)
- Prints model parameter count

#### `train_stage()`

Trains the model for a specific curriculum stage.

```python
def train_stage(self, 
                stage: CurriculumStage, 
                stage_idx: int) -> Dict[str, Any]
```

**Parameters:**
- `stage` (CurriculumStage): Stage configuration object
- `stage_idx` (int): Zero-based stage index

**Returns:**
- `Dict[str, Any]`: Stage training metrics

#### `run_progressive_training()`

Executes the complete progressive training strategy through all 5 stages.

```python
def run_progressive_training(self) -> Dict[str, Any]
```

**Returns:**
- `Dict[str, Any]`: Comprehensive training report

## Configuration Parameters

### Model Configuration File Structure

```yaml
# model_config.yaml
model:
  name: "KhmerTextOCR"
  vocab_size: 115
  hidden_dim: 512
  num_layers: 6
  
training:
  batch_size: 32
  learning_rate: 0.001
  weight_decay: 0.0001
  
data:
  image_height: 32
  image_width: 128
  max_sequence_length: 20
```

### Command Line Arguments

```bash
python src/sample_scripts/progressive_training_strategy.py [OPTIONS]
```

**Options:**
- `--config` (str): Path to configuration file
- `--output-dir` (str): Output directory
- `--dry-run`: Test configuration without training

## Return Types and Data Structures

### Training Report

```python
TrainingReport = {
    'training_completed': str,         # ISO 8601 timestamp
    'total_stages': int,               # Number of stages completed
    'total_epochs': int,               # Sum of epochs across stages
    'final_performance': float,        # Final validation accuracy
    'stage_performances': List[Dict],  # Per-stage results
    'training_duration': str           # Human-readable duration
}
```

### Stage Metrics

```python
StageMetrics = {
    'stage_name': str,             # Stage display name
    'best_val_accuracy': float,    # Best validation accuracy achieved
    'epochs_trained': int          # Number of epochs completed
}
```

## Integration Points

### Curriculum Learning Integration

```python
from modules.trainers import CurriculumLearningManager, CurriculumStage, DifficultyLevel

# Access curriculum components
manager = strategy.curriculum_manager
current_stage = manager.current_stage
stage_level = current_stage.level  # DifficultyLevel enum
```

### Model Factory Integration

```python
from models import ModelFactory

# Model creation through factory
factory = ModelFactory(config)
model = factory.create_model("KhmerTextOCR")
```

## Usage Examples

### Basic Progressive Training

```python
from progressive_training_strategy import ProgressiveTrainingStrategy

# Initialize strategy
strategy = ProgressiveTrainingStrategy(
    config_path="config/model_config.yaml",
    output_dir="experiment_001"
)

# Run complete training
report = strategy.run_progressive_training()

# Access results
print(f"Final accuracy: {report['final_performance']:.3f}")
```

### Custom Stage Training

```python
# Initialize components
strategy = ProgressiveTrainingStrategy("config/model_config.yaml")
curriculum_manager = strategy.create_curriculum_manager()
strategy.initialize_model()

# Train specific stage
stage = curriculum_manager.stages[2]  # Stage 3: Complex Combinations
metrics = strategy.train_stage(stage, 2)
```

### Error Handling

```python
try:
    strategy = ProgressiveTrainingStrategy("config.yaml")
    report = strategy.run_progressive_training()
except FileNotFoundError:
    print("Configuration file not found")
except Exception as e:
    print(f"Training failed: {e}")
```

## Performance Considerations

### Memory Usage

- **Model Size**: ~50-100MB depending on configuration
- **Training Data**: ~1-5GB for full dataset generation
- **Checkpoints**: ~50MB per stage checkpoint

### Training Time

- **Total Duration**: 2-8 hours depending on hardware
- **GPU Acceleration**: 3-5x faster than CPU training

## Version Compatibility

### Supported Versions

- **Python**: 3.8+
- **PyTorch**: 1.9+
- **CUDA**: 10.2+ (if using GPU)

### Dependencies

```python
torch >= 1.9.0
torchvision >= 0.10.0
pyyaml >= 5.4.0
numpy >= 1.20.0
tqdm >= 4.60.0
```
