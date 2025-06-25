# Trainers Module Quick Reference

## Quick Start

### Basic Training Setup
```python
from src.modules.trainers import OCRTrainer, TrainingConfig
from src.modules.data_utils import create_data_loaders
from src.models import create_model

# 1. Load configuration
config = TrainingConfig.from_yaml("config/training_config.yaml")

# 2. Create data loaders
train_loader, val_loader = create_data_loaders(
    metadata_path=config.metadata_path,
    batch_size=config.batch_size
)

# 3. Create model
model = create_model(config.model_name)

# 4. Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 5. Create and run trainer
trainer = OCRTrainer(model, train_loader, val_loader, config, device)
results = trainer.train()
```

## Common Configurations

### Small Model, Fast Training
```yaml
model_name: "small"
batch_size: 64
num_epochs: 20
learning_rate: 0.002
loss_type: "crossentropy"
scheduler_type: "steplr"
early_stopping_patience: 5
```

### Large Model, Production Training
```yaml
model_name: "large" 
batch_size: 16
num_epochs: 100
learning_rate: 0.0005
loss_type: "focal"
scheduler_type: "cosine"
early_stopping_patience: 15
mixed_precision: true
```

### CTC Training
```yaml
model_name: "ctc_medium"
loss_type: "ctc"
scheduler_type: "plateau"
lr_patience: 5
```

## Loss Functions

### CrossEntropy (Default)
```python
config.loss_type = "crossentropy"
config.label_smoothing = 0.1  # Optional regularization
```

### Focal Loss (Class Imbalance)
```python
config.loss_type = "focal"
config.focal_gamma = 2.0      # Focus on hard examples
config.focal_alpha = 0.25     # Class weighting
```

### CTC Loss (Alignment-Free)
```python
config.loss_type = "ctc"
# Requires CTC-compatible model
```

## Learning Rate Scheduling

### StepLR (Default)
```python
config.scheduler_type = "steplr"
config.step_size = 10         # Decay every 10 epochs
config.gamma = 0.5            # Multiply by 0.5
```

### Cosine Annealing
```python
config.scheduler_type = "cosine"
config.cosine_t_max = config.num_epochs
```

### Reduce on Plateau
```python
config.scheduler_type = "plateau"
config.lr_patience = 5        # Wait 5 epochs
config.lr_factor = 0.5        # Reduce by 50%
```

## Metrics and Evaluation

### Calculate Metrics
```python
from src.modules.trainers import OCRMetrics

# Create metrics calculator
idx_to_char = {0: '០', 1: '១', 2: '២', ...}
metrics = OCRMetrics(idx_to_char)

# Update with batch results
metrics.update(predictions, targets)

# Get results
results = metrics.compute()
print(f"Character Accuracy: {results['char_accuracy']:.3f}")
print(f"Sequence Accuracy: {results['seq_accuracy']:.3f}")
print(f"Edit Distance: {results['edit_distance']:.3f}")
```

### Confusion Matrix
```python
confusion_matrix = metrics.get_confusion_matrix()
per_class_acc = metrics.get_per_class_accuracy()
```

## Checkpointing

### Auto Checkpointing
```python
config.save_every_n_epochs = 5    # Save every 5 epochs
config.keep_n_checkpoints = 3     # Keep last 3 checkpoints
```

### Manual Checkpoint Management
```python
from src.modules.trainers import CheckpointManager

manager = CheckpointManager("checkpoints/", keep_n_checkpoints=5)

# Save checkpoint
checkpoint_path = manager.save_checkpoint(
    model, optimizer, scheduler, epoch, metrics, is_best=True
)

# Load checkpoint
checkpoint = manager.load_checkpoint(checkpoint_path)
model.load_state_dict(checkpoint['model_state_dict'])

# Load best model
best_checkpoint = manager.load_best_model()
```

## Early Stopping

### Basic Early Stopping
```python
config.early_stopping_patience = 10      # Wait 10 epochs
config.early_stopping_min_delta = 0.001  # Minimum improvement
```

### Manual Early Stopping
```python
from src.modules.trainers import EarlyStopping

early_stopping = EarlyStopping(patience=10, min_delta=0.001)

for epoch in range(num_epochs):
    val_loss = validate()
    if early_stopping(val_loss, epoch):
        print(f"Early stopping at epoch {epoch}")
        break
```

## Environment Setup

### Setup Training Environment
```python
from src.modules.trainers import setup_training_environment

# Creates directories, sets up logging
output_dir = setup_training_environment(config)
```

### Device Configuration
```python
config.device = "auto"        # Auto-detect GPU/CPU
config.device = "cuda"        # Force GPU
config.device = "cpu"         # Force CPU
```

## Performance Optimization

### Memory Optimization
```python
config.mixed_precision = True        # Enable AMP
config.gradient_checkpointing = True # Save memory
config.pin_memory = True             # Faster data loading
```

### Speed Optimization
```python
config.num_workers = 4              # Parallel data loading
config.persistent_workers = True    # Keep workers alive
config.gradient_accumulation_steps = 4  # Simulate larger batch
```

## Logging and Monitoring

### TensorBoard Logging
```python
config.use_tensorboard = True
config.log_every_n_steps = 25        # Log frequency
```

### Custom Logging
```python
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Logs are automatically created in training loop
```

## Debugging

### Debug Mode
```python
config.debug_mode = True             # Extra validation
config.log_every_n_steps = 1        # Frequent logging
config.num_epochs = 2               # Quick test
```

### Common Issues

#### CUDA Out of Memory
```python
config.batch_size = 16              # Reduce batch size
config.mixed_precision = True       # Use AMP
config.gradient_checkpointing = True
```

#### Slow Training
```python
config.num_workers = 8              # More data workers
config.pin_memory = True
config.persistent_workers = True
```

#### Poor Convergence
```python
config.learning_rate = 0.0001       # Lower learning rate
config.loss_type = "focal"          # Try focal loss
config.gradient_clip_norm = 0.5     # Gradient clipping
```

## Configuration Patterns

### Development Config
```yaml
num_epochs: 5
batch_size: 8
use_tensorboard: true
log_every_n_steps: 1
debug_mode: true
```

### Production Config
```yaml
num_epochs: 100
batch_size: 32
mixed_precision: true
early_stopping_patience: 15
save_every_n_epochs: 10
use_tensorboard: true
```

### Hyperparameter Search
```yaml
# Grid search friendly
learning_rate: 0.001
batch_size: 32
loss_type: "crossentropy"
# Easy to modify and sweep
```

## Factory Functions

### Create Components
```python
from src.modules.trainers import create_loss_function, create_metrics_calculator

# Create loss function
loss_fn = create_loss_function("focal", focal_gamma=2.0)

# Create metrics calculator
metrics = create_metrics_calculator(idx_to_char)
```

## Error Handling

### Robust Training
```python
try:
    results = trainer.train()
except RuntimeError as e:
    if "out of memory" in str(e):
        print("Reducing batch size...")
        config.batch_size //= 2
        trainer = OCRTrainer(model, train_loader, val_loader, config, device)
        results = trainer.train()
    else:
        raise e
```

### Checkpoint Recovery
```python
try:
    # Try to resume from checkpoint
    checkpoint = manager.load_best_model()
    if checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
except FileNotFoundError:
    # Start fresh training
    start_epoch = 0
```

## Command Line Usage

### Training Script Template
```python
#!/usr/bin/env python3
import argparse
from src.modules.trainers import TrainingConfig, OCRTrainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/training_config.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()
    
    config = TrainingConfig.from_yaml(args.config)
    if args.epochs:
        config.num_epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    
    # Setup and train...
    trainer = OCRTrainer(model, train_loader, val_loader, config, device)
    results = trainer.train()
    
    print(f"Training completed. Best val loss: {results['best_val_loss']:.4f}")

if __name__ == "__main__":
    main()
```

### Run Training
```bash
# Basic training
python train.py

# Custom config
python train.py --config custom_config.yaml

# Override parameters
python train.py --epochs 20 --batch-size 16
```

## Integration Examples

### With Custom Dataset
```python
from torch.utils.data import DataLoader
from src.modules.trainers import OCRTrainer

# Custom dataset
dataset = MyCustomDataset(...)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Train with custom data
trainer = OCRTrainer(model, train_loader, val_loader, config, device)
```

### With Model Variants
```python
# Different model presets
for model_name in ["small", "medium", "large"]:
    config.model_name = model_name
    model = create_model(model_name)
    trainer = OCRTrainer(model, train_loader, val_loader, config, device)
    results = trainer.train()
    print(f"{model_name}: {results['best_val_loss']:.4f}")
```

### Multi-Experiment Runner
```python
configs = [
    {"loss_type": "crossentropy", "learning_rate": 0.001},
    {"loss_type": "focal", "learning_rate": 0.001},
    {"loss_type": "ctc", "learning_rate": 0.0005},
]

for i, config_updates in enumerate(configs):
    config = TrainingConfig.from_yaml("config/training_config.yaml")
    for key, value in config_updates.items():
        setattr(config, key, value)
    
    config.experiment_name = f"experiment_{i}"
    trainer = OCRTrainer(model, train_loader, val_loader, config, device)
    results = trainer.train()
```

---

## Quick Commands

```bash
# Test training infrastructure
python src/sample_scripts/test_training_infrastructure.py

# Generate training data
python src/sample_scripts/generate_dataset.py

# Quick model test
python src/sample_scripts/simple_model_test.py

# View TensorBoard logs
tensorboard --logdir training_output/logs/
``` 