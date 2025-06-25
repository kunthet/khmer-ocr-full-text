# Trainers Module Examples

## Basic Training Examples

### Example 1: Simple Training Setup

```python
import torch
from src.modules.trainers import OCRTrainer, TrainingConfig
from src.modules.data_utils import create_data_loaders
from src.models import create_model

def basic_training():
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
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create and run trainer
    trainer = OCRTrainer(model, train_loader, val_loader, config, device)
    results = trainer.train()
    
    print(f"Training completed!")
    print(f"Best validation loss: {results['best_val_loss']:.4f}")
    print(f"Final character accuracy: {results['final_char_accuracy']:.3f}")
    
    return results

if __name__ == "__main__":
    results = basic_training()
```

### Example 2: Custom Configuration Training

```python
from src.modules.trainers import TrainingConfig, OCRTrainer
from src.modules.data_utils import create_data_loaders
from src.models import create_model

def custom_config_training():
    # Create custom configuration
    config = TrainingConfig()
    
    # Customize parameters
    config.model_name = "large"
    config.num_epochs = 25
    config.learning_rate = 0.0005
    config.batch_size = 16
    config.loss_type = "focal"
    config.focal_gamma = 2.0
    config.scheduler_type = "cosine"
    config.early_stopping_patience = 8
    config.mixed_precision = True
    
    # Save custom configuration
    config.save_yaml("config/custom_training_config.yaml")
    
    # Setup training
    train_loader, val_loader = create_data_loaders(
        metadata_path=config.metadata_path,
        batch_size=config.batch_size
    )
    
    model = create_model(config.model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Train
    trainer = OCRTrainer(model, train_loader, val_loader, config, device)
    results = trainer.train()
    
    return results
```

### Example 3: Multi-Experiment Training

```python
import torch
from src.modules.trainers import TrainingConfig, OCRTrainer
from src.modules.data_utils import create_data_loaders
from src.models import create_model

def multi_experiment_training():
    # Define experiments
    experiments = [
        {
            "name": "baseline",
            "model_name": "medium",
            "loss_type": "crossentropy",
            "learning_rate": 0.001
        },
        {
            "name": "focal_loss",
            "model_name": "medium",
            "loss_type": "focal",
            "learning_rate": 0.001,
            "focal_gamma": 2.0
        },
        {
            "name": "large_model",
            "model_name": "large",
            "loss_type": "crossentropy",
            "learning_rate": 0.0005
        },
        {
            "name": "ctc_training",
            "model_name": "ctc_medium",
            "loss_type": "ctc",
            "learning_rate": 0.0005
        }
    ]
    
    results = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for exp in experiments:
        print(f"\n=== Running Experiment: {exp['name']} ===")
        
        # Load base configuration
        config = TrainingConfig.from_yaml("config/training_config.yaml")
        
        # Apply experiment settings
        for key, value in exp.items():
            if key != "name":
                setattr(config, key, value)
        
        # Set experiment name for logging
        config.experiment_name = f"khmer_ocr_{exp['name']}"
        config.num_epochs = 10  # Quick experiments
        
        # Create data loaders
        train_loader, val_loader = create_data_loaders(
            metadata_path=config.metadata_path,
            batch_size=config.batch_size
        )
        
        # Create model
        model = create_model(config.model_name)
        
        # Train
        trainer = OCRTrainer(model, train_loader, val_loader, config, device)
        exp_results = trainer.train()
        
        results[exp['name']] = exp_results
        print(f"Best val loss: {exp_results['best_val_loss']:.4f}")
    
    # Compare results
    print("\n=== Experiment Comparison ===")
    for name, result in results.items():
        print(f"{name:15}: {result['best_val_loss']:.4f}")
    
    return results
```

## Advanced Training Examples

### Example 4: Resume Training from Checkpoint

```python
import torch
from pathlib import Path
from src.modules.trainers import OCRTrainer, TrainingConfig, CheckpointManager
from src.modules.data_utils import create_data_loaders
from src.models import create_model

def resume_training():
    config = TrainingConfig.from_yaml("config/training_config.yaml")
    
    # Create components
    train_loader, val_loader = create_data_loaders(
        metadata_path=config.metadata_path,
        batch_size=config.batch_size
    )
    
    model = create_model(config.model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Setup checkpoint manager
    checkpoint_dir = Path("training_output") / config.experiment_name / "checkpoints"
    checkpoint_manager = CheckpointManager(str(checkpoint_dir))
    
    # Try to load best checkpoint
    start_epoch = 0
    best_checkpoint = checkpoint_manager.load_best_model()
    
    if best_checkpoint:
        print(f"Resuming from epoch {best_checkpoint['epoch']}")
        model.load_state_dict(best_checkpoint['model_state_dict'])
        start_epoch = best_checkpoint['epoch'] + 1
    else:
        print("Starting fresh training")
    
    # Create trainer
    trainer = OCRTrainer(model, train_loader, val_loader, config, device)
    
    # Resume training
    results = trainer.train()
    
    return results
```

### Example 5: Custom Metrics and Analysis

```python
import torch
import numpy as np
from src.modules.trainers import OCRMetrics, OCRTrainer, TrainingConfig
from src.modules.data_utils import create_data_loaders
from src.models import create_model

def detailed_metrics_analysis():
    # Standard training setup
    config = TrainingConfig.from_yaml("config/training_config.yaml")
    config.num_epochs = 5  # Quick training for demo
    
    train_loader, val_loader = create_data_loaders(
        metadata_path=config.metadata_path,
        batch_size=config.batch_size
    )
    
    model = create_model(config.model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Train model
    trainer = OCRTrainer(model, train_loader, val_loader, config, device)
    results = trainer.train()
    
    # Detailed evaluation
    model.eval()
    
    # Character mappings
    idx_to_char = {
        0: '០', 1: '១', 2: '២', 3: '៣', 4: '៤',
        5: '៥', 6: '៦', 7: '៧', 8: '៨', 9: '៩',
        10: '<EOS>', 11: '<PAD>', 12: '<BLANK>'
    }
    
    # Create metrics calculator
    metrics = OCRMetrics(idx_to_char)
    
    # Evaluate on validation set
    with torch.no_grad():
        for batch_idx, (images, targets, _) in enumerate(val_loader):
            if batch_idx >= 10:  # Limit for demo
                break
                
            images = images.to(device)
            targets = targets.to(device)
            
            # Get predictions
            predictions = model(images)
            metrics.update(predictions, targets)
    
    # Compute comprehensive metrics
    results = metrics.compute()
    
    print("=== Detailed Metrics Analysis ===")
    print(f"Character Accuracy: {results['char_accuracy']:.3f}")
    print(f"Sequence Accuracy: {results['seq_accuracy']:.3f}")
    print(f"Edit Distance: {results['edit_distance']:.3f}")
    
    # Per-class accuracy
    per_class_acc = metrics.get_per_class_accuracy()
    print("\n=== Per-Character Accuracy ===")
    for char, acc in per_class_acc.items():
        if char not in ['<PAD>', '<EOS>', '<BLANK>']:
            print(f"Character '{char}': {acc:.3f}")
    
    # Confusion matrix analysis
    confusion_matrix = metrics.get_confusion_matrix()
    print(f"\nConfusion Matrix Shape: {confusion_matrix.shape}")
    
    # Find most confused pairs
    most_confused = []
    for i in range(10):  # Only digits
        for j in range(10):
            if i != j and confusion_matrix[i, j] > 0:
                most_confused.append((
                    idx_to_char[i], 
                    idx_to_char[j], 
                    confusion_matrix[i, j]
                ))
    
    most_confused.sort(key=lambda x: x[2], reverse=True)
    
    print("\n=== Most Common Confusions ===")
    for true_char, pred_char, count in most_confused[:5]:
        print(f"'{true_char}' confused with '{pred_char}': {count} times")
    
    return results, metrics
```

### Example 6: Hyperparameter Tuning

```python
import itertools
from src.modules.trainers import TrainingConfig, OCRTrainer
from src.modules.data_utils import create_data_loaders
from src.models import create_model

def hyperparameter_tuning():
    # Define hyperparameter grid
    param_grid = {
        'learning_rate': [0.001, 0.0005, 0.0001],
        'batch_size': [16, 32],
        'loss_type': ['crossentropy', 'focal'],
        'scheduler_type': ['steplr', 'cosine']
    }
    
    # Generate all combinations
    keys = param_grid.keys()
    values = param_grid.values()
    combinations = list(itertools.product(*values))
    
    results = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for i, combo in enumerate(combinations[:8]):  # Limit for demo
        params = dict(zip(keys, combo))
        print(f"\n=== Experiment {i+1}/{len(combinations[:8])} ===")
        print(f"Parameters: {params}")
        
        # Create configuration
        config = TrainingConfig()
        for key, value in params.items():
            setattr(config, key, value)
        
        # Quick training settings
        config.num_epochs = 5
        config.early_stopping_patience = 3
        config.experiment_name = f"hp_tune_{i+1}"
        
        try:
            # Create data loaders
            train_loader, val_loader = create_data_loaders(
                metadata_path=config.metadata_path,
                batch_size=config.batch_size
            )
            
            # Create model
            model = create_model(config.model_name)
            
            # Train
            trainer = OCRTrainer(model, train_loader, val_loader, config, device)
            exp_results = trainer.train()
            
            # Store results
            result_entry = {
                'params': params,
                'best_val_loss': exp_results['best_val_loss'],
                'final_char_accuracy': exp_results.get('final_char_accuracy', 0.0)
            }
            results.append(result_entry)
            
            print(f"Best val loss: {exp_results['best_val_loss']:.4f}")
            
        except Exception as e:
            print(f"Experiment failed: {e}")
            continue
    
    # Find best configuration
    best_result = min(results, key=lambda x: x['best_val_loss'])
    print(f"\n=== Best Configuration ===")
    print(f"Parameters: {best_result['params']}")
    print(f"Best val loss: {best_result['best_val_loss']:.4f}")
    print(f"Character accuracy: {best_result['final_char_accuracy']:.3f}")
    
    return results
```

## Loss Function Examples

### Example 7: Comparing Loss Functions

```python
import torch
from src.modules.trainers import OCRLoss

def loss_function_comparison():
    # Create dummy data
    batch_size, seq_len, num_classes = 4, 8, 13
    predictions = torch.randn(batch_size, seq_len, num_classes)
    targets = torch.randint(0, 10, (batch_size, seq_len))
    
    # Pad some sequences
    targets[0, 6:] = 11  # PAD token
    targets[1, 5:] = 11
    
    print("=== Loss Function Comparison ===")
    
    # CrossEntropy Loss
    ce_loss = OCRLoss(loss_type='crossentropy')
    ce_result = ce_loss(predictions, targets)
    print(f"CrossEntropy Loss: {ce_result['loss'].item():.4f}")
    
    # CrossEntropy with Label Smoothing
    ce_smooth_loss = OCRLoss(loss_type='crossentropy', label_smoothing=0.1)
    ce_smooth_result = ce_smooth_loss(predictions, targets)
    print(f"CrossEntropy + Smoothing: {ce_smooth_result['loss'].item():.4f}")
    
    # Focal Loss
    focal_loss = OCRLoss(loss_type='focal', focal_gamma=2.0)
    focal_result = focal_loss(predictions, targets)
    print(f"Focal Loss (γ=2.0): {focal_result['loss'].item():.4f}")
    
    # CTC Loss
    input_lengths = torch.tensor([8, 8, 8, 8])
    target_lengths = torch.tensor([6, 5, 8, 7])
    
    # Convert targets for CTC (remove padding)
    ctc_targets = []
    for i, length in enumerate(target_lengths):
        ctc_targets.append(targets[i, :length])
    ctc_targets = torch.cat(ctc_targets)
    
    ctc_loss = OCRLoss(loss_type='ctc')
    ctc_result = ctc_loss(
        predictions.log_softmax(dim=-1).transpose(0, 1),  # [T, N, C]
        ctc_targets,
        input_lengths=input_lengths,
        target_lengths=target_lengths
    )
    print(f"CTC Loss: {ctc_result['loss'].item():.4f}")
```

### Example 8: Custom Training Loop

```python
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from src.modules.trainers import OCRLoss, OCRMetrics, CheckpointManager, EarlyStopping

def custom_training_loop():
    # Setup (simplified)
    from src.modules.data_utils import create_data_loaders
    from src.models import create_model
    from src.modules.trainers import TrainingConfig
    
    config = TrainingConfig()
    config.num_epochs = 10
    
    train_loader, val_loader = create_data_loaders(
        metadata_path=config.metadata_path,
        batch_size=config.batch_size
    )
    
    model = create_model(config.model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Training components
    loss_fn = OCRLoss(loss_type='crossentropy')
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    # Utilities
    checkpoint_manager = CheckpointManager("custom_training_checkpoints")
    early_stopping = EarlyStopping(patience=5)
    writer = SummaryWriter("custom_training_logs")
    
    # Character mapping for metrics
    idx_to_char = {i: str(i) for i in range(13)}
    metrics_calculator = OCRMetrics(idx_to_char)
    
    # Training loop
    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch+1}/{config.num_epochs}")
        
        # Training phase
        model.train()
        train_loss = 0.0
        num_batches = 0
        
        for batch_idx, (images, targets, _) in enumerate(train_loader):
            images, targets = images.to(device), targets.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            predictions = model(images)
            
            # Calculate loss
            loss_result = loss_fn(predictions, targets)
            loss = loss_result['loss']
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            num_batches += 1
            
            # Log every 20 batches
            if batch_idx % 20 == 0:
                step = epoch * len(train_loader) + batch_idx
                writer.add_scalar('Loss/Train_Batch', loss.item(), step)
                writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], step)
        
        avg_train_loss = train_loss / num_batches
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        metrics_calculator.reset()
        
        with torch.no_grad():
            for images, targets, _ in val_loader:
                images, targets = images.to(device), targets.to(device)
                
                predictions = model(images)
                loss_result = loss_fn(predictions, targets)
                val_loss += loss_result['loss'].item()
                
                # Update metrics
                metrics_calculator.update(predictions, targets)
        
        avg_val_loss = val_loss / len(val_loader)
        metrics = metrics_calculator.compute()
        
        # Logging
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}")
        print(f"Char Accuracy: {metrics['char_accuracy']:.3f}")
        print(f"Seq Accuracy: {metrics['seq_accuracy']:.3f}")
        
        # TensorBoard logging
        writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        writer.add_scalar('Loss/Validation', avg_val_loss, epoch)
        writer.add_scalar('Metrics/Char_Accuracy', metrics['char_accuracy'], epoch)
        writer.add_scalar('Metrics/Seq_Accuracy', metrics['seq_accuracy'], epoch)
        
        # Save checkpoint
        if (epoch + 1) % 3 == 0:
            checkpoint_manager.save_checkpoint(
                model, optimizer, scheduler, epoch, metrics,
                is_best=(avg_val_loss < early_stopping.best_score)
            )
        
        # Early stopping check
        if early_stopping(avg_val_loss, epoch):
            print(f"Early stopping at epoch {epoch+1}")
            break
        
        # Learning rate scheduling
        scheduler.step()
    
    writer.close()
    print("Training completed!")
    
    return {
        'final_train_loss': avg_train_loss,
        'final_val_loss': avg_val_loss,
        'final_metrics': metrics
    }
```

## Error Handling Examples

### Example 9: Robust Training with Error Recovery

```python
import torch
from src.modules.trainers import OCRTrainer, TrainingConfig
from src.modules.data_utils import create_data_loaders
from src.models import create_model

def robust_training():
    config = TrainingConfig.from_yaml("config/training_config.yaml")
    
    # Retry mechanism for different batch sizes
    batch_sizes = [config.batch_size, config.batch_size // 2, config.batch_size // 4]
    
    for batch_size in batch_sizes:
        print(f"Trying batch size: {batch_size}")
        
        try:
            # Create data loaders with current batch size
            train_loader, val_loader = create_data_loaders(
                metadata_path=config.metadata_path,
                batch_size=batch_size
            )
            
            # Create model
            model = create_model(config.model_name)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Update config
            config.batch_size = batch_size
            
            # Create trainer
            trainer = OCRTrainer(model, train_loader, val_loader, config, device)
            
            # Try training
            results = trainer.train()
            
            print(f"Training successful with batch size {batch_size}")
            return results
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"CUDA OOM with batch size {batch_size}, trying smaller...")
                torch.cuda.empty_cache()
                continue
            else:
                print(f"Other error: {e}")
                raise e
        except Exception as e:
            print(f"Unexpected error: {e}")
            raise e
    
    raise RuntimeError("Could not complete training with any batch size")

def safe_training_with_checkpoints():
    """Training with automatic checkpoint recovery"""
    config = TrainingConfig.from_yaml("config/training_config.yaml")
    
    try:
        # Normal training
        return robust_training()
        
    except Exception as e:
        print(f"Training failed: {e}")
        print("Attempting recovery from checkpoint...")
        
        # Try to recover from best checkpoint
        from src.modules.trainers import CheckpointManager
        checkpoint_manager = CheckpointManager("training_output/checkpoints")
        
        best_checkpoint = checkpoint_manager.load_best_model()
        if best_checkpoint:
            print(f"Found checkpoint from epoch {best_checkpoint['epoch']}")
            print(f"Best val loss: {best_checkpoint['val_loss']:.4f}")
            return best_checkpoint
        else:
            print("No checkpoint found for recovery")
            raise e
```

This examples file provides practical, working code that demonstrates all the key features of the training infrastructure. Each example is self-contained and shows different aspects of using the trainers module effectively. 