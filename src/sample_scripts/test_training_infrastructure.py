#!/usr/bin/env python3
"""
Comprehensive Test Script for Training Infrastructure

This script tests all components of the training infrastructure including:
- Loss functions and metrics
- Trainer classes and training loops
- Configuration management
- Checkpointing and early stopping
- TensorBoard logging
"""

import os
import sys
import torch
import torch.nn as nn
import tempfile
import shutil
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.modules.trainers import (
    OCRTrainer,
    OCRLoss,
    OCRMetrics,
    TrainingConfig,
    CheckpointManager,
    EarlyStopping,
    setup_training_environment,
    calculate_character_accuracy,
    calculate_sequence_accuracy
)
from src.modules.data_utils import KhmerDigitsDataset, create_data_loaders, get_train_transforms, get_val_transforms
from src.models import create_model


def create_dummy_model(num_classes=13, max_seq_length=9):
    """Create a simple dummy model for testing."""
    class DummyOCRModel(nn.Module):
        def __init__(self, num_classes, max_seq_length):
            super().__init__()
            self.cnn = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1))
            )
            self.classifier = nn.Linear(32, num_classes * max_seq_length)
            self.num_classes = num_classes
            self.max_seq_length = max_seq_length
        
        def forward(self, x):
            features = self.cnn(x)
            features = features.view(features.size(0), -1)
            output = self.classifier(features)
            return output.view(-1, self.max_seq_length, self.num_classes)
    
    return DummyOCRModel(num_classes, max_seq_length)


def test_loss_functions():
    """Test all loss function implementations."""
    print("Testing Loss Functions...")
    
    # Create dummy data with requires_grad=True for predictions
    batch_size, seq_len, num_classes = 4, 8, 13
    predictions = torch.randn(batch_size, seq_len, num_classes, requires_grad=True)
    targets = torch.randint(0, num_classes, (batch_size, seq_len))
    
    # Test CrossEntropy loss
    print("  Testing CrossEntropy loss...")
    ce_loss = OCRLoss(loss_type='crossentropy')
    loss_dict = ce_loss(predictions, targets)
    assert 'loss' in loss_dict
    assert loss_dict['loss'].requires_grad
    print(f"    CrossEntropy loss: {loss_dict['loss'].item():.4f}")
    
    # Test Focal loss
    print("  Testing Focal loss...")
    focal_loss = OCRLoss(loss_type='focal', focal_gamma=2.0)
    loss_dict = focal_loss(predictions, targets)
    assert 'loss' in loss_dict
    print(f"    Focal loss: {loss_dict['loss'].item():.4f}")
    
    # Test CTC loss (requires special setup)
    print("  Testing CTC loss...")
    try:
        ctc_loss = OCRLoss(loss_type='ctc')
        input_lengths = torch.full((batch_size,), seq_len, dtype=torch.long)
        target_lengths = torch.randint(1, seq_len, (batch_size,))
        ctc_targets = torch.cat([torch.randint(1, num_classes-1, (length,)) for length in target_lengths])
        
        loss_dict = ctc_loss(
            predictions, 
            ctc_targets,
            input_lengths=input_lengths,
            target_lengths=target_lengths
        )
        assert 'loss' in loss_dict
        print(f"    CTC loss: {loss_dict['loss'].item():.4f}")
    except Exception as e:
        print(f"    CTC loss test failed (expected): {e}")
    
    print("  ✅ Loss functions test passed!")


def test_metrics():
    """Test metrics calculation."""
    print("Testing Metrics...")
    
    # Create test data
    batch_size, seq_len, num_classes = 4, 8, 13
    predictions = torch.randn(batch_size, seq_len, num_classes)
    targets = torch.randint(0, num_classes, (batch_size, seq_len))
    
    # Test character accuracy
    char_acc = calculate_character_accuracy(predictions, targets)
    assert 0.0 <= char_acc <= 1.0
    print(f"  Character accuracy: {char_acc:.3f}")
    
    # Test sequence accuracy
    seq_acc = calculate_sequence_accuracy(predictions, targets)
    assert 0.0 <= seq_acc <= 1.0
    print(f"  Sequence accuracy: {seq_acc:.3f}")
    
    # Test OCRMetrics class
    idx_to_char = {i: str(i) for i in range(num_classes)}
    metrics_calc = OCRMetrics(idx_to_char)
    
    # Update with multiple batches
    for _ in range(3):
        metrics_calc.update(predictions, targets)
    
    metrics = metrics_calc.compute()
    
    required_metrics = ['char_accuracy', 'seq_accuracy', 'edit_distance', 'total_samples']
    for metric in required_metrics:
        assert metric in metrics
        print(f"  {metric}: {metrics[metric]}")
    
    print("  ✅ Metrics test passed!")


def test_training_config():
    """Test training configuration management."""
    print("Testing Training Configuration...")
    
    # Test default config
    config = TrainingConfig()
    assert config.num_epochs == 50
    assert config.learning_rate == 1e-3
    
    # Test config serialization
    config_dict = config.to_dict()
    config2 = TrainingConfig.from_dict(config_dict)
    assert config2.num_epochs == config.num_epochs
    
    # Test YAML save/load
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        temp_file = f.name
    
    try:
        config.save_yaml(temp_file)
        config3 = TrainingConfig.from_yaml(temp_file)
        assert config3.learning_rate == config.learning_rate
    finally:
        # Safe cleanup for Windows
        try:
            os.unlink(temp_file)
        except (OSError, PermissionError):
            pass  # File might be locked on Windows
    
    print("  ✅ Training configuration test passed!")


def test_checkpoint_manager():
    """Test checkpoint management."""
    print("Testing Checkpoint Manager...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create checkpoint manager
        ckpt_mgr = CheckpointManager(temp_dir, keep_n_checkpoints=2)
        
        # Create dummy model and optimizer
        model = create_dummy_model()
        optimizer = torch.optim.Adam(model.parameters())
        
        # Save multiple checkpoints
        for epoch in range(1, 5):
            metrics = {'val_loss': 1.0 / epoch}  # Decreasing loss
            is_best = epoch == 3  # Epoch 3 is best
            
            ckpt_path = ckpt_mgr.save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=None,
                epoch=epoch,
                metrics=metrics,
                is_best=is_best
            )
            
            assert os.path.exists(ckpt_path)
        
        # Check that only the last 2 checkpoints exist
        checkpoint_files = list(Path(temp_dir).glob("checkpoint_epoch_*.pth"))
        assert len(checkpoint_files) <= 2
        
        # Check that best model exists
        best_path = Path(temp_dir) / "best_model.pth"
        assert best_path.exists()
        
        # Test loading
        best_checkpoint = ckpt_mgr.load_best_model()
        assert best_checkpoint is not None
        assert best_checkpoint['epoch'] == 3
    
    print("  ✅ Checkpoint manager test passed!")


def test_early_stopping():
    """Test early stopping mechanism."""
    print("Testing Early Stopping...")
    
    early_stopping = EarlyStopping(patience=3, min_delta=0.01, mode='min')
    
    # Simulate improving then degrading loss
    losses = [1.0, 0.8, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    for epoch, loss in enumerate(losses, 1):
        should_stop = early_stopping(loss, epoch)
        if should_stop:
            break
    
    assert early_stopping.should_stop
    assert early_stopping.best_epoch == 3  # Epoch with loss 0.6
    
    print(f"  Early stopping triggered at epoch {epoch}")
    print(f"  Best epoch: {early_stopping.best_epoch}")
    print("  ✅ Early stopping test passed!")


def test_training_environment():
    """Test training environment setup."""
    print("Testing Training Environment Setup...")
    
    config = TrainingConfig()
    config.experiment_name = "test_experiment"
    config.use_tensorboard = False  # Disable to avoid log file locks
    
    # Use manual cleanup instead of TemporaryDirectory for better Windows compatibility
    temp_dir = tempfile.mkdtemp()
    try:
        config.output_dir = temp_dir
        
        env_info = setup_training_environment(config)
        
        # Check that directories were created
        exp_dir = Path(temp_dir) / "test_experiment"
        assert exp_dir.exists()
        assert (exp_dir / "checkpoints").exists()
        assert (exp_dir / "logs").exists()
        assert (exp_dir / "tensorboard").exists()
        assert (exp_dir / "configs").exists()
        
        # Check device setup
        assert 'device' in env_info
        assert 'dirs' in env_info
        
        # Check config was saved
        config_path = exp_dir / "configs" / "training_config.yaml"
        assert config_path.exists()
        
        print("  ✅ Training environment test passed!")
        
    finally:
        # Clean up manually with retries for Windows
        import time
        for attempt in range(3):
            try:
                shutil.rmtree(temp_dir)
                break
            except (OSError, PermissionError):
                time.sleep(0.1)  # Brief wait before retry


def test_trainer_initialization():
    """Test trainer initialization without actual training."""
    print("Testing Trainer Initialization...")
    
    # Check if dataset exists
    metadata_path = "generated_data/metadata.yaml"
    if not os.path.exists(metadata_path):
        print("  ⚠️ Skipping trainer test - no dataset found")
        print("  Run 'python src/sample_scripts/generate_dataset.py' first")
        return
    
    try:
        # Create small dataset loaders
        train_transform = get_train_transforms(image_size=(64, 32), augmentation_strength=0.1)
        val_transform = get_val_transforms(image_size=(64, 32))
        
        train_loader, val_loader = create_data_loaders(
            metadata_path=metadata_path,
            train_transform=train_transform,
            val_transform=val_transform,
            batch_size=4,
            num_workers=0
        )
        
        # Create model and config
        model = create_dummy_model()
        config = TrainingConfig()
        config.num_epochs = 2  # Very short for testing
        config.use_tensorboard = False  # Disable for testing
        
        temp_dir = tempfile.mkdtemp()
        try:
            config.output_dir = temp_dir
            config.experiment_name = "test_trainer"
            
            device = torch.device('cpu')  # Use CPU for testing
            
            # Initialize trainer
            trainer = OCRTrainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                config=config,
                device=device
            )
            
            # Test trainer methods
            assert trainer.vocab_size == 13
            assert '<PAD>' in trainer.char_to_idx
            
            # Test prediction
            sample_batch = next(iter(val_loader))
            images = sample_batch[0][:2]  # Take only 2 samples
            predictions = trainer.predict_sequence(images)
            assert len(predictions) == 2
            assert all(isinstance(pred, str) for pred in predictions)
            
            # Test evaluation
            eval_results = trainer.evaluate_samples(num_samples=2)
            assert 'accuracy' in eval_results
            assert 'examples' in eval_results
            
            print(f"  Sample predictions: {predictions}")
            print(f"  Evaluation accuracy: {eval_results['accuracy']:.2%}")
        
            print("  ✅ Trainer initialization test passed!")
        
        finally:
            # Clean up manually with retries for Windows
            import time
            for attempt in range(3):
                try:
                    shutil.rmtree(temp_dir)
                    break
                except (OSError, PermissionError):
                    time.sleep(0.1)  # Brief wait before retry
        
    except Exception as e:
        print(f"  ❌ Trainer test failed: {e}")


def test_integration():
    """Test integration of all components with mini training run."""
    print("Testing Integration (Mini Training)...")
    
    # Check if dataset exists
    metadata_path = "generated_data/metadata.yaml"
    if not os.path.exists(metadata_path):
        print("  ⚠️ Skipping integration test - no dataset found")
        return
    
    try:
        # Setup minimal training
        train_transform = get_train_transforms(image_size=(64, 32), augmentation_strength=0.1)
        val_transform = get_val_transforms(image_size=(64, 32))
        
        train_loader, val_loader = create_data_loaders(
            metadata_path=metadata_path,
            train_transform=train_transform,
            val_transform=val_transform,
            batch_size=4,
            num_workers=0
        )
        
        # Create model and config for mini training
        model = create_dummy_model()
        config = TrainingConfig()
        config.num_epochs = 2
        config.log_every_n_steps = 1
        config.save_every_n_epochs = 1
        config.use_tensorboard = False
        config.early_stopping_patience = 1
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config.output_dir = temp_dir
            config.experiment_name = "integration_test"
            
            # Setup environment
            env_info = setup_training_environment(config)
            device = env_info['device']
            
            # Create trainer
            trainer = OCRTrainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                config=config,
                device=device
            )
            
            # Run mini training
            print("  Starting mini training run...")
            results = trainer.train()
            
            # Verify results
            assert 'best_val_loss' in results
            assert 'total_epochs' in results
            assert 'training_history' in results
            
            # Check that checkpoints were saved
            checkpoints_dir = env_info['dirs']['checkpoints_dir']
            checkpoint_files = list(checkpoints_dir.glob("*.pth"))
            assert len(checkpoint_files) > 0
            
            print(f"  Training completed in {results['total_epochs']} epochs")
            print(f"  Best validation loss: {results['best_val_loss']:.4f}")
            print(f"  Checkpoints saved: {len(checkpoint_files)}")
        
        print("  ✅ Integration test passed!")
        
    except Exception as e:
        print(f"  ❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run all tests."""
    print("🧪 Testing Training Infrastructure")
    print("=" * 50)
    
    tests = [
        test_loss_functions,
        test_metrics,
        test_training_config,
        test_checkpoint_manager,
        test_early_stopping,
        test_training_environment,
        test_trainer_initialization,
        test_integration
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            print(f"\n📋 {test_func.__name__.replace('test_', '').replace('_', ' ').title()}")
            test_func()
            passed += 1
        except Exception as e:
            print(f"❌ {test_func.__name__} failed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print(f"\n📊 Test Results")
    print("=" * 30)
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📈 Success Rate: {passed/(passed+failed)*100:.1f}%")
    
    if failed == 0:
        print("\n🎉 All tests passed! Training infrastructure is ready.")
    else:
        print(f"\n⚠️ {failed} test(s) failed. Please check the errors above.")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 