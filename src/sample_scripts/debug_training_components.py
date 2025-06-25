#!/usr/bin/env python3
"""
Debug Training Components

Simple script to test each training component individually to identify issues.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import logging

def test_data_loading():
    """Test data loading functionality."""
    print("🔍 Testing data loading...")
    
    try:
        from modules.data_utils import KhmerDigitsDataset, create_data_loaders, get_train_transforms, get_val_transforms
        
        # Test dataset creation
        metadata_path = 'generated_data/metadata.yaml'
        
        train_dataset = KhmerDigitsDataset(
            metadata_path=metadata_path,
            split='train'
        )
        
        val_dataset = KhmerDigitsDataset(
            metadata_path=metadata_path,
            split='val'
        )
        
        print(f"  ✅ Training dataset: {len(train_dataset)} samples")
        print(f"  ✅ Validation dataset: {len(val_dataset)} samples")
        
        # Test data loaders
        train_transform = get_train_transforms()
        val_transform = get_val_transforms()
        
        train_loader, val_loader = create_data_loaders(
            metadata_path=metadata_path,
            train_transform=train_transform,
            val_transform=val_transform,
            batch_size=4,
            num_workers=0,
            pin_memory=False
        )
        
        print(f"  ✅ Data loaders created successfully")
        
        # Test getting a batch
        for images, labels, metadata in train_loader:
            print(f"  ✅ Batch shapes: images={images.shape}, labels={labels.shape}")
            break
            
        return True, train_dataset, val_dataset, train_loader, val_loader
        
    except Exception as e:
        print(f"  ❌ Data loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None, None, None

def test_model_creation():
    """Test model creation functionality."""
    print("🔍 Testing model creation...")
    
    try:
        from models import create_model
        
        # Create model with correct sequence length using factory pattern
        model = create_model(
            preset='small',  # Use small preset for faster testing
            max_sequence_length=9,  # 8 digits + 1 EOS token
            vocab_size=13
        )
        print(f"  ✅ Model created: {type(model).__name__}")
        
        # Test forward pass
        dummy_input = torch.randn(2, 3, 64, 128)
        output = model(dummy_input)
        print(f"  ✅ Forward pass works: output shape={output.shape}")
        
        return True, model
        
    except Exception as e:
        print(f"  ❌ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_trainer_creation():
    """Test trainer creation."""
    print("🔍 Testing trainer creation...")
    
    try:
        from modules.trainers import OCRTrainer
        from modules.trainers.utils import TrainingConfig
        
        # Get basic components first
        success_data, train_dataset, val_dataset, train_loader, val_loader = test_data_loading()
        if not success_data:
            return False, None
            
        success_model, model = test_model_creation()
        if not success_model:
            return False, None
        
        # Create basic config using TrainingConfig class
        config = TrainingConfig(
            learning_rate=0.001,
            weight_decay=1e-4,
            num_epochs=5,
            gradient_clip_norm=1.0,
            loss_type='crossentropy',
            mixed_precision=False,  # Disable for debugging
            early_stopping_patience=3,
            early_stopping_min_delta=0.001,
            scheduler_type='steplr',
            step_size=5,
            gamma=0.5
        )
        
        device = torch.device('cpu')  # Use CPU for debugging
        model = model.to(device)
        
        trainer = OCRTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=device,
            char_to_idx=train_dataset.char_to_idx,
            idx_to_char=train_dataset.idx_to_char
        )
        
        print(f"  ✅ Trainer created successfully")
        
        return True, trainer
        
    except Exception as e:
        print(f"  ❌ Trainer creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_single_training_step():
    """Test a single training step."""
    print("🔍 Testing single training step...")
    
    try:
        success, trainer = test_trainer_creation()
        if not success:
            return False
        
        # Get a batch
        data_iter = iter(trainer.train_loader)
        images, labels, metadata = next(data_iter)
        
        # Move to device
        images = images.to(trainer.device)
        labels = labels.to(trainer.device)
        
        # Forward pass
        trainer.model.train()
        predictions = trainer.model(images)
        print(f"  ✅ Forward pass: predictions shape={predictions.shape}")
        
        # Calculate loss
        print(f"  🔍 Debug shapes: predictions={predictions.shape}, labels={labels.shape}")
        print(f"  🔍 Labels content: {labels}")
        
        loss_dict = trainer.criterion(predictions, labels)
        loss = loss_dict['loss']
        print(f"  ✅ Loss calculation: loss={loss.item():.4f}")
        
        # Backward pass
        trainer.optimizer.zero_grad()
        loss.backward()
        trainer.optimizer.step()
        
        print(f"  ✅ Single training step completed")
        return True
        
    except Exception as e:
        print(f"  ❌ Training step failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all component tests."""
    print("🧪 Debugging Training Components")
    print("=" * 50)
    
    # Test each component
    test_data_loading()
    print()
    
    test_model_creation()
    print()
    
    test_trainer_creation()
    print()
    
    test_single_training_step()
    print()
    
    print("🎉 Component debugging complete!")

if __name__ == "__main__":
    main() 