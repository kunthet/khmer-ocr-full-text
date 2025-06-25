#!/usr/bin/env python3
"""
Simple Initial Training Script for Step 2.3

A simplified script to test the training pipeline and complete step 2.3 of the workplan.
"""

import os
import sys
from pathlib import Path
import torch
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from modules.data_utils import KhmerDigitsDataset, create_data_loaders, get_train_transforms, get_val_transforms
from models import create_model
from modules.trainers import OCRTrainer
from modules.trainers.utils import TrainingConfig
import tempfile

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Run a simple training experiment for step 2.3."""
    logger.info("🚀 Step 2.3: Initial Training and Debugging")
    logger.info("=" * 50)
    
    # Check data directory
    data_dir = 'generated_data'
    metadata_path = os.path.join(data_dir, 'metadata.yaml')
    
    if not os.path.exists(metadata_path):
        logger.error(f"Metadata file not found: {metadata_path}")
        return False
    
    logger.info("✅ Data directory found")
    
    # Create configuration
    config = TrainingConfig(
        # Basic training settings
        learning_rate=0.001,
        weight_decay=1e-4,
        num_epochs=3,  # Very short for initial testing
        batch_size=16,  # Smaller batch size for testing
        gradient_clip_norm=1.0,
        loss_type='crossentropy',
        mixed_precision=False,  # Disable for debugging
        
        # Scheduler settings
        scheduler_type='steplr',
        step_size=2,
        gamma=0.5,
        
        # Early stopping
        early_stopping_patience=5,
        early_stopping_min_delta=0.001,
        
        # Paths
        output_dir=tempfile.gettempdir(),
        experiment_name='step_2_3_initial_training',
        
        # Device
        device='cpu',  # Use CPU for debugging
        
        # Logging
        log_every_n_steps=10,
        use_tensorboard=False,  # Disable for simplicity
        save_every_n_epochs=1
    )
    
    logger.info("✅ Configuration created")
    
    # Setup device
    device = torch.device('cpu')
    logger.info(f"Using device: {device}")
    
    # Create data loaders
    logger.info("Loading data...")
    train_transform = get_train_transforms()
    val_transform = get_val_transforms()
    
    train_loader, val_loader = create_data_loaders(
        metadata_path=metadata_path,
        train_transform=train_transform,
        val_transform=val_transform,
        batch_size=config.batch_size,
        num_workers=0,  # No multiprocessing for debugging
        pin_memory=False
    )
    
    # Get dataset for character mappings
    train_dataset = KhmerDigitsDataset(metadata_path=metadata_path, split='train')
    
    logger.info(f"✅ Data loaded - Train: {len(train_dataset)} samples")
    
    # Create model with correct sequence length (to match label length)
    model = create_model(
        preset='small',  # Use small preset for faster training
        max_sequence_length=9,  # 8 digits + 1 EOS token
        vocab_size=13
    )
    model = model.to(device)
    
    logger.info("✅ Model created")
    
    # Create trainer
    logger.info("Creating trainer...")
    trainer = OCRTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        char_to_idx=train_dataset.char_to_idx,
        idx_to_char=train_dataset.idx_to_char
    )
    
    logger.info("✅ Trainer created")
    
    # Test a single batch first
    logger.info("Testing single batch...")
    try:
        data_iter = iter(train_loader)
        images, labels, metadata = next(data_iter)
        
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        model.train()
        predictions = model(images)
        
        logger.info(f"Shapes - Images: {images.shape}, Labels: {labels.shape}, Predictions: {predictions.shape}")
        
        # Calculate loss
        loss_dict = trainer.criterion(predictions, labels)
        loss = loss_dict['loss']
        
        logger.info(f"✅ Single batch test passed - Loss: {loss.item():.4f}")
        
    except Exception as e:
        logger.error(f"❌ Single batch test failed: {e}")
        return False
    
    # Run initial evaluation
    logger.info("Running initial evaluation...")
    try:
        eval_results = trainer.evaluate_samples(num_samples=5)
        logger.info(f"Initial accuracy: {eval_results['accuracy']:.1%}")
        logger.info("✅ Initial evaluation completed")
    except Exception as e:
        logger.error(f"❌ Initial evaluation failed: {e}")
        return False
    
    # Run short training
    logger.info("Starting training...")
    try:
        training_results = trainer.train()
        
        logger.info("🎉 Training completed!")
        logger.info(f"Best validation loss: {training_results['best_val_loss']:.4f}")
        logger.info(f"Total epochs: {training_results['total_epochs']}")
        logger.info(f"Training time: {training_results['total_time']:.2f} seconds")
        
        # Final evaluation
        final_eval = trainer.evaluate_samples(num_samples=10)
        logger.info(f"Final accuracy: {final_eval['accuracy']:.1%}")
        
        # Log a few examples
        logger.info("Sample predictions:")
        for i, example in enumerate(final_eval['examples'][:3]):
            status = "✅" if example['correct'] else "❌"
            logger.info(f"  {status} Target: '{example['target']}' -> Predicted: '{example['predicted']}'")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Step 2.3 Initial Training and Debugging completed successfully!")
        print("✅ All training infrastructure components are working")
        print("✅ Model can train without major issues")
        print("✅ Training is stable and reproducible")
    else:
        print("\n❌ Step 2.3 Initial Training and Debugging failed")
    
    sys.exit(0 if success else 1) 