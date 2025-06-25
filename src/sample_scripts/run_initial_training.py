#!/usr/bin/env python3
"""
Initial Training Script for Khmer Digits OCR

This script runs initial training experiments with comprehensive debugging
and analysis to establish baseline performance and identify issues.
"""

import os
import sys
import logging
import torch
import yaml
from pathlib import Path
from datetime import datetime
import tempfile
import argparse

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from modules.data_utils import KhmerDigitsDataset, create_data_loaders
from models import create_model
from modules.trainers import OCRTrainer
from modules.trainers.utils import setup_training_environment, TrainingConfig
# Scheduler is handled internally by the trainer


def setup_logging(log_level: str = "INFO", log_file: str = None):
    """Setup logging configuration."""
    # Configure logging to handle Unicode on Windows
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            *([] if log_file is None else [logging.FileHandler(log_file, encoding='utf-8')])
        ]
    )

def create_training_config():
    """Create a configuration for initial training experiments."""
    config = {
        # Model configuration
        'model': {
            'type': 'ocr_model',
            'characters': {
                'total_classes': 13,
                'max_sequence_length': 9
            },
            'cnn': {
                'type': 'resnet18',
                'feature_size': 512,
                'pretrained': True
            },
            'rnn': {
                'encoder': {
                    'type': 'bilstm',
                    'hidden_size': 256,
                    'num_layers': 2,
                    'dropout': 0.1
                },
                'decoder': {
                    'hidden_size': 256,
                    'num_layers': 1
                },
                'attention': {
                    'hidden_size': 256
                }
            }
        },
        
        # Training configuration
        'training': {
            'batch_size': 32,
            'learning_rate': 0.001,
            'weight_decay': 1e-4,
            'num_epochs': 20,
            'gradient_clip_norm': 1.0,
            'loss_type': 'crossentropy',
            'label_smoothing': 0.0,
            'mixed_precision': True
        },
        
        # Optimizer configuration
        'optimizer': {
            'type': 'adam',
            'learning_rate': 0.001,
            'weight_decay': 1e-4,
            'betas': [0.9, 0.999]
        },
        
        # Scheduler configuration
        'scheduler': {
            'type': 'cosine',
            'warmup_epochs': 2,
            'min_lr': 1e-6
        },
        
        # Early stopping configuration
        'early_stopping': {
            'patience': 5,
            'min_delta': 0.001,
            'monitor': 'val_char_accuracy',
            'mode': 'max'
        },
        
        # Data configuration
        'data': {
            'train_split': 'train',
            'val_split': 'val',
            'num_workers': 2,
            'pin_memory': True,
            'augmentation': True
        },
        
        # Debugging configuration
        'debug': {
            'check_gradients': True,
            'log_sample_predictions': True,
            'save_training_samples': True,
            'analyze_errors': True,
            'confusion_matrix': True
        }
    }
    
    return config

def analyze_gradient_flow(model):
    """Analyze gradient flow through the model."""
    total_norm = 0
    param_count = 0
    gradient_info = {}
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            gradient_info[name] = {
                'norm': param_norm.item(),
                'shape': list(param.shape),
                'mean': param.grad.data.mean().item(),
                'std': param.grad.data.std().item()
            }
            param_count += 1
    
    total_norm = total_norm ** (1. / 2)
    
    return {
        'total_norm': total_norm,
        'param_count': param_count,
        'avg_norm': total_norm / max(param_count, 1),
        'details': gradient_info
    }

def debug_training_step(trainer, batch_idx, images, labels, metadata, config):
    """Debug a single training step."""
    debug_info = {}
    
    # Check input data
    debug_info['batch_info'] = {
        'batch_size': images.size(0),
        'image_shape': list(images.shape),
        'label_shape': list(labels.shape),
        'image_range': (images.min().item(), images.max().item()),
        'unique_labels': len(torch.unique(labels))
    }
    
    # Forward pass
    trainer.model.train()
    images = images.to(trainer.device)
    labels = labels.to(trainer.device)
    
    # Get predictions
    predictions = trainer.model(images, target_sequences=labels)
    
    # Check predictions
    debug_info['predictions'] = {
        'shape': list(predictions.shape),
        'range': (predictions.min().item(), predictions.max().item()),
        'mean': predictions.mean().item(),
        'std': predictions.std().item()
    }
    
    # Calculate loss
    loss_dict = trainer.criterion(predictions, labels)
    loss = loss_dict['loss']
    
    debug_info['loss'] = {
        'value': loss.item(),
        'requires_grad': loss.requires_grad
    }
    
    # Backward pass
    trainer.optimizer.zero_grad()
    loss.backward()
    
    # Check gradients
    if config.get('debug', {}).get('check_gradients', True):
        debug_info['gradients'] = analyze_gradient_flow(trainer.model)
    
    # Update parameters
    if config['training'].get('gradient_clip_norm', 0) > 0:
        torch.nn.utils.clip_grad_norm_(
            trainer.model.parameters(), 
            config['training']['gradient_clip_norm']
        )
    
    trainer.optimizer.step()
    
    return debug_info

def run_training_experiment(config, data_dir, experiment_name):
    """Run a training experiment with debugging."""
    logger = logging.getLogger(__name__)
    
    # Create TrainingConfig object
    training_config = TrainingConfig(
        experiment_name=experiment_name,
        metadata_path=os.path.join(data_dir, 'metadata.yaml'),
        batch_size=config['training']['batch_size'],
        num_epochs=config['training']['num_epochs'],
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        gradient_clip_norm=config['training']['gradient_clip_norm'],
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory'],
        mixed_precision=config['training']['mixed_precision']
    )
    
    # Setup experiment directory
    env_info = setup_training_environment(training_config)
    exp_dir = env_info['dirs']['experiment_dir']
    device = env_info['device']
    
    logger.info(f"Starting training experiment: {experiment_name}")
    logger.info(f"Experiment directory: {exp_dir}")
    logger.info(f"Using device: {device}")
    
        # Create datasets
    logger.info("Loading datasets...")
    metadata_path = os.path.join(data_dir, 'metadata.yaml')
    
    # Create data loaders with transforms
    from modules.data_utils import get_train_transforms, get_val_transforms
    train_transform = get_train_transforms() if config['data']['augmentation'] else get_val_transforms()
    val_transform = get_val_transforms()
    
    train_loader, val_loader = create_data_loaders(
        metadata_path=metadata_path,
        train_transform=train_transform,
        val_transform=val_transform,
        batch_size=config['training']['batch_size'],
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory']
    )
    
    # Get dataset objects for character mappings
    train_dataset = KhmerDigitsDataset(metadata_path=metadata_path, split='train')
    val_dataset = KhmerDigitsDataset(metadata_path=metadata_path, split='val')
    
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    
    # Create model
    logger.info("Creating model...")
    model = create_model(
        config['model'],
        max_sequence_length=9,
        vocab_size=13
    )
    logger.info(f"Model max_sequence_length: {model.max_sequence_length}")
    model = model.to(device)
    
    # Create trainer
    logger.info("Initializing trainer...")
    trainer = OCRTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=training_config,
        device=device,
        char_to_idx=train_dataset.char_to_idx,
        idx_to_char=train_dataset.idx_to_char
    )
    
    # Initial evaluation
    logger.info("Running initial evaluation...")
    initial_results = trainer.evaluate_samples(num_samples=10)
    logger.info(f"Initial accuracy: {initial_results['accuracy']:.1%}")
    
    # Debug first training batch
    logger.info("Debugging first training batch...")
    data_iter = iter(train_loader)
    images, labels, metadata = next(data_iter)
    
    debug_info = debug_training_step(trainer, 0, images, labels, metadata, config)
    logger.info(f"Debug info: {debug_info}")
    
    # Train model
    logger.info("Starting training...")
    try:
        training_results = trainer.train()
        logger.info("Training completed successfully!")
        
        # Final evaluation
        logger.info("Running final evaluation...")
        final_results = trainer.evaluate_samples(num_samples=20)
        logger.info(f"Final accuracy: {final_results['accuracy']:.1%}")
        
        # Error analysis
        if config.get('debug', {}).get('analyze_errors', True):
            logger.info("Analyzing errors...")
            error_analysis = trainer.analyze_errors(num_samples=50)
            logger.info(f"Error analysis completed. Found {len(error_analysis.get('errors', []))} errors.")
        
        # Save model
        model_path = os.path.join(exp_dir, 'final_model.pth')
        trainer.save_model(model_path)
        logger.info(f"Model saved: {model_path}")
        
        return {
            'success': True,
            'experiment_dir': exp_dir,
            'training_results': training_results,
            'initial_results': initial_results,
            'final_results': final_results,
            'model_path': model_path
        }
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        return {
            'success': False,
            'error': str(e),
            'experiment_dir': exp_dir
        }

def main():
    """Main function to run initial training experiments."""
    parser = argparse.ArgumentParser(description='Run initial training experiments')
    parser.add_argument('--data-dir', type=str, default='generated_data',
                       help='Path to generated data directory')
    parser.add_argument('--experiment-name', type=str, default=None,
                       help='Name for the experiment')
    parser.add_argument('--config-file', type=str, default=None,
                       help='Path to configuration file')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f"training_log_{timestamp}.txt"
    setup_logging(args.log_level, log_file)
    
    logger = logging.getLogger(__name__)
    logger.info("Starting Initial Training and Debugging")
    logger.info("=" * 50)
    
    # Create experiment name
    if args.experiment_name is None:
        args.experiment_name = f"initial_training_{timestamp}"
    
    # Load or create configuration
    if args.config_file and os.path.exists(args.config_file):
        with open(args.config_file, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from: {args.config_file}")
    else:
        config = create_training_config()
        logger.info("Using default configuration")
    
    # Check data directory
    if not os.path.exists(args.data_dir):
        logger.error(f"Data directory not found: {args.data_dir}")
        return False
    
    # Run training experiment
    results = run_training_experiment(config, args.data_dir, args.experiment_name)
    
    # Report results
    if results['success']:
        logger.info("Training experiment completed successfully!")
        logger.info(f"Results saved in: {results['experiment_dir']}")
        logger.info(f"Final accuracy: {results['final_results']['accuracy']:.1%}")
    else:
        logger.error("Training experiment failed!")
        logger.error(f"Error: {results['error']}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)