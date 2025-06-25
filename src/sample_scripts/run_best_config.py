#!/usr/bin/env python3
"""
Run Best Configuration from Hyperparameter Tuning

Based on the Phase 3.1 results, this script runs the best performing
configuration (conservative_small) with full training epochs.
"""

import os
import sys
import yaml
import logging
import time
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from modules.data_utils import KhmerDigitsDataset, create_data_loaders
from models import create_model
from modules.trainers import OCRTrainer
from modules.trainers.utils import setup_training_environment, TrainingConfig


def main():
    """Run the best configuration from hyperparameter tuning."""
    
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"best_config_training_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding='utf-8')
        ]
    )
    logger = logging.getLogger(__name__)
    
    logger.info("🚀 Starting training with best configuration: conservative_small")
    
    # Load configuration
    config_file = "config/phase3_training_configs.yaml"
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get the best experiment configuration
    best_config = config['experiments']['conservative_small']
    best_config['experiment_name'] = f"best_conservative_small_{timestamp}"
    
    logger.info("📋 Configuration Summary:")
    logger.info(f"  Model: {best_config['model']['name']}")
    logger.info(f"  Batch Size: {best_config['training']['batch_size']}")
    logger.info(f"  Learning Rate: {best_config['training']['learning_rate']}")
    logger.info(f"  Epochs: {best_config['training']['num_epochs']}")
    logger.info(f"  Loss Type: {best_config['training']['loss_type']}")
    
    try:
        # Create training configuration
        training_config = TrainingConfig(
            experiment_name=best_config['experiment_name'],
            model_name=best_config['model']['name'],
            model_config_path=best_config['model']['config_path'],
            metadata_path=best_config['data']['metadata_path'],
            batch_size=best_config['training']['batch_size'],
            num_workers=best_config['data']['num_workers'],
            pin_memory=best_config['data']['pin_memory'],
            learning_rate=best_config['training']['learning_rate'],
            weight_decay=best_config['training']['weight_decay'],
            num_epochs=best_config['training']['num_epochs'],
            device=best_config['training']['device'],
            mixed_precision=best_config['training']['mixed_precision'],
            gradient_clip_norm=best_config['training']['gradient_clip_norm'],
            loss_type=best_config['training']['loss_type'],
            label_smoothing=best_config['training'].get('label_smoothing', 0.0),
            scheduler_type=best_config['scheduler']['type'],
            early_stopping_patience=best_config['early_stopping']['patience'],
            early_stopping_min_delta=best_config['early_stopping']['min_delta'],
            log_every_n_steps=best_config['training']['log_every_n_steps'],
            save_every_n_epochs=best_config['training']['save_every_n_epochs'],
            keep_n_checkpoints=best_config['training']['keep_n_checkpoints'],
            use_tensorboard=best_config['training']['use_tensorboard'],
            output_dir="training_output"
        )
        
        # Setup environment
        logger.info("🔧 Setting up training environment...")
        env_info = setup_training_environment(training_config)
        device = env_info['device']
        exp_dir = env_info['dirs']['experiment_dir']
        
        # Load datasets
        logger.info("📚 Loading datasets...")
        from modules.data_utils.preprocessing import get_train_transforms, get_val_transforms
        
        train_dataset = KhmerDigitsDataset(
            metadata_path=training_config.metadata_path,
            split='train',
            transform=get_train_transforms()
        )
        val_dataset = KhmerDigitsDataset(
            metadata_path=training_config.metadata_path,
            split='val',
            transform=get_val_transforms()
        )
        
        # Create data loaders
        from torch.utils.data import DataLoader
        from modules.data_utils.dataset import collate_fn
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=training_config.batch_size,
            shuffle=True,
            num_workers=training_config.num_workers,
            pin_memory=training_config.pin_memory,
            collate_fn=collate_fn
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=training_config.batch_size,
            shuffle=False,
            num_workers=training_config.num_workers,
            pin_memory=training_config.pin_memory,
            collate_fn=collate_fn
        )
        
        logger.info(f"📊 Dataset Info:")
        logger.info(f"  Training samples: {len(train_dataset)}")
        logger.info(f"  Validation samples: {len(val_dataset)}")
        logger.info(f"  Vocabulary size: {len(train_dataset.char_to_idx)}")
        
        # Create model
        logger.info("🏗️  Creating model...")
        model = create_model(
            preset=training_config.model_name,
            vocab_size=len(train_dataset.char_to_idx),
            max_sequence_length=train_dataset.max_sequence_length + 1  # +1 for EOS token
        )
        
        # Initialize trainer
        trainer = OCRTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=training_config,
            device=device
        )
        
        # Run training
        logger.info("🎯 Starting training...")
        start_time = time.time()
        training_history = trainer.train()
        end_time = time.time()
        
        # Training completed
        total_time = end_time - start_time
        logger.info(f"✅ Training completed in {total_time:.2f} seconds")
        
        # Extract final metrics
        training_hist = training_history.get('training_history', {})
        val_metrics_list = training_hist.get('val_metrics', [])
        
        if val_metrics_list:
            final_metrics = val_metrics_list[-1]
            best_char_acc = max([m.get('char_accuracy', 0.0) for m in val_metrics_list])
            best_seq_acc = max([m.get('seq_accuracy', 0.0) for m in val_metrics_list])
            
            logger.info("🏆 Final Results:")
            logger.info(f"  Best Character Accuracy: {best_char_acc:.1%}")
            logger.info(f"  Best Sequence Accuracy: {best_seq_acc:.1%}")
            logger.info(f"  Final Character Accuracy: {final_metrics.get('char_accuracy', 0.0):.1%}")
            logger.info(f"  Final Sequence Accuracy: {final_metrics.get('seq_accuracy', 0.0):.1%}")
            logger.info(f"  Training Time: {total_time:.1f}s")
            
            # Check if we exceeded target accuracy
            target_char_acc = 0.85  # 85% target from config
            if best_char_acc >= target_char_acc:
                logger.info(f"🎉 TARGET ACHIEVED! Character accuracy {best_char_acc:.1%} >= {target_char_acc:.1%}")
            else:
                logger.info(f"📈 Progress: {best_char_acc:.1%} / {target_char_acc:.1%} character accuracy")
        
        # Save results summary
        results_summary = {
            'experiment_name': best_config['experiment_name'],
            'completion_time': datetime.now().isoformat(),
            'training_time_seconds': total_time,
            'configuration': {
                'model': best_config['model']['name'],
                'batch_size': best_config['training']['batch_size'],
                'learning_rate': best_config['training']['learning_rate'],
                'epochs_completed': training_history.get('total_epochs', 0),
                'loss_type': best_config['training']['loss_type']
            }
        }
        
        if val_metrics_list:
            results_summary['performance'] = {
                'best_char_accuracy': best_char_acc,
                'best_seq_accuracy': best_seq_acc,
                'final_char_accuracy': final_metrics.get('char_accuracy', 0.0),
                'final_seq_accuracy': final_metrics.get('seq_accuracy', 0.0)
            }
        
        # Save to file
        results_file = f"best_config_results_{timestamp}.json"
        import json
        with open(results_file, 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        logger.info(f"📁 Results saved to: {results_file}")
        logger.info(f"📁 Model checkpoints in: {exp_dir}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Training failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False


if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎯 Best configuration training completed successfully!")
        print("📊 Check the log file and results JSON for detailed metrics.")
    else:
        print("\n❌ Training failed. Check the log file for details.")
        sys.exit(1) 