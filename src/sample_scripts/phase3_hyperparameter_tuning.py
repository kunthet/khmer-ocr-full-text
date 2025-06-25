#!/usr/bin/env python3
"""
Phase 3.1 Hyperparameter Tuning Script

This script runs systematic hyperparameter tuning experiments to optimize
the Khmer digits OCR model performance beyond the baseline 24% accuracy.
"""

import os
import sys
import yaml
import json
import logging
import argparse
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import torch

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from modules.data_utils import KhmerDigitsDataset, create_data_loaders
from models import create_model
from modules.trainers import OCRTrainer
from modules.trainers.utils import setup_training_environment, TrainingConfig


class HyperparameterTuner:
    """Systematic hyperparameter tuning for Khmer OCR model."""
    
    def __init__(self, config_file: str):
        """Initialize the hyperparameter tuner."""
        self.config_file = config_file
        self.results = []
        self.best_result = None
        self.experiments_completed = 0
        
        # Load configuration
        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup logging
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging for the tuning process."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"hyperparameter_tuning_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(log_file, encoding='utf-8')
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def create_training_config(self, experiment_config: Dict) -> TrainingConfig:
        """Create TrainingConfig object from experiment configuration."""
        config = TrainingConfig(
            experiment_name=experiment_config['experiment_name'],
            model_name=experiment_config['model']['name'],
            model_config_path=experiment_config['model']['config_path'],
            metadata_path=experiment_config['data']['metadata_path'],
            batch_size=experiment_config['training']['batch_size'],
            num_workers=experiment_config['data']['num_workers'],
            pin_memory=experiment_config['data']['pin_memory'],
            learning_rate=experiment_config['training']['learning_rate'],
            weight_decay=experiment_config['training']['weight_decay'],
            num_epochs=experiment_config['training']['num_epochs'],
            device=experiment_config['training']['device'],
            mixed_precision=experiment_config['training']['mixed_precision'],
            gradient_clip_norm=experiment_config['training']['gradient_clip_norm'],
            loss_type=experiment_config['training']['loss_type'],
            label_smoothing=experiment_config['training'].get('label_smoothing', 0.0),
            scheduler_type=experiment_config['scheduler']['type'],
            step_size=experiment_config['scheduler'].get('step_size', 10),
            gamma=experiment_config['scheduler'].get('gamma', 0.5),
            early_stopping_patience=experiment_config['early_stopping']['patience'],
            early_stopping_min_delta=experiment_config['early_stopping']['min_delta'],
            log_every_n_steps=experiment_config['training']['log_every_n_steps'],
            save_every_n_epochs=experiment_config['training']['save_every_n_epochs'],
            keep_n_checkpoints=experiment_config['training']['keep_n_checkpoints'],
            use_tensorboard=experiment_config['training']['use_tensorboard'],
            output_dir="training_output"
        )
        
        return config

    def run_single_experiment(self, experiment_name: str, experiment_config: Dict) -> Dict:
        """Run a single hyperparameter experiment."""
        self.logger.info(f"Starting experiment: {experiment_name}")
        
        start_time = time.time()
        
        try:
            # Create training configuration
            training_config = self.create_training_config(experiment_config)
            
            # Setup environment
            env_info = setup_training_environment(training_config)
            device = env_info['device']
            exp_dir = env_info['dirs']['experiment_dir']
            
            # Load datasets
            self.logger.info("Loading datasets...")
            
            # Import transforms
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
            
            self.logger.info(f"Training samples: {len(train_dataset)}")
            self.logger.info(f"Validation samples: {len(val_dataset)}")
            
            # Create model
            self.logger.info("Creating model...")
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
            self.logger.info("Starting training...")
            training_history = trainer.train()
            
            # Calculate metrics
            end_time = time.time()
            training_time = end_time - start_time
            
            # Extract metrics correctly from training history
            training_hist = training_history.get('training_history', {})
            val_metrics_list = training_hist.get('val_metrics', [])
            
            # Calculate best metrics across all epochs
            best_char_acc = 0.0
            best_seq_acc = 0.0
            
            if val_metrics_list:
                char_accuracies = [m.get('char_accuracy', 0.0) for m in val_metrics_list]
                seq_accuracies = [m.get('seq_accuracy', 0.0) for m in val_metrics_list]
                best_char_acc = max(char_accuracies) if char_accuracies else 0.0
                best_seq_acc = max(seq_accuracies) if seq_accuracies else 0.0
            
            # Get final loss
            train_losses = training_hist.get('train_loss', [])
            final_train_loss = train_losses[-1] if train_losses else float('inf')
            
            # Create result
            result = {
                'experiment_name': experiment_name,
                'status': 'completed',
                'training_time': training_time,
                'best_val_char_accuracy': best_char_acc,
                'best_val_seq_accuracy': best_seq_acc,
                'final_train_loss': final_train_loss,
                'hyperparameters': {
                    'model_name': experiment_config['model']['name'],
                    'batch_size': experiment_config['training']['batch_size'],
                    'learning_rate': experiment_config['training']['learning_rate'],
                    'weight_decay': experiment_config['training']['weight_decay'],
                    'loss_type': experiment_config['training']['loss_type'],
                    'scheduler_type': experiment_config['scheduler']['type']
                },
                'training_history': training_hist  # Include full history for analysis
            }
            
            self.logger.info(f"Experiment {experiment_name} completed successfully!")
            self.logger.info(f"Best character accuracy: {result['best_val_char_accuracy']:.4f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Experiment {experiment_name} failed: {str(e)}")
            return {
                'experiment_name': experiment_name,
                'status': 'failed',
                'error': str(e),
                'training_time': time.time() - start_time
            }

    def run_experiments(self, experiment_names: List[str] = None):
        """Run all or specified experiments."""
        experiments = self.config['experiments']
        
        if experiment_names:
            experiments = {name: config for name, config in experiments.items() 
                          if name in experiment_names}
        
        self.logger.info(f"Starting hyperparameter tuning with {len(experiments)} experiments")
        
        for exp_name, exp_config in experiments.items():
            result = self.run_single_experiment(exp_name, exp_config)
            self.results.append(result)
            
            # Update best result
            if (result.get('status') == 'completed' and 
                (self.best_result is None or 
                 result['best_val_char_accuracy'] > 
                 self.best_result['best_val_char_accuracy'])):
                self.best_result = result
            
            self.experiments_completed += 1

    def save_results(self):
        """Save tuning results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"hyperparameter_tuning_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump({
                'best_result': self.best_result,
                'all_results': self.results
            }, f, indent=2)
        
        self.logger.info(f"Results saved to {results_file}")


def main():
    """Main function for hyperparameter tuning."""
    parser = argparse.ArgumentParser(description="Phase 3.1 Hyperparameter Tuning")
    parser.add_argument("--config", default="config/phase3_training_configs.yaml",
                        help="Configuration file for experiments")
    parser.add_argument("--experiments", nargs='+', 
                        help="Specific experiments to run (default: all)")
    
    args = parser.parse_args()
    
    # Initialize tuner
    tuner = HyperparameterTuner(args.config)
    
    # Run experiments
    tuner.run_experiments(args.experiments)
    
    # Save results
    tuner.save_results()
    
    print(f"\n✅ Hyperparameter tuning completed!")
    if tuner.best_result:
        print(f"🏆 Best result: {tuner.best_result['experiment_name']}")
        print(f"   Character accuracy: {tuner.best_result['best_val_char_accuracy']:.4f}")


if __name__ == "__main__":
    main() 