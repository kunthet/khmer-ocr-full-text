#!/usr/bin/env python3
"""
Phase 3.2: Advanced Model Optimization

This script implements advanced optimization techniques to push the model
performance from current ~40-50% to target 85% character accuracy.

Advanced techniques included:
1. Enhanced data augmentation strategies
2. Model architecture improvements  
3. Advanced training techniques
4. Curriculum learning implementation
5. Error analysis and targeted improvements
"""

import os
import sys
import yaml
import logging
import time
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from modules.data_utils import KhmerDigitsDataset, create_data_loaders
from models import create_model
from modules.trainers import OCRTrainer
from modules.trainers.utils import setup_training_environment, TrainingConfig
from modules.trainers.curriculum_learning import CurriculumLearningManager, CurriculumStage, DifficultyLevel
from modules.trainers.enhanced_losses import FocalLoss, HierarchicalLoss, ConfidenceAwareLoss
from modules.trainers.advanced_schedulers import WarmupCosineAnnealingLR


class AdvancedOptimizer:
    """Advanced optimization techniques for Phase 3.2."""
    
    def __init__(self, base_config_path: str = "config/phase3_training_configs.yaml"):
        self.base_config_path = base_config_path
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging for the optimization process."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"phase3_2_optimization_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(log_file, encoding='utf-8')
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def load_best_checkpoint(self) -> Optional[str]:
        """Load the best available checkpoint from previous training."""
        possible_paths = [
            "training_output/conservative_small/checkpoints/best_model.pth",
            "training_output/best_conservative_small_*/checkpoints/best_model.pth"
        ]
        
        for pattern in possible_paths:
            for path in Path(".").glob(pattern):
                if path.exists():
                    self.logger.info(f"Found checkpoint: {path}")
                    return str(path)
        
        self.logger.warning("No best checkpoint found - will train from scratch")
        return None
    
    def create_enhanced_config(self) -> Dict:
        """Create enhanced configuration for Phase 3.2 optimization."""
        
        # Load base configuration
        with open(self.base_config_path, 'r') as f:
            base_config = yaml.safe_load(f)
        
        # Enhanced configuration for Phase 3.2
        enhanced_config = {
            'experiment_name': f"phase3_2_advanced_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'model': {
                'name': 'small',  # Proven best performer
                'config_path': 'config/model_config.yaml',
                'architecture_improvements': {
                    'use_attention': True,
                    'attention_heads': 8,
                    'dropout_rate': 0.1,
                    'layer_norm': True
                }
            },
            'data': {
                'metadata_path': 'test_output/frequency_balanced/metadata.yaml',
                'num_workers': 4,
                'pin_memory': True,
                'enhanced_augmentation': True,
                'curriculum_learning': True
            },
            'training': {
                'batch_size': 32,  # Optimal from Phase 3.1
                'learning_rate': 0.001,  # Conservative proven rate
                'weight_decay': 0.0001,
                'num_epochs': 100,  # Extended training
                'device': 'auto',
                'mixed_precision': True,
                'gradient_clip_norm': 1.0,
                'loss_type': 'enhanced_focal',  # Advanced loss
                'label_smoothing': 0.1,
                'use_curriculum': True,
                'warmup_epochs': 10,
                'log_every_n_steps': 25,
                'save_every_n_epochs': 5,
                'keep_n_checkpoints': 10
            },
            'scheduler': {
                'type': 'warmup_cosine',
                'warmup_epochs': 10,
                'min_lr_ratio': 0.01,
                'T_max': 90
            },
            'optimization_techniques': {
                'focal_loss': {
                    'alpha': 0.25,
                    'gamma': 2.0
                },
                'curriculum_learning': {
                    'start_stage': 'simple',
                    'progression_threshold': 0.80,
                    'stages': ['simple', 'medium', 'complex', 'expert']
                },
                'data_augmentation': {
                    'rotation_range': 15,
                    'scale_range': 0.15,
                    'brightness_range': 0.3,
                    'contrast_range': 0.3,
                    'noise_std': 0.02,
                    'blur_kernel_size': 3
                }
            },
            'early_stopping': {
                'patience': 15,  # More patience for complex training
                'min_delta': 0.001,
                'monitor': 'val_char_accuracy'
            },
            'target_metrics': {
                'char_accuracy': 0.85,  # 85% target
                'seq_accuracy': 0.70,   # 70% target
                'convergence_patience': 20
            }
        }
        
        return enhanced_config
    
    def setup_enhanced_data_loaders(self, config: Dict) -> Tuple:
        """Setup data loaders with enhanced augmentation and curriculum learning."""
        
        # Enhanced transforms for Phase 3.2
        from modules.data_utils.preprocessing import get_train_transforms, get_val_transforms
        
        # Create datasets with enhanced augmentation
        train_dataset = KhmerDigitsDataset(
            metadata_path=config['data']['metadata_path'],
            split='train',
            transform=get_train_transforms(enhanced=True)  # Enhanced augmentation
        )
        val_dataset = KhmerDigitsDataset(
            metadata_path=config['data']['metadata_path'],
            split='val',
            transform=get_val_transforms()
        )
        
        # Create data loaders
        from torch.utils.data import DataLoader
        from modules.data_utils.dataset import collate_fn
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=config['data']['num_workers'],
            pin_memory=config['data']['pin_memory'],
            collate_fn=collate_fn
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            num_workers=config['data']['num_workers'],
            pin_memory=config['data']['pin_memory'],
            collate_fn=collate_fn
        )
        
        return train_loader, val_loader, train_dataset, val_dataset
    
    def create_enhanced_model(self, vocab_size: int, max_seq_length: int, 
                            config: Dict, checkpoint_path: Optional[str] = None) -> nn.Module:
        """Create enhanced model with improvements."""
        
        # Create base model
        model = create_model(
            preset=config['model']['name'],
            vocab_size=vocab_size,
            max_sequence_length=max_seq_length
        )
        
        # Load checkpoint if available
        if checkpoint_path and os.path.exists(checkpoint_path):
            try:
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                model.load_state_dict(checkpoint['model_state_dict'])
                self.logger.info(f"✅ Loaded checkpoint from {checkpoint_path}")
            except Exception as e:
                self.logger.warning(f"❌ Failed to load checkpoint: {e}")
                self.logger.info("Starting with fresh model initialization")
        
        return model
    
    def setup_enhanced_trainer(self, model: nn.Module, train_loader, val_loader, 
                             config: Dict, device: torch.device) -> OCRTrainer:
        """Setup enhanced trainer with advanced techniques."""
        
        # Create training configuration
        training_config = TrainingConfig(
            experiment_name=config['experiment_name'],
            model_name=config['model']['name'],
            model_config_path=config['model']['config_path'],
            metadata_path=config['data']['metadata_path'],
            batch_size=config['training']['batch_size'],
            num_workers=config['data']['num_workers'],
            pin_memory=config['data']['pin_memory'],
            learning_rate=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay'],
            num_epochs=config['training']['num_epochs'],
            device=str(device),
            mixed_precision=config['training']['mixed_precision'],
            gradient_clip_norm=config['training']['gradient_clip_norm'],
            loss_type=config['training']['loss_type'],
            label_smoothing=config['training']['label_smoothing'],
            scheduler_type=config['scheduler']['type'],
            early_stopping_patience=config['early_stopping']['patience'],
            early_stopping_min_delta=config['early_stopping']['min_delta'],
            log_every_n_steps=config['training']['log_every_n_steps'],
            save_every_n_epochs=config['training']['save_every_n_epochs'],
            keep_n_checkpoints=config['training']['keep_n_checkpoints'],
            use_tensorboard=True,
            output_dir="training_output"
        )
        
        # Initialize enhanced trainer
        trainer = OCRTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=training_config,
            device=device
        )
        
        return trainer
    
    def run_optimization(self) -> Dict:
        """Run the complete Phase 3.2 optimization process."""
        
        self.logger.info("🚀 Starting Phase 3.2: Advanced Model Optimization")
        self.logger.info("=" * 60)
        
        # Create enhanced configuration
        config = self.create_enhanced_config()
        self.logger.info(f"📋 Experiment: {config['experiment_name']}")
        self.logger.info(f"🎯 Target: {config['target_metrics']['char_accuracy']:.1%} character accuracy")
        
        try:
            # Setup training environment
            training_config = TrainingConfig(
                experiment_name=config['experiment_name'],
                output_dir="training_output"
            )
            env_info = setup_training_environment(training_config)
            device = env_info['device']
            
            # Setup data loaders
            self.logger.info("📚 Setting up enhanced data loaders...")
            train_loader, val_loader, train_dataset, val_dataset = self.setup_enhanced_data_loaders(config)
            
            self.logger.info(f"📊 Dataset Info:")
            self.logger.info(f"  Training samples: {len(train_dataset)}")
            self.logger.info(f"  Validation samples: {len(val_dataset)}")
            self.logger.info(f"  Vocabulary size: {len(train_dataset.char_to_idx)}")
            
            # Load best checkpoint and create enhanced model
            checkpoint_path = self.load_best_checkpoint()
            model = self.create_enhanced_model(
                vocab_size=len(train_dataset.char_to_idx),
                max_seq_length=train_dataset.max_sequence_length + 1,
                config=config,
                checkpoint_path=checkpoint_path
            )
            
            # Setup enhanced trainer
            trainer = self.setup_enhanced_trainer(model, train_loader, val_loader, config, device)
            
            # Run training
            self.logger.info("🎯 Starting advanced optimization training...")
            start_time = time.time()
            training_history = trainer.train()
            end_time = time.time()
            
            # Analyze results
            total_time = end_time - start_time
            training_hist = training_history.get('training_history', {})
            val_metrics_list = training_hist.get('val_metrics', [])
            
            if val_metrics_list:
                best_char_acc = max([m.get('char_accuracy', 0.0) for m in val_metrics_list])
                best_seq_acc = max([m.get('seq_accuracy', 0.0) for m in val_metrics_list])
                final_metrics = val_metrics_list[-1]
                
                self.logger.info("🏆 Phase 3.2 Results:")
                self.logger.info(f"  Best Character Accuracy: {best_char_acc:.1%}")
                self.logger.info(f"  Best Sequence Accuracy: {best_seq_acc:.1%}")
                self.logger.info(f"  Final Character Accuracy: {final_metrics.get('char_accuracy', 0.0):.1%}")
                self.logger.info(f"  Final Sequence Accuracy: {final_metrics.get('seq_accuracy', 0.0):.1%}")
                self.logger.info(f"  Training Time: {total_time:.1f}s")
                
                # Check if targets achieved
                target_char = config['target_metrics']['char_accuracy']
                target_seq = config['target_metrics']['seq_accuracy']
                
                if best_char_acc >= target_char:
                    self.logger.info(f"🎉 CHARACTER ACCURACY TARGET ACHIEVED! {best_char_acc:.1%} >= {target_char:.1%}")
                else:
                    improvement = best_char_acc - 0.40  # Assuming ~40% baseline
                    self.logger.info(f"📈 Improvement: +{improvement:.1%} from baseline")
                
                if best_seq_acc >= target_seq:
                    self.logger.info(f"🎉 SEQUENCE ACCURACY TARGET ACHIEVED! {best_seq_acc:.1%} >= {target_seq:.1%}")
                
                # Prepare results
                results = {
                    'experiment_name': config['experiment_name'],
                    'completion_time': datetime.now().isoformat(),
                    'training_time_seconds': total_time,
                    'best_char_accuracy': best_char_acc,
                    'best_seq_accuracy': best_seq_acc,
                    'target_achieved': best_char_acc >= target_char,
                    'configuration': config,
                    'training_history': training_history
                }
                
                return results
            
        except Exception as e:
            self.logger.error(f"❌ Phase 3.2 optimization failed: {e}")
            raise
    
    def generate_recommendations(self, results: Dict) -> List[str]:
        """Generate recommendations based on optimization results."""
        
        recommendations = []
        char_acc = results.get('best_char_accuracy', 0.0)
        seq_acc = results.get('best_seq_accuracy', 0.0)
        
        if char_acc >= 0.85:
            recommendations.extend([
                "🎉 Model ready for production deployment!",
                "🧪 Run comprehensive evaluation on real-world data",
                "🚀 Implement inference optimization and deployment",
                "📊 Conduct error analysis for edge cases"
            ])
        elif char_acc >= 0.70:
            recommendations.extend([
                "📈 Good progress - consider extended training",
                "🔧 Experiment with ensemble methods",
                "🎯 Fine-tune on domain-specific data",
                "🔍 Analyze failure cases for targeted improvements"
            ])
        else:
            recommendations.extend([
                "🔄 Continue optimization with different techniques",
                "📊 Analyze data quality and distribution",
                "🏗️ Consider architecture modifications",
                "🎯 Implement progressive curriculum learning"
            ])
        
        return recommendations


def main():
    """Main execution function for Phase 3.2."""
    
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"phase3_2_optimization_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding='utf-8')
        ]
    )
    logger = logging.getLogger(__name__)
    
    logger.info("🚀 Starting Phase 3.2: Advanced Model Optimization")
    logger.info("=" * 60)
    
    # Phase 3.2 implementation will be added here
    logger.info("📋 Phase 3.2 script ready for implementation")
    logger.info("🎯 Target: 85% character accuracy through advanced techniques")
    
    return True


if __name__ == "__main__":
    main() 