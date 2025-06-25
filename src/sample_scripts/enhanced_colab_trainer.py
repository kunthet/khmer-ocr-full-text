"""
Enhanced Khmer OCR Hyperparameter Tuning for Google Colab
Includes Google Drive integration and resumable training capabilities.

Usage in Colab:
1. Mount Google Drive
2. Upload project files
3. Run this script with: exec(open('src/sample_scripts/enhanced_colab_trainer.py').read())
"""

import yaml
import json
import logging
import time
import shutil
import glob
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
import torch
from IPython.display import display, HTML, clear_output
from dataclasses import dataclass, asdict

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from modules.data_utils import KhmerDigitsDataset
    from models import create_model
    from modules.trainers import OCRTrainer
    from modules.trainers.utils import setup_training_environment, TrainingConfig
    from modules.data_utils.preprocessing import get_train_transforms, get_val_transforms
    from modules.data_utils.dataset import collate_fn
    from torch.utils.data import DataLoader
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure you've uploaded your project and set up paths correctly.")
    exit(1)

# Enhanced Training Config with Resume Support
@dataclass 
class EnhancedTrainingConfig(TrainingConfig):
    """Enhanced TrainingConfig with resume functionality."""
    resume_from_checkpoint: Optional[str] = None
    drive_output_dir: Optional[str] = None
    auto_resume: bool = True

class EnhancedOCRTrainer(OCRTrainer):
    """Enhanced OCR Trainer with resumable training support."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.resume_epoch = 0
        
        # Check for resume
        if hasattr(self.config, 'resume_from_checkpoint') and self.config.resume_from_checkpoint:
            self._load_resume_checkpoint()
    
    def _load_resume_checkpoint(self):
        """Load checkpoint for resuming training."""
        try:
            checkpoint_path = self.config.resume_from_checkpoint
            logger.info(f"🔄 Loading checkpoint: {checkpoint_path}")
            
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Load states
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if 'scheduler_state_dict' in checkpoint and self.scheduler:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            # Load training state
            self.resume_epoch = checkpoint['epoch']
            self.current_epoch = self.resume_epoch
            self.global_step = checkpoint.get('global_step', 0)
            
            if 'training_history' in checkpoint:
                self.training_history = checkpoint['training_history']
            
            self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            
            logger.info(f"✅ Resumed from epoch {self.resume_epoch}")
            
        except Exception as e:
            logger.error(f"❌ Resume failed: {e}")
            self.resume_epoch = 0
    
    def train(self) -> Dict[str, Any]:
        """Enhanced training loop with resume support."""
        start_epoch = self.resume_epoch + 1 if self.resume_epoch > 0 else 1
        
        if self.resume_epoch > 0:
            logger.info(f"🔄 Resuming from epoch {start_epoch}")
        
        logger.info(f"📈 Training epochs {start_epoch} to {self.config.num_epochs}")
        logger.info(f"📊 Training samples: {len(self.train_loader.dataset)}")
        logger.info(f"📊 Validation samples: {len(self.val_loader.dataset)}")
        
        training_start_time = time.time()
        
        for epoch in range(start_epoch, self.config.num_epochs + 1):
            self.current_epoch = epoch
            
            # Training and validation
            train_results = self._train_epoch()
            val_results = self._validate_epoch()
            
            # Update learning rate
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_results['loss'])
                else:
                    self.scheduler.step()
            
            # Log results
            self._log_epoch_results(train_results, val_results)
            
            # Save checkpoint with enhanced state
            is_best = val_results['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_results['loss']
            
            if epoch % self.config.save_every_n_epochs == 0 or is_best:
                self.checkpoint_manager.save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch,
                    metrics=val_results,
                    is_best=is_best,
                    extra_state={
                        'config': self.config.to_dict(),
                        'training_history': self.training_history,
                        'global_step': self.global_step,
                        'best_val_loss': self.best_val_loss
                    }
                )
            
            # Early stopping
            if self.early_stopping(val_results['loss'], epoch):
                logger.info(f"⚠️ Early stopping at epoch {epoch}")
                break
        
        # Training completed
        total_time = time.time() - training_start_time
        epochs_trained = self.current_epoch - self.resume_epoch
        
        logger.info(f"✅ Training completed in {total_time:.2f}s")
        logger.info(f"📈 Trained {epochs_trained} epochs")
        
        # Load best model
        best_checkpoint = self.checkpoint_manager.load_best_model()
        if best_checkpoint:
            self.model.load_state_dict(best_checkpoint['model_state_dict'])
            logger.info(f"🏆 Loaded best model from epoch {best_checkpoint['epoch']}")
        
        if self.writer:
            self.writer.close()
        
        return {
            'best_val_loss': self.best_val_loss,
            'total_epochs': self.current_epoch,
            'epochs_trained_this_session': epochs_trained,
            'resumed_from_epoch': self.resume_epoch,
            'total_time': total_time,
            'training_history': self.training_history
        }

class EnhancedColabHyperparameterTuner:
    """Enhanced Colab hyperparameter tuner with Google Drive and resumable training."""
    
    def __init__(self, 
                 config_file: str = "config/phase3_training_configs.yaml",
                 drive_models_path: str = None,
                 drive_results_path: str = None,
                 auto_resume: bool = True):
        
        self.config_file = config_file
        self.results = []
        self.best_result = None
        self.experiments_completed = 0
        self.auto_resume = auto_resume
        
        # Drive paths
        self.drive_models_path = drive_models_path
        self.drive_results_path = drive_results_path
        self.use_drive = drive_models_path is not None
        
        # Output directories
        self.output_dir = drive_models_path if self.use_drive else "training_output"
        self.results_dir = drive_results_path if self.use_drive else "."
        
        # Load configuration
        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Load existing results if resuming
        if auto_resume:
            self.load_existing_results()
    
    def find_resumable_experiments(self) -> List[Dict]:
        """Find experiments that can be resumed."""
        resumable = []
        if not self.use_drive:
            return resumable
            
        exp_dirs = glob.glob(f'{self.drive_models_path}/*')
        
        for exp_dir in exp_dirs:
            if os.path.isdir(exp_dir):
                exp_name = os.path.basename(exp_dir)
                checkpoint_dir = os.path.join(exp_dir, 'checkpoints')
                
                if os.path.exists(checkpoint_dir):
                    checkpoints = glob.glob(f'{checkpoint_dir}/checkpoint_epoch_*.pth')
                    if checkpoints:
                        latest_checkpoint = sorted(checkpoints)[-1]
                        epoch_num = int(latest_checkpoint.split('_epoch_')[1].split('.pth')[0])
                        
                        # Check config for total epochs
                        total_epochs = 50  # Default
                        config_file = os.path.join(exp_dir, 'config.yaml')
                        if os.path.exists(config_file):
                            try:
                                with open(config_file, 'r') as f:
                                    config = yaml.safe_load(f)
                                    total_epochs = config.get('num_epochs', 50)
                            except:
                                pass
                        
                        is_complete = epoch_num >= total_epochs
                        
                        resumable.append({
                            'experiment_name': exp_name,
                            'latest_epoch': epoch_num,
                            'total_epochs': total_epochs,
                            'is_complete': is_complete,
                            'checkpoint_path': latest_checkpoint,
                            'experiment_dir': exp_dir,
                            'progress': f"{epoch_num}/{total_epochs}"
                        })
        
        return resumable
    
    def display_resumable_table(self, experiments: List[Dict]):
        """Display resumable experiments in HTML table."""
        if not experiments:
            print("📂 No previous experiments found.")
            return
            
        html = """
        <div style="border: 2px solid #2196F3; padding: 15px; margin: 10px 0; border-radius: 10px; background: #f8f9fa;">
            <h3>🔄 Found Resumable Experiments</h3>
            <table style="width: 100%; border-collapse: collapse; border: 1px solid #ddd;">
                <thead>
                    <tr style="background: #2196F3; color: white;">
                        <th style="border: 1px solid #ddd; padding: 8px;">Experiment</th>
                        <th style="border: 1px solid #ddd; padding: 8px;">Progress</th>
                        <th style="border: 1px solid #ddd; padding: 8px;">Status</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        for exp in experiments:
            status = "✅ Complete" if exp['is_complete'] else "🔄 Resumable"
            bg_color = "#e8f5e8" if exp['is_complete'] else "#fff3cd"
            
            html += f"""
                <tr style="background: {bg_color};">
                    <td style="border: 1px solid #ddd; padding: 8px;">{exp['experiment_name']}</td>
                    <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">{exp['progress']}</td>
                    <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">{status}</td>
                </tr>
            """
        
        html += """
                </tbody>
            </table>
            <p style="margin-top: 10px; color: #666; font-style: italic;">
                💡 Incomplete experiments will automatically resume from their last checkpoint.
            </p>
        </div>
        """
        
        display(HTML(html))
    
    def load_existing_results(self):
        """Load results from previous runs and detect resumable experiments."""
        try:
            # Find resumable experiments
            resumable = self.find_resumable_experiments()
            
            for exp in resumable:
                result = {
                    'experiment_name': exp['experiment_name'],
                    'status': 'completed' if exp['is_complete'] else 'resumable',
                    'latest_epoch': exp['latest_epoch'],
                    'total_epochs': exp['total_epochs'],
                    'progress': exp['progress'],
                    'checkpoint_path': exp['checkpoint_path'],
                    'experiment_dir': exp['experiment_dir']
                }
                self.results.append(result)
                
            if resumable:
                logger.info(f"📂 Found {len(resumable)} existing experiments")
                self.display_resumable_table(resumable)
                
            # Load previous results files
            results_files = glob.glob(f'{self.results_dir}/colab_hyperparameter_results_*.json')
            if results_files:
                latest_results_file = sorted(results_files)[-1]
                with open(latest_results_file, 'r') as f:
                    previous_results = json.load(f)
                    
                if previous_results.get('best_result'):
                    self.best_result = previous_results['best_result']
                    
                logger.info(f"📊 Loaded previous results from {os.path.basename(latest_results_file)}")
                
        except Exception as e:
            logger.warning(f"⚠️ Could not load previous results: {e}")
    
    def create_training_config(self, experiment_config: Dict, resume_from: str = None) -> EnhancedTrainingConfig:
        """Create enhanced training configuration."""
        config = EnhancedTrainingConfig(
            experiment_name=experiment_config['experiment_name'],
            model_name=experiment_config['model']['name'],
            model_config_path=experiment_config['model']['config_path'],
            metadata_path=experiment_config['data']['metadata_path'],
            batch_size=experiment_config['training']['batch_size'],
            num_workers=2,  # Reduced for Colab
            pin_memory=True,  # Enable for GPU
            learning_rate=experiment_config['training']['learning_rate'],
            weight_decay=experiment_config['training']['weight_decay'],
            num_epochs=experiment_config['training']['num_epochs'],
            device="auto",
            mixed_precision=True,  # Enable for GPU speedup
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
            keep_n_checkpoints=3,
            use_tensorboard=True,
            output_dir=self.output_dir,
            resume_from_checkpoint=resume_from,
            drive_output_dir=self.drive_models_path,
            auto_resume=True
        )
        return config

    def check_experiment_status(self, experiment_name: str) -> Dict[str, Any]:
        """Check if experiment is completed or can be resumed."""
        for result in self.results:
            if result['experiment_name'] == experiment_name:
                if result.get('status') == 'completed':
                    return {'status': 'completed', 'result': result}
                elif result.get('status') == 'resumable':
                    return {'status': 'resumable', 'result': result}
        return {'status': 'new'}

    def display_progress(self, current: int, total: int, exp_name: str, status: str):
        """Display live progress."""
        progress_html = f"""
        <div style="border: 2px solid #4CAF50; padding: 10px; margin: 10px 0; border-radius: 5px;">
            <h3>🚀 Hyperparameter Tuning Progress</h3>
            <p><strong>Experiment:</strong> {current}/{total} - {exp_name}</p>
            <p><strong>Status:</strong> {status}</p>
            <div style="background-color: #f0f0f0; border-radius: 10px; padding: 3px;">
                <div style="background-color: #4CAF50; height: 20px; border-radius: 10px; width: {(current/total)*100}%;"></div>
            </div>
            <p>{(current/total)*100:.1f}% Complete</p>
        </div>
        """
        display(HTML(progress_html))

    def run_single_experiment(self, experiment_name: str, experiment_config: Dict) -> Dict:
        """Run a single hyperparameter experiment with resumable training."""
        logger.info(f"🚀 Starting experiment: {experiment_name}")
        
        start_time = time.time()
        
        try:
            # Check if can resume
            status_check = self.check_experiment_status(experiment_name)
            resume_from = None
            
            if status_check['status'] == 'completed':
                logger.info(f"✅ Experiment {experiment_name} already completed")
                return status_check['result']
            elif status_check['status'] == 'resumable':
                resume_from = status_check['result']['checkpoint_path']
                logger.info(f"🔄 Will resume {experiment_name} from {resume_from}")
            
            # Create training configuration
            training_config = self.create_training_config(experiment_config, resume_from)
            
            # Setup environment
            env_info = setup_training_environment(training_config)
            device = env_info['device']
            exp_dir = env_info['dirs']['experiment_dir']
            
            logger.info(f"🖥️ Using device: {device}")
            if self.use_drive:
                logger.info(f"💾 Saving to Drive: {exp_dir}")
            
            # Load datasets
            logger.info("📚 Loading datasets...")
            
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
            
            logger.info(f"📊 Training samples: {len(train_dataset)}")
            logger.info(f"📊 Validation samples: {len(val_dataset)}")
            
            # Create model
            logger.info("🏗️ Creating model...")
            model = create_model(
                preset=training_config.model_name,
                vocab_size=len(train_dataset.char_to_idx),
                max_sequence_length=train_dataset.max_sequence_length + 1
            )
            
            # Initialize enhanced trainer
            trainer = EnhancedOCRTrainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                config=training_config,
                device=device,
                char_to_idx=train_dataset.char_to_idx,
                idx_to_char=train_dataset.idx_to_char
            )
            
            # Run training
            logger.info("🎯 Starting training...")
            training_history = trainer.train()
            
            # Calculate metrics
            end_time = time.time()
            training_time = end_time - start_time
            
            # Extract metrics from training history
            training_hist = training_history.get('training_history', {})
            val_metrics_list = training_hist.get('val_metrics', [])
            
            # Calculate best metrics
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
                'resumed_from_epoch': training_history.get('resumed_from_epoch', 0),
                'total_epochs_trained': training_history.get('total_epochs', 0),
                'epochs_trained_this_session': training_history.get('epochs_trained_this_session', 0),
                'best_val_char_accuracy': best_char_acc,
                'best_val_seq_accuracy': best_seq_acc,
                'final_train_loss': final_train_loss,
                'device': str(device),
                'drive_path': exp_dir if self.use_drive else None,
                'hyperparameters': {
                    'model_name': experiment_config['model']['name'],
                    'batch_size': experiment_config['training']['batch_size'],
                    'learning_rate': experiment_config['training']['learning_rate'],
                    'weight_decay': experiment_config['training']['weight_decay'],
                    'loss_type': experiment_config['training']['loss_type'],
                    'scheduler_type': experiment_config['scheduler']['type']
                },
                'training_history': training_hist
            }
            
            logger.info(f"✅ Experiment {experiment_name} completed!")
            logger.info(f"🏆 Best character accuracy: {result['best_val_char_accuracy']:.4f}")
            logger.info(f"🏆 Best sequence accuracy: {result['best_val_seq_accuracy']:.4f}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Experiment {experiment_name} failed: {str(e)}")
            return {
                'experiment_name': experiment_name,
                'status': 'failed',
                'error': str(e),
                'training_time': time.time() - start_time
            }

    def display_results_table(self):
        """Display results in HTML table."""
        if not self.results:
            return
            
        completed_results = [r for r in self.results if r.get('status') == 'completed']
        completed_results.sort(key=lambda x: x.get('best_val_char_accuracy', 0), reverse=True)
        
        html = """
        <div style="margin: 20px 0;">
            <h3>📊 Current Results</h3>
            <table style="border-collapse: collapse; width: 100%; border: 1px solid #ddd;">
                <thead>
                    <tr style="background-color: #f2f2f2;">
                        <th style="border: 1px solid #ddd; padding: 8px;">Rank</th>
                        <th style="border: 1px solid #ddd; padding: 8px;">Experiment</th>
                        <th style="border: 1px solid #ddd; padding: 8px;">Char Acc</th>
                        <th style="border: 1px solid #ddd; padding: 8px;">Seq Acc</th>
                        <th style="border: 1px solid #ddd; padding: 8px;">Model</th>
                        <th style="border: 1px solid #ddd; padding: 8px;">Status</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        for i, result in enumerate(completed_results, 1):
            char_acc = result.get('best_val_char_accuracy', 0) * 100
            seq_acc = result.get('best_val_seq_accuracy', 0) * 100
            model_name = result.get('hyperparameters', {}).get('model_name', 'unknown')
            
            html += f"""
                <tr>
                    <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">{i}</td>
                    <td style="border: 1px solid #ddd; padding: 8px;">{result['experiment_name']}</td>
                    <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">{char_acc:.1f}%</td>
                    <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">{seq_acc:.1f}%</td>
                    <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">{model_name}</td>
                    <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">✅</td>
                </tr>
            """
        
        # Add failed experiments
        failed_results = [r for r in self.results if r.get('status') == 'failed']
        for result in failed_results:
            html += f"""
                <tr style="background-color: #ffe6e6;">
                    <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">-</td>
                    <td style="border: 1px solid #ddd; padding: 8px;">{result['experiment_name']}</td>
                    <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">FAILED</td>
                    <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">FAILED</td>
                    <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">-</td>
                    <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">❌</td>
                </tr>
            """
        
        html += """
                </tbody>
            </table>
        </div>
        """
        
        display(HTML(html))

    def run_experiments(self, experiment_names: List[str] = None):
        """Run experiments with live progress tracking."""
        experiments = self.config['experiments']
        
        if experiment_names:
            experiments = {name: config for name, config in experiments.items() 
                          if name in experiment_names}
        
        total_experiments = len(experiments)
        logger.info(f"🚀 Starting {total_experiments} experiments")
        
        for i, (exp_name, exp_config) in enumerate(experiments.items(), 1):
            # Update progress
            clear_output(wait=True)
            self.display_progress(i-1, total_experiments, exp_name, "Starting...")
            
            result = self.run_single_experiment(exp_name, exp_config)
            
            # Update results list (remove old entry if exists)
            self.results = [r for r in self.results if r['experiment_name'] != exp_name]
            self.results.append(result)
            
            # Update best result
            if (result.get('status') == 'completed' and 
                (self.best_result is None or 
                 result['best_val_char_accuracy'] > 
                 self.best_result['best_val_char_accuracy'])):
                self.best_result = result
            
            self.experiments_completed += 1
            
            # Update progress
            status = "✅ Completed" if result.get('status') == 'completed' else "❌ Failed"
            self.display_progress(i, total_experiments, exp_name, status)
            
            # Display results table
            self.display_results_table()

    def save_results(self):
        """Save results to Drive."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"colab_hyperparameter_results_{timestamp}.json"
        
        if self.use_drive:
            results_path = os.path.join(self.drive_results_path, results_file)
        else:
            results_path = results_file
        
        results_data = {
            'timestamp': timestamp,
            'platform': 'Google Colab',
            'gpu_info': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
            'drive_integration': self.use_drive,
            'drive_models_path': self.drive_models_path,
            'best_result': self.best_result,
            'all_results': self.results
        }
        
        with open(results_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        logger.info(f"💾 Results saved to {results_path}")
        return results_path

def setup_drive_integration():
    """Setup Google Drive integration - call this first in Colab."""
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        
        DRIVE_ROOT = '/content/drive/MyDrive'
        PROJECT_DRIVE_PATH = f'{DRIVE_ROOT}/Khmer_OCR_Experiments'
        MODELS_DRIVE_PATH = f'{PROJECT_DRIVE_PATH}/training_output'
        RESULTS_DRIVE_PATH = f'{PROJECT_DRIVE_PATH}/results'
        
        for path in [PROJECT_DRIVE_PATH, MODELS_DRIVE_PATH, RESULTS_DRIVE_PATH]:
            os.makedirs(path, exist_ok=True)
        
        print(f"✅ Drive mounted: {PROJECT_DRIVE_PATH}")
        print(f"🏗️ Models: {MODELS_DRIVE_PATH}")
        print(f"📊 Results: {RESULTS_DRIVE_PATH}")
        
        return MODELS_DRIVE_PATH, RESULTS_DRIVE_PATH
        
    except ImportError:
        print("❌ Not running in Colab - Drive integration disabled")
        return None, None

# Example usage in Colab:
if __name__ == "__main__":
    print("🚀 Enhanced Khmer OCR Hyperparameter Tuning")
    print("=" * 50)
    
    # Setup Drive (only works in Colab)
    try:
        models_path, results_path = setup_drive_integration()
        
        # Initialize enhanced tuner
        tuner = EnhancedColabHyperparameterTuner(
            drive_models_path=models_path,
            drive_results_path=results_path,
            auto_resume=True
        )
        
        print("\n🎯 Ready to run experiments!")
        print("Example usage:")
        print("tuner.run_experiments(['conservative_small'])")  # Run single experiment
        print("tuner.run_experiments()  # Run all experiments")
        
    except:
        print("⚠️ Initialize manually with:")
        print("tuner = EnhancedColabHyperparameterTuner()")