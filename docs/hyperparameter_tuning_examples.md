# Hyperparameter Tuning Examples

This document provides practical examples for using the Khmer OCR hyperparameter tuning system.

## Quick Start Examples

### 1. Run All Predefined Experiments

```bash
# Run all experiments with default configuration
python src/sample_scripts/phase3_hyperparameter_tuning.py
```

This will execute:
- `conservative_small`: Safe baseline experiment
- `baseline_optimized`: Balanced optimization  
- `high_learning_rate`: Aggressive training approach

### 2. Run Specific Experiments

```bash
# Run only the conservative experiment
python src/sample_scripts/phase3_hyperparameter_tuning.py --experiments conservative_small

# Run multiple specific experiments
python src/sample_scripts/phase3_hyperparameter_tuning.py --experiments conservative_small baseline_optimized
```

### 3. Use Custom Configuration

```bash
# Use custom configuration file
python src/sample_scripts/phase3_hyperparameter_tuning.py --config my_experiments.yaml
```

## Configuration Examples

### Example 1: Fast Prototyping Setup

```yaml
# config/fast_prototyping.yaml
experiments:
  quick_test:
    experiment_name: "quick_test"
    model:
      name: "small"
      config_path: "config/model_config.yaml"
    
    data:
      metadata_path: "generated_data/metadata.yaml"
      train_split: "train"
      val_split: "val"
      num_workers: 2
      pin_memory: false
      augmentation: true
    
    training:
      device: "auto"
      mixed_precision: false
      gradient_clip_norm: 1.0
      log_every_n_steps: 10
      save_every_n_epochs: 5
      keep_n_checkpoints: 2
      use_tensorboard: true
      batch_size: 32
      learning_rate: 0.002
      weight_decay: 0.0001
      num_epochs: 10        # Fast: only 10 epochs
      loss_type: "crossentropy"
      label_smoothing: 0.1
    
    optimizer:
      type: "adam"
      learning_rate: 0.002
      weight_decay: 0.0001
      betas: [0.9, 0.999]
    
    scheduler:
      type: "steplr"
      step_size: 5
      gamma: 0.5
    
    early_stopping:
      patience: 3           # Early stop after 3 epochs
      min_delta: 0.001
      monitor: "val_char_accuracy"
      mode: "max"

targets:
  character_accuracy: 0.50  # Lower target for quick testing
  sequence_accuracy: 0.30
  training_time_per_epoch: 200
  convergence_epochs: 10
```

### Example 2: Comprehensive Model Comparison

```yaml
# config/model_comparison.yaml
experiments:
  small_model_test:
    experiment_name: "small_model_test"
    model:
      name: "small"
      config_path: "config/model_config.yaml"
    training:
      batch_size: 64
      learning_rate: 0.002
      num_epochs: 30
      loss_type: "crossentropy"
      label_smoothing: 0.1
    optimizer:
      type: "adamw"
    scheduler:
      type: "cosine"
      warmup_epochs: 3
    early_stopping:
      patience: 8
      min_delta: 0.001
      monitor: "val_char_accuracy"
      mode: "max"
  
  medium_model_test:
    experiment_name: "medium_model_test"
    model:
      name: "medium"
      config_path: "config/model_config.yaml"
    training:
      batch_size: 48
      learning_rate: 0.0015
      num_epochs: 30
      loss_type: "crossentropy"
      label_smoothing: 0.1
    optimizer:
      type: "adamw"
    scheduler:
      type: "cosine"
      warmup_epochs: 3
    early_stopping:
      patience: 8
      min_delta: 0.001
      monitor: "val_char_accuracy"
      mode: "max"
  
  large_model_test:
    experiment_name: "large_model_test"
    model:
      name: "large"
      config_path: "config/model_config.yaml"
    training:
      batch_size: 24        # Smaller batch for larger model
      learning_rate: 0.001  # Lower LR for larger model
      num_epochs: 25
      loss_type: "crossentropy"
      label_smoothing: 0.15
    optimizer:
      type: "adamw"
    scheduler:
      type: "plateau"
      patience: 5
      factor: 0.5
    early_stopping:
      patience: 10
      min_delta: 0.001
      monitor: "val_char_accuracy"
      mode: "max"

targets:
  character_accuracy: 0.85
  sequence_accuracy: 0.70
  training_time_per_epoch: 400
  convergence_epochs: 20
```

### Example 3: Learning Rate Sweep

```yaml
# config/learning_rate_sweep.yaml
experiments:
  lr_0001:
    experiment_name: "lr_0001"
    model:
      name: "medium"
      config_path: "config/model_config.yaml"
    training:
      batch_size: 64
      learning_rate: 0.001
      num_epochs: 25
      loss_type: "crossentropy"
      label_smoothing: 0.1
    optimizer:
      type: "adamw"
      learning_rate: 0.001
    scheduler:
      type: "cosine"
      warmup_epochs: 2
    early_stopping:
      patience: 8
      monitor: "val_char_accuracy"
      mode: "max"
  
  lr_0002:
    experiment_name: "lr_0002"
    model:
      name: "medium"
      config_path: "config/model_config.yaml"
    training:
      batch_size: 64
      learning_rate: 0.002
      num_epochs: 25
      loss_type: "crossentropy"
      label_smoothing: 0.1
    optimizer:
      type: "adamw"
      learning_rate: 0.002
    scheduler:
      type: "cosine"
      warmup_epochs: 2
    early_stopping:
      patience: 8
      monitor: "val_char_accuracy"
      mode: "max"
  
  lr_0003:
    experiment_name: "lr_0003"
    model:
      name: "medium"
      config_path: "config/model_config.yaml"
    training:
      batch_size: 64
      learning_rate: 0.003
      num_epochs: 25
      loss_type: "crossentropy"
      label_smoothing: 0.1
    optimizer:
      type: "adamw"
      learning_rate: 0.003
    scheduler:
      type: "cosine"
      warmup_epochs: 2
    early_stopping:
      patience: 8
      monitor: "val_char_accuracy"
      mode: "max"
  
  lr_0005:
    experiment_name: "lr_0005"
    model:
      name: "medium"
      config_path: "config/model_config.yaml"
    training:
      batch_size: 64
      learning_rate: 0.005
      num_epochs: 25
      loss_type: "crossentropy"
      label_smoothing: 0.1
    optimizer:
      type: "adamw"
      learning_rate: 0.005
    scheduler:
      type: "cosine"
      warmup_epochs: 2
    early_stopping:
      patience: 6        # More aggressive for high LR
      monitor: "val_char_accuracy"
      mode: "max"

targets:
  character_accuracy: 0.85
  sequence_accuracy: 0.70
  training_time_per_epoch: 300
  convergence_epochs: 15
```

## Python Script Examples

### Example 1: Automated Experiment Runner

```python
#!/usr/bin/env python3
"""
Automated experiment runner with custom configurations.
"""

import yaml
import json
import time
from pathlib import Path
from src.sample_scripts.phase3_hyperparameter_tuning import HyperparameterTuner

def create_batch_size_experiments():
    """Create experiments testing different batch sizes."""
    
    experiments = {}
    batch_sizes = [16, 32, 48, 64, 96]
    
    for batch_size in batch_sizes:
        exp_name = f"batch_size_{batch_size}"
        experiments[exp_name] = {
            'experiment_name': exp_name,
            'model': {
                'name': 'medium',
                'config_path': 'config/model_config.yaml'
            },
            'training': {
                'batch_size': batch_size,
                'learning_rate': 0.002,
                'num_epochs': 20,
                'loss_type': 'crossentropy',
                'label_smoothing': 0.1
            },
            'optimizer': {
                'type': 'adamw'
            },
            'scheduler': {
                'type': 'cosine',
                'warmup_epochs': 2
            },
            'early_stopping': {
                'patience': 6,
                'monitor': 'val_char_accuracy',
                'mode': 'max'
            }
        }
    
    config = {
        'experiments': experiments,
        'targets': {
            'character_accuracy': 0.85,
            'sequence_accuracy': 0.70,
            'training_time_per_epoch': 300,
            'convergence_epochs': 15
        }
    }
    
    return config

def main():
    """Run batch size comparison experiments."""
    
    print("Creating batch size comparison experiments...")
    
    # Generate configuration
    config = create_batch_size_experiments()
    
    # Save configuration
    config_path = 'config/batch_size_experiments.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Configuration saved to: {config_path}")
    
    # Run experiments
    print("Starting experiments...")
    tuner = HyperparameterTuner(config_path)
    
    start_time = time.time()
    results = tuner.run_experiments()
    end_time = time.time()
    
    # Print summary
    total_time = end_time - start_time
    completed = len([r for r in results['all_results'] if r['status'] == 'completed'])
    
    print(f"\n=== EXPERIMENT SUMMARY ===")
    print(f"Total experiments: {len(results['all_results'])}")
    print(f"Completed: {completed}")
    print(f"Total time: {total_time:.1f} seconds")
    
    if results['best_result']:
        best = results['best_result']
        print(f"Best result: {best['experiment_name']}")
        print(f"Character accuracy: {best['best_val_char_accuracy']:.4f}")
        print(f"Batch size: {best['hyperparameters']['batch_size']}")

if __name__ == "__main__":
    main()
```

### Example 2: Results Analysis Script

```python
#!/usr/bin/env python3
"""
Analyze hyperparameter tuning results and generate insights.
"""

import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def load_all_results():
    """Load all hyperparameter tuning result files."""
    
    result_files = glob.glob("hyperparameter_tuning_results_*.json")
    all_experiments = []
    
    for file_path in result_files:
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        for experiment in data['all_results']:
            if experiment['status'] == 'completed':
                all_experiments.append(experiment)
    
    return all_experiments

def analyze_hyperparameter_impact(experiments):
    """Analyze the impact of different hyperparameters."""
    
    # Convert to DataFrame for easier analysis
    df_data = []
    for exp in experiments:
        row = {
            'experiment_name': exp['experiment_name'],
            'char_accuracy': exp['best_val_char_accuracy'],
            'seq_accuracy': exp['best_val_seq_accuracy'],
            'training_time': exp['training_time'],
            'total_epochs': exp['total_epochs']
        }
        
        # Add hyperparameters
        for key, value in exp['hyperparameters'].items():
            row[key] = value
            
        df_data.append(row)
    
    df = pd.DataFrame(df_data)
    
    print("=== HYPERPARAMETER IMPACT ANALYSIS ===")
    
    # Analyze learning rate impact
    if 'learning_rate' in df.columns:
        lr_analysis = df.groupby('learning_rate')['char_accuracy'].agg(['mean', 'std', 'count'])
        print("\nLearning Rate Impact:")
        print(lr_analysis)
    
    # Analyze batch size impact
    if 'batch_size' in df.columns:
        batch_analysis = df.groupby('batch_size')['char_accuracy'].agg(['mean', 'std', 'count'])
        print("\nBatch Size Impact:")
        print(batch_analysis)
    
    # Analyze model size impact
    if 'model_name' in df.columns:
        model_analysis = df.groupby('model_name')['char_accuracy'].agg(['mean', 'std', 'count'])
        print("\nModel Size Impact:")
        print(model_analysis)
    
    return df

def generate_visualizations(df):
    """Generate visualization plots."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Learning rate vs accuracy
    if 'learning_rate' in df.columns:
        lr_groups = df.groupby('learning_rate')['char_accuracy'].mean()
        axes[0, 0].bar(range(len(lr_groups)), lr_groups.values)
        axes[0, 0].set_title('Learning Rate vs Character Accuracy')
        axes[0, 0].set_xlabel('Learning Rate')
        axes[0, 0].set_ylabel('Character Accuracy')
        axes[0, 0].set_xticks(range(len(lr_groups)))
        axes[0, 0].set_xticklabels([f'{lr:.4f}' for lr in lr_groups.index])
    
    # Batch size vs accuracy
    if 'batch_size' in df.columns:
        batch_groups = df.groupby('batch_size')['char_accuracy'].mean()
        axes[0, 1].bar(range(len(batch_groups)), batch_groups.values)
        axes[0, 1].set_title('Batch Size vs Character Accuracy')
        axes[0, 1].set_xlabel('Batch Size')
        axes[0, 1].set_ylabel('Character Accuracy')
        axes[0, 1].set_xticks(range(len(batch_groups)))
        axes[0, 1].set_xticklabels(batch_groups.index)
    
    # Training time vs accuracy
    axes[1, 0].scatter(df['training_time'], df['char_accuracy'])
    axes[1, 0].set_title('Training Time vs Character Accuracy')
    axes[1, 0].set_xlabel('Training Time (seconds)')
    axes[1, 0].set_ylabel('Character Accuracy')
    
    # Accuracy distribution
    axes[1, 1].hist(df['char_accuracy'], bins=10, alpha=0.7)
    axes[1, 1].set_title('Character Accuracy Distribution')
    axes[1, 1].set_xlabel('Character Accuracy')
    axes[1, 1].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('hyperparameter_analysis.png', dpi=300, bbox_inches='tight')
    print("Visualization saved to: hyperparameter_analysis.png")

def find_optimal_configurations(df):
    """Find optimal hyperparameter configurations."""
    
    print("\n=== OPTIMAL CONFIGURATIONS ===")
    
    # Best overall performance
    best_overall = df.loc[df['char_accuracy'].idxmax()]
    print("\nBest Overall Performance:")
    print(f"Experiment: {best_overall['experiment_name']}")
    print(f"Character Accuracy: {best_overall['char_accuracy']:.4f}")
    print(f"Training Time: {best_overall['training_time']:.1f}s")
    
    # Best efficiency (accuracy per time)
    df['efficiency'] = df['char_accuracy'] / (df['training_time'] / 3600)  # accuracy per hour
    best_efficiency = df.loc[df['efficiency'].idxmax()]
    print("\nBest Efficiency (Accuracy per Hour):")
    print(f"Experiment: {best_efficiency['experiment_name']}")
    print(f"Character Accuracy: {best_efficiency['char_accuracy']:.4f}")
    print(f"Training Time: {best_efficiency['training_time']:.1f}s")
    print(f"Efficiency: {best_efficiency['efficiency']:.4f} acc/hour")
    
    # Fast convergence
    fast_convergence = df.loc[df['total_epochs'].idxmin()]
    print("\nFastest Convergence:")
    print(f"Experiment: {fast_convergence['experiment_name']}")
    print(f"Character Accuracy: {fast_convergence['char_accuracy']:.4f}")
    print(f"Epochs: {fast_convergence['total_epochs']}")

def main():
    """Main analysis function."""
    
    print("Loading hyperparameter tuning results...")
    experiments = load_all_results()
    
    if not experiments:
        print("No completed experiments found!")
        return
    
    print(f"Found {len(experiments)} completed experiments")
    
    # Analyze hyperparameter impact
    df = analyze_hyperparameter_impact(experiments)
    
    # Generate visualizations
    generate_visualizations(df)
    
    # Find optimal configurations
    find_optimal_configurations(df)
    
    # Save analysis results
    df.to_csv('hyperparameter_analysis.csv', index=False)
    print("\nAnalysis results saved to: hyperparameter_analysis.csv")

if __name__ == "__main__":
    main()
```

### Example 3: Progressive Hyperparameter Search

```python
#!/usr/bin/env python3
"""
Progressive hyperparameter search starting from best known configuration.
"""

import yaml
import json
import numpy as np
from pathlib import Path
from src.sample_scripts.phase3_hyperparameter_tuning import HyperparameterTuner

class ProgressiveHyperparameterSearch:
    """Progressive search starting from best known configuration."""
    
    def __init__(self, base_config_path="config/phase3_simple_configs.yaml"):
        self.base_config_path = base_config_path
        self.best_config = None
        self.best_score = 0.0
        
    def load_best_configuration(self):
        """Load best configuration from previous results."""
        
        result_files = list(Path('.').glob('hyperparameter_tuning_results_*.json'))
        
        if not result_files:
            print("No previous results found, using default configuration")
            return None
            
        # Load most recent results
        latest_file = max(result_files, key=lambda f: f.stat().st_mtime)
        
        with open(latest_file) as f:
            results = json.load(f)
        
        if results['best_result']:
            self.best_config = results['best_result']['hyperparameters']
            self.best_score = results['best_result']['best_val_char_accuracy']
            print(f"Loaded best configuration: {self.best_score:.4f} accuracy")
            return self.best_config
        
        return None
    
    def generate_search_space(self, center_config, search_radius=0.2):
        """Generate search space around best configuration."""
        
        if not center_config:
            # Use default search space
            return {
                'learning_rate': [0.001, 0.002, 0.003],
                'batch_size': [32, 48, 64],
                'weight_decay': [0.0001, 0.0005, 0.001]
            }
        
        # Generate variations around best configuration
        search_space = {}
        
        # Learning rate variations
        base_lr = center_config.get('learning_rate', 0.002)
        search_space['learning_rate'] = [
            base_lr * (1 - search_radius),
            base_lr,
            base_lr * (1 + search_radius)
        ]
        
        # Batch size variations
        base_batch = center_config.get('batch_size', 64)
        batch_variations = [
            max(16, int(base_batch * (1 - search_radius))),
            base_batch,
            min(128, int(base_batch * (1 + search_radius)))
        ]
        search_space['batch_size'] = sorted(set(batch_variations))
        
        # Weight decay variations
        base_wd = center_config.get('weight_decay', 0.0001)
        search_space['weight_decay'] = [
            base_wd * 0.5,
            base_wd,
            base_wd * 2.0
        ]
        
        return search_space
    
    def create_progressive_experiments(self, search_space):
        """Create experiments from search space."""
        
        experiments = {}
        exp_count = 0
        
        for lr in search_space['learning_rate']:
            for batch_size in search_space['batch_size']:
                for wd in search_space['weight_decay']:
                    exp_name = f"progressive_{exp_count:03d}"
                    
                    experiments[exp_name] = {
                        'experiment_name': exp_name,
                        'model': {
                            'name': 'medium',
                            'config_path': 'config/model_config.yaml'
                        },
                        'training': {
                            'batch_size': int(batch_size),
                            'learning_rate': float(lr),
                            'weight_decay': float(wd),
                            'num_epochs': 25,
                            'loss_type': 'crossentropy',
                            'label_smoothing': 0.1
                        },
                        'optimizer': {
                            'type': 'adamw',
                            'learning_rate': float(lr),
                            'weight_decay': float(wd)
                        },
                        'scheduler': {
                            'type': 'cosine',
                            'warmup_epochs': 2
                        },
                        'early_stopping': {
                            'patience': 6,
                            'monitor': 'val_char_accuracy',
                            'mode': 'max'
                        }
                    }
                    
                    exp_count += 1
        
        return experiments
    
    def run_progressive_search(self):
        """Run progressive hyperparameter search."""
        
        print("=== PROGRESSIVE HYPERPARAMETER SEARCH ===")
        
        # Load best known configuration
        best_config = self.load_best_configuration()
        
        # Generate search space
        search_space = self.generate_search_space(best_config)
        print(f"Search space: {search_space}")
        
        # Create experiments
        experiments = self.create_progressive_experiments(search_space)
        print(f"Generated {len(experiments)} experiments")
        
        # Create configuration file
        config = {
            'experiments': experiments,
            'targets': {
                'character_accuracy': 0.90,  # Higher target for progressive search
                'sequence_accuracy': 0.75,
                'training_time_per_epoch': 300,
                'convergence_epochs': 20
            }
        }
        
        config_path = 'config/progressive_search.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"Configuration saved to: {config_path}")
        
        # Run experiments
        tuner = HyperparameterTuner(config_path)
        results = tuner.run_experiments()
        
        # Compare with previous best
        if results['best_result']:
            new_best = results['best_result']['best_val_char_accuracy']
            print(f"\n=== PROGRESSIVE SEARCH RESULTS ===")
            print(f"Previous best: {self.best_score:.4f}")
            print(f"New best: {new_best:.4f}")
            print(f"Improvement: {new_best - self.best_score:.4f}")
            
            if new_best > self.best_score:
                print("🎉 Found better configuration!")
            else:
                print("No improvement found, consider expanding search space")

def main():
    """Run progressive hyperparameter search."""
    
    searcher = ProgressiveHyperparameterSearch()
    searcher.run_progressive_search()

if __name__ == "__main__":
    main()
```

## Usage Scenarios

### Scenario 1: Quick Model Validation

**Goal**: Quickly validate that the model can learn from the data

```bash
# Create fast validation configuration
cat > config/quick_validation.yaml << EOF
experiments:
  quick_check:
    experiment_name: "quick_check"
    model:
      name: "small"
    training:
      batch_size: 32
      learning_rate: 0.003
      num_epochs: 5
      loss_type: "crossentropy"
    optimizer:
      type: "adam"
    scheduler:
      type: "steplr"
      step_size: 2
    early_stopping:
      patience: 3
      monitor: "val_char_accuracy"
      mode: "max"

targets:
  character_accuracy: 0.30
  sequence_accuracy: 0.10
EOF

# Run quick validation
python src/sample_scripts/phase3_hyperparameter_tuning.py --config config/quick_validation.yaml
```

### Scenario 2: Production Optimization

**Goal**: Find best configuration for production deployment

```bash
# Run comprehensive optimization
python src/sample_scripts/phase3_hyperparameter_tuning.py --config config/model_comparison.yaml

# Analyze results
python analyze_results.py

# Run progressive search based on best results
python progressive_search.py
```

### Scenario 3: Resource-Constrained Environment

**Goal**: Optimize for limited computational resources

```yaml
# config/resource_constrained.yaml
experiments:
  efficient_small:
    experiment_name: "efficient_small"
    model:
      name: "small"
    training:
      batch_size: 16     # Small batch for memory
      learning_rate: 0.002
      num_epochs: 20     # Limited epochs
      loss_type: "crossentropy"
    optimizer:
      type: "adam"       # Faster than adamw
    scheduler:
      type: "steplr"     # Simple scheduler
      step_size: 8
    early_stopping:
      patience: 5        # Early stopping to save time
      monitor: "val_char_accuracy"
      mode: "max"

targets:
  character_accuracy: 0.70  # Lower target for constraints
  sequence_accuracy: 0.50
  training_time_per_epoch: 150
  convergence_epochs: 15
```

### Scenario 4: Debugging Poor Performance

**Goal**: Diagnose why model is not learning well

```yaml
# config/debugging.yaml
experiments:
  debug_high_lr:
    experiment_name: "debug_high_lr"
    model:
      name: "small"
    training:
      batch_size: 32
      learning_rate: 0.01    # Very high LR to test instability
      num_epochs: 15
      loss_type: "crossentropy"
      label_smoothing: 0.0   # No smoothing for debugging
    optimizer:
      type: "adam"
    scheduler:
      type: "steplr"
      step_size: 5
    early_stopping:
      patience: 10           # Patient for debugging
      monitor: "val_char_accuracy"
      mode: "max"
  
  debug_low_lr:
    experiment_name: "debug_low_lr"
    model:
      name: "small"
    training:
      batch_size: 32
      learning_rate: 0.0001  # Very low LR to test slow learning
      num_epochs: 15
      loss_type: "crossentropy"
      label_smoothing: 0.0
    optimizer:
      type: "adam"
    scheduler:
      type: "plateau"        # Plateau scheduler for low LR
      patience: 3
    early_stopping:
      patience: 10
      monitor: "val_char_accuracy"
      mode: "max"

targets:
  character_accuracy: 0.40
  sequence_accuracy: 0.20
  training_time_per_epoch: 200
  convergence_epochs: 10
```

## Monitoring and Analysis

### Real-time Monitoring

```bash
# Monitor experiment progress
watch -n 30 'ls -la training_output/*/logs/ | tail -10'

# Watch latest log file
tail -f training_output/$(ls -t training_output/ | head -1)/logs/training_*.log

# Monitor TensorBoard
tensorboard --logdir training_output/ --port 6006
```

### Post-experiment Analysis

```bash
# Generate analysis report
python analyze_results.py > analysis_report.txt

# Create visualizations
python visualize_results.py

# Compare with baseline
python compare_with_baseline.py --baseline 0.24 --target 0.85
```

---

*These examples provide comprehensive coverage of common use cases for the hyperparameter tuning system. Adapt the configurations based on your specific requirements and computational constraints.* 