# Progressive Training Strategy Examples

## Overview

This document provides practical examples for using the Progressive Training Strategy implementation. Each example includes complete code, expected output, and explanations.

## Table of Contents

1. [Basic Usage Examples](#basic-usage-examples)
2. [Testing and Validation](#testing-and-validation)
3. [Custom Configuration](#custom-configuration)
4. [Integration Examples](#integration-examples)
5. [Monitoring and Analysis](#monitoring-and-analysis)
6. [Error Handling](#error-handling)
7. [Advanced Use Cases](#advanced-use-cases)

## Basic Usage Examples

### Example 1: Simple Progressive Training

**Scenario**: Run the complete 5-stage progressive training with default settings.

```bash
# Command line usage
python src/sample_scripts/progressive_training_strategy.py \
    --config config/model_config.yaml \
    --output-dir basic_training_example
```

**Expected Output**:
```
🚀 Progressive Training Strategy (Phase 3.1)
   Device: cuda
   Output: basic_training_example

=============== Stage 1: Single Character ===============
🚀 Training Stage 1: Single Character
   Success threshold: 92.0%
   Epoch  1: Simulated Acc=0.450
   Epoch  2: Simulated Acc=0.550
   ...
   Epoch  8: Simulated Acc=0.924
✅ Stage 1: Single Character completed!

=============== Stage 2: Simple Combinations ===============
...

🎉 Progressive Training Strategy Completed!
Final Performance: 0.753
Total Epochs: 65
```

### Example 2: Python Script Integration

**Scenario**: Use the progressive training strategy within a Python script.

```python
#!/usr/bin/env python3
"""
Example: Basic Progressive Training Integration
"""

import os
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sample_scripts.progressive_training_strategy import ProgressiveTrainingStrategy

def main():
    # Initialize the strategy
    strategy = ProgressiveTrainingStrategy(
        config_path="config/model_config.yaml",
        output_dir="python_example_output"
    )
    
    print("🚀 Starting Progressive Training Example")
    
    try:
        # Run the complete training pipeline
        report = strategy.run_progressive_training()
        
        # Print results
        print(f"✅ Training completed successfully!")
        print(f"   Final Performance: {report['final_performance']:.3f}")
        print(f"   Total Epochs: {report['total_epochs']}")
        print(f"   Duration: {report['training_duration']}")
        
        # Print stage-by-stage results
        print("\n📊 Stage Results:")
        for stage_perf in report['stage_performances']:
            print(f"   Stage {stage_perf['stage_idx'] + 1}: "
                  f"{stage_perf['best_accuracy']:.3f} "
                  f"({stage_perf['epochs_trained']} epochs)")
        
        return report
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return None

if __name__ == "__main__":
    report = main()
    exit(0 if report else 1)
```

## Testing and Validation

### Example 3: Comprehensive Testing

**Scenario**: Run all available tests to verify the system is working correctly.

```bash
# Run the test suite
python src/sample_scripts/test_progressive_training.py
```

**Expected Output**:
```
🚀 Testing Progressive Training Strategy Implementation (Phase 3.1)
======================================================================
🔍 Testing Curriculum Stages...
  ✅ CurriculumStage creation works

🔍 Testing Progressive Training Strategy...
  ✅ Progressive training strategy works
     Created 5 stages

🔍 Testing Phase 3.1 Requirements...
  ✅ All requirements satisfied

======================================================================
🎯 Overall: 3/3 tests passed (100.0%)
🎉 All tests passed! Progressive Training Strategy (Phase 3.1) is ready!
```

### Example 4: Dry Run Validation

**Scenario**: Test configuration and system setup without actual training.

```bash
# Dry run test
python src/sample_scripts/progressive_training_strategy.py --dry-run
```

## Custom Configuration

### Example 5: Custom Output Directory

```bash
# Custom experiment directory
mkdir -p experiments/progressive_v1
python src/sample_scripts/progressive_training_strategy.py \
    --config config/model_config.yaml \
    --output-dir experiments/progressive_v1
```

### Example 6: Custom Configuration File

```yaml
# custom_config.yaml
model:
  name: "KhmerTextOCR"
  vocab_size: 115
  hidden_dim: 256        # Reduced from 512
  num_layers: 4          # Reduced from 6

training:
  batch_size: 16         # Reduced from 32
  learning_rate: 0.0005  # Reduced from 0.001
  weight_decay: 0.0001

data:
  image_height: 32
  image_width: 128
  max_sequence_length: 20
```

## Integration Examples

### Example 7: Curriculum Stage Analysis

```python
#!/usr/bin/env python3
"""
Example: Curriculum Stage Analysis
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sample_scripts.progressive_training_strategy import ProgressiveTrainingStrategy

def analyze_curriculum():
    # Initialize strategy
    strategy = ProgressiveTrainingStrategy("config/model_config.yaml")
    
    # Create curriculum manager
    curriculum_manager = strategy.create_curriculum_manager()
    
    print("📊 Curriculum Analysis")
    print("=" * 50)
    
    total_min_epochs = 0
    total_max_epochs = 0
    
    for i, stage in enumerate(curriculum_manager.stages):
        print(f"\n🎯 Stage {i+1}: {stage.name}")
        print(f"   Description: {stage.description}")
        print(f"   Difficulty Level: {stage.level.value}")
        print(f"   Success Threshold: {stage.success_threshold:.1%}")
        print(f"   Epoch Range: {stage.min_epochs}-{stage.max_epochs}")
        
        total_min_epochs += stage.min_epochs
        total_max_epochs += stage.max_epochs
    
    print(f"\n📈 Overall Statistics:")
    print(f"   Total Stages: {len(curriculum_manager.stages)}")
    print(f"   Minimum Total Epochs: {total_min_epochs}")
    print(f"   Maximum Total Epochs: {total_max_epochs}")
    print(f"   Estimated Duration: {total_min_epochs * 2}-{total_max_epochs * 2} minutes")

if __name__ == "__main__":
    analyze_curriculum()
```

## Error Handling

### Example 8: Robust Error Handling

```python
#!/usr/bin/env python3
"""
Example: Robust Error Handling
"""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sample_scripts.progressive_training_strategy import ProgressiveTrainingStrategy

def setup_logging():
    """Setup comprehensive logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('progressive_training_errors.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def robust_progressive_training(config_path, output_dir):
    """Run progressive training with comprehensive error handling."""
    
    logger = setup_logging()
    
    try:
        # Validate configuration file
        if not Path(config_path).exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        logger.info(f"Starting progressive training with config: {config_path}")
        
        # Initialize strategy
        strategy = ProgressiveTrainingStrategy(
            config_path=config_path,
            output_dir=output_dir
        )
        
        logger.info("Strategy initialized successfully")
        
        # Run training with error recovery
        report = strategy.run_progressive_training()
        
        logger.info(f"Training completed successfully. Final performance: {report['final_performance']:.3f}")
        
        return report
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        print(f"❌ Error: {e}")
        print("💡 Solution: Check file path and ensure it exists")
        return None
        
    except ImportError as e:
        logger.error(f"Import error: {e}")
        print(f"❌ Import Error: {e}")
        print("💡 Solution: Ensure you're running from project root directory")
        return None
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        print(f"❌ Unexpected Error: {e}")
        print("💡 Check the log file for detailed error information")
        return None

def main():
    """Main function with error handling examples."""
    
    print("🛡️ Progressive Training with Error Handling")
    
    # Test with valid configuration
    result = robust_progressive_training(
        config_path='config/model_config.yaml',
        output_dir='error_handling_test'
    )
    
    if result:
        print(f"✅ Training completed successfully")
    else:
        print(f"❌ Training failed")

if __name__ == "__main__":
    main()
```

## Advanced Use Cases

### Example 9: Custom Stage Implementation

```python
#!/usr/bin/env python3
"""
Example: Custom Stage Implementation
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sample_scripts.progressive_training_strategy import ProgressiveTrainingStrategy
from modules.trainers import CurriculumLearningManager, CurriculumStage, DifficultyLevel

class CustomProgressiveTraining(ProgressiveTrainingStrategy):
    """Custom progressive training with modified stages."""
    
    def create_curriculum_manager(self):
        """Create custom curriculum with modified stages."""
        
        # Define custom stages with different thresholds
        custom_stages = [
            CurriculumStage(
                level=DifficultyLevel.SINGLE_CHAR,
                name="Enhanced Stage 1: Character Mastery",
                description="Extended single character training with higher threshold",
                min_epochs=8,           # Increased from 5
                max_epochs=20,          # Increased from 15
                success_threshold=0.95  # Increased from 0.92
            ),
            CurriculumStage(
                level=DifficultyLevel.SIMPLE_COMBO,
                name="Enhanced Stage 2: Basic Combinations",
                description="Simple combinations with extended training",
                min_epochs=10,          # Increased from 8
                max_epochs=25,          # Increased from 20
                success_threshold=0.90  # Increased from 0.88
            ),
            # ... additional custom stages
        ]
        
        return CurriculumLearningManager(
            stages=custom_stages,
            auto_progression=True
        )
    
    def train_stage(self, stage, stage_idx):
        """Custom stage training with additional monitoring."""
        
        print(f"\n🎯 Custom Training for {stage.name}")
        print(f"   Enhanced Threshold: {stage.success_threshold:.1%}")
        print(f"   Extended Epoch Range: {stage.min_epochs}-{stage.max_epochs}")
        
        # Call parent implementation
        metrics = super().train_stage(stage, stage_idx)
        
        # Additional custom analysis
        threshold_met = metrics['best_val_accuracy'] >= stage.success_threshold
        performance_quality = "Excellent" if threshold_met else "Needs Improvement"
        
        print(f"   Performance Quality: {performance_quality}")
        
        return metrics

def main():
    """Run custom progressive training."""
    
    print("🎨 Custom Progressive Training Example")
    
    # Use custom implementation
    custom_strategy = CustomProgressiveTraining(
        config_path="config/model_config.yaml",
        output_dir="custom_progressive_output"
    )
    
    # Run custom training
    report = custom_strategy.run_progressive_training()
    
    print(f"\n🎉 Custom Training Completed!")
    print(f"   Final Performance: {report['final_performance']:.3f}")
    
    return report

if __name__ == "__main__":
    main()
```

## Summary

These examples demonstrate the flexibility and power of the Progressive Training Strategy implementation. Key takeaways:

1. **Simple Usage**: Basic command-line and Python integration
2. **Testing**: Comprehensive validation and dry-run capabilities  
3. **Customization**: Flexible configuration and custom implementations
4. **Monitoring**: Advanced progress tracking and analysis
5. **Error Handling**: Robust error recovery and troubleshooting
6. **Integration**: Seamless connection with other system components

Each example can be adapted for specific use cases and requirements.