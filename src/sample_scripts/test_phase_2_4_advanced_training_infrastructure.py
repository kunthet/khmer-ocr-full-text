"""
Test Script for Phase 2.4: Advanced Training Infrastructure

This script comprehensively tests all components of the advanced training infrastructure
including curriculum learning, multi-task learning, enhanced loss functions,
advanced schedulers, and training utilities.
"""

import sys
import os
import warnings
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, List
import numpy as np
import tempfile
import shutil


def test_curriculum_learning():
    """Test curriculum learning manager."""
    print("Testing Curriculum Learning...")
    
    try:
        from src.modules.trainers import (
            CurriculumLearningManager,
            CurriculumStage,
            DifficultyLevel
        )
        
        # Test default curriculum stages
        curriculum = CurriculumLearningManager()
        print(f"  ✓ Default curriculum created with {len(curriculum.stages)} stages")
        
        # Test curriculum progression
        assert curriculum.current_stage.level == DifficultyLevel.SINGLE_CHAR
        print(f"  ✓ Started at correct stage: {curriculum.current_stage.name}")
        
        # Simulate training epochs
        performance_progression = [0.5, 0.7, 0.85, 0.9, 0.92]
        for i, performance in enumerate(performance_progression):
            update_info = curriculum.update_epoch(performance)
            print(f"    Epoch {i+1}: Performance={performance:.2f}, Stage={update_info['current_stage_name']}")
        
        # Test stage progression
        assert curriculum.current_stage_idx > 0, "Curriculum should have progressed"
        print(f"  ✓ Curriculum progressed to stage {curriculum.current_stage_idx + 1}")
        
        # Test data loader configuration
        config = curriculum.get_data_loader_config()
        assert 'sample_filters' in config
        assert 'augmentation_strength' in config
        print(f"  ✓ Data loader config: {config}")
        
        # Test training adjustments
        adjustments = curriculum.get_training_config_adjustments()
        assert 'learning_rate_multiplier' in adjustments
        print(f"  ✓ Training adjustments: {adjustments}")
        
        # Test summary
        summary = curriculum.get_curriculum_summary()
        assert summary['total_stages'] == 5
        assert len(summary['stage_history']) > 0
        print(f"  ✓ Curriculum summary generated successfully")
        
        print("  ✅ Curriculum learning test passed!")
        return True
        
    except Exception as e:
        print(f"  ❌ Curriculum learning test failed: {e}")
        return False


def test_multi_task_learning():
    """Test multi-task learning components."""
    print("Testing Multi-Task Learning...")
    
    try:
        from src.modules.trainers import (
            MultiTaskTrainer,
            MultiTaskLoss,
            TaskConfig,
            TaskType,
            FocalLoss
        )
        
        # Create task configurations
        task_configs = [
            TaskConfig(
                task_type=TaskType.CHARACTER_RECOGNITION,
                name="char_recognition",
                weight=1.0,
                loss_type="crossentropy",
                metrics=["accuracy"],
                output_size=115
            ),
            TaskConfig(
                task_type=TaskType.CONFIDENCE_PREDICTION,
                name="confidence",
                weight=0.3,
                loss_type="mse",
                metrics=["mae"],
                output_size=1
            ),
            TaskConfig(
                task_type=TaskType.HIERARCHICAL_CLASSIFICATION,
                name="base_chars",
                weight=0.5,
                loss_type="focal",
                metrics=["accuracy", "f1_score"],
                output_size=35
            )
        ]
        
        print(f"  ✓ Created {len(task_configs)} task configurations")
        
        # Test multi-task loss
        multi_loss = MultiTaskLoss(task_configs)
        print(f"  ✓ Multi-task loss created with {len(multi_loss.task_losses)} loss functions")
        
        # Test loss calculation with dummy data
        batch_size, seq_len = 4, 8
        predictions = {
            'char_recognition': torch.randn(batch_size * seq_len, 115),
            'confidence': torch.randn(batch_size, seq_len, 1),
            'base_chars': torch.randn(batch_size * seq_len, 35)
        }
        targets = {
            'char_recognition': torch.randint(0, 115, (batch_size * seq_len,)),
            'confidence': torch.randn(batch_size, seq_len, 1),
            'base_chars': torch.randint(0, 35, (batch_size * seq_len,))
        }
        
        loss_results = multi_loss(predictions, targets)
        assert 'total_loss' in loss_results
        assert len(loss_results) == len(task_configs) + 1  # +1 for total_loss
        print(f"  ✓ Multi-task loss calculation successful")
        
        # Test task weights
        weights = multi_loss.get_task_weights()
        assert len(weights) == len(task_configs)
        print(f"  ✓ Task weights: {weights}")
        
        # Test focal loss
        focal_loss = FocalLoss(alpha=1.0, gamma=2.0)
        dummy_pred = torch.randn(10, 5)
        dummy_target = torch.randint(0, 5, (10,))
        focal_result = focal_loss(dummy_pred, dummy_target)
        assert focal_result.dim() == 0  # Scalar loss
        print(f"  ✓ Focal loss working correctly")
        
        # Test multi-task trainer (without actual training)
        class DummyTrainer:
            pass
        
        trainer = MultiTaskTrainer(task_configs, DummyTrainer())
        print(f"  ✓ Multi-task trainer initialized")
        
        # Test metrics calculation - reshape predictions for metrics
        metrics_predictions = {
            'char_recognition': predictions['char_recognition'].view(batch_size, seq_len, 115),
            'confidence': predictions['confidence'],
            'base_chars': predictions['base_chars'].view(batch_size, seq_len, 35)
        }
        metrics_targets = {
            'char_recognition': targets['char_recognition'].view(batch_size, seq_len),
            'confidence': targets['confidence'],
            'base_chars': targets['base_chars'].view(batch_size, seq_len)
        }
        metrics = trainer.calculate_task_metrics(metrics_predictions, metrics_targets)
        print(f"  ✓ Task metrics calculated: {list(metrics.keys())}")
        
        # Test summary
        summary = trainer.get_multi_task_summary()
        assert summary['total_tasks'] == 3
        assert len(summary['enabled_tasks']) == 3
        print(f"  ✓ Multi-task summary: {summary}")
        
        print("  ✅ Multi-task learning test passed!")
        return True
        
    except Exception as e:
        print(f"  ❌ Multi-task learning test failed: {e}")
        return False


def test_enhanced_losses():
    """Test enhanced loss functions."""
    print("Testing Enhanced Loss Functions...")
    
    try:
        from src.modules.trainers import (
            HierarchicalLoss,
            ConfidenceAwareLoss,
            CurriculumLoss,
            OnlineHardExampleMining,
            DistillationLoss
        )
        
        batch_size, seq_len, num_classes = 4, 8, 115
        
        # Test Hierarchical Loss
        hierarchical_loss = HierarchicalLoss()
        hierarchical_preds = {
            'base_chars': torch.randn(batch_size * seq_len, 35),
            'modifiers': torch.randn(batch_size * seq_len, 22),
            'combinations': torch.randn(batch_size, seq_len, 10)
        }
        hierarchical_targets = {
            'base_chars': torch.randint(0, 35, (batch_size * seq_len,)),
            'modifiers': torch.randint(0, 22, (batch_size * seq_len,)),
            'combinations': torch.randn(batch_size, seq_len, 10)
        }
        
        hier_results = hierarchical_loss(hierarchical_preds, hierarchical_targets)
        assert 'total_loss' in hier_results
        assert 'base_loss' in hier_results
        print(f"  ✓ Hierarchical loss: {list(hier_results.keys())}")
        
        # Test Confidence-Aware Loss
        confidence_loss = ConfidenceAwareLoss()
        predictions = torch.randn(batch_size, seq_len, num_classes)
        targets = torch.randint(0, num_classes, (batch_size, seq_len))
        confidence_scores = torch.rand(batch_size, seq_len)
        
        conf_results = confidence_loss(predictions, targets, confidence_scores)
        assert 'total_loss' in conf_results
        assert 'confidence_loss' in conf_results
        print(f"  ✓ Confidence-aware loss: {list(conf_results.keys())}")
        
        # Test Curriculum Loss
        curriculum_loss = CurriculumLoss()
        difficulty_scores = torch.randn(batch_size)
        
        curr_results = curriculum_loss(predictions, targets, difficulty_scores, curriculum_stage=2)
        assert 'total_loss' in curr_results
        assert 'stage_weight' in curr_results
        print(f"  ✓ Curriculum loss: {list(curr_results.keys())}")
        
        # Test Online Hard Example Mining
        ohem_loss = OnlineHardExampleMining(keep_ratio=0.7)
        ohem_results = ohem_loss(predictions, targets)
        assert 'total_loss' in ohem_results
        assert 'kept_ratio' in ohem_results
        print(f"  ✓ OHEM loss: {list(ohem_results.keys())}")
        
        # Test Distillation Loss
        distill_loss = DistillationLoss(temperature=4.0, alpha=0.3)
        student_logits = torch.randn(batch_size, num_classes)
        teacher_logits = torch.randn(batch_size, num_classes)
        targets_1d = torch.randint(0, num_classes, (batch_size,))
        
        distill_results = distill_loss(student_logits, teacher_logits, targets_1d)
        assert 'total_loss' in distill_results
        assert 'distillation_loss' in distill_results
        print(f"  ✓ Distillation loss: {list(distill_results.keys())}")
        
        print("  ✅ Enhanced loss functions test passed!")
        return True
        
    except Exception as e:
        print(f"  ❌ Enhanced loss functions test failed: {e}")
        return False


def test_advanced_schedulers():
    """Test advanced learning rate schedulers."""
    print("Testing Advanced Schedulers...")
    
    try:
        from src.modules.trainers import (
            WarmupCosineAnnealingLR,
            CurriculumAwareLR,
            AdaptiveLR,
            GradualWarmupScheduler,
            SchedulerFactory
        )
        
        # Create dummy model and optimizer
        model = nn.Linear(10, 5)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        
        # Test Warmup Cosine Annealing
        warmup_scheduler = WarmupCosineAnnealingLR(
            optimizer, 
            warmup_epochs=5, 
            max_epochs=50, 
            min_lr_ratio=0.01
        )
        
        initial_lr = warmup_scheduler.get_lr()[0]
        for epoch in range(10):
            warmup_scheduler.step()
        
        print(f"  ✓ Warmup cosine scheduler: initial_lr={initial_lr:.6f}, epoch_10_lr={warmup_scheduler.get_lr()[0]:.6f}")
        
        # Test Curriculum-Aware Scheduler
        curriculum_scheduler = CurriculumAwareLR(
            optimizer,
            stage_lr_multipliers=[1.0, 0.8, 0.6, 0.4, 0.2],
            stage_transitions=[10, 25, 45, 70, 100]
        )
        
        curriculum_scheduler.step(epoch=30)  # Should be in stage 2
        stage_2_lr = curriculum_scheduler.get_lr()[0]
        print(f"  ✓ Curriculum scheduler at epoch 30: lr={stage_2_lr:.6f}")
        
        # Test Adaptive Scheduler
        adaptive_scheduler = AdaptiveLR(
            optimizer,
            metric_name='val_loss',
            patience=3,
            factor=0.5
        )
        
        # Simulate metrics with plateauing
        metrics_sequence = [
            {'val_loss': 1.0},
            {'val_loss': 0.9},
            {'val_loss': 0.85},
            {'val_loss': 0.84},  # Start plateauing
            {'val_loss': 0.84},
            {'val_loss': 0.84},
            {'val_loss': 0.84}   # Should trigger LR reduction
        ]
        
        for metrics in metrics_sequence:
            adaptive_scheduler.step(metrics)
        
        print(f"  ✓ Adaptive scheduler after plateau: lr={optimizer.param_groups[0]['lr']:.6f}")
        
        # Test Scheduler Factory
        scheduler_configs = [
            {
                'type': 'cosine_warmup',
                'warmup_epochs': 5,
                'max_epochs': 100,
                'min_lr_ratio': 0.01
            },
            {
                'type': 'curriculum',
                'stage_multipliers': [1.0, 0.8, 0.6],
                'stage_transitions': [20, 40, 80]
            },
            {
                'type': 'adaptive',
                'metric_name': 'val_accuracy',
                'mode': 'max',
                'patience': 5
            }
        ]
        
        for i, config in enumerate(scheduler_configs):
            scheduler = SchedulerFactory.create_scheduler(optimizer, config)
            print(f"  ✓ Factory created {config['type']} scheduler")
        
        print("  ✅ Advanced schedulers test passed!")
        return True
        
    except Exception as e:
        print(f"  ❌ Advanced schedulers test failed: {e}")
        return False


def test_training_utilities():
    """Test advanced training utilities."""
    print("Testing Advanced Training Utilities...")
    
    try:
        from src.modules.trainers import (
            GradientAccumulator,
            MixedPrecisionManager,
            EnhancedCheckpointManager
        )
        
        # Test Gradient Accumulator
        accumulator = GradientAccumulator(
            accumulation_steps=4,
            max_grad_norm=1.0
        )
        
        model = nn.Linear(10, 5)
        optimizer = optim.Adam(model.parameters())
        
        # Simulate gradient accumulation
        for step in range(8):
            dummy_loss = torch.tensor(0.5, requires_grad=True)
            scaled_loss = accumulator.accumulate_loss(dummy_loss)
            
            # Simulate forward and backward pass
            dummy_input = torch.randn(5, 10)
            output = model(dummy_input)
            loss = output.mean() + scaled_loss
            loss.backward()
            
            step_info = accumulator.step_optimizer(optimizer, model)
            
            if step_info['avg_accumulated_loss'] > 0:
                print(f"    Step {step}: Updated optimizer, avg_loss={step_info['avg_accumulated_loss']:.3f}")
        
        grad_stats = accumulator.get_gradient_stats()
        print(f"  ✓ Gradient accumulator: {grad_stats}")
        
        # Test Mixed Precision Manager
        if torch.cuda.is_available():
            mp_manager = MixedPrecisionManager(enabled=True)
            print(f"  ✓ Mixed precision enabled: {mp_manager.enabled}")
            
            scale_info = mp_manager.get_scale_info()
            print(f"  ✓ Scale info: {scale_info}")
        else:
            mp_manager = MixedPrecisionManager(enabled=False)
            print(f"  ✓ Mixed precision disabled (CUDA not available)")
        
        # Test Enhanced Checkpoint Manager
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_manager = EnhancedCheckpointManager(
                checkpoint_dir=temp_dir,
                keep_n_checkpoints=3,
                save_optimizer_state=True,
                save_scheduler_state=True
            )
            
            # Save a checkpoint
            checkpoint_path = checkpoint_manager.save_checkpoint(
                model=model,
                epoch=10,
                optimizer=optimizer,
                metrics={'val_accuracy': 0.85, 'val_loss': 0.3},
                is_best=True,
                tag="test"
            )
            
            print(f"  ✓ Checkpoint saved: {Path(checkpoint_path).name}")
            
            # Load checkpoint
            new_model = nn.Linear(10, 5)
            new_optimizer = optim.Adam(new_model.parameters())
            
            checkpoint_info = checkpoint_manager.load_checkpoint(
                checkpoint_path, new_model, new_optimizer
            )
            
            print(f"  ✓ Checkpoint loaded: epoch={checkpoint_info['epoch']}")
            
            # Test checkpoint summary
            summary = checkpoint_manager.get_checkpoint_summary()
            assert summary['total_checkpoints'] == 1
            assert summary['latest_epoch'] == 10
            print(f"  ✓ Checkpoint summary: {summary}")
        
        print("  ✅ Training utilities test passed!")
        return True
        
    except Exception as e:
        print(f"  ❌ Training utilities test failed: {e}")
        return False


def test_integration():
    """Test integration of all components."""
    print("Testing Component Integration...")
    
    try:
        from src.modules.trainers import (
            CurriculumLearningManager,
            MultiTaskTrainer,
            TaskConfig,
            TaskType,
            HierarchicalLoss,
            WarmupCosineAnnealingLR,
            GradientAccumulator,
            MixedPrecisionManager
        )
        
        # Create a comprehensive training setup
        task_configs = [
            TaskConfig(
                task_type=TaskType.CHARACTER_RECOGNITION,
                name="char_recognition",
                weight=1.0,
                metrics=["accuracy"]
            )
        ]
        
        # Initialize all components
        curriculum = CurriculumLearningManager()
        
        model = nn.Linear(100, 115)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        
        scheduler = WarmupCosineAnnealingLR(
            optimizer, 
            warmup_epochs=5, 
            max_epochs=50
        )
        
        accumulator = GradientAccumulator(accumulation_steps=2)
        mp_manager = MixedPrecisionManager(enabled=False)  # Disable for CPU testing
        
        class DummyTrainer:
            pass
        
        multi_task_trainer = MultiTaskTrainer(task_configs, DummyTrainer())
        
        print(f"  ✓ All components initialized successfully")
        
        # Simulate a mini training loop
        for epoch in range(3):
            # Update curriculum
            performance = 0.7 + epoch * 0.1
            curriculum_info = curriculum.update_epoch(performance)
            
            # Get curriculum adjustments
            adjustments = curriculum.get_training_config_adjustments()
            
            # Simulate training step
            dummy_loss = torch.tensor(0.5)
            scaled_loss = accumulator.accumulate_loss(dummy_loss)
            
            # Step scheduler
            scheduler.step()
            
            print(f"    Epoch {epoch+1}: Stage={curriculum_info['current_stage_name']}, "
                  f"LR={optimizer.param_groups[0]['lr']:.6f}, "
                  f"Performance={performance:.2f}")
        
        print(f"  ✓ Integration test completed successfully")
        
        # Test summary generation
        curriculum_summary = curriculum.get_curriculum_summary()
        multi_task_summary = multi_task_trainer.get_multi_task_summary()
        grad_stats = accumulator.get_gradient_stats()
        
        print(f"  ✓ All summaries generated successfully")
        
        print("  ✅ Integration test passed!")
        return True
        
    except Exception as e:
        print(f"  ❌ Integration test failed: {e}")
        return False


def main():
    """Main test function."""
    print("🧪 Testing Phase 2.4: Advanced Training Infrastructure")
    print("=" * 60)
    
    # Test categories
    tests = [
        ("Curriculum Learning", test_curriculum_learning),
        ("Multi-Task Learning", test_multi_task_learning),
        ("Enhanced Loss Functions", test_enhanced_losses),
        ("Advanced Schedulers", test_advanced_schedulers),
        ("Training Utilities", test_training_utilities),
        ("Component Integration", test_integration)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 40)
        results[test_name] = test_func()
    
    # Summary
    print(f"\n📊 Test Results Summary")
    print("=" * 40)
    
    passed_tests = sum(1 for result in results.values() if result)
    total_tests = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:.<30} {status}")
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed ({passed_tests/total_tests*100:.1f}%)")
    
    if passed_tests == total_tests:
        print("\n🎉 All tests passed! Phase 2.4 Advanced Training Infrastructure is ready.")
        return True
    else:
        print(f"\n⚠️  {total_tests - passed_tests} test(s) failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 