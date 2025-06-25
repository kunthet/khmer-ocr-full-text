#!/usr/bin/env python3
"""
Test Script for Progressive Training Strategy (Phase 3.1)
"""

import os
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_curriculum_stages():
    """Test curriculum stages creation."""
    print("🔍 Testing Curriculum Stages...")
    
    try:
        from modules.trainers import CurriculumLearningManager, CurriculumStage, DifficultyLevel
        
        stage = CurriculumStage(
            level=DifficultyLevel.SINGLE_CHAR,
            name="Test Stage",
            description="Test stage description",
            min_epochs=5,
            max_epochs=15,
            success_threshold=0.90
        )
        
        assert stage.name == "Test Stage"
        assert stage.min_epochs == 5
        print("  ✅ CurriculumStage creation works")
        return True
        
    except Exception as e:
        print(f"  ❌ Curriculum stage test failed: {e}")
        return False

def test_progressive_training_strategy():
    """Test the progressive training strategy class."""
    print("\n🔍 Testing Progressive Training Strategy...")
    
    try:
        sys.path.append(str(Path(__file__).parent))
        from progressive_training_strategy import ProgressiveTrainingStrategy
        
        strategy = ProgressiveTrainingStrategy(
            config_path="config/model_config.yaml",
            output_dir="test_output"
        )
        
        curriculum_manager = strategy.create_curriculum_manager()
        
        assert len(curriculum_manager.stages) == 5
        print("  ✅ Progressive training strategy works")
        print(f"     Created {len(curriculum_manager.stages)} stages")
        
        for i, stage in enumerate(curriculum_manager.stages):
            print(f"     Stage {i+1}: {stage.name}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Progressive training strategy test failed: {e}")
        return False

def test_phase_3_1_requirements():
    """Test Phase 3.1 requirements."""
    print("\n🔍 Testing Phase 3.1 Requirements...")
    
    try:
        sys.path.append(str(Path(__file__).parent))
        from progressive_training_strategy import ProgressiveTrainingStrategy
        from modules.trainers import DifficultyLevel
        
        strategy = ProgressiveTrainingStrategy(config_path="config/model_config.yaml")
        curriculum_manager = strategy.create_curriculum_manager()
        stages = curriculum_manager.stages
        
        requirements = {
            "Stage 1: Single character": stages[0].level == DifficultyLevel.SINGLE_CHAR,
            "Stage 2: Simple combinations": stages[1].level == DifficultyLevel.SIMPLE_COMBO,
            "Stage 3: Complex combinations": stages[2].level == DifficultyLevel.COMPLEX_COMBO,
            "Stage 4: Word-level": stages[3].level == DifficultyLevel.WORD_LEVEL,
            "Stage 5: Multi-word": stages[4].level == DifficultyLevel.MULTI_WORD,
            "Progressive difficulty": all(
                stages[i].success_threshold >= stages[i+1].success_threshold 
                for i in range(len(stages)-1)
            )
        }
        
        all_passed = True
        for requirement, passed in requirements.items():
            status = "✅" if passed else "❌"
            print(f"  {status} {requirement}")
            if not passed:
                all_passed = False
        
        return all_passed
        
    except Exception as e:
        print(f"  ❌ Phase 3.1 requirements test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Testing Progressive Training Strategy Implementation (Phase 3.1)")
    print("=" * 70)
    
    tests = [
        ("Curriculum Stages", test_curriculum_stages),
        ("Progressive Training Strategy", test_progressive_training_strategy),
        ("Phase 3.1 Requirements", test_phase_3_1_requirements)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 70)
    print("📊 Test Results Summary:")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All tests passed! Progressive Training Strategy (Phase 3.1) is ready!")
        return 0
    else:
        print("⚠️ Some tests failed. Please review the implementation.")
        return 1

if __name__ == '__main__':
    exit(main())
