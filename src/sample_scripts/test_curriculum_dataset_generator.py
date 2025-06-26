#!/usr/bin/env python3
"""
Test script for the Curriculum Dataset Generator.

This script demonstrates the advanced curriculum learning dataset generation capabilities
including predefined curricula, custom curricula, and comprehensive analytics.
"""

import os
import sys
import json
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.synthetic_data_generator import (
    CurriculumDatasetGenerator, 
    CurriculumStage, 
    CurriculumConfig
)


def test_predefined_curricula():
    """Test all predefined curriculum configurations."""
    print("🎓 Testing Predefined Curricula")
    print("=" * 50)
    
    # Initialize generator
    config_path = "config/model_config.yaml"
    fonts_dir = "src/fonts"
    output_dir = "test_output/curriculum_datasets"
    
    generator = CurriculumDatasetGenerator(config_path, fonts_dir, output_dir)
    
    # List available curricula
    curricula = generator.list_available_curricula()
    print(f"📚 Available curricula: {curricula}")
    
    # Get detailed info for each curriculum
    for curriculum_name in curricula:
        print(f"\n📖 {curriculum_name.upper()} CURRICULUM")
        print("-" * 30)
        
        try:
            info = generator.get_curriculum_info(curriculum_name)
            print(f"Description: {info['description']}")
            print(f"Total stages: {info['total_stages']}")
            print(f"Total samples: {info['total_samples']}")
            print(f"Difficulty range: {info['difficulty_range'][0]}-{info['difficulty_range'][1]}")
            
            print("\nStages:")
            for stage in info['stages']:
                print(f"  {stage['index']}. {stage['name']} (difficulty: {stage['difficulty']})")
                print(f"     {stage['description']}")
                print(f"     Samples: {stage['samples']}, Content: {stage['content_types']}")
                if stage['corpus_ratio'] > 0:
                    print(f"     Corpus ratio: {stage['corpus_ratio']:.1%}")
                print()
        
        except Exception as e:
            print(f"❌ Error getting info for {curriculum_name}: {e}")


def test_small_curriculum_generation():
    """Test generating a small curriculum dataset."""
    print("\n🏗️ Testing Small Curriculum Generation")
    print("=" * 50)
    
    # Initialize generator
    config_path = "config/model_config.yaml"
    fonts_dir = "src/fonts"
    output_dir = "test_output/curriculum_small"
    
    generator = CurriculumDatasetGenerator(config_path, fonts_dir, output_dir)
    
    try:
        # Generate digits curriculum (fast and simple)
        print("Generating digits curriculum...")
        metadata = generator.generate_curriculum_dataset(
            curriculum_name="digits_only",
            train_split=0.8,
            save_images=True,
            show_progress=True
        )
        
        print("\n📊 Generation Results:")
        stats = metadata['statistics']
        print(f"Total stages: {stats['total_stages']}")
        print(f"Total samples: {stats['total_samples']}")
        print(f"Training samples: {stats['total_train_samples']}")
        print(f"Validation samples: {stats['total_val_samples']}")
        
        print("\nStage progression:")
        for stage_info in stats['difficulty_progression']:
            print(f"  {stage_info['stage']}: difficulty {stage_info['difficulty']}, {stage_info['samples']} samples")
        
        print(f"\nGeneration type distribution: {stats['generation_type_distribution']}")
        print(f"Content type distribution: {stats['content_type_distribution']}")
        
    except Exception as e:
        print(f"❌ Error during generation: {e}")
        import traceback
        traceback.print_exc()


def test_custom_curriculum():
    """Test creating and using a custom curriculum."""
    print("\n🎨 Testing Custom Curriculum")
    print("=" * 50)
    
    # Create custom curriculum stages
    stages = [
        CurriculumStage(
            name="test_foundation",
            description="Very basic test stage",
            difficulty_level=1,
            content_types=['digits'],
            content_weights=[1.0],
            length_range=(1, 2),
            min_accuracy_threshold=0.95,
            samples_per_stage=200,
            use_corpus=False,
            corpus_ratio=0.0
        ),
        CurriculumStage(
            name="test_progression",
            description="Slightly more complex test stage",
            difficulty_level=3,
            content_types=['digits', 'characters'],
            content_weights=[0.7, 0.3],
            length_range=(2, 4),
            min_accuracy_threshold=0.90,
            samples_per_stage=300,
            use_corpus=False,
            corpus_ratio=0.0
        )
    ]
    
    # Create custom curriculum configuration
    custom_curriculum = CurriculumConfig(
        name="test_custom",
        description="Custom test curriculum for demonstration",
        stages=stages,
        progression_strategy="accuracy_based",
        global_settings={
            'train_split': 0.75,
            'save_images': True,
            'show_progress': True
        }
    )
    
    # Initialize generator
    config_path = "config/model_config.yaml"
    fonts_dir = "src/fonts"
    output_dir = "test_output/curriculum_custom"
    
    generator = CurriculumDatasetGenerator(config_path, fonts_dir, output_dir)
    
    try:
        # Validate custom curriculum
        is_valid, errors = generator.validate_curriculum_config(custom_curriculum)
        if not is_valid:
            print("❌ Custom curriculum validation failed:")
            for error in errors:
                print(f"  - {error}")
            return
        
        print("✅ Custom curriculum validation passed")
        
        # Generate custom curriculum dataset
        print("Generating custom curriculum...")
        metadata = generator.generate_curriculum_dataset(
            custom_curriculum=custom_curriculum,
            train_split=0.75,
            save_images=True,
            show_progress=True
        )
        
        print("\n📊 Custom Curriculum Results:")
        stats = metadata['statistics']
        print(f"Total stages: {stats['total_stages']}")
        print(f"Total samples: {stats['total_samples']}")
        print(f"Training samples: {stats['total_train_samples']}")
        print(f"Validation samples: {stats['total_val_samples']}")
        
        # Print stage details
        for stage_name, stage_data in metadata['stages'].items():
            stage_config = stage_data['stage_config']
            print(f"\nStage: {stage_name}")
            print(f"  Difficulty: {stage_config['difficulty_level']}")
            print(f"  Content types: {stage_config['content_types']}")
            print(f"  Generation type: {stage_data['generation_type']}")
            if 'combined_train_samples' in stage_data:
                print(f"  Train samples: {stage_data['combined_train_samples']}")
                print(f"  Val samples: {stage_data['combined_val_samples']}")
            else:
                print(f"  Train samples: {len(stage_data['train']['samples'])}")
                print(f"  Val samples: {len(stage_data['val']['samples'])}")
        
    except Exception as e:
        print(f"❌ Error during custom curriculum generation: {e}")
        import traceback
        traceback.print_exc()


def test_curriculum_analytics():
    """Test curriculum analytics and validation features."""
    print("\n📈 Testing Curriculum Analytics")
    print("=" * 50)
    
    # Initialize generator
    config_path = "config/model_config.yaml"
    fonts_dir = "src/fonts"
    output_dir = "test_output/curriculum_analytics"
    
    generator = CurriculumDatasetGenerator(config_path, fonts_dir, output_dir)
    
    # Test validation with invalid curriculum
    print("Testing curriculum validation...")
    
    # Create invalid curriculum (weights don't sum to 1.0)
    invalid_stage = CurriculumStage(
        name="invalid_test",
        description="Invalid test stage",
        difficulty_level=1,
        content_types=['digits', 'characters'],
        content_weights=[0.5, 0.3],  # Only sums to 0.8, not 1.0
        length_range=(1, 2),
        samples_per_stage=100
    )
    
    invalid_curriculum = CurriculumConfig(
        name="invalid_test",
        description="Invalid curriculum for testing validation",
        stages=[invalid_stage]
    )
    
    is_valid, errors = generator.validate_curriculum_config(invalid_curriculum)
    print(f"Validation result: {'✅ Valid' if is_valid else '❌ Invalid'}")
    if not is_valid:
        print("Validation errors:")
        for error in errors:
            print(f"  - {error}")
    
    # Test with valid curriculum
    print("\nTesting with valid curriculum...")
    valid_stage = CurriculumStage(
        name="valid_test",
        description="Valid test stage",
        difficulty_level=5,
        content_types=['digits'],
        content_weights=[1.0],
        length_range=(1, 3),
        samples_per_stage=100
    )
    
    valid_curriculum = CurriculumConfig(
        name="valid_test",
        description="Valid curriculum for testing",
        stages=[valid_stage]
    )
    
    is_valid, errors = generator.validate_curriculum_config(valid_curriculum)
    print(f"Validation result: {'✅ Valid' if is_valid else '❌ Invalid'}")
    if errors:
        for error in errors:
            print(f"  - {error}")


def main():
    """Main test function."""
    print("🎓 CURRICULUM DATASET GENERATOR TESTS")
    print("=" * 60)
    
    # Ensure output directories exist
    os.makedirs("test_output", exist_ok=True)
    
    # Run tests
    try:
        test_predefined_curricula()
        test_curriculum_analytics() 
        test_small_curriculum_generation()
        test_custom_curriculum()
        
        print("\n✅ All curriculum dataset generator tests completed!")
        
    except Exception as e:
        print(f"\n❌ Test execution failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 