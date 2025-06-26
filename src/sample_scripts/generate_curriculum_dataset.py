#!/usr/bin/env python3
"""
Production script for generating curriculum datasets for Khmer OCR training.

This script provides a comprehensive interface for generating curriculum learning datasets
with various predefined curricula and custom configuration options.
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.synthetic_data_generator import (
    CurriculumDatasetGenerator, 
    CurriculumStage, 
    CurriculumConfig
)


def setup_output_directory(base_output_dir: str, curriculum_name: str) -> Path:
    """Setup and create output directory with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(base_output_dir) / f"curriculum_{curriculum_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Output directory: {output_dir}")
    return output_dir


def print_curriculum_info(generator: CurriculumDatasetGenerator, curriculum_name: str):
    """Print detailed information about a curriculum."""
    try:
        info = generator.get_curriculum_info(curriculum_name)
        
        print(f"\n📖 CURRICULUM: {curriculum_name.upper()}")
        print("=" * 60)
        print(f"Description: {info['description']}")
        print(f"Total stages: {info['total_stages']}")
        print(f"Total samples: {info['total_samples']:,}")
        print(f"Difficulty range: {info['difficulty_range'][0]}-{info['difficulty_range'][1]}")
        
        print(f"\nStages breakdown:")
        for stage in info['stages']:
            corpus_info = f" (corpus: {stage['corpus_ratio']:.1%})" if stage['corpus_ratio'] > 0 else ""
            print(f"  {stage['index']}. {stage['name']} - Difficulty: {stage['difficulty']}")
            print(f"     Samples: {stage['samples']:,}, Length: {stage['length_range']}{corpus_info}")
            print(f"     Content: {', '.join(stage['content_types'])}")
        
        print()
        
    except Exception as e:
        print(f"❌ Error getting curriculum info: {e}")


def generate_curriculum_with_reporting(
    generator: CurriculumDatasetGenerator,
    curriculum_name: str,
    output_dir: Path,
    train_split: float = 0.8,
    save_images: bool = True,
    show_progress: bool = True,
    custom_curriculum: Optional[CurriculumConfig] = None
) -> Dict:
    """Generate curriculum dataset with comprehensive reporting."""
    
    print(f"\n🏗️ GENERATING CURRICULUM DATASET")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        # Print curriculum information
        if not custom_curriculum:
            print_curriculum_info(generator, curriculum_name)
        else:
            print(f"Using custom curriculum: {custom_curriculum.name}")
        
        # Generate the dataset
        print("🚀 Starting dataset generation...")
        metadata = generator.generate_curriculum_dataset(
            curriculum_name=curriculum_name if not custom_curriculum else "custom",
            custom_curriculum=custom_curriculum,
            train_split=train_split,
            save_images=save_images,
            show_progress=show_progress
        )
        
        generation_time = time.time() - start_time
        
        # Print generation summary
        print_generation_summary(metadata, generation_time, output_dir)
        
        # Save detailed report
        save_generation_report(metadata, generation_time, output_dir, curriculum_name)
        
        return metadata
        
    except Exception as e:
        print(f"\n❌ Error during generation: {e}")
        import traceback
        traceback.print_exc()
        return {}


def print_generation_summary(metadata: Dict, generation_time: float, output_dir: Path):
    """Print a comprehensive generation summary."""
    if not metadata or 'statistics' not in metadata:
        print("❌ No generation statistics available")
        return
    
    stats = metadata['statistics']
    
    print(f"\n✅ GENERATION COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"⏱️  Total time: {generation_time:.1f} seconds ({generation_time/60:.1f} minutes)")
    print(f"📊 Total samples: {stats['total_samples']:,}")
    print(f"📚 Training samples: {stats['total_train_samples']:,}")
    print(f"🧪 Validation samples: {stats['total_val_samples']:,}")
    print(f"🎯 Total stages: {stats['total_stages']}")
    print(f"📁 Output directory: {output_dir}")
    
    print(f"\n📈 Stage Progression:")
    for stage_info in stats['difficulty_progression']:
        print(f"  • {stage_info['stage']}: Difficulty {stage_info['difficulty']}, {stage_info['samples']:,} samples")
    
    print(f"\n🎨 Content Distribution:")
    for content_type, count in stats['content_type_distribution'].items():
        percentage = (count / stats['total_samples']) * 100
        print(f"  • {content_type}: {count:,} samples ({percentage:.1f}%)")
    
    print(f"\n🔧 Generation Methods:")
    for gen_type, count in stats['generation_type_distribution'].items():
        percentage = (count / stats['total_samples']) * 100
        print(f"  • {gen_type}: {count:,} samples ({percentage:.1f}%)")
    
    # Calculate generation rate
    samples_per_second = stats['total_samples'] / generation_time
    print(f"\n⚡ Performance:")
    print(f"  • Generation rate: {samples_per_second:.1f} samples/second")
    print(f"  • Average time per sample: {generation_time/stats['total_samples']*1000:.1f} ms")


def save_generation_report(metadata: Dict, generation_time: float, output_dir: Path, curriculum_name: str):
    """Save a detailed generation report to files."""
    
    # Create reports directory
    reports_dir = output_dir / 'reports'
    reports_dir.mkdir(exist_ok=True)
    
    # Prepare report data
    report_data = {
        'generation_info': {
            'curriculum_name': curriculum_name,
            'generation_time_seconds': generation_time,
            'generation_time_minutes': generation_time / 60,
            'timestamp': datetime.now().isoformat(),
            'output_directory': str(output_dir)
        },
        'metadata': metadata
    }
    
    # Save JSON report
    json_report_path = reports_dir / 'generation_report.json'
    with open(json_report_path, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False, default=str)
    
    # Save human-readable summary
    summary_path = reports_dir / 'generation_summary.txt'
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"CURRICULUM DATASET GENERATION REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"=" * 60 + "\n\n")
        
        f.write(f"Curriculum: {curriculum_name}\n")
        f.write(f"Generation time: {generation_time:.1f} seconds ({generation_time/60:.1f} minutes)\n")
        f.write(f"Output directory: {output_dir}\n\n")
        
        if 'statistics' in metadata:
            stats = metadata['statistics']
            f.write(f"DATASET STATISTICS\n")
            f.write(f"-" * 30 + "\n")
            f.write(f"Total samples: {stats['total_samples']:,}\n")
            f.write(f"Training samples: {stats['total_train_samples']:,}\n")
            f.write(f"Validation samples: {stats['total_val_samples']:,}\n")
            f.write(f"Total stages: {stats['total_stages']}\n\n")
            
            f.write(f"STAGE PROGRESSION\n")
            f.write(f"-" * 30 + "\n")
            for stage_info in stats['difficulty_progression']:
                f.write(f"{stage_info['stage']}: Difficulty {stage_info['difficulty']}, {stage_info['samples']:,} samples\n")
            
            f.write(f"\nCONTENT DISTRIBUTION\n")
            f.write(f"-" * 30 + "\n")
            for content_type, count in stats['content_type_distribution'].items():
                percentage = (count / stats['total_samples']) * 100
                f.write(f"{content_type}: {count:,} samples ({percentage:.1f}%)\n")
    
    print(f"\n📋 Reports saved:")
    print(f"  • JSON report: {json_report_path}")
    print(f"  • Summary report: {summary_path}")


def create_custom_quick_test_curriculum() -> CurriculumConfig:
    """Create a custom curriculum for quick testing."""
    stages = [
        CurriculumStage(
            name="quick_test",
            description="Quick test with minimal samples",
            difficulty_level=1,
            content_types=['digits'],
            content_weights=[1.0],
            length_range=(1, 3),
            min_accuracy_threshold=0.95,
            samples_per_stage=100,
            use_corpus=False,
            corpus_ratio=0.0
        )
    ]
    
    return CurriculumConfig(
        name="quick_test",
        description="Quick test curriculum with minimal samples for testing",
        stages=stages,
        progression_strategy="accuracy_based"
    )


def main():
    """Main script function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Generate curriculum datasets for Khmer OCR training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate basic curriculum
  python generate_curriculum_dataset.py --curriculum basic_khmer

  # Generate digits curriculum with custom split
  python generate_curriculum_dataset.py --curriculum digits_only --train-split 0.9

  # Generate custom quick test curriculum
  python generate_curriculum_dataset.py --curriculum custom

  # List available curricula
  python generate_curriculum_dataset.py --list-curricula
        """
    )
    
    parser.add_argument(
        '--curriculum', '-c',
        choices=['basic_khmer', 'advanced_khmer', 'comprehensive', 'digits_only', 'corpus_intensive', 'custom'],
        default='basic_khmer',
        help='Curriculum to generate (default: basic_khmer)'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        default='curriculum_output',
        help='Base output directory (default: curriculum_output)'
    )
    
    parser.add_argument(
        '--config-path',
        default='config/model_config.yaml',
        help='Path to model configuration file (default: config/model_config.yaml)'
    )
    
    parser.add_argument(
        '--fonts-dir',
        default='src/fonts',
        help='Directory containing Khmer fonts (default: src/fonts)'
    )
    
    parser.add_argument(
        '--train-split',
        type=float,
        default=0.8,
        help='Train/validation split ratio (default: 0.8)'
    )
    
    parser.add_argument(
        '--no-images',
        action='store_true',
        help='Generate metadata only, do not save images'
    )
    
    parser.add_argument(
        '--no-progress',
        action='store_true',
        help='Disable progress bars'
    )
    
    parser.add_argument(
        '--list-curricula',
        action='store_true',
        help='List all available curricula and exit'
    )
    
    parser.add_argument(
        '--info',
        help='Show detailed information about a curriculum and exit'
    )
    
    args = parser.parse_args()
    
    # Print header
    print("🎓 CURRICULUM DATASET GENERATOR")
    print("=" * 60)
    print(f"🕐 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Initialize generator
        generator = CurriculumDatasetGenerator(
            config_path=args.config_path,
            fonts_dir=args.fonts_dir,
            output_dir=args.output_dir
        )
        
        # Handle list curricula option
        if args.list_curricula:
            curricula = generator.list_available_curricula()
            print(f"\n📚 Available curricula:")
            for curriculum in curricula:
                print(f"  • {curriculum}")
            return
        
        # Handle info option
        if args.info:
            print_curriculum_info(generator, args.info)
            return
        
        # Setup output directory
        output_dir = setup_output_directory(args.output_dir, args.curriculum)
        
        # Handle custom curriculum
        custom_curriculum = None
        if args.curriculum == 'custom':
            print(f"📝 Creating custom quick test curriculum")
            custom_curriculum = create_custom_quick_test_curriculum()
        
        # Generate the curriculum dataset
        metadata = generate_curriculum_with_reporting(
            generator=generator,
            curriculum_name=args.curriculum,
            output_dir=output_dir,
            train_split=args.train_split,
            save_images=not args.no_images,
            show_progress=not args.no_progress,
            custom_curriculum=custom_curriculum
        )
        
        if metadata:
            print(f"\n🎉 Dataset generation completed successfully!")
            print(f"📁 All files saved to: {output_dir}")
        else:
            print(f"\n❌ Dataset generation failed!")
            return 1
    
    except KeyboardInterrupt:
        print(f"\n\n⏹️ Generation interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Script execution failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 