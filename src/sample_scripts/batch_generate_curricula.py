#!/usr/bin/env python3
"""
Batch curriculum dataset generation script for Khmer OCR training.

This script generates multiple curriculum datasets efficiently with parallel processing,
comprehensive configuration options, and detailed batch reporting.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.synthetic_data_generator import (
    CurriculumDatasetGenerator, 
    CurriculumStage, 
    CurriculumConfig
)


class BatchCurriculumGenerator:
    """Manages batch generation of multiple curriculum datasets."""
    
    def __init__(self, config_path: str, fonts_dir: str, base_output_dir: str):
        """Initialize the batch generator."""
        self.config_path = config_path
        self.fonts_dir = fonts_dir
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Thread-safe progress tracking
        self.lock = threading.Lock()
        self.completed_curricula = []
        self.failed_curricula = []
        
        print(f"🏭 Batch Curriculum Generator initialized")
        print(f"📁 Base output directory: {self.base_output_dir}")
    
    def create_generator(self, output_subdir: str) -> CurriculumDatasetGenerator:
        """Create a curriculum generator for a specific output subdirectory."""
        output_dir = self.base_output_dir / output_subdir
        return CurriculumDatasetGenerator(
            config_path=self.config_path,
            fonts_dir=self.fonts_dir,
            output_dir=str(output_dir)
        )
    
    def generate_single_curriculum(self, 
                                 curriculum_name: str,
                                 output_subdir: str,
                                 train_split: float = 0.8,
                                 save_images: bool = True,
                                 custom_curriculum: Optional[CurriculumConfig] = None) -> Dict:
        """Generate a single curriculum dataset."""
        
        start_time = time.time()
        thread_id = threading.get_ident()
        
        try:
            print(f"🔄 [Thread {thread_id}] Starting {curriculum_name}...")
            
            generator = self.create_generator(output_subdir)
            
            metadata = generator.generate_curriculum_dataset(
                curriculum_name=curriculum_name if not custom_curriculum else "custom",
                custom_curriculum=custom_curriculum,
                train_split=train_split,
                save_images=save_images,
                show_progress=False  # Disable progress bars for batch processing
            )
            
            generation_time = time.time() - start_time
            
            result = {
                'curriculum_name': curriculum_name,
                'status': 'success',
                'generation_time': generation_time,
                'output_dir': str(self.base_output_dir / output_subdir),
                'metadata': metadata,
                'thread_id': thread_id
            }
            
            with self.lock:
                self.completed_curricula.append(result)
            
            if 'statistics' in metadata:
                stats = metadata['statistics']
                print(f"✅ [Thread {thread_id}] {curriculum_name} completed: {stats['total_samples']} samples in {generation_time:.1f}s")
            else:
                print(f"✅ [Thread {thread_id}] {curriculum_name} completed in {generation_time:.1f}s")
            
            return result
            
        except Exception as e:
            generation_time = time.time() - start_time
            
            result = {
                'curriculum_name': curriculum_name,
                'status': 'failed',
                'generation_time': generation_time,
                'error': str(e),
                'output_dir': str(self.base_output_dir / output_subdir),
                'thread_id': thread_id
            }
            
            with self.lock:
                self.failed_curricula.append(result)
            
            print(f"❌ [Thread {thread_id}] {curriculum_name} failed: {e}")
            return result
    
    def generate_batch(self, 
                      curricula_configs: List[Dict],
                      max_workers: int = 2,
                      train_split: float = 0.8,
                      save_images: bool = True) -> Dict:
        """Generate multiple curricula in parallel."""
        
        print(f"\n🚀 Starting batch generation of {len(curricula_configs)} curricula")
        print(f"⚙️ Max workers: {max_workers}")
        print(f"📊 Train split: {train_split}")
        print(f"💾 Save images: {save_images}")
        
        batch_start_time = time.time()
        
        # Prepare tasks
        tasks = []
        for i, config in enumerate(curricula_configs):
            curriculum_name = config['name']
            output_subdir = f"{curriculum_name}_{i:02d}"
            
            task_args = {
                'curriculum_name': curriculum_name,
                'output_subdir': output_subdir,
                'train_split': train_split,
                'save_images': save_images,
                'custom_curriculum': config.get('custom_curriculum')
            }
            tasks.append(task_args)
        
        # Execute tasks in parallel
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(self.generate_single_curriculum, **task): task 
                for task in tasks
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_task):
                result = future.result()
                results.append(result)
                
                # Print progress
                completed = len(results)
                total = len(tasks)
                print(f"📈 Progress: {completed}/{total} curricula completed ({completed/total*100:.1f}%)")
        
        batch_time = time.time() - batch_start_time
        
        # Create batch summary
        batch_summary = self.create_batch_summary(results, batch_time)
        
        # Save batch report
        self.save_batch_report(batch_summary)
        
        return batch_summary
    
    def create_batch_summary(self, results: List[Dict], batch_time: float) -> Dict:
        """Create a comprehensive batch summary."""
        
        successful = [r for r in results if r['status'] == 'success']
        failed = [r for r in results if r['status'] == 'failed']
        
        total_samples = 0
        total_train_samples = 0
        total_val_samples = 0
        
        for result in successful:
            if 'metadata' in result and 'statistics' in result['metadata']:
                stats = result['metadata']['statistics']
                total_samples += stats.get('total_samples', 0)
                total_train_samples += stats.get('total_train_samples', 0)
                total_val_samples += stats.get('total_val_samples', 0)
        
        summary = {
            'batch_info': {
                'total_curricula': len(results),
                'successful_curricula': len(successful),
                'failed_curricula': len(failed),
                'success_rate': len(successful) / len(results) * 100 if results else 0,
                'batch_time_seconds': batch_time,
                'batch_time_minutes': batch_time / 60,
                'timestamp': datetime.now().isoformat(),
                'output_directory': str(self.base_output_dir)
            },
            'dataset_totals': {
                'total_samples': total_samples,
                'total_train_samples': total_train_samples,
                'total_val_samples': total_val_samples,
                'samples_per_second': total_samples / batch_time if batch_time > 0 else 0
            },
            'successful_curricula': successful,
            'failed_curricula': failed
        }
        
        return summary
    
    def save_batch_report(self, batch_summary: Dict):
        """Save comprehensive batch report."""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        reports_dir = self.base_output_dir / f'batch_reports_{timestamp}'
        reports_dir.mkdir(exist_ok=True)
        
        # Save JSON report
        json_path = reports_dir / 'batch_report.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(batch_summary, f, indent=2, ensure_ascii=False, default=str)
        
        # Save human-readable summary
        summary_path = reports_dir / 'batch_summary.txt'
        with open(summary_path, 'w', encoding='utf-8') as f:
            batch_info = batch_summary['batch_info']
            dataset_totals = batch_summary['dataset_totals']
            
            f.write("BATCH CURRICULUM GENERATION REPORT\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("BATCH SUMMARY\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total curricula: {batch_info['total_curricula']}\n")
            f.write(f"Successful: {batch_info['successful_curricula']}\n")
            f.write(f"Failed: {batch_info['failed_curricula']}\n")
            f.write(f"Success rate: {batch_info['success_rate']:.1f}%\n")
            f.write(f"Batch time: {batch_info['batch_time_minutes']:.1f} minutes\n\n")
            
            f.write("DATASET TOTALS\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total samples: {dataset_totals['total_samples']:,}\n")
            f.write(f"Training samples: {dataset_totals['total_train_samples']:,}\n")
            f.write(f"Validation samples: {dataset_totals['total_val_samples']:,}\n")
            f.write(f"Generation rate: {dataset_totals['samples_per_second']:.1f} samples/second\n")
        
        print(f"\n📋 Batch reports saved:")
        print(f"  • JSON report: {json_path}")
        print(f"  • Summary report: {summary_path}")
    
    def print_batch_summary(self, batch_summary: Dict):
        """Print batch generation summary."""
        
        batch_info = batch_summary['batch_info']
        dataset_totals = batch_summary['dataset_totals']
        
        print(f"\n🎉 BATCH GENERATION COMPLETED!")
        print("=" * 60)
        print(f"⏱️  Total time: {batch_info['batch_time_minutes']:.1f} minutes")
        print(f"📊 Total curricula: {batch_info['total_curricula']}")
        print(f"✅ Successful: {batch_info['successful_curricula']}")
        print(f"❌ Failed: {batch_info['failed_curricula']}")
        print(f"📈 Success rate: {batch_info['success_rate']:.1f}%")
        
        print(f"\n📚 Dataset Totals:")
        print(f"  • Total samples: {dataset_totals['total_samples']:,}")
        print(f"  • Training samples: {dataset_totals['total_train_samples']:,}")
        print(f"  • Validation samples: {dataset_totals['total_val_samples']:,}")
        print(f"  • Generation rate: {dataset_totals['samples_per_second']:.1f} samples/second")
        
        print(f"\n📁 Output directory: {batch_info['output_directory']}")


def create_predefined_batch_configs() -> Dict[str, List[Dict]]:
    """Create predefined batch configurations."""
    
    configs = {
        'all_standard': [
            {'name': 'basic_khmer'},
            {'name': 'digits_only'},
            {'name': 'comprehensive'}
        ],
        
        'essential': [
            {'name': 'basic_khmer'},
            {'name': 'digits_only'}
        ],
        
        'quick_test': [
            {
                'name': 'quick_test_digits',
                'custom_curriculum': CurriculumConfig(
                    name="quick_test_digits",
                    description="Quick test with digits only",
                    stages=[
                        CurriculumStage(
                            name="test_stage",
                            description="Quick digits test",
                            difficulty_level=1,
                            content_types=['digits'],
                            content_weights=[1.0],
                            length_range=(1, 3),
                            samples_per_stage=50,  # Reduced for faster testing
                            use_corpus=False,
                            corpus_ratio=0.0
                        )
                    ]
                )
            }
        ]
    }
    
    return configs


def main():
    """Main script function."""
    parser = argparse.ArgumentParser(
        description="Batch generate multiple curriculum datasets for Khmer OCR training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate all standard curricula
  python batch_generate_curricula.py --batch all_standard

  # Generate essential curricula with 4 workers
  python batch_generate_curricula.py --batch essential --workers 4

  # Generate quick test batch
  python batch_generate_curricula.py --batch quick_test

  # List available batch configurations
  python batch_generate_curricula.py --list-batches
        """
    )
    
    parser.add_argument(
        '--batch', '-b',
        choices=['all_standard', 'essential', 'quick_test'],
        default='essential',
        help='Batch configuration to generate (default: essential)'
    )
    
    parser.add_argument(
        '--workers', '-w',
        type=int,
        default=2,
        help='Number of parallel workers (default: 2)'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        default='batch_curriculum_output',
        help='Base output directory (default: batch_curriculum_output)'
    )
    
    parser.add_argument(
        '--config-path',
        default='config/model_config.yaml',
        help='Path to model configuration file'
    )
    
    parser.add_argument(
        '--fonts-dir',
        default='src/fonts',
        help='Directory containing Khmer fonts'
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
        '--list-batches',
        action='store_true',
        help='List available batch configurations and exit'
    )
    
    args = parser.parse_args()
    
    # Print header
    print("🏭 BATCH CURRICULUM DATASET GENERATOR")
    print("=" * 60)
    print(f"🕐 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Handle list batches option
        if args.list_batches:
            configs = create_predefined_batch_configs()
            print(f"\n📚 Available batch configurations:")
            for batch_name, curricula in configs.items():
                print(f"\n  • {batch_name}: {len(curricula)} curricula")
                for curriculum in curricula:
                    print(f"    - {curriculum['name']}")
            return
        
        # Get batch configuration
        batch_configs = create_predefined_batch_configs()
        curricula_configs = batch_configs[args.batch]
        
        # Initialize batch generator
        batch_generator = BatchCurriculumGenerator(
            config_path=args.config_path,
            fonts_dir=args.fonts_dir,
            base_output_dir=args.output_dir
        )
        
        # Generate batch
        batch_summary = batch_generator.generate_batch(
            curricula_configs=curricula_configs,
            max_workers=args.workers,
            train_split=args.train_split,
            save_images=not args.no_images
        )
        
        # Print final summary
        batch_generator.print_batch_summary(batch_summary)
        
        return 0
    
    except KeyboardInterrupt:
        print(f"\n\n⏹️ Batch generation interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Batch generation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 