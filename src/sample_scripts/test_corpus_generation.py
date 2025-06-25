#!/usr/bin/env python3
"""
Test script for corpus-based text generation vs synthetic generation.
Demonstrates the quality and characteristics of both approaches.
"""

import os
import sys
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'src' / 'modules'))

from synthetic_data_generator.generator import SyntheticDataGenerator
from synthetic_data_generator.utils import (
    analyze_corpus_segments, load_khmer_corpus, segment_corpus_text,
    extract_corpus_segments_by_complexity, generate_corpus_based_text
)

def test_corpus_loading():
    """Test corpus loading and basic analysis."""
    print("="*60)
    print("CORPUS LOADING AND ANALYSIS")
    print("="*60)
    
    corpus_lines = load_khmer_corpus()
    if not corpus_lines:
        print("❌ Could not load corpus")
        return False
    
    print(f"✅ Corpus loaded successfully")
    print(f"Lines: {len(corpus_lines)}")
    print(f"Total characters: {sum(len(line) for line in corpus_lines):,}")
    print(f"Average line length: {sum(len(line) for line in corpus_lines) / len(corpus_lines):.1f}")
    
    return corpus_lines

def test_corpus_segmentation(corpus_lines):
    """Test corpus text segmentation."""
    print("\n" + "="*60)
    print("CORPUS SEGMENTATION TESTING")
    print("="*60)
    
    # Test different length segments
    lengths = [3, 8, 15, 25]
    
    for target_length in lengths:
        print(f"\n✅ Testing {target_length}-character segments:")
        
        for i in range(5):
            segment = segment_corpus_text(
                corpus_lines,
                target_length=target_length,
                min_length=max(1, target_length - 3),
                max_length=target_length + 5
            )
            
            print(f"  {i+1}. '{segment}' (length: {len(segment)})")

def test_complexity_extraction(corpus_lines):
    """Test complexity-based segment extraction."""
    print("\n" + "="*60)
    print("COMPLEXITY-BASED EXTRACTION")
    print("="*60)
    
    complexity_levels = ['simple', 'medium', 'complex']
    
    for complexity in complexity_levels:
        print(f"\n✅ {complexity.upper()} complexity segments:")
        
        segments = extract_corpus_segments_by_complexity(
            corpus_lines, 
            complexity_level=complexity, 
            num_segments=5
        )
        
        for i, segment in enumerate(segments):
            print(f"  {i+1}. '{segment}' (length: {len(segment)})")

def compare_generation_methods():
    """Compare corpus-based vs synthetic generation."""
    print("\n" + "="*60)
    print("GENERATION METHOD COMPARISON")
    print("="*60)
    
    config_path = project_root / 'config' / 'model_config.yaml'
    fonts_dir = project_root / 'src' / 'fonts'
    output_dir = project_root / 'test_output' / 'corpus_comparison'
    
    # Test both approaches
    methods = [
        ("Corpus-based", True),
        ("Synthetic", False)
    ]
    
    for method_name, use_corpus in methods:
        print(f"\n🔍 {method_name} Generation:")
        
        generator = SyntheticDataGenerator(
            config_path=str(config_path),
            fonts_dir=str(fonts_dir),
            output_dir=str(output_dir),
            mode='full_text',
            use_corpus=use_corpus
        )
        
        # Generate samples with different content types
        content_types = ['characters', 'words', 'phrases']
        
        for content_type in content_types:
            print(f"\n  {content_type.upper()}:")
            
            for i in range(3):
                try:
                    text = generator._generate_text_content(
                        content_type=content_type,
                        length_range=(5, 15)
                    )
                    
                    print(f"    {i+1}. '{text}' (length: {len(text)})")
                    
                except Exception as e:
                    print(f"    {i+1}. Error: {e}")

def test_curriculum_with_corpus():
    """Test curriculum learning with corpus-based generation."""
    print("\n" + "="*60)
    print("CURRICULUM LEARNING WITH CORPUS")
    print("="*60)
    
    config_path = project_root / 'config' / 'model_config.yaml'
    fonts_dir = project_root / 'src' / 'fonts'
    output_dir = project_root / 'test_output' / 'corpus_curriculum'
    
    generator = SyntheticDataGenerator(
        config_path=str(config_path),
        fonts_dir=str(fonts_dir),
        output_dir=str(output_dir),
        mode='full_text',
        use_corpus=True
    )
    
    # Test curriculum stage generation
    stages = ['stage1', 'stage2', 'stage3']
    
    for stage in stages:
        print(f"\n✅ Testing {stage} with corpus:")
        
        try:
            dataset = generator.generate_curriculum_dataset(
                stage=stage,
                num_samples=5,
                save_images=False,
                show_progress=False
            )
            
            # Show sample labels
            sample_labels = [s['label'] for s in dataset['train']['samples']]
            print(f"  Sample labels: {sample_labels}")
            print(f"  Character subset size: {dataset['dataset_info']['character_subset_size']}")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")

def analyze_text_quality():
    """Analyze the quality of corpus vs synthetic text."""
    print("\n" + "="*60)
    print("TEXT QUALITY ANALYSIS")
    print("="*60)
    
    # Analyze corpus segments
    print("🔍 Analyzing corpus segments...")
    analysis = analyze_corpus_segments(num_samples=30)
    
    if 'error' not in analysis:
        print(f"\n📊 Corpus Statistics:")
        stats = analysis['corpus_stats']
        print(f"  Total lines: {stats['total_lines']:,}")
        print(f"  Average line length: {stats['avg_line_length']:.1f}")
        print(f"  Total characters: {stats['total_characters']:,}")
        
        print(f"\n📈 Segment Analysis:")
        for complexity, data in analysis['segment_analysis'].items():
            print(f"  {complexity.upper()}:")
            print(f"    Count: {data['count']}")
            print(f"    Avg length: {data['avg_length']:.1f}")
            print(f"    Length range: {data['min_length']}-{data['max_length']}")
            print(f"    Unique characters: {data['unique_characters']}")
            print(f"    Sample: {data['sample_segments'][0] if data['sample_segments'] else 'None'}")
    
def generate_comparison_samples():
    """Generate side-by-side comparison samples."""
    print("\n" + "="*60)
    print("GENERATING COMPARISON SAMPLES")
    print("="*60)
    
    config_path = project_root / 'config' / 'model_config.yaml'
    fonts_dir = project_root / 'src' / 'fonts'
    output_dir = project_root / 'test_output' / 'comparison_samples'
    
    # Create directories
    corpus_dir = Path(output_dir) / 'corpus_based'
    synthetic_dir = Path(output_dir) / 'synthetic'
    corpus_dir.mkdir(parents=True, exist_ok=True)
    synthetic_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate samples with both methods
    for use_corpus, output_folder, method_name in [
        (True, corpus_dir, "Corpus-based"),
        (False, synthetic_dir, "Synthetic")
    ]:
        print(f"\n🎨 Generating {method_name} samples...")
        
        generator = SyntheticDataGenerator(
            config_path=str(config_path),
            fonts_dir=str(fonts_dir),
            output_dir=str(output_folder),
            mode='full_text',
            use_corpus=use_corpus
        )
        
        # Generate a small dataset for comparison
        dataset = generator.generate_dataset(
            num_samples=10,
            train_split=0.8,
            save_images=True,
            show_progress=False
        )
        
        print(f"  ✅ Generated {dataset['dataset_info']['total_samples']} samples")
        
        # Show sample labels
        sample_labels = [s['label'] for s in dataset['train']['samples'][:5]]
        print(f"  Sample labels: {sample_labels}")

def main():
    """Run comprehensive corpus vs synthetic comparison."""
    print("🚀 CORPUS vs SYNTHETIC TEXT GENERATION COMPARISON")
    print("Testing corpus-based text generation capabilities")
    
    try:
        # Test corpus loading
        corpus_lines = test_corpus_loading()
        if not corpus_lines:
            print("❌ Cannot proceed without corpus")
            return 1
        
        # Run all tests
        test_corpus_segmentation(corpus_lines)
        test_complexity_extraction(corpus_lines)
        compare_generation_methods()
        test_curriculum_with_corpus()
        analyze_text_quality()
        generate_comparison_samples()
        
        print("\n" + "="*60)
        print("🎉 COMPARISON COMPLETE!")
        print("="*60)
        print("✅ Corpus-based text generation provides:")
        print("  - Authentic Khmer language patterns")
        print("  - Natural word formations and grammar")
        print("  - Real COENG stacking and diacritics")
        print("  - Proper sentence structures")
        print("  - Realistic character combinations")
        print("\n✅ Best practice: Use corpus for words/phrases, synthetic for characters/syllables")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 