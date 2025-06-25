#!/usr/bin/env python3
"""
Test script for the data pipeline and utilities module.

This script demonstrates the usage of all data pipeline components
including dataset loading, preprocessing, visualization, and analysis.
"""

import os
import sys
import argparse
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.data_utils import (
    KhmerDigitsDataset, create_data_loaders,
    get_train_transforms, get_val_transforms,
    DataVisualizer, plot_samples, plot_dataset_stats,
    DatasetAnalyzer, calculate_dataset_metrics, validate_dataset_quality
)

import torch
import matplotlib.pyplot as plt


def test_dataset_loading(metadata_path: str):
    """Test basic dataset loading functionality."""
    print("=== Testing Dataset Loading ===")
    
    try:
        # Test different splits
        for split in ['train', 'val', 'all']:
            print(f"\nTesting split: {split}")
            dataset = KhmerDigitsDataset(metadata_path, split=split)
            print(f"  Loaded {len(dataset)} samples")
            
            # Test sample access
            if len(dataset) > 0:
                image, label, metadata = dataset[0]
                print(f"  First sample: '{metadata['original_label']}' "
                      f"(shape: {image.size if hasattr(image, 'size') else 'N/A'})")
                
                # Test character mappings
                char_to_idx, idx_to_char = dataset.get_character_mappings()
                print(f"  Character set size: {len(char_to_idx)}")
                
                # Test dataset stats
                stats = dataset.get_dataset_stats()
                print(f"  Sequence length range: {stats['sequence_length_stats']['min']}-{stats['sequence_length_stats']['max']}")
    
    except Exception as e:
        print(f"Error in dataset loading: {e}")
        return False
    
    print("✓ Dataset loading tests passed")
    return True


def test_preprocessing(metadata_path: str):
    """Test image preprocessing pipeline."""
    print("\n=== Testing Preprocessing Pipeline ===")
    
    try:
        # Create transforms
        train_transform = get_train_transforms(
            image_size=(128, 64),
            augmentation_strength=0.3,
            normalize=True
        )
        
        val_transform = get_val_transforms(
            image_size=(128, 64),
            normalize=True
        )
        
        print("✓ Created training and validation transforms")
        
        # Test with dataset
        train_dataset = KhmerDigitsDataset(
            metadata_path, 
            split='train', 
            transform=train_transform
        )
        
        val_dataset = KhmerDigitsDataset(
            metadata_path, 
            split='val', 
            transform=val_transform
        )
        
        if len(train_dataset) > 0:
            # Test transform application
            image, label, metadata = train_dataset[0]
            print(f"✓ Applied training transform - image shape: {image.shape}")
            
            image, label, metadata = val_dataset[0] if len(val_dataset) > 0 else train_dataset[0]
            print(f"✓ Applied validation transform - image shape: {image.shape}")
    
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return False
    
    print("✓ Preprocessing tests passed")
    return True


def test_data_loaders(metadata_path: str):
    """Test data loader creation and batching."""
    print("\n=== Testing Data Loaders ===")
    
    try:
        # Create transforms
        train_transform = get_train_transforms()
        val_transform = get_val_transforms()
        
        # Create data loaders
        train_loader, val_loader = create_data_loaders(
            metadata_path=metadata_path,
            train_transform=train_transform,
            val_transform=val_transform,
            batch_size=8,
            num_workers=0,  # Use 0 for testing to avoid multiprocessing issues
            shuffle_train=True
        )
        
        print(f"✓ Created data loaders:")
        print(f"  Training batches: {len(train_loader)}")
        print(f"  Validation batches: {len(val_loader)}")
        
        # Test batch loading
        if len(train_loader) > 0:
            images, labels, metadata_list = next(iter(train_loader))
            print(f"✓ Loaded training batch:")
            print(f"  Images shape: {images.shape}")
            print(f"  Labels shape: {labels.shape}")
            print(f"  Metadata items: {len(metadata_list)}")
            try:
                print(f"  Sample labels: {[m['original_label'] for m in metadata_list[:3]]}")
            except Exception as e:
                print(f"  Error accessing metadata: {e}")
                print(f"  Metadata type: {type(metadata_list)}")
                if len(metadata_list) > 0:
                    print(f"  First metadata item type: {type(metadata_list[0])}")
        
        if len(val_loader) > 0:
            images, labels, metadata_list = next(iter(val_loader))
            print(f"✓ Loaded validation batch:")
            print(f"  Images shape: {images.shape}")
            print(f"  Labels shape: {labels.shape}")
    
    except Exception as e:
        print(f"Error in data loader testing: {e}")
        return False
    
    print("✓ Data loader tests passed")
    return True


def test_visualization(metadata_path: str, output_dir: str):
    """Test visualization utilities."""
    print("\n=== Testing Visualization ===")
    
    try:
        # Create dataset
        dataset = KhmerDigitsDataset(metadata_path, split='train')
        
        if len(dataset) == 0:
            print("No samples in dataset for visualization")
            return False
        
        # Create visualizer
        visualizer = DataVisualizer()
        
        # Test sample plotting
        print("Creating sample visualization...")
        fig1 = visualizer.plot_samples(
            dataset, 
            num_samples=min(16, len(dataset)), 
            save_path=os.path.join(output_dir, 'samples.png')
        )
        plt.close(fig1)
        print("✓ Sample visualization saved")
        
        # Test dataset statistics plotting
        print("Creating dataset statistics visualization...")
        fig2 = visualizer.plot_dataset_statistics(
            dataset,
            save_path=os.path.join(output_dir, 'dataset_stats.png')
        )
        plt.close(fig2)
        print("✓ Dataset statistics visualization saved")
        
        # Test convenience functions
        fig3 = plot_samples(dataset, num_samples=8, save_path=os.path.join(output_dir, 'samples_simple.png'))
        plt.close(fig3)
        print("✓ Simple sample plot saved")
        
        # Test batch visualization with data loader
        train_transform = get_train_transforms()
        train_loader, _ = create_data_loaders(
            metadata_path, 
            train_transform=train_transform,
            val_transform=train_transform,
            batch_size=8,
            num_workers=0
        )
        
        if len(train_loader) > 0:
            fig4 = visualizer.plot_batch_samples(
                train_loader,
                num_batches=1,
                save_path=os.path.join(output_dir, 'batch_samples.png')
            )
            plt.close(fig4)
            print("✓ Batch visualization saved")
    
    except Exception as e:
        print(f"Error in visualization testing: {e}")
        return False
    
    print("✓ Visualization tests passed")
    return True


def test_analysis(metadata_path: str, output_dir: str):
    """Test dataset analysis utilities."""
    print("\n=== Testing Dataset Analysis ===")
    
    try:
        # Create dataset
        dataset = KhmerDigitsDataset(metadata_path, split='all')
        
        if len(dataset) == 0:
            print("No samples in dataset for analysis")
            return False
        
        # Create analyzer
        analyzer = DatasetAnalyzer(dataset)
        
        # Test sequence pattern analysis
        print("Analyzing sequence patterns...")
        seq_analysis = analyzer.analyze_sequence_patterns()
        print(f"✓ Found {seq_analysis['unique_sequences']} unique sequences")
        print(f"  Average length: {seq_analysis['average_length']:.1f}")
        print(f"  Most common sequence: {seq_analysis['most_common_sequences'][0] if seq_analysis['most_common_sequences'] else 'N/A'}")
        
        # Test visual analysis
        print("Analyzing visual properties...")
        visual_analysis = analyzer.analyze_visual_properties()
        print(f"✓ Analyzed {visual_analysis['samples_analyzed']} samples")
        print(f"  Average brightness: {visual_analysis['brightness_stats']['mean']:.1f}")
        print(f"  Average contrast: {visual_analysis['contrast_stats']['mean']:.1f}")
        
        # Test augmentation analysis
        print("Analyzing augmentations...")
        aug_analysis = analyzer.analyze_augmentation_impact()
        print(f"✓ Augmentation rate: {aug_analysis['augmentation_rate']:.1%}")
        
        # Test quality validation
        print("Validating data quality...")
        quality_validation = analyzer.validate_data_quality()
        print(f"✓ Dataset valid: {quality_validation['is_valid']}")
        if quality_validation['issues']:
            print(f"  Issues: {quality_validation['issues']}")
        if quality_validation['warnings']:
            print(f"  Warnings: {quality_validation['warnings']}")
        
        # Test comprehensive report
        print("Generating comprehensive report...")
        report = analyzer.generate_comprehensive_report(
            save_path=os.path.join(output_dir, 'analysis_report.json')
        )
        print("✓ Analysis report saved")
        
        # Test analysis plots
        print("Creating analysis plots...")
        plots = analyzer.create_analysis_plots(save_dir=output_dir)
        for plot_name in plots:
            plt.close(plots[plot_name])
        print(f"✓ Created {len(plots)} analysis plots")
        
        # Test metric calculation
        print("Calculating dataset metrics...")
        metrics = calculate_dataset_metrics(dataset)
        print(f"✓ Dataset metrics:")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.3f}")
            else:
                print(f"  {key}: {value}")
        
        # Test quality validation
        print("Validating dataset quality...")
        is_valid, issues = validate_dataset_quality(dataset)
        print(f"✓ Quality validation: {'PASS' if is_valid else 'FAIL'}")
        if issues:
            print(f"  Issues: {issues}")
    
    except Exception as e:
        print(f"Error in analysis testing: {e}")
        return False
    
    print("✓ Analysis tests passed")
    return True


def test_integration(metadata_path: str):
    """Test end-to-end integration."""
    print("\n=== Testing Integration ===")
    
    try:
        # Create full pipeline
        train_transform = get_train_transforms(image_size=(128, 64), augmentation_strength=0.2)
        val_transform = get_val_transforms(image_size=(128, 64))
        
        train_loader, val_loader = create_data_loaders(
            metadata_path=metadata_path,
            train_transform=train_transform,
            val_transform=val_transform,
            batch_size=4,
            num_workers=0
        )
        
        print(f"✓ Created full pipeline with {len(train_loader)} train batches, {len(val_loader)} val batches")
        
        # Test multiple batch loading
        train_samples = 0
        val_samples = 0
        
        for i, (images, labels, metadata_list) in enumerate(train_loader):
            train_samples += images.shape[0]
            if i >= 2:  # Test first 3 batches
                break
        
        for i, (images, labels, metadata_list) in enumerate(val_loader):
            val_samples += images.shape[0]
            if i >= 1:  # Test first 2 batches
                break
        
        print(f"✓ Successfully loaded {train_samples} training samples and {val_samples} validation samples")
        
        # Test character encoding/decoding
        if len(train_loader) > 0:
            images, labels, metadata_list = next(iter(train_loader))
            dataset = train_loader.dataset.dataset if hasattr(train_loader.dataset, 'dataset') else train_loader.dataset
            
            # Test encoding/decoding consistency
            for i in range(min(3, len(metadata_list))):
                original_label = metadata_list[i]['original_label']
                encoded_label = labels[i]
                decoded_label = dataset._decode_label(encoded_label)
                
                print(f"  Sample {i}: '{original_label}' -> {encoded_label[:len(original_label)+1].tolist()} -> '{decoded_label}'")
                
                if original_label != decoded_label:
                    print(f"  Warning: Encoding/decoding mismatch for sample {i}!")
    
    except Exception as e:
        print(f"Error in integration testing: {e}")
        return False
    
    print("✓ Integration tests passed")
    return True


def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description='Test data pipeline utilities')
    parser.add_argument('--metadata', type=str, default='generated_data/metadata.yaml',
                      help='Path to metadata file')
    parser.add_argument('--output-dir', type=str, default='data_pipeline_test_output',
                      help='Output directory for test results')
    parser.add_argument('--skip-visualization', action='store_true',
                      help='Skip visualization tests (faster)')
    parser.add_argument('--skip-analysis', action='store_true',
                      help='Skip analysis tests (faster)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.metadata):
        print(f"Error: Metadata file not found: {args.metadata}")
        print("Please run the data generation script first:")
        print("  python src/sample_scripts/generate_dataset.py --num-samples 100")
        return 1
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=== Data Pipeline and Utilities Test ===")
    print(f"Metadata file: {args.metadata}")
    print(f"Output directory: {args.output_dir}")
    print()
    
    # Run tests
    test_results = []
    
    test_results.append(test_dataset_loading(args.metadata))
    test_results.append(test_preprocessing(args.metadata))
    test_results.append(test_data_loaders(args.metadata))
    
    if not args.skip_visualization:
        test_results.append(test_visualization(args.metadata, args.output_dir))
    
    if not args.skip_analysis:
        test_results.append(test_analysis(args.metadata, args.output_dir))
    
    test_results.append(test_integration(args.metadata))
    
    # Summary
    print("\n=== Test Summary ===")
    passed_tests = sum(test_results)
    total_tests = len(test_results)
    
    print(f"Tests passed: {passed_tests}/{total_tests}")
    
    if passed_tests == total_tests:
        print("✅ All tests passed! Data pipeline is ready for training.")
        if not args.skip_visualization:
            print(f"📊 Visualizations saved to: {args.output_dir}")
        return 0
    else:
        print("❌ Some tests failed. Please check the error messages above.")
        return 1


if __name__ == '__main__':
    exit(main()) 