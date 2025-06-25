"""
Dataset analysis and validation utilities for Khmer digits OCR.
"""

import os
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import torch
from torch.utils.data import DataLoader


class DatasetAnalyzer:
    """
    Comprehensive analysis utilities for the Khmer digits dataset.
    
    Provides detailed statistics, quality metrics, and validation
    for synthetic training data.
    """
    
    def __init__(self, dataset):
        """
        Initialize the dataset analyzer.
        
        Args:
            dataset: KhmerDigitsDataset instance
        """
        self.dataset = dataset
        self.metadata = dataset.metadata
        self.samples = dataset.samples
        
    def analyze_sequence_patterns(self) -> Dict[str, Any]:
        """
        Analyze digit sequence patterns and distributions.
        
        Returns:
            Dictionary with sequence pattern analysis
        """
        sequences = [sample['label'] for sample in self.samples]
        
        # Sequence length distribution
        lengths = [len(seq) for seq in sequences]
        length_counts = Counter(lengths)
        
        # Digit frequency analysis
        all_digits = ''.join(sequences)
        digit_counts = Counter(all_digits)
        
        # Digit position analysis
        position_digit_counts = defaultdict(Counter)
        for seq in sequences:
            for pos, digit in enumerate(seq):
                position_digit_counts[pos][digit] += 1
        
        # Most common sequences
        sequence_counts = Counter(sequences)
        most_common_sequences = [(seq, count) for seq, count in sequence_counts.most_common(20)]
        
        # Digit pair analysis
        digit_pairs = []
        for seq in sequences:
            for i in range(len(seq) - 1):
                digit_pairs.append(seq[i:i+2])
        pair_counts = Counter(digit_pairs)
        
        return {
            'length_distribution': dict(length_counts),
            'digit_frequency': dict(digit_counts),
            'position_digit_frequency': {str(pos): dict(counts) for pos, counts in position_digit_counts.items()},
            'most_common_sequences': most_common_sequences,
            'digit_pair_frequency': {pair: count for pair, count in pair_counts.most_common(50)},
            'total_sequences': len(sequences),
            'unique_sequences': len(set(sequences)),
            'average_length': float(np.mean(lengths)),
            'length_std': float(np.std(lengths))
        }
    
    def analyze_visual_properties(self) -> Dict[str, Any]:
        """
        Analyze visual properties of the dataset images.
        
        Returns:
            Dictionary with visual analysis results
        """
        # Sample a subset for analysis (to avoid loading all images)
        sample_indices = np.random.choice(
            len(self.samples), 
            min(1000, len(self.samples)), 
            replace=False
        )
        
        image_stats = {
            'brightness': [],
            'contrast': [],
            'sizes': [],
            'aspect_ratios': []
        }
        
        font_stats = defaultdict(list)
        
        for idx in sample_indices:
            sample = self.samples[idx]
            
            try:
                # Load image
                image_path = sample['image_path']
                if not os.path.isabs(image_path):
                    # The paths in metadata are already relative to project root
                    image_path = image_path.replace('\\', '/')
                
                image = Image.open(image_path).convert('RGB')
                img_array = np.array(image)
                
                # Calculate brightness (average pixel intensity)
                brightness = np.mean(img_array)
                image_stats['brightness'].append(brightness)
                
                # Calculate contrast (standard deviation of pixel intensities)
                contrast = np.std(img_array)
                image_stats['contrast'].append(contrast)
                
                # Image size
                image_stats['sizes'].append(image.size)
                image_stats['aspect_ratios'].append(image.size[0] / image.size[1])
                
                # Font-specific stats
                font_name = sample.get('font', 'unknown')
                font_stats[font_name].append({
                    'brightness': brightness,
                    'contrast': contrast,
                    'font_size': sample.get('font_size', 0)
                })
                
            except Exception as e:
                print(f"Warning: Could not analyze image {image_path}: {e}")
                continue
        
        # Calculate statistics
        brightness_stats = {
            'mean': float(np.mean(image_stats['brightness'])),
            'std': float(np.std(image_stats['brightness'])),
            'min': float(np.min(image_stats['brightness'])),
            'max': float(np.max(image_stats['brightness']))
        }
        
        contrast_stats = {
            'mean': float(np.mean(image_stats['contrast'])),
            'std': float(np.std(image_stats['contrast'])),
            'min': float(np.min(image_stats['contrast'])),
            'max': float(np.max(image_stats['contrast']))
        }
        
        # Font-specific analysis
        font_analysis = {}
        for font, stats_list in font_stats.items():
            if stats_list:
                font_analysis[font] = {
                    'count': len(stats_list),
                    'avg_brightness': np.mean([s['brightness'] for s in stats_list]),
                    'avg_contrast': np.mean([s['contrast'] for s in stats_list]),
                    'avg_font_size': np.mean([s['font_size'] for s in stats_list if s['font_size'] > 0])
                }
        
        return {
            'brightness_stats': brightness_stats,
            'contrast_stats': contrast_stats,
            'image_sizes': {str(size): count for size, count in Counter(image_stats['sizes']).items()},
            'aspect_ratio_stats': {
                'mean': float(np.mean(image_stats['aspect_ratios'])),
                'std': float(np.std(image_stats['aspect_ratios']))
            },
            'font_analysis': font_analysis,
            'samples_analyzed': len(sample_indices)
        }
    
    def analyze_augmentation_impact(self) -> Dict[str, Any]:
        """
        Analyze the impact and distribution of augmentations.
        
        Returns:
            Dictionary with augmentation analysis
        """
        augmented_samples = [s for s in self.samples if s.get('augmented', False)]
        original_samples = [s for s in self.samples if not s.get('augmented', False)]
        
        augmentation_rate = len(augmented_samples) / len(self.samples)
        
        # Font distribution comparison
        aug_fonts = Counter([s.get('font', 'unknown') for s in augmented_samples])
        orig_fonts = Counter([s.get('font', 'unknown') for s in original_samples])
        
        # Sequence length comparison
        aug_lengths = [len(s['label']) for s in augmented_samples]
        orig_lengths = [len(s['label']) for s in original_samples]
        
        return {
            'augmentation_rate': augmentation_rate,
            'augmented_count': len(augmented_samples),
            'original_count': len(original_samples),
            'font_distribution': {
                'augmented': dict(aug_fonts),
                'original': dict(orig_fonts)
            },
            'length_distribution': {
                'augmented': {
                    'mean': float(np.mean(aug_lengths)) if aug_lengths else 0.0,
                    'std': float(np.std(aug_lengths)) if aug_lengths else 0.0
                },
                'original': {
                    'mean': float(np.mean(orig_lengths)) if orig_lengths else 0.0,
                    'std': float(np.std(orig_lengths)) if orig_lengths else 0.0
                }
            }
        }
    
    def validate_data_quality(self) -> Dict[str, Any]:
        """
        Validate data quality and identify potential issues.
        
        Returns:
            Dictionary with validation results and issues
        """
        issues = []
        warnings = []
        
        # Check for missing files
        missing_files = []
        for sample in self.samples:
            image_path = sample['image_path']
            if not os.path.isabs(image_path):
                metadata_dir = os.path.dirname(self.dataset.metadata_path)
                image_path = os.path.join(metadata_dir, image_path)
            
            if not os.path.exists(image_path):
                missing_files.append(image_path)
        
        if missing_files:
            issues.append(f"Missing {len(missing_files)} image files")
        
        # Check label consistency
        char_to_idx, _ = self.dataset.get_character_mappings()
        unknown_chars = set()
        for sample in self.samples:
            label = sample['label']
            for char in label:
                if char not in char_to_idx:
                    unknown_chars.add(char)
        
        if unknown_chars:
            issues.append(f"Unknown characters found: {unknown_chars}")
        
        # Check sequence length distribution
        lengths = [len(sample['label']) for sample in self.samples]
        if max(lengths) > self.dataset.max_sequence_length:
            warnings.append(f"Some sequences exceed max length ({self.dataset.max_sequence_length})")
        
        # Check font distribution balance
        fonts = [sample.get('font', 'unknown') for sample in self.samples]
        font_counts = Counter(fonts)
        if font_counts:
            min_count = min(font_counts.values())
            max_count = max(font_counts.values())
            imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
            
            if imbalance_ratio > 5:  # More than 5x difference
                warnings.append(f"Font distribution imbalanced (ratio: {imbalance_ratio:.1f})")
        
        # Check for duplicate samples
        labels = [sample['label'] for sample in self.samples]
        unique_labels = set(labels)
        duplicate_rate = 1 - (len(unique_labels) / len(labels))
        
        if duplicate_rate > 0.5:  # More than 50% duplicates
            warnings.append(f"High duplicate rate: {duplicate_rate:.1%}")
        
        return {
            'is_valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'missing_files_count': len(missing_files),
            'unknown_characters': list(unknown_chars),
            'font_imbalance_ratio': imbalance_ratio if 'imbalance_ratio' in locals() else 1.0,
            'duplicate_rate': duplicate_rate,
            'total_samples_checked': len(self.samples)
        }
    
    def generate_comprehensive_report(self, save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a comprehensive analysis report.
        
        Args:
            save_path: Optional path to save the report as JSON
            
        Returns:
            Complete analysis report dictionary
        """
        report = {
            'dataset_overview': self.dataset.get_dataset_stats(),
            'sequence_analysis': self.analyze_sequence_patterns(),
            'visual_analysis': self.analyze_visual_properties(),
            'augmentation_analysis': self.analyze_augmentation_impact(),
            'quality_validation': self.validate_data_quality(),
            'analysis_timestamp': pd.Timestamp.now().isoformat()
        }
        
        if save_path:
            import json
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"Analysis report saved to: {save_path}")
        
        return report
    
    def create_analysis_plots(self, save_dir: Optional[str] = None) -> Dict[str, plt.Figure]:
        """
        Create comprehensive analysis plots.
        
        Args:
            save_dir: Optional directory to save plots
            
        Returns:
            Dictionary of plot names to matplotlib figures
        """
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        figures = {}
        
        # 1. Sequence pattern analysis
        seq_analysis = self.analyze_sequence_patterns()
        
        fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Length distribution
        lengths = list(seq_analysis['length_distribution'].keys())
        counts = list(seq_analysis['length_distribution'].values())
        ax1.bar(lengths, counts, alpha=0.7)
        ax1.set_xlabel('Sequence Length')
        ax1.set_ylabel('Count')
        ax1.set_title('Sequence Length Distribution')
        ax1.grid(True, alpha=0.3)
        
        # Digit frequency
        digits = list(seq_analysis['digit_frequency'].keys())
        digit_counts = list(seq_analysis['digit_frequency'].values())
        ax2.bar(digits, digit_counts, alpha=0.7)
        ax2.set_xlabel('Khmer Digits')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Digit Frequency Distribution')
        ax2.grid(True, alpha=0.3)
        
        # Most common sequences (top 10)
        common_seqs = seq_analysis['most_common_sequences'][:10]
        if common_seqs:
            seqs, seq_counts = zip(*common_seqs)
            ax3.barh(range(len(seqs)), seq_counts, alpha=0.7)
            ax3.set_yticks(range(len(seqs)))
            ax3.set_yticklabels(seqs)
            ax3.set_xlabel('Count')
            ax3.set_title('Most Common Sequences (Top 10)')
            ax3.grid(True, alpha=0.3)
        
        # Digit pair frequency (top 15)
        pairs = list(seq_analysis['digit_pair_frequency'].items())[:15]
        if pairs:
            pair_names, pair_counts = zip(*pairs)
            ax4.barh(range(len(pair_names)), pair_counts, alpha=0.7)
            ax4.set_yticks(range(len(pair_names)))
            ax4.set_yticklabels(pair_names)
            ax4.set_xlabel('Count')
            ax4.set_title('Most Common Digit Pairs (Top 15)')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        figures['sequence_analysis'] = fig1
        
        if save_dir:
            fig1.savefig(os.path.join(save_dir, 'sequence_analysis.png'), dpi=150, bbox_inches='tight')
        
        # 2. Visual properties analysis
        visual_analysis = self.analyze_visual_properties()
        
        fig2, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Font analysis
        font_data = visual_analysis['font_analysis']
        if font_data:
            fonts = list(font_data.keys())
            brightnesses = [font_data[f]['avg_brightness'] for f in fonts]
            contrasts = [font_data[f]['avg_contrast'] for f in fonts]
            counts = [font_data[f]['count'] for f in fonts]
            
            # Font brightness
            ax1.bar(range(len(fonts)), brightnesses, alpha=0.7)
            ax1.set_xticks(range(len(fonts)))
            ax1.set_xticklabels([f[:10] for f in fonts], rotation=45, ha='right')
            ax1.set_ylabel('Average Brightness')
            ax1.set_title('Average Brightness by Font')
            ax1.grid(True, alpha=0.3)
            
            # Font contrast
            ax2.bar(range(len(fonts)), contrasts, alpha=0.7)
            ax2.set_xticks(range(len(fonts)))
            ax2.set_xticklabels([f[:10] for f in fonts], rotation=45, ha='right')
            ax2.set_ylabel('Average Contrast')
            ax2.set_title('Average Contrast by Font')
            ax2.grid(True, alpha=0.3)
            
            # Sample counts per font
            ax3.bar(range(len(fonts)), counts, alpha=0.7)
            ax3.set_xticks(range(len(fonts)))
            ax3.set_xticklabels([f[:10] for f in fonts], rotation=45, ha='right')
            ax3.set_ylabel('Sample Count')
            ax3.set_title('Samples per Font')
            ax3.grid(True, alpha=0.3)
        
        # Brightness and contrast correlation
        brightness_stats = visual_analysis['brightness_stats']
        contrast_stats = visual_analysis['contrast_stats']
        
        ax4.text(0.1, 0.7, f"Brightness: {brightness_stats['mean']:.1f} ± {brightness_stats['std']:.1f}", 
                transform=ax4.transAxes, fontsize=12)
        ax4.text(0.1, 0.5, f"Contrast: {contrast_stats['mean']:.1f} ± {contrast_stats['std']:.1f}", 
                transform=ax4.transAxes, fontsize=12)
        ax4.text(0.1, 0.3, f"Samples analyzed: {visual_analysis['samples_analyzed']}", 
                transform=ax4.transAxes, fontsize=12)
        ax4.set_title('Visual Properties Summary')
        ax4.axis('off')
        
        plt.tight_layout()
        figures['visual_analysis'] = fig2
        
        if save_dir:
            fig2.savefig(os.path.join(save_dir, 'visual_analysis.png'), dpi=150, bbox_inches='tight')
        
        return figures


def calculate_dataset_metrics(dataset) -> Dict[str, float]:
    """
    Calculate key metrics for dataset quality assessment.
    
    Args:
        dataset: KhmerDigitsDataset instance
        
    Returns:
        Dictionary of calculated metrics
    """
    analyzer = DatasetAnalyzer(dataset)
    
    # Get basic stats
    stats = dataset.get_dataset_stats()
    
    # Calculate diversity metrics
    seq_analysis = analyzer.analyze_sequence_patterns()
    visual_analysis = analyzer.analyze_visual_properties()
    
    # Diversity score based on unique sequences ratio
    diversity_score = seq_analysis['unique_sequences'] / seq_analysis['total_sequences']
    
    # Balance score based on font distribution
    font_counts = list(stats['font_distribution'].values())
    if font_counts:
        font_balance = min(font_counts) / max(font_counts)
    else:
        font_balance = 1.0
    
    # Character coverage score
    total_possible_chars = 10  # 10 Khmer digits
    unique_chars = len(seq_analysis['digit_frequency'])
    char_coverage = unique_chars / total_possible_chars
    
    return {
        'total_samples': stats['total_samples'],
        'diversity_score': diversity_score,
        'font_balance_score': font_balance,
        'character_coverage': char_coverage,
        'augmentation_rate': stats['augmentation_rate'],
        'avg_sequence_length': stats['sequence_length_stats']['mean'],
        'samples_analyzed': visual_analysis['samples_analyzed']
    }


def validate_dataset_quality(dataset, 
                           min_samples: int = 1000,
                           min_diversity: float = 0.7,
                           min_font_balance: float = 0.2) -> Tuple[bool, List[str]]:
    """
    Validate dataset quality against minimum requirements.
    
    Args:
        dataset: KhmerDigitsDataset instance
        min_samples: Minimum number of samples required
        min_diversity: Minimum diversity score required
        min_font_balance: Minimum font balance score required
        
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    analyzer = DatasetAnalyzer(dataset)
    validation = analyzer.validate_data_quality()
    metrics = calculate_dataset_metrics(dataset)
    
    issues = validation['issues'].copy()
    
    # Check metrics against thresholds
    if metrics['total_samples'] < min_samples:
        issues.append(f"Insufficient samples: {metrics['total_samples']} < {min_samples}")
    
    if metrics['diversity_score'] < min_diversity:
        issues.append(f"Low diversity: {metrics['diversity_score']:.2f} < {min_diversity}")
    
    if metrics['font_balance_score'] < min_font_balance:
        issues.append(f"Poor font balance: {metrics['font_balance_score']:.2f} < {min_font_balance}")
    
    if metrics['character_coverage'] < 1.0:
        issues.append(f"Incomplete character coverage: {metrics['character_coverage']:.2f}")
    
    is_valid = len(issues) == 0
    
    return is_valid, issues 