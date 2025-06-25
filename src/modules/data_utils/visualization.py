"""
Data visualization utilities for Khmer digits OCR.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import matplotlib.font_manager as fm
import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image
from typing import List, Dict, Tuple, Optional, Union
import seaborn as sns
from collections import Counter
import warnings
import os
from pathlib import Path

# Import the new font utilities
from .font_utils import safe_khmer_text as _safe_khmer_text


class DataVisualizer:
    """
    Comprehensive data visualization utilities for the Khmer digits dataset.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize the data visualizer.
        
        Args:
            figsize: Default figure size for plots
        """
        self.figsize = figsize
        plt.style.use('default')
        sns.set_palette("husl")
    
    def plot_samples(self, 
                    dataset,
                    num_samples: int = 16,
                    cols: int = 4,
                    show_metadata: bool = True,
                    save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot a grid of sample images with their labels.
        
        Args:
            dataset: KhmerDigitsDataset instance
            num_samples: Number of samples to display
            cols: Number of columns in the grid
            show_metadata: Whether to show metadata information
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure
        """
        rows = (num_samples + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        
        if rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        
        for i in range(num_samples):
            row, col = i // cols, i % cols
            ax = axes[row, col]
            
            if i < len(dataset):
                # Get sample
                image, label_tensor, metadata = dataset[i]
                
                # Handle different image formats
                if torch.is_tensor(image):
                    if image.shape[0] == 3:  # CHW format
                        # Convert to HWC and denormalize if needed
                        img_np = image.permute(1, 2, 0).numpy()
                        if img_np.min() < 0:  # Likely normalized
                            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
                    else:  # HWC format
                        img_np = image.numpy()
                else:
                    img_np = np.array(image)
                
                # Display image
                ax.imshow(img_np, cmap='gray' if len(img_np.shape) == 2 else None)
                
                # Add label and metadata with safe Khmer text
                label_text, font_props = _safe_khmer_text(
                    metadata['original_label'], 
                    fallback_format="Sample {}", 
                    index=i
                )
                
                title_lines = [f"Label: {label_text}"]
                if show_metadata:
                    title_lines.extend([
                        f"Font: {metadata['font'][:15]}",
                        f"Size: {metadata['font_size']}, Aug: {metadata['augmented']}"
                    ])
                
                ax.set_title('\n'.join(title_lines), fontsize=10, **font_props)
            else:
                ax.axis('off')
            
            ax.set_xticks([])
            ax.set_yticks([])
        
        # Remove empty subplots
        for i in range(num_samples, rows * cols):
            row, col = i // cols, i % cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_dataset_statistics(self,
                               dataset,
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot comprehensive dataset statistics.
        
        Args:
            dataset: KhmerDigitsDataset instance
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure
        """
        stats = dataset.get_dataset_stats()
        
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 3, figure=fig)
        
        # 1. Sequence Length Distribution
        ax1 = fig.add_subplot(gs[0, 0])
        sequence_lengths = [sample['sequence_length'] for sample in dataset.samples]
        ax1.hist(sequence_lengths, bins=range(1, max(sequence_lengths) + 2), 
                alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Sequence Length')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Sequence Length Distribution')
        ax1.grid(True, alpha=0.3)
        
        # 2. Font Distribution
        ax2 = fig.add_subplot(gs[0, 1])
        fonts = list(stats['font_distribution'].keys())
        font_counts = list(stats['font_distribution'].values())
        
        bars = ax2.bar(range(len(fonts)), font_counts, alpha=0.7)
        ax2.set_xlabel('Font')
        ax2.set_ylabel('Number of Samples')
        ax2.set_title('Font Distribution')
        ax2.set_xticks(range(len(fonts)))
        ax2.set_xticklabels([f[:10] + '...' if len(f) > 10 else f for f in fonts], 
                           rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        # Add count labels on bars
        for bar, count in zip(bars, font_counts):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(count), ha='center', va='bottom', fontsize=9)
        
        # 3. Character Frequency
        ax3 = fig.add_subplot(gs[0, 2])
        char_freq = self._calculate_character_frequency(dataset)
        chars = list(char_freq.keys())
        frequencies = list(char_freq.values())
        
        # Use safe text for character labels
        safe_chars = []
        for i, char in enumerate(chars):
            safe_char, _ = _safe_khmer_text(char, fallback_format="Char{}", index=i)
            safe_chars.append(safe_char)
        
        bars = ax3.bar(range(len(chars)), frequencies, alpha=0.7)
        ax3.set_xlabel('Khmer Digits')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Character Frequency Distribution')
        ax3.set_xticks(range(len(chars)))
        ax3.set_xticklabels(safe_chars)
        ax3.grid(True, alpha=0.3)
        
        # 4. Augmentation Rate
        ax4 = fig.add_subplot(gs[1, 0])
        aug_rate = stats['augmentation_rate']
        labels = ['Augmented', 'Original']
        sizes = [aug_rate, 1 - aug_rate]
        colors = ['lightcoral', 'lightblue']
        
        ax4.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax4.set_title('Augmentation Distribution')
        
        # 5. Sample Image Examples
        ax5 = fig.add_subplot(gs[1, 1:])
        self._plot_sample_grid(dataset, ax5, num_samples=6, title="Sample Images")
        
        # 6. Statistics Summary
        ax6 = fig.add_subplot(gs[2, :])
        ax6.axis('off')
        
        summary_text = f"""
Dataset Summary:
• Total Samples: {stats['total_samples']:,}
• Sequence Length: {stats['sequence_length_stats']['min']}-{stats['sequence_length_stats']['max']} (avg: {stats['sequence_length_stats']['mean']:.1f})
• Fonts Used: {len(stats['font_distribution'])}
• Character Set Size: {stats['character_set_size']}
• Augmentation Rate: {aug_rate:.1%}
• Max Sequence Length: {stats['max_sequence_length']}
        """.strip()
        
        ax6.text(0.1, 0.5, summary_text, fontsize=12, va='center', 
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.7))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_batch_samples(self,
                          dataloader: DataLoader,
                          num_batches: int = 1,
                          save_path: Optional[str] = None) -> plt.Figure:
        """
        Visualize samples from data loader batches.
        
        Args:
            dataloader: DataLoader instance
            num_batches: Number of batches to visualize
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(num_batches, 8, figsize=(16, 2 * num_batches))
        if num_batches == 1:
            axes = axes.reshape(1, -1)
        
        for batch_idx, (images, labels, metadata_list) in enumerate(dataloader):
            if batch_idx >= num_batches:
                break
            
            for i in range(min(8, images.shape[0])):
                ax = axes[batch_idx, i]
                
                # Get image
                img = images[i]
                if img.shape[0] == 3:  # CHW format
                    img = img.permute(1, 2, 0)
                
                # Denormalize if needed
                img_np = img.numpy()
                if img_np.min() < 0:
                    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
                
                ax.imshow(img_np)
                
                # Decode label
                label_tensor = labels[i]
                original_label = metadata_list[i]['original_label']
                
                # Use safe Khmer text
                label_text, font_props = _safe_khmer_text(
                    original_label,
                    fallback_format="Sample {}",
                    index=i
                )
                ax.set_title(f"Label: {label_text}", fontsize=10, **font_props)
                ax.set_xticks([])
                ax.set_yticks([])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_transforms_comparison(self,
                                 dataset,
                                 sample_idx: int,
                                 transforms_list: List[Tuple[str, callable]],
                                 save_path: Optional[str] = None) -> plt.Figure:
        """
        Compare different transforms applied to the same sample.
        
        Args:
            dataset: KhmerDigitsDataset instance
            sample_idx: Index of the sample to transform
            transforms_list: List of (name, transform) tuples
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure
        """
        # Get original image path
        sample = dataset.samples[sample_idx]
        image_path = sample['image_path']
        original_image = Image.open(image_path).convert('RGB')
        
        num_transforms = len(transforms_list)
        fig, axes = plt.subplots(1, num_transforms, figsize=(4 * num_transforms, 4))
        
        if num_transforms == 1:
            axes = [axes]
        
        for i, (name, transform) in enumerate(transforms_list):
            ax = axes[i]
            
            # Apply transform
            transformed = transform(original_image)
            
            # Handle tensor format
            if torch.is_tensor(transformed):
                if transformed.shape[0] == 3:  # CHW
                    img_np = transformed.permute(1, 2, 0).numpy()
                else:
                    img_np = transformed.numpy()
                
                # Denormalize if needed
                if img_np.min() < 0:
                    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
            else:
                img_np = np.array(transformed)
            
            ax.imshow(img_np)
            
            # Use safe Khmer text
            label_text, font_props = _safe_khmer_text(
                sample['label'],
                fallback_format="Transform {}",
                index=i
            )
            ax.set_title(f"{name}\nLabel: {label_text}", fontsize=12, **font_props)
            ax.set_xticks([])
            ax.set_yticks([])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def _calculate_character_frequency(self, dataset) -> Dict[str, int]:
        """Calculate frequency of each Khmer digit in the dataset."""
        char_counts = Counter()
        
        for sample in dataset.samples:
            label = sample['label']
            for char in label:
                char_counts[char] += 1
        
        return dict(char_counts)
    
    def _plot_sample_grid(self, dataset, ax, num_samples: int = 6, title: str = ""):
        """Plot a grid of samples in a single axis."""
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')
        
        # Create a grid of subplots within this axis
        for i in range(num_samples):
            if i >= len(dataset):
                break
            
            image, _, metadata = dataset[i]
            
            # Handle tensor format
            if torch.is_tensor(image):
                if image.shape[0] == 3:  # CHW
                    img_np = image.permute(1, 2, 0).numpy()
                else:
                    img_np = image.numpy()
                
                if img_np.min() < 0:  # Denormalize
                    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
            else:
                img_np = np.array(image)
            
            # Calculate position for this sample
            x_pos = (i % 3) * 0.33
            y_pos = 0.7 - (i // 3) * 0.4
            
            # Create inset axes
            from mpl_toolkits.axes_grid1.inset_locator import inset_axes
            inset_ax = ax.inset_axes([x_pos, y_pos, 0.3, 0.3])
            inset_ax.imshow(img_np)
            
            # Use safe Khmer text
            label_text, font_props = _safe_khmer_text(
                metadata['original_label'],
                fallback_format="Grid {}",
                index=i
            )
            inset_ax.set_title(label_text, fontsize=10, **font_props)
            inset_ax.set_xticks([])
            inset_ax.set_yticks([])


def plot_samples(dataset, 
                num_samples: int = 16,
                cols: int = 4,
                show_metadata: bool = True,
                save_path: Optional[str] = None) -> plt.Figure:
    """
    Convenience function to plot dataset samples.
    
    Args:
        dataset: KhmerDigitsDataset instance
        num_samples: Number of samples to display
        cols: Number of columns in the grid
        show_metadata: Whether to show metadata information
        save_path: Optional path to save the figure
        
    Returns:
        Matplotlib figure
    """
    visualizer = DataVisualizer()
    return visualizer.plot_samples(dataset, num_samples, cols, show_metadata, save_path)


def plot_dataset_stats(dataset, save_path: Optional[str] = None) -> plt.Figure:
    """
    Convenience function to plot dataset statistics.
    
    Args:
        dataset: KhmerDigitsDataset instance
        save_path: Optional path to save the figure
        
    Returns:
        Matplotlib figure
    """
    visualizer = DataVisualizer()
    return visualizer.plot_dataset_statistics(dataset, save_path)


def plot_training_progress(train_losses: List[float],
                         val_losses: List[float],
                         train_accuracies: List[float],
                         val_accuracies: List[float],
                         save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot training progress curves.
    
    Args:
        train_losses: Training loss values
        val_losses: Validation loss values
        train_accuracies: Training accuracy values
        val_accuracies: Validation accuracy values
        save_path: Optional path to save the figure
        
    Returns:
        Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot losses
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracies
    ax2.plot(epochs, train_accuracies, 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, val_accuracies, 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig 