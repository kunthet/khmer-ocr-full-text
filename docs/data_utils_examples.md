# Data Utils Module - Usage Examples

This document provides complete, working examples for common tasks using the data_utils module.

## Example 1: Basic Dataset Exploration

```python
#!/usr/bin/env python3
"""
Example 1: Explore dataset contents and statistics
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.modules.data_utils import KhmerDigitsDataset, plot_samples, plot_dataset_stats

def explore_dataset():
    # Load dataset
    dataset = KhmerDigitsDataset(
        metadata_path='generated_data/metadata.yaml',
        split='train',
        transform=None
    )
    
    print(f"Dataset Overview:")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Split: train")
    
    # Character mappings
    char_to_idx, idx_to_char = dataset.get_character_mappings()
    print(f"  Character set size: {len(char_to_idx)}")
    print(f"  Characters: {list(char_to_idx.keys())}")
    
    # Sample inspection
    print(f"\nSample Inspection:")
    for i in range(5):
        image, label_tensor, metadata = dataset[i]
        print(f"  Sample {i}: '{metadata['original_label']}' "
              f"(font: {metadata['font']}, aug: {metadata['augmented']})")
    
    # Dataset statistics
    stats = dataset.get_dataset_stats()
    print(f"\nDataset Statistics:")
    print(f"  Sequence length range: {stats['sequence_length_stats']['min']}-{stats['sequence_length_stats']['max']}")
    print(f"  Average length: {stats['sequence_length_stats']['mean']:.1f}")
    print(f"  Fonts used: {len(stats['font_distribution'])}")
    print(f"  Augmentation rate: {stats['augmentation_rate']:.1%}")
    
    # Create visualizations
    plot_samples(dataset, num_samples=16, save_path='exploration_samples.png')
    plot_dataset_stats(dataset, save_path='exploration_stats.png')
    
    print(f"\n✅ Visualizations saved: exploration_samples.png, exploration_stats.png")

if __name__ == '__main__':
    explore_dataset()
```

## Example 2: Training Pipeline Setup

```python
#!/usr/bin/env python3
"""
Example 2: Complete training pipeline setup
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.modules.data_utils import (
    create_data_loaders, get_train_transforms, get_val_transforms
)

def setup_training_pipeline():
    # Configuration
    config = {
        'metadata_path': 'generated_data/metadata.yaml',
        'batch_size': 32,
        'image_size': (128, 64),
        'augmentation_strength': 0.3,
        'num_workers': 4,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print(f"Training Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Create transforms
    train_transform = get_train_transforms(
        image_size=config['image_size'],
        augmentation_strength=config['augmentation_strength'],
        normalize=True
    )
    
    val_transform = get_val_transforms(
        image_size=config['image_size'],
        normalize=True
    )
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        metadata_path=config['metadata_path'],
        train_transform=train_transform,
        val_transform=val_transform,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        shuffle_train=True
    )
    
    print(f"\nDataLoader Setup:")
    print(f"  Training batches: {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")
    
    # Test batch loading
    print(f"\nTesting Batch Loading:")
    train_iter = iter(train_loader)
    val_iter = iter(val_loader)
    
    # Load training batch
    train_images, train_labels, train_metadata = next(train_iter)
    print(f"  Training batch - Images: {train_images.shape}, Labels: {train_labels.shape}")
    print(f"  Sample labels: {[m['original_label'] for m in train_metadata[:3]]}")
    
    # Load validation batch
    val_images, val_labels, val_metadata = next(val_iter)
    print(f"  Validation batch - Images: {val_images.shape}, Labels: {val_labels.shape}")
    
    # Test character encoding/decoding
    print(f"\nTesting Character Encoding:")
    dataset = train_loader.dataset
    for i in range(3):
        original = train_metadata[i]['original_label']
        encoded = train_labels[i]
        decoded = dataset._decode_label(encoded)
        print(f"  '{original}' -> {encoded[:len(original)+1].tolist()} -> '{decoded}'")
        assert original == decoded, f"Encoding mismatch: {original} != {decoded}"
    
    print(f"\n✅ Training pipeline setup complete!")
    
    return train_loader, val_loader, config

# Example training function
def example_training_step(train_loader, val_loader, config):
    """Example training step (without actual model)"""
    device = torch.device(config['device'])
    
    # Simulate training loop
    print(f"\nSimulating Training Loop:")
    for epoch in range(2):  # 2 epochs for demo
        print(f"  Epoch {epoch + 1}:")
        
        # Training
        total_samples = 0
        for batch_idx, (images, labels, metadata) in enumerate(train_loader):
            total_samples += images.size(0)
            
            # Move to device (simulated)
            # images = images.to(device)
            # labels = labels.to(device)
            
            if batch_idx >= 3:  # Process first 4 batches for demo
                break
        
        print(f"    Processed {total_samples} training samples")
        
        # Validation
        total_val_samples = 0
        for batch_idx, (images, labels, metadata) in enumerate(val_loader):
            total_val_samples += images.size(0)
            if batch_idx >= 1:  # Process first 2 validation batches
                break
        
        print(f"    Processed {total_val_samples} validation samples")
    
    print(f"  ✅ Training simulation complete!")

if __name__ == '__main__':
    train_loader, val_loader, config = setup_training_pipeline()
    example_training_step(train_loader, val_loader, config)
```

## Example 3: Dataset Analysis and Quality Validation

```python
#!/usr/bin/env python3
"""
Example 3: Comprehensive dataset analysis
"""
from src.modules.data_utils import (
    KhmerDigitsDataset, DatasetAnalyzer,
    calculate_dataset_metrics, validate_dataset_quality
)
import json

def analyze_dataset():
    # Load complete dataset
    dataset = KhmerDigitsDataset(
        metadata_path='generated_data/metadata.yaml',
        split='all',  # Analyze entire dataset
        transform=None
    )
    
    print(f"Dataset Analysis Report")
    print(f"=" * 50)
    
    # Basic metrics
    print(f"\n1. Basic Metrics:")
    metrics = calculate_dataset_metrics(dataset)
    print(f"   Total samples: {metrics['total_samples']:,}")
    print(f"   Diversity score: {metrics['diversity_score']:.3f}")
    print(f"   Font balance score: {metrics['font_balance_score']:.3f}")
    print(f"   Character coverage: {metrics['character_coverage']:.3f}")
    print(f"   Augmentation rate: {metrics['augmentation_rate']:.1%}")
    print(f"   Average sequence length: {metrics['avg_sequence_length']:.1f}")
    
    # Quality validation
    print(f"\n2. Quality Validation:")
    is_valid, issues = validate_dataset_quality(
        dataset,
        min_samples=1000,
        min_diversity=0.7,
        min_font_balance=0.2
    )
    
    print(f"   Dataset quality: {'✅ PASS' if is_valid else '❌ FAIL'}")
    if issues:
        print(f"   Issues found:")
        for issue in issues:
            print(f"     - {issue}")
    else:
        print(f"   No issues found!")
    
    # Detailed analysis
    print(f"\n3. Detailed Analysis:")
    analyzer = DatasetAnalyzer(dataset)
    
    # Sequence patterns
    seq_analysis = analyzer.analyze_sequence_patterns()
    print(f"   Sequence Analysis:")
    print(f"     Total sequences: {seq_analysis['total_sequences']:,}")
    print(f"     Unique sequences: {seq_analysis['unique_sequences']:,}")
    print(f"     Average length: {seq_analysis['average_length']:.1f}")
    print(f"     Length std: {seq_analysis['length_std']:.1f}")
    
    # Most common sequences
    print(f"     Most common sequences:")
    for seq, count in seq_analysis['most_common_sequences'][:5]:
        print(f"       '{seq}': {count} times")
    
    # Visual properties
    print(f"\n   Visual Analysis:")
    visual_analysis = analyzer.analyze_visual_properties()
    brightness = visual_analysis['brightness_stats']
    contrast = visual_analysis['contrast_stats']
    print(f"     Samples analyzed: {visual_analysis['samples_analyzed']}")
    print(f"     Average brightness: {brightness['mean']:.1f} ± {brightness['std']:.1f}")
    print(f"     Average contrast: {contrast['mean']:.1f} ± {contrast['std']:.1f}")
    
    # Font analysis
    print(f"     Font distribution:")
    for font, analysis in visual_analysis['font_analysis'].items():
        print(f"       {font}: {analysis['count']} samples")
    
    # Augmentation analysis
    print(f"\n   Augmentation Analysis:")
    aug_analysis = analyzer.analyze_augmentation_impact()
    print(f"     Augmentation rate: {aug_analysis['augmentation_rate']:.1%}")
    print(f"     Augmented samples: {aug_analysis['augmented_count']:,}")
    print(f"     Original samples: {aug_analysis['original_count']:,}")
    
    # Generate comprehensive report
    print(f"\n4. Generating Reports:")
    report = analyzer.generate_comprehensive_report('dataset_analysis_report.json')
    print(f"   ✅ Comprehensive report saved: dataset_analysis_report.json")
    
    # Create analysis plots
    plots = analyzer.create_analysis_plots(save_dir='analysis_plots')
    print(f"   ✅ Analysis plots created: {len(plots)} plots in analysis_plots/")
    
    # Quality summary
    print(f"\n5. Quality Summary:")
    if metrics['diversity_score'] >= 0.7:
        print(f"   ✅ Good diversity ({metrics['diversity_score']:.1%})")
    else:
        print(f"   ⚠️  Low diversity ({metrics['diversity_score']:.1%})")
    
    if metrics['font_balance_score'] >= 0.5:
        print(f"   ✅ Well-balanced fonts ({metrics['font_balance_score']:.1%})")
    else:
        print(f"   ⚠️  Imbalanced fonts ({metrics['font_balance_score']:.1%})")
    
    if metrics['character_coverage'] >= 1.0:
        print(f"   ✅ Complete character coverage")
    else:
        print(f"   ⚠️  Incomplete character coverage ({metrics['character_coverage']:.1%})")
    
    print(f"\n✅ Dataset analysis complete!")

if __name__ == '__main__':
    analyze_dataset()
```

## Example 4: Custom Preprocessing Pipeline

```python
#!/usr/bin/env python3
"""
Example 4: Custom preprocessing and augmentation
"""
import torch
import torchvision.transforms as transforms
from PIL import Image
from src.modules.data_utils import KhmerDigitsDataset
from src.modules.data_utils.preprocessing import (
    ImagePreprocessor, AddGaussianNoise, RandomBlur,
    TestTimeAugmentation, create_preprocessing_pipeline
)

def custom_preprocessing_example():
    print("Custom Preprocessing Pipeline Example")
    print("=" * 40)
    
    # 1. Basic custom preprocessor
    print("\n1. Basic Custom Preprocessor:")
    preprocessor = ImagePreprocessor(
        image_size=(128, 64),
        normalize=True,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    )
    
    # Different transform types
    base_transform = preprocessor.get_base_transforms()
    light_aug = preprocessor.get_train_transforms(augmentation_strength=0.2)
    heavy_aug = preprocessor.get_train_transforms(augmentation_strength=0.8)
    val_transform = preprocessor.get_val_transforms()
    
    print(f"   ✅ Created 4 different transform pipelines")
    
    # 2. Custom augmentation pipeline
    print("\n2. Custom Augmentation Pipeline:")
    custom_transform = transforms.Compose([
        transforms.Resize((128, 64)),
        
        # Custom augmentations
        RandomBlur(radius_range=(0.1, 1.0), p=0.3),
        
        # Standard transforms
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.1, contrast=0.1)
        ], p=0.5),
        
        transforms.ToTensor(),
        
        # Custom noise
        AddGaussianNoise(mean=0.0, std=0.02),
        
        # Normalization
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    print(f"   ✅ Custom augmentation pipeline created")
    
    # 3. Configuration-based pipeline
    print("\n3. Configuration-based Pipeline:")
    config = {
        'image_size': [128, 64],
        'normalize': True,
        'augmentation_strength': 0.4
    }
    
    train_transform, val_transform = create_preprocessing_pipeline(config)
    print(f"   ✅ Pipeline created from config")
    
    # 4. Test with actual images
    print("\n4. Testing with Dataset:")
    dataset = KhmerDigitsDataset(
        metadata_path='generated_data/metadata.yaml',
        split='train',
        transform=None  # No transform initially
    )
    
    # Test different transforms on same image
    sample_idx = 0
    original_image, _, metadata = dataset[sample_idx]
    print(f"   Testing on sample: '{metadata['original_label']}'")
    
    transforms_to_test = [
        ("Original", lambda x: x),
        ("Base", base_transform),
        ("Light Aug", light_aug),
        ("Heavy Aug", heavy_aug),
        ("Custom", custom_transform),
        ("Validation", val_transform)
    ]
    
    results = {}
    for name, transform in transforms_to_test:
        try:
            if name == "Original":
                result = original_image
                shape = f"{result.size}"
            else:
                result = transform(original_image)
                shape = f"{result.shape}"
            
            results[name] = result
            print(f"     {name}: {shape} ✅")
        except Exception as e:
            print(f"     {name}: Error - {e} ❌")
    
    # 5. Test-Time Augmentation
    print("\n5. Test-Time Augmentation:")
    tta = TestTimeAugmentation(val_transform, num_augmentations=5)
    tta_results = tta(original_image)
    print(f"   ✅ Generated {len(tta_results)} augmented versions")
    print(f"   Each version shape: {tta_results[0].shape}")
    
    # 6. Batch processing test
    print("\n6. Batch Processing Test:")
    from torch.utils.data import DataLoader
    from src.modules.data_utils.dataset import collate_fn
    
    # Create dataset with custom transform
    custom_dataset = KhmerDigitsDataset(
        metadata_path='generated_data/metadata.yaml',
        split='train',
        transform=custom_transform
    )
    
    # Create data loader
    custom_loader = DataLoader(
        custom_dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Test batch
    images, labels, metadata_list = next(iter(custom_loader))
    print(f"   ✅ Batch processing successful:")
    print(f"     Images shape: {images.shape}")
    print(f"     Labels shape: {labels.shape}")
    print(f"     Batch size: {len(metadata_list)}")
    
    print(f"\n✅ Custom preprocessing examples complete!")

if __name__ == '__main__':
    custom_preprocessing_example()
```

## Example 5: Advanced Visualization

```python
#!/usr/bin/env python3
"""
Example 5: Advanced visualization and debugging
"""
import matplotlib.pyplot as plt
from src.modules.data_utils import (
    KhmerDigitsDataset, DataVisualizer, create_data_loaders,
    get_train_transforms, get_val_transforms
)

def advanced_visualization_example():
    print("Advanced Visualization Example")
    print("=" * 35)
    
    # Setup
    dataset = KhmerDigitsDataset('generated_data/metadata.yaml', split='train')
    visualizer = DataVisualizer(figsize=(15, 10))
    
    # 1. Sample grid with metadata
    print("\n1. Creating Sample Visualizations:")
    
    # Basic samples
    fig1 = visualizer.plot_samples(
        dataset,
        num_samples=20,
        cols=5,
        show_metadata=True,
        save_path='advanced_samples.png'
    )
    plt.close(fig1)
    print("   ✅ Sample grid saved: advanced_samples.png")
    
    # Comprehensive statistics
    fig2 = visualizer.plot_dataset_statistics(
        dataset,
        save_path='advanced_statistics.png'
    )
    plt.close(fig2)
    print("   ✅ Statistics plot saved: advanced_statistics.png")
    
    # 2. Transform comparison
    print("\n2. Transform Comparison:")
    
    # Create different transforms
    transforms_list = [
        ('Original', lambda x: x),
        ('Training (Light)', get_train_transforms(augmentation_strength=0.2)),
        ('Training (Heavy)', get_train_transforms(augmentation_strength=0.8)),
        ('Validation', get_val_transforms())
    ]
    
    fig3 = visualizer.plot_transforms_comparison(
        dataset,
        sample_idx=0,
        transforms_list=transforms_list,
        save_path='transform_comparison.png'
    )
    plt.close(fig3)
    print("   ✅ Transform comparison saved: transform_comparison.png")
    
    # 3. Batch visualization
    print("\n3. Batch Visualization:")
    
    # Create data loaders
    train_transform = get_train_transforms()
    val_transform = get_val_transforms()
    
    train_loader, val_loader = create_data_loaders(
        'generated_data/metadata.yaml',
        train_transform=train_transform,
        val_transform=val_transform,
        batch_size=16,
        num_workers=0  # Avoid multiprocessing for example
    )
    
    # Visualize training batches
    fig4 = visualizer.plot_batch_samples(
        train_loader,
        num_batches=2,
        save_path='batch_visualization.png'
    )
    plt.close(fig4)
    print("   ✅ Batch visualization saved: batch_visualization.png")
    
    # 4. Custom visualization function
    print("\n4. Custom Visualization Functions:")
    
    def plot_font_comparison(dataset, save_path=None):
        """Plot samples grouped by font"""
        # Get samples from different fonts
        font_samples = {}
        for i in range(len(dataset)):
            _, _, metadata = dataset[i]
            font = metadata['font']
            if font not in font_samples:
                font_samples[font] = []
            if len(font_samples[font]) < 3:  # 3 samples per font
                font_samples[font].append(i)
        
        # Create plot
        num_fonts = len(font_samples)
        fig, axes = plt.subplots(num_fonts, 3, figsize=(12, num_fonts * 3))
        
        if num_fonts == 1:
            axes = axes.reshape(1, -1)
        
        for row, (font, indices) in enumerate(font_samples.items()):
            for col, idx in enumerate(indices):
                image, _, metadata = dataset[idx]
                
                ax = axes[row, col] if num_fonts > 1 else axes[col]
                ax.imshow(image)
                ax.set_title(f"{font}\n{metadata['original_label']}")
                ax.axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return fig
    
    # Use custom function
    fig5 = plot_font_comparison(dataset, 'font_comparison.png')
    print("   ✅ Font comparison saved: font_comparison.png")
    
    # 5. Sequence length analysis
    def plot_sequence_length_samples(dataset, save_path=None):
        """Plot samples grouped by sequence length"""
        length_samples = {}
        for i in range(len(dataset)):
            _, _, metadata = dataset[i]
            length = metadata['sequence_length']
            if length not in length_samples:
                length_samples[length] = []
            if len(length_samples[length]) < 4:  # 4 samples per length
                length_samples[length].append(i)
        
        # Sort by length
        sorted_lengths = sorted(length_samples.keys())
        
        fig, axes = plt.subplots(len(sorted_lengths), 4, figsize=(16, len(sorted_lengths) * 3))
        
        if len(sorted_lengths) == 1:
            axes = axes.reshape(1, -1)
        
        for row, length in enumerate(sorted_lengths):
            indices = length_samples[length]
            for col, idx in enumerate(indices):
                image, _, metadata = dataset[idx]
                
                ax = axes[row, col] if len(sorted_lengths) > 1 else axes[col]
                ax.imshow(image)
                ax.set_title(f"Length {length}\n{metadata['original_label']}")
                ax.axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return fig
    
    fig6 = plot_sequence_length_samples(dataset, 'sequence_length_samples.png')
    print("   ✅ Sequence length samples saved: sequence_length_samples.png")
    
    # 6. Training progress simulation
    print("\n5. Training Progress Visualization:")
    
    # Simulate training data
    import numpy as np
    epochs = range(1, 21)
    train_losses = [2.5 * np.exp(-0.15 * e) + 0.1 + np.random.normal(0, 0.05) for e in epochs]
    val_losses = [2.7 * np.exp(-0.12 * e) + 0.15 + np.random.normal(0, 0.08) for e in epochs]
    train_accs = [100 * (1 - np.exp(-0.2 * e)) + np.random.normal(0, 2) for e in epochs]
    val_accs = [100 * (1 - np.exp(-0.18 * e)) + np.random.normal(0, 3) for e in epochs]
    
    from src.modules.data_utils.visualization import plot_training_progress
    
    fig7 = plot_training_progress(
        train_losses, val_losses,
        train_accs, val_accs,
        save_path='training_progress_simulation.png'
    )
    plt.close(fig7)
    print("   ✅ Training progress simulation saved: training_progress_simulation.png")
    
    print(f"\n✅ Advanced visualization examples complete!")
    print(f"Generated visualizations:")
    print(f"  - advanced_samples.png")
    print(f"  - advanced_statistics.png")
    print(f"  - transform_comparison.png")
    print(f"  - batch_visualization.png")
    print(f"  - font_comparison.png")
    print(f"  - sequence_length_samples.png")
    print(f"  - training_progress_simulation.png")

if __name__ == '__main__':
    advanced_visualization_example()
```

## Example 6: Production Training Script

```python
#!/usr/bin/env python3
"""
Example 6: Production-ready training script template
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os
from datetime import datetime

from src.modules.data_utils import (
    create_data_loaders, get_train_transforms, get_val_transforms,
    DatasetAnalyzer, calculate_dataset_metrics
)

class SimpleOCRModel(nn.Module):
    """Simple CNN model for demonstration"""
    def __init__(self, num_classes=13, max_seq_length=9):
        super().__init__()
        self.max_seq_length = max_seq_length
        
        # CNN backbone
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Classifier
        self.classifier = nn.Linear(128, num_classes * max_seq_length)
        self.num_classes = num_classes
    
    def forward(self, x):
        features = self.cnn(x)
        features = features.view(features.size(0), -1)
        output = self.classifier(features)
        return output.view(-1, self.max_seq_length, self.num_classes)

def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    total_samples = 0
    
    for batch_idx, (images, labels, metadata) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        total_samples += images.size(0)
        
        if batch_idx % 50 == 0:
            print(f'Epoch {epoch}, Batch {batch_idx:4d}, Loss: {loss.item():.4f}')
    
    avg_loss = total_loss / len(train_loader)
    return avg_loss

def validate_epoch(model, val_loader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0.0
    correct_chars = 0
    total_chars = 0
    
    with torch.no_grad():
        for images, labels, metadata in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
            
            total_loss += loss.item()
            
            # Calculate accuracy (ignore PAD tokens)
            predictions = torch.argmax(outputs, dim=-1)
            mask = labels != 11  # PAD token
            correct_chars += (predictions == labels)[mask].sum().item()
            total_chars += mask.sum().item()
    
    avg_loss = total_loss / len(val_loader)
    accuracy = correct_chars / total_chars if total_chars > 0 else 0
    return avg_loss, accuracy

def main():
    parser = argparse.ArgumentParser(description='Khmer Digits OCR Training')
    parser.add_argument('--metadata-path', default='generated_data/metadata.yaml',
                       help='Path to dataset metadata')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--output-dir', default='training_output',
                       help='Output directory for models and logs')
    parser.add_argument('--device', default='auto',
                       help='Device to use (cuda/cpu/auto)')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Production Training Script")
    print(f"=" * 30)
    print(f"Configuration:")
    print(f"  Metadata path: {args.metadata_path}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Device: {device}")
    print(f"  Output directory: {args.output_dir}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Analyze dataset first
    print(f"\nDataset Analysis:")
    from src.modules.data_utils import KhmerDigitsDataset
    full_dataset = KhmerDigitsDataset(args.metadata_path, split='all')
    metrics = calculate_dataset_metrics(full_dataset)
    
    print(f"  Total samples: {metrics['total_samples']:,}")
    print(f"  Diversity score: {metrics['diversity_score']:.3f}")
    print(f"  Font balance: {metrics['font_balance_score']:.3f}")
    print(f"  Character coverage: {metrics['character_coverage']:.3f}")
    
    # Create data loaders
    print(f"\nCreating Data Loaders:")
    train_transform = get_train_transforms(
        image_size=(128, 64),
        augmentation_strength=0.3,
        normalize=True
    )
    val_transform = get_val_transforms(
        image_size=(128, 64),
        normalize=True
    )
    
    train_loader, val_loader = create_data_loaders(
        metadata_path=args.metadata_path,
        train_transform=train_transform,
        val_transform=val_transform,
        batch_size=args.batch_size,
        num_workers=4,
        shuffle_train=True
    )
    
    print(f"  Training batches: {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")
    
    # Create model
    print(f"\nCreating Model:")
    model = SimpleOCRModel(num_classes=13, max_seq_length=9).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Setup training
    criterion = nn.CrossEntropyLoss(ignore_index=11)  # Ignore PAD token
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    # Training loop
    print(f"\nStarting Training:")
    best_val_acc = 0.0
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"-" * 20)
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        train_losses.append(train_loss)
        
        # Validate
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val Accuracy: {val_acc:.1%}")
        print(f"Learning Rate: {current_lr:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss
            }, os.path.join(args.output_dir, 'best_model.pth'))
            print(f"✅ New best model saved (Val Acc: {val_acc:.1%})")
    
    # Save final model
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies
    }, os.path.join(args.output_dir, 'final_model.pth'))
    
    # Plot training progress
    print(f"\nGenerating Training Plots:")
    from src.modules.data_utils.visualization import plot_training_progress
    
    # Convert accuracy to percentage
    val_accs_percent = [acc * 100 for acc in val_accuracies]
    train_accs_percent = [100 - loss * 10 for loss in train_losses]  # Approximate
    
    fig = plot_training_progress(
        train_losses, val_losses,
        train_accs_percent, val_accs_percent,
        save_path=os.path.join(args.output_dir, 'training_progress.png')
    )
    
    print(f"\nTraining Complete!")
    print(f"  Best validation accuracy: {best_val_acc:.1%}")
    print(f"  Final validation accuracy: {val_accuracies[-1]:.1%}")
    print(f"  Models saved in: {args.output_dir}")
    print(f"  ✅ Training progress plot: {os.path.join(args.output_dir, 'training_progress.png')}")

if __name__ == '__main__':
    main()
```

## Running the Examples

To run these examples:

```bash
# Make sure you're in the project root directory
cd /path/to/kh_ocr_prototype

# Run individual examples
python docs/examples/example_1_exploration.py
python docs/examples/example_2_training_pipeline.py
python docs/examples/example_3_analysis.py
python docs/examples/example_4_preprocessing.py
python docs/examples/example_5_visualization.py

# Run production training (example 6)
python docs/examples/example_6_production_training.py --epochs 5 --batch-size 16
```

Each example is self-contained and demonstrates specific aspects of the data_utils module. The examples progress from basic usage to advanced production scenarios.

## Example 7: Font Detection and Troubleshooting

```python
#!/usr/bin/env python3
"""
Example 7: Khmer font detection and troubleshooting
"""
from src.modules.data_utils import (
    KhmerFontManager, print_font_status, safe_khmer_text,
    plot_samples, KhmerDigitsDataset
)

def font_detection_example():
    print("Khmer Font Detection and Management Example")
    print("=" * 50)
    
    # 1. Check current font status
    print("\n1. Font Detection Status:")
    print_font_status()
    
    # 2. Test font manager directly
    print("\n2. Font Manager Details:")
    manager = KhmerFontManager()
    
    print(f"   Total fonts detected: {len(manager.available_fonts)}")
    print(f"   Selected font: {manager.current_font}")
    
    if manager.available_fonts:
        print(f"   Available fonts:")
        for font_name, font_path in manager.available_fonts.items():
            path_type = "project" if "src/fonts" in str(font_path) else "system"
            print(f"     • {font_name} ({path_type})")
    
    # 3. Test safe text rendering
    print("\n3. Safe Text Rendering Tests:")
    
    test_cases = [
        ("១២៣៤៥", "Sample {}"),
        ("០៩៨៧៦", "Number {}"),
        ("៣៣៣៣៣៣៣", "Sequence {}"),
        ("៤២", "Short {}")
    ]
    
    for i, (khmer_text, fallback_format) in enumerate(test_cases):
        display_text, font_props = safe_khmer_text(
            khmer_text, 
            fallback_format, 
            i
        )
        
        font_info = ""
        if font_props:
            font_info = f" (using {font_props['fontname']})"
        
        print(f"   Test {i+1}: '{khmer_text}' -> '{display_text}'{font_info}")
    
    # 4. Test visualization with font handling
    print("\n4. Testing Visualization with Font Handling:")
    
    try:
        # Load a small dataset sample
        dataset = KhmerDigitsDataset(
            'generated_data/metadata.yaml', 
            split='train', 
            transform=None
        )
        
        # Create visualization with proper font handling
        print("   Creating sample visualization...")
        fig = plot_samples(
            dataset,
            num_samples=8,
            cols=4,
            show_metadata=True,
            save_path='font_test_samples.png'
        )
        
        print("   ✅ Visualization created: font_test_samples.png")
        
        # Show sample labels
        print("   Sample labels from dataset:")
        for i in range(5):
            _, _, metadata = dataset[i]
            original = metadata['original_label']
            safe_text, props = safe_khmer_text(original, "Sample {}", i)
            
            print(f"     Sample {i}: '{original}' -> display as '{safe_text}'")
        
    except Exception as e:
        print(f"   ❌ Visualization test failed: {e}")
    
    # 5. Font troubleshooting guide
    print("\n5. Font Troubleshooting:")
    
    if not manager.available_fonts:
        print("   ❌ No Khmer fonts detected")
        print("   Recommendations:")
        print("     • Install system Khmer fonts:")
        print("       - Windows: Khmer UI, Khmer OS")
        print("       - macOS: Khmer Sangam MN")
        print("       - Linux: sudo apt-get install fonts-khmeros")
        print("     • Verify project fonts in src/fonts/ are valid TTF files")
        
    elif not manager.current_font:
        print("   ⚠️  Fonts detected but none selected")
        print("   This may indicate font registration issues")
        
    else:
        print("   ✅ Font system working properly")
        print(f"   Current setup: {manager.current_font}")
        print("   Visualizations should display proper Khmer text")
    
    print("\n✅ Font detection and management example completed!")

if __name__ == '__main__':
    font_detection_example()
``` 