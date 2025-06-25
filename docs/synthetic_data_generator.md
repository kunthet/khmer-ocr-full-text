# SyntheticDataGenerator Documentation

## Overview

The `SyntheticDataGenerator` is a comprehensive Python module designed to create realistic synthetic training data for Khmer digits OCR (Optical Character Recognition). It generates diverse images containing Khmer digit sequences (១-៨ digits) with various fonts, backgrounds, and augmentations to train robust machine learning models.

## Purpose

- **Primary Goal**: Generate high-quality synthetic training data for Khmer digits OCR
- **Target Use Case**: Training CNN-RNN hybrid models for digit sequence recognition
- **Language Focus**: Khmer digits (០១២៣៤៥៦៧៨៩)
- **Sequence Range**: 1-8 digit sequences
- **Output Format**: 128x64 RGB images with metadata

## Architecture

### Module Structure

```
src/modules/synthetic_data_generator/
├── __init__.py                 # Package initialization and exports
├── generator.py               # Main SyntheticDataGenerator class
├── backgrounds.py             # Background generation utilities
├── augmentation.py            # Image augmentation pipeline
└── utils.py                   # Utility functions and helpers
```

### Core Components

#### 1. **SyntheticDataGenerator** (`generator.py`)
- **Main orchestrator class** that coordinates all components
- Manages font loading, validation, and dataset generation
- Implements adaptive font sizing and safe text positioning
- Handles train/validation splitting and metadata creation

#### 2. **BackgroundGenerator** (`backgrounds.py`)
- Generates diverse background types for realistic image variety
- **9 Background Types**: solid colors, gradients, noise textures, paper simulation, patterns
- Automatic text color optimization based on background brightness

#### 3. **ImageAugmentor** (`augmentation.py`)
- Applies realistic image transformations for data diversity
- **7 Augmentation Techniques**: rotation, scaling, brightness, contrast, noise, blur, perspective
- Text-safe augmentation parameters to prevent digit cropping

#### 4. **Utility Functions** (`utils.py`)
- Unicode normalization for consistent Khmer text handling
- Font management and validation systems
- Dataset statistics calculation and validation
- Character mapping and sequence generation

## Key Features

### 🎨 **Diverse Visual Generation**

**Background Variety:**
- **Solid Colors**: Random light colors optimized for text visibility
- **Gradients**: Horizontal, vertical, and diagonal color transitions
- **Noise Textures**: Subtle noise patterns for realistic document simulation
- **Paper Textures**: Fiber-like patterns mimicking real paper documents
- **Subtle Patterns**: Dots, lines, and grid patterns for variety

**Font Support:**
- **8 Khmer Fonts**: Complete KhmerOS font family support
- **Automatic Validation**: Font rendering capability testing
- **Even Distribution**: Balanced font usage across dataset

### 🔧 **Intelligent Text Positioning**

**Adaptive Font Sizing:**
- Dynamic font size calculation based on sequence length
- Longer sequences automatically use smaller fonts
- Iterative testing to ensure text fits within image bounds
- Minimum font size protection (12px) for readability

**Safe Text Placement:**
- 10% margin buffers on all image edges
- Boundary validation prevents text cropping
- Smart positioning within safe zones
- Fallback sizing if text doesn't fit

### 🌟 **Advanced Augmentation**

**Text-Safe Transformations:**
- **Rotation**: ±5° range to prevent cropping
- **Scaling**: 95-105% range for subtle variation
- **Brightness/Contrast**: ±20% adjustment range
- **Noise**: Gaussian and speckle noise simulation
- **Blur**: Motion and Gaussian blur effects
- **Perspective**: Subtle viewing angle changes

**Intelligent Application:**
- Reduced aggressive parameters compared to standard augmentation
- Selective augmentation application (e.g., 30% rotation probability)
- Text preservation prioritized over augmentation variety

### 📊 **Comprehensive Metadata**

**Per-Sample Tracking:**
```yaml
label: "០១២៣"              # Khmer digit sequence
font: "KhmerOS"              # Font name used
font_size: 32                # Actual font size
text_color: [0, 0, 0]        # RGB text color
text_position: [45, 20]      # Text position (x, y)
sequence_length: 4           # Number of digits
augmented: true              # Augmentation applied
```

**Dataset-Level Information:**
- Total sample counts and train/val splits
- Font distribution statistics
- Character frequency analysis
- Image size and configuration parameters

## Usage Examples

### Basic Dataset Generation

```python
from src.modules.synthetic_data_generator import SyntheticDataGenerator

# Initialize generator
generator = SyntheticDataGenerator(
    config_path="config/model_config.yaml",
    fonts_dir="src/fonts",
    output_dir="generated_data"
)

# Generate dataset
metadata = generator.generate_dataset(
    num_samples=15000,
    train_split=0.8,
    save_images=True,
    show_progress=True
)
```

### Preview Generation

```python
# Generate preview samples
samples = generator.preview_samples(num_samples=10)

for i, (image, label) in enumerate(samples):
    print(f"Sample {i+1}: '{label}' ({len(label)} digits)")
    image.save(f"preview_{i:02d}_{label}.png")
```

### Single Image Generation

```python
# Generate specific text
image, metadata = generator.generate_single_image(
    text="០១២៣",
    font_name="KhmerOS",
    apply_augmentation=True
)

# Generate random sequence
image, metadata = generator.generate_single_image()
print(f"Generated: '{metadata['label']}'")
```

### Command Line Interface

```bash
# Generate full dataset
python src/sample_scripts/generate_dataset.py --num-samples 15000

# Generate preview only
python src/sample_scripts/generate_dataset.py --preview-only

# Validate fonts
python src/sample_scripts/generate_dataset.py --validate-fonts

# Custom configuration
python src/sample_scripts/generate_dataset.py \
    --config custom_config.yaml \
    --fonts-dir custom_fonts/ \
    --output-dir my_dataset/ \
    --num-samples 5000 \
    --train-split 0.9
```

## Configuration

### Model Configuration (`config/model_config.yaml`)

```yaml
model:
  input:
    image_size: [128, 64]     # Width x Height
    channels: 3               # RGB
  
  characters:
    khmer_digits: ["០", "១", "២", "៣", "៤", "៥", "៦", "៧", "៨", "៩"]
    special_tokens: ["<EOS>", "<PAD>", "<BLANK>"]
    max_sequence_length: 8

data:
  dataset_size: 15000
  train_split: 0.8
  val_split: 0.2
  
  augmentation:
    rotation: [-15, 15]       # Degree range
    scaling: [0.8, 1.2]       # Scale factor range
    noise:
      gaussian_std: 0.01      # Noise intensity
    brightness: [-0.2, 0.2]   # Brightness adjustment
    contrast: [-0.2, 0.2]     # Contrast adjustment
```

## Output Structure

### Dataset Organization

```
generated_data/
├── metadata.yaml           # Complete dataset metadata
├── train/                  # Training images (80%)
│   ├── train_000000.png   
│   ├── train_000001.png
│   └── ...
├── val/                    # Validation images (20%)
│   ├── val_000000.png
│   ├── val_000001.png
│   └── ...
└── preview/                # Preview samples (if generated)
    ├── preview_00_០១២៣.png
    └── ...
```

### Metadata Format

```yaml
dataset_info:
  total_samples: 15000
  train_samples: 12000
  val_samples: 3000
  image_size: [128, 64]
  max_sequence_length: 8
  fonts_used: ["KhmerOS", "KhmerOSbattambang", ...]
  character_set: ["០", "១", "២", ...]
  generated_by: "SyntheticDataGenerator v1.0"

train:
  samples:
    - label: "០១២៣"
      font: "KhmerOS"
      font_size: 32
      text_color: [0, 0, 0]
      text_position: [45, 20]
      sequence_length: 4
      augmented: true
      image_path: "train/train_000000.png"
      image_filename: "train_000000.png"
```

## Technical Specifications

### Performance Characteristics

**Generation Speed:**
- ~50-60 samples/second on modern hardware
- Parallel processing for optimal throughput
- Progress tracking with tqdm integration

**Memory Usage:**
- Low memory footprint with batch processing
- Images generated and saved individually
- Configurable batch sizes for large datasets

**Quality Assurance:**
- Automatic font validation before generation
- Text fit validation with fallback sizing
- Comprehensive boundary checking
- Statistical validation of output distribution

### Unicode Support

**Text Normalization:**
- NFC (Canonical Decomposition + Canonical Composition) normalization
- Consistent character encoding across all samples
- Proper handling of Khmer Unicode ranges (U+1780-U+17FF)

**Character Set:**
```python
khmer_digits = ["០", "១", "២", "៣", "៤", "៥", "៦", "៧", "៨", "៩"]
special_tokens = ["<EOS>", "<PAD>", "<BLANK>"]
total_classes = 13  # 10 digits + 3 special tokens
```

### Font Management

**Supported Fonts:**
- KhmerOS (regular)
- KhmerOSbattambang
- KhmerOSbokor
- KhmerOSfasthand
- KhmerOSmetalchrieng
- KhmerOSmuol
- KhmerOSmuollight
- KhmerOSsiemreap

**Validation Process:**
1. Font file existence check
2. Khmer character rendering test
3. Bounding box calculation validation
4. Working font filtering and reporting

## Error Handling

### Common Issues and Solutions

**Font Loading Errors:**
```python
# Automatic fallback to working fonts
if not self.working_fonts:
    raise ValueError("No working Khmer fonts found!")
```

**Text Positioning Failures:**
```python
# Fallback sizing with validation
if not self._validate_text_fits(text, font, (x, y), image_size):
    # Try smaller font sizes until text fits
    for fallback_size in range(font_size - 2, 8, -2):
        # Test and apply working font size
```

**YAML Serialization Issues:**
```python
# Convert tuples to lists for YAML compatibility
metadata = {
    'text_color': list(text_color),    # tuple -> list
    'text_position': [x, y],           # tuple -> list
    'image_size': list(self.image_size) # tuple -> list
}
```

## Integration with Training Pipeline

### PyTorch Dataset Integration

```python
from torch.utils.data import Dataset
import yaml
from PIL import Image

class KhmerDigitsDataset(Dataset):
    def __init__(self, metadata_path, transform=None):
        with open(metadata_path, 'r') as f:
            self.metadata = yaml.safe_load(f)
        self.samples = self.metadata['train']['samples']  # or 'val'
        self.transform = transform
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample['image_path'])
        label = sample['label']
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
```

### Data Loading Example

```python
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# Define transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

# Create datasets
train_dataset = KhmerDigitsDataset('generated_data/metadata.yaml', transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Training loop
for batch_idx, (images, labels) in enumerate(train_loader):
    # Forward pass, loss calculation, backpropagation
    pass
```

## Troubleshooting

### Common Solutions

**Issue: "No working fonts found"**
```bash
# Validate font installation
python src/sample_scripts/test_fonts.py
```

**Issue: "Text cropping in images"**
- Check image size configuration
- Verify font size calculation
- Ensure margin settings are appropriate

**Issue: "YAML serialization errors"**
- Verify metadata structure
- Check for tuple/list conversion
- Validate Unicode character handling

**Issue: "Slow generation speed"**
- Reduce augmentation complexity
- Disable progress bars for automated runs
- Check available system resources

## Version History

**v1.0.0 (Current)**
- Initial implementation with complete pipeline
- Support for 8 Khmer fonts with validation
- 9 background types and 7 augmentation techniques
- Adaptive font sizing and safe text positioning
- Comprehensive metadata tracking and statistics
- Train/validation splitting with YAML output
- Text cropping prevention with boundary validation

## Dependencies

**Core Requirements:**
- Python 3.8+
- PIL/Pillow 9.0+
- NumPy 1.21+
- PyYAML 6.0+
- tqdm 4.64+
- OpenCV 4.5+ (for advanced augmentations)

**Development Dependencies:**
- pytest (for testing)
- matplotlib (for visualization)
- jupyter (for notebooks)

## License and Usage

This synthetic data generator is part of the Khmer OCR Prototype project and follows the same licensing terms. Generated synthetic data can be used freely for research and development purposes.

---

*For technical support or feature requests, please refer to the project documentation or create an issue in the project repository.* 