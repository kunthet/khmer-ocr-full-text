# Data Utils Module - Quick Reference

## Essential Imports
```python
from src.modules.data_utils import (
    KhmerDigitsDataset, create_data_loaders,
    get_train_transforms, get_val_transforms,
    plot_samples, plot_dataset_stats,
    DataVisualizer, DatasetAnalyzer,
    calculate_dataset_metrics, validate_dataset_quality,
    KhmerFontManager, print_font_status, safe_khmer_text
)
```

## 1. Basic Dataset Usage

### Load Dataset
```python
# Load training data
dataset = KhmerDigitsDataset(
    metadata_path='generated_data/metadata.yaml',
    split='train',  # 'train', 'val', 'all'
    transform=None
)

print(f"Dataset size: {len(dataset)}")
```

### Access Sample
```python
image, label_tensor, metadata = dataset[0]
print(f"Label: {metadata['original_label']}")
print(f"Encoded: {label_tensor.tolist()}")
print(f"Image size: {image.size}")
```

### Character Mappings
```python
char_to_idx, idx_to_char = dataset.get_character_mappings()
print(f"Character set: {list(char_to_idx.keys())}")
# Output: ['០', '១', '២', '៣', '៤', '៥', '៦', '៧', '៨', '៩', '<EOS>', '<PAD>', '<BLANK>']
```

## 2. Data Loading Pipeline

### Create Transforms
```python
# Training (with augmentation)
train_transform = get_train_transforms(
    image_size=(128, 64),
    augmentation_strength=0.3
)

# Validation (no augmentation)
val_transform = get_val_transforms(image_size=(128, 64))
```

### Create DataLoaders
```python
train_loader, val_loader = create_data_loaders(
    metadata_path='generated_data/metadata.yaml',
    train_transform=train_transform,
    val_transform=val_transform,
    batch_size=32,
    num_workers=4
)

print(f"Training batches: {len(train_loader)}")
print(f"Validation batches: {len(val_loader)}")
```

### Process Batches
```python
for batch_idx, (images, labels, metadata_list) in enumerate(train_loader):
    # images: [batch_size, 3, height, width]
    # labels: [batch_size, max_seq_length + 1]
    # metadata_list: List of dictionaries
    
    print(f"Batch {batch_idx}:")
    print(f"  Images: {images.shape}")
    print(f"  Labels: {labels.shape}")
    print(f"  Sample labels: {[m['original_label'] for m in metadata_list[:3]]}")
    
    if batch_idx >= 2:  # Process first 3 batches
        break
```

## 3. Visualization

### Basic Plotting
```python
# Plot sample images
plot_samples(dataset, num_samples=16, save_path='samples.png')

# Plot dataset statistics
plot_dataset_stats(dataset, save_path='stats.png')
```

### Advanced Visualization
```python
visualizer = DataVisualizer()

# Comprehensive dataset statistics
fig = visualizer.plot_dataset_statistics(dataset, save_path='detailed_stats.png')

# Batch visualization
fig = visualizer.plot_batch_samples(train_loader, num_batches=2)
```

## 4. Dataset Analysis

### Quick Metrics
```python
metrics = calculate_dataset_metrics(dataset)
print(f"Diversity score: {metrics['diversity_score']:.3f}")
print(f"Font balance: {metrics['font_balance_score']:.3f}")
print(f"Character coverage: {metrics['character_coverage']:.3f}")
```

### Comprehensive Analysis
```python
analyzer = DatasetAnalyzer(dataset)

# Generate complete report
report = analyzer.generate_comprehensive_report('analysis_report.json')

# Create analysis plots
plots = analyzer.create_analysis_plots(save_dir='analysis_plots')
```

### Quality Validation
```python
is_valid, issues = validate_dataset_quality(
    dataset,
    min_samples=1000,
    min_diversity=0.7,
    min_font_balance=0.2
)

print(f"Dataset quality: {'PASS' if is_valid else 'FAIL'}")
for issue in issues:
    print(f"  - {issue}")
```

## 5. Training Integration

### Simple Training Loop
```python
import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = YourOCRModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss(ignore_index=11)  # Ignore PAD token

model.train()
for epoch in range(num_epochs):
    for images, labels, metadata in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### Inference Example
```python
def predict_sequence(model, image_path, transform, device):
    from PIL import Image
    
    # Load and preprocess
    image = Image.open(image_path).convert('RGB')
    tensor = transform(image).unsqueeze(0).to(device)
    
    # Predict
    model.eval()
    with torch.no_grad():
        outputs = model(tensor)
        predicted_indices = torch.argmax(outputs, dim=-1)
    
    # Decode
    dataset = KhmerDigitsDataset('generated_data/metadata.yaml', split='train')
    predicted_text = dataset._decode_label(predicted_indices[0])
    
    return predicted_text

# Usage
val_transform = get_val_transforms()
prediction = predict_sequence(model, 'test_image.png', val_transform, device)
print(f"Predicted: {prediction}")
```

## 6. Configuration Examples

### Dataset Config
```python
dataset_config = {
    'metadata_path': 'generated_data/metadata.yaml',
    'split': 'train',
    'max_sequence_length': 8,
    'transform': train_transform
}
dataset = KhmerDigitsDataset(**dataset_config)
```

### DataLoader Config
```python
dataloader_config = {
    'batch_size': 32,
    'shuffle': True,
    'num_workers': 4,
    'pin_memory': True,
    'drop_last': True
}
```

### Preprocessing Config
```python
from src.modules.data_utils.preprocessing import create_preprocessing_pipeline

config = {
    'image_size': [128, 64],
    'normalize': True,
    'augmentation_strength': 0.3
}

train_transform, val_transform = create_preprocessing_pipeline(config)
```

## 7. Common Patterns

### Custom Dataset Split
```python
# Use 90% for training, 10% for validation
from src.modules.data_utils import create_combined_data_loader

train_loader, val_loader = create_combined_data_loader(
    metadata_path='generated_data/metadata.yaml',
    train_ratio=0.9,
    batch_size=32
)
```

### Debug Dataset Issues
```python
# Check dataset integrity
try:
    for i in range(10):
        image, label, metadata = dataset[i]
        print(f"Sample {i}: {metadata['original_label']} - OK")
except Exception as e:
    print(f"Error at sample {i}: {e}")

# Validate character encoding
test_labels = ["០", "១២៣", "៤៥៦៧", "៨៩០១២៣៤៥"]
for label in test_labels:
    encoded = dataset._encode_label(label)
    decoded = dataset._decode_label(encoded)
    assert label == decoded, f"Mismatch: {label} != {decoded}"
print("Character encoding validation passed!")
```

### Memory Optimization
```python
# For limited memory
train_loader = DataLoader(
    dataset,
    batch_size=16,  # Reduce batch size
    num_workers=2,  # Reduce workers
    pin_memory=False
)
```

### Performance Optimization
```python
import multiprocessing

# Optimal settings
optimal_workers = min(4, multiprocessing.cpu_count())
train_loader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=optimal_workers,
    persistent_workers=True,
    pin_memory=True,
    prefetch_factor=2
)
```

## 8. Error Handling

### Common Issues
```python
# Handle missing files gracefully
try:
    dataset = KhmerDigitsDataset('generated_data/metadata.yaml')
except FileNotFoundError:
    print("Generate dataset first: python src/sample_scripts/generate_dataset.py")

# Handle transform errors
try:
    transformed = train_transform(image)
except Exception as e:
    print(f"Transform error: {e}")
    # Fallback to simple transform
    simple_transform = transforms.Compose([
        transforms.Resize((128, 64)),
        transforms.ToTensor()
    ])
```

## 9. Testing and Validation

### Quick Test
```python
# Run comprehensive test
import subprocess
result = subprocess.run([
    'python', 'src/sample_scripts/test_data_pipeline.py',
    '--metadata', 'generated_data/metadata.yaml',
    '--output-dir', 'test_output'
], capture_output=True, text=True)

if result.returncode == 0:
    print("✅ All tests passed!")
else:
    print("❌ Tests failed:")
    print(result.stdout)
```

### Manual Validation
```python
# Test dataset loading
dataset = KhmerDigitsDataset('generated_data/metadata.yaml', split='all')
print(f"Total samples: {len(dataset)}")

# Test data loaders
train_loader, val_loader = create_data_loaders(
    'generated_data/metadata.yaml', batch_size=8, num_workers=0
)

# Test batch loading
images, labels, metadata = next(iter(train_loader))
print(f"Batch loaded successfully: {images.shape}")

# Test visualization
plot_samples(dataset, num_samples=4, save_path='test_samples.png')
print("✅ Visualization test passed!")
```

## 10. Font Management

### Check Font Status
```python
from src.modules.data_utils import print_font_status

# Check detected Khmer fonts
print_font_status()
```

### Safe Text Rendering
```python
from src.modules.data_utils import safe_khmer_text

# Automatic font handling
khmer_text = "០១២៣៤"
display_text, font_props = safe_khmer_text(khmer_text, "Sample {}", 0)
print(f"Display: {display_text}")  # Either Khmer digits or fallback
```

### Font Manager
```python
from src.modules.data_utils import KhmerFontManager

manager = KhmerFontManager()
print(f"Available fonts: {len(manager.available_fonts)}")
print(f"Selected font: {manager.current_font}")
```

### Visualization with Fonts
```python
# These automatically use proper Khmer fonts
plot_samples(dataset, save_path='samples.png')        # ✅ Proper labels
plot_dataset_stats(dataset, save_path='stats.png')     # ✅ Correct text
```

## 11. Character Set Reference

```python
# Khmer Digits (10 total)
KHMER_DIGITS = {
    "០": 0,  # ZERO
    "១": 1,  # ONE
    "២": 2,  # TWO
    "៣": 3,  # THREE
    "៤": 4,  # FOUR
    "៥": 5,  # FIVE
    "៦": 6,  # SIX
    "៧": 7,  # SEVEN
    "៨": 8,  # EIGHT
    "៩": 9   # NINE
}

# Special Tokens (3 total)
SPECIAL_TOKENS = {
    "<EOS>": 10,   # End of sequence
    "<PAD>": 11,   # Padding token
    "<BLANK>": 12  # Blank/background token
}

# Total character set: 13 characters
```

For detailed documentation, see `docs/data_pipeline_documentation.md` 