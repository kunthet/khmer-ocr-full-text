# Khmer Digits OCR - Models Module Documentation

## Overview

The `src/models` module contains the complete implementation of the CNN-RNN hybrid architecture for Khmer digits OCR. This module provides a comprehensive, production-ready implementation with multiple model variants, configuration management, and extensive utilities.

## Architecture Overview

The OCR architecture follows a modern encoder-decoder design with attention mechanism:

```
Input Image → CNN Backbone → RNN Encoder → Attention Decoder → Character Sequence
[3,64,128]    [8,512]        [8,256]       [8,13]             "០១២៣"
```

### Core Components

1. **CNN Backbone**: Feature extraction from images (ResNet-18/EfficientNet-B0)
2. **RNN Encoder**: Bidirectional LSTM for sequence modeling  
3. **Attention Mechanism**: Bahdanau attention for alignment
4. **RNN Decoder**: LSTM decoder with attention for character generation
5. **Model Factory**: Configuration-driven model creation and management

## Module Structure

```
src/models/
├── __init__.py          # Module exports and imports
├── backbone.py          # CNN backbone architectures  
├── encoder.py           # RNN encoder components
├── attention.py         # Attention mechanisms
├── decoder.py           # RNN decoder components
├── ocr_model.py         # Complete OCR model
├── model_factory.py     # Model factory and presets
└── utils.py             # Model utilities and analysis
```

## Quick Start Guide

### Basic Usage

```python
from models import create_model, KhmerDigitsOCR
import torch

# Create a model using preset
model = create_model(preset='medium')

# Load sample image
image = torch.randn(1, 3, 64, 128)  # [batch, channels, height, width]

# Make prediction
model.eval()
with torch.no_grad():
    predictions = model(image)  # [1, 8, 13]
    text = model.predict(image)[0]  # "០១២៣"

print(f"Predicted text: {text}")
```

### Advanced Configuration

```python
from models import ModelFactory
import yaml

# Create custom model
model = create_model(
    preset='medium',
    vocab_size=13,
    max_sequence_length=8,
    encoder_hidden_size=512,
    decoder_hidden_size=512,
    dropout=0.2
)

# Load from configuration file
model = create_model(config_path='config/model_config.yaml')

# Get model information
info = model.get_model_info()
print(f"Parameters: {info['total_parameters']:,}")
print(f"Model size: {info['model_size_mb']:.1f} MB")
```

## Component Documentation

### 1. CNN Backbone (`backbone.py`)

Provides feature extraction from input images using pretrained CNN architectures.

#### Classes

**`CNNBackbone` (Abstract Base Class)**
- Base interface for all CNN backbone implementations
- Defines standard forward pass and output shape properties

**`ResNetBackbone`**
- ResNet-18 based feature extractor
- Adaptive pooling to fixed sequence length (8 positions)
- Feature projection to configurable dimensions

```python
from models.backbone import ResNetBackbone

backbone = ResNetBackbone(
    feature_size=512,     # Output feature dimension
    pretrained=True,      # Use ImageNet pretrained weights
    dropout=0.1           # Dropout rate for regularization
)

# Input: [batch_size, 3, 64, 128]
# Output: [batch_size, 8, 512]
features = backbone(images)
```

**`EfficientNetBackbone`**
- EfficientNet-B0 based feature extractor
- More parameter efficient than ResNet-18
- Requires `efficientnet-pytorch` package

```python
from models.backbone import EfficientNetBackbone

backbone = EfficientNetBackbone(
    feature_size=512,
    pretrained=True,
    dropout=0.1
)
```

#### Factory Function

```python
from models.backbone import create_backbone

backbone = create_backbone(
    backbone_type='resnet18',  # or 'efficientnet-b0'
    feature_size=512,
    pretrained=True
)
```

### 2. RNN Encoder (`encoder.py`)

Processes CNN features into contextual sequence representations.

#### Classes

**`RNNEncoder` (Abstract Base Class)**
- Base interface for all RNN encoder implementations

**`BiLSTMEncoder`**
- Bidirectional LSTM encoder
- Layer normalization and dropout for stability
- Proper gradient flow initialization

```python
from models.encoder import BiLSTMEncoder

encoder = BiLSTMEncoder(
    input_size=512,       # CNN feature size
    hidden_size=256,      # LSTM hidden dimension
    num_layers=2,         # Number of LSTM layers
    dropout=0.1,          # Dropout between layers
    batch_first=True      # Batch dimension first
)

# Input: [batch_size, seq_len, input_size]
# Output: encoded_features [batch_size, seq_len, hidden_size]
#         final_hidden [batch_size, hidden_size]
encoded, hidden = encoder(cnn_features)
```

#### Factory Function

```python
from models.encoder import create_encoder

encoder = create_encoder(
    encoder_type='bilstm',
    input_size=512,
    hidden_size=256,
    num_layers=2,
    dropout=0.1
)
```

### 3. Attention Mechanism (`attention.py`)

Implements attention mechanisms for focusing on relevant image regions.

#### Classes

**`BahdanauAttention`**
- Additive attention mechanism
- Learnable alignment between encoder and decoder states
- Normalized attention weights

```python
from models.attention import BahdanauAttention

attention = BahdanauAttention(
    encoder_hidden_size=256,  # Encoder state dimension
    decoder_hidden_size=256,  # Decoder state dimension  
    attention_size=256        # Attention projection dimension
)

# Compute attention
context_vector, attention_weights = attention(
    encoder_states,  # [batch_size, seq_len, encoder_hidden_size]
    decoder_state,   # [batch_size, decoder_hidden_size]
    encoder_mask     # [batch_size, seq_len] (optional)
)
```

### 4. RNN Decoder (`decoder.py`)

Generates character sequences from encoded features.

#### Classes

**`RNNDecoder` (Abstract Base Class)**
- Base interface for all decoder implementations

**`AttentionDecoder`**
- LSTM decoder with Bahdanau attention
- Teacher forcing for training, autoregressive for inference
- Character embedding and output projection

```python
from models.decoder import AttentionDecoder

decoder = AttentionDecoder(
    vocab_size=13,                # Character vocabulary size
    encoder_hidden_size=256,      # Encoder output dimension
    decoder_hidden_size=256,      # Decoder LSTM dimension
    num_layers=1,                 # Number of LSTM layers
    dropout=0.1,                  # Dropout rate
    attention_size=256            # Attention mechanism size
)

# Training mode (teacher forcing)
predictions = decoder(
    encoder_features,    # [batch_size, seq_len, encoder_hidden_size]
    target_sequence,     # [batch_size, target_len]
    max_length=8
)

# Inference mode (autoregressive)
predictions = decoder(encoder_features, max_length=8)
```

**`CTCDecoder`**
- Connectionist Temporal Classification decoder
- Alignment-free training approach
- Simpler than attention but less flexible

```python
from models.decoder import CTCDecoder

decoder = CTCDecoder(
    vocab_size=13,
    encoder_hidden_size=256,
    dropout=0.1
)

# Output log probabilities for CTC loss
log_probs = decoder(encoder_features)
```

#### Factory Function

```python
from models.decoder import create_decoder

# Attention decoder
decoder = create_decoder(
    decoder_type='attention',
    vocab_size=13,
    encoder_hidden_size=256,
    decoder_hidden_size=256
)

# CTC decoder  
decoder = create_decoder(
    decoder_type='ctc',
    vocab_size=13,
    encoder_hidden_size=256
)
```

### 5. Complete OCR Model (`ocr_model.py`)

Integrates all components into an end-to-end OCR system.

#### Classes

**`KhmerDigitsOCR`**
- Complete OCR model combining all components
- Character encoding/decoding utilities
- Configuration management and model information

```python
from models.ocr_model import KhmerDigitsOCR

model = KhmerDigitsOCR(
    vocab_size=13,                    # 10 digits + 3 special tokens
    max_sequence_length=8,            # Maximum digit sequence length
    cnn_type='resnet18',              # CNN backbone type
    encoder_type='bilstm',            # Encoder type
    decoder_type='attention',         # Decoder type
    feature_size=512,                 # CNN output features
    encoder_hidden_size=256,          # Encoder LSTM size
    decoder_hidden_size=256,          # Decoder LSTM size
    attention_size=256,               # Attention mechanism size
    num_encoder_layers=2,             # Number of encoder layers
    num_decoder_layers=1,             # Number of decoder layers
    dropout=0.1,                      # Dropout rate
    pretrained_cnn=True               # Use pretrained CNN
)
```

#### Key Methods

**Forward Pass**
```python
# Training mode
predictions = model(
    images,           # [batch_size, 3, height, width]
    target_sequences, # [batch_size, seq_len] (optional)
    sequence_lengths  # [batch_size] (optional)
)

# Inference mode
predictions = model(images)
```

**Text Prediction**
```python
# Predict text from images
texts = model.predict(images)  # List of predicted strings
print(texts)  # ['០១២', '៣៤៥៦', ...]
```

**Text Encoding/Decoding**
```python
# Encode text to indices
text = "០១២៣"
indices = model.encode_text(text, max_length=8)

# Decode indices to text
text = model._decode_sequence(indices)
```

**Model Information**
```python
info = model.get_model_info()
print(f"Parameters: {info['total_parameters']:,}")
print(f"Model size: {info['model_size_mb']:.1f} MB")
```

**Configuration Management**
```python
# Create from config file
model = KhmerDigitsOCR.from_config('config/model_config.yaml')

# Save configuration
model.save_config('saved_config.yaml')
```

### 6. Model Factory (`model_factory.py`)

Provides factory methods and predefined model configurations.

#### Classes

**`ModelFactory`**
- Factory class with predefined model presets
- Configuration file loading
- Model creation and management utilities

#### Model Presets

```python
from models.model_factory import ModelFactory

# List available presets
presets = ModelFactory.list_presets()
print(presets.keys())  # ['small', 'medium', 'large', 'ctc_small', 'ctc_medium']

# Get preset information
info = ModelFactory.get_preset_info('medium')
print(f"Estimated parameters: {info['estimated_parameters']['total']:,}")
print(f"Configuration: {info['configuration']}")
```

**Preset Specifications:**

| Preset | Parameters | CNN | Encoder Hidden | Decoder Type | Description |
|--------|------------|-----|----------------|--------------|-------------|
| `small` | ~12.5M | ResNet-18 | 128 | Attention | Lightweight model |
| `medium` | ~16.2M | ResNet-18 | 256 | Attention | Balanced performance |
| `large` | ~30M | EfficientNet-B0 | 512 | Attention | High-performance model |
| `ctc_small` | ~12.3M | ResNet-18 | 128 | CTC | Fast inference |
| `ctc_medium` | ~16M | ResNet-18 | 256 | CTC | Balanced CTC model |

#### Factory Functions

```python
from models.model_factory import create_model, load_model, save_model

# Create from preset
model = create_model(preset='medium')

# Create from config file
model = create_model(config_path='config/model_config.yaml')

# Create with custom parameters
model = create_model(
    preset='medium',
    encoder_hidden_size=512,  # Override preset value
    dropout=0.2
)

# Load saved model
model = load_model('checkpoints/best_model.pth')

# Save model checkpoint
save_model(
    model=model,
    checkpoint_path='checkpoints/epoch_10.pth',
    optimizer=optimizer,
    epoch=10,
    loss=0.45,
    metrics={'accuracy': 0.95}
)
```

### 7. Model Utilities (`utils.py`)

Provides utilities for model analysis, debugging, and performance profiling.

#### Classes

**`ModelSummary`**
- Comprehensive model analysis and summary generation
- Parameter counting and memory usage estimation
- Layer-by-layer information

```python
from models.utils import ModelSummary

summary_tool = ModelSummary(model)

# Generate summary
summary = summary_tool.summary(
    input_size=(3, 64, 128),
    batch_size=1,
    device='cpu'
)

# Print formatted summary
summary_tool.print_summary()

# Save summary to file
summary_tool.save_summary('model_summary.json')
```

#### Utility Functions

**Parameter Counting**
```python
from models.utils import count_parameters, get_model_info

# Count parameters
total_params = count_parameters(model, trainable_only=False)
trainable_params = count_parameters(model, trainable_only=True)

# Get comprehensive model info
info = get_model_info(model)
print(f"Model class: {info['model_class']}")
print(f"Devices: {info['devices']}")
print(f"Parameter types: {info['parameter_dtypes']}")
```

**Performance Profiling**
```python
from models.utils import profile_model

metrics = profile_model(
    model=model,
    input_size=(3, 64, 128),
    batch_size=1,
    num_runs=10,
    device='cpu'
)

print(f"Inference time: {metrics['avg_inference_time_ms']:.2f} ms")
print(f"Throughput: {metrics['throughput_samples_per_sec']:.1f} samples/sec")
```

**Model Comparison**
```python
from models.utils import compare_models

models = {
    'small': create_model(preset='small'),
    'medium': create_model(preset='medium'),
    'large': create_model(preset='large')
}

comparison = compare_models(models)
for name, stats in comparison.items():
    print(f"{name}: {stats['parameters']:,} params, {stats['model_size_mb']:.1f} MB")
```

**Architecture Visualization**
```python
from models.utils import visualize_architecture

# Generate text-based architecture diagram
diagram = visualize_architecture(
    model=model,
    input_size=(3, 64, 128),
    save_path='model_architecture.txt'
)
print(diagram)
```

## Configuration System

The models module supports YAML-based configuration management for reproducible experiments.

### Configuration File Format

```yaml
# config/model_config.yaml
model:
  name: "khmer_digits_ocr"
  architecture: "cnn_rnn_attention"
  
  # Input specifications
  input:
    image_size: [128, 64]  # [width, height]
    channels: 3
    normalization:
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]
  
  # Character set
  characters:
    khmer_digits: ["០", "១", "២", "៣", "៤", "៥", "៦", "៧", "៨", "៩"]
    special_tokens: ["<EOS>", "<PAD>", "<BLANK>"]
    total_classes: 13
    max_sequence_length: 8
  
  # CNN Backbone
  cnn:
    type: "resnet18"
    pretrained: true
    feature_size: 512
  
  # RNN Components
  rnn:
    encoder:
      type: "bidirectional_lstm"
      hidden_size: 256
      num_layers: 2
      dropout: 0.1
    
    decoder:
      type: "lstm"
      hidden_size: 256
      num_layers: 1
      dropout: 0.1
    
    attention:
      type: "bahdanau"
      hidden_size: 256
```

### Using Configuration

```python
from models import create_model

# Load model from configuration
model = create_model(config_path='config/model_config.yaml')

# Override specific parameters
model = create_model(
    config_path='config/model_config.yaml',
    encoder_hidden_size=512,  # Override config value
    dropout=0.2
)
```

## Character Set and Vocabulary

The models use a 13-class vocabulary for Khmer digits:

```python
character_mappings = {
    '០': 0,   # Khmer digit 0
    '១': 1,   # Khmer digit 1  
    '២': 2,   # Khmer digit 2
    '៣': 3,   # Khmer digit 3
    '៤': 4,   # Khmer digit 4
    '៥': 5,   # Khmer digit 5
    '៦': 6,   # Khmer digit 6
    '៧': 7,   # Khmer digit 7
    '៨': 8,   # Khmer digit 8
    '៩': 9,   # Khmer digit 9
    '<EOS>': 10,   # End of sequence
    '<PAD>': 11,   # Padding token
    '<BLANK>': 12  # CTC blank token
}
```

### Sequence Handling

- **Maximum Length**: 8 characters (configurable)
- **Variable Length**: Sequences from 1-8 digits supported
- **Padding**: Shorter sequences padded with `<PAD>` token
- **End Marker**: Sequences terminated with `<EOS>` token

## Memory and Performance Specifications

### Model Size Comparison

| Model | Parameters | Memory (MB) | Inference (ms) | Accuracy Target |
|-------|------------|-------------|----------------|-----------------|
| Small | 12.5M | 47.6 | ~50 | >90% |
| Medium | 16.2M | 61.8 | ~70 | >95% |
| Large | 30M+ | 115+ | ~120 | >98% |
| CTC Small | 12.3M | 46.9 | ~30 | >88% |

### Hardware Requirements

**Minimum Requirements:**
- RAM: 2GB for inference, 4GB for training
- GPU: Optional, but recommended for training
- Storage: 500MB for model files

**Recommended Requirements:**
- RAM: 8GB+ for comfortable training
- GPU: 4GB+ VRAM for batch training
- Storage: 2GB+ for experiments and checkpoints

## Error Handling and Debugging

### Common Issues and Solutions

**1. Import Errors**
```python
# If you get import errors, ensure models is in Python path
import sys
sys.path.append('src')
from models import create_model
```

**2. Memory Issues**
```python
# Reduce batch size or model size for limited memory
model = create_model(preset='small')  # Use smaller model

# Clear cache
torch.cuda.empty_cache()  # If using GPU
```

**3. Shape Mismatches**
```python
# Ensure correct input shape
images = images.view(-1, 3, 64, 128)  # Reshape if needed

# Check model output shapes
summary = ModelSummary(model).summary()
print(summary['layer_details'])
```

### Debugging Tools

```python
from models.utils import ModelSummary, visualize_architecture

# Model architecture inspection
print(visualize_architecture(model))

# Layer-by-layer analysis
summary = ModelSummary(model)
summary.print_summary()

# Parameter analysis
info = get_model_info(model)
print(f"Trainable parameters: {info['trainable_parameters']:,}")
```

## Integration Examples

### Training Pipeline Integration

```python
import torch
from torch.utils.data import DataLoader
from models import create_model
from modules.data_utils import KhmerDigitsDataset

# Create model
model = create_model(preset='medium')
model.train()

# Setup data loading
dataset = KhmerDigitsDataset('generated_data/metadata.yaml')
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Training loop
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss(ignore_index=11)  # Ignore PAD token

for epoch in range(10):
    for batch_idx, (images, labels, metadata) in enumerate(dataloader):
        optimizer.zero_grad()
        
        # Forward pass
        predictions = model(images, labels)
        
        # Compute loss
        loss = criterion(
            predictions.view(-1, predictions.size(-1)),
            labels.view(-1)
        )
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        if batch_idx % 10 == 0:
            print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
```

### Inference Pipeline

```python
import torch
from PIL import Image
from models import create_model
from modules.data_utils.preprocessing import get_default_transforms

# Load model
model = create_model(preset='medium')
model.eval()

# Load and preprocess image
transform = get_default_transforms()
image = Image.open('sample_image.png').convert('RGB')
tensor_image = transform(image).unsqueeze(0)

# Make prediction
with torch.no_grad():
    predicted_text = model.predict(tensor_image)[0]
    print(f"Predicted: {predicted_text}")
```

### Model Conversion and Deployment

```python
import torch
from models import create_model

# Create and load trained model
model = create_model(preset='medium')
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

# Convert to TorchScript for deployment
sample_input = torch.randn(1, 3, 64, 128)
traced_model = torch.jit.trace(model, sample_input)
traced_model.save('model_traced.pt')

# Export to ONNX
torch.onnx.export(
    model,
    sample_input,
    'model.onnx',
    export_params=True,
    opset_version=11,
    input_names=['images'],
    output_names=['predictions']
)
```

## API Reference Summary

### Core Classes
- `KhmerDigitsOCR`: Main OCR model class
- `ModelFactory`: Model creation and management
- `ModelSummary`: Model analysis and inspection

### Factory Functions
- `create_model()`: Create model from preset or config
- `load_model()`: Load model from checkpoint
- `save_model()`: Save model checkpoint

### Component Classes
- `ResNetBackbone`, `EfficientNetBackbone`: CNN backbones
- `BiLSTMEncoder`: Bidirectional LSTM encoder
- `BahdanauAttention`: Attention mechanism
- `AttentionDecoder`, `CTCDecoder`: Sequence decoders

### Utility Functions
- `count_parameters()`: Count model parameters
- `get_model_info()`: Get model information
- `profile_model()`: Performance profiling
- `compare_models()`: Compare multiple models
- `visualize_architecture()`: Architecture visualization

## Future Extensions

The models module is designed for extensibility:

1. **Additional Backbones**: EfficientNet variants, Vision Transformers
2. **Advanced Attention**: Multi-head attention, self-attention
3. **Alternative Decoders**: Transformer decoders, beam search
4. **Model Compression**: Quantization, pruning, distillation
5. **Full Khmer Script**: Extension to complete character set (84+ characters)

This documentation provides comprehensive coverage of the models module. For specific implementation details, refer to the source code and inline documentation in each component file. 