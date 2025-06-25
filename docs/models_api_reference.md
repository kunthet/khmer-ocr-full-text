# Models Module API Reference

## Quick Navigation
- [Core Classes](#core-classes)
- [Factory Functions](#factory-functions)
- [Component Classes](#component-classes)
- [Utility Functions](#utility-functions)
- [Configuration](#configuration)
- [Examples](#examples)

## Core Classes

### `KhmerDigitsOCR`
**Location**: `models.ocr_model.KhmerDigitsOCR`

Complete OCR model integrating all components.

#### Constructor
```python
KhmerDigitsOCR(
    vocab_size: int = 13,
    max_sequence_length: int = 8,
    cnn_type: str = 'resnet18',
    encoder_type: str = 'bilstm',
    decoder_type: str = 'attention',
    feature_size: int = 512,
    encoder_hidden_size: int = 256,
    decoder_hidden_size: int = 256,
    attention_size: int = 256,
    num_encoder_layers: int = 2,
    num_decoder_layers: int = 1,
    dropout: float = 0.1,
    pretrained_cnn: bool = True
)
```

#### Methods
| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `forward()` | `images`, `target_sequences=None`, `sequence_lengths=None` | `torch.Tensor` | Forward pass |
| `predict()` | `images`, `return_attention=False` | `List[str]` | Predict text from images |
| `encode_text()` | `text`, `max_length=None` | `torch.Tensor` | Encode text to indices |
| `get_model_info()` | - | `Dict[str, Any]` | Get model information |
| `from_config()` | `config_path` | `KhmerDigitsOCR` | Create from config file |
| `save_config()` | `config_path` | `None` | Save configuration |

#### Properties
- `char_to_idx`: Character to index mapping
- `idx_to_char`: Index to character mapping
- `vocab_size`: Vocabulary size
- `max_sequence_length`: Maximum sequence length

---

### `ModelFactory`
**Location**: `models.model_factory.ModelFactory`

Factory for creating and managing models.

#### Class Methods
| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `create_model()` | `config=None`, `preset=None`, `**kwargs` | `KhmerDigitsOCR` | Create model |
| `list_presets()` | - | `Dict[str, Dict]` | List available presets |
| `get_preset_info()` | `preset` | `Dict[str, Any]` | Get preset information |

#### Available Presets
- `small`: 12.5M parameters, ResNet-18 + BiLSTM(128)
- `medium`: 16.2M parameters, ResNet-18 + BiLSTM(256)  
- `large`: 30M+ parameters, EfficientNet + BiLSTM(512)
- `ctc_small`: 12.3M parameters, CTC decoder
- `ctc_medium`: 16M parameters, CTC decoder

---

### `ModelSummary`
**Location**: `models.utils.ModelSummary`

Model analysis and summary generation.

#### Constructor
```python
ModelSummary(model: nn.Module)
```

#### Methods
| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `summary()` | `input_size=(3,64,128)`, `batch_size=1`, `device='cpu'` | `Dict[str, Any]` | Generate summary |
| `print_summary()` | `input_size=(3,64,128)`, `batch_size=1` | `None` | Print formatted summary |
| `save_summary()` | `filepath`, `**kwargs` | `None` | Save summary to file |

## Factory Functions

### `create_model()`
**Location**: `models.model_factory.create_model`

Create OCR model from preset or configuration.

```python
create_model(
    config_path: Optional[str] = None,
    preset: Optional[str] = None,
    **kwargs
) -> KhmerDigitsOCR
```

**Examples:**
```python
# From preset
model = create_model(preset='medium')

# From config file
model = create_model(config_path='config/model.yaml')

# With overrides
model = create_model(preset='small', dropout=0.2)
```

---

### `load_model()`
**Location**: `models.model_factory.load_model`

Load model from checkpoint.

```python
load_model(
    checkpoint_path: str,
    map_location: Optional[str] = None
) -> KhmerDigitsOCR
```

---

### `save_model()`
**Location**: `models.model_factory.save_model`

Save model checkpoint.

```python
save_model(
    model: KhmerDigitsOCR,
    checkpoint_path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: Optional[int] = None,
    loss: Optional[float] = None,
    metrics: Optional[Dict[str, float]] = None
)
```

## Component Classes

### CNN Backbones

#### `ResNetBackbone`
**Location**: `models.backbone.ResNetBackbone`

```python
ResNetBackbone(
    feature_size: int = 512,
    pretrained: bool = True,
    dropout: float = 0.1
)
```

**Output Shape**: `[batch_size, 8, feature_size]`

#### `EfficientNetBackbone`
**Location**: `models.backbone.EfficientNetBackbone`

```python
EfficientNetBackbone(
    feature_size: int = 512,
    pretrained: bool = True,
    dropout: float = 0.1
)
```

**Requirements**: `pip install efficientnet-pytorch`

#### `create_backbone()`
```python
create_backbone(
    backbone_type: str,  # 'resnet18' or 'efficientnet-b0'
    feature_size: int = 512,
    pretrained: bool = True,
    **kwargs
) -> CNNBackbone
```

---

### RNN Encoders

#### `BiLSTMEncoder`
**Location**: `models.encoder.BiLSTMEncoder`

```python
BiLSTMEncoder(
    input_size: int,
    hidden_size: int,
    num_layers: int = 2,
    dropout: float = 0.1,
    batch_first: bool = True
)
```

**Input**: `[batch_size, seq_len, input_size]`  
**Output**: `encoded_features [batch_size, seq_len, hidden_size]`, `final_hidden [batch_size, hidden_size]`

#### `create_encoder()`
```python
create_encoder(
    encoder_type: str,  # 'bilstm'
    input_size: int,
    hidden_size: int,
    **kwargs
) -> RNNEncoder
```

---

### Attention Mechanisms

#### `BahdanauAttention`
**Location**: `models.attention.BahdanauAttention`

```python
BahdanauAttention(
    encoder_hidden_size: int,
    decoder_hidden_size: int,
    attention_size: int = 256
)
```

**Method**: `forward(encoder_states, decoder_state, encoder_mask=None)`  
**Returns**: `context_vector [batch_size, encoder_hidden_size]`, `attention_weights [batch_size, seq_len]`

---

### RNN Decoders

#### `AttentionDecoder`
**Location**: `models.decoder.AttentionDecoder`

```python
AttentionDecoder(
    vocab_size: int,
    encoder_hidden_size: int,
    decoder_hidden_size: int = 256,
    num_layers: int = 1,
    dropout: float = 0.1,
    attention_size: int = 256
)
```

#### `CTCDecoder`
**Location**: `models.decoder.CTCDecoder`

```python
CTCDecoder(
    vocab_size: int,
    encoder_hidden_size: int,
    dropout: float = 0.1
)
```

#### `create_decoder()`
```python
create_decoder(
    decoder_type: str,  # 'attention' or 'ctc'
    vocab_size: int,
    encoder_hidden_size: int,
    **kwargs
) -> nn.Module
```

## Utility Functions

### Parameter Analysis

#### `count_parameters()`
**Location**: `models.utils.count_parameters`

```python
count_parameters(
    model: nn.Module,
    trainable_only: bool = False
) -> int
```

#### `get_model_info()`
**Location**: `models.utils.get_model_info`

```python
get_model_info(model: nn.Module) -> Dict[str, Any]
```

**Returns**:
- `model_class`: Model class name
- `total_parameters`: Total parameter count
- `trainable_parameters`: Trainable parameter count
- `model_size_mb`: Model size in MB
- `devices`: List of devices
- `parameter_dtypes`: Parameter data types
- `training_mode`: Whether in training mode

---

### Performance Analysis

#### `profile_model()`
**Location**: `models.utils.profile_model`

```python
profile_model(
    model: nn.Module,
    input_size: Tuple[int, ...] = (3, 64, 128),
    batch_size: int = 1,
    num_runs: int = 10,
    device: str = 'cpu'
) -> Dict[str, float]
```

**Returns**:
- `avg_inference_time_ms`: Average inference time
- `throughput_samples_per_sec`: Throughput
- `memory_allocated_mb`: Memory allocated (CUDA)
- `memory_reserved_mb`: Memory reserved (CUDA)

#### `compare_models()`
**Location**: `models.utils.compare_models`

```python
compare_models(
    models: Dict[str, nn.Module],
    input_size: Tuple[int, ...] = (3, 64, 128),
    batch_size: int = 1
) -> Dict[str, Dict[str, Any]]
```

---

### Visualization

#### `visualize_architecture()`
**Location**: `models.utils.visualize_architecture`

```python
visualize_architecture(
    model: nn.Module,
    input_size: Tuple[int, ...] = (3, 64, 128),
    save_path: Optional[str] = None
) -> str
```

## Configuration

### Character Set
```python
character_mappings = {
    '០': 0, '១': 1, '២': 2, '៣': 3, '៤': 4,
    '៥': 5, '៦': 6, '៧': 7, '៨': 8, '៩': 9,
    '<EOS>': 10, '<PAD>': 11, '<BLANK>': 12
}
```

### Model Presets Configuration

| Parameter | Small | Medium | Large | CTC Small |
|-----------|-------|--------|-------|-----------|
| CNN Type | ResNet-18 | ResNet-18 | EfficientNet-B0 | ResNet-18 |
| Feature Size | 256 | 512 | 512 | 256 |
| Encoder Hidden | 128 | 256 | 512 | 128 |
| Decoder Hidden | 128 | 256 | 512 | - |
| Encoder Layers | 1 | 2 | 3 | 1 |
| Decoder Layers | 1 | 1 | 2 | - |
| Parameters | 12.5M | 16.2M | 30M+ | 12.3M |

## Examples

### Basic Usage
```python
from models import create_model
import torch

# Create model
model = create_model(preset='medium')

# Make prediction
image = torch.randn(1, 3, 64, 128)
text = model.predict(image)[0]
print(f"Predicted: {text}")
```

### Training Setup
```python
from models import create_model
from torch.utils.data import DataLoader

# Create model
model = create_model(preset='medium')
model.train()

# Setup optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss(ignore_index=11)

# Training loop
for images, labels, _ in dataloader:
    optimizer.zero_grad()
    predictions = model(images, labels)
    loss = criterion(predictions.view(-1, 13), labels.view(-1))
    loss.backward()
    optimizer.step()
```

### Model Analysis
```python
from models import create_model
from models.utils import ModelSummary, get_model_info

# Create and analyze model
model = create_model(preset='medium')

# Basic info
info = get_model_info(model)
print(f"Parameters: {info['total_parameters']:,}")

# Detailed summary
summary = ModelSummary(model)
summary.print_summary()

# Performance profiling
from models.utils import profile_model
metrics = profile_model(model)
print(f"Inference time: {metrics['avg_inference_time_ms']:.2f} ms")
```

### Custom Configuration
```python
# Create custom model
model = create_model(
    preset='medium',
    encoder_hidden_size=512,
    dropout=0.2,
    num_encoder_layers=3
)

# Save custom configuration
model.save_config('custom_config.yaml')

# Load from custom config
model = create_model(config_path='custom_config.yaml')
```

### Model Checkpoint Management
```python
from models.model_factory import save_model, load_model

# Save checkpoint
save_model(
    model=model,
    checkpoint_path='checkpoints/epoch_10.pth',
    optimizer=optimizer,
    epoch=10,
    loss=0.45,
    metrics={'accuracy': 0.95}
)

# Load checkpoint
model = load_model('checkpoints/epoch_10.pth')
```

## Error Codes and Troubleshooting

### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `ImportError: No module named efficientnet_pytorch` | EfficientNet not installed | `pip install efficientnet-pytorch` |
| `RuntimeError: view of a leaf Variable` | In-place operation issue | Update to latest code version |
| `RuntimeError: Expected tensor for argument #1` | Wrong input type | Ensure input is `torch.Tensor` |
| `RuntimeError: size mismatch` | Wrong input shape | Use shape `[batch, 3, 64, 128]` |

### Debug Commands
```python
# Check model structure
from models.utils import visualize_architecture
print(visualize_architecture(model))

# Verify input shapes
print(f"Expected input: [batch_size, 3, 64, 128]")
print(f"Your input: {images.shape}")

# Check parameter count
from models.utils import count_parameters
print(f"Total parameters: {count_parameters(model):,}")
```

This API reference provides quick access to all classes, methods, and functions in the models module. For detailed explanations and examples, refer to the main documentation. 