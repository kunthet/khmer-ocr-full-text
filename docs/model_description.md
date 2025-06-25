# Khmer Digits OCR Prototype - Model Description

## Overview

This prototype focuses on recognizing Khmer digits (០១២៣៤៥៦៧៨៩) in synthetic images containing 1-8 digit sequences. The system uses a hybrid CNN-RNN architecture optimized for sequential digit recognition with attention mechanisms.

## Model Architecture

### 1. Convolutional Neural Network (CNN) Backbone
- **Purpose**: Feature extraction from input images
- **Architecture**: Modified ResNet-18 or EfficientNet-B0
- **Input**: RGB images (128x64 pixels, normalized)
- **Output**: Feature maps (512-dimensional feature vectors)

### 2. Sequence-to-Sequence Model
- **Encoder**: Bidirectional LSTM (256 hidden units)
- **Decoder**: LSTM with attention mechanism (256 hidden units)
- **Attention**: Bahdanau attention for focusing on relevant image regions

### 3. Character Classification Head
- **Output Layer**: Softmax over 13 classes (10 Khmer digits + special tokens)
- **Loss Function**: Cross-entropy loss with sequence padding

## Technical Specifications

### Character Set
- **Khmer Digits**: ០ (0), ១ (1), ២ (2), ៣ (3), ៤ (4), ៥ (5), ៦ (6), ៧ (7), ៨ (8), ៩ (9)
- **Special Tokens**: `<EOS>` (End of Sequence), `<PAD>` (Padding), `<BLANK>` (CTC Blank)
- **Total Classes**: 13 (including special tokens)
- **Scalability**: Architecture designed for easy expansion to 84+ characters for full Khmer script

### Input Specifications
- **Image Size**: 128x64 pixels (width x height)
- **Color Space**: RGB
- **Normalization**: Mean=[0.485, 0.456, 0.406], Std=[0.229, 0.224, 0.225]
- **Sequence Length**: Variable (1-8 digits) with maximum length of 8
- **Unicode Handling**: NFC normalization for consistent character encoding

### Model Parameters
- **CNN Features**: ~11M parameters (ResNet-18 backbone)
- **RNN Components**: ~2M parameters
- **Total Model Size**: ~13M parameters
- **Memory Requirements**: ~50MB for inference

## Training Strategy

### 1. Synthetic Data Generation
- **Font Variations**: Multiple Khmer fonts (Khmer OS, Khmer Unicode, etc.)
- **Backgrounds**: Solid colors, textures, and simple patterns
- **Augmentations**: 
  - Rotation: ±15 degrees
  - Scaling: 0.8-1.2x
  - Gaussian noise: σ=0.01
  - Brightness/contrast variations: ±20%
  - Perspective transformations: slight skewing

### 2. Training Configuration
- **Optimizer**: AdamW with learning rate scheduling
- **Learning Rate**: 1e-3 with cosine annealing
- **Batch Size**: 32-64 samples
- **Epochs**: 50-100 epochs
- **Validation Split**: 20% of generated data

### 3. Loss Function
```
Total Loss = CrossEntropy Loss + 0.1 * Sequence Length Penalty
```

## Performance Metrics

### Primary Metrics
- **Character Accuracy**: Percentage of correctly recognized individual digits
- **Sequence Accuracy**: Percentage of completely correct digit sequences
- **Edit Distance**: Average Levenshtein distance between predicted and ground truth

### Secondary Metrics
- **Inference Speed**: Target <100ms per image on CPU
- **Model Size**: Target <20MB for deployment
- **Memory Usage**: Target <512MB RAM during inference

## Implementation Framework

### Core Technologies
- **Deep Learning**: PyTorch 2.0+ or TensorFlow 2.10+
- **Computer Vision**: OpenCV for image preprocessing
- **Data Generation**: Pillow (PIL) for synthetic image creation
- **Text Processing**: Custom tokenization for Khmer digits

### Dependencies
- `torch` or `tensorflow`
- `torchvision` or `tensorflow-addons`
- `opencv-python`
- `pillow`
- `numpy`
- `matplotlib` (for visualization)
- `tensorboard` (for monitoring)
- `unicodedata` (for Unicode normalization)

## Model Advantages

1. **Simplicity**: Focused on a small character set (10 digits)
2. **Efficiency**: Lightweight architecture suitable for deployment
3. **Robustness**: Attention mechanism handles variable sequence lengths
4. **Scalability**: Architecture can be extended to full Khmer character set
5. **Synthetic Training**: No need for manual data collection and annotation

## Limitations and Considerations

1. **Domain Gap**: Synthetic data may not capture all real-world variations
2. **Font Dependency**: Performance may vary with unseen fonts
3. **Background Complexity**: Limited to simple backgrounds in prototype
4. **Sequence Length**: Fixed maximum length of 8 digits (allows future flexibility)
5. **Context**: No semantic understanding of digit sequences

## Future Enhancements

1. **Transformer Architecture**: Replace RNN with transformer encoder-decoder
2. **Data Augmentation**: Add more sophisticated augmentation techniques
3. **Multi-Scale Features**: Implement feature pyramid networks
4. **Real Data Integration**: Fine-tune on real Khmer document images
5. **End-to-End Training**: Joint optimization of feature extraction and sequence modeling

## Validation Strategy

### Test Scenarios
1. **Clean Images**: High-quality synthetic images
2. **Noisy Images**: Images with various noise types
3. **Font Variations**: Unseen fonts not used in training
4. **Degraded Quality**: Blurred, low-resolution images
5. **Edge Cases**: Single digits, maximum length sequences (up to 8 digits)

### Success Criteria
- **Character Accuracy**: >95% on clean synthetic data
- **Sequence Accuracy**: >90% on clean synthetic data
- **Robustness**: >85% accuracy on noisy/degraded images
- **Speed**: <100ms inference time on standard CPU

This prototype serves as a foundation for developing a comprehensive Khmer OCR system, providing insights into model architecture, training strategies, and performance characteristics specific to Khmer script recognition. 