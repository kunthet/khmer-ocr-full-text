# Khmer OCR Inference Guide

This guide shows how to use your trained `conservative_small.pth` model for inference on new images.

## Quick Start

1. **Test your setup** (recommended first step):
```bash
python test_inference.py
```

2. **Generate and test samples**:
```bash
python run_inference.py --generate --num_samples 10 --confidence
```

3. **Process a single image**:
```bash
python run_inference.py --image path/to/your/image.png --confidence --visualize
```

4. **Process a directory of images**:
```bash
python run_inference.py --directory path/to/images/ --batch_size 16
```

## Your Trained Model

Your `conservative_small.pth` model was trained with these performance metrics:
- **Train Char Accuracy**: 96.4%
- **Val Char Accuracy**: 89.7%
- **Train Sequence Accuracy**: 85.3%
- **Val Sequence Accuracy**: 78.6%
- **Trained Epochs**: 45

The model uses a "small" architecture preset which is efficient while maintaining good accuracy.

## Available Scripts

### 1. `test_inference.py` - Quick Validation
Tests if your inference setup is working correctly:
- ✅ Loads your trained model
- ✅ Generates sample images
- ✅ Runs inference and shows accuracy
- ✅ Validates the complete pipeline

### 2. `run_inference.py` - Full Inference CLI
Comprehensive inference script with many options:

#### Input Options
- `--image path.png` - Process single image
- `--directory path/` - Process all images in directory
- `--generate` - Generate test samples automatically

#### Model Options
- `--checkpoint path.pth` - Custom checkpoint (default: your conservative_small.pth)
- `--device cpu/cuda/auto` - Device selection
- `--model_preset small/medium/large` - Model architecture preset

#### Output Options
- `--confidence` - Show confidence scores
- `--visualize` - Create annotated images
- `--output_dir results/` - Output directory
- `--batch_size 16` - Batch processing size

#### Examples
```bash
# Generate samples and test accuracy
python run_inference.py --generate --num_samples 20 --confidence

# Process single image with visualization
python run_inference.py --image test.png --visualize --confidence

# Batch process directory
python run_inference.py --directory /path/to/images --batch_size 32

# Use different checkpoint
python run_inference.py --checkpoint other_model.pth --image test.png
```

## Inference Engine API

You can also use the inference engine directly in your Python code:

```python
from src.inference.inference_engine import KhmerOCRInference

# Initialize inference engine
engine = KhmerOCRInference(
    checkpoint_path="training_output/checkpoints/conservative_small.pth",
    model_preset="small"
)

# Single image prediction
prediction = engine.predict_single("image.png")
print(f"Predicted text: {prediction}")

# With confidence score
prediction, confidence = engine.predict_single("image.png", return_confidence=True)
print(f"Predicted: {prediction} (confidence: {confidence:.3f})")

# Batch prediction
image_list = ["img1.png", "img2.png", "img3.png"]
predictions = engine.predict_batch(image_list)

# Directory processing
results = engine.predict_from_directory("image_folder/")

# Get model information
info = engine.get_model_info()
print(f"Model has {info['total_parameters']:,} parameters")
```

## Supported Input Formats

- **Image formats**: PNG, JPG, JPEG
- **Input types**: File paths, PIL Images, NumPy arrays
- **Preprocessing**: Automatic resizing to 128x64, normalization
- **Character set**: Khmer digits ០-៩ plus special tokens

## Expected Model Performance

Based on your training results, you can expect:
- **High accuracy** on clear, well-formed Khmer digits
- **Good generalization** to different fonts and styles
- **Robust performance** on synthetic and real images
- **Fast inference** - suitable for real-time applications

## Troubleshooting

### Model Loading Issues
```bash
# Check if checkpoint exists
ls -la training_output/checkpoints/conservative_small.pth

# Test with verbose logging
python run_inference.py --generate --verbose
```

### Import Errors
```bash
# Make sure you're in the project root directory
pwd  # Should show khmer-ocr-digits/

# Check Python path
python -c "import sys; print('\n'.join(sys.path))"
```

### Performance Issues
```bash
# Use CPU explicitly if CUDA issues
python run_inference.py --device cpu --generate

# Reduce batch size for memory issues
python run_inference.py --directory images/ --batch_size 4
```

## File Structure

```
khmer-ocr-digits/
├── training_output/checkpoints/
│   ├── conservative_small.pth          # Your trained model
│   └── conservative_small.txt          # Training log
├── src/inference/
│   ├── __init__.py
│   └── inference_engine.py             # Main inference class
├── run_inference.py                    # Full CLI script
├── test_inference.py                   # Quick test script
└── INFERENCE_README.md                 # This file
```

## Next Steps

1. **Test with your own images**: Try the inference on real Khmer digit images
2. **Integrate into applications**: Use the Python API in your projects
3. **Evaluate on your dataset**: Process your validation images to assess real-world performance
4. **Fine-tune if needed**: If performance isn't sufficient, consider additional training

## Support

If you encounter issues:
1. Run `test_inference.py` first to isolate the problem
2. Check the error messages with `--verbose` flag
3. Verify your checkpoint file exists and is not corrupted
4. Ensure all dependencies are installed correctly 