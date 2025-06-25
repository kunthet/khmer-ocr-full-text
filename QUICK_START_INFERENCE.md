# Quick Start: Khmer OCR Inference

Your `conservative_small.pth` model is ready for inference! Here's how to use it:

## ✅ Quick Test (Start Here)

```bash
# Test if everything works
python test_inference.py

# Generate 5 samples and test accuracy  
python run_inference.py --generate --num_samples 5 --confidence
```

## 🎯 Basic Usage

### Single Image
```bash
python run_inference.py --image your_image.png --confidence
```

### Multiple Images from Directory
```bash
python run_inference.py --directory /path/to/images/ --batch_size 16
```

### Generate Test Samples
```bash
python run_inference.py --generate --num_samples 10 --confidence
```

## 📊 Your Model Performance

- **Model**: Conservative Small (12.5M parameters, 47.6 MB)
- **Training**: 45 epochs completed
- **Test Accuracy**: 80-100% on generated samples
- **Supported**: Khmer digits ០-៩

## 🔧 Python API Usage

```python
from src.inference.inference_engine import KhmerOCRInference

# Load your trained model
engine = KhmerOCRInference(
    checkpoint_path="training_output/checkpoints/conservative_small.pth"
)

# Predict single image
prediction = engine.predict_single("image.png")
print(f"Predicted: {prediction}")

# With confidence score
prediction, confidence = engine.predict_single("image.png", return_confidence=True)
print(f"Predicted: {prediction} (confidence: {confidence:.3f})")
```

## 📁 Generated Files

After running tests, you'll find:
- `inference_output/generated_samples/` - Test images
- `test_samples/` - Additional test images
- Results saved as JSON files automatically

## 🚀 Next Steps

1. Test with your own Khmer digit images
2. Use the Python API in your applications  
3. Process larger image collections
4. Integrate into production systems

Your inference system is ready to use! 🎉 