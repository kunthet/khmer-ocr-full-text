#!/usr/bin/env python3
"""
Simple test script for Khmer OCR inference engine.

This script provides a quick way to test if the inference setup is working correctly
with your trained conservative_small.pth model.
"""

import os
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Try to import the inference engine
try:
    from inference.inference_engine import KhmerOCRInference, setup_logging
    print("✓ Successfully imported inference engine")
except ImportError as e:
    print(f"✗ Failed to import inference engine: {e}")
    sys.exit(1)

def test_model_loading():
    """Test if the model can be loaded successfully."""
    print("\n" + "="*50)
    print("TESTING MODEL LOADING")
    print("="*50)
    
    checkpoint_path = "training_output/checkpoints/conservative_small.pth"
    
    if not os.path.exists(checkpoint_path):
        print(f"✗ Checkpoint file not found: {checkpoint_path}")
        return False
    
    try:
        # Setup logging
        setup_logging(logging.INFO)
        
        # Initialize inference engine
        print(f"Loading model from: {checkpoint_path}")
        inference_engine = KhmerOCRInference(
            checkpoint_path=checkpoint_path,
            model_preset="small"
        )
        
        # Get model info
        model_info = inference_engine.get_model_info()
        
        print("✓ Model loaded successfully!")
        print(f"  - Device: {model_info.get('device', 'N/A')}")
        print(f"  - Parameters: {model_info.get('total_parameters', 'N/A'):,}")
        print(f"  - Model Size: {model_info.get('model_size_mb', 'N/A'):.1f} MB")
        
        if 'epoch' in model_info:
            print(f"  - Trained Epochs: {model_info['epoch']}")
            print(f"  - Train Char Accuracy: {model_info.get('train_char_accuracy', 'N/A')}")
            print(f"  - Val Char Accuracy: {model_info.get('val_char_accuracy', 'N/A')}")
            print(f"  - Train Seq Accuracy: {model_info.get('train_seq_accuracy', 'N/A')}")
            print(f"  - Val Seq Accuracy: {model_info.get('val_seq_accuracy', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return False

def test_sample_generation():
    """Test generating and processing sample images."""
    print("\n" + "="*50)
    print("TESTING SAMPLE GENERATION & INFERENCE")
    print("="*50)
    
    try:
        from modules.synthetic_data_generator.generator import SyntheticDataGenerator
        
        # Generate a few sample images - need to create generator with proper config
        generator = SyntheticDataGenerator(
            config_path="config/model_config.yaml",
            fonts_dir="src/fonts",
            output_dir="test_samples"
        )
        print("✓ Successfully imported data generator")
        
        # Create test samples directory
        test_dir = Path("test_samples")
        test_dir.mkdir(exist_ok=True)
        
        # Generate test images
        test_sequences = ["১२३", "៤៥", "៦៧៨៩", "០", "១២៣៤៥"]
        generated_files = []
        
        print("Generating test images...")
        for i, sequence in enumerate(test_sequences):
            try:
                image, metadata = generator.generate_single_image(
                    text=sequence,
                    apply_augmentation=False
                )
                
                filename = f"test_{i:02d}_{sequence}.png"
                filepath = test_dir / filename
                image.save(filepath)
                generated_files.append(str(filepath))
                print(f"  - Generated: {filename}")
                
            except Exception as e:
                print(f"  - Failed to generate {sequence}: {e}")
        
        return generated_files
        
    except ImportError as e:
        print(f"✗ Failed to import data generator: {e}")
        return []
    except Exception as e:
        print(f"✗ Failed to generate samples: {e}")
        return []

def test_inference(image_files):
    """Test inference on generated images."""
    if not image_files:
        print("No images to test inference on")
        return
    
    print("\n" + "="*50)
    print("TESTING INFERENCE")
    print("="*50)
    
    try:
        checkpoint_path = "training_output/checkpoints/conservative_small.pth"
        inference_engine = KhmerOCRInference(
            checkpoint_path=checkpoint_path,
            model_preset="small"
        )
        
        print(f"Running inference on {len(image_files)} images...")
        
        for image_file in image_files:
            try:
                # Extract expected sequence from filename
                filename = os.path.basename(image_file)
                expected = filename.split('_')[-1].replace('.png', '')
                
                # Run inference
                prediction, confidence = inference_engine.predict_single(
                    image_file, 
                    return_confidence=True
                )
                
                # Check if correct
                is_correct = prediction == expected
                status = "✓" if is_correct else "✗"
                
                print(f"  {status} {filename}")
                print(f"     Expected: {expected}")
                print(f"     Predicted: {prediction} (confidence: {confidence:.3f})")
                
            except Exception as e:
                print(f"  ✗ Failed to process {image_file}: {e}")
        
        print("✓ Inference testing completed")
        
    except Exception as e:
        print(f"✗ Inference testing failed: {e}")

def main():
    print("Khmer OCR Inference Test Script")
    print("This script tests if your trained model can be loaded and used for inference.")
    
    # Test 1: Model loading
    if not test_model_loading():
        print("\n❌ Model loading failed. Please check your checkpoint file.")
        sys.exit(1)
    
    # Test 2: Sample generation
    image_files = test_sample_generation()
    
    # Test 3: Inference
    if image_files:
        test_inference(image_files)
    
    print("\n" + "="*50)
    print("✓ ALL TESTS COMPLETED")
    print("="*50)
    print("\nYour inference setup is working! You can now use:")
    print("  python run_inference.py --help")
    print("\nFor example:")
    print("  python run_inference.py --generate --num_samples 5 --confidence")
    print("  python run_inference.py --image path/to/your/image.png --visualize")

if __name__ == "__main__":
    main() 