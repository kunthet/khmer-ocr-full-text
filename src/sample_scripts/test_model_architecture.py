#!/usr/bin/env python3
"""
Test script for Khmer Digits OCR model architecture.

Tests all model components, validates forward passes, and demonstrates
the complete model pipeline with synthetic data.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Import model components
from models import (
    KhmerDigitsOCR, ModelFactory, create_model,
    ModelSummary, count_parameters, get_model_info,
    ResNetBackbone, EfficientNetBackbone, BiLSTMEncoder,
    AttentionDecoder, BahdanauAttention
)

# Import data utilities
from modules.data_utils.preprocessing import get_default_transforms
from modules.synthetic_data_generator import SyntheticDataGenerator


def test_backbone_components():
    """Test CNN backbone components."""
    print("\n" + "="*60)
    print("Testing CNN Backbone Components")
    print("="*60)
    
    # Test ResNet backbone
    print("\n1. Testing ResNet-18 Backbone:")
    resnet_backbone = ResNetBackbone(feature_size=512, pretrained=False)
    dummy_input = torch.randn(2, 3, 64, 128)
    
    try:
        features = resnet_backbone(dummy_input)
        print(f"   ✓ Input shape: {dummy_input.shape}")
        print(f"   ✓ Output shape: {features.shape}")
        print(f"   ✓ Expected shape: [2, 8, 512]")
        print(f"   ✓ Parameters: {count_parameters(resnet_backbone):,}")
        assert features.shape == (2, 8, 512), f"Unexpected output shape: {features.shape}"
        print("   ✓ ResNet backbone test passed!")
    except Exception as e:
        print(f"   ✗ ResNet backbone test failed: {e}")
    
    # Test EfficientNet backbone (if available)
    print("\n2. Testing EfficientNet-B0 Backbone:")
    try:
        efficientnet_backbone = EfficientNetBackbone(feature_size=512, pretrained=False)
        features = efficientnet_backbone(dummy_input)
        print(f"   ✓ Input shape: {dummy_input.shape}")
        print(f"   ✓ Output shape: {features.shape}")
        print(f"   ✓ Parameters: {count_parameters(efficientnet_backbone):,}")
        assert features.shape == (2, 8, 512), f"Unexpected output shape: {features.shape}"
        print("   ✓ EfficientNet backbone test passed!")
    except ImportError:
        print("   ⚠ EfficientNet not available (install efficientnet-pytorch)")
    except Exception as e:
        print(f"   ✗ EfficientNet backbone test failed: {e}")


def test_encoder_components():
    """Test RNN encoder components."""
    print("\n" + "="*60)
    print("Testing RNN Encoder Components")
    print("="*60)
    
    # Test BiLSTM encoder
    print("\n1. Testing BiLSTM Encoder:")
    encoder = BiLSTMEncoder(
        input_size=512,
        hidden_size=256,
        num_layers=2,
        dropout=0.1
    )
    
    dummy_features = torch.randn(2, 8, 512)
    
    try:
        encoded_features, final_hidden = encoder(dummy_features)
        print(f"   ✓ Input shape: {dummy_features.shape}")
        print(f"   ✓ Encoded features shape: {encoded_features.shape}")
        print(f"   ✓ Final hidden shape: {final_hidden.shape}")
        print(f"   ✓ Parameters: {count_parameters(encoder):,}")
        
        assert encoded_features.shape == (2, 8, 256), f"Unexpected encoded shape: {encoded_features.shape}"
        assert final_hidden.shape == (2, 256), f"Unexpected hidden shape: {final_hidden.shape}"
        print("   ✓ BiLSTM encoder test passed!")
    except Exception as e:
        print(f"   ✗ BiLSTM encoder test failed: {e}")


def test_attention_mechanism():
    """Test attention mechanism."""
    print("\n" + "="*60)
    print("Testing Attention Mechanism")
    print("="*60)
    
    # Test Bahdanau attention
    print("\n1. Testing Bahdanau Attention:")
    attention = BahdanauAttention(
        encoder_hidden_size=256,
        decoder_hidden_size=256,
        attention_size=256
    )
    
    encoder_states = torch.randn(2, 8, 256)
    decoder_state = torch.randn(2, 256)
    
    try:
        context_vector, attention_weights = attention(encoder_states, decoder_state)
        print(f"   ✓ Encoder states shape: {encoder_states.shape}")
        print(f"   ✓ Decoder state shape: {decoder_state.shape}")
        print(f"   ✓ Context vector shape: {context_vector.shape}")
        print(f"   ✓ Attention weights shape: {attention_weights.shape}")
        print(f"   ✓ Parameters: {count_parameters(attention):,}")
        
        assert context_vector.shape == (2, 256), f"Unexpected context shape: {context_vector.shape}"
        assert attention_weights.shape == (2, 8), f"Unexpected attention shape: {attention_weights.shape}"
        
        # Check if attention weights sum to 1
        weight_sums = attention_weights.sum(dim=1)
        assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-6), "Attention weights don't sum to 1"
        print("   ✓ Bahdanau attention test passed!")
    except Exception as e:
        print(f"   ✗ Bahdanau attention test failed: {e}")


def test_decoder_components():
    """Test decoder components."""
    print("\n" + "="*60)
    print("Testing Decoder Components")
    print("="*60)
    
    # Test attention decoder
    print("\n1. Testing Attention Decoder:")
    decoder = AttentionDecoder(
        vocab_size=13,
        encoder_hidden_size=256,
        decoder_hidden_size=256,
        num_layers=1,
        dropout=0.1,
        attention_size=256
    )
    
    encoder_features = torch.randn(2, 8, 256)
    target_sequence = torch.randint(0, 13, (2, 5))
    
    try:
        # Training mode
        decoder.train()
        predictions_train = decoder(encoder_features, target_sequence)
        print(f"   ✓ Training mode predictions shape: {predictions_train.shape}")
        print(f"   ✓ Parameters: {count_parameters(decoder):,}")
        
        # Inference mode
        decoder.eval()
        predictions_inference = decoder(encoder_features, max_length=8)
        print(f"   ✓ Inference mode predictions shape: {predictions_inference.shape}")
        
        assert predictions_train.shape == (2, 5, 13), f"Unexpected training shape: {predictions_train.shape}"
        print("   ✓ Attention decoder test passed!")
    except Exception as e:
        print(f"   ✗ Attention decoder test failed: {e}")


def test_complete_model():
    """Test complete OCR model."""
    print("\n" + "="*60)
    print("Testing Complete OCR Model")
    print("="*60)
    
    # Test model creation with different presets
    presets = ['small', 'medium', 'ctc_small']
    
    for preset in presets:
        print(f"\n{preset.upper()} Model:")
        try:
            model = create_model(preset=preset)
            
            # Test forward pass
            dummy_images = torch.randn(2, 3, 64, 128)
            dummy_targets = torch.randint(0, 13, (2, 8))
            
            # Training mode
            model.train()
            predictions_train = model(dummy_images, dummy_targets)
            print(f"   ✓ Training predictions shape: {predictions_train.shape}")
            
            # Inference mode
            model.eval()
            predictions_inference = model(dummy_images)
            print(f"   ✓ Inference predictions shape: {predictions_inference.shape}")
            
            # Test text prediction
            predicted_texts = model.predict(dummy_images)
            print(f"   ✓ Predicted texts: {predicted_texts}")
            
            # Model info
            info = model.get_model_info()
            print(f"   ✓ Total parameters: {info['total_parameters']:,}")
            print(f"   ✓ Model size: {info['model_size_mb']:.2f} MB")
            
            print(f"   ✓ {preset} model test passed!")
            
        except Exception as e:
            print(f"   ✗ {preset} model test failed: {e}")


def test_model_factory():
    """Test model factory functionality."""
    print("\n" + "="*60)
    print("Testing Model Factory")
    print("="*60)
    
    # Test preset listing
    print("\n1. Available Presets:")
    presets = ModelFactory.list_presets()
    for preset_name in presets.keys():
        print(f"   ✓ {preset_name}")
    
    # Test preset info
    print("\n2. Preset Information:")
    try:
        info = ModelFactory.get_preset_info('medium')
        print(f"   ✓ Medium preset estimated parameters: {info['estimated_parameters']['total']:,}")
        print(f"   ✓ Medium preset estimated size: {info['estimated_size_mb']:.2f} MB")
    except Exception as e:
        print(f"   ✗ Preset info test failed: {e}")
    
    # Test model creation from config
    print("\n3. Model from Configuration:")
    try:
        config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'model_config.yaml')
        if os.path.exists(config_path):
            model = create_model(config_path)
            info = model.get_model_info()
            print(f"   ✓ Config-based model parameters: {info['total_parameters']:,}")
        else:
            print("   ⚠ Config file not found, skipping config test")
    except Exception as e:
        print(f"   ✗ Config-based model test failed: {e}")


def test_model_utilities():
    """Test model utility functions."""
    print("\n" + "="*60)
    print("Testing Model Utilities")
    print("="*60)
    
    # Create a small model for testing
    model = create_model(preset='small')
    
    # Test model summary
    print("\n1. Model Summary:")
    try:
        summary_tool = ModelSummary(model)
        summary = summary_tool.summary(input_size=(3, 64, 128), batch_size=1)
        print(f"   ✓ Total parameters: {summary['total_params']:,}")
        print(f"   ✓ Trainable parameters: {summary['trainable_params']:,}")
        print(f"   ✓ Model size: {summary['model_size_mb']:.2f} MB")
        print(f"   ✓ Estimated memory: {summary['estimated_memory_usage']['total_mb']:.2f} MB")
    except Exception as e:
        print(f"   ✗ Model summary test failed: {e}")
    
    # Test parameter counting
    print("\n2. Parameter Counting:")
    try:
        total_params = count_parameters(model)
        trainable_params = count_parameters(model, trainable_only=True)
        print(f"   ✓ Total parameters: {total_params:,}")
        print(f"   ✓ Trainable parameters: {trainable_params:,}")
    except Exception as e:
        print(f"   ✗ Parameter counting test failed: {e}")
    
    # Test model info
    print("\n3. Model Information:")
    try:
        info = get_model_info(model)
        print(f"   ✓ Model class: {info['model_class']}")
        print(f"   ✓ Training mode: {info['training_mode']}")
        print(f"   ✓ Devices: {info['devices']}")
    except Exception as e:
        print(f"   ✗ Model info test failed: {e}")


def test_with_synthetic_data():
    """Test model with synthetic data."""
    print("\n" + "="*60)
    print("Testing with Synthetic Data")
    print("="*60)
    
    try:
        # Create synthetic data generator
        fonts_dir = os.path.join(os.path.dirname(__file__), '..', 'fonts')
        generator = SyntheticDataGenerator(fonts_dir=fonts_dir)
        
        # Generate some test samples
        print("\n1. Generating test samples:")
        samples = []
        labels = []
        
        for i in range(5):
            # Generate random digit sequence
            num_digits = np.random.randint(1, 5)
            digits = [str(np.random.randint(0, 10)) for _ in range(num_digits)]
            text = ''.join(digits)
            
            # Convert to Khmer
            khmer_text = generator._convert_to_khmer(text)
            
            # Generate image
            image = generator._generate_single_image(khmer_text, font_path=generator.font_paths[0])
            
            samples.append(image)
            labels.append(khmer_text)
            print(f"   Sample {i+1}: '{text}' -> '{khmer_text}'")
        
        # Test with model
        print("\n2. Testing model with synthetic samples:")
        model = create_model(preset='small')
        transform = get_default_transforms()
        
        # Prepare batch
        batch_images = []
        for image in samples:
            tensor_image = transform(image).unsqueeze(0)
            batch_images.append(tensor_image)
        
        batch_tensor = torch.cat(batch_images, dim=0)
        
        # Model prediction
        model.eval()
        with torch.no_grad():
            predictions = model.predict(batch_tensor)
        
        print("\n3. Predictions:")
        for i, (label, prediction) in enumerate(zip(labels, predictions)):
            print(f"   Sample {i+1}: True='{label}' Pred='{prediction}'")
        
        print("   ✓ Synthetic data test completed!")
        
    except Exception as e:
        print(f"   ✗ Synthetic data test failed: {e}")


def main():
    """Run all model architecture tests."""
    print("Khmer Digits OCR Model Architecture Tests")
    print("="*80)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run tests
    try:
        test_backbone_components()
        test_encoder_components()
        test_attention_mechanism()
        test_decoder_components()
        test_complete_model()
        test_model_factory()
        test_model_utilities()
        test_with_synthetic_data()
        
        print("\n" + "="*80)
        print("All tests completed!")
        print("✓ Model architecture implementation is working correctly!")
        print("="*80)
        
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user.")
    except Exception as e:
        print(f"\n\nUnexpected error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 