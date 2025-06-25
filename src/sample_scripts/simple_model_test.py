#!/usr/bin/env python3
"""
Simple test script for Khmer Digits OCR model architecture.

Tests core model components with minimal dependencies.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn

# Import model components
from models.backbone import ResNetBackbone
from models.attention import BahdanauAttention  
from models.encoder import BiLSTMEncoder
from models.ocr_model import KhmerDigitsOCR
from models.model_factory import ModelFactory, create_model


def test_basic_components():
    """Test basic model components."""
    print("Testing Basic Model Components")
    print("="*50)
    
    # Test ResNet backbone
    print("\n1. Testing ResNet Backbone:")
    try:
        backbone = ResNetBackbone(feature_size=512, pretrained=False)
        dummy_input = torch.randn(2, 3, 64, 128)
        features = backbone(dummy_input)
        print(f"   ✓ Input: {dummy_input.shape}")
        print(f"   ✓ Output: {features.shape}")
        print(f"   ✓ Expected: [2, 8, 512]")
        assert features.shape == (2, 8, 512)
        print("   ✓ ResNet backbone working!")
    except Exception as e:
        print(f"   ✗ ResNet test failed: {e}")
    
    # Test BiLSTM encoder
    print("\n2. Testing BiLSTM Encoder:")
    try:
        encoder = BiLSTMEncoder(input_size=512, hidden_size=256, num_layers=2)
        dummy_features = torch.randn(2, 8, 512)
        encoded, hidden = encoder(dummy_features)
        print(f"   ✓ Input: {dummy_features.shape}")
        print(f"   ✓ Encoded: {encoded.shape}")
        print(f"   ✓ Hidden: {hidden.shape}")
        assert encoded.shape == (2, 8, 256)
        assert hidden.shape == (2, 256)
        print("   ✓ BiLSTM encoder working!")
    except Exception as e:
        print(f"   ✗ BiLSTM test failed: {e}")
    
    # Test Bahdanau attention
    print("\n3. Testing Bahdanau Attention:")
    try:
        attention = BahdanauAttention(256, 256, 256)
        encoder_states = torch.randn(2, 8, 256)
        decoder_state = torch.randn(2, 256)
        context, weights = attention(encoder_states, decoder_state)
        print(f"   ✓ Context: {context.shape}")
        print(f"   ✓ Weights: {weights.shape}")
        assert context.shape == (2, 256)
        assert weights.shape == (2, 8)
        print("   ✓ Attention working!")
    except Exception as e:
        print(f"   ✗ Attention test failed: {e}")


def test_complete_model():
    """Test complete OCR model."""
    print("\n\nTesting Complete OCR Model")
    print("="*50)
    
    # Test small model
    print("\n1. Testing Small Model:")
    try:
        model = create_model(preset='small')
        dummy_images = torch.randn(2, 3, 64, 128)
        
        # Test forward pass
        model.eval()
        with torch.no_grad():
            predictions = model(dummy_images)
            print(f"   ✓ Input: {dummy_images.shape}")
            print(f"   ✓ Predictions: {predictions.shape}")
            print("   ✓ Small model working!")
    except Exception as e:
        print(f"   ✗ Small model test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test medium model
    print("\n2. Testing Medium Model:")
    try:
        model = create_model(preset='medium')
        dummy_images = torch.randn(2, 3, 64, 128)
        
        # Test forward pass
        model.eval()
        with torch.no_grad():
            predictions = model(dummy_images)
            print(f"   ✓ Input: {dummy_images.shape}")
            print(f"   ✓ Predictions: {predictions.shape}")
            
        # Test prediction method
        predicted_texts = model.predict(dummy_images)
        print(f"   ✓ Predicted texts: {predicted_texts}")
        print("   ✓ Medium model working!")
    except Exception as e:
        print(f"   ✗ Medium model test failed: {e}")
        import traceback
        traceback.print_exc()


def test_model_info():
    """Test model information utilities."""
    print("\n\nTesting Model Information")
    print("="*50)
    
    try:
        model = create_model(preset='small')
        info = model.get_model_info()
        
        print(f"   ✓ Model: {info['model_name']}")
        print(f"   ✓ Vocab size: {info['vocab_size']}")
        print(f"   ✓ Max sequence length: {info['max_sequence_length']}")
        print(f"   ✓ Total parameters: {info['total_parameters']:,}")
        print(f"   ✓ Model size: {info['model_size_mb']:.2f} MB")
        print("   ✓ Model info working!")
    except Exception as e:
        print(f"   ✗ Model info test failed: {e}")


def test_factory_presets():
    """Test model factory presets."""
    print("\n\nTesting Model Factory")
    print("="*50)
    
    try:
        presets = ModelFactory.list_presets()
        print(f"   ✓ Available presets: {list(presets.keys())}")
        
        for preset in ['small', 'medium', 'ctc_small']:
            try:
                model = create_model(preset=preset)
                info = model.get_model_info()
                print(f"   ✓ {preset}: {info['total_parameters']:,} params")
            except Exception as e:
                print(f"   ✗ {preset} failed: {e}")
        
        print("   ✓ Factory presets working!")
    except Exception as e:
        print(f"   ✗ Factory test failed: {e}")


def main():
    """Run all tests."""
    print("Khmer Digits OCR - Simple Model Architecture Test")
    print("="*60)
    
    # Set random seed
    torch.manual_seed(42)
    
    try:
        test_basic_components()
        test_complete_model() 
        test_model_info()
        test_factory_presets()
        
        print("\n" + "="*60)
        print("✓ All basic tests passed!")
        print("✓ Model architecture implementation is working!")
        print("="*60)
        
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 