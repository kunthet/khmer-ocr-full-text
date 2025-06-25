"""
Test script for enhanced Khmer OCR model architecture.

Tests the new KhmerTextOCR model with:
- Full Khmer character vocabulary (102+ characters)
- Hierarchical character recognition
- Enhanced attention mechanisms
- Confidence scoring
- Beam search decoding

Usage:
    python test_enhanced_model_architecture.py
"""

import sys
import os
import torch
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.models import (
    create_text_model, 
    create_digit_model,
    ModelFactory,
    KhmerTextOCR,
    KhmerDigitsOCR
)


def test_model_creation():
    """Test creating various model configurations."""
    print("=" * 60)
    print("Testing Model Creation")
    print("=" * 60)
    
    try:
        # Test digit model creation
        print("1. Creating digit model...")
        digit_model = create_digit_model('medium')
        print(f"   ✓ Digit model created: {type(digit_model).__name__}")
        print(f"   ✓ Vocab size: {digit_model.vocab_size}")
        
        # Test text model creation
        print("\n2. Creating text models...")
        
        presets = ['text_small', 'text_medium', 'text_hierarchical', 'text_fast']
        for preset in presets:
            print(f"   Testing preset: {preset}")
            text_model = create_text_model(preset)
            print(f"   ✓ {preset} created: vocab_size={text_model.vocab_size}")
            
            # Test model info
            info = text_model.get_model_info()
            print(f"   ✓ Total parameters: {info['total_parameters']:,}")
            print(f"   ✓ Hierarchical: {info['enable_hierarchical']}")
            print(f"   ✓ Confidence scoring: {info['enable_confidence_scoring']}")
            print(f"   ✓ Multi-head attention: {info['has_multi_head_attention']}")
            print()
        
        return True
        
    except Exception as e:
        print(f"   ✗ Error in model creation: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_vocabulary_support():
    """Test full Khmer character vocabulary support."""
    print("=" * 60)
    print("Testing Vocabulary Support")
    print("=" * 60)
    
    try:
        # Create text model
        model = create_text_model('text_medium')
        
        # Test vocabulary info
        vocab_info = model.get_vocabulary_info()
        print(f"1. Vocabulary Information:")
        print(f"   ✓ Total vocabulary size: {vocab_info['vocab_size']}")
        print(f"   ✓ Uses full Khmer: {vocab_info['use_full_khmer']}")
        print(f"   ✓ Total Khmer characters: {vocab_info['total_khmer_chars']}")
        
        print(f"\n2. Character Categories:")
        for category, count in vocab_info['character_categories'].items():
            print(f"   ✓ {category}: {count} characters")
        
        # Test character mapping
        print(f"\n3. Testing character mappings:")
        test_chars = ['ក', 'ខ', 'គ', 'ា', 'ិ', 'ំ', '០', '១', '២']
        for char in test_chars:
            if char in model.char_to_idx:
                idx = model.char_to_idx[char]
                decoded = model.idx_to_char[idx]
                print(f"   ✓ '{char}' → {idx} → '{decoded}'")
            else:
                print(f"   ✗ Character '{char}' not found in vocabulary")
        
        return True
        
    except Exception as e:
        print(f"   ✗ Error in vocabulary testing: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_forward_pass():
    """Test forward pass with different input sizes."""
    print("=" * 60)
    print("Testing Forward Pass")
    print("=" * 60)
    
    try:
        # Create model
        model = create_text_model('text_medium')
        model.eval()
        
        # Test different input sizes
        test_cases = [
            (1, 3, 64, 256),   # Single image
            (4, 3, 64, 256),   # Batch of 4
            (2, 3, 32, 128),   # Smaller images
        ]
        
        for i, (batch_size, channels, height, width) in enumerate(test_cases, 1):
            print(f"{i}. Testing input shape: {(batch_size, channels, height, width)}")
            
            # Create random input
            images = torch.randn(batch_size, channels, height, width)
            
            # Forward pass
            with torch.no_grad():
                # Basic forward pass
                outputs = model.forward(images)
                predictions = outputs['predictions']
                
                print(f"   ✓ Output shape: {predictions.shape}")
                print(f"   ✓ Expected: ({batch_size}, {model.max_sequence_length}, {model.vocab_size})")
                
                # Forward pass with confidence scoring
                outputs_conf = model.forward(images, return_confidence=True)
                if 'char_confidence' in outputs_conf:
                    char_conf = outputs_conf['char_confidence']
                    word_conf = outputs_conf['word_confidence']
                    print(f"   ✓ Character confidence shape: {char_conf.shape}")
                    print(f"   ✓ Word confidence shape: {word_conf.shape}")
                
                # Forward pass with hierarchical predictions
                if model.enable_hierarchical:
                    outputs_hier = model.forward(images, return_hierarchical=True)
                    if 'category_predictions' in outputs_hier:
                        cat_pred = outputs_hier['category_predictions']
                        print(f"   ✓ Category predictions shape: {cat_pred.shape}")
                
                print()
        
        return True
        
    except Exception as e:
        print(f"   ✗ Error in forward pass: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_beam_search():
    """Test beam search decoding."""
    print("=" * 60)
    print("Testing Beam Search Decoding")
    print("=" * 60)
    
    try:
        # Create model
        model = create_text_model('text_medium')
        model.eval()
        
        # Create test input
        batch_size = 2
        images = torch.randn(batch_size, 3, 64, 256)
        
        print("1. Testing beam search with different parameters:")
        
        beam_sizes = [1, 3, 5]
        length_norms = [0.0, 0.6, 1.0]
        
        for beam_size in beam_sizes:
            for length_norm in length_norms:
                print(f"   Testing beam_size={beam_size}, length_norm={length_norm}")
                
                with torch.no_grad():
                    results = model.predict_with_beam_search(
                        images, 
                        beam_size=beam_size,
                        length_normalization=length_norm,
                        return_confidence=True
                    )
                
                print(f"   ✓ Generated {len(results)} predictions")
                for i, result in enumerate(results):
                    text = result['text']
                    score = result['score']
                    print(f"   ✓ Sample {i}: '{text}' (score: {score:.4f})")
                
                print()
        
        return True
        
    except Exception as e:
        print(f"   ✗ Error in beam search: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_presets():
    """Test all available model presets."""
    print("=" * 60)
    print("Testing Model Presets")
    print("=" * 60)
    
    try:
        # List all presets
        digit_presets = ModelFactory.list_presets('digits')
        text_presets = ModelFactory.list_presets('text')
        
        print("1. Available digit presets:")
        for preset in digit_presets:
            print(f"   ✓ {preset}")
        
        print(f"\n2. Available text presets:")
        for preset in text_presets:
            print(f"   ✓ {preset}")
        
        # Test preset information
        print(f"\n3. Testing preset information:")
        test_presets = ['medium', 'text_medium', 'text_hierarchical']
        
        for preset in test_presets:
            print(f"\n   Preset: {preset}")
            try:
                info = ModelFactory.get_preset_info(preset)
                print(f"   ✓ Model type: {info['model_type']}")
                print(f"   ✓ Total parameters: {info['estimated_parameters']['total']:,}")
                print(f"   ✓ Memory estimate: {info['memory_estimate_mb']:.1f} MB")
            except Exception as e:
                print(f"   ✗ Error getting preset info: {e}")
        
        # Test preset comparison
        print(f"\n4. Testing preset comparison:")
        comparison = ModelFactory.compare_presets(['text_small', 'text_medium', 'text_large'])
        
        for preset, info in comparison.items():
            if 'error' not in info:
                params = info['estimated_parameters']['total']
                memory = info['memory_estimate_mb']
                print(f"   ✓ {preset}: {params:,} params, {memory:.1f} MB")
            else:
                print(f"   ✗ {preset}: {info['error']}")
        
        return True
        
    except Exception as e:
        print(f"   ✗ Error in preset testing: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_enhanced_attention():
    """Test enhanced attention mechanisms."""
    print("=" * 60)
    print("Testing Enhanced Attention Mechanisms")
    print("=" * 60)
    
    try:
        from src.models.attention import (
            EnhancedBahdanauAttention,
            MultiHeadAttention,
            HierarchicalAttention
        )
        
        # Test dimensions
        batch_size = 2
        seq_len = 10
        feature_size = 256
        
        # Test Enhanced Bahdanau Attention
        print("1. Testing Enhanced Bahdanau Attention:")
        enhanced_attention = EnhancedBahdanauAttention(
            encoder_hidden_size=feature_size,
            decoder_hidden_size=feature_size,
            attention_size=256,
            use_coverage=True,
            use_gating=True
        )
        
        encoder_states = torch.randn(batch_size, seq_len, feature_size)
        decoder_state = torch.randn(batch_size, feature_size)
        
        context, weights, coverage = enhanced_attention(encoder_states, decoder_state)
        print(f"   ✓ Context shape: {context.shape}")
        print(f"   ✓ Attention weights shape: {weights.shape}")
        print(f"   ✓ Coverage shape: {coverage.shape}")
        
        # Test Multi-Head Attention
        print("\n2. Testing Multi-Head Attention:")
        multi_head = MultiHeadAttention(d_model=feature_size, num_heads=8)
        
        output, attn_weights = multi_head(encoder_states, encoder_states, encoder_states)
        print(f"   ✓ Output shape: {output.shape}")
        print(f"   ✓ Attention weights shape: {attn_weights.shape}")
        
        # Test Hierarchical Attention
        print("\n3. Testing Hierarchical Attention:")
        hierarchical = HierarchicalAttention(feature_size=feature_size)
        
        fused_context, attn_info = hierarchical(encoder_states, decoder_state)
        print(f"   ✓ Fused context shape: {fused_context.shape}")
        print(f"   ✓ Character weights shape: {attn_info['char_weights'].shape}")
        print(f"   ✓ Word weights shape: {attn_info['word_weights'].shape}")
        
        return True
        
    except Exception as e:
        print(f"   ✗ Error in attention testing: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_memory_efficiency():
    """Test memory efficiency of different model sizes."""
    print("=" * 60)
    print("Testing Memory Efficiency")
    print("=" * 60)
    
    try:
        presets = ['text_small', 'text_medium', 'text_large']
        
        for preset in presets:
            print(f"Testing {preset}:")
            
            # Create model
            model = create_text_model(preset)
            
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            print(f"   ✓ Total parameters: {total_params:,}")
            print(f"   ✓ Trainable parameters: {trainable_params:,}")
            
            # Estimate memory usage
            param_memory = total_params * 4 / (1024 * 1024)  # Float32 in MB
            print(f"   ✓ Parameter memory: {param_memory:.1f} MB")
            
            # Test forward pass memory
            model.eval()
            with torch.no_grad():
                images = torch.randn(1, 3, 64, 256)
                outputs = model.forward(images)
                predictions = outputs['predictions']
                print(f"   ✓ Forward pass successful")
                print(f"   ✓ Output shape: {predictions.shape}")
            
            print()
        
        return True
        
    except Exception as e:
        print(f"   ✗ Error in memory testing: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("Enhanced Khmer OCR Model Architecture Tests")
    print("=" * 60)
    
    tests = [
        ("Model Creation", test_model_creation),
        ("Vocabulary Support", test_vocabulary_support),
        ("Forward Pass", test_forward_pass),
        ("Beam Search Decoding", test_beam_search),
        ("Model Presets", test_model_presets),
        ("Enhanced Attention", test_enhanced_attention),
        ("Memory Efficiency", test_memory_efficiency)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"Running: {test_name}")
        print(f"{'='*60}")
        
        try:
            success = test_func()
            results.append((test_name, success))
            
            if success:
                print(f"\n✓ {test_name} PASSED")
            else:
                print(f"\n✗ {test_name} FAILED")
                
        except Exception as e:
            print(f"\n✗ {test_name} FAILED with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"{status:4} | {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Enhanced model architecture is working correctly.")
        return True
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 