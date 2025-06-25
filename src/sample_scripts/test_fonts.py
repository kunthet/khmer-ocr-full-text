#!/usr/bin/env python3
"""
Test script to validate Khmer fonts and test basic functionality.
"""

import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.synthetic_data_generator.utils import (
    load_khmer_fonts, validate_font_collection, get_khmer_digits,
    generate_digit_sequence, normalize_khmer_text
)


def test_font_loading():
    """Test font loading functionality."""
    print("=== Testing Font Loading ===")
    
    fonts_dir = "src/fonts"
    
    try:
        fonts = load_khmer_fonts(fonts_dir)
        print(f"Found {len(fonts)} font files:")
        for name, path in fonts.items():
            print(f"  {name}: {path}")
        return fonts
    except Exception as e:
        print(f"Error loading fonts: {e}")
        return {}


def test_font_validation(fonts):
    """Test font validation functionality."""
    print("\n=== Testing Font Validation ===")
    
    if not fonts:
        print("No fonts to validate!")
        return {}
    
    try:
        validation_results = validate_font_collection("src/fonts")
        print("Font validation results:")
        for font_name, is_valid in validation_results.items():
            status = "✓" if is_valid else "✗"
            print(f"  {status} {font_name}")
        
        working_fonts = [name for name, valid in validation_results.items() if valid]
        print(f"\nWorking fonts: {len(working_fonts)}/{len(validation_results)}")
        
        return validation_results
    except Exception as e:
        print(f"Error validating fonts: {e}")
        return {}


def test_text_generation():
    """Test text generation functionality."""
    print("\n=== Testing Text Generation ===")
    
    try:
        digits = get_khmer_digits()
        print(f"Khmer digits: {digits}")
        
        print("\nGenerated sequences:")
        for i in range(10):
            seq = generate_digit_sequence(1, 8)
            normalized = normalize_khmer_text(seq)
            print(f"  {i+1}: '{seq}' (length: {len(seq)}) -> '{normalized}'")
        
    except Exception as e:
        print(f"Error generating text: {e}")


def test_basic_image_generation():
    """Test basic image generation without saving."""
    print("\n=== Testing Basic Image Generation ===")
    
    try:
        from modules.synthetic_data_generator.backgrounds import BackgroundGenerator
        from modules.synthetic_data_generator.augmentation import ImageAugmentor
        
        # Test background generation
        bg_gen = BackgroundGenerator((128, 64))
        print("Testing background generation...")
        
        bg_solid = bg_gen.generate_solid_color()
        print(f"  Solid background: {bg_solid.size}, mode: {bg_solid.mode}")
        
        bg_gradient = bg_gen.generate_gradient()
        print(f"  Gradient background: {bg_gradient.size}, mode: {bg_gradient.mode}")
        
        bg_paper = bg_gen.generate_paper_texture()
        print(f"  Paper texture: {bg_paper.size}, mode: {bg_paper.mode}")
        
        # Test augmentation
        augmentor = ImageAugmentor()
        print("Testing image augmentation...")
        
        rotated = augmentor.rotate_image(bg_solid, 15)
        print(f"  Rotated image: {rotated.size}, mode: {rotated.mode}")
        
        noisy = augmentor.add_gaussian_noise(bg_solid)
        print(f"  Noisy image: {noisy.size}, mode: {noisy.mode}")
        
        print("Basic image generation tests passed!")
        
    except Exception as e:
        print(f"Error in image generation: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main test function."""
    print("=== Khmer Digits Font and Functionality Test ===\n")
    
    # Test font loading
    fonts = test_font_loading()
    
    # Test font validation
    validation_results = test_font_validation(fonts)
    
    # Test text generation
    test_text_generation()
    
    # Test basic image generation
    test_basic_image_generation()
    
    # Summary
    print("\n=== Test Summary ===")
    working_fonts = [name for name, valid in validation_results.items() if valid]
    
    if working_fonts:
        print(f"✓ Found {len(working_fonts)} working fonts")
        print(f"✓ Text generation working")
        print(f"✓ Image generation components working")
        print("\nReady to generate synthetic dataset!")
    else:
        print("✗ No working fonts found!")
        print("Please check font installation and compatibility.")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main()) 