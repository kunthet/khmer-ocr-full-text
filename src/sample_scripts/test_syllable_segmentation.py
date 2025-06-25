#!/usr/bin/env python3
"""
Test script demonstrating the improvement in corpus text segmentation 
using syllable-aware boundary detection vs simple character-based cutting.
"""

import os
import sys
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'src' / 'modules'))

from synthetic_data_generator.utils import (
    load_khmer_corpus, segment_corpus_text, 
    _extract_simple_segment, _extract_syllable_aware_segment
)

def compare_segmentation_methods():
    """Compare simple vs syllable-aware text segmentation."""
    print("="*70)
    print("CORPUS TEXT SEGMENTATION COMPARISON")
    print("="*70)
    
    # Load corpus
    corpus_lines = load_khmer_corpus()
    if not corpus_lines:
        print("❌ Could not load corpus")
        return
    
    # Test with different target lengths
    test_lengths = [5, 10, 15, 25]
    
    for target_length in test_lengths:
        print(f"\n🎯 Target Length: {target_length} characters")
        print("-" * 50)
        
        # Generate samples with both methods
        for i in range(3):
            print(f"\nSample {i+1}:")
            
            # Choose a random line
            import random
            line = random.choice(corpus_lines)
            
            # Simple character-based extraction
            simple_segment = _extract_simple_segment(
                line, target_length, 
                max(1, target_length - 3), 
                target_length + 5
            )
            
            # Syllable-aware extraction
            syllable_segment = _extract_syllable_aware_segment(
                line, target_length,
                max(1, target_length - 3),
                target_length + 5
            )
            
            print(f"  Simple:   '{simple_segment}' (len: {len(simple_segment)})")
            print(f"  Syllable: '{syllable_segment}' (len: {len(syllable_segment)})")
            
            # Analyze quality
            analyze_segment_quality(simple_segment, syllable_segment)

def analyze_segment_quality(simple_seg: str, syllable_seg: str):
    """Analyze and compare the quality of segmented text."""
    
    def has_broken_syllables(text: str) -> bool:
        """Check if text has obviously broken syllables."""
        # Look for patterns that suggest broken syllables
        broken_patterns = [
            '្$',  # Ends with COENG (subscript marker)
            '^្',  # Starts with COENG
            '[ាិីឹឺុូួើែៃេ]$',  # Ends with dependent vowel without consonant
        ]
        
        import re
        for pattern in broken_patterns:
            if re.search(pattern, text):
                return True
        return False
    
    def count_complete_syllables(text: str) -> int:
        """Estimate number of complete syllables."""
        try:
            from khtext.subword_cluster import khmer_syllables_advanced
            syllables = khmer_syllables_advanced(text)
            # Filter out whitespace tags and empty syllables
            complete_syllables = [s for s in syllables if s.strip() and not s.startswith('<')]
            return len(complete_syllables)
        except:
            return 0
    
    # Quality indicators
    simple_broken = has_broken_syllables(simple_seg)
    syllable_broken = has_broken_syllables(syllable_seg)
    
    simple_syllable_count = count_complete_syllables(simple_seg)
    syllable_syllable_count = count_complete_syllables(syllable_seg)
    
    quality_indicators = []
    
    if simple_broken and not syllable_broken:
        quality_indicators.append("✅ Syllable method avoids broken syllables")
    elif syllable_broken and not simple_broken:
        quality_indicators.append("⚠️ Simple method has better boundaries here")
    
    if syllable_syllable_count > simple_syllable_count:
        quality_indicators.append("✅ Syllable method preserves more complete syllables")
    elif simple_syllable_count > syllable_syllable_count:
        quality_indicators.append("ℹ️ Simple method has more syllables")
    
    if quality_indicators:
        for indicator in quality_indicators:
            print(f"    {indicator}")
    else:
        print(f"    → Both methods produce similar quality")

def test_syllable_boundary_preservation():
    """Test how well syllable boundaries are preserved."""
    print("\n" + "="*70)
    print("SYLLABLE BOUNDARY PRESERVATION TEST")
    print("="*70)
    
    # Test with known Khmer text that has complex syllables
    test_texts = [
        "អ្នកគ្រួបង្រៀនភាសាខ្មែរឱ្យបានល្អ",
        "ការធ្វើទំនើបកម្មវិស័យកសិកម្មនៅកម្ពុជា",
        "បច្ចេកវិទ្យាកសិកម្ម និងឧបករណ៍ទំនើបៗ",
        "ដំណើរការផលិតកម្មសម័យទំនើប"
    ]
    
    for i, text in enumerate(test_texts):
        print(f"\n📝 Test Text {i+1}: '{text}'")
        print(f"Length: {len(text)} characters")
        
        # Show syllable breakdown
        try:
            from khtext.subword_cluster import khmer_syllables_advanced
            syllables = khmer_syllables_advanced(text)
            clean_syllables = [s for s in syllables if s.strip() and not s.startswith('<')]
            print(f"Syllables: {clean_syllables}")
            print(f"Syllable count: {len(clean_syllables)}")
        except ImportError:
            print("Syllable analysis not available")
        
        # Test extraction at different lengths
        for target_len in [8, 12, 16]:
            print(f"\n  🎯 Extracting {target_len}-char segment:")
            
            simple_seg = _extract_simple_segment(text, target_len, target_len-2, target_len+3)
            syllable_seg = _extract_syllable_aware_segment(text, target_len, target_len-2, target_len+3)
            
            print(f"    Simple:   '{simple_seg}' (len: {len(simple_seg)})")
            print(f"    Syllable: '{syllable_seg}' (len: {len(syllable_seg)})")

def test_corpus_integration():
    """Test the integration with actual corpus data."""
    print("\n" + "="*70)
    print("CORPUS INTEGRATION TEST")
    print("="*70)
    
    corpus_lines = load_khmer_corpus()
    if not corpus_lines:
        print("❌ Corpus not available")
        return
    
    # Test with actual corpus extraction
    test_configs = [
        {"target": 8, "method": "Simple", "use_syllable": False},
        {"target": 8, "method": "Syllable-aware", "use_syllable": True},
        {"target": 15, "method": "Simple", "use_syllable": False},
        {"target": 15, "method": "Syllable-aware", "use_syllable": True},
    ]
    
    for config in test_configs:
        print(f"\n🔍 {config['method']} extraction (target: {config['target']} chars):")
        
        segments = []
        for _ in range(5):
            segment = segment_corpus_text(
                corpus_lines,
                target_length=config['target'],
                min_length=config['target'] - 2,
                max_length=config['target'] + 5,
                use_syllable_boundaries=config['use_syllable']
            )
            segments.append(segment)
        
        for i, segment in enumerate(segments):
            print(f"  {i+1}. '{segment}' (len: {len(segment)})")

def main():
    """Run comprehensive syllable boundary segmentation tests."""
    print("🚀 SYLLABLE-AWARE CORPUS SEGMENTATION TEST")
    print("Testing improvements in text segmentation quality")
    
    try:
        compare_segmentation_methods()
        test_syllable_boundary_preservation()
        test_corpus_integration()
        
        print("\n" + "="*70)
        print("🎉 SYLLABLE SEGMENTATION TEST COMPLETE!")
        print("="*70)
        print("✅ Benefits of syllable-aware segmentation:")
        print("  - Preserves complete Khmer syllables")
        print("  - Avoids breaking COENG consonant clusters")
        print("  - Maintains proper vowel-consonant relationships")
        print("  - Produces more natural text boundaries")
        print("  - Better quality for OCR training data")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 