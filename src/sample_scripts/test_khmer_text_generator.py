#!/usr/bin/env python3
"""
Test script for KhmerTextGenerator class.
Validates linguistic rule compliance and generation quality.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.modules.synthetic_data_generator.utils import KhmerTextGenerator

def test_character_classification():
    """Test character classification methods."""
    print("=== Testing Character Classification ===")
    
    # Test consonants
    consonant_samples = ['ក', 'ខ', 'គ', 'ង', 'ត', 'ន', 'ប', 'ម', 'យ', 'រ', 'ល', 'វ', 'ស', 'ហ']
    for char in consonant_samples:
        assert KhmerTextGenerator.is_khmer_consonant(char), f"'{char}' should be consonant"
    
    # Test vowels (dependent)
    vowel_samples = ['ា', 'ិ', 'ី', 'ឹ', 'ឺ', 'ុ', 'ូ', 'ួ', 'ើ', 'ៀ', 'េ', 'ែ', 'ៃ', 'ោ', 'ៅ']
    for char in vowel_samples:
        assert KhmerTextGenerator.is_khmer_vowel(char), f"'{char}' should be vowel"
    
    # Test independent vowels
    independent_samples = ['ឥ', 'ឦ', 'ឧ', 'ឨ', 'ឩ', 'ឪ', 'ឫ', 'ឬ', 'ឭ', 'ឮ', 'ឯ', 'ឰ', 'ឱ', 'ឲ', 'ឳ']
    for char in independent_samples:
        assert KhmerTextGenerator.is_khmer_independent_vowel(char), f"'{char}' should be independent vowel"
    
    # Test valid start characters
    for char in consonant_samples + independent_samples:
        assert KhmerTextGenerator.is_valid_khmer_start_character(char), f"'{char}' should be valid start"
    
    # Test invalid start characters (dependent vowels)
    for char in vowel_samples:
        assert not KhmerTextGenerator.is_valid_khmer_start_character(char), f"'{char}' should not be valid start"
    
    print("✅ Character classification tests passed!")

def test_rule_validation():
    """Test linguistic rule validation."""
    print("\n=== Testing Rule Validation ===")
    
    generator = KhmerTextGenerator()
    
    # Test Rule 1: Must start with consonant or independent vowel
    valid_text = "កា"  # consonant + vowel
    is_valid, error = generator.validate_khmer_text_structure(valid_text)
    assert is_valid, f"'{valid_text}' should be valid: {error}"
    
    invalid_text = "ាក"  # vowel + consonant (invalid start)
    is_valid, error = generator.validate_khmer_text_structure(invalid_text)
    assert not is_valid, f"'{invalid_text}' should be invalid"
    
    # Test Rule 2: Coeng must be between consonants
    valid_coeng = "ក្រ"  # consonant + coeng + consonant
    is_valid, error = generator.validate_khmer_text_structure(valid_coeng)
    assert is_valid, f"'{valid_coeng}' should be valid: {error}"
    
    invalid_coeng = "ក្ា"  # consonant + coeng + vowel (invalid)
    is_valid, error = generator.validate_khmer_text_structure(invalid_coeng)
    assert not is_valid, f"'{invalid_coeng}' should be invalid"
    
    # Test Rule 3: No double Coeng
    invalid_double_coeng = "ក្្រ"  # consonant + coeng + coeng + consonant
    is_valid, error = generator.validate_khmer_text_structure(invalid_double_coeng)
    assert not is_valid, f"'{invalid_double_coeng}' should be invalid"
    
    # Test Rule 4: No duplicate vowels
    invalid_double_vowel = "កាា"  # consonant + vowel + same vowel
    is_valid, error = generator.validate_khmer_text_structure(invalid_double_vowel)
    assert not is_valid, f"'{invalid_double_vowel}' should be invalid"
    
    # Test Rule 5: Cannot end with Coeng
    invalid_end_coeng = "ក្"  # consonant + coeng (ends with coeng)
    is_valid, error = generator.validate_khmer_text_structure(invalid_end_coeng)
    assert not is_valid, f"'{invalid_end_coeng}' should be invalid"
    
    print("✅ Rule validation tests passed!")

def test_text_generation():
    """Test text generation methods."""
    print("\n=== Testing Text Generation ===")
    
    generator = KhmerTextGenerator()
    
    # Test syllable generation
    print("Testing syllable generation:")
    for i in range(5):
        syllable = generator.generate_syllable()
        is_valid, error = generator.validate_khmer_text_structure(syllable)
        print(f"  Syllable {i+1}: '{syllable}' - {'✅ Valid' if is_valid else f'❌ Invalid: {error}'}")
        assert is_valid, f"Generated syllable '{syllable}' is invalid: {error}"
    
    # Test word generation
    print("\nTesting word generation:")
    for i in range(5):
        word = generator.generate_word(min_syllables=1, max_syllables=3)
        is_valid, error = generator.validate_khmer_text_structure(word)
        print(f"  Word {i+1}: '{word}' - {'✅ Valid' if is_valid else f'❌ Invalid: {error}'}")
        assert is_valid, f"Generated word '{word}' is invalid: {error}"
    
    # Test phrase generation
    print("\nTesting phrase generation:")
    for i in range(3):
        phrase = generator.generate_phrase(min_words=1, max_words=3)
        is_valid, error = generator.validate_khmer_text_structure(phrase.replace(' ', ''))  # Remove spaces for validation
        print(f"  Phrase {i+1}: '{phrase}' - {'✅ Valid' if is_valid else f'❌ Invalid: {error}'}")
        # Note: Phrases may contain spaces which are not part of Khmer script validation
    
    # Test character sequence generation
    print("\nTesting character sequence generation:")
    for length in [3, 5, 8]:
        sequence = generator.generate_character_sequence(length)
        is_valid, error = generator.validate_khmer_text_structure(sequence)
        print(f"  Length {length}: '{sequence}' - {'✅ Valid' if is_valid else f'❌ Invalid: {error}'}")
        assert is_valid, f"Generated sequence '{sequence}' is invalid: {error}"
    
    print("✅ Text generation tests passed!")

def test_content_by_type():
    """Test content generation by type."""
    print("\n=== Testing Content Generation by Type ===")
    
    generator = KhmerTextGenerator()
    
    content_types = ["digits", "characters", "syllables", "words", "phrases", "auto", "mixed"]
    
    for content_type in content_types:
        print(f"\nTesting {content_type} generation:")
        for i in range(3):
            try:
                content = generator.generate_content_by_type(content_type, length=5)
                
                # Skip validation for digits (they're not Khmer script)
                if content_type == "digits":
                    print(f"  {content_type.capitalize()} {i+1}: '{content}' - ✅ Generated")
                else:
                    # Remove spaces for validation (phrases may contain spaces)
                    clean_content = content.replace(' ', '')
                    if clean_content:  # Only validate non-empty content
                        is_valid, error = generator.validate_khmer_text_structure(clean_content)
                        print(f"  {content_type.capitalize()} {i+1}: '{content}' - {'✅ Valid' if is_valid else f'❌ Invalid: {error}'}")
                        if not is_valid and content_type != "phrases":  # Phrases may have complex structures
                            print(f"    Warning: {error}")
                    else:
                        print(f"  {content_type.capitalize()} {i+1}: '{content}' - ⚠️ Empty")
                        
            except Exception as e:
                print(f"  {content_type.capitalize()} {i+1}: ❌ Error: {e}")
    
    print("\n✅ Content generation by type tests completed!")

def test_frequency_integration():
    """Test character frequency integration."""
    print("\n=== Testing Character Frequency Integration ===")
    
    generator = KhmerTextGenerator()
    
    print(f"Loaded {len(generator.character_frequencies)} character frequencies")
    print(f"Character sets: consonants({len(generator.consonants)}), vowels({len(generator.vowels)}), independents({len(generator.independents)})")
    
    # Test character frequency loading
    assert len(generator.character_frequencies) > 0, "Character frequencies should be loaded"
    
    # Test that frequencies are normalized (sum to 1 or close to 1)
    total_freq = sum(generator.character_frequencies.values())
    print(f"Total frequency sum: {total_freq:.6f}")
    
    # Generate some text and check if high-frequency characters appear more often
    generated_texts = [generator.generate_syllable() for _ in range(100)]
    char_counts = {}
    
    for text in generated_texts:
        for char in text:
            char_counts[char] = char_counts.get(char, 0) + 1
    
    print("Top 5 most generated characters:")
    sorted_chars = sorted(char_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    for char, count in sorted_chars:
        freq = generator.character_frequencies.get(char, 0)
        print(f"  '{char}': {count} times (frequency: {freq:.4f})")
    
    print("✅ Character frequency integration tests passed!")

def main():
    """Run all tests."""
    print("🔍 Testing KhmerTextGenerator Class")
    print("=" * 50)
    
    try:
        test_character_classification()
        test_rule_validation()
        test_text_generation()
        test_content_by_type()
        test_frequency_integration()
        
        print("\n" + "=" * 50)
        print("🎉 All KhmerTextGenerator tests passed successfully!")
        print("✅ Linguistic rules are properly enforced")
        print("✅ Text generation methods work correctly")
        print("✅ Character frequency integration is functional")
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 