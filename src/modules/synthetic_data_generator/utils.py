"""
Utilities for synthetic data generation including font management,
text generation, and validation functions.
"""

import os
import yaml
import random
import unicodedata
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import json
import sys

# Add khtext module to path for syllable segmentation
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root / 'src' / 'modules'))

try:
    from khtext.subword_cluster import khmer_syllables_advanced, restore_whitespace_tags
    SYLLABLE_SEGMENTATION_AVAILABLE = True
except ImportError:
    print("Warning: Khmer syllable segmentation not available")
    SYLLABLE_SEGMENTATION_AVAILABLE = False

def normalize_khmer_text(text: str) -> str:
    """
    Normalize Khmer text using Unicode NFC normalization.
    
    Args:
        text: Input Khmer text
        
    Returns:
        Normalized text
    """
    return unicodedata.normalize('NFC', text)


def load_khmer_fonts(fonts_dir: str) -> Dict[str, str]:
    """
    Load all TTF fonts from the specified directory.
    
    Args:
        fonts_dir: Path to fonts directory
        
    Returns:
        Dictionary mapping font names to font file paths
    """
    fonts = {}
    fonts_path = Path(fonts_dir)
    
    if not fonts_path.exists():
        raise FileNotFoundError(f"Fonts directory not found: {fonts_dir}")
    
    for font_file in fonts_path.glob("*.ttf"):
        font_name = font_file.stem
        fonts[font_name] = str(font_file)
    
    if not fonts:
        raise ValueError(f"No TTF fonts found in {fonts_dir}")
    
    return fonts


def get_khmer_digits() -> List[str]:
    """
    Get the list of Khmer digits.
    
    Returns:
        List of Khmer digit characters
    """
    return ["០", "១", "២", "៣", "៤", "៥", "៦", "៧", "៨", "៩"]


def get_full_khmer_characters() -> Dict[str, List[str]]:
    """
    Get the complete Khmer character set organized by categories.
    
    Returns:
        Dictionary with character categories
    """
    try:
        # Import from khchar if available
        import sys
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'khtext'))
        from khchar import CONSONANTS, VOWELS, INDEPENDENTS, SIGN_CHARS, DIGITS, LEK_ATTAK
        
        return {
            'consonants': [chr(c) for c in CONSONANTS],
            'vowels': [chr(c) for c in VOWELS],
            'independents': [chr(c) for c in INDEPENDENTS],
            'signs': [chr(c) for c in SIGN_CHARS],
            'digits': [chr(c) for c in DIGITS],
            'lek_attak': [chr(c) for c in LEK_ATTAK]
        }
    except ImportError:
        # Fallback to basic character set
        return {
            'consonants': [chr(i) for i in range(0x1780, 0x17A3)],  # 33 consonants
            'vowels': [chr(i) for i in range(0x17B6, 0x17C6)],      # 16 vowels
            'independents': [chr(i) for i in range(0x17A5, 0x17B6)], # 14 independent vowels
            'signs': [chr(i) for i in range(0x17C6, 0x17D4)],       # 13 signs
            'digits': [chr(i) for i in range(0x17E0, 0x17EA)],      # 10 digits
            'lek_attak': [chr(i) for i in range(0x17F0, 0x17FA)]    # 10 lek attak
        }


def load_character_frequencies(analysis_file: str = "khmer_text_analysis_results.json") -> Dict[str, float]:
    """
    Load character frequencies from analysis results.
    
    Args:
        analysis_file: Path to analysis results JSON file
        
    Returns:
        Dictionary mapping characters to normalized frequencies
    """
    frequencies = {}
    
    try:
        if os.path.exists(analysis_file):
            with open(analysis_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if 'top_frequencies' in data:
                total_freq = sum(freq for char_code, freq in data['top_frequencies'])
                for char_code, freq in data['top_frequencies']:
                    char = chr(char_code)
                    frequencies[char] = freq / total_freq
    except Exception as e:
        print(f"Warning: Could not load character frequencies: {e}")
    
    # Fallback to uniform distribution if no frequencies available
    if not frequencies:
        khmer_chars = get_full_khmer_characters()
        all_chars = []
        for category in khmer_chars.values():
            all_chars.extend(category)
        
        uniform_freq = 1.0 / len(all_chars)
        frequencies = {char: uniform_freq for char in all_chars}
    
    return frequencies


def get_special_tokens() -> List[str]:
    """
    Get the list of special tokens used in the model.
    
    Returns:
        List of special token strings
    """
    return ["<EOS>", "<PAD>", "<BLANK>"]


def create_character_mapping(use_full_khmer: bool = True) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Create character to index and index to character mappings.
    
    Args:
        use_full_khmer: If True, use full Khmer character set, else just digits
    
    Returns:
        Tuple of (char_to_idx, idx_to_char) dictionaries
    """
    if use_full_khmer:
        khmer_chars = get_full_khmer_characters()
        chars = []
        for category in khmer_chars.values():
            chars.extend(category)
        chars.extend(get_special_tokens())
    else:
        chars = get_khmer_digits() + get_special_tokens()
    
    char_to_idx = {char: idx for idx, char in enumerate(chars)}
    idx_to_char = {idx: char for idx, char in enumerate(chars)}
    
    return char_to_idx, idx_to_char


def generate_digit_sequence(min_length: int = 1, max_length: int = 8) -> str:
    """
    Generate a random sequence of Khmer digits.
    
    Args:
        min_length: Minimum sequence length
        max_length: Maximum sequence length
        
    Returns:
        Random Khmer digit sequence
    """
    digits = get_khmer_digits()
    length = np.random.randint(min_length, max_length + 1)
    sequence = ''.join(np.random.choice(digits, size=length))
    return normalize_khmer_text(sequence)


def generate_weighted_character_sequence(length: int, 
                                       character_frequencies: Optional[Dict[str, float]] = None,
                                       character_set: Optional[List[str]] = None) -> str:
    """
    Generate a character sequence using frequency weighting.
    
    Args:
        length: Length of sequence to generate
        character_frequencies: Dictionary of character frequencies
        character_set: List of characters to choose from
        
    Returns:
        Generated character sequence
    """
    if character_frequencies is None:
        character_frequencies = load_character_frequencies()
    
    if character_set is None:
        character_set = list(character_frequencies.keys())
    
    # Filter frequencies for available character set
    available_chars = [char for char in character_set if char in character_frequencies]
    if not available_chars:
        # Fallback to uniform sampling
        available_chars = character_set
        weights = [1.0] * len(available_chars)
    else:
        weights = [character_frequencies[char] for char in available_chars]
    
    # Normalize weights
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]
    
    # Generate sequence
    sequence = ''.join(np.random.choice(available_chars, size=length, p=weights))
    return normalize_khmer_text(sequence)


def generate_khmer_syllable() -> str:
    """
    Generate a realistic Khmer syllable structure.
    
    Returns:
        Generated Khmer syllable
    """
    khmer_chars = get_full_khmer_characters()
    frequencies = load_character_frequencies()
    
    # Basic syllable structure: [Consonant] + [Vowel/Signs]
    consonants = khmer_chars['consonants']
    vowels = khmer_chars['vowels']
    signs = khmer_chars['signs']
    
    syllable = ""
    
    # Start with consonant (high probability)
    if consonants and random.random() < 0.9:
        consonant_weights = [frequencies.get(c, 0.01) for c in consonants]
        total_weight = sum(consonant_weights)
        if total_weight > 0:
            consonant_weights = [w / total_weight for w in consonant_weights]
            syllable += np.random.choice(consonants, p=consonant_weights)
    
    # Add vowel (moderate probability)
    if vowels and random.random() < 0.7:
        vowel_weights = [frequencies.get(v, 0.01) for v in vowels]
        total_weight = sum(vowel_weights)
        if total_weight > 0:
            vowel_weights = [w / total_weight for w in vowel_weights]
            syllable += np.random.choice(vowels, p=vowel_weights)
    
    # Add signs/diacritics (lower probability)
    if signs and random.random() < 0.3:
        sign_weights = [frequencies.get(s, 0.01) for s in signs]
        total_weight = sum(sign_weights)
        if total_weight > 0:
            sign_weights = [w / total_weight for w in sign_weights]
            syllable += np.random.choice(signs, p=sign_weights)
    
    return normalize_khmer_text(syllable) if syllable else generate_weighted_character_sequence(1)


def generate_khmer_word(min_syllables: int = 1, max_syllables: int = 4) -> str:
    """
    Generate a realistic Khmer word with multiple syllables.
    
    Args:
        min_syllables: Minimum number of syllables
        max_syllables: Maximum number of syllables
        
    Returns:
        Generated Khmer word
    """
    num_syllables = random.randint(min_syllables, max_syllables)
    word = ""
    
    for i in range(num_syllables):
        syllable = generate_khmer_syllable()
        word += syllable
        
        # Add COENG (stacking character) between syllables sometimes
        if i < num_syllables - 1 and random.random() < 0.3:
            word += chr(0x17D2)  # KHMER SIGN COENG
    
    return normalize_khmer_text(word)


def generate_khmer_phrase(min_words: int = 1, max_words: int = 5) -> str:
    """
    Generate a Khmer phrase with multiple words.
    
    Args:
        min_words: Minimum number of words
        max_words: Maximum number of words
        
    Returns:
        Generated Khmer phrase
    """
    num_words = random.randint(min_words, max_words)
    words = []
    
    for _ in range(num_words):
        word = generate_khmer_word()
        words.append(word)
    
    # Join words with space (though Khmer traditionally doesn't use spaces)
    phrase = " ".join(words) if random.random() < 0.3 else "".join(words)
    return normalize_khmer_text(phrase)


def generate_mixed_content(length: int = None, content_type: str = "mixed") -> str:
    """
    Generate mixed content including digits, text, and special characters.
    
    Args:
        length: Target length (if None, random length chosen)
        content_type: Type of content ('digits', 'characters', 'syllables', 'words', 'mixed')
        
    Returns:
        Generated content
    """
    if length is None:
        length = random.randint(1, 20)
    
    if content_type == "digits":
        return generate_digit_sequence(min(1, length), min(8, length))
    elif content_type == "characters":
        return generate_weighted_character_sequence(length)
    elif content_type == "syllables":
        num_syllables = max(1, length // 3)
        return "".join([generate_khmer_syllable() for _ in range(num_syllables)])
    elif content_type == "words":
        if length <= 5:
            return generate_khmer_word(1, 2)
        else:
            return generate_khmer_phrase(1, max(1, length // 10))
    else:  # mixed
        content_types = ["digits", "characters", "syllables", "words"]
        weights = [0.2, 0.3, 0.3, 0.2]  # Balanced distribution
        chosen_type = np.random.choice(content_types, p=weights)
        return generate_mixed_content(length, chosen_type)


def test_font_rendering(font_path: str, test_text: str = "០១២៣៤កខគ") -> bool:
    """
    Test if a font can properly render Khmer text.
    
    Args:
        font_path: Path to font file
        test_text: Test text to render
        
    Returns:
        True if font renders properly, False otherwise
    """
    try:
        font = ImageFont.truetype(font_path, size=48)
        # Try to get text bounding box - this will fail if characters are not supported
        bbox = font.getbbox(test_text)
        return bbox[2] > bbox[0] and bbox[3] > bbox[1]  # width > 0 and height > 0
    except Exception:
        return False


def validate_font_collection(fonts_dir: str) -> Dict[str, bool]:
    """
    Validate all fonts in the collection for Khmer text support.
    
    Args:
        fonts_dir: Path to fonts directory
        
    Returns:
        Dictionary mapping font names to validation status
    """
    fonts = load_khmer_fonts(fonts_dir)
    validation_results = {}
    
    # Test with a mix of digits and characters
    test_text = "០១២៣៤កខគឃង"  # Mix of digits and consonants
    
    for font_name, font_path in fonts.items():
        validation_results[font_name] = test_font_rendering(font_path, test_text)
    
    return validation_results


def validate_dataset(dataset_path: str, expected_size: int) -> Dict[str, any]:
    """
    Validate a generated dataset.
    
    Args:
        dataset_path: Path to dataset directory
        expected_size: Expected number of samples
        
    Returns:
        Dictionary with validation results
    """
    dataset_dir = Path(dataset_path)
    
    if not dataset_dir.exists():
        return {"valid": False, "error": "Dataset directory does not exist"}
    
    # Count image files
    image_files = list(dataset_dir.glob("*.png")) + list(dataset_dir.glob("*.jpg"))
    
    # Check for metadata file
    metadata_file = dataset_dir / "metadata.yaml"
    has_metadata = metadata_file.exists()
    
    # Load and validate metadata if it exists
    metadata = None
    if has_metadata:
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = yaml.safe_load(f)
        except Exception as e:
            return {"valid": False, "error": f"Failed to load metadata: {e}"}
    
    return {
        "valid": True,
        "num_images": len(image_files),
        "expected_size": expected_size,
        "size_match": len(image_files) == expected_size,
        "has_metadata": has_metadata,
        "metadata": metadata
    }


def calculate_dataset_statistics(dataset_path: str) -> Dict[str, any]:
    """
    Calculate statistics for a generated dataset.
    
    Args:
        dataset_path: Path to dataset directory
        
    Returns:
        Dictionary with dataset statistics
    """
    metadata_file = Path(dataset_path) / "metadata.yaml"
    
    if not metadata_file.exists():
        return {"error": "No metadata file found"}
    
    try:
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = yaml.safe_load(f)
    except Exception as e:
        return {"error": f"Failed to load metadata: {e}"}
    
    # Calculate statistics by combining train and val samples
    all_samples = []
    if 'train' in metadata and 'samples' in metadata['train']:
        all_samples.extend(metadata['train']['samples'])
    if 'val' in metadata and 'samples' in metadata['val']:
        all_samples.extend(metadata['val']['samples'])
    
    # If no train/val structure, try direct samples
    if not all_samples and 'samples' in metadata:
        all_samples = metadata['samples']
    
    sequence_lengths = [len(item['label']) for item in all_samples]
    fonts_used = [item['font'] for item in all_samples]
    
    stats = {
        "total_samples": len(all_samples),
        "sequence_length_distribution": {
            "min": min(sequence_lengths) if sequence_lengths else 0,
            "max": max(sequence_lengths) if sequence_lengths else 0,
            "mean": np.mean(sequence_lengths) if sequence_lengths else 0,
            "std": np.std(sequence_lengths) if sequence_lengths else 0
        },
        "font_distribution": {font: fonts_used.count(font) for font in set(fonts_used)},
        "character_frequency": {}
    }
    
    # Calculate character frequencies from all labels
    all_chars = ''.join([item['label'] for item in all_samples])
    khmer_chars = get_full_khmer_characters()
    for category_chars in khmer_chars.values():
        for char in category_chars:
            stats["character_frequency"][char] = all_chars.count(char)
    
    return stats


def load_khmer_corpus(corpus_file: str = "data/khmer_clean_text.txt") -> List[str]:
    """
    Load and prepare Khmer corpus text for segmentation.
    
    Args:
        corpus_file: Path to the corpus text file
        
    Returns:
        List of text lines from the corpus
    """
    if not os.path.exists(corpus_file):
        print(f"Warning: Corpus file not found: {corpus_file}")
        return []
    
    try:
        with open(corpus_file, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        print(f"Loaded corpus: {len(lines)} lines, total ~{sum(len(line) for line in lines):,} characters")
        return lines
        
    except Exception as e:
        print(f"Error loading corpus: {e}")
        return []


def segment_corpus_text(corpus_lines: List[str], 
                       target_length: int,
                       min_length: int = 1,
                       max_length: int = 50,
                       allowed_characters: Optional[List[str]] = None,
                       use_syllable_boundaries: bool = True) -> str:
    """
    Extract a random text segment from the corpus with specified length constraints.
    Uses advanced Khmer syllable segmentation for proper text boundaries.
    
    Args:
        corpus_lines: List of corpus text lines
        target_length: Desired segment length (approximate)
        min_length: Minimum acceptable length
        max_length: Maximum acceptable length
        allowed_characters: List of allowed characters for curriculum filtering
        use_syllable_boundaries: Whether to use syllable-aware boundary detection
        
    Returns:
        Text segment from corpus
    """
    if not corpus_lines:
        return generate_weighted_character_sequence(target_length)
    
    # Choose random line
    line = random.choice(corpus_lines)
    
    # Find suitable segment
    attempts = 0
    max_attempts = 50
    
    while attempts < max_attempts:
        # For very short target lengths, use simple character-based extraction
        if target_length <= 3 or not use_syllable_boundaries or not SYLLABLE_SEGMENTATION_AVAILABLE:
            segment = _extract_simple_segment(line, target_length, min_length, max_length)
        else:
            # Use syllable-aware extraction for longer segments
            segment = _extract_syllable_aware_segment(line, target_length, min_length, max_length)
        
        if not segment:
            attempts += 1
            continue
            
        # Clean and normalize
        segment = normalize_khmer_text(segment.strip())
        
        # Check length constraints
        if min_length <= len(segment) <= max_length:
            # Check character constraints for curriculum learning
            if allowed_characters is None or _text_uses_allowed_characters(segment, allowed_characters):
                return segment
        
        attempts += 1
    
    # Fallback: generate synthetic text if no suitable segment found
    print(f"Warning: Could not find suitable corpus segment, generating synthetic text")
    if allowed_characters:
        return generate_weighted_character_sequence(
            target_length, 
            character_frequencies={char: load_character_frequencies().get(char, 0.001) 
                                 for char in allowed_characters},
            character_set=allowed_characters
        )
    else:
        return generate_weighted_character_sequence(target_length)


def _extract_simple_segment(line: str, target_length: int, min_length: int, max_length: int) -> str:
    """Extract segment using simple character-based approach."""
    if len(line) <= target_length:
        return line
    
    # Random starting position
    start_pos = random.randint(0, len(line) - target_length)
    end_pos = start_pos + target_length
    
    # Adjust to word boundaries if possible
    segment = line[start_pos:end_pos]
    
    # Try to end at word boundary (space or punctuation)
    word_boundaries = [' ', '។', '៕', '៖', 'ៗ', '\n']
    for i in range(len(segment) - 1, max(0, len(segment) - 10), -1):
        if segment[i] in word_boundaries:
            segment = segment[:i].strip()
            break
    
    return segment


def _extract_syllable_aware_segment(line: str, target_length: int, min_length: int, max_length: int) -> str:
    """Extract segment using syllable-aware boundary detection."""
    try:
        # Segment the entire line into syllables
        syllables = khmer_syllables_advanced(line)
        
        if not syllables:
            return _extract_simple_segment(line, target_length, min_length, max_length)
        
        # Calculate approximate syllables needed
        avg_syllable_length = len(line) / len(syllables) if syllables else 3
        target_syllables = max(1, int(target_length / avg_syllable_length))
        
        # Choose random starting position in syllables
        if len(syllables) <= target_syllables:
            selected_syllables = syllables
        else:
            start_idx = random.randint(0, len(syllables) - target_syllables)
            
            # Try different segment lengths around target
            best_segment = None
            best_length_diff = float('inf')
            
            for length_adjustment in range(-2, 3):  # Try ±2 syllables
                adjusted_length = max(1, target_syllables + length_adjustment)
                end_idx = min(len(syllables), start_idx + adjusted_length)
                
                segment_syllables = syllables[start_idx:end_idx]
                segment_text = restore_whitespace_tags(''.join(segment_syllables))
                
                if min_length <= len(segment_text) <= max_length:
                    length_diff = abs(len(segment_text) - target_length)
                    if length_diff < best_length_diff:
                        best_length_diff = length_diff
                        best_segment = segment_text
            
            if best_segment:
                return best_segment
            
            # Fallback to target length
            selected_syllables = syllables[start_idx:start_idx + target_syllables]
        
        # Join syllables and restore whitespace
        segment = restore_whitespace_tags(''.join(selected_syllables))
        return segment
        
    except Exception as e:
        print(f"Warning: Syllable segmentation failed: {e}, falling back to simple extraction")
        return _extract_simple_segment(line, target_length, min_length, max_length)


def _text_uses_allowed_characters(text: str, allowed_chars: List[str]) -> bool:
    """Check if text only uses allowed characters."""
    allowed_set = set(allowed_chars)
    text_chars = set(text)
    khmer_chars = {char for char in text_chars if '\u1780' <= char <= '\u17FF'}
    return khmer_chars.issubset(allowed_set)


def extract_corpus_segments_by_complexity(corpus_lines: List[str],
                                         complexity_level: str = "medium",
                                         num_segments: int = 100,
                                         use_syllable_boundaries: bool = True) -> List[str]:
    """
    Extract corpus segments categorized by complexity level using syllable-aware boundaries.
    
    Args:
        corpus_lines: List of corpus text lines
        complexity_level: 'simple', 'medium', or 'complex'
        num_segments: Number of segments to extract
        use_syllable_boundaries: Whether to use syllable-aware boundary detection
        
    Returns:
        List of extracted text segments
    """
    complexity_configs = {
        'simple': {'min_length': 1, 'max_length': 5, 'target_length': 3},
        'medium': {'min_length': 6, 'max_length': 15, 'target_length': 10},
        'complex': {'min_length': 16, 'max_length': 50, 'target_length': 25}
    }
    
    if complexity_level not in complexity_configs:
        complexity_level = 'medium'
    
    config = complexity_configs[complexity_level]
    segments = []
    
    for _ in range(num_segments * 3):  # Try more to get enough good segments
        segment = segment_corpus_text(
            corpus_lines,
            target_length=random.randint(config['min_length'], config['max_length']),
            min_length=config['min_length'],
            max_length=config['max_length'],
            use_syllable_boundaries=use_syllable_boundaries
        )
        
        if segment and len(segment) >= config['min_length']:
            segments.append(segment)
        
        if len(segments) >= num_segments:
            break
    
    return segments[:num_segments]


def generate_corpus_based_text(corpus_lines: Optional[List[str]] = None,
                              target_length: int = None,
                              content_type: str = "auto",
                              allowed_characters: Optional[List[str]] = None) -> str:
    """
    Generate text using corpus segmentation with fallback to synthetic generation.
    
    Args:
        corpus_lines: Pre-loaded corpus lines (if None, will load automatically)
        target_length: Target text length
        content_type: Type of content to generate
        allowed_characters: Allowed characters for curriculum learning
        
    Returns:
        Generated or extracted text
    """
    if corpus_lines is None:
        corpus_lines = load_khmer_corpus()
    
    if target_length is None:
        target_length = random.randint(1, 20)
    
    # For certain content types, prefer corpus extraction
    if content_type in ["auto", "words", "phrases", "mixed"] and corpus_lines:
        # Try corpus extraction first
        corpus_text = segment_corpus_text(
            corpus_lines,
            target_length=target_length,
            min_length=max(1, target_length - 5),
            max_length=target_length + 10,
            allowed_characters=allowed_characters
        )
        
        if corpus_text and len(corpus_text) >= 1:
            return corpus_text
    
    # Fallback to synthetic generation
    if content_type == "digits":
        return generate_digit_sequence(1, min(8, target_length))
    elif content_type == "characters":
        return generate_weighted_character_sequence(
            target_length, 
            character_frequencies={char: load_character_frequencies().get(char, 0.001) 
                                 for char in allowed_characters} if allowed_characters else None,
            character_set=allowed_characters
        )
    elif content_type == "syllables":
        num_syllables = max(1, target_length // 3)
        return "".join([generate_khmer_syllable() for _ in range(num_syllables)])
    else:
        return generate_mixed_content(target_length, "mixed")


def analyze_corpus_segments(corpus_file: str = "data/khmer_clean_text.txt", 
                           num_samples: int = 100) -> Dict:
    """
    Analyze corpus text segments to understand their characteristics.
    
    Args:
        corpus_file: Path to corpus file
        num_samples: Number of segments to analyze
        
    Returns:
        Analysis results dictionary
    """
    corpus_lines = load_khmer_corpus(corpus_file)
    if not corpus_lines:
        return {"error": "Could not load corpus"}
    
    segments = {
        'simple': extract_corpus_segments_by_complexity(corpus_lines, 'simple', num_samples // 3),
        'medium': extract_corpus_segments_by_complexity(corpus_lines, 'medium', num_samples // 3),
        'complex': extract_corpus_segments_by_complexity(corpus_lines, 'complex', num_samples // 3)
    }
    
    analysis = {
        'corpus_stats': {
            'total_lines': len(corpus_lines),
            'avg_line_length': sum(len(line) for line in corpus_lines) / len(corpus_lines),
            'total_characters': sum(len(line) for line in corpus_lines)
        },
        'segment_analysis': {}
    }
    
    for complexity, segs in segments.items():
        if segs:
            lengths = [len(seg) for seg in segs]
            char_counts = {}
            for seg in segs:
                for char in seg:
                    if '\u1780' <= char <= '\u17FF':  # Khmer range
                        char_counts[char] = char_counts.get(char, 0) + 1
            
            analysis['segment_analysis'][complexity] = {
                'count': len(segs),
                'avg_length': sum(lengths) / len(lengths),
                'min_length': min(lengths),
                'max_length': max(lengths),
                'unique_characters': len(char_counts),
                'sample_segments': segs[:5],
                'top_characters': sorted(char_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            }
    
    return analysis 