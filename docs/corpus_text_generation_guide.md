# Corpus-Based Text Generation Guide

## Overview

The corpus-based text generation system provides authentic Khmer text for OCR training by extracting segments from real Khmer corpus data. This approach significantly improves training quality compared to purely synthetic text generation by providing natural language patterns, proper grammar, and authentic character combinations.

## Key Features

### 🎯 Core Capabilities
- **Authentic Language Patterns**: Extract real Khmer text from 40.9MB corpus (644 lines, 13.9M characters)
- **Syllable-Aware Segmentation**: Uses `khmer_syllables_advanced` for proper boundary detection
- **Curriculum Learning Integration**: Compatible with character subset filtering for progressive training
- **Complexity-Based Extraction**: Generate text at different complexity levels (simple/medium/complex)
- **Fallback Mechanisms**: Graceful degradation to synthetic generation when needed

### 🔧 Technical Features
- **Performance**: 50+ samples/second generation speed
- **Quality Assurance**: Automatic validation and filtering of extracted segments
- **Backward Compatibility**: Works alongside existing synthetic generation methods
- **Configurable Parameters**: Flexible length ranges, character constraints, and boundary detection options

## Architecture

### System Components

```
Corpus Text Generation Pipeline
├── corpus_loader.py          # Load and preprocess corpus files
├── syllable_segmenter.py     # Advanced Khmer syllable boundary detection
├── text_extractor.py         # Extract segments with quality validation
├── curriculum_filter.py      # Apply character constraints for learning stages
└── quality_analyzer.py       # Analyze and validate text quality
```

### Integration Points

```
SyntheticDataGenerator
├── use_corpus=True           # Enable corpus-based generation
├── corpus_lines              # Pre-loaded corpus data
├── syllable_boundaries       # Advanced boundary detection
└── fallback_synthetic        # Backup generation method
```

## Quick Start

### Basic Usage

```python
from synthetic_data_generator.generator import SyntheticDataGenerator

# Initialize with corpus enabled
generator = SyntheticDataGenerator(
    config_path="config/model_config.yaml",
    fonts_dir="src/fonts",
    output_dir="training_data",
    mode='full_text',
    use_corpus=True  # Enable corpus-based generation
)

# Generate authentic Khmer text
text = generator._generate_text_content(
    content_type="phrases",      # words, phrases, mixed
    length_range=(10, 20)        # Target length range
)

print(f"Generated: '{text}'")
# Output: Generated: 'ការធ្វើទំនើបកម្មវិស័យ'
```

### Advanced Configuration

```python
# Custom corpus file and syllable boundaries
from synthetic_data_generator.utils import segment_corpus_text, load_khmer_corpus

# Load corpus
corpus_lines = load_khmer_corpus("path/to/corpus.txt")

# Extract with syllable boundaries
segment = segment_corpus_text(
    corpus_lines=corpus_lines,
    target_length=15,
    min_length=10,
    max_length=20,
    use_syllable_boundaries=True,  # Enable syllable-aware cutting
    allowed_characters=['ក', 'ា', 'រ']  # Character constraints
)
```

## API Reference

### Core Functions

#### `load_khmer_corpus(corpus_file)`
Load and prepare Khmer corpus text for segmentation.

**Parameters:**
- `corpus_file` (str): Path to corpus text file (default: "data/khmer_clean_text.txt")

**Returns:**
- `List[str]`: List of text lines from corpus

**Example:**
```python
corpus_lines = load_khmer_corpus()
print(f"Loaded {len(corpus_lines)} lines")
# Output: Loaded 644 lines
```

#### `segment_corpus_text(corpus_lines, target_length, **kwargs)`
Extract text segment with advanced boundary detection.

**Parameters:**
- `corpus_lines` (List[str]): Pre-loaded corpus lines
- `target_length` (int): Desired segment length (approximate)
- `min_length` (int): Minimum acceptable length (default: 1)
- `max_length` (int): Maximum acceptable length (default: 50)
- `allowed_characters` (Optional[List[str]]): Character constraints for curriculum
- `use_syllable_boundaries` (bool): Enable syllable-aware cutting (default: True)

**Returns:**
- `str`: Extracted text segment

**Example:**
```python
segment = segment_corpus_text(
    corpus_lines=corpus_lines,
    target_length=12,
    min_length=10,
    max_length=15,
    use_syllable_boundaries=True
)
print(f"Segment: '{segment}' (length: {len(segment)})")
# Output: Segment: 'ការធ្វើទំនើប' (length: 12)
```

#### `extract_corpus_segments_by_complexity(corpus_lines, complexity_level, num_segments)`
Extract multiple segments categorized by complexity.

**Parameters:**
- `corpus_lines` (List[str]): Pre-loaded corpus lines
- `complexity_level` (str): 'simple', 'medium', or 'complex'
- `num_segments` (int): Number of segments to extract
- `use_syllable_boundaries` (bool): Enable syllable-aware cutting (default: True)

**Complexity Configurations:**
- **Simple**: 1-5 characters, basic text elements
- **Medium**: 6-15 characters, word-level content  
- **Complex**: 16-50 characters, phrase and sentence fragments

**Returns:**
- `List[str]`: List of extracted segments

**Example:**
```python
segments = extract_corpus_segments_by_complexity(
    corpus_lines=corpus_lines,
    complexity_level='medium',
    num_segments=10
)
print(f"Generated {len(segments)} medium complexity segments")
# Output: Generated 10 medium complexity segments
```

#### `generate_corpus_based_text(corpus_lines, target_length, content_type, allowed_characters)`
Unified corpus and synthetic text generation with intelligent fallback.

**Parameters:**
- `corpus_lines` (Optional[List[str]]): Pre-loaded corpus (auto-loads if None)
- `target_length` (Optional[int]): Target text length (random if None)
- `content_type` (str): 'auto', 'words', 'phrases', 'mixed', 'characters', 'syllables', 'digits'
- `allowed_characters` (Optional[List[str]]): Character constraints

**Content Type Behavior:**
- **auto/words/phrases/mixed**: Prefer corpus extraction
- **characters/syllables**: Use synthetic generation
- **digits**: Always use synthetic digit sequences

**Returns:**
- `str`: Generated text (corpus or synthetic)

**Example:**
```python
text = generate_corpus_based_text(
    corpus_lines=corpus_lines,
    target_length=15,
    content_type='phrases',
    allowed_characters=None
)
print(f"Generated: '{text}'")
# Output: Generated: 'ដំណើរការផលិតកម្ម'
```

### Syllable Boundary Detection

#### `_extract_syllable_aware_segment(line, target_length, min_length, max_length)`
Extract segment using advanced syllable boundary detection.

**Process:**
1. Segment entire line using `khmer_syllables_advanced()`
2. Calculate optimal syllable count for target length
3. Try different segment lengths (±2 syllables) to find best fit
4. Join syllables and restore whitespace using `restore_whitespace_tags()`
5. Fallback to simple extraction if syllable segmentation fails

**Quality Benefits:**
- Preserves complete Khmer syllables
- Avoids breaking COENG consonant clusters
- Maintains proper vowel-consonant relationships
- Produces natural text boundaries

**Example:**
```python
# Input text: "អ្នកគ្រួបង្រៀនភាសាខ្មែរឱ្យបានល្អ"
# Syllables: ['អ្ន', 'ក', 'គ្រួ', 'ប', 'ង្រៀ', 'ន', 'ភា', 'សា', 'ខ្មែ', 'រ', 'ឱ្យ', 'បា', 'ន', 'ល្អ']

segment = _extract_syllable_aware_segment(line, 12, 10, 15)
# Result: 'បង្រៀនភាសាខ្មែរ' (proper syllable boundaries)
```

#### `_extract_simple_segment(line, target_length, min_length, max_length)`
Fallback character-based extraction with word boundary awareness.

**Process:**
1. Random character-based positioning
2. Attempt to end at word boundaries (spaces, punctuation)
3. Simple character counting approach

**Usage:**
- Target lengths ≤ 3 characters
- When syllable segmentation unavailable
- As fallback when syllable method fails

### Quality Analysis

#### `analyze_corpus_segments(corpus_file, num_samples)`
Comprehensive analysis of corpus text segment characteristics.

**Returns:**
```python
{
    'corpus_stats': {
        'total_lines': 644,
        'avg_line_length': 21607.8,
        'total_characters': 13915420
    },
    'segment_analysis': {
        'simple': {
            'count': 10,
            'avg_length': 2.9,
            'min_length': 1,
            'max_length': 5,
            'unique_characters': 20,
            'sample_segments': ['៏មា', 'នៃ', ...],
            'top_characters': [('្', 145), ('ា', 98), ...]
        },
        'medium': {...},
        'complex': {...}
    }
}
```

## Configuration

### Generator Configuration

```python
# Full configuration example
generator = SyntheticDataGenerator(
    config_path="config/model_config.yaml",  # Model configuration
    fonts_dir="src/fonts",                   # Khmer fonts directory
    output_dir="training_data",              # Output directory
    mode='full_text',                        # 'digits', 'full_text', 'mixed'
    use_corpus=True                          # Enable corpus-based generation
)
```

### Corpus File Requirements

**Format:**
- UTF-8 encoded text file
- One paragraph/document per line
- Authentic Khmer text content

**Recommended Characteristics:**
- Large corpus size (40MB+ recommended)
- Diverse content (multiple domains, writing styles)
- Clean text (minimal OCR errors or formatting issues)
- Representative character distribution

**Example Line:**
```
ការធ្វើទំនើបកម្មវិស័យកសិកម្មនៅកម្ពុជាទាមទារនូវការផ្លាស់ប្ដូរផ្នត់គំនិត និងការវិនិយោគយ៉ាងសន្ធឹកសន្ធាប់។
```

## Integration with Curriculum Learning

### Character Subset Filtering

The corpus generation system seamlessly integrates with curriculum learning by respecting character constraints:

```python
# Stage 1: High-frequency characters only
high_freq_chars = ['ក', 'ា', 'រ', 'ន', 'ង', '្', 'ត', 'ម', 'ល', 'ស']

segment = segment_corpus_text(
    corpus_lines=corpus_lines,
    target_length=10,
    allowed_characters=high_freq_chars  # Only use these characters
)
```

### Curriculum Stages

**Stage 1 (Simple):**
- Top 30 high-frequency characters
- Length range: 1-5 characters
- Content: Characters and simple syllables

**Stage 2 (Medium):**
- Top 60 medium-frequency characters  
- Length range: 6-15 characters
- Content: Syllables and words

**Stage 3 (Complex):**
- All 112+ characters
- Length range: 16-50 characters
- Content: Words, phrases, and complex structures

## Performance Optimization

### Caching Strategies

```python
# Pre-load corpus for multiple generations
corpus_lines = load_khmer_corpus()

# Reuse loaded corpus
for i in range(1000):
    segment = segment_corpus_text(corpus_lines, target_length=15)
```

### Batch Processing

```python
# Generate multiple segments efficiently
segments = extract_corpus_segments_by_complexity(
    corpus_lines=corpus_lines,
    complexity_level='medium',
    num_segments=100  # Batch generation
)
```

### Memory Management

- Corpus loads ~14MB into memory
- Syllable segmentation adds ~10% overhead
- Recommended for systems with 4GB+ RAM

## Quality Comparison

### Corpus vs Synthetic Examples

**Corpus-Based (Natural):**
```python
# Examples from real text extraction
"ការធ្វើទំនើបកម្ម"      # Modernization process
"ងមុន។"                # Previous period  
"ារអនុវត"               # Implementation
"តបានជាបរិយាកាស"        # Environmental context
```

**Synthetic (Artificial):**
```python
# Examples from algorithmic generation
"រ្"                    # Isolated subscript
"កៀៜ"                   # Random combination
"ជិ្ដាយៅ្"              # Unrealistic pattern
```

### Quality Metrics

| **Metric** | **Corpus** | **Synthetic** |
|------------|------------|---------------|
| **Language Authenticity** | ✅ Natural | ❌ Artificial |
| **Grammar Correctness** | ✅ Proper | ⚠️ Variable |
| **Character Combinations** | ✅ Realistic | ❌ Random |
| **COENG Usage** | ✅ Authentic | ⚠️ Basic |
| **Training Effectiveness** | ✅ Superior | ✅ Good |

## Troubleshooting

### Common Issues

**1. Corpus Not Loading**
```python
# Check file path and encoding
import os
print(os.path.exists("data/khmer_clean_text.txt"))  # Should be True
print(os.path.getsize("data/khmer_clean_text.txt"))  # Should be > 1MB
```

**Solution:**
- Verify file path is correct
- Ensure UTF-8 encoding
- Check file permissions

**2. Syllable Segmentation Unavailable**
```
Warning: Khmer syllable segmentation not available
```

**Solution:**
```python
# Check if subword_cluster module is accessible
try:
    from khtext.subword_cluster import khmer_syllables_advanced
    print("✅ Syllable segmentation available")
except ImportError:
    print("❌ Install syllable segmentation module")
```

**3. Poor Quality Segments**
```python
# Increase attempts for better quality
segment = segment_corpus_text(
    corpus_lines=corpus_lines,
    target_length=15,
    min_length=12,    # Tighter constraints
    max_length=18     # Better filtering
)
```

**4. Curriculum Character Conflicts**
```python
# Check character coverage
allowed_chars = set(['ក', 'ា', 'រ'])
segment_chars = set(segment)
khmer_chars = {c for c in segment_chars if '\u1780' <= c <= '\u17FF'}
print(f"Conflicts: {khmer_chars - allowed_chars}")
```

### Performance Issues

**Slow Generation:**
- Pre-load corpus once, reuse for multiple generations
- Use batch processing for large datasets
- Consider reducing max_attempts in segment_corpus_text

**Memory Usage:**
- Monitor corpus size (current: ~14MB)
- Consider line-by-line processing for very large corpora
- Clear unused variables after generation

## Best Practices

### 1. Content Type Selection

```python
# Use corpus for realistic content
content_types = {
    'characters': 'synthetic',    # Better control
    'syllables': 'synthetic',     # Structured learning
    'words': 'corpus',           # Natural patterns
    'phrases': 'corpus',         # Authentic language
    'mixed': 'corpus'            # Complex structures
}
```

### 2. Length Optimization

```python
# Optimal length ranges for different complexity
length_configs = {
    'simple': (1, 5),      # Character-level
    'medium': (6, 15),     # Word-level  
    'complex': (16, 50)    # Phrase-level
}
```

### 3. Quality Validation

```python
def validate_segment(segment):
    """Validate extracted segment quality."""
    # Check for broken syllables
    if segment.startswith('្') or segment.endswith('្'):
        return False
    
    # Check minimum content
    if len(segment.strip()) < 1:
        return False
        
    # Check character validity
    if not any('\u1780' <= c <= '\u17FF' for c in segment):
        return False
        
    return True
```

### 4. Curriculum Integration

```python
# Progressive character introduction
stages = {
    'stage1': {'chars': 30, 'complexity': 'simple'},
    'stage2': {'chars': 60, 'complexity': 'medium'},  
    'stage3': {'chars': 115, 'complexity': 'complex'}
}

for stage_name, config in stages.items():
    dataset = generator.generate_curriculum_dataset(
        stage=stage_name,
        num_samples=1000,
        use_corpus=True  # Enable authentic text
    )
```

## Future Enhancements

### Planned Features

1. **Domain-Specific Corpora**: Agriculture, technology, literature, news
2. **Advanced Filtering**: Part-of-speech, semantic categories
3. **Context Awareness**: Maintain semantic coherence in segments
4. **Quality Metrics**: Automated assessment of extracted text quality
5. **Multi-Corpus Support**: Blend multiple corpus sources

### Research Directions

1. **Semantic Segmentation**: Meaning-preserving text boundaries
2. **Style Transfer**: Adapt corpus text to different writing styles
3. **Frequency Balancing**: Dynamic character frequency adjustment
4. **Active Learning**: Iterative corpus expansion based on model needs

## Conclusion

The corpus-based text generation system with syllable-aware segmentation provides a significant upgrade in OCR training data quality. By leveraging authentic Khmer text and proper script structure preservation, it enables training of more accurate and robust OCR models for real-world Khmer text recognition tasks.

For technical support or advanced configurations, refer to the API documentation or contact the development team. 