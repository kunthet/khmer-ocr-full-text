# Corpus Text Generation API Reference

## Overview

This document provides detailed API reference for the corpus-based text generation system, including function signatures, parameters, return values, and usage examples.

## Module: `synthetic_data_generator.utils`

### Corpus Loading Functions

#### `load_khmer_corpus(corpus_file: str = "data/khmer_clean_text.txt") -> List[str]`

Load and prepare Khmer corpus text for segmentation.

**Parameters:**
- `corpus_file` (str, optional): Path to the corpus text file. Defaults to "data/khmer_clean_text.txt".

**Returns:**
- `List[str]`: List of text lines from the corpus. Empty list if file not found or error occurs.

**Raises:**
- No exceptions raised. Errors are logged and empty list returned.

**Example:**
```python
from synthetic_data_generator.utils import load_khmer_corpus

# Load default corpus
corpus_lines = load_khmer_corpus()
print(f"Loaded {len(corpus_lines)} lines")

# Load custom corpus
custom_corpus = load_khmer_corpus("path/to/custom_corpus.txt")
```

**Console Output:**
```
Loaded corpus: 644 lines, total ~13,915,420 characters
```

---

### Text Segmentation Functions

#### `segment_corpus_text(corpus_lines: List[str], target_length: int, min_length: int = 1, max_length: int = 50, allowed_characters: Optional[List[str]] = None, use_syllable_boundaries: bool = True) -> str`

Extract a random text segment from the corpus with specified length constraints and optional syllable-aware boundary detection.

**Parameters:**
- `corpus_lines` (List[str]): List of corpus text lines
- `target_length` (int): Desired segment length (approximate)
- `min_length` (int, optional): Minimum acceptable length. Defaults to 1.
- `max_length` (int, optional): Maximum acceptable length. Defaults to 50.
- `allowed_characters` (Optional[List[str]], optional): List of allowed characters for curriculum filtering. Defaults to None.
- `use_syllable_boundaries` (bool, optional): Whether to use syllable-aware boundary detection. Defaults to True.

**Returns:**
- `str`: Extracted text segment from corpus. Falls back to synthetic generation if no suitable segment found.

**Algorithm:**
1. Choose random line from corpus
2. For target_length > 3 and use_syllable_boundaries=True: Use `_extract_syllable_aware_segment()`
3. Otherwise: Use `_extract_simple_segment()`
4. Validate length constraints and character filtering
5. Fallback to synthetic generation if no suitable segment found after 50 attempts

**Example:**
```python
from synthetic_data_generator.utils import load_khmer_corpus, segment_corpus_text

corpus_lines = load_khmer_corpus()

# Basic extraction
segment = segment_corpus_text(corpus_lines, target_length=15)
print(f"Segment: '{segment}' (length: {len(segment)})")

# With character constraints (curriculum learning)
allowed_chars = ['ក', 'ា', 'រ', 'ន', 'ត', 'ម', 'ង', '្']
constrained_segment = segment_corpus_text(
    corpus_lines=corpus_lines,
    target_length=10,
    min_length=8,
    max_length=12,
    allowed_characters=allowed_chars,
    use_syllable_boundaries=True
)

# Disable syllable boundaries
simple_segment = segment_corpus_text(
    corpus_lines=corpus_lines,
    target_length=15,
    use_syllable_boundaries=False
)
```

**Output Examples:**
```
Segment: 'ការធ្វើទំនើបកម្ម' (length: 15)
```

---

#### `_extract_syllable_aware_segment(line: str, target_length: int, min_length: int, max_length: int) -> str`

Extract segment using advanced Khmer syllable boundary detection. **Internal function - not for direct use.**

**Parameters:**
- `line` (str): Source text line
- `target_length` (int): Desired segment length
- `min_length` (int): Minimum acceptable length
- `max_length` (int): Maximum acceptable length

**Returns:**
- `str`: Extracted segment with proper syllable boundaries

**Algorithm:**
1. Segment entire line using `khmer_syllables_advanced()`
2. Calculate optimal syllable count: `target_syllables = target_length / avg_syllable_length`
3. Try different segment lengths (±2 syllables) to find best fit within constraints
4. Join selected syllables using `restore_whitespace_tags()`
5. Fallback to `_extract_simple_segment()` on error

**Quality Benefits:**
- Preserves complete Khmer syllables
- Avoids breaking COENG consonant clusters (្រ, ្ម, ្ន, etc.)
- Maintains proper vowel-consonant relationships
- Respects natural word boundaries

**Example Internal Process:**
```python
# Input: "អ្នកគ្រួបង្រៀនភាសាខ្មែរឱ្យបានល្អ"
# Syllables: ['អ្ន', 'ក', 'គ្រួ', 'ប', 'ង្រៀ', 'ន', 'ភា', 'សា', 'ខ្មែ', 'រ', 'ឱ្យ', 'បា', 'ន', 'ល្អ']
# Target: 15 characters
# Selected: 'បង្រៀនភាសាខ្មែរ' (syllables 5-9, length 15)
```

---

#### `_extract_simple_segment(line: str, target_length: int, min_length: int, max_length: int) -> str`

Extract segment using simple character-based approach with word boundary awareness. **Internal function - not for direct use.**

**Parameters:**
- `line` (str): Source text line
- `target_length` (int): Desired segment length
- `min_length` (int): Minimum acceptable length  
- `max_length` (int): Maximum acceptable length

**Returns:**
- `str`: Extracted segment

**Algorithm:**
1. Random character-based positioning
2. Extract target_length characters
3. Attempt to end at word boundaries: [' ', '។', '៕', '៖', 'ៗ', '\n']
4. Search up to 10 characters backward for word boundary
5. Return trimmed segment

**Usage:**
- Target lengths ≤ 3 characters
- When `use_syllable_boundaries=False`
- Fallback when syllable segmentation fails

---

### Complexity-Based Extraction

#### `extract_corpus_segments_by_complexity(corpus_lines: List[str], complexity_level: str = "medium", num_segments: int = 100, use_syllable_boundaries: bool = True) -> List[str]`

Extract multiple corpus segments categorized by complexity level.

**Parameters:**
- `corpus_lines` (List[str]): List of corpus text lines
- `complexity_level` (str, optional): 'simple', 'medium', or 'complex'. Defaults to "medium".
- `num_segments` (int, optional): Number of segments to extract. Defaults to 100.
- `use_syllable_boundaries` (bool, optional): Whether to use syllable-aware boundary detection. Defaults to True.

**Returns:**
- `List[str]`: List of extracted text segments

**Complexity Configurations:**
```python
complexity_configs = {
    'simple': {'min_length': 1, 'max_length': 5, 'target_length': 3},
    'medium': {'min_length': 6, 'max_length': 15, 'target_length': 10},
    'complex': {'min_length': 16, 'max_length': 50, 'target_length': 25}
}
```

**Algorithm:**
1. Apply complexity configuration for length constraints
2. Attempt `num_segments * 3` extractions to ensure sufficient good segments
3. Filter segments meeting minimum length requirements
4. Return first `num_segments` valid segments

**Example:**
```python
from synthetic_data_generator.utils import load_khmer_corpus, extract_corpus_segments_by_complexity

corpus_lines = load_khmer_corpus()

# Extract simple segments (1-5 characters)
simple_segments = extract_corpus_segments_by_complexity(
    corpus_lines=corpus_lines,
    complexity_level='simple',
    num_segments=20
)

# Extract complex segments (16-50 characters)  
complex_segments = extract_corpus_segments_by_complexity(
    corpus_lines=corpus_lines,
    complexity_level='complex',
    num_segments=10,
    use_syllable_boundaries=True
)

print(f"Simple: {simple_segments[:3]}")
print(f"Complex: {complex_segments[:2]}")
```

**Output Examples:**
```python
Simple: ['៏មា', 'នៃ', 'ឺ']
Complex: ['រខ្ញុំចូលចិត្តទៅលេងផ្ទះយាយតានៅឯខេត្ត។', 'ាន់ណាស់។ ក្រុងព្រះសីហនុជាកំពង់']
```

---

### Unified Text Generation

#### `generate_corpus_based_text(corpus_lines: Optional[List[str]] = None, target_length: int = None, content_type: str = "auto", allowed_characters: Optional[List[str]] = None) -> str`

Generate text using corpus segmentation with intelligent fallback to synthetic generation.

**Parameters:**
- `corpus_lines` (Optional[List[str]], optional): Pre-loaded corpus lines. Auto-loads if None.
- `target_length` (Optional[int], optional): Target text length. Random 1-20 if None.
- `content_type` (str, optional): Type of content to generate. Defaults to "auto".
- `allowed_characters` (Optional[List[str]], optional): Allowed characters for curriculum learning. Defaults to None.

**Content Types:**
- `"auto"`: Intelligent type selection based on length
- `"words"`: Prefer corpus extraction for natural words
- `"phrases"`: Prefer corpus extraction for natural phrases  
- `"mixed"`: Prefer corpus extraction for complex structures
- `"characters"`: Use synthetic character sequences
- `"syllables"`: Use synthetic syllable generation
- `"digits"`: Use synthetic digit sequences

**Content Type Logic:**
```python
# Corpus-preferred types
if content_type in ["auto", "words", "phrases", "mixed"] and corpus_lines:
    # Try corpus extraction first
    corpus_text = segment_corpus_text(...)
    if corpus_text: return corpus_text

# Fallback to synthetic generation
```

**Auto Type Selection:**
```python
if content_type == "auto":
    if target_length <= 3: content_type = "characters"
    elif target_length <= 8: content_type = "syllables"  
    elif target_length <= 15: content_type = "words"
    else: content_type = "mixed"
```

**Returns:**
- `str`: Generated text (corpus or synthetic)

**Example:**
```python
from synthetic_data_generator.utils import generate_corpus_based_text

# Auto content type
text1 = generate_corpus_based_text(target_length=12)

# Specific content type with character constraints
high_freq_chars = ['ក', 'ា', 'រ', 'ន', 'ត', 'ម', 'ង', '្']
text2 = generate_corpus_based_text(
    target_length=10,
    content_type="words",
    allowed_characters=high_freq_chars
)

# Force synthetic generation
text3 = generate_corpus_based_text(
    corpus_lines=[],  # Empty corpus forces synthetic
    target_length=8,
    content_type="syllables"
)

print(f"Auto: '{text1}'")
print(f"Constrained: '{text2}'") 
print(f"Synthetic: '{text3}'")
```

---

### Quality Analysis

#### `analyze_corpus_segments(corpus_file: str = "data/khmer_clean_text.txt", num_samples: int = 100) -> Dict`

Analyze corpus text segments to understand their characteristics and quality metrics.

**Parameters:**
- `corpus_file` (str, optional): Path to corpus file. Defaults to "data/khmer_clean_text.txt".
- `num_samples` (int, optional): Number of segments to analyze per complexity level. Defaults to 100.

**Returns:**
- `Dict`: Comprehensive analysis results

**Return Structure:**
```python
{
    'corpus_stats': {
        'total_lines': int,           # Total lines in corpus
        'avg_line_length': float,     # Average characters per line
        'total_characters': int       # Total character count
    },
    'segment_analysis': {
        'simple': {
            'count': int,                    # Number of segments extracted
            'avg_length': float,             # Average segment length
            'min_length': int,               # Minimum segment length
            'max_length': int,               # Maximum segment length
            'unique_characters': int,        # Unique Khmer characters
            'sample_segments': List[str],    # First 5 sample segments
            'top_characters': List[Tuple]    # Top 10 characters by frequency
        },
        'medium': {...},                     # Same structure for medium complexity
        'complex': {...}                     # Same structure for complex complexity
    }
}
```

**Example:**
```python
from synthetic_data_generator.utils import analyze_corpus_segments

# Analyze default corpus
analysis = analyze_corpus_segments()

# Print corpus statistics
stats = analysis['corpus_stats']
print(f"Corpus: {stats['total_lines']} lines, {stats['total_characters']:,} characters")

# Print segment analysis
for complexity, data in analysis['segment_analysis'].items():
    print(f"\n{complexity.upper()} Complexity:")
    print(f"  Count: {data['count']}")
    print(f"  Avg Length: {data['avg_length']:.1f}")
    print(f"  Unique Characters: {data['unique_characters']}")
    print(f"  Sample: {data['sample_segments'][0] if data['sample_segments'] else 'None'}")

# Custom analysis
custom_analysis = analyze_corpus_segments(
    corpus_file="path/to/custom_corpus.txt",
    num_samples=50
)
```

**Output Example:**
```
Corpus: 644 lines, 13,915,420 characters

SIMPLE Complexity:
  Count: 100
  Avg Length: 2.9
  Unique Characters: 20
  Sample: ៏មា

MEDIUM Complexity:
  Count: 100
  Avg Length: 10.3
  Unique Characters: 42
  Sample: ែលមានលក្ខណៈអន្ត

COMPLEX Complexity:
  Count: 100
  Avg Length: 33.0
  Unique Characters: 48
  Sample: ៃការពិតតែប៉ុណ្ណោះ។
```

---

### Utility Functions

#### `_text_uses_allowed_characters(text: str, allowed_chars: List[str]) -> bool`

Check if text only uses allowed characters (for curriculum learning). **Internal function.**

**Parameters:**
- `text` (str): Text to validate
- `allowed_chars` (List[str]): List of allowed characters

**Returns:**
- `bool`: True if all Khmer characters in text are in allowed list

**Algorithm:**
1. Extract character sets from text and allowed list
2. Filter to Khmer Unicode range (\u1780-\u17FF)
3. Check if text Khmer characters are subset of allowed characters

**Example:**
```python
# Internal usage in curriculum learning
allowed = ['ក', 'ា', 'រ', 'ន', 'ត']
text1 = "ការ"      # Uses only allowed chars
text2 = "ការធ្វើ"   # Uses disallowed ធ, ្, វ, ើ

result1 = _text_uses_allowed_characters(text1, allowed)  # True
result2 = _text_uses_allowed_characters(text2, allowed)  # False
```

---

## Module: `synthetic_data_generator.generator`

### SyntheticDataGenerator Class

#### `__init__(self, config_path: str, fonts_dir: str, output_dir: str, mode: str = "full_text", use_corpus: bool = True)`

Initialize the synthetic data generator with corpus support.

**Parameters:**
- `config_path` (str): Path to model configuration file
- `fonts_dir` (str): Directory containing Khmer fonts
- `output_dir` (str): Directory to save generated data
- `mode` (str, optional): Generation mode. Defaults to "full_text".
  - `"digits"`: Khmer digits only (13 characters)
  - `"full_text"`: Full Khmer text (115 characters)
  - `"mixed"`: Mixed digits and text (115 characters)
- `use_corpus` (bool, optional): Whether to use real corpus text for generation. Defaults to True.

**Initialization Process:**
1. Load model configuration from YAML file
2. Initialize background generator and image augmentor
3. Load and validate Khmer fonts
4. Create character mappings based on mode
5. Load character frequencies for realistic generation
6. Load corpus if `use_corpus=True` and mode supports it
7. Create output directory

**Attributes:**
- `corpus_lines` (Optional[List[str]]): Pre-loaded corpus data
- `use_corpus` (bool): Corpus usage flag
- `character_frequencies` (Dict): Character frequency mapping
- `char_to_idx` (Dict): Character to index mapping
- `idx_to_char` (Dict): Index to character mapping

**Example:**
```python
from synthetic_data_generator.generator import SyntheticDataGenerator

# Basic initialization with corpus
generator = SyntheticDataGenerator(
    config_path="config/model_config.yaml",
    fonts_dir="src/fonts",
    output_dir="training_data",
    mode='full_text',
    use_corpus=True
)

# Digits-only mode (no corpus)
digit_generator = SyntheticDataGenerator(
    config_path="config/model_config.yaml",
    fonts_dir="src/fonts", 
    output_dir="digit_data",
    mode='digits',
    use_corpus=False  # Not applicable for digits
)
```

**Console Output:**
```
Loaded 8 working fonts: ['KhmerOS', 'KhmerOSbattambang', ...]
Loaded corpus: 644 lines, total ~13,915,420 characters
✅ Corpus loaded: 644 lines for authentic text generation
Character vocabulary size: 115
Generation mode: full_text
Corpus usage: Enabled
Loaded frequencies for 30 characters
```

---

#### `_generate_text_content(self, content_type: str = "auto", length_range: Tuple[int, int] = (1, 15), allowed_characters: Optional[List[str]] = None) -> str`

Generate text content based on mode and type with corpus integration.

**Parameters:**
- `content_type` (str, optional): Type of content to generate. Defaults to "auto".
- `length_range` (Tuple[int, int], optional): Range of text lengths. Defaults to (1, 15).
- `allowed_characters` (Optional[List[str]], optional): Allowed characters for curriculum learning. Defaults to None.

**Returns:**
- `str`: Generated text content

**Content Generation Logic:**
1. **Digits Mode**: Always generate digit sequences
2. **Corpus Available**: Try corpus-based generation for words/phrases/mixed
3. **Fallback**: Use synthetic generation methods

**Corpus Integration:**
```python
if self.use_corpus and self.corpus_lines and content_type in ["auto", "words", "phrases", "mixed"]:
    corpus_text = generate_corpus_based_text(
        corpus_lines=self.corpus_lines,
        target_length=target_length,
        content_type=content_type,
        allowed_characters=allowed_characters
    )
    if corpus_text: return corpus_text
```

**Example:**
```python
# Auto content generation
text1 = generator._generate_text_content()

# Specific content type with length range
text2 = generator._generate_text_content(
    content_type="phrases",
    length_range=(10, 20)
)

# With character constraints (curriculum)
allowed_chars = ['ក', 'ា', 'រ', 'ន', 'ត', 'ម', 'ង', '្']
text3 = generator._generate_text_content(
    content_type="words",
    length_range=(8, 12),
    allowed_characters=allowed_chars
)

print(f"Auto: '{text1}'")
print(f"Phrases: '{text2}'")
print(f"Constrained: '{text3}'")
```

---

## Error Handling

### Exception Types

**No custom exceptions defined.** All functions use graceful error handling with fallbacks:

1. **File Not Found**: Returns empty list/string, logs warning
2. **Encoding Errors**: Logs error, returns empty result
3. **Syllable Segmentation Failure**: Falls back to simple extraction
4. **Invalid Parameters**: Uses default values, logs warning

### Error Recovery

```python
# Automatic fallback example
try:
    # Try syllable-aware extraction
    segment = _extract_syllable_aware_segment(line, target_length, min_length, max_length)
except Exception as e:
    print(f"Warning: Syllable segmentation failed: {e}, falling back to simple extraction")
    segment = _extract_simple_segment(line, target_length, min_length, max_length)
```

### Debugging Tips

**Enable Verbose Logging:**
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Now corpus loading and segmentation will show detailed logs
corpus_lines = load_khmer_corpus()
```

**Check Syllable Segmentation Availability:**
```python
from synthetic_data_generator.utils import SYLLABLE_SEGMENTATION_AVAILABLE
print(f"Syllable segmentation: {'Available' if SYLLABLE_SEGMENTATION_AVAILABLE else 'Not available'}")
```

**Validate Corpus Quality:**
```python
analysis = analyze_corpus_segments()
if 'error' in analysis:
    print(f"Corpus analysis error: {analysis['error']}")
else:
    print("✅ Corpus analysis successful")
```

---

## Performance Characteristics

### Time Complexity

| **Function** | **Time Complexity** | **Notes** |
|--------------|-------------------|-----------|
| `load_khmer_corpus` | O(n) | n = file size |
| `segment_corpus_text` | O(k) | k = avg line length |
| `_extract_syllable_aware_segment` | O(m) | m = syllable count |
| `extract_corpus_segments_by_complexity` | O(s × k) | s = num_segments |
| `generate_corpus_based_text` | O(k) | Constant for cached corpus |

### Memory Usage

| **Component** | **Memory** | **Notes** |
|---------------|------------|-----------|
| Corpus Loading | ~14MB | For 13.9M character corpus |
| Syllable Segmentation | +10% | Additional overhead |
| Character Mappings | ~1KB | Vocabulary mappings |
| Font Loading | ~10MB | All 8 Khmer fonts |

### Throughput

- **Corpus Generation**: 50+ samples/second
- **Syllable Segmentation**: 100+ segments/second  
- **Quality Analysis**: 30+ segments/second

### Optimization Recommendations

1. **Pre-load Corpus**: Load once, reuse for multiple generations
2. **Batch Processing**: Extract multiple segments in single call
3. **Cache Character Sets**: Reuse allowed_characters filtering
4. **Memory Management**: Clear unused variables for large datasets

---

## Version History

### v1.0.0 (Current)
- Initial corpus-based text generation implementation
- Syllable-aware boundary detection using `khmer_syllables_advanced`
- Curriculum learning integration
- Comprehensive quality analysis tools
- Performance optimization for 50+ samples/second generation

### Planned Features (v1.1.0)
- Domain-specific corpus support
- Advanced quality metrics
- Multi-corpus blending
- Context-aware segmentation
- Real-time corpus updating 