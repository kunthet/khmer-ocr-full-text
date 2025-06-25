# Corpus Text Generation Quick Reference

## Quick Start

```python
from synthetic_data_generator.generator import SyntheticDataGenerator

# Enable corpus-based generation
generator = SyntheticDataGenerator(
    config_path="config/model_config.yaml",
    fonts_dir="src/fonts",
    output_dir="training_data",
    mode='full_text',
    use_corpus=True  # Enable authentic text
)

# Generate authentic Khmer text
text = generator._generate_text_content(
    content_type="phrases",
    length_range=(10, 20)
)
```

## Key Functions

### Corpus Loading
```python
from synthetic_data_generator.utils import load_khmer_corpus

corpus_lines = load_khmer_corpus()  # Default corpus
# corpus_lines = load_khmer_corpus("custom_corpus.txt")  # Custom corpus
```

### Text Extraction
```python
from synthetic_data_generator.utils import segment_corpus_text

# Basic extraction
segment = segment_corpus_text(corpus_lines, target_length=15)

# With constraints (curriculum learning)
segment = segment_corpus_text(
    corpus_lines=corpus_lines,
    target_length=10,
    min_length=8,
    max_length=12,
    allowed_characters=['ក', 'ា', 'រ', 'ន'],  # Character constraints
    use_syllable_boundaries=True  # Syllable-aware cutting
)
```

### Batch Extraction
```python
from synthetic_data_generator.utils import extract_corpus_segments_by_complexity

# Extract by complexity
simple_segments = extract_corpus_segments_by_complexity(
    corpus_lines, 'simple', num_segments=100
)
medium_segments = extract_corpus_segments_by_complexity(
    corpus_lines, 'medium', num_segments=100
)
complex_segments = extract_corpus_segments_by_complexity(
    corpus_lines, 'complex', num_segments=100
)
```

## Content Types

| Content Type | Best Source | Use Case | Example Output |
|--------------|-------------|----------|----------------|
| `characters` | Synthetic | Character recognition | `'ការរ្'` |
| `syllables` | Synthetic | Syllable learning | `'កាគ្រួ'` |
| `words` | **Corpus** | Word recognition | `'ការធ្វើ'` |
| `phrases` | **Corpus** | Sentence recognition | `'ការធ្វើទំនើបកម្ម'` |
| `mixed` | **Corpus** | Complex structures | `'ការធ្វើទំនើបកម្មវិស័យ'` |
| `digits` | Synthetic | Number recognition | `'៤៥៦៧៨'` |

## Complexity Levels

| Level | Length Range | Character Count | Content Type | Use Case |
|-------|--------------|-----------------|--------------|----------|
| **Simple** | 1-5 chars | 30 high-freq | Characters, syllables | Early training |
| **Medium** | 6-15 chars | 60 med-freq | Words, syllables | Intermediate |
| **Complex** | 16-50 chars | All 115+ | Phrases, sentences | Advanced |

## Curriculum Learning

```python
# Stage 1: High-frequency characters
high_freq_chars = ['ក', 'ា', 'រ', 'ន', 'ត', 'ម', 'ង', '្', 'ស', 'ល']
stage1_dataset = generator.generate_curriculum_dataset(
    stage='stage1',
    num_samples=1000
)

# Stage 2: Medium-frequency characters  
stage2_dataset = generator.generate_curriculum_dataset(
    stage='stage2',
    num_samples=2000
)

# Stage 3: All characters
stage3_dataset = generator.generate_curriculum_dataset(
    stage='stage3', 
    num_samples=3000
)
```

## Quality Analysis

```python
from synthetic_data_generator.utils import analyze_corpus_segments

# Analyze corpus quality
analysis = analyze_corpus_segments(num_samples=100)

# Check results
if 'error' not in analysis:
    stats = analysis['corpus_stats']
    print(f"Lines: {stats['total_lines']:,}")
    print(f"Characters: {stats['total_characters']:,}")
    
    for complexity, data in analysis['segment_analysis'].items():
        print(f"{complexity}: {data['avg_length']:.1f} avg chars")
```

## Performance Tips

### ✅ Do This
```python
# Pre-load corpus once
corpus_lines = load_khmer_corpus()

# Reuse for multiple generations
for i in range(1000):
    segment = segment_corpus_text(corpus_lines, target_length=10)

# Use batch processing
segments = extract_corpus_segments_by_complexity(
    corpus_lines, 'medium', num_segments=1000
)
```

### ❌ Avoid This
```python
# Don't reload corpus repeatedly
for i in range(1000):
    corpus_lines = load_khmer_corpus()  # Slow!
    segment = segment_corpus_text(corpus_lines, target_length=10)

# Don't generate one by one when you need many
segments = []
for i in range(1000):
    segment = segment_corpus_text(corpus_lines, target_length=10)  # Slow!
    segments.append(segment)
```

## Common Parameters

### `segment_corpus_text()`
- `corpus_lines`: Pre-loaded corpus (required)
- `target_length`: Desired segment length (required)
- `min_length`: Minimum acceptable length (default: 1)
- `max_length`: Maximum acceptable length (default: 50)
- `allowed_characters`: Character constraints for curriculum (default: None)
- `use_syllable_boundaries`: Syllable-aware cutting (default: True)

### `SyntheticDataGenerator()`
- `config_path`: Model config file (required)
- `fonts_dir`: Khmer fonts directory (required)
- `output_dir`: Output directory (required)
- `mode`: 'digits', 'full_text', or 'mixed' (default: 'full_text')
- `use_corpus`: Enable corpus generation (default: True)

## Error Handling

### Common Issues & Solutions

**Corpus not loading:**
```python
# Check file exists
import os
print(os.path.exists("data/khmer_clean_text.txt"))

# Check encoding
corpus_lines = load_khmer_corpus()
if not corpus_lines:
    print("Corpus loading failed - check file path and encoding")
```

**Syllable segmentation unavailable:**
```python
from synthetic_data_generator.utils import SYLLABLE_SEGMENTATION_AVAILABLE
if not SYLLABLE_SEGMENTATION_AVAILABLE:
    print("Syllable segmentation not available - using simple extraction")
```

**Poor quality segments:**
```python
# Use stricter constraints
segment = segment_corpus_text(
    corpus_lines,
    target_length=15,
    min_length=12,    # Tighter range
    max_length=18,
    use_syllable_boundaries=True  # Better quality
)
```

## Performance Benchmarks

| Operation | Rate | Notes |
|-----------|------|--------|
| Corpus loading | ~1s | 13.9M chars, one-time cost |
| Individual extraction | 30-50/sec | Single segment generation |
| Batch extraction | 80-120/sec | Multiple segments at once |
| Syllable-aware | 40-60/sec | +10% overhead, better quality |
| Simple extraction | 50-70/sec | Faster, lower quality |

## Output Examples

### Corpus-Based (Authentic)
```python
# Natural Khmer language patterns
"ការធ្វើទំនើបកម្ម"      # Modernization process
"ងមុន។"                # Previous period  
"ារអនុវត"               # Implementation
"តបានជាបរិយាកាស"        # Environmental context
"ការគ្រប់គ្រងដំណាំ"      # Crop management
```

### Synthetic (Artificial)
```python
# Algorithmic patterns
"រ្"                    # Isolated subscript
"កៀៜ"                   # Random combination
"ជិ្ដាយៅ្"              # Unrealistic pattern
"សូទ្ចពនរល"             # Character sequence
```

## Best Practices

### 🎯 Content Strategy
- **Characters/Syllables**: Use synthetic for controlled learning
- **Words/Phrases**: Use corpus for authentic language
- **Mixed Content**: Combine both approaches

### 📚 Curriculum Design
1. **Stage 1**: Simple synthetic + high-frequency corpus
2. **Stage 2**: Medium complexity corpus with character constraints
3. **Stage 3**: Full corpus complexity with all characters

### ⚡ Performance Optimization
1. Pre-load corpus once and reuse
2. Use batch processing for large datasets
3. Enable syllable boundaries for quality
4. Monitor generation rates (target: 50+ samples/sec)

### 🔍 Quality Validation
```python
def validate_segment(segment):
    # Check for broken syllables
    if segment.startswith('្') or segment.endswith('្'):
        return False
    # Check minimum content
    if len(segment.strip()) < 1:
        return False
    return True
```

## Integration Examples

### Basic OCR Dataset
```python
generator = SyntheticDataGenerator(
    config_path="config/model_config.yaml",
    fonts_dir="src/fonts",
    output_dir="ocr_data",
    use_corpus=True
)

dataset = generator.generate_dataset(
    num_samples=5000,
    train_split=0.8,
    save_images=True
)
```

### Curriculum Training
```python
for stage in ['stage1', 'stage2', 'stage3']:
    dataset = generator.generate_curriculum_dataset(
        stage=stage,
        num_samples=2000,
        save_images=True
    )
    print(f"Generated {stage}: {len(dataset['train']['samples'])} samples")
```

### Custom Content
```python
# Load custom corpus
custom_corpus = load_khmer_corpus("domain_specific_corpus.txt")

# Extract domain-specific text
segments = extract_corpus_segments_by_complexity(
    custom_corpus, 'complex', num_segments=500
)
```

## Quick Troubleshooting

| Issue | Check | Solution |
|-------|-------|----------|
| No corpus loaded | File path, encoding | Verify `data/khmer_clean_text.txt` exists |
| Poor segment quality | Syllable boundaries | Set `use_syllable_boundaries=True` |
| Slow generation | Pre-loading, batching | Load corpus once, use batch functions |
| Character violations | Curriculum constraints | Check `allowed_characters` parameter |
| Memory usage | Large datasets | Use batch processing, clear variables |

## Version Information

- **Current Version**: 1.0.0
- **Corpus Support**: 40.9MB Khmer text (644 lines)
- **Character Coverage**: 115+ Khmer characters
- **Syllable Segmentation**: `khmer_syllables_advanced` integration
- **Performance**: 50+ samples/second generation rate 