# Progressive Training Strategy Guide

## Overview

The Progressive Training Strategy is a systematic approach to training Khmer OCR models through five stages of increasing complexity. This guide provides comprehensive instructions for using the progressive training system implemented in Phase 3.1.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Understanding Progressive Training](#understanding-progressive-training)
3. [Installation and Setup](#installation-and-setup)
4. [Usage Examples](#usage-examples)
5. [Configuration Options](#configuration-options)
6. [Monitoring and Analysis](#monitoring-and-analysis)
7. [Troubleshooting](#troubleshooting)
8. [Advanced Usage](#advanced-usage)

## Quick Start

### Prerequisites
- Python 3.8+
- PyTorch 1.9+
- Valid model configuration file
- Khmer fonts installed

### Basic Usage

1. **Test the system** (recommended first step):
```bash
python src/sample_scripts/test_progressive_training.py
```

2. **Run a dry test** to verify configuration:
```bash
python src/sample_scripts/progressive_training_strategy.py --dry-run
```

3. **Start progressive training**:
```bash
python src/sample_scripts/progressive_training_strategy.py \
    --config config/model_config.yaml \
    --output-dir my_progressive_training
```

## Understanding Progressive Training

### The Five Training Stages

The progressive training strategy follows a curriculum-based approach with five distinct stages:

#### Stage 1: Single Character Recognition
- **Purpose**: Establish basic character recognition capabilities
- **Data**: Individual Khmer characters and digits
- **Success Threshold**: 92% accuracy
- **Duration**: 5-15 epochs
- **Transfer Learning**: Can initialize from pre-trained digits model

#### Stage 2: Simple Character Combinations
- **Purpose**: Learn basic character relationships
- **Data**: Simple combinations (consonant + vowel)
- **Success Threshold**: 88% accuracy
- **Duration**: 8-20 epochs
- **Focus**: Basic syllable structures

#### Stage 3: Complex Character Combinations
- **Purpose**: Master complex Khmer structures
- **Data**: Stacked consonants, multiple diacritics
- **Success Threshold**: 82% accuracy
- **Duration**: 12-25 epochs
- **Focus**: Advanced Khmer typography

#### Stage 4: Word-Level Recognition
- **Purpose**: Understand word boundaries and spacing
- **Data**: Complete words with proper spacing
- **Success Threshold**: 78% accuracy
- **Duration**: 15-30 epochs
- **Focus**: Word segmentation and spacing

#### Stage 5: Multi-Word Recognition
- **Purpose**: Handle sentences and complex text
- **Data**: Multi-word phrases and sentences
- **Success Threshold**: 75% accuracy
- **Duration**: 20-40 epochs
- **Focus**: Full text recognition

### Automatic Progression Logic

The system automatically progresses between stages based on:
- **Performance Thresholds**: Must achieve target accuracy
- **Minimum Epochs**: Ensures adequate training time
- **Maximum Epochs**: Prevents infinite training
- **Improvement Tracking**: Monitors learning progress
