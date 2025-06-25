# Khmer Digits OCR Prototype

A proof-of-concept OCR system for recognizing Khmer digits (០១២៣៤៥៦៧៨៩) in synthetic images containing 1-8 digit sequences.

## 🎯 Project Overview

This prototype uses a hybrid CNN-RNN architecture with attention mechanisms to recognize Khmer digits. The system is designed to validate the OCR concept and provide a foundation for scaling to full Khmer text recognition.

## 📁 Project Structure

```
kh_ocr_prototype/
├── docs/                           # Documentation
│   ├── model_description.md        # Technical model specifications
│   ├── workplan.md                 # Development plan
│   ├── technical_analysis.md       # Architecture validation
│   └── changes.md                  # Change tracking
├── src/                            # Source code
│   ├── fonts/                      # Khmer fonts (8 fonts included)
│   ├── core/                       # Core functionality
│   │   └── models/                 # Model architectures
│   ├── modules/                    # Main modules
│   │   ├── synthetic_data_generator/ # Data generation
│   │   ├── ocr_models/             # OCR model implementations
│   │   ├── trainers/               # Training logic
│   │   └── training_scripts/       # Training scripts
│   └── sample_scripts/             # Example scripts
├── tests/                          # Unit tests
├── generated_data/                 # Generated synthetic datasets
├── config/                         # Configuration files
│   └── model_config.yaml          # Model configuration
├── requirements.txt                # Python dependencies
├── .gitignore                      # Git ignore patterns
└── README.md                       # This file
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Verify Font Installation

```bash
python -c "
import os
font_dir = 'src/fonts'
fonts = [f for f in os.listdir(font_dir) if f.endswith('.ttf')]
print(f'Found {len(fonts)} Khmer fonts:')
for font in fonts:
    print(f'  - {font}')
"
```

### 3. Generate Synthetic Data

```bash
# Will be implemented in Phase 1
python src/modules/synthetic_data_generator/generate_dataset.py
```

### 4. Train Model

```bash
# Will be implemented in Phase 2
python src/modules/training_scripts/train_model.py --config config/model_config.yaml
```

## 🔧 Technical Specifications

- **Input**: RGB images (128x64 pixels)
- **Sequence Length**: 1-8 Khmer digits
- **Architecture**: CNN (ResNet-18) + Bidirectional LSTM + Attention
- **Character Set**: 10 Khmer digits + 3 special tokens (13 classes total)
- **Training Data**: 15,000 synthetic images with augmentation

## 📊 Performance Targets

- **Character Accuracy**: >95%
- **Sequence Accuracy**: >90%
- **Inference Speed**: <100ms per image
- **Model Size**: <20MB

## 🎨 Available Fonts

The project includes 8 high-quality Khmer fonts:

1. KhmerOS.ttf
2. KhmerOSbattambang.ttf
3. KhmerOSbokor.ttf
4. KhmerOSfasthand.ttf
5. KhmerOSmetalchrieng.ttf
6. KhmerOSmuol.ttf
7. KhmerOSmuollight.ttf
8. KhmerOSsiemreap.ttf

## 🗓️ Development Timeline

- **Week 1**: Environment setup and data generation
- **Week 2**: Model development and training infrastructure
- **Week 3**: Model optimization and evaluation
- **Week 4**: Integration, testing, and documentation

## 🔮 Future Expansion

This prototype is designed for easy scaling to full Khmer text recognition:

- **Character Set**: Expandable from 10 digits to 84+ Khmer characters
- **Sequence Length**: Configurable from 8 digits to 32-64 characters for full words
- **Input Size**: Scalable to larger images for document processing

## 📝 Development Status

Current implementation status:
- [x] Project structure created
- [x] Documentation completed
- [x] Configuration files prepared
- [x] Khmer fonts collected (8 fonts)
- [x] Synthetic data generator (Phase 1) - **COMPLETED**
- [x] Dataset generation (5,000 samples: 4,000 train + 1,000 val)
- [x] Data pipeline and utilities - **COMPLETED**
- [x] Model architecture implementation (Phase 2) - **COMPLETED**
- [x] Training infrastructure - **COMPLETED**
- [x] Initial training and debugging - **COMPLETED**
- [x] Training pipeline validation - **COMPLETED**
- [🔄] Model optimization and hyperparameter tuning (Phase 3.1) - **IN PROGRESS**
- [ ] Comprehensive evaluation framework (Phase 3.2)
- [ ] Performance optimization and GPU training (Phase 3.3)
- [ ] Production deployment utilities (Phase 4)

**🎯 Current Status:** Phase 3.1 hyperparameter tuning **IN PROGRESS**. Conservative small model experiment running in background. Successfully implemented systematic hyperparameter tuning infrastructure targeting 85% character accuracy from 24% baseline.

## 📄 License

This project is developed for research and educational purposes.

## 🤝 Contributing

This is a prototype project. Please refer to the workplan and technical documentation for implementation details.

## 🔗 References

- [Model Description](docs/model_description.md)
- [Development Workplan](docs/workplan.md)
- [Technical Analysis](docs/technical_analysis.md) 