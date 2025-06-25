# Progressive Training Strategy Quick Reference

## Essential Commands

### Testing
```bash
# Run all tests
python src/sample_scripts/test_progressive_training.py

# Dry run test
python src/sample_scripts/progressive_training_strategy.py --dry-run
```

### Training
```bash
# Basic training
python src/sample_scripts/progressive_training_strategy.py

# Custom output directory
python src/sample_scripts/progressive_training_strategy.py \
    --output-dir my_experiment

# Custom config
python src/sample_scripts/progressive_training_strategy.py \
    --config my_config.yaml
```

## 5 Training Stages

| Stage | Focus | Threshold | Epochs | Duration |
|-------|-------|-----------|---------|----------|
| 1 | Single Characters | 92% | 5-15 | Quick |
| 2 | Simple Combinations | 88% | 8-20 | Short |
| 3 | Complex Combinations | 82% | 12-25 | Medium |
| 4 | Word Level | 78% | 15-30 | Long |
| 5 | Multi-word | 75% | 20-40 | Longest |

## Key File Locations

### Scripts
- **Main Script**: `src/sample_scripts/progressive_training_strategy.py`
- **Test Script**: `src/sample_scripts/test_progressive_training.py`

### Configuration
- **Model Config**: `config/model_config.yaml`
- **Documentation**: `docs/progressive_training_strategy_*.md`

### Output Structure
```
progressive_training_output/
├── checkpoints/          # Model checkpoints
├── training_data/        # Generated data
└── progressive_training_report.json
```

## Basic Python Usage

```python
from progressive_training_strategy import ProgressiveTrainingStrategy

# Initialize
strategy = ProgressiveTrainingStrategy("config/model_config.yaml")

# Run training
report = strategy.run_progressive_training()

# Check results
print(f"Final accuracy: {report['final_performance']:.3f}")
```

## Configuration Template

```yaml
# model_config.yaml
model:
  name: "KhmerTextOCR"
  vocab_size: 115
  hidden_dim: 512

training:
  batch_size: 32
  learning_rate: 0.001

data:
  image_height: 32
  image_width: 128
```

## Troubleshooting

### Common Issues
- **Import Error**: Check you're in project root directory
- **Config Not Found**: Verify file path and existence
- **GPU Memory**: Reduce batch size or use CPU
- **Slow Progress**: Check success thresholds aren't too high

### Quick Fixes
```bash
# Check current directory
pwd

# Verify config exists
ls config/model_config.yaml

# Run with CPU only
CUDA_VISIBLE_DEVICES="" python src/sample_scripts/progressive_training_strategy.py
```

## Integration Points

### With Existing Components
- **Phase 2.3**: Uses KhmerTextOCR model
- **Phase 2.4**: Uses curriculum learning infrastructure
- **Data Generation**: Compatible with enhanced dataset generation

### Expected Test Output
```
🎯 Overall: 3/3 tests passed (100.0%)
🎉 All tests passed! Progressive Training Strategy (Phase 3.1) is ready!
```

## Performance Expectations

### Timing
- **Setup**: < 1 minute
- **Each Stage**: 15-60 minutes
- **Total Training**: 2-8 hours

### Accuracy Progression
- Stage 1: ~92% (characters)
- Stage 2: ~88% (simple combinations)  
- Stage 3: ~82% (complex combinations)
- Stage 4: ~78% (words)
- Stage 5: ~75% (sentences)

## Next Steps

After successful training:
1. **Check final model**: `progressive_training_output/checkpoints/stage_5_best.pth`
2. **Review report**: `progressive_training_output/progressive_training_report.json`
3. **Proceed to Phase 3.2**: Hyperparameter optimization

## Support

- **Full Guide**: `docs/progressive_training_strategy_guide.md`
- **API Reference**: `docs/progressive_training_strategy_api_reference.md`
- **Implementation Summary**: `PHASE_3_1_PROGRESSIVE_TRAINING_STRATEGY_SUMMARY.md` 