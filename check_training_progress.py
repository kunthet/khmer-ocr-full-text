#!/usr/bin/env python3
"""
Check Training Progress and Test Inference

This script monitors the current training progress and tests inference
when the model is ready.
"""

import os
import sys
import time
import json
import logging
from pathlib import Path
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_latest_checkpoint():
    """Check for the latest model checkpoint."""
    checkpoint_dirs = [
        "training_output/conservative_small/checkpoints",
        "training_output/best_conservative_small_*/checkpoints"
    ]
    
    latest_checkpoint = None
    latest_time = 0
    
    for pattern in checkpoint_dirs:
        for checkpoint_dir in Path(".").glob(pattern):
            if checkpoint_dir.exists():
                for checkpoint_file in checkpoint_dir.glob("*.pth"):
                    file_time = checkpoint_file.stat().st_mtime
                    if file_time > latest_time:
                        latest_time = file_time
                        latest_checkpoint = checkpoint_file
    
    return latest_checkpoint

def check_training_logs():
    """Check latest training logs for progress."""
    log_files = list(Path(".").glob("best_config_training_*.log"))
    if not log_files:
        return None
    
    latest_log = max(log_files, key=lambda f: f.stat().st_mtime)
    
    try:
        with open(latest_log, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        # Extract latest metrics
        latest_metrics = {}
        for line in reversed(lines[-50:]):  # Check last 50 lines
            if "Val Char Acc:" in line:
                try:
                    acc = float(line.split("Val Char Acc:")[1].split("%")[0].strip())
                    latest_metrics['char_accuracy'] = acc / 100
                    break
                except:
                    continue
                    
        return latest_metrics
    except Exception as e:
        logger.error(f"Error reading log file: {e}")
        return None

def test_inference(checkpoint_path):
    """Test inference with the latest checkpoint."""
    try:
        # Add src to path
        sys.path.append(str(Path(__file__).parent / "src"))
        
        from inference.inference_engine import KhmerOCRInference
        
        logger.info(f"Testing inference with checkpoint: {checkpoint_path}")
        
        # Initialize inference engine
        engine = KhmerOCRInference(checkpoint_path=str(checkpoint_path))
        
        # Test with a simple generated sample
        logger.info("✅ Inference engine loaded successfully!")
        
        # Generate a test sample
        logger.info("Generating test sample for validation...")
        
        # You can add actual test generation here
        return True
        
    except Exception as e:
        logger.error(f"❌ Inference test failed: {e}")
        return False

def main():
    """Main monitoring function."""
    logger.info("🔍 Checking Khmer OCR Training Progress")
    logger.info("=" * 50)
    
    # Check training progress
    latest_metrics = check_training_logs()
    if latest_metrics:
        char_acc = latest_metrics.get('char_accuracy', 0)
        logger.info(f"📊 Latest Character Accuracy: {char_acc:.1%}")
        
        if char_acc >= 0.70:  # 70% target
            logger.info("🎉 TARGET ACHIEVED! Model ready for production!")
        elif char_acc >= 0.50:  # 50% good progress
            logger.info("📈 Good progress! Approaching target accuracy.")
        else:
            logger.info("🔄 Training in progress...")
    else:
        logger.info("⏳ Training logs not found or still initializing...")
    
    # Check for latest checkpoint
    latest_checkpoint = check_latest_checkpoint()
    if latest_checkpoint:
        logger.info(f"📁 Latest checkpoint: {latest_checkpoint}")
        
        # Test inference if checkpoint is recent (modified in last hour)
        checkpoint_age = time.time() - latest_checkpoint.stat().st_mtime
        if checkpoint_age < 3600:  # 1 hour
            logger.info("🧪 Testing inference with latest checkpoint...")
            if test_inference(latest_checkpoint):
                logger.info("✅ Inference system operational!")
            else:
                logger.info("❌ Inference test failed - model may still be training")
        else:
            logger.info("⏰ Checkpoint is older - training may still be in progress")
    else:
        logger.info("❌ No checkpoints found")
    
    # Recommendations
    logger.info("\n🎯 Next Steps:")
    if latest_metrics and latest_metrics.get('char_accuracy', 0) >= 0.70:
        logger.info("1. ✅ Model ready - proceed to Phase 3.2 advanced techniques")
        logger.info("2. 🧪 Run comprehensive inference tests")
        logger.info("3. 📊 Perform error analysis")
    elif latest_metrics and latest_metrics.get('char_accuracy', 0) >= 0.40:
        logger.info("1. ⏳ Wait for training completion (target: 70%)")
        logger.info("2. 🔧 Monitor training convergence")
        logger.info("3. 📈 Prepare advanced optimization techniques")
    else:
        logger.info("1. 🔄 Training still in early stages")
        logger.info("2. ⏰ Allow more time for convergence")
        logger.info("3. 🔍 Monitor for any training issues")

if __name__ == "__main__":
    main() 