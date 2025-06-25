#!/usr/bin/env python3
"""
Comprehensive Progress Monitor

This script monitors all training processes and provides recommendations
for next steps based on current progress.
"""

import os
import time
import glob
import json
import logging
from pathlib import Path
from datetime import datetime

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def check_training_progress():
    """Check progress of all training processes."""
    logger = setup_logging()
    
    logger.info("🔍 COMPREHENSIVE PROGRESS MONITOR")
    logger.info("=" * 60)
    
    # Check best config training
    logger.info("📊 PHASE 3.1 TRAINING STATUS:")
    best_config_logs = list(Path(".").glob("best_config_training_*.log"))
    if best_config_logs:
        latest_log = max(best_config_logs, key=lambda f: f.stat().st_mtime)
        
        try:
            with open(latest_log, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Find latest epoch info
            latest_epoch = 0
            latest_char_acc = 0
            latest_seq_acc = 0
            
            for line in reversed(lines[-100:]):  # Check last 100 lines
                if "Epoch" in line and "Results:" in line:
                    try:
                        epoch_line = line.split("Epoch")[1].split("Results:")[0].strip()
                        latest_epoch = int(epoch_line)
                        break
                    except:
                        continue
            
            for line in reversed(lines[-20:]):  # Check for latest metrics
                if "Val Char Acc:" in line:
                    try:
                        acc = float(line.split("Val Char Acc:")[1].split("%")[0].strip())
                        latest_char_acc = acc
                        break
                    except:
                        continue
            
            for line in reversed(lines[-20:]):
                if "Val Seq Acc:" in line:
                    try:
                        acc = float(line.split("Val Seq Acc:")[1].split("%")[0].strip())
                        latest_seq_acc = acc
                        break
                    except:
                        continue
            
            logger.info(f"  📈 Current Epoch: {latest_epoch}/50")
            logger.info(f"  🎯 Character Accuracy: {latest_char_acc:.1f}%")
            logger.info(f"  📝 Sequence Accuracy: {latest_seq_acc:.1f}%")
            
            # Progress assessment
            if latest_epoch >= 45:
                logger.info("  ✅ Training nearly complete!")
            elif latest_epoch >= 25:
                logger.info("  🔄 Training more than halfway done")
            elif latest_epoch >= 10:
                logger.info("  ⏳ Training in progress")
            else:
                logger.info("  🚀 Training in early stages")
                
        except Exception as e:
            logger.error(f"  ❌ Error reading log: {e}")
    else:
        logger.info("  ⚠️ No best config training logs found")
    
    # Check checkpoints
    logger.info("\n📁 CHECKPOINT STATUS:")
    checkpoint_dirs = list(Path("training_output").glob("*/checkpoints"))
    
    for checkpoint_dir in checkpoint_dirs:
        if checkpoint_dir.exists():
            checkpoints = list(checkpoint_dir.glob("*.pth"))
            if checkpoints:
                latest_checkpoint = max(checkpoints, key=lambda f: f.stat().st_mtime)
                age_hours = (time.time() - latest_checkpoint.stat().st_mtime) / 3600
                logger.info(f"  📁 {checkpoint_dir.parent.name}: {len(checkpoints)} checkpoints")
                logger.info(f"     Latest: {latest_checkpoint.name} ({age_hours:.1f}h ago)")
    
    # Check Phase 3.2 logs
    logger.info("\n🚀 PHASE 3.2 STATUS:")
    phase32_logs = list(Path(".").glob("phase3_2_optimization_*.log"))
    if phase32_logs:
        latest_log = max(phase32_logs, key=lambda f: f.stat().st_mtime)
        logger.info(f"  📋 Phase 3.2 log found: {latest_log.name}")
        
        try:
            with open(latest_log, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            if len(lines) > 5:
                logger.info("  🔄 Phase 3.2 optimization in progress")
            else:
                logger.info("  🚀 Phase 3.2 optimization starting")
                
        except Exception as e:
            logger.error(f"  ❌ Error reading Phase 3.2 log: {e}")
    else:
        logger.info("  ⏳ Phase 3.2 not yet started")
    
    # Recommendations
    logger.info("\n🎯 RECOMMENDATIONS:")
    
    # Based on current progress
    if latest_char_acc >= 50:
        logger.info("1. ✅ Great progress! Consider starting inference testing")
        logger.info("2. 🧪 Run comprehensive evaluation suite")
        logger.info("3. 📊 Analyze error patterns for targeted improvements")
    elif latest_char_acc >= 35:
        logger.info("1. 📈 Good progress - continue current training")
        logger.info("2. ⏰ Wait for training completion before major changes")
        logger.info("3. 🔧 Prepare Phase 3.2 enhancements")
    else:
        logger.info("1. ⏳ Allow more time for current training")
        logger.info("2. 🔍 Monitor for convergence issues")
        logger.info("3. 🚀 Prepare Phase 3.2 optimization")
    
    # Next steps based on time and progress
    current_hour = datetime.now().hour
    if 6 <= current_hour <= 22:  # Daytime
        logger.info("\n⏰ NEXT ACTIONS (Daytime):")
        logger.info("1. 🔄 Monitor training completion")
        logger.info("2. 🧪 Test inference with latest checkpoints")
        logger.info("3. 📊 Prepare comprehensive evaluation")
    else:  # Night time - background processes
        logger.info("\n🌙 NEXT ACTIONS (Background):")
        logger.info("1. 🔄 Let training processes continue")
        logger.info("2. 📋 Review progress in the morning")
        logger.info("3. 🚀 Phase 3.2 can run overnight")
    
    logger.info("\n📞 SUPPORT COMMANDS:")
    logger.info("  Check progress: python monitor_progress.py")
    logger.info("  Test inference: python run_inference.py --generate --num_samples 5")
    logger.info("  View logs: tail -f best_config_training_*.log")

if __name__ == "__main__":
    check_training_progress() 