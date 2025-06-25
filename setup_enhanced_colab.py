"""
Quick Setup Script for Enhanced Colab Training
Run this in any Colab cell to get Google Drive integration and resumable training.

Usage:
1. Copy this file to your project
2. In Colab, run: exec(open('setup_enhanced_colab.py').read())
3. Follow the prompts
"""

import os
import sys
from pathlib import Path

def setup_enhanced_colab():
    """One-click setup for enhanced Colab training."""
    
    print("🚀 Setting up Enhanced Colab Training...")
    print("=" * 50)
    
    # Step 1: Check GPU
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        print(f"🎮 GPU Available: {gpu_available}")
        if gpu_available:
            print(f"   Device: {torch.cuda.get_device_name(0)}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("   ⚠️ No GPU detected. Enable GPU: Runtime → Change runtime type → GPU")
    except ImportError:
        print("❌ PyTorch not available")
        return False
    
    # Step 2: Check if we're in Colab
    try:
        from google.colab import drive, files
        print("✅ Running in Google Colab")
        in_colab = True
    except ImportError:
        print("⚠️ Not running in Google Colab - some features disabled")
        in_colab = False
    
    # Step 3: Install dependencies
    print("\n📦 Installing dependencies...")
    os.system("pip install -q opencv-python-headless albumentations tensorboard efficientnet-pytorch")
    print("✅ Dependencies installed")
    
    # Step 4: Setup Google Drive (if in Colab)
    if in_colab:
        print("\n💾 Setting up Google Drive...")
        try:
            drive.mount('/content/drive')
            
            # Create directory structure
            DRIVE_ROOT = '/content/drive/MyDrive'
            PROJECT_DRIVE_PATH = f'{DRIVE_ROOT}/Khmer_OCR_Experiments'
            MODELS_DRIVE_PATH = f'{PROJECT_DRIVE_PATH}/training_output'
            RESULTS_DRIVE_PATH = f'{PROJECT_DRIVE_PATH}/results'
            
            for path in [PROJECT_DRIVE_PATH, MODELS_DRIVE_PATH, RESULTS_DRIVE_PATH]:
                os.makedirs(path, exist_ok=True)
            
            print(f"✅ Drive mounted: {PROJECT_DRIVE_PATH}")
            print(f"📁 Models: {MODELS_DRIVE_PATH}")
            print(f"📊 Results: {RESULTS_DRIVE_PATH}")
            
            # Store paths globally
            globals()['MODELS_DRIVE_PATH'] = MODELS_DRIVE_PATH
            globals()['RESULTS_DRIVE_PATH'] = RESULTS_DRIVE_PATH
            
        except Exception as e:
            print(f"❌ Drive setup failed: {e}")
            return False
    
    # Step 5: Check project structure
    print("\n📁 Checking project structure...")
    
    # Find project root
    project_root = None
    for root in ['.', './khmer-ocr-digits', '../', '/content']:
        if os.path.exists(os.path.join(root, 'src')):
            project_root = Path(root).resolve()
            break
    
    if project_root:
        os.chdir(project_root)
        sys.path.append(str(project_root / 'src'))
        print(f"✅ Project root: {project_root}")
        
        # Check required files
        required_files = [
            'src/sample_scripts/enhanced_colab_trainer.py',
            'config/phase3_training_configs.yaml',
            'generated_data/metadata.yaml'
        ]
        
        missing_files = []
        for file_path in required_files:
            if os.path.exists(file_path):
                print(f"✅ {file_path}")
            else:
                print(f"❌ {file_path} - NOT FOUND")
                missing_files.append(file_path)
        
        if missing_files:
            print(f"\n⚠️ Missing {len(missing_files)} required files")
            print("Please upload your complete project to Colab")
            return False
            
    else:
        print("❌ Could not find project root")
        print("Please upload your project files to Colab")
        print("Current directory contents:")
        os.system("ls -la")
        return False
    
    # Step 6: Load enhanced trainer
    print("\n🔧 Loading enhanced trainer...")
    try:
        enhanced_trainer_path = 'src/sample_scripts/enhanced_colab_trainer.py'
        if os.path.exists(enhanced_trainer_path):
            with open(enhanced_trainer_path, 'r') as f:
                exec(f.read(), globals())
            print("✅ Enhanced trainer loaded")
        else:
            print("❌ Enhanced trainer script not found")
            return False
    except Exception as e:
        print(f"❌ Failed to load enhanced trainer: {e}")
        return False
    
    # Step 7: Test imports
    print("\n🧪 Testing imports...")
    try:
        from modules.data_utils import KhmerDigitsDataset
        from models import create_model
        from modules.trainers import OCRTrainer
        print("✅ All imports successful")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please check your project structure")
        return False
    
    print("\n🎉 Setup Complete!")
    print("=" * 50)
    
    # Step 8: Show usage instructions
    if in_colab and 'MODELS_DRIVE_PATH' in globals():
        print("\n🚀 Ready to run enhanced training!")
        print("\nQuick start:")
        print("```python")
        print("# Initialize enhanced tuner")
        print("tuner = EnhancedColabHyperparameterTuner(")
        print(f"    drive_models_path='{globals()['MODELS_DRIVE_PATH']}',")
        print(f"    drive_results_path='{globals()['RESULTS_DRIVE_PATH']}',")
        print("    auto_resume=True")
        print(")")
        print("")
        print("# Run all experiments (auto-resume if interrupted)")
        print("tuner.run_experiments()")
        print("")
        print("# Or run specific experiments:")
        print("tuner.run_experiments(['conservative_small', 'baseline_optimized'])")
        print("")
        print("# Save results to Drive")
        print("results_file = tuner.save_results()")
        print("```")
        
        # Actually create the tuner for immediate use
        try:
            tuner = globals()['EnhancedColabHyperparameterTuner'](
                drive_models_path=globals()['MODELS_DRIVE_PATH'],
                drive_results_path=globals()['RESULTS_DRIVE_PATH'],
                auto_resume=True
            )
            globals()['tuner'] = tuner
            print("\n✨ Enhanced tuner created and ready to use!")
            print("Just run: tuner.run_experiments()")
            
        except Exception as e:
            print(f"\n⚠️ Tuner creation failed: {e}")
            print("You can create it manually using the code above")
    
    else:
        print("\n💡 Basic setup complete (no Drive integration)")
        print("Run the enhanced trainer manually for full features")
    
    return True

# Auto-run if executed directly
if __name__ == "__main__":
    success = setup_enhanced_colab()
    if not success:
        print("\n❌ Setup failed. Please check the requirements and try again.")
    else:
        print("\n🎯 You're all set! Happy training!")

# Also make it easy to run from exec()
if 'get_ipython' in globals():  # Running in Jupyter/Colab
    setup_enhanced_colab() 