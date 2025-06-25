#!/usr/bin/env python3
"""
Wrapper script to fix OpenMP issues and run training
"""
import os
import subprocess
import sys

# Set environment variable to fix OpenMP conflict
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Also try setting other environment variables that might help
os.environ['OMP_NUM_THREADS'] = '1'  # Limit threads to avoid conflicts

print("🔧 Setting OpenMP environment variables...")
print(f"KMP_DUPLICATE_LIB_OK = {os.environ.get('KMP_DUPLICATE_LIB_OK')}")
print(f"OMP_NUM_THREADS = {os.environ.get('OMP_NUM_THREADS')}")

try:
    # Run the training script
    print("🚀 Running training script...")
    result = subprocess.run([
        sys.executable, 
        'src/sample_scripts/simple_initial_training.py'
    ], capture_output=True, text=True)
    
    # Print output
    if result.stdout:
        print("STDOUT:")
        print(result.stdout)
    
    if result.stderr:
        print("STDERR:")
        print(result.stderr)
    
    print(f"Exit code: {result.returncode}")
    
except Exception as e:
    print(f"Error running script: {e}")
    sys.exit(1) 