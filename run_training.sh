#!/bin/bash
# Wrapper script to fix OpenMP library conflict and run training

export KMP_DUPLICATE_LIB_OK=TRUE
python src/sample_scripts/simple_initial_training.py 