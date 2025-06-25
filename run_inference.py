#!/usr/bin/env python3
"""
Khmer Digits OCR Inference Script

This script demonstrates how to use the trained Khmer OCR model for inference.
It can process single images, batches of images, or entire directories.

Usage examples:
    # Single image inference
    python run_inference.py --image path/to/image.png
    
    # Batch inference from directory
    python run_inference.py --directory path/to/images --batch_size 16
    
    # With custom checkpoint
    python run_inference.py --checkpoint custom_model.pth --image test.png
    
    # With visualization
    python run_inference.py --image test.png --visualize --output_dir results/
"""

import argparse
import os
import sys
import logging
from pathlib import Path
from typing import List, Optional
import json

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from inference.inference_engine import KhmerOCRInference, setup_logging
from modules.synthetic_data_generator.generator import SyntheticDataGenerator


def main():
    parser = argparse.ArgumentParser(
        description="Khmer Digits OCR Inference Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--image', '-i',
        type=str,
        help='Path to a single image file'
    )
    input_group.add_argument(
        '--directory', '-d',
        type=str,
        help='Path to directory containing images'
    )
    input_group.add_argument(
        '--generate', '-g',
        action='store_true',
        help='Generate sample images for testing'
    )
    
    # Model options
    parser.add_argument(
        '--checkpoint', '-c',
        type=str,
        default='training_output/checkpoints/conservative_small.pth',
        help='Path to model checkpoint (default: training_output/checkpoints/conservative_small.pth)'
    )
    parser.add_argument(
        '--config',
        type=str,
        help='Path to model configuration file (optional)'
    )
    parser.add_argument(
        '--device',
        type=str,
        choices=['cpu', 'cuda', 'auto'],
        default='auto',
        help='Device to run inference on (default: auto)'
    )
    parser.add_argument(
        '--model_preset',
        type=str,
        default='small',
        choices=['small', 'medium', 'large'],
        help='Model preset to use if config not provided (default: small)'
    )
    
    # Processing options
    parser.add_argument(
        '--batch_size', '-b',
        type=int,
        default=8,
        help='Batch size for processing multiple images (default: 8)'
    )
    parser.add_argument(
        '--confidence',
        action='store_true',
        help='Show confidence scores with predictions'
    )
    
    # Output options
    parser.add_argument(
        '--output_dir', '-o',
        type=str,
        default='inference_output',
        help='Output directory for results (default: inference_output)'
    )
    parser.add_argument(
        '--visualize', '-v',
        action='store_true',
        help='Create visualization images with predictions'
    )
    parser.add_argument(
        '--save_results',
        action='store_true',
        default=True,
        help='Save results to JSON file (default: True)'
    )
    
    # Generation options (when using --generate)
    parser.add_argument(
        '--num_samples',
        type=int,
        default=10,
        help='Number of sample images to generate for testing (default: 10)'
    )
    
    # Other options
    parser.add_argument(
        '--verbose', '-V',
        action='store_true',
        help='Enable verbose logging'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress output except errors'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    if args.quiet:
        log_level = logging.ERROR
    elif args.verbose:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO
    
    setup_logging(log_level)
    logger = logging.getLogger(__name__)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Check if checkpoint exists
        if not os.path.exists(args.checkpoint):
            logger.error(f"Checkpoint file not found: {args.checkpoint}")
            sys.exit(1)
        
        # Initialize inference engine
        logger.info("Initializing inference engine...")
        device = None if args.device == 'auto' else args.device
        
        inference_engine = KhmerOCRInference(
            checkpoint_path=args.checkpoint,
            config_path=args.config,
            device=device,
            model_preset=args.model_preset
        )
        
        # Print model information
        model_info = inference_engine.get_model_info()
        logger.info("Model Information:")
        logger.info(f"  - Model: {model_info.get('model_name', 'N/A')}")
        logger.info(f"  - Parameters: {model_info.get('total_parameters', 'N/A'):,}")
        logger.info(f"  - Device: {model_info.get('device', 'N/A')}")
        if 'epoch' in model_info:
            logger.info(f"  - Trained Epochs: {model_info['epoch']}")
            logger.info(f"  - Val Char Accuracy: {model_info.get('val_char_accuracy', 'N/A')}")
            logger.info(f"  - Val Seq Accuracy: {model_info.get('val_seq_accuracy', 'N/A')}")
        
        if args.generate:
            # Generate sample images for testing
            logger.info(f"Generating {args.num_samples} sample images...")
            
            # Create generator with proper config
            generator = SyntheticDataGenerator(
                config_path="config/model_config.yaml",
                fonts_dir="src/fonts",
                output_dir=str(output_dir / 'generated_samples')
            )
            sample_dir = output_dir / 'generated_samples'
            sample_dir.mkdir(exist_ok=True)
            
            generated_files = []
            for i in range(args.num_samples):
                # Generate random number sequence using Khmer digits
                import random
                sequence_length = random.randint(1, 6)
                khmer_digits = ['០', '១', '២', '៣', '៤', '៥', '៦', '៧', '៨', '៩']
                sequence = ''.join([random.choice(khmer_digits) for _ in range(sequence_length)])
                
                # Generate image
                image, metadata = generator.generate_single_image(
                    text=sequence,
                    apply_augmentation=False
                )
                
                # Save image
                filename = f"sample_{i:03d}_{sequence}.png"
                filepath = sample_dir / filename
                image.save(filepath)
                generated_files.append(str(filepath))
                
                if not args.quiet:
                    print(f"Generated: {filename} (sequence: {sequence})")
            
            # Process generated images
            logger.info("Processing generated images...")
            predictions = inference_engine.predict_batch(
                generated_files, 
                batch_size=args.batch_size,
                return_confidence=args.confidence
            )
            
            # Print results
            print("\n" + "="*60)
            print("GENERATED SAMPLES INFERENCE RESULTS")
            print("="*60)
            
            correct = 0
            total = len(generated_files)
            
            for filepath, result in zip(generated_files, predictions):
                filename = os.path.basename(filepath)
                # Extract true sequence from filename
                true_sequence = filename.split('_')[-1].replace('.png', '')
                
                if args.confidence:
                    predicted_text, confidence = result
                    is_correct = predicted_text == true_sequence
                    if is_correct:
                        correct += 1
                    
                    status = "✓" if is_correct else "✗"
                    print(f"{status} {filename}")
                    print(f"   True: {true_sequence}")
                    print(f"   Pred: {predicted_text} (conf: {confidence:.3f})")
                else:
                    predicted_text = result
                    is_correct = predicted_text == true_sequence
                    if is_correct:
                        correct += 1
                    
                    status = "✓" if is_correct else "✗"
                    print(f"{status} {filename}: {true_sequence} → {predicted_text}")
            
            accuracy = correct / total if total > 0 else 0
            print(f"\nAccuracy: {correct}/{total} ({accuracy:.1%})")
            
        elif args.image:
            # Single image inference
            logger.info(f"Processing single image: {args.image}")
            
            if not os.path.exists(args.image):
                logger.error(f"Image file not found: {args.image}")
                sys.exit(1)
            
            # Predict
            if args.confidence:
                prediction, confidence = inference_engine.predict_single(
                    args.image, return_confidence=True
                )
                print(f"Prediction: {prediction}")
                print(f"Confidence: {confidence:.3f}")
            else:
                prediction = inference_engine.predict_single(args.image)
                print(f"Prediction: {prediction}")
            
            # Visualize if requested
            if args.visualize:
                vis_path = output_dir / f"visualization_{Path(args.image).stem}.png"
                pred_text, vis_image = inference_engine.visualize_prediction(
                    args.image, save_path=str(vis_path)
                )
                logger.info(f"Visualization saved to: {vis_path}")
            
        elif args.directory:
            # Directory inference
            logger.info(f"Processing images from directory: {args.directory}")
            
            if not os.path.exists(args.directory):
                logger.error(f"Directory not found: {args.directory}")
                sys.exit(1)
            
            # Process directory
            results_file = output_dir / 'results.json' if args.save_results else None
            
            predictions = inference_engine.predict_from_directory(
                args.directory,
                batch_size=args.batch_size,
                save_results=args.save_results,
                output_path=str(results_file) if results_file else None
            )
            
            # Print results
            print(f"\nProcessed {len(predictions)} images:")
            for filename, prediction in predictions.items():
                print(f"  {filename}: {prediction}")
            
            if args.save_results and results_file:
                logger.info(f"Detailed results saved to: {results_file}")
        
        logger.info("Inference completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Inference interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Inference failed: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 