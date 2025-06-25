"""
Main synthetic data generator for Khmer OCR (digits and full text).
"""

import os
import yaml
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from .utils import (
    load_khmer_fonts, validate_font_collection, generate_digit_sequence,
    normalize_khmer_text, create_character_mapping, get_full_khmer_characters,
    load_character_frequencies, generate_weighted_character_sequence,
    generate_khmer_syllable, generate_khmer_word, generate_khmer_phrase,
    generate_mixed_content, load_khmer_corpus, generate_corpus_based_text,
    segment_corpus_text
)
from .backgrounds import BackgroundGenerator
from .augmentation import ImageAugmentor


class SyntheticDataGenerator:
    """
    Generates synthetic training data for Khmer OCR (digits and full text).
    """
    
    def __init__(self, 
                 config_path: str,
                 fonts_dir: str,
                 output_dir: str,
                 mode: str = "full_text",
                 use_corpus: bool = True):
        """
        Initialize the synthetic data generator.
        
        Args:
            config_path: Path to model configuration file
            fonts_dir: Directory containing Khmer fonts
            output_dir: Directory to save generated data
            mode: Generation mode ('digits', 'full_text', 'mixed')
            use_corpus: Whether to use real corpus text for generation
        """
        self.config_path = config_path
        self.fonts_dir = fonts_dir
        self.output_dir = output_dir
        self.mode = mode
        self.use_corpus = use_corpus
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize components
        self.image_size = tuple(self.config['model']['input']['image_size'])
        self.max_sequence_length = self.config['model']['characters'].get('max_sequence_length', 20)
        
        self.background_generator = BackgroundGenerator(self.image_size)
        self.augmentor = ImageAugmentor(
            rotation_range=tuple(self.config['data']['augmentation']['rotation']),
            scale_range=tuple(self.config['data']['augmentation']['scaling']),
            noise_std=self.config['data']['augmentation']['noise']['gaussian_std'],
            brightness_range=tuple(self.config['data']['augmentation']['brightness']),
            contrast_range=tuple(self.config['data']['augmentation']['contrast'])
        )
        
        # Load and validate fonts
        self.fonts = load_khmer_fonts(fonts_dir)
        self.font_validation = validate_font_collection(fonts_dir)
        
        # Filter to only working fonts
        self.working_fonts = {
            name: path for name, path in self.fonts.items() 
            if self.font_validation[name]
        }
        
        if not self.working_fonts:
            raise ValueError("No working Khmer fonts found!")
        
        print(f"Loaded {len(self.working_fonts)} working fonts: {list(self.working_fonts.keys())}")
        
        # Create character mappings
        use_full_khmer = (mode in ["full_text", "mixed"])
        self.char_to_idx, self.idx_to_char = create_character_mapping(use_full_khmer)
        
        # Load character frequencies for realistic text generation
        self.character_frequencies = load_character_frequencies()
        self.khmer_characters = get_full_khmer_characters()
        
        # Load corpus if using corpus-based generation
        self.corpus_lines = None
        if self.use_corpus and mode in ["full_text", "mixed"]:
            self.corpus_lines = load_khmer_corpus()
            if self.corpus_lines:
                print(f"✅ Corpus loaded: {len(self.corpus_lines)} lines for authentic text generation")
            else:
                print("⚠️ Corpus not available, using synthetic generation only")
                self.use_corpus = False
        
        print(f"Character vocabulary size: {len(self.char_to_idx)}")
        print(f"Generation mode: {mode}")
        print(f"Corpus usage: {'Enabled' if self.use_corpus else 'Disabled'}")
        print(f"Loaded frequencies for {len(self.character_frequencies)} characters")
        
        # Create output directory
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
    
    def _load_config(self) -> Dict:
        """Load configuration from YAML file."""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _generate_text_content(self, 
                              content_type: str = "auto",
                              length_range: Tuple[int, int] = (1, 15),
                              allowed_characters: Optional[List[str]] = None) -> str:
        """
        Generate text content based on mode and type.
        
        Args:
            content_type: Type of content to generate
            length_range: Range of text lengths
            allowed_characters: Allowed characters for curriculum learning
            
        Returns:
            Generated text content
        """
        target_length = random.randint(*length_range)
        
        if self.mode == "digits":
            return generate_digit_sequence(1, min(8, target_length))
        
        # Use corpus-based generation if available and appropriate
        if self.use_corpus and self.corpus_lines and content_type in ["auto", "words", "phrases", "mixed"]:
            corpus_text = generate_corpus_based_text(
                corpus_lines=self.corpus_lines,
                target_length=target_length,
                content_type=content_type,
                allowed_characters=allowed_characters
            )
            
            # Validate the corpus text meets our requirements
            if corpus_text and len(corpus_text) >= 1:
                return corpus_text
        
        # Fallback to synthetic generation
        if content_type == "auto":
            # Intelligent content type selection based on length
            if target_length <= 3:
                content_type = "characters"
            elif target_length <= 8:
                content_type = "syllables"  
            elif target_length <= 15:
                content_type = "words"
            else:
                content_type = "mixed"
        
        if content_type == "digits":
            return generate_digit_sequence(1, min(8, target_length))
        elif content_type == "characters":
            return generate_weighted_character_sequence(
                target_length, 
                character_frequencies={char: self.character_frequencies.get(char, 0.001) 
                                     for char in allowed_characters} if allowed_characters else self.character_frequencies,
                character_set=allowed_characters
            )
        elif content_type == "syllables":
            num_syllables = max(1, target_length // 3)
            return "".join([generate_khmer_syllable() for _ in range(num_syllables)])
        elif content_type == "words":
            return generate_khmer_word(1, max(1, target_length // 4))
        elif content_type == "phrases":
            return generate_khmer_phrase(1, max(1, target_length // 8))
        else:  # mixed
            return generate_mixed_content(target_length, "mixed")
    
    def _get_adaptive_font_size(self, text: str) -> int:
        """Get a font size that fits the text within image boundaries."""
        # Start with a reasonable base size
        base_size = int(self.image_size[1] * 0.6)
        
        # Adjust based on text length - longer sequences need smaller fonts
        length_factor = max(0.3, 1.0 - (len(text) - 1) * 0.05)
        target_size = int(base_size * length_factor)
        
        # Test different font sizes to ensure text fits
        for font_size in range(target_size, 12, -2):  # Minimum font size of 12
            # Test with a representative font
            test_font_path = list(self.working_fonts.values())[0]
            test_font = ImageFont.truetype(test_font_path, font_size)
            
            bbox = self._get_text_bbox(text, test_font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # Check if text fits with some margin (leave 20% margin on each side)
            margin_width = self.image_size[0] * 0.1
            margin_height = self.image_size[1] * 0.1
            
            if (text_width <= self.image_size[0] - 2 * margin_width and 
                text_height <= self.image_size[1] - 2 * margin_height):
                return font_size
        
        # Fallback to minimum size
        return 12
    
    def _get_text_bbox(self, text: str, font: ImageFont.FreeTypeFont) -> Tuple[int, int, int, int]:
        """Get tight bounding box for text."""
        # Create temporary image to measure text
        temp_img = Image.new('RGB', (1000, 1000))
        temp_draw = ImageDraw.Draw(temp_img)
        bbox = temp_draw.textbbox((0, 0), text, font=font)
        return bbox
    
    def _safe_position_text(self, text: str, font: ImageFont.FreeTypeFont, 
                           image_size: Tuple[int, int]) -> Tuple[int, int]:
        """Calculate safe position for text ensuring it fits within image bounds."""
        bbox = self._get_text_bbox(text, font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Define safe margins (10% of image size)
        margin_x = int(image_size[0] * 0.1)
        margin_y = int(image_size[1] * 0.1)
        
        # Calculate available space for text placement
        available_width = image_size[0] - 2 * margin_x
        available_height = image_size[1] - 2 * margin_y
        
        # Center text within available space
        x = margin_x + (available_width - text_width) // 2
        y = margin_y + (available_height - text_height) // 2
        
        # Adjust for bbox offset
        x -= bbox[0]
        y -= bbox[1]
        
        # Ensure text doesn't go outside bounds
        x = max(margin_x - bbox[0], min(x, image_size[0] - margin_x - text_width - bbox[0]))
        y = max(margin_y - bbox[1], min(y, image_size[1] - margin_y - text_height - bbox[1]))
        
        return x, y
    
    def _validate_text_fits(self, text: str, font: ImageFont.FreeTypeFont, 
                           position: Tuple[int, int], image_size: Tuple[int, int]) -> bool:
        """Validate that text fits completely within image boundaries."""
        x, y = position
        bbox = self._get_text_bbox(text, font)
        
        # Calculate actual text boundaries in the image
        text_left = x + bbox[0]
        text_top = y + bbox[1]
        text_right = x + bbox[2]
        text_bottom = y + bbox[3]
        
        # Check if text is within image bounds
        return (text_left >= 0 and text_top >= 0 and 
                text_right <= image_size[0] and text_bottom <= image_size[1])
    
    def generate_single_image(self, 
                             text: Optional[str] = None,
                             font_name: Optional[str] = None,
                             content_type: str = "auto",
                             apply_augmentation: bool = True) -> Tuple[Image.Image, Dict]:
        """
        Generate a single synthetic image with text.
        
        Args:
            text: Text to render, if None generates based on mode
            font_name: Font to use, if None chooses random font
            content_type: Type of content to generate ('auto', 'digits', 'characters', 'syllables', 'words', 'phrases', 'mixed')
            apply_augmentation: Whether to apply augmentation
            
        Returns:
            Tuple of (image, metadata)
        """
        # Generate text if not provided
        if text is None:
            text = self._generate_text_content(content_type)
        text = normalize_khmer_text(text)
        
        # Choose font
        if font_name is None:
            font_name = random.choice(list(self.working_fonts.keys()))
        font_path = self.working_fonts[font_name]
        
        # Generate background
        background = self.background_generator.generate_random_background()
        
        # Get optimal text color for this background
        text_color = self.background_generator.get_optimal_text_color(background)
        
        # Create font object with adaptive size
        font_size = self._get_adaptive_font_size(text)
        font = ImageFont.truetype(font_path, font_size)
        
        # Calculate safe text position
        text_position = self._safe_position_text(text, font, self.image_size)
        
        # Create image with text
        image = background.copy()
        draw = ImageDraw.Draw(image)
        
        # Draw text
        draw.text(text_position, text, font=font, fill=text_color)
        
        # Validate text positioning
        if not self._validate_text_fits(text, font, text_position, self.image_size):
            print(f"Warning: Text may be cropped - '{text}' with font size {font_size}")
        
        # Apply augmentation if requested
        augmentation_params = {}
        if apply_augmentation:
            image, augmentation_params = self.augmentor.apply_random_augmentation(image)
        
        # Create metadata
        metadata = {
            'label': text,
            'font': font_name,
            'font_size': font_size,
            'text_position': text_position,
            'text_color': text_color,
            'image_size': self.image_size,
            'character_count': len(text),
            'content_type': content_type if content_type != "auto" else self._classify_content_type(text),
            'augmentation': augmentation_params
        }
        
        return image, metadata
    
    def _classify_content_type(self, text: str) -> str:
        """Classify the type of generated content."""
        digits = set("០១២៣៤៥៦៧៨៩")
        khmer_chars = set()
        for category_chars in self.khmer_characters.values():
            khmer_chars.update(category_chars)
        
        text_chars = set(text)
        
        if text_chars.issubset(digits):
            return "digits"
        elif len(text) <= 3:
            return "characters"
        elif len(text) <= 8:
            return "syllables"
        elif len(text) <= 15:
            return "words"
        else:
            return "phrases"
    
    def generate_dataset(self, 
                        num_samples: int,
                        train_split: float = 0.8,
                        save_images: bool = True,
                        show_progress: bool = True) -> Dict:
        """
        Generate a complete dataset.
        
        Args:
            num_samples: Total number of samples to generate
            train_split: Fraction of samples for training
            save_images: Whether to save images to disk
            show_progress: Whether to show progress bar
            
        Returns:
            Dictionary with dataset information
        """
        # Calculate splits
        num_train = int(num_samples * train_split)
        num_val = num_samples - num_train
        
        # Create directories
        train_dir = Path(self.output_dir) / 'train'
        val_dir = Path(self.output_dir) / 'val'
        
        if save_images:
            train_dir.mkdir(exist_ok=True)
            val_dir.mkdir(exist_ok=True)
        
        # Generate samples
        all_metadata = {
            'train': {'samples': []},
            'val': {'samples': []}
        }
        
        # Progress bar setup
        total_progress = tqdm(total=num_samples, desc="Generating dataset") if show_progress else None
        
        # Generate training samples
        for i in range(num_train):
            image, metadata = self.generate_single_image()
            
            if save_images:
                image_filename = f"train_{i:06d}.png"
                image_path = train_dir / image_filename
                image.save(image_path)
                metadata['image_path'] = str(image_path)
                metadata['image_filename'] = image_filename
            
            all_metadata['train']['samples'].append(metadata)
            
            if total_progress:
                total_progress.update(1)
        
        # Generate validation samples
        for i in range(num_val):
            image, metadata = self.generate_single_image()
            
            if save_images:
                image_filename = f"val_{i:06d}.png"
                image_path = val_dir / image_filename
                image.save(image_path)
                metadata['image_path'] = str(image_path)
                metadata['image_filename'] = image_filename
            
            all_metadata['val']['samples'].append(metadata)
            
            if total_progress:
                total_progress.update(1)
        
        if total_progress:
            total_progress.close()
        
        # Add dataset-level metadata
        dataset_info = {
            'total_samples': num_samples,
            'train_samples': num_train,
            'val_samples': num_val,
            'image_size': list(self.image_size),  # Convert tuple to list for YAML compatibility
            'max_sequence_length': self.max_sequence_length,
            'fonts_used': list(self.working_fonts.keys()),
            'character_set': list(self.char_to_idx.keys()),
            'generated_by': 'SyntheticDataGenerator v1.0'
        }
        
        all_metadata['dataset_info'] = dataset_info
        
        # Save metadata
        if save_images:
            metadata_path = Path(self.output_dir) / 'metadata.yaml'
            with open(metadata_path, 'w', encoding='utf-8') as f:
                yaml.dump(all_metadata, f, default_flow_style=False, allow_unicode=True)
        
        print(f"\nDataset generated successfully!")
        print(f"Total samples: {num_samples}")
        print(f"Training samples: {num_train}")
        print(f"Validation samples: {num_val}")
        print(f"Fonts used: {len(self.working_fonts)}")
        
        return all_metadata
    
    def generate_samples_by_length(self, 
                                  samples_per_length: int = 100,
                                  save_images: bool = True) -> Dict:
        """
        Generate samples with balanced sequence lengths.
        
        Args:
            samples_per_length: Number of samples per sequence length
            save_images: Whether to save images to disk
            
        Returns:
            Dictionary with dataset information
        """
        all_metadata = {'samples': []}
        
        # Create output directory
        if save_images:
            output_dir = Path(self.output_dir) / 'balanced'
            output_dir.mkdir(exist_ok=True)
        
        sample_count = 0
        
        for length in range(1, self.max_sequence_length + 1):
            print(f"Generating {samples_per_length} samples with {length} digit(s)...")
            
            for i in tqdm(range(samples_per_length), desc=f"Length {length}"):
                # Generate text with specific length
                text = generate_digit_sequence(length, length)
                image, metadata = self.generate_single_image(text=text)
                
                if save_images:
                    image_filename = f"sample_{sample_count:06d}_len{length}.png"
                    image_path = output_dir / image_filename
                    image.save(image_path)
                    metadata['image_path'] = str(image_path)
                    metadata['image_filename'] = image_filename
                
                all_metadata['samples'].append(metadata)
                sample_count += 1
        
        # Add dataset info
        dataset_info = {
            'total_samples': sample_count,
            'samples_per_length': samples_per_length,
            'sequence_lengths': list(range(1, self.max_sequence_length + 1)),
            'image_size': self.image_size,
            'fonts_used': list(self.working_fonts.keys()),
            'character_set': list(self.char_to_idx.keys()),
            'generated_by': 'SyntheticDataGenerator v1.0 (balanced)'
        }
        
        all_metadata['dataset_info'] = dataset_info
        
        # Save metadata
        if save_images:
            metadata_path = Path(self.output_dir) / 'balanced' / 'metadata.yaml'
            with open(metadata_path, 'w', encoding='utf-8') as f:
                yaml.dump(all_metadata, f, default_flow_style=False, allow_unicode=True)
        
        print(f"\nBalanced dataset generated successfully!")
        print(f"Total samples: {sample_count}")
        print(f"Samples per length: {samples_per_length}")
        
        return all_metadata
    
    def preview_samples(self, num_samples: int = 10) -> List[Tuple[Image.Image, str]]:
        """
        Generate preview samples for visual inspection.
        
        Args:
            num_samples: Number of preview samples
            
        Returns:
            List of (image, label) tuples
        """
        samples = []
        
        for _ in range(num_samples):
            image, metadata = self.generate_single_image()
            samples.append((image, metadata['label']))
        
        return samples
    
    def generate_curriculum_dataset(self,
                                  stage: str = "stage1",
                                  num_samples: int = 1000,
                                  train_split: float = 0.8,
                                  save_images: bool = True,
                                  show_progress: bool = True) -> Dict:
        """
        Generate dataset for curriculum learning stages.
        
        Args:
            stage: Curriculum stage ('stage1', 'stage2', 'stage3', 'mixed')
            num_samples: Total number of samples to generate
            train_split: Fraction of samples for training
            save_images: Whether to save images to disk
            show_progress: Whether to show progress bar
            
        Returns:
            Dictionary with dataset information
        """
        # Define curriculum stages based on character frequency analysis
        curriculum_stages = {
            'stage1': {
                'description': 'High-frequency characters (top 30)',
                'content_types': ['characters', 'syllables'],
                'content_weights': [0.6, 0.4],
                'length_range': (1, 5),
                'character_limit': 30  # Top 30 most frequent characters
            },
            'stage2': {
                'description': 'Medium-frequency characters (top 60)',
                'content_types': ['characters', 'syllables', 'words'],
                'content_weights': [0.4, 0.4, 0.2],
                'length_range': (1, 10),
                'character_limit': 60  # Top 60 most frequent characters
            },
            'stage3': {
                'description': 'All characters with complex structures',
                'content_types': ['syllables', 'words', 'phrases'],
                'content_weights': [0.3, 0.5, 0.2],
                'length_range': (1, 20),
                'character_limit': None  # All characters
            },
            'mixed': {
                'description': 'Mixed content including digits',
                'content_types': ['digits', 'characters', 'syllables', 'words'],
                'content_weights': [0.2, 0.3, 0.3, 0.2],
                'length_range': (1, 15),
                'character_limit': None
            }
        }
        
        if stage not in curriculum_stages:
            raise ValueError(f"Unknown curriculum stage: {stage}. Available: {list(curriculum_stages.keys())}")
        
        stage_config = curriculum_stages[stage]
        print(f"Generating curriculum dataset - {stage}: {stage_config['description']}")
        
        # Get character subset for this stage
        character_subset = self._get_character_subset(stage_config['character_limit'])
        
        # Calculate splits
        num_train = int(num_samples * train_split)
        num_val = num_samples - num_train
        
        # Create directories
        stage_dir = Path(self.output_dir) / f'curriculum_{stage}'
        train_dir = stage_dir / 'train'
        val_dir = stage_dir / 'val'
        
        if save_images:
            train_dir.mkdir(parents=True, exist_ok=True)
            val_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate samples
        all_metadata = {
            'train': {'samples': []},
            'val': {'samples': []},
            'curriculum_stage': stage,
            'stage_config': stage_config
        }
        
        # Progress bar setup
        total_progress = tqdm(total=num_samples, desc=f"Generating {stage} dataset") if show_progress else None
        
        # Generate training samples
        for i in range(num_train):
            content_type = np.random.choice(
                stage_config['content_types'], 
                p=stage_config['content_weights']
            )
            
            image, metadata = self.generate_single_image(
                content_type=content_type,
                apply_augmentation=True
            )
            
            # Filter content to stage character subset if applicable
            if character_subset and not self._text_uses_allowed_characters(metadata['label'], character_subset):
                # Regenerate with allowed characters
                text = self._generate_stage_appropriate_text(content_type, stage_config, character_subset)
                image, metadata = self.generate_single_image(
                    text=text,
                    content_type=content_type,
                    apply_augmentation=True
                )
            
            if save_images:
                image_filename = f"train_{i:06d}.png"
                image_path = train_dir / image_filename
                image.save(image_path)
                metadata['image_path'] = str(image_path)
                metadata['image_filename'] = image_filename
            
            metadata['curriculum_stage'] = stage
            all_metadata['train']['samples'].append(metadata)
            
            if total_progress:
                total_progress.update(1)
        
        # Generate validation samples
        for i in range(num_val):
            content_type = np.random.choice(
                stage_config['content_types'], 
                p=stage_config['content_weights']
            )
            
            image, metadata = self.generate_single_image(
                content_type=content_type,
                apply_augmentation=False  # No augmentation for validation
            )
            
            # Filter content to stage character subset if applicable
            if character_subset and not self._text_uses_allowed_characters(metadata['label'], character_subset):
                # Regenerate with allowed characters
                text = self._generate_stage_appropriate_text(content_type, stage_config, character_subset)
                image, metadata = self.generate_single_image(
                    text=text,
                    content_type=content_type,
                    apply_augmentation=False
                )
            
            if save_images:
                image_filename = f"val_{i:06d}.png"
                image_path = val_dir / image_filename
                image.save(image_path)
                metadata['image_path'] = str(image_path)
                metadata['image_filename'] = image_filename
            
            metadata['curriculum_stage'] = stage
            all_metadata['val']['samples'].append(metadata)
            
            if total_progress:
                total_progress.update(1)
        
        if total_progress:
            total_progress.close()
        
        # Add dataset-level metadata
        dataset_info = {
            'curriculum_stage': stage,
            'stage_description': stage_config['description'],
            'total_samples': num_samples,
            'train_samples': num_train,
            'val_samples': num_val,
            'image_size': list(self.image_size),
            'content_types': stage_config['content_types'],
            'length_range': stage_config['length_range'],
            'character_subset_size': len(character_subset) if character_subset else len(self.character_frequencies),
            'fonts_used': list(self.working_fonts.keys()),
            'generated_by': f'SyntheticDataGenerator v2.0 - Curriculum Learning'
        }
        
        all_metadata['dataset_info'] = dataset_info
        
        # Save metadata
        if save_images:
            metadata_path = stage_dir / 'metadata.yaml'
            with open(metadata_path, 'w', encoding='utf-8') as f:
                yaml.dump(all_metadata, f, default_flow_style=False, allow_unicode=True)
        
        print(f"\nCurriculum dataset ({stage}) generated successfully!")
        print(f"Total samples: {num_samples}")
        print(f"Training samples: {num_train}")
        print(f"Validation samples: {num_val}")
        print(f"Character subset size: {len(character_subset) if character_subset else 'All characters'}")
        
        return all_metadata
    
    def _get_character_subset(self, limit: Optional[int]) -> Optional[List[str]]:
        """Get top N characters by frequency."""
        if limit is None:
            return None
        
        # Sort characters by frequency
        sorted_chars = sorted(
            self.character_frequencies.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Return top N characters
        return [char for char, freq in sorted_chars[:limit]]
    
    def _text_uses_allowed_characters(self, text: str, allowed_chars: List[str]) -> bool:
        """Check if text only uses allowed characters."""
        allowed_set = set(allowed_chars)
        text_chars = set(text)
        return text_chars.issubset(allowed_set)
    
    def _generate_stage_appropriate_text(self, 
                                       content_type: str, 
                                       stage_config: Dict, 
                                       character_subset: List[str]) -> str:
        """Generate text appropriate for curriculum stage."""
        length_range = stage_config['length_range']
        
        # Use the enhanced text generation method
        return self._generate_text_content(
            content_type=content_type,
            length_range=length_range,
            allowed_characters=character_subset
        )
    
    def generate_frequency_balanced_dataset(self,
                                          num_samples: int = 1000,
                                          balance_factor: float = 0.5,
                                          train_split: float = 0.8,
                                          save_images: bool = True) -> Dict:
        """
        Generate dataset with frequency-balanced character distribution.
        
        Args:
            num_samples: Total number of samples
            balance_factor: 0.0 = pure frequency weighting, 1.0 = uniform distribution
            train_split: Fraction for training
            save_images: Whether to save images
            
        Returns:
            Dataset metadata
        """
        print(f"Generating frequency-balanced dataset (balance_factor: {balance_factor})")
        
        # Create balanced character frequencies
        balanced_frequencies = {}
        uniform_weight = 1.0 / len(self.character_frequencies)
        
        for char, freq in self.character_frequencies.items():
            # Interpolate between original frequency and uniform distribution
            balanced_freq = (1 - balance_factor) * freq + balance_factor * uniform_weight
            balanced_frequencies[char] = balanced_freq
        
        # Temporarily replace character frequencies
        original_frequencies = self.character_frequencies
        self.character_frequencies = balanced_frequencies
        
        try:
            # Generate dataset with balanced frequencies
            dataset = self.generate_dataset(
                num_samples=num_samples,
                train_split=train_split,
                save_images=save_images
            )
            
            # Add balance information to metadata
            dataset['dataset_info']['balance_factor'] = balance_factor
            dataset['dataset_info']['generation_type'] = 'frequency_balanced'
            
            return dataset
            
        finally:
            # Restore original frequencies
            self.character_frequencies = original_frequencies
    
    def generate_mixed_complexity_dataset(self,
                                        num_samples: int = 1000,
                                        train_split: float = 0.8,
                                        save_images: bool = True) -> Dict:
        """
        Generate dataset with mixed complexity levels.
        
        Args:
            num_samples: Total number of samples
            train_split: Fraction for training
            save_images: Whether to save images
            
        Returns:
            Dataset metadata
        """
        print("Generating mixed complexity dataset")
        
        # Define complexity distribution
        complexity_levels = {
            'simple': {'weight': 0.4, 'content_types': ['digits', 'characters'], 'length_range': (1, 3)},
            'medium': {'weight': 0.4, 'content_types': ['syllables', 'words'], 'length_range': (4, 10)},
            'complex': {'weight': 0.2, 'content_types': ['words', 'phrases'], 'length_range': (11, 20)}
        }
        
        # Calculate splits
        num_train = int(num_samples * train_split)
        num_val = num_samples - num_train
        
        # Create directories
        output_dir = Path(self.output_dir) / 'mixed_complexity'
        train_dir = output_dir / 'train'
        val_dir = output_dir / 'val'
        
        if save_images:
            train_dir.mkdir(parents=True, exist_ok=True)
            val_dir.mkdir(parents=True, exist_ok=True)
        
        all_metadata = {
            'train': {'samples': []},
            'val': {'samples': []},
            'complexity_levels': complexity_levels
        }
        
        # Generate samples with complexity distribution
        for split_name, num_split, split_dir in [('train', num_train, train_dir), ('val', num_val, val_dir)]:
            for i in tqdm(range(num_split), desc=f"Generating {split_name} samples"):
                # Choose complexity level
                complexity = np.random.choice(
                    list(complexity_levels.keys()),
                    p=[level['weight'] for level in complexity_levels.values()]
                )
                
                level_config = complexity_levels[complexity]
                content_type = random.choice(level_config['content_types'])
                
                # Generate with specific complexity
                image, metadata = self.generate_single_image(
                    content_type=content_type,
                    apply_augmentation=(split_name == 'train')
                )
                
                if save_images:
                    image_filename = f"{split_name}_{i:06d}.png"
                    image_path = split_dir / image_filename
                    image.save(image_path)
                    metadata['image_path'] = str(image_path)
                    metadata['image_filename'] = image_filename
                
                metadata['complexity_level'] = complexity
                all_metadata[split_name]['samples'].append(metadata)
        
        # Add dataset info
        dataset_info = {
            'generation_type': 'mixed_complexity',
            'total_samples': num_samples,
            'train_samples': num_train,
            'val_samples': num_val,
            'complexity_distribution': complexity_levels,
            'image_size': list(self.image_size),
            'fonts_used': list(self.working_fonts.keys()),
            'generated_by': 'SyntheticDataGenerator v2.0 - Mixed Complexity'
        }
        
        all_metadata['dataset_info'] = dataset_info
        
        # Save metadata
        if save_images:
            metadata_path = output_dir / 'metadata.yaml'
            with open(metadata_path, 'w', encoding='utf-8') as f:
                yaml.dump(all_metadata, f, default_flow_style=False, allow_unicode=True)
        
        print(f"Mixed complexity dataset generated: {num_samples} samples")
        return all_metadata 