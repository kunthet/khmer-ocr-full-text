"""
Image preprocessing and transformation utilities for Khmer digits OCR.
"""

import torch
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
from PIL import Image, ImageFilter, ImageEnhance
import random
import numpy as np
from typing import Tuple, Optional, List, Union


class ImagePreprocessor:
    """
    Image preprocessing pipeline for Khmer digits OCR.
    
    Provides consistent preprocessing for training and inference,
    including normalization, resizing, and format conversion.
    """
    
    def __init__(self,
                 image_size: Tuple[int, int] = (128, 64),
                 normalize: bool = True,
                 mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
                 std: Tuple[float, float, float] = (0.229, 0.224, 0.225)):
        """
        Initialize the image preprocessor.
        
        Args:
            image_size: Target image size as (width, height)
            normalize: Whether to normalize images
            mean: Normalization mean values for RGB channels
            std: Normalization std values for RGB channels
        """
        self.image_size = image_size
        self.normalize = normalize
        self.mean = mean
        self.std = std
    
    def get_base_transforms(self) -> transforms.Compose:
        """Get basic preprocessing transforms."""
        transform_list = [
            transforms.Resize(self.image_size),
            transforms.ToTensor()
        ]
        
        if self.normalize:
            transform_list.append(
                transforms.Normalize(mean=self.mean, std=self.std)
            )
        
        return transforms.Compose(transform_list)
    
    def get_train_transforms(self, 
                           augmentation_strength: float = 0.3) -> transforms.Compose:
        """
        Get training transforms with augmentation.
        
        Args:
            augmentation_strength: Strength of augmentation (0.0 to 1.0)
            
        Returns:
            Composed transforms for training
        """
        transform_list = [
            transforms.Resize(self.image_size),
            
            # Light augmentations to preserve text readability
            transforms.RandomApply([
                transforms.ColorJitter(
                    brightness=0.1 * augmentation_strength,
                    contrast=0.1 * augmentation_strength,
                    saturation=0.05 * augmentation_strength,
                    hue=0.02 * augmentation_strength
                )
            ], p=0.5),
            
            # Slight rotation
            transforms.RandomApply([
                transforms.RandomRotation(
                    degrees=3 * augmentation_strength,
                    fill=255  # White fill for rotated areas
                )
            ], p=0.3),
            
            # Random perspective for slight 3D effect
            transforms.RandomApply([
                transforms.RandomPerspective(
                    distortion_scale=0.1 * augmentation_strength,
                    p=0.5,
                    fill=255
                )
            ], p=0.2),
            
            transforms.ToTensor(),
            
            # Add slight noise
            AddGaussianNoise(
                mean=0.0, 
                std=0.01 * augmentation_strength
            ),
        ]
        
        if self.normalize:
            transform_list.append(
                transforms.Normalize(mean=self.mean, std=self.std)
            )
        
        return transforms.Compose(transform_list)
    
    def get_val_transforms(self) -> transforms.Compose:
        """Get validation transforms (no augmentation)."""
        return self.get_base_transforms()
    
    def preprocess_image(self, image: Union[Image.Image, np.ndarray]) -> torch.Tensor:
        """
        Preprocess a single image for inference.
        
        Args:
            image: Input image as PIL Image or numpy array
            
        Returns:
            Preprocessed image tensor
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        transform = self.get_base_transforms()
        return transform(image)
    
    def denormalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Denormalize a tensor for visualization.
        
        Args:
            tensor: Normalized image tensor
            
        Returns:
            Denormalized tensor
        """
        if not self.normalize:
            return tensor
        
        mean = torch.tensor(self.mean).view(3, 1, 1)
        std = torch.tensor(self.std).view(3, 1, 1)
        
        return tensor * std + mean


class AddGaussianNoise:
    """Add Gaussian noise to image tensors."""
    
    def __init__(self, mean: float = 0.0, std: float = 0.01):
        self.mean = mean
        self.std = std
    
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.std > 0:
            noise = torch.randn_like(tensor) * self.std + self.mean
            return torch.clamp(tensor + noise, 0.0, 1.0)
        return tensor


class RandomBlur:
    """Apply random Gaussian blur to images."""
    
    def __init__(self, radius_range: Tuple[float, float] = (0.1, 1.0), p: float = 0.3):
        self.radius_range = radius_range
        self.p = p
    
    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() < self.p:
            radius = random.uniform(*self.radius_range)
            return image.filter(ImageFilter.GaussianBlur(radius=radius))
        return image


class RandomBrightnessContrast:
    """Apply random brightness and contrast adjustments."""
    
    def __init__(self, 
                 brightness_range: Tuple[float, float] = (0.8, 1.2),
                 contrast_range: Tuple[float, float] = (0.8, 1.2),
                 p: float = 0.5):
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.p = p
    
    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() < self.p:
            # Apply brightness
            brightness_factor = random.uniform(*self.brightness_range)
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(brightness_factor)
            
            # Apply contrast
            contrast_factor = random.uniform(*self.contrast_range)
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(contrast_factor)
        
        return image


def get_train_transforms(image_size: Tuple[int, int] = (128, 64),
                        augmentation_strength: float = 0.3,
                        normalize: bool = True) -> transforms.Compose:
    """
    Get training transforms with configurable augmentation.
    
    Args:
        image_size: Target image size as (width, height)
        augmentation_strength: Strength of augmentation (0.0 to 1.0)
        normalize: Whether to apply ImageNet normalization
        
    Returns:
        Composed transforms for training
    """
    preprocessor = ImagePreprocessor(
        image_size=image_size,
        normalize=normalize
    )
    
    return preprocessor.get_train_transforms(augmentation_strength)


def get_val_transforms(image_size: Tuple[int, int] = (128, 64),
                      normalize: bool = True) -> transforms.Compose:
    """
    Get validation transforms (no augmentation).
    
    Args:
        image_size: Target image size as (width, height)
        normalize: Whether to apply ImageNet normalization
        
    Returns:
        Composed transforms for validation
    """
    preprocessor = ImagePreprocessor(
        image_size=image_size,
        normalize=normalize
    )
    
    return preprocessor.get_val_transforms()


def get_inference_transforms(image_size: Tuple[int, int] = (128, 64),
                           normalize: bool = True) -> transforms.Compose:
    """
    Get transforms for inference (same as validation).
    
    Args:
        image_size: Target image size as (width, height)
        normalize: Whether to apply ImageNet normalization
        
    Returns:
        Composed transforms for inference
    """
    return get_val_transforms(image_size, normalize)


class TestTimeAugmentation:
    """
    Test-time augmentation for improved inference accuracy.
    
    Applies multiple transformations to the same image and
    averages the predictions.
    """
    
    def __init__(self, 
                 base_transform: transforms.Compose,
                 num_augmentations: int = 5):
        """
        Initialize TTA.
        
        Args:
            base_transform: Base transform to apply
            num_augmentations: Number of augmented versions to create
        """
        self.base_transform = base_transform
        self.num_augmentations = num_augmentations
        
        # Light augmentations for TTA
        self.tta_transforms = [
            transforms.Compose([
                transforms.RandomRotation(degrees=2, fill=255),
                base_transform
            ]),
            transforms.Compose([
                transforms.ColorJitter(brightness=0.05, contrast=0.05),
                base_transform
            ]),
            transforms.Compose([
                transforms.RandomPerspective(distortion_scale=0.05, p=0.5, fill=255),
                base_transform
            ]),
            base_transform,  # Original
        ]
    
    def __call__(self, image: Image.Image) -> List[torch.Tensor]:
        """
        Apply TTA to an image.
        
        Args:
            image: Input PIL Image
            
        Returns:
            List of augmented image tensors
        """
        augmented_images = []
        
        # Apply base transform
        augmented_images.append(self.base_transform(image))
        
        # Apply TTA transforms
        for i in range(self.num_augmentations - 1):
            transform = random.choice(self.tta_transforms[:-1])  # Exclude original
            augmented_images.append(transform(image))
        
        return augmented_images


def create_preprocessing_pipeline(config: dict) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Create preprocessing pipeline from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (train_transform, val_transform)
    """
    # Extract config parameters
    image_size = tuple(config.get('image_size', [128, 64]))
    normalize = config.get('normalize', True)
    augmentation_strength = config.get('augmentation_strength', 0.3)
    
    # Create transforms
    train_transform = get_train_transforms(
        image_size=image_size,
        augmentation_strength=augmentation_strength,
        normalize=normalize
    )
    
    val_transform = get_val_transforms(
        image_size=image_size,
        normalize=normalize
    )
    
    return train_transform, val_transform


def visualize_transforms(image: Image.Image, 
                        transform: transforms.Compose,
                        num_samples: int = 4) -> List[torch.Tensor]:
    """
    Visualize the effect of transforms on an image.
    
    Args:
        image: Input PIL Image
        transform: Transform to apply
        num_samples: Number of transformed samples to generate
        
    Returns:
        List of transformed image tensors
    """
    samples = []
    for _ in range(num_samples):
        transformed = transform(image)
        samples.append(transformed)
    
    return samples 