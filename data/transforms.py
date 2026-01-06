"""
Data augmentation transforms for remote sensing images
Uses albumentations library for efficient augmentation
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


def get_training_augmentation(image_size=256):
    """
    Get training augmentation pipeline
    
    Args:
        image_size: Target image size
    
    Returns:
        Albumentations transform
    """
    train_transform = A.Compose([
        # Resize
        A.Resize(image_size, image_size, interpolation=cv2.INTER_LINEAR),
        
        # Geometric transforms
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=30, p=0.3, border_mode=cv2.BORDER_CONSTANT),
        
        # Color augmentation
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1),
            A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=1),
        ], p=0.3),
        
        # Noise
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
        
        # Blur
        A.OneOf([
            A.GaussianBlur(blur_limit=3, p=1),
            A.MedianBlur(blur_limit=3, p=1),
        ], p=0.1),
        
        # Normalize
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        
        # Convert to tensor
        ToTensorV2(),
    ])
    
    return train_transform


def get_validation_augmentation(image_size=256):
    """
    Get validation augmentation pipeline (only resize and normalize)
    
    Args:
        image_size: Target image size
    
    Returns:
        Albumentations transform
    """
    val_transform = A.Compose([
        A.Resize(image_size, image_size, interpolation=cv2.INTER_LINEAR),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    
    return val_transform


def get_test_augmentation(image_size=256):
    """
    Get test augmentation pipeline (same as validation)
    
    Args:
        image_size: Target image size
    
    Returns:
        Albumentations transform
    """
    return get_validation_augmentation(image_size)


def get_light_augmentation(image_size=256):
    """
    Get light augmentation for faster training
    
    Args:
        image_size: Target image size
    
    Returns:
        Albumentations transform
    """
    light_transform = A.Compose([
        A.Resize(image_size, image_size, interpolation=cv2.INTER_LINEAR),
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    
    return light_transform


def get_strong_augmentation(image_size=256):
    """
    Get strong augmentation for robust training
    
    Args:
        image_size: Target image size
    
    Returns:
        Albumentations transform
    """
    strong_transform = A.Compose([
        A.Resize(image_size, image_size, interpolation=cv2.INTER_LINEAR),
        
        # Geometric transforms
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=45, p=0.5, border_mode=cv2.BORDER_CONSTANT),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, p=0.5),
        
        # Elastic transforms
        A.ElasticTransform(p=0.2),
        A.GridDistortion(p=0.2),
        
        # Color augmentation
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1),
            A.HueSaturationValue(hue_shift_limit=30, sat_shift_limit=40, val_shift_limit=30, p=1),
            A.RGBShift(r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=1),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1),
        ], p=0.5),
        
        # Noise and blur
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 100.0), p=1),
            A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=1),
        ], p=0.3),
        
        A.OneOf([
            A.GaussianBlur(blur_limit=5, p=1),
            A.MedianBlur(blur_limit=5, p=1),
            A.MotionBlur(blur_limit=5, p=1),
        ], p=0.2),
        
        # Weather effects
        A.OneOf([
            A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=1),
            A.RandomRain(blur_value=3, brightness_coefficient=0.95, p=1),
            A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), p=1),
        ], p=0.1),
        
        # Normalize
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        
        # Convert to tensor
        ToTensorV2(),
    ])
    
    return strong_transform


def get_transform(split='train', augmentation_level='normal', image_size=256):
    """
    Factory function to get transforms
    
    Args:
        split: 'train', 'val', or 'test'
        augmentation_level: 'light', 'normal', or 'strong'
        image_size: Target image size
    
    Returns:
        Albumentations transform
    """
    if split == 'train':
        if augmentation_level == 'light':
            return get_light_augmentation(image_size)
        elif augmentation_level == 'normal':
            return get_training_augmentation(image_size)
        elif augmentation_level == 'strong':
            return get_strong_augmentation(image_size)
        else:
            raise ValueError(f"Unknown augmentation level: {augmentation_level}")
    
    elif split in ['val', 'test']:
        return get_validation_augmentation(image_size)
    
    else:
        raise ValueError(f"Unknown split: {split}")


if __name__ == "__main__":
    import numpy as np
    from PIL import Image
    
    print("Testing data augmentation transforms...")
    
    # Create dummy image and mask
    image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    mask = np.random.randint(0, 6, (256, 256), dtype=np.uint8)
    
    # Test different transforms
    print("\n1. Testing training transform:")
    train_transform = get_training_augmentation(256)
    augmented = train_transform(image=image, mask=mask)
    print(f"  Image shape: {augmented['image'].shape}")
    print(f"  Mask shape: {augmented['mask'].shape}")
    
    print("\n2. Testing validation transform:")
    val_transform = get_validation_augmentation(256)
    augmented = val_transform(image=image, mask=mask)
    print(f"  Image shape: {augmented['image'].shape}")
    print(f"  Mask shape: {augmented['mask'].shape}")
    
    print("\n3. Testing light augmentation:")
    light_transform = get_light_augmentation(256)
    augmented = light_transform(image=image, mask=mask)
    print(f"  Image shape: {augmented['image'].shape}")
    print(f"  Mask shape: {augmented['mask'].shape}")
    
    print("\n4. Testing strong augmentation:")
    strong_transform = get_strong_augmentation(256)
    augmented = strong_transform(image=image, mask=mask)
    print(f"  Image shape: {augmented['image'].shape}")
    print(f"  Mask shape: {augmented['mask'].shape}")
    
    print("\nAll transforms tested successfully!")
