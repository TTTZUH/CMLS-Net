"""
Dataset loaders for WHDLD, Vaihingen, and Potsdam datasets
"""

import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import glob


class RemoteSensingDataset(Dataset):
    """
    Generic Remote Sensing Dataset
    Supports WHDLD, Vaihingen, and Potsdam datasets
    """
    
    def __init__(self, root_dir, split='train', transform=None, dataset_type='whdld'):
        """
        Args:
            root_dir: Root directory of the dataset
            split: 'train', 'val', or 'test'
            transform: Transformations to apply
            dataset_type: 'whdld', 'vaihingen', or 'potsdam'
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.dataset_type = dataset_type.lower()
        
        # Load file lists
        self.images, self.labels = self._load_file_lists()
        
        print(f"Loaded {len(self.images)} {split} samples from {dataset_type}")
    
    def _load_file_lists(self):
        """Load lists of image and label files"""
        if self.dataset_type == 'whdld':
            return self._load_whdld()
        elif self.dataset_type == 'vaihingen':
            return self._load_vaihingen()
        elif self.dataset_type == 'potsdam':
            return self._load_potsdam()
        else:
            raise ValueError(f"Unknown dataset type: {self.dataset_type}")
    
    def _load_whdld(self):
        """Load WHDLD dataset"""
        img_dir = os.path.join(self.root_dir, self.split, 'images')
        label_dir = os.path.join(self.root_dir, self.split, 'labels')
        
        images = sorted(glob.glob(os.path.join(img_dir, '*.jpg')))
        if not images:
            images = sorted(glob.glob(os.path.join(img_dir, '*.png')))
        
        labels = sorted(glob.glob(os.path.join(label_dir, '*.png')))
        
        assert len(images) == len(labels), f"Mismatch: {len(images)} images vs {len(labels)} labels"
        
        return images, labels
    
    def _load_vaihingen(self):
        """Load Vaihingen dataset"""
        if self.split == 'train':
            split_dir = 'training'
        else:
            split_dir = self.split
        
        img_dir = os.path.join(self.root_dir, split_dir, 'images')
        label_dir = os.path.join(self.root_dir, split_dir, 'labels')
        
        images = sorted(glob.glob(os.path.join(img_dir, '*.tif')))
        if not images:
            images = sorted(glob.glob(os.path.join(img_dir, '*.png')))
        
        labels = sorted(glob.glob(os.path.join(label_dir, '*.png')))
        
        return images, labels
    
    def _load_potsdam(self):
        """Load Potsdam dataset"""
        if self.split == 'train':
            split_dir = 'training'
        else:
            split_dir = self.split
        
        img_dir = os.path.join(self.root_dir, split_dir, 'images')
        label_dir = os.path.join(self.root_dir, split_dir, 'labels')
        
        images = sorted(glob.glob(os.path.join(img_dir, '*.tif')))
        if not images:
            images = sorted(glob.glob(os.path.join(img_dir, '*.png')))
        
        labels = sorted(glob.glob(os.path.join(label_dir, '*.png')))
        
        return images, labels
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        """Get a sample"""
        # Load image
        image_path = self.images[idx]
        image = Image.open(image_path).convert('RGB')
        
        # Load label
        label_path = self.labels[idx]
        label = Image.open(label_path)
        
        # Convert to numpy arrays
        image = np.array(image, dtype=np.uint8)
        label = np.array(label, dtype=np.int64)
        
        # Apply transforms
        if self.transform is not None:
            augmented = self.transform(image=image, mask=label)
            image = augmented['image']
            label = augmented['mask']
        
        # Convert to tensors
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        if not isinstance(label, torch.Tensor):
            label = torch.from_numpy(label).long()
        
        return image, label


class WHDLDDataset(RemoteSensingDataset):
    """WHDLD Dataset"""
    def __init__(self, root_dir, split='train', transform=None):
        super().__init__(root_dir, split, transform, dataset_type='whdld')


class VaihingenDataset(RemoteSensingDataset):
    """Vaihingen Dataset"""
    def __init__(self, root_dir, split='train', transform=None):
        super().__init__(root_dir, split, transform, dataset_type='vaihingen')


class PotsdamDataset(RemoteSensingDataset):
    """Potsdam Dataset"""
    def __init__(self, root_dir, split='train', transform=None):
        super().__init__(root_dir, split, transform, dataset_type='potsdam')


def get_dataset(dataset_name, root_dir, split='train', transform=None):
    """
    Factory function to get dataset
    
    Args:
        dataset_name: 'whdld', 'vaihingen', or 'potsdam'
        root_dir: Root directory of the dataset
        split: 'train', 'val', or 'test'
        transform: Transformations to apply
    
    Returns:
        Dataset instance
    """
    dataset_name = dataset_name.lower()
    
    if dataset_name == 'whdld':
        return WHDLDDataset(root_dir, split, transform)
    elif dataset_name == 'vaihingen':
        return VaihingenDataset(root_dir, split, transform)
    elif dataset_name == 'potsdam':
        return PotsdamDataset(root_dir, split, transform)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


if __name__ == "__main__":
    # Test dataset loading
    print("Testing dataset loaders...")
    
    # Note: This assumes you have the datasets downloaded
    # You'll need to update the paths accordingly
    
    print("\nDataset loaders are ready!")
    print("To test with actual data, please provide dataset paths.")
