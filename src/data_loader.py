"""
Data loading and preprocessing utilities for brain bleeding classification.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np


class BrainBleedingDataset(Dataset):
    """
    Dataset class for brain bleeding images.
    """
    
    def __init__(self, data_dir, split='train', transform=None, img_size=(224, 224)):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Root directory containing train/val/test folders
            split: One of 'train', 'val', or 'test'
            transform: Optional transform to be applied on images
            img_size: Target image size (height, width)
        """
        self.data_dir = data_dir
        self.split = split
        self.img_size = img_size
        self.transform = transform
        
        # Define class names
        self.classes = ['no_bleeding', 'bleeding']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Load image paths and labels
        self.samples = []
        split_dir = os.path.join(data_dir, split)
        
        if not os.path.exists(split_dir):
            raise ValueError(f"Split directory not found: {split_dir}")
        
        for class_name in self.classes:
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"Warning: Class directory not found: {class_dir}")
                continue
            
            class_idx = self.class_to_idx[class_name]
            
            # Get all image files
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append((img_path, class_idx))
        
        if len(self.samples) == 0:
            raise ValueError(f"No images found in {split_dir}")
        
        print(f"Loaded {len(self.samples)} images from {split} set")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Index of the sample
        
        Returns:
            Tuple of (image, label)
        """
        img_path, label = self.samples[idx]
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
            image = np.array(image)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image as fallback
            image = np.zeros((self.img_size[0], self.img_size[1], 3), dtype=np.uint8)
        
        # Apply transforms
        if self.transform:
            if isinstance(self.transform, A.Compose):
                # Albumentations transform
                transformed = self.transform(image=image)
                image = transformed['image']
            else:
                # PyTorch transform
                image = Image.fromarray(image)
                image = self.transform(image)
        
        return image, label


def get_transforms(split='train', img_size=(224, 224), use_albumentations=True):
    """
    Get data augmentation transforms.
    
    Args:
        split: One of 'train', 'val', or 'test'
        img_size: Target image size
        use_albumentations: Whether to use Albumentations library
    
    Returns:
        Transform pipeline
    """
    if use_albumentations:
        if split == 'train':
            # Training transforms with augmentation
            transform = A.Compose([
                A.Resize(img_size[0], img_size[1]),
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.5
                ),
                A.GaussNoise(std_range=(0.1, 0.5), p=0.3),
                A.GaussianBlur(blur_limit=3, p=0.3),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])
        else:
            # Validation/Test transforms (no augmentation)
            transform = A.Compose([
                A.Resize(img_size[0], img_size[1]),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])
    else:
        # PyTorch transforms (fallback)
        if split == 'train':
            transform = transforms.Compose([
                transforms.Resize(img_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(10),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(img_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
    
    return transform


def get_data_loaders(data_dir, batch_size=32, img_size=(224, 224), num_workers=4, use_albumentations=True):
    """
    Create data loaders for train, validation, and test sets.
    
    Args:
        data_dir: Root directory containing train/val/test folders
        batch_size: Batch size for data loaders
        img_size: Target image size
        num_workers: Number of worker processes for data loading
        use_albumentations: Whether to use Albumentations
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Get transforms
    train_transform = get_transforms('train', img_size, use_albumentations)
    val_transform = get_transforms('val', img_size, use_albumentations)
    test_transform = get_transforms('test', img_size, use_albumentations)
    
    # Create datasets
    train_dataset = BrainBleedingDataset(
        data_dir=data_dir,
        split='train',
        transform=train_transform,
        img_size=img_size
    )
    
    val_dataset = BrainBleedingDataset(
        data_dir=data_dir,
        split='val',
        transform=val_transform,
        img_size=img_size
    )
    
    test_dataset = BrainBleedingDataset(
        data_dir=data_dir,
        split='test',
        transform=test_transform,
        img_size=img_size
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader, test_loader

