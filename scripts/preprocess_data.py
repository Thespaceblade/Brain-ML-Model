"""
Data preprocessing script for brain bleeding classification dataset.
"""

import argparse
import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
import numpy as np


def organize_data(args):
    """
    Organize dataset into train/val/test splits.
    
    Args:
        args: Command line arguments
    """
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    # Create output directories
    splits = ['train', 'val', 'test']
    classes = ['bleeding', 'no_bleeding']
    
    for split in splits:
        for class_name in classes:
            os.makedirs(output_dir / split / class_name, exist_ok=True)
    
    # If data is already organized by class
    if (input_dir / 'bleeding').exists() and (input_dir / 'no_bleeding').exists():
        print("Organizing data from class-based structure...")
        
        for class_name in classes:
            class_dir = input_dir / class_name
            if not class_dir.exists():
                print(f"Warning: Class directory not found: {class_dir}")
                continue
            
            # Get all images
            image_files = list(class_dir.glob('*.*'))
            image_files = [f for f in image_files if f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']]
            
            if len(image_files) == 0:
                print(f"Warning: No images found in {class_dir}")
                continue
            
            print(f"Found {len(image_files)} images in {class_name}")
            
            # Split data
            train_files, temp_files = train_test_split(
                image_files,
                test_size=1 - args.train_ratio,
                random_state=args.random_seed
            )
            
            val_files, test_files = train_test_split(
                temp_files,
                test_size=args.test_ratio / (args.val_ratio + args.test_ratio),
                random_state=args.random_seed
            )
            
            # Copy files
            for file in train_files:
                shutil.copy2(file, output_dir / 'train' / class_name / file.name)
            
            for file in val_files:
                shutil.copy2(file, output_dir / 'val' / class_name / file.name)
            
            for file in test_files:
                shutil.copy2(file, output_dir / 'test' / class_name / file.name)
            
            print(f"  Train: {len(train_files)} images")
            print(f"  Val: {len(val_files)} images")
            print(f"  Test: {len(test_files)} images")
    
    else:
        print("Warning: Expected class-based structure (bleeding/ and no_bleeding/ folders)")
        print("Please organize your data first or provide a different input structure")
    
    print(f"\nData organization complete!")
    print(f"Output directory: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Preprocess brain bleeding dataset')
    
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Input directory containing images')
    parser.add_argument('--output_dir', type=str, default='data',
                        help='Output directory for organized data')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                        help='Ratio of training data')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                        help='Ratio of validation data')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                        help='Ratio of test data')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for splitting')
    
    args = parser.parse_args()
    
    # Validate ratios
    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")
    
    organize_data(args)


if __name__ == '__main__':
    main()



