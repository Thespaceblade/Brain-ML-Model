"""
Split existing training data into train/val/test splits.
"""

import argparse
import shutil
import random
from pathlib import Path


def split_data(data_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42):
    """
    Split data in train directory into train/val/test.
    
    Args:
        data_dir: Root data directory
        train_ratio: Ratio for training data
        val_ratio: Ratio for validation data
        test_ratio: Ratio for test data
        random_seed: Random seed for reproducibility
    """
    data_dir = Path(data_dir)
    train_dir = data_dir / 'train'
    
    if not train_dir.exists():
        print(f"Error: Training directory not found: {train_dir}")
        return
    
    # Set random seed
    random.seed(random_seed)
    
    # Process each class
    classes = ['bleeding', 'no_bleeding']
    
    for class_name in classes:
        class_dir = train_dir / class_name
        if not class_dir.exists():
            print(f"Warning: Class directory not found: {class_dir}")
            continue
        
        # Get all images
        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')
        images = [f for f in class_dir.iterdir() 
                 if f.is_file() and f.suffix.lower() in image_extensions]
        
        if len(images) == 0:
            print(f"Warning: No images found in {class_dir}")
            continue
        
        print(f"\nProcessing {class_name}: {len(images)} images")
        
        # Shuffle images
        random.shuffle(images)
        
        # Calculate split indices
        train_end = int(train_ratio * len(images))
        val_end = train_end + int(val_ratio * len(images))
        
        train_imgs = images[:train_end]
        val_imgs = images[train_end:val_end]
        test_imgs = images[val_end:]
        
        # Create output directories
        val_dir = data_dir / 'val' / class_name
        test_dir = data_dir / 'test' / class_name
        val_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)
        
        # Move validation images
        print(f"  Moving {len(val_imgs)} images to val/{class_name}...")
        for img in val_imgs:
            try:
                shutil.move(str(img), str(val_dir / img.name))
            except Exception as e:
                print(f"    Warning: Could not move {img.name}: {e}")
        
        # Move test images
        print(f"  Moving {len(test_imgs)} images to test/{class_name}...")
        for img in test_imgs:
            try:
                shutil.move(str(img), str(test_dir / img.name))
            except Exception as e:
                print(f"    Warning: Could not move {img.name}: {e}")
        
        print(f"  Result: {len(train_imgs)} train, {len(val_imgs)} val, {len(test_imgs)} test")
    
    # Print summary
    print(f"\n{'='*60}")
    print("Data splitting complete!")
    print(f"{'='*60}")
    
    for split in ['train', 'val', 'test']:
        split_total = 0
        for class_name in classes:
            split_dir = data_dir / split / class_name
            if split_dir.exists():
                images = [f for f in split_dir.iterdir() 
                         if f.is_file() and f.suffix.lower() in image_extensions]
                count = len(images)
                split_total += count
                if count > 0:
                    print(f"  {split}/{class_name}: {count} images")
        if split_total > 0:
            print(f"  {split} total: {split_total} images")


def main():
    parser = argparse.ArgumentParser(description='Split training data into train/val/test')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Root data directory')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                        help='Ratio for training data')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                        help='Ratio for validation data')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                        help='Ratio for test data')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Validate ratios
    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")
    
    split_data(args.data_dir, args.train_ratio, args.val_ratio, args.test_ratio, args.random_seed)


if __name__ == '__main__':
    main()

