"""
Script to download brain tumor MRI dataset from Kaggle.
"""

import argparse
import os
import sys
import kagglehub
from pathlib import Path
import shutil


def download_dataset(args):
    """
    Download dataset from Kaggle.
    
    Args:
        args: Command line arguments
    """
    print("Downloading brain tumor MRI dataset from Kaggle...")
    print("-" * 60)
    
    try:
        # Download latest version
        print("Connecting to Kaggle...")
        dataset_path = kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset")
        print(f"[OK] Dataset downloaded successfully!")
        print(f"Path to dataset files: {dataset_path}")
        
        # If output directory is specified, copy/move files
        if args.output_dir:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"\nOrganizing dataset to: {output_dir}")
            
            # Find all images in the downloaded dataset
            dataset_path = Path(dataset_path)
            
            # Look for common dataset structures
            # First, check what's in the root directory
            print(f"\nExploring dataset structure in: {dataset_path}")
            all_items = list(dataset_path.iterdir())
            print(f"Found {len(all_items)} items in root directory")
            
            possible_paths = [
                dataset_path / "Training",
                dataset_path / "training",
                dataset_path / "Train",
                dataset_path / "train",
                dataset_path / "Test",
                dataset_path / "test",
                dataset_path,
            ]
            
            training_dir = None
            test_dir = None
            
            # Find training and test directories
            for path in possible_paths:
                if path.exists() and path.is_dir():
                    # Check if it contains class folders (subdirectories with images)
                    subdirs = [d for d in path.iterdir() if d.is_dir()]
                    image_files = [f for f in path.iterdir() 
                                 if f.is_file() and f.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')]
                    
                    # If it has subdirectories, it's likely a class-organized folder
                    if len(subdirs) > 0:
                        print(f"  Found directory with {len(subdirs)} subdirectories: {path.name}")
                        # Determine if it's training or test based on name
                        path_name_lower = path.name.lower()
                        if 'train' in path_name_lower:
                            training_dir = path
                            print(f"    -> Identified as training directory")
                        elif 'test' in path_name_lower:
                            test_dir = path
                            print(f"    -> Identified as test directory")
                        elif training_dir is None:
                            training_dir = path
                            print(f"    -> Assuming training directory")
                        elif test_dir is None:
                            test_dir = path
                            print(f"    -> Assuming test directory")
                    
                    # If it has image files directly, it might be a flat structure
                    elif len(image_files) > 0:
                        print(f"  Found directory with {len(image_files)} images: {path.name}")
            
            # If we didn't find training/test splits, use the entire dataset
            if training_dir is None and test_dir is None:
                print("\nNo clear training/test split found. Using entire dataset as training data.")
                training_dir = dataset_path
            
            # Organize data
            if training_dir:
                print(f"\nProcessing training data from: {training_dir}")
                # Process training data
                class_dirs = [d for d in training_dir.iterdir() if d.is_dir()]
                
                if len(class_dirs) == 0:
                    # Flat structure - might need manual organization
                    print("  Warning: No class subdirectories found. Dataset might need manual organization.")
                    print("  Please check the dataset structure and organize manually if needed.")
                else:
                    for class_dir in class_dirs:
                        class_name = class_dir.name.lower()
                        
                        # Map class names to our naming convention
                        if any(keyword in class_name for keyword in ['no', 'normal', 'healthy', 'benign', 'negative']):
                            target_class = 'no_bleeding'
                        elif any(keyword in class_name for keyword in ['yes', 'tumor', 'bleeding', 'positive', 'malignant', 'glioma', 'meningioma', 'pituitary']):
                            target_class = 'bleeding'
                        else:
                            # Default mapping - check if it looks like a positive case
                            # Brain tumor datasets often have specific tumor types
                            target_class = 'bleeding' if any(keyword in class_name for keyword in ['tumor', 'glioma', 'meningioma', 'pituitary']) else 'no_bleeding'
                        
                        target_dir = output_dir / 'train' / target_class
                        target_dir.mkdir(parents=True, exist_ok=True)
                        
                        # Copy images
                        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')
                        images = [f for f in class_dir.iterdir() 
                                if f.is_file() and f.suffix.lower() in image_extensions]
                        
                        if len(images) > 0:
                            for img in images:
                                shutil.copy2(img, target_dir / img.name)
                            
                            print(f"  [OK] Copied {len(images)} images from '{class_dir.name}' -> train/{target_class}")
                        else:
                            print(f"  [WARNING] No images found in {class_dir.name}")
            
            if test_dir:
                print(f"\nProcessing test data from: {test_dir}")
                # Process test data
                class_dirs = [d for d in test_dir.iterdir() if d.is_dir()]
                
                for class_dir in class_dirs:
                    class_name = class_dir.name.lower()
                    
                    # Map class names
                    if any(keyword in class_name for keyword in ['no', 'normal', 'healthy', 'benign', 'negative']):
                        target_class = 'no_bleeding'
                    elif any(keyword in class_name for keyword in ['yes', 'tumor', 'bleeding', 'positive', 'malignant', 'glioma', 'meningioma', 'pituitary']):
                        target_class = 'bleeding'
                    else:
                        target_class = 'bleeding' if any(keyword in class_name for keyword in ['tumor', 'glioma', 'meningioma', 'pituitary']) else 'no_bleeding'
                    
                    target_dir = output_dir / 'test' / target_class
                    target_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Copy images
                    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')
                    images = [f for f in class_dir.iterdir() 
                            if f.is_file() and f.suffix.lower() in image_extensions]
                    
                    if len(images) > 0:
                        for img in images:
                            shutil.copy2(img, target_dir / img.name)
                        
                        print(f"  [OK] Copied {len(images)} images from '{class_dir.name}' -> test/{target_class}")
                    else:
                        print(f"  [WARNING] No images found in {class_dir.name}")
            
            # If we only have training data, split it into train/val/test
            if training_dir and not test_dir:
                print("\nSplitting training data into train/val/test...")
                import random
                
                # Split the training data
                for target_class in ['bleeding', 'no_bleeding']:
                    class_dir = output_dir / 'train' / target_class
                    if class_dir.exists():
                        images = list(class_dir.glob('*.*'))
                        images = [f for f in images if f.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')]
                        
                        if len(images) > 0:
                            # Split: 70% train, 15% val, 15% test
                            random.seed(42)
                            random.shuffle(images)
                            
                            train_end = int(0.7 * len(images))
                            val_end = int(0.85 * len(images))
                            
                            train_imgs = images[:train_end]
                            val_imgs = images[train_end:val_end]
                            test_imgs = images[val_end:]
                            
                            # Move validation images
                            val_dir = output_dir / 'val' / target_class
                            val_dir.mkdir(parents=True, exist_ok=True)
                            for img in val_imgs:
                                shutil.move(img, val_dir / img.name)
                            
                            # Move test images
                            test_dir_output = output_dir / 'test' / target_class
                            test_dir_output.mkdir(parents=True, exist_ok=True)
                            for img in test_imgs:
                                shutil.move(img, test_dir_output / img.name)
                            
                            print(f"  [OK] {target_class}: {len(train_imgs)} train, {len(val_imgs)} val, {len(test_imgs)} test")
            
            # Print summary
            print(f"\n{'='*60}")
            print("Dataset organization complete!")
            print(f"{'='*60}")
            print(f"Data saved to: {output_dir}")
            
            # Count images in each split
            splits = ['train', 'val', 'test']
            classes = ['bleeding', 'no_bleeding']
            total_images = 0
            
            print(f"\nDataset Summary:")
            for split in splits:
                split_total = 0
                for class_name in classes:
                    split_dir = output_dir / split / class_name
                    if split_dir.exists():
                        images = list(split_dir.glob('*.*'))
                        images = [f for f in images if f.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')]
                        count = len(images)
                        split_total += count
                        if count > 0:
                            print(f"  {split}/{class_name}: {count} images")
                total_images += split_total
                if split_total > 0:
                    print(f"  {split} total: {split_total} images")
            
            print(f"\nTotal images: {total_images}")
            print(f"\nNext steps:")
            print("1. Review the data organization in the output directory")
            print("2. Start training: python scripts/train.py --data_dir data")
            print(f"3. Or evaluate: python scripts/evaluate.py --model_path models/best_model.pth --data_dir data")
        else:
            print(f"\nDataset is available at: {dataset_path}")
            print("You can now organize it using preprocess_data.py script")
    
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("\nMake sure you have:")
        print("1. Installed kagglehub: pip install kagglehub")
        print("2. Set up Kaggle API credentials (if required)")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description='Download brain tumor MRI dataset from Kaggle')
    
    parser.add_argument('--output_dir', type=str, default='data',
                        help='Output directory for organized dataset')
    
    args = parser.parse_args()
    
    download_dataset(args)


if __name__ == '__main__':
    main()



