"""
Prepare Kaggle brain tumor dataset for brain bleeding classification.
Maps tumor classes to bleeding/no_bleeding classes.
"""

import argparse
import shutil
import random
from pathlib import Path


def prepare_data(kaggle_path, output_dir='data', train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42):
    """
    Prepare Kaggle dataset by mapping classes and creating train/val/test splits.
    
    Args:
        kaggle_path: Path to downloaded Kaggle dataset
        output_dir: Output directory for prepared data
        train_ratio: Ratio for training data
        val_ratio: Ratio for validation data
        test_ratio: Ratio for test data
        random_seed: Random seed for reproducibility
    """
    kaggle_path = Path(kaggle_path)
    output_dir = Path(output_dir)
    
    # Class mapping: tumor types -> bleeding, notumor -> no_bleeding
    tumor_classes = ['glioma', 'meningioma', 'pituitary']
    no_tumor_class = 'notumor'
    output_classes = ['bleeding', 'no_bleeding']
    
    # Set random seed
    random.seed(random_seed)
    
    # Create output directories
    for split in ['train', 'val', 'test']:
        for class_name in output_classes:
            (output_dir / split / class_name).mkdir(parents=True, exist_ok=True)
    
    # Process Training data (will be split into train/val)
    training_dir = kaggle_path / 'Training'
    if not training_dir.exists():
        raise ValueError(f"Training directory not found: {training_dir}")
    
    # Collect all images with their mapped labels
    all_samples = {'bleeding': [], 'no_bleeding': []}
    
    # Process tumor classes (bleeding)
    for tumor_class in tumor_classes:
        class_dir = training_dir / tumor_class
        if class_dir.exists():
            image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')
            images = [f for f in class_dir.iterdir() 
                     if f.is_file() and f.suffix.lower() in image_extensions]
            all_samples['bleeding'].extend(images)
            print(f"Found {len(images)} {tumor_class} images (-> bleeding)")
    
    # Process no tumor class (no_bleeding)
    no_tumor_dir = training_dir / no_tumor_class
    if no_tumor_dir.exists():
        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')
        images = [f for f in no_tumor_dir.iterdir() 
                 if f.is_file() and f.suffix.lower() in image_extensions]
        all_samples['no_bleeding'].extend(images)
        print(f"Found {len(images)} {no_tumor_class} images (-> no_bleeding)")
    
    # Split training data into train/val
    print(f"\nSplitting training data...")
    for class_name, images in all_samples.items():
        if len(images) == 0:
            continue
        
        random.shuffle(images)
        train_end = int(train_ratio * len(images))
        val_end = train_end + int(val_ratio * len(images))
        
        train_imgs = images[:train_end]
        val_imgs = images[train_end:val_end]
        test_imgs = images[val_end:]  # This will be replaced by Testing data
        
        # Copy train images
        print(f"  Copying {len(train_imgs)} {class_name} images to train/...")
        for img in train_imgs:
            shutil.copy2(img, output_dir / 'train' / class_name / img.name)
        
        # Copy val images
        print(f"  Copying {len(val_imgs)} {class_name} images to val/...")
        for img in val_imgs:
            shutil.copy2(img, output_dir / 'val' / class_name / img.name)
    
    # Process Testing data (use as test set)
    testing_dir = kaggle_path / 'Testing'
    if testing_dir.exists():
        print(f"\nProcessing Testing data...")
        test_samples = {'bleeding': [], 'no_bleeding': []}
        
        # Process tumor classes (bleeding)
        for tumor_class in tumor_classes:
            class_dir = testing_dir / tumor_class
            if class_dir.exists():
                image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')
                images = [f for f in class_dir.iterdir() 
                         if f.is_file() and f.suffix.lower() in image_extensions]
                test_samples['bleeding'].extend(images)
                print(f"  Found {len(images)} {tumor_class} test images (-> bleeding)")
        
        # Process no tumor class (no_bleeding)
        no_tumor_dir = testing_dir / no_tumor_class
        if no_tumor_dir.exists():
            image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')
            images = [f for f in no_tumor_dir.iterdir() 
                     if f.is_file() and f.suffix.lower() in image_extensions]
            test_samples['no_bleeding'].extend(images)
            print(f"  Found {len(images)} {no_tumor_class} test images (-> no_bleeding)")
        
        # Copy test images
        for class_name, images in test_samples.items():
            print(f"  Copying {len(images)} {class_name} images to test/...")
            for img in images:
                shutil.copy2(img, output_dir / 'test' / class_name / img.name)
    
    # Print summary
    print(f"\n{'='*60}")
    print("Data preparation complete!")
    print(f"{'='*60}")
    
    for split in ['train', 'val', 'test']:
        split_total = 0
        for class_name in output_classes:
            split_dir = output_dir / split / class_name
            if split_dir.exists():
                image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')
                images = [f for f in split_dir.iterdir() 
                         if f.is_file() and f.suffix.lower() in image_extensions]
                count = len(images)
                split_total += count
                if count > 0:
                    print(f"  {split}/{class_name}: {count} images")
        if split_total > 0:
            print(f"  {split} total: {split_total} images")


def main():
    parser = argparse.ArgumentParser(description='Prepare Kaggle brain tumor dataset')
    parser.add_argument('--kaggle_path', type=str, 
                        default='/Users/jasoncharwin/.cache/kagglehub/datasets/masoudnickparvar/brain-tumor-mri-dataset/versions/1',
                        help='Path to downloaded Kaggle dataset')
    parser.add_argument('--output_dir', type=str, default='data',
                        help='Output directory for prepared data')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                        help='Ratio for training data')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                        help='Ratio for validation data')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                        help='Ratio for test data (not used if Testing folder exists)')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Validate ratios
    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")
    
    prepare_data(args.kaggle_path, args.output_dir, args.train_ratio, 
                 args.val_ratio, args.test_ratio, args.random_seed)


if __name__ == '__main__':
    main()


