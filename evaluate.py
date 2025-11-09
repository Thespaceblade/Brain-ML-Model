"""
Evaluation script for brain bleeding classification model.
"""

import argparse
import os
import sys
import torch
from torch.utils.data import DataLoader

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import get_model
from src.data_loader import BrainBleedingDataset, get_transforms
from src.utils import (
    load_checkpoint,
    evaluate_model,
    plot_confusion_matrix,
    print_metrics,
    get_device
)
from sklearn.metrics import classification_report


def evaluate(args):
    """
    Evaluate model on test set.
    
    Args:
        args: Command line arguments
    """
    # Set device
    device = get_device()
    
    # Load model
    print(f"Loading model from {args.model_path}")
    model = get_model(
        model_name=args.model_name,
        num_classes=2,
        pretrained=False
    )
    model = model.to(device)
    
    # Load checkpoint
    checkpoint = load_checkpoint(args.model_path, model)
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"Checkpoint accuracy: {checkpoint['accuracy']:.4f}")
    
    # Get transforms
    transform = get_transforms(
        split='test',
        img_size=(args.img_size, args.img_size),
        use_albumentations=args.use_albumentations
    )
    
    # Create dataset and data loader
    dataset = BrainBleedingDataset(
        data_dir=args.data_dir,
        split=args.split,
        transform=transform,
        img_size=(args.img_size, args.img_size)
    )
    
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    # Evaluate model
    print(f"\nEvaluating on {args.split} set...")
    print("-" * 60)
    
    criterion = torch.nn.CrossEntropyLoss()
    results = evaluate_model(model, data_loader, device, criterion)
    
    # Print metrics
    print_metrics(results['metrics'], prefix=f'{args.split.capitalize()} ')
    
    # Classification report
    print(f"\nClassification Report:")
    print("-" * 60)
    print(classification_report(
        results['labels'],
        results['predictions'],
        target_names=['No Bleeding', 'Bleeding']
    ))
    
    # Plot confusion matrix
    if args.save_plots:
        os.makedirs(args.output_dir, exist_ok=True)
        cm_path = os.path.join(
            args.output_dir,
            f'confusion_matrix_{args.split}.png'
        )
        plot_confusion_matrix(
            results['labels'],
            results['predictions'],
            class_names=['No Bleeding', 'Bleeding'],
            save_path=cm_path
        )
        print(f"\nConfusion matrix saved to: {cm_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate brain bleeding classification model')
    
    # Model arguments
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--model_name', type=str, default='resnet50',
                        choices=['resnet50', 'efficientnet_b0', 'efficientnet_b1'],
                        help='Model architecture')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Root directory containing train/val/test folders')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Dataset split to evaluate on')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Image size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of worker processes')
    parser.add_argument('--use_albumentations', action='store_true', default=True,
                        help='Use Albumentations for transforms')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory to save results')
    parser.add_argument('--save_plots', action='store_true', default=True,
                        help='Save confusion matrix plot')
    
    args = parser.parse_args()
    
    evaluate(args)


if __name__ == '__main__':
    main()



