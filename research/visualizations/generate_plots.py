"""
Script to generate all evaluation visualizations from model predictions.
"""

import argparse
import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.model import get_model
from src.data_loader import BrainBleedingDataset, get_transforms
from src.utils import load_checkpoint, get_device
from research.visualizations.plotter import generate_all_visualizations
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def get_predictions_and_scores(model, data_loader, device):
    """
    Get predictions and prediction scores from model.
    
    Args:
        model: Trained model
        data_loader: Data loader
        device: Device to run on
    
    Returns:
        Tuple of (y_true, y_pred, y_scores)
    """
    model.eval()
    all_labels = []
    all_preds = []
    all_scores = []
    
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_scores.extend(probabilities[:, 1].cpu().numpy())  # Probability of positive class
    
    return np.array(all_labels), np.array(all_preds), np.array(all_scores)


def load_training_history(log_dir):
    """
    Load training history from logs if available.
    
    Args:
        log_dir: Directory containing training logs
    
    Returns:
        Dictionary with training history or None
    """
    # This is a placeholder - implement based on your logging format
    # For now, return None
    return None


def main():
    parser = argparse.ArgumentParser(description='Generate evaluation visualizations')
    
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
    parser.add_argument('--output_dir', type=str, default='research/visualizations/output',
                      help='Directory to save visualizations')
    parser.add_argument('--display_name', type=str, default=None,
                      help='Display name for the model in plots')
    
    args = parser.parse_args()
    
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
        split=args.split,
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
    
    # Get predictions
    print(f"\nGenerating predictions on {args.split} set...")
    y_true, y_pred, y_scores = get_predictions_and_scores(model, data_loader, device)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    print(f"\nMetrics on {args.split} set:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    
    # Load training history if available
    history = load_training_history('logs')
    
    # Generate visualizations
    model_display_name = args.display_name or f"{args.model_name}_{args.split}"
    
    print(f"\nGenerating visualizations...")
    saved_paths = generate_all_visualizations(
        y_true=y_true,
        y_pred=y_pred,
        y_scores=y_scores,
        history=history,
        model_name=model_display_name,
        output_dir=args.output_dir,
        class_names=['No Bleeding', 'Bleeding']
    )
    
    print(f"\nVisualization generation complete!")
    print(f"All plots saved to: {args.output_dir}")


if __name__ == '__main__':
    main()

