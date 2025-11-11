"""
Training script for brain bleeding classification model.
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import get_model
from src.data_loader import get_data_loaders
from src.utils import (
    save_checkpoint,
    evaluate_model,
    plot_training_history,
    print_metrics,
    get_device
)


def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train for one epoch.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to run on
    
    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(train_loader, desc='Training')
    for images, labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100 * correct / total:.2f}%'
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc


def train(args):
    """
    Main training function.
    
    Args:
        args: Command line arguments
    """
    # Set device
    device = get_device()
    
    # Create output directories
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Get data loaders
    print("Loading data...")
    train_loader, val_loader, test_loader = get_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        img_size=(args.img_size, args.img_size),
        num_workers=args.num_workers,
        use_albumentations=args.use_albumentations
    )
    
    # Create model
    print(f"Creating model: {args.model_name}")
    model = get_model(
        model_name=args.model_name,
        num_classes=2,
        pretrained=args.pretrained,
        freeze_backbone=args.freeze_backbone
    )
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    
    if args.optimizer == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.learning_rate,
            momentum=0.9,
            weight_decay=args.weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")
    
    # Learning rate scheduler
    if args.scheduler == 'reduce_on_plateau':
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
    elif args.scheduler == 'step':
        scheduler = StepLR(
            optimizer,
            step_size=args.step_size,
            gamma=args.gamma
        )
    else:
        scheduler = None
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # Best model tracking
    best_val_acc = 0.0
    best_model_path = os.path.join(args.model_dir, 'best_model.pth')
    patience_counter = 0
    
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Model will be saved to: {args.model_dir}")
    print(f"Logs will be saved to: {args.log_dir}")
    print("-" * 60)
    
    # Training loop
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 60)
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        print("Validating...")
        val_results = evaluate_model(model, val_loader, device, criterion)
        val_loss = val_results['metrics']['loss']
        val_acc = val_results['metrics']['accuracy']
        
        # Update learning rate
        if scheduler:
            if args.scheduler == 'reduce_on_plateau':
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print metrics
        print(f"\nTrain Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                model, optimizer, epoch, val_loss, val_acc, best_model_path
            )
            print(f"✓ New best model saved! (Val Acc: {val_acc:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if args.early_stopping > 0 and patience_counter >= args.early_stopping:
            print(f"\nEarly stopping triggered after {epoch} epochs")
            break
        
        # Save checkpoint periodically
        if epoch % args.save_every == 0:
            checkpoint_path = os.path.join(
                args.model_dir, f'checkpoint_epoch_{epoch}.pth'
            )
            save_checkpoint(
                model, optimizer, epoch, val_loss, val_acc, checkpoint_path
            )
    
    # Save final model
    final_model_path = os.path.join(args.model_dir, 'final_model.pth')
    save_checkpoint(
        model, optimizer, args.epochs, val_loss, val_acc, final_model_path
    )
    
    # Plot training history
    history_path = os.path.join(args.log_dir, 'training_history.png')
    plot_training_history(history, save_path=history_path)
    
    # Evaluate on test set
    print("\n" + "=" * 60)
    print("Evaluating on test set...")
    print("=" * 60)
    test_results = evaluate_model(model, test_loader, device, criterion)
    print_metrics(test_results['metrics'], prefix='Test ')
    
    print(f"\nTraining completed!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Best model saved to: {best_model_path}")


def main():
    parser = argparse.ArgumentParser(description='Train brain bleeding classification model')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Root directory containing train/val/test folders')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Image size for training')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of worker processes for data loading')
    parser.add_argument('--use_albumentations', action='store_true', default=True,
                        help='Use Albumentations for data augmentation')
    
    # Model arguments
    parser.add_argument('--model_name', type=str, default='resnet50',
                        choices=['resnet50', 'efficientnet_b0', 'efficientnet_b1'],
                        help='Model architecture to use')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Use pretrained weights')
    parser.add_argument('--freeze_backbone', action='store_true', default=False,
                        help='Freeze backbone for fine-tuning')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay for optimizer')
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['adam', 'sgd'],
                        help='Optimizer to use')
    parser.add_argument('--scheduler', type=str, default='reduce_on_plateau',
                        choices=['reduce_on_plateau', 'step', 'none'],
                        help='Learning rate scheduler')
    parser.add_argument('--step_size', type=int, default=10,
                        help='Step size for step scheduler')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='Gamma for step scheduler')
    
    # Checkpointing arguments
    parser.add_argument('--model_dir', type=str, default='models',
                        help='Directory to save models')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory to save logs')
    parser.add_argument('--save_every', type=int, default=10,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--early_stopping', type=int, default=10,
                        help='Early stopping patience (0 to disable)')
    
    args = parser.parse_args()
    
    train(args)


if __name__ == '__main__':
    main()

