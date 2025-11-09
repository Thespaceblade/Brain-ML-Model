"""
Utility functions for training, evaluation, and visualization.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from typing import Dict, List, Tuple
import os


def save_checkpoint(model, optimizer, epoch, loss, accuracy, filepath):
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        loss: Current loss
        accuracy: Current accuracy
        filepath: Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(filepath, model, optimizer=None):
    """
    Load model checkpoint.
    
    Args:
        filepath: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optional optimizer to load state into
    
    Returns:
        Dictionary with checkpoint data
    """
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Checkpoint loaded from {filepath}")
    return checkpoint


def calculate_metrics(y_true, y_pred, average='binary'):
    """
    Calculate classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        average: Averaging strategy for multi-class
    
    Returns:
        Dictionary with metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'f1': f1_score(y_true, y_pred, average=average, zero_division=0)
    }
    return metrics


def plot_confusion_matrix(y_true, y_pred, class_names=None, save_path=None):
    """
    Plot and save confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Path to save the plot
    """
    if class_names is None:
        class_names = ['No Bleeding', 'Bleeding']
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_training_history(history: Dict[str, List[float]], save_path=None):
    """
    Plot training history (loss and accuracy curves).
    
    Args:
        history: Dictionary with 'train_loss', 'val_loss', 'train_acc', 'val_acc' keys
        save_path: Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracy
    ax2.plot(history['train_acc'], label='Train Accuracy')
    ax2.plot(history['val_acc'], label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def evaluate_model(model, data_loader, device, criterion=None):
    """
    Evaluate model on a dataset.
    
    Args:
        model: Model to evaluate
        data_loader: Data loader for evaluation
        device: Device to run evaluation on
        criterion: Loss function (optional)
    
    Returns:
        Dictionary with metrics and predictions
    """
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            # Calculate loss if criterion is provided
            if criterion is not None:
                loss = criterion(outputs, labels)
                total_loss += loss.item()
            
            # Store predictions and labels
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    metrics = calculate_metrics(all_labels, all_preds)
    
    if criterion is not None:
        metrics['loss'] = total_loss / len(data_loader)
    
    return {
        'metrics': metrics,
        'predictions': all_preds,
        'labels': all_labels
    }


def print_metrics(metrics: Dict[str, float], prefix=''):
    """
    Print metrics in a formatted way.
    
    Args:
        metrics: Dictionary with metric values
        prefix: Prefix to add to each metric name
    """
    print(f"\n{prefix}Metrics:")
    print("-" * 40)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key.capitalize()}: {value:.4f}")
        else:
            print(f"{key.capitalize()}: {value}")
    print("-" * 40)


def get_device():
    """
    Get the appropriate device (GPU if available, else CPU).
    
    Returns:
        torch.device object
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device

