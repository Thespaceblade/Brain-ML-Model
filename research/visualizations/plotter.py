"""
Publication-quality visualization functions for model evaluation metrics.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, 
    roc_auc_score, 
    precision_recall_curve, 
    average_precision_score,
    confusion_matrix
)
from sklearn.calibration import calibration_curve
from typing import Dict, List, Tuple, Optional, Union
import os


# Set publication-quality style
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except OSError:
    try:
        plt.style.use('seaborn-whitegrid')
    except OSError:
        plt.style.use('default')
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3

sns.set_palette("husl")

# Custom color palette for medical imaging
MEDICAL_COLORS = {
    'primary': '#2E86AB',      # Deep blue
    'secondary': '#A23B72',     # Deep purple
    'accent': '#F18F01',        # Orange
    'success': '#06A77D',       # Green
    'danger': '#D00000',        # Red
    'neutral': '#6C757D',       # Gray
    'light': '#F8F9FA',         # Light gray
}

# Figure settings
FIGURE_DPI = 300
FIGURE_FORMAT = 'png'
FONT_SIZE = 11
TITLE_SIZE = 14
LABEL_SIZE = 12
LEGEND_SIZE = 10


def _setup_plot_style(ax, title: str, xlabel: str, ylabel: str, 
                     grid: bool = True, legend: bool = True):
    """Setup consistent plot styling."""
    ax.set_title(title, fontsize=TITLE_SIZE, fontweight='bold', pad=15)
    ax.set_xlabel(xlabel, fontsize=LABEL_SIZE, fontweight='medium')
    ax.set_ylabel(ylabel, fontsize=LABEL_SIZE, fontweight='medium')
    
    if grid:
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_axisbelow(True)
    
    if legend:
        ax.legend(fontsize=LEGEND_SIZE, framealpha=0.9, loc='best')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    
    plt.tight_layout()


def plot_roc_curve(y_true: np.ndarray, y_scores: np.ndarray, 
                   model_name: str = "Model", 
                   save_path: Optional[str] = None,
                   show_plot: bool = False,
                   figsize: Tuple[int, int] = (8, 6)) -> plt.Figure:
    """
    Plot ROC curve with AUC score.
    
    Args:
        y_true: True binary labels
        y_scores: Target scores (probabilities of positive class)
        model_name: Name of the model for legend
        save_path: Path to save the figure
        show_plot: Whether to display the plot
        figsize: Figure size (width, height)
    
    Returns:
        matplotlib Figure object
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    auc_score = roc_auc_score(y_true, y_scores)
    
    fig, ax = plt.subplots(figsize=figsize, dpi=FIGURE_DPI)
    
    ax.plot(fpr, tpr, 
            color=MEDICAL_COLORS['primary'], 
            linewidth=2.5,
            label=f'{model_name} (AUC = {auc_score:.3f})',
            alpha=0.9)
    
    # Diagonal reference line
    ax.plot([0, 1], [0, 1], 
            color=MEDICAL_COLORS['neutral'], 
            linestyle='--', 
            linewidth=1.5,
            alpha=0.5,
            label='Random Classifier')
    
    ax.fill_between(fpr, tpr, alpha=0.2, color=MEDICAL_COLORS['primary'])
    
    _setup_plot_style(
        ax, 
        'Receiver Operating Characteristic (ROC) Curve',
        'False Positive Rate (1 - Specificity)',
        'True Positive Rate (Sensitivity)',
        grid=True,
        legend=True
    )
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_aspect('equal', adjustable='box')
    
    if save_path:
        fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"ROC curve saved to: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_pr_curve(y_true: np.ndarray, y_scores: np.ndarray,
                  model_name: str = "Model",
                  save_path: Optional[str] = None,
                  show_plot: bool = False,
                  figsize: Tuple[int, int] = (8, 6)) -> plt.Figure:
    """
    Plot Precision-Recall curve with AP score.
    
    Args:
        y_true: True binary labels
        y_scores: Target scores (probabilities of positive class)
        model_name: Name of the model for legend
        save_path: Path to save the figure
        show_plot: Whether to display the plot
        figsize: Figure size (width, height)
    
    Returns:
        matplotlib Figure object
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    ap_score = average_precision_score(y_true, y_scores)
    
    # Baseline (random classifier)
    baseline = np.sum(y_true) / len(y_true)
    
    fig, ax = plt.subplots(figsize=figsize, dpi=FIGURE_DPI)
    
    ax.plot(recall, precision,
            color=MEDICAL_COLORS['secondary'],
            linewidth=2.5,
            label=f'{model_name} (AP = {ap_score:.3f})',
            alpha=0.9)
    
    # Baseline reference line
    ax.axhline(y=baseline, 
               color=MEDICAL_COLORS['neutral'],
               linestyle='--',
               linewidth=1.5,
               alpha=0.5,
               label=f'Baseline (AP = {baseline:.3f})')
    
    ax.fill_between(recall, precision, alpha=0.2, color=MEDICAL_COLORS['secondary'])
    
    _setup_plot_style(
        ax,
        'Precision-Recall Curve',
        'Recall (Sensitivity)',
        'Precision',
        grid=True,
        legend=True
    )
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    
    if save_path:
        fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"PR curve saved to: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                          class_names: Optional[List[str]] = None,
                          normalize: bool = False,
                          model_name: str = "Model",
                          save_path: Optional[str] = None,
                          show_plot: bool = False,
                          figsize: Tuple[int, int] = (8, 6)) -> plt.Figure:
    """
    Plot confusion matrix with annotations.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        normalize: Whether to normalize the confusion matrix
        model_name: Name of the model
        save_path: Path to save the figure
        show_plot: Whether to display the plot
        figsize: Figure size (width, height)
    
    Returns:
        matplotlib Figure object
    """
    if class_names is None:
        class_names = ['No Bleeding', 'Bleeding']
    
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title_suffix = ' (Normalized)'
    else:
        fmt = 'd'
        title_suffix = ''
    
    fig, ax = plt.subplots(figsize=figsize, dpi=FIGURE_DPI)
    
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count' if not normalize else 'Proportion'},
                linewidths=0.5, linecolor='gray',
                square=True, ax=ax)
    
    ax.set_title(f'Confusion Matrix{title_suffix} - {model_name}',
                fontsize=TITLE_SIZE, fontweight='bold', pad=15)
    ax.set_ylabel('True Label', fontsize=LABEL_SIZE, fontweight='medium')
    ax.set_xlabel('Predicted Label', fontsize=LABEL_SIZE, fontweight='medium')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"Confusion matrix saved to: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_metrics_comparison(metrics_dict: Dict[str, Dict[str, float]],
                           save_path: Optional[str] = None,
                           show_plot: bool = False,
                           figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
    """
    Plot bar chart comparing metrics across different models.
    
    Args:
        metrics_dict: Dictionary mapping model names to their metrics
        save_path: Path to save the figure
        show_plot: Whether to display the plot
        figsize: Figure size (width, height)
    
    Returns:
        matplotlib Figure object
    """
    models = list(metrics_dict.keys())
    metric_names = list(metrics_dict[models[0]].keys())
    
    x = np.arange(len(metric_names))
    width = 0.8 / len(models)
    
    fig, ax = plt.subplots(figsize=figsize, dpi=FIGURE_DPI)
    
    colors = [MEDICAL_COLORS['primary'], MEDICAL_COLORS['secondary'], 
              MEDICAL_COLORS['accent'], MEDICAL_COLORS['success']]
    
    for i, model in enumerate(models):
        values = [metrics_dict[model].get(metric, 0) for metric in metric_names]
        offset = (i - len(models) / 2 + 0.5) * width
        ax.bar(x + offset, values, width, label=model,
               color=colors[i % len(colors)], alpha=0.8, edgecolor='white', linewidth=1)
    
    ax.set_xlabel('Metrics', fontsize=LABEL_SIZE, fontweight='medium')
    ax.set_ylabel('Score', fontsize=LABEL_SIZE, fontweight='medium')
    ax.set_title('Model Performance Comparison',
                fontsize=TITLE_SIZE, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, rotation=45, ha='right')
    ax.set_ylim([0, 1.1])
    ax.legend(fontsize=LEGEND_SIZE, framealpha=0.9)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Add value labels on bars
    for i, model in enumerate(models):
        values = [metrics_dict[model].get(metric, 0) for metric in metric_names]
        offset = (i - len(models) / 2 + 0.5) * width
        for j, v in enumerate(values):
            ax.text(j + offset, v + 0.01, f'{v:.3f}', 
                   ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"Metrics comparison saved to: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_training_curves(history: Dict[str, List[float]],
                        save_path: Optional[str] = None,
                        show_plot: bool = False,
                        figsize: Tuple[int, int] = (14, 5)) -> plt.Figure:
    """
    Plot training and validation curves for loss and accuracy.
    
    Args:
        history: Dictionary with 'train_loss', 'val_loss', 'train_acc', 'val_acc' keys
        save_path: Path to save the figure
        show_plot: Whether to display the plot
        figsize: Figure size (width, height)
    
    Returns:
        matplotlib Figure object
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, dpi=FIGURE_DPI)
    
    # Plot loss
    ax1.plot(epochs, history['train_loss'], 
            color=MEDICAL_COLORS['primary'], 
            linewidth=2, 
            marker='o', 
            markersize=4,
            label='Training Loss',
            alpha=0.8)
    ax1.plot(epochs, history['val_loss'],
            color=MEDICAL_COLORS['secondary'],
            linewidth=2,
            marker='s',
            markersize=4,
            label='Validation Loss',
            alpha=0.8)
    
    _setup_plot_style(ax1, 'Training and Validation Loss',
                     'Epoch', 'Loss', grid=True, legend=True)
    
    # Plot accuracy
    ax2.plot(epochs, history['train_acc'],
            color=MEDICAL_COLORS['success'],
            linewidth=2,
            marker='o',
            markersize=4,
            label='Training Accuracy',
            alpha=0.8)
    ax2.plot(epochs, history['val_acc'],
            color=MEDICAL_COLORS['accent'],
            linewidth=2,
            marker='s',
            markersize=4,
            label='Validation Accuracy',
            alpha=0.8)
    
    _setup_plot_style(ax2, 'Training and Validation Accuracy',
                     'Epoch', 'Accuracy', grid=True, legend=True)
    ax2.set_ylim([0, 1.05])
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"Training curves saved to: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_calibration_curve(y_true: np.ndarray, y_scores: np.ndarray,
                          model_name: str = "Model",
                          n_bins: int = 10,
                          save_path: Optional[str] = None,
                          show_plot: bool = False,
                          figsize: Tuple[int, int] = (8, 6)) -> plt.Figure:
    """
    Plot calibration curve to assess prediction reliability.
    
    Args:
        y_true: True binary labels
        y_scores: Predicted probabilities
        model_name: Name of the model
        n_bins: Number of bins for calibration
        save_path: Path to save the figure
        show_plot: Whether to display the plot
        figsize: Figure size (width, height)
    
    Returns:
        matplotlib Figure object
    """
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_scores, n_bins=n_bins, strategy='uniform'
    )
    
    fig, ax = plt.subplots(figsize=figsize, dpi=FIGURE_DPI)
    
    ax.plot(mean_predicted_value, fraction_of_positives,
            color=MEDICAL_COLORS['primary'],
            linewidth=2.5,
            marker='o',
            markersize=6,
            label=model_name,
            alpha=0.9)
    
    # Perfect calibration line
    ax.plot([0, 1], [0, 1],
            color=MEDICAL_COLORS['neutral'],
            linestyle='--',
            linewidth=1.5,
            alpha=0.5,
            label='Perfect Calibration')
    
    _setup_plot_style(ax, 'Calibration Curve (Reliability Diagram)',
                     'Mean Predicted Probability',
                     'Fraction of Positives',
                     grid=True,
                     legend=True)
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"Calibration curve saved to: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_metric_distribution(metrics_list: List[Dict[str, float]],
                           metric_name: str,
                           model_names: Optional[List[str]] = None,
                           save_path: Optional[str] = None,
                           show_plot: bool = False,
                           figsize: Tuple[int, int] = (8, 6)) -> plt.Figure:
    """
    Plot distribution of a metric across multiple runs or folds.
    
    Args:
        metrics_list: List of metric dictionaries
        metric_name: Name of the metric to plot
        model_names: Optional list of model names
        save_path: Path to save the figure
        show_plot: Whether to display the plot
        figsize: Figure size (width, height)
    
    Returns:
        matplotlib Figure object
    """
    values = [m[metric_name] for m in metrics_list]
    
    fig, ax = plt.subplots(figsize=figsize, dpi=FIGURE_DPI)
    
    ax.hist(values, bins=15, color=MEDICAL_COLORS['primary'],
           alpha=0.7, edgecolor='white', linewidth=1.5)
    
    # Add mean line
    mean_val = np.mean(values)
    std_val = np.std(values)
    ax.axvline(mean_val, color=MEDICAL_COLORS['danger'],
              linestyle='--', linewidth=2,
              label=f'Mean = {mean_val:.3f} ± {std_val:.3f}')
    
    _setup_plot_style(ax, f'Distribution of {metric_name}',
                     metric_name, 'Frequency',
                     grid=True, legend=True)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"Metric distribution saved to: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_model_comparison(results_dict: Dict[str, Dict],
                         save_path: Optional[str] = None,
                         show_plot: bool = False,
                         figsize: Tuple[int, int] = (14, 8)) -> plt.Figure:
    """
    Create a comprehensive comparison plot with multiple metrics.
    
    Args:
        results_dict: Dictionary mapping model names to their results
        save_path: Path to save the figure
        show_plot: Whether to display the plot
        figsize: Figure size (width, height)
    
    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize, dpi=FIGURE_DPI)
    
    models = list(results_dict.keys())
    colors = [MEDICAL_COLORS['primary'], MEDICAL_COLORS['secondary'],
              MEDICAL_COLORS['accent'], MEDICAL_COLORS['success']]
    
    # Extract metrics
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        values = [results_dict[model].get(metric, 0) for model in models]
        
        bars = ax.bar(models, values, color=colors[:len(models)],
                     alpha=0.8, edgecolor='white', linewidth=1.5)
        
        ax.set_title(metric.capitalize(), fontsize=LABEL_SIZE, fontweight='bold')
        ax.set_ylabel('Score', fontsize=10)
        ax.set_ylim([0, 1.1])
        ax.grid(True, alpha=0.3, axis='y', linestyle='--', linewidth=0.5)
        ax.set_axisbelow(True)
        
        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.suptitle('Model Performance Comparison', fontsize=TITLE_SIZE,
                fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"Model comparison saved to: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return fig


def generate_all_visualizations(y_true: np.ndarray,
                               y_pred: np.ndarray,
                               y_scores: np.ndarray,
                               history: Optional[Dict[str, List[float]]] = None,
                               model_name: str = "Model",
                               output_dir: str = "research/visualizations/output",
                               class_names: Optional[List[str]] = None) -> Dict[str, str]:
    """
    Generate all visualizations and save them to output directory.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_scores: Prediction scores/probabilities
        history: Training history dictionary
        model_name: Name of the model
        output_dir: Directory to save visualizations
        class_names: List of class names
    
    Returns:
        Dictionary mapping visualization names to file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    
    saved_paths = {}
    
    # ROC curve
    roc_path = os.path.join(output_dir, f'roc_curve_{model_name.lower()}.png')
    plot_roc_curve(y_true, y_scores, model_name, roc_path)
    saved_paths['roc_curve'] = roc_path
    
    # PR curve
    pr_path = os.path.join(output_dir, f'pr_curve_{model_name.lower()}.png')
    plot_pr_curve(y_true, y_scores, model_name, pr_path)
    saved_paths['pr_curve'] = pr_path
    
    # Confusion matrix (raw)
    cm_path = os.path.join(output_dir, f'confusion_matrix_{model_name.lower()}.png')
    plot_confusion_matrix(y_true, y_pred, class_names, False, model_name, cm_path)
    saved_paths['confusion_matrix'] = cm_path
    
    # Confusion matrix (normalized)
    cm_norm_path = os.path.join(output_dir, f'confusion_matrix_normalized_{model_name.lower()}.png')
    plot_confusion_matrix(y_true, y_pred, class_names, True, model_name, cm_norm_path)
    saved_paths['confusion_matrix_normalized'] = cm_norm_path
    
    # Calibration curve
    cal_path = os.path.join(output_dir, f'calibration_curve_{model_name.lower()}.png')
    plot_calibration_curve(y_true, y_scores, model_name, save_path=cal_path)
    saved_paths['calibration_curve'] = cal_path
    
    # Training curves (if available)
    if history:
        train_path = os.path.join(output_dir, f'training_curves_{model_name.lower()}.png')
        plot_training_curves(history, train_path)
        saved_paths['training_curves'] = train_path
    
    print(f"\nAll visualizations saved to: {output_dir}")
    return saved_paths

