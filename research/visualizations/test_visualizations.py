"""
Test script to generate sample visualizations with synthetic data.
Useful for testing visualization functions without running full model evaluation.
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

import numpy as np
from research.visualizations.plotter import (
    plot_roc_curve,
    plot_pr_curve,
    plot_confusion_matrix,
    plot_metrics_comparison,
    plot_training_curves,
    plot_calibration_curve,
    plot_model_comparison,
    generate_all_visualizations
)
import os

# Create output directory (relative to this file)
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, "output", "test")
os.makedirs(output_dir, exist_ok=True)

print("Generating test visualizations with synthetic data...")

# Generate synthetic data
np.random.seed(42)
n_samples = 1000

# True labels
y_true = np.random.binomial(1, 0.4, n_samples)

# Prediction scores (simulate a good model)
y_scores = np.random.beta(2, 5, n_samples)
y_scores[y_true == 1] = np.random.beta(5, 2, np.sum(y_true == 1))
y_scores = np.clip(y_scores, 0.01, 0.99)

# Predictions
y_pred = (y_scores > 0.5).astype(int)

# Training history
history = {
    'train_loss': np.exp(-np.linspace(0, 2, 50)) + np.random.normal(0, 0.02, 50),
    'val_loss': np.exp(-np.linspace(0, 1.8, 50)) + np.random.normal(0, 0.02, 50),
    'train_acc': 1 - np.exp(-np.linspace(0, 2, 50)) + np.random.normal(0, 0.02, 50),
    'val_acc': 1 - np.exp(-np.linspace(0, 1.8, 50)) + np.random.normal(0, 0.02, 50)
}

# Clip to valid ranges
history['train_loss'] = np.clip(history['train_loss'], 0, 2)
history['val_loss'] = np.clip(history['val_loss'], 0, 2)
history['train_acc'] = np.clip(history['train_acc'], 0, 1)
history['val_acc'] = np.clip(history['val_acc'], 0, 1)

print("\n1. Generating ROC curve...")
plot_roc_curve(y_true, y_scores, "Test Model", 
               os.path.join(output_dir, "test_roc_curve.png"))

print("2. Generating PR curve...")
plot_pr_curve(y_true, y_scores, "Test Model",
              os.path.join(output_dir, "test_pr_curve.png"))

print("3. Generating confusion matrix...")
plot_confusion_matrix(y_true, y_pred, ["No Bleeding", "Bleeding"],
                     False, "Test Model",
                     os.path.join(output_dir, "test_confusion_matrix.png"))

print("4. Generating normalized confusion matrix...")
plot_confusion_matrix(y_true, y_pred, ["No Bleeding", "Bleeding"],
                     True, "Test Model",
                     os.path.join(output_dir, "test_confusion_matrix_normalized.png"))

print("5. Generating calibration curve...")
plot_calibration_curve(y_true, y_scores, "Test Model",
                      save_path=os.path.join(output_dir, "test_calibration_curve.png"))

print("6. Generating training curves...")
plot_training_curves(history,
                     save_path=os.path.join(output_dir, "test_training_curves.png"))

print("7. Generating metrics comparison...")
metrics_dict = {
    "ResNet50": {"accuracy": 0.92, "precision": 0.89, "recall": 0.91, "f1": 0.90},
    "EfficientNet-B0": {"accuracy": 0.90, "precision": 0.87, "recall": 0.89, "f1": 0.88},
    "EfficientNet-B1": {"accuracy": 0.91, "precision": 0.88, "recall": 0.90, "f1": 0.89}
}
plot_metrics_comparison(metrics_dict,
                       save_path=os.path.join(output_dir, "test_metrics_comparison.png"))

print("8. Generating model comparison...")
results_dict = {
    "ResNet50": {"accuracy": 0.92, "precision": 0.89, "recall": 0.91, "f1": 0.90},
    "EfficientNet-B0": {"accuracy": 0.90, "precision": 0.87, "recall": 0.89, "f1": 0.88}
}
plot_model_comparison(results_dict,
                     save_path=os.path.join(output_dir, "test_model_comparison.png"))

print("9. Generating all visualizations at once...")
generate_all_visualizations(
    y_true=y_true,
    y_pred=y_pred,
    y_scores=y_scores,
    history=history,
    model_name="TestModel",
    output_dir=os.path.join(output_dir, "all_plots"),
    class_names=["No Bleeding", "Bleeding"]
)

print(f"\nAll test visualizations generated successfully!")
print(f"Output directory: {output_dir}")

