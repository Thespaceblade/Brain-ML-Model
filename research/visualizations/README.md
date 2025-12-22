# Visualizations Module

This module provides publication-quality visualization functions for model evaluation metrics.

## Features

- **ROC Curves**: Receiver Operating Characteristic curves with AUC scores
- **PR Curves**: Precision-Recall curves with Average Precision scores
- **Confusion Matrices**: Both raw and normalized confusion matrices
- **Calibration Curves**: Reliability diagrams for probability calibration
- **Training Curves**: Loss and accuracy over training epochs
- **Metric Comparisons**: Bar charts comparing metrics across models
- **Metric Distributions**: Histograms of metric values across runs

## Usage

### Command Line

Generate all visualizations from a trained model:

```bash
python research/visualizations/generate_plots.py \
    --model_path models/best_model.pth \
    --model_name resnet50 \
    --data_dir data \
    --split test \
    --output_dir research/visualizations/output
```

### Python API

```python
from research.visualizations.plotter import (
    plot_roc_curve,
    plot_pr_curve,
    plot_confusion_matrix,
    generate_all_visualizations
)
import numpy as np

# Example: Generate ROC curve
y_true = np.array([0, 1, 1, 0, 1])
y_scores = np.array([0.1, 0.9, 0.8, 0.2, 0.95])

plot_roc_curve(
    y_true=y_true,
    y_scores=y_scores,
    model_name="ResNet50",
    save_path="research/visualizations/output/roc_curve.png"
)

# Generate all visualizations
saved_paths = generate_all_visualizations(
    y_true=y_true,
    y_pred=y_pred,
    y_scores=y_scores,
    model_name="ResNet50",
    output_dir="research/visualizations/output"
)
```

## Output Files

The module generates the following visualization files:

- `roc_curve_{model_name}.png` - ROC curve with AUC score
- `pr_curve_{model_name}.png` - Precision-Recall curve with AP score
- `confusion_matrix_{model_name}.png` - Raw confusion matrix
- `confusion_matrix_normalized_{model_name}.png` - Normalized confusion matrix
- `calibration_curve_{model_name}.png` - Calibration curve
- `training_curves_{model_name}.png` - Training history (if available)

## Customization

All plots use a consistent medical imaging color palette and publication-quality styling:

- High DPI (300) for publication
- Clean, minimal design
- Professional color scheme
- Consistent typography
- Grid lines for readability

## Requirements

- matplotlib
- seaborn
- numpy
- scikit-learn

