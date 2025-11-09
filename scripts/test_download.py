"""
Simple test script to download dataset from Kaggle.
This matches the basic download functionality.
"""

import kagglehub

# Download latest version
path = kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset")

print("Path to dataset files:", path)

