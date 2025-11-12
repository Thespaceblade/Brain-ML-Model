"""
Setup script to download model files for deployment.
This script checks if model files exist locally, and if not, 
attempts to download them from cloud storage.
"""

import os
import urllib.request
import hashlib

MODEL_URL = os.getenv("MODEL_URL", "")  # Set via environment variable or Streamlit secrets
MODEL_PATH = "models/best_model.pth"
MODEL_DIR = "models"


def download_model(url, destination):
    """Download model file from URL."""
    print(f"Downloading model from {url}...")
    try:
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        urllib.request.urlretrieve(url, destination)
        print(f"Model downloaded successfully to {destination}")
        return True
    except Exception as e:
        print(f"Error downloading model: {str(e)}")
        return False


def check_model_exists(path):
    """Check if model file exists and is valid."""
    if not os.path.exists(path):
        return False
    
    # Check file size (should be > 0)
    file_size = os.path.getsize(path)
    if file_size == 0:
        return False
    
    # Model file should be at least 1MB (basic validation)
    if file_size < 1024 * 1024:
        return False
    
    return True


def setup_model():
    """Setup model file for deployment."""
    # Check if model already exists
    if check_model_exists(MODEL_PATH):
        print(f"Model file already exists at {MODEL_PATH}")
        file_size = os.path.getsize(MODEL_PATH) / (1024 * 1024)
        print(f"Model size: {file_size:.2f} MB")
        return True
    
    # Try to download from URL if provided
    if MODEL_URL:
        print(f"Model file not found. Attempting to download from {MODEL_URL}...")
        if download_model(MODEL_URL, MODEL_PATH):
            return True
        else:
            print("Failed to download model from URL.")
            return False
    else:
        print(f"Model file not found at {MODEL_PATH}")
        print("To download model automatically, set MODEL_URL environment variable.")
        print("Alternatively, ensure the model file is in the repository or use Git LFS.")
        return False


if __name__ == "__main__":
    setup_model()

