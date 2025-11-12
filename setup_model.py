"""
Setup script to download model files for deployment.
This script checks if model files exist locally, and if not, 
attempts to download them from cloud storage.
"""

import os
import urllib.request
import hashlib

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


def setup_model(model_url=None):
    """Setup model file for deployment.
    
    Args:
        model_url: Optional URL to download model from. If not provided, will check:
            - Streamlit secrets (MODEL_URL)
            - Environment variable (MODEL_URL)
    """
    # Check if model already exists
    if check_model_exists(MODEL_PATH):
        print(f"Model file already exists at {MODEL_PATH}")
        file_size = os.path.getsize(MODEL_PATH) / (1024 * 1024)
        print(f"Model size: {file_size:.2f} MB")
        return True
    
    # Get model URL from parameter, Streamlit secrets, or environment variable
    url = model_url
    if not url:
        try:
            import streamlit as st
            # Check if we're in a Streamlit context and secrets are available
            if hasattr(st, 'secrets') and 'MODEL_URL' in st.secrets:
                url = st.secrets['MODEL_URL']
        except:
            pass
        
        # Fall back to environment variable
        if not url:
            url = os.getenv("MODEL_URL", "")
    
    # Try to download from URL if provided
    if url:
        print(f"Model file not found. Attempting to download from {url}...")
        if download_model(url, MODEL_PATH):
            return True
        else:
            print("Failed to download model from URL.")
            return False
    else:
        print(f"Model file not found at {MODEL_PATH}")
        print("To download model automatically, set MODEL_URL in Streamlit secrets or as environment variable.")
        print("Alternatively, ensure the model file is in the repository or use Git LFS.")
        return False


if __name__ == "__main__":
    setup_model()

