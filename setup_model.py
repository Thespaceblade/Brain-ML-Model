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
    
    Returns:
        bool: True if model exists or was successfully downloaded, False otherwise
    """
    try:
        # Check if model already exists
        if check_model_exists(MODEL_PATH):
            file_size = os.path.getsize(MODEL_PATH) / (1024 * 1024)
            return True
        
        # Get model URL from parameter, Streamlit secrets, or environment variable
        url = model_url
        if not url:
            try:
                import streamlit as st
                # Check if we're in a Streamlit context and secrets are available
                if hasattr(st, 'secrets') and st.secrets and 'MODEL_URL' in st.secrets:
                    url = st.secrets.get('MODEL_URL', '')
            except (ImportError, AttributeError, TypeError):
                # Not in Streamlit context or secrets not available
                pass
            
            # Fall back to environment variable
            if not url:
                url = os.getenv("MODEL_URL", "")
        
        # Try to download from URL if provided
        if url:
            if download_model(url, MODEL_PATH):
                return True
            else:
                return False
        else:
            return False
    except Exception as e:
        # Silently fail - don't break the app if setup fails
        return False


if __name__ == "__main__":
    setup_model()

