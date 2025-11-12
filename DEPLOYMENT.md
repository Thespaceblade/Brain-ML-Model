# Deployment Guide for Streamlit App

## Issues Fixed

### 1. Model Loading Issues
- **Problem**: Model files were not loading correctly in Streamlit Cloud deployments
- **Root Cause**: 
  - Model files are ignored by git (`.gitignore` includes `models/` and `*.pth`)
  - Path resolution issues in cloud environments
  - No caching mechanism for model loading
  - No proper error handling for missing files

### 2. Solutions Implemented

#### a. Path Resolution
- Added `resolve_model_path()` function that tries multiple possible locations:
  - Original path (as provided)
  - Relative to `app.py` file
  - Relative to current working directory
- Provides clear error messages when model file is not found
- Shows current working directory and app directory for debugging

#### b. Streamlit Caching
- Added `@st.cache_resource` decorator to `load_model_robust()` function
- Caches model in memory to avoid reloading on every rerun
- Uses absolute paths as cache keys for consistency across environments
- Cache can be manually cleared using "Clear Cache" button

#### c. Auto-Load Functionality
- Automatically loads model if it exists at the default path (`models/best_model.pth`)
- Only attempts auto-load once per session
- Falls back to manual loading if auto-load fails
- Shows clear status messages for model loading state

#### d. Better Error Handling
- Comprehensive error messages with file path information
- Shows current working directory and app directory when file is not found
- Provides tips for Streamlit Cloud deployment
- Detailed error messages in expandable sections

## Deployment Requirements

### For Streamlit Cloud Deployment

#### Option 1: Include Model in Repository (Recommended for Small Models)

1. **Remove model files from `.gitignore`**:
   ```bash
   # Edit .gitignore and remove or comment out:
   # models/
   # *.pth
   ```

2. **Add model files to git**:
   ```bash
   git add models/best_model.pth
   git commit -m "Add model file for deployment"
   git push
   ```

3. **Note**: This works well for smaller models (< 100MB). For larger models, use Option 2.

#### Option 2: Use Git LFS (Recommended for Large Models)

1. **Install Git LFS**:
   ```bash
   git lfs install
   ```

2. **Track model files with Git LFS**:
   ```bash
   git lfs track "*.pth"
   git lfs track "models/*"
   ```

3. **Add and commit**:
   ```bash
   git add .gitattributes
   git add models/best_model.pth
   git commit -m "Add model file with Git LFS"
   git push
   ```

#### Option 3: Download Model During Deployment

1. **Create a setup script** (`setup_model.py`):
   ```python
   import os
   import urllib.request
   
   def download_model():
       model_url = "YOUR_MODEL_URL"  # e.g., Google Drive, S3, etc.
       model_path = "models/best_model.pth"
       
       os.makedirs("models", exist_ok=True)
       
       if not os.path.exists(model_path):
           print("Downloading model...")
           urllib.request.urlretrieve(model_url, model_path)
           print("Model downloaded successfully!")
   
   if __name__ == "__main__":
       download_model()
   ```

2. **Add to `requirements.txt`**:
   ```
   # Add any additional dependencies needed for downloading
   ```

3. **Modify `app.py` to run setup on first load**:
   ```python
   # At the top of app.py, after imports
   import setup_model
   setup_model.download_model()
   ```

#### Option 4: Use Cloud Storage (Recommended for Production)

1. **Upload model to cloud storage** (e.g., AWS S3, Google Cloud Storage, Azure Blob):
   ```python
   # Example: Download from S3
   import boto3
   
   def download_model_from_s3():
       s3 = boto3.client('s3')
       s3.download_file('your-bucket', 'models/best_model.pth', 'models/best_model.pth')
   ```

2. **Add credentials as Streamlit secrets**:
   - Go to Streamlit Cloud settings
   - Add secrets in `.streamlit/secrets.toml` format
   - Access in app: `st.secrets["aws_access_key_id"]`

### For Local Deployment

1. **Ensure model file exists**:
   ```bash
   # Check if model exists
   ls -lh models/best_model.pth
   ```

2. **Run the app**:
   ```bash
   streamlit run app.py
   ```

3. **If model is not found**:
   - Check the path in the sidebar (default: `models/best_model.pth`)
   - Verify the model file exists in the correct location
   - Update the path in the sidebar if needed

## Troubleshooting

### Model File Not Found

**Error**: `Model file not found at: models/best_model.pth`

**Solutions**:
1. Check if the model file exists in the repository
2. Verify the path is correct (relative to `app.py`)
3. Check current working directory in the error message
4. Ensure model file is committed to git (if using Streamlit Cloud)

### Model Loading Errors

**Error**: `Failed to load model checkpoint`

**Solutions**:
1. Check model file format (should be PyTorch `.pth` file)
2. Verify model architecture matches (ResNet50, EfficientNet, etc.)
3. Check if model was trained with the same number of classes (2)
4. Inspect checkpoint using "Inspect Checkpoint" button

### Cache Issues

**Problem**: Model not updating after changes

**Solutions**:
1. Click "Clear Cache" button in the sidebar
2. Restart the Streamlit app
3. Clear browser cache if issues persist

### Path Resolution Issues

**Problem**: Model found locally but not in deployment

**Solutions**:
1. Use absolute paths in deployment
2. Ensure model file is in the repository root or same directory as `app.py`
3. Check Streamlit Cloud file structure
4. Use environment variables for model paths if needed

## Best Practices

1. **Model File Size**:
   - Use Git LFS for models > 100MB
   - Consider model quantization for smaller file sizes
   - Use cloud storage for very large models

2. **Security**:
   - Never commit API keys or secrets
   - Use Streamlit secrets for sensitive data
   - Validate model files before loading

3. **Performance**:
   - Use `@st.cache_resource` for model loading
   - Cache preprocessed images if processing multiple times
   - Use GPU if available (Streamlit Cloud provides CPU by default)

4. **Error Handling**:
   - Always provide clear error messages
   - Show helpful tips for common issues
   - Log errors for debugging

## Testing Deployment

1. **Test locally first**:
   ```bash
   streamlit run app.py
   ```

2. **Test with different paths**:
   - Try absolute paths
   - Try relative paths
   - Test with missing files

3. **Test on Streamlit Cloud**:
   - Deploy to Streamlit Cloud
   - Test model loading
   - Verify predictions work correctly

## Additional Resources

- [Streamlit Cloud Documentation](https://docs.streamlit.io/streamlit-community-cloud)
- [Git LFS Documentation](https://git-lfs.github.com/)
- [PyTorch Model Saving/Loading](https://pytorch.org/tutorials/beginner/saving_loading_models.html)

