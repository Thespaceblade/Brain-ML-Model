# Streamlit Cloud Deployment - Model Setup Guide

## Quick Setup Steps

To get your model to load automatically on Streamlit Cloud, you have **3 options**:

### Option 1: Use Git LFS (Recommended for Large Models)

1. **Install Git LFS** (if not already installed):
   ```bash
   git lfs install
   ```

2. **Track model files**:
   ```bash
   git lfs track "*.pth"
   git lfs track "models/*"
   git add .gitattributes
   ```

3. **Add and commit your model**:
   ```bash
   git add models/best_model.pth
   git commit -m "Add model with Git LFS"
   git push
   ```

4. **Deploy to Streamlit Cloud** - the model will be included automatically!

### Option 2: Download from Cloud Storage (Recommended for Production)

1. **Upload your model to cloud storage**:
   - **Google Drive**: Upload `models/best_model.pth` and get a shareable link
   - **Dropbox**: Upload and get a direct download link
   - **AWS S3**: Upload to a public bucket
   - **Any other cloud storage** with direct download URLs

2. **Convert Google Drive link** (if using Google Drive):
   - Original: `https://drive.google.com/file/d/FILE_ID/view?usp=sharing`
   - Convert to: `https://drive.google.com/uc?export=download&id=FILE_ID`

3. **Set MODEL_URL in Streamlit Cloud Secrets**:
   - Go to your Streamlit Cloud dashboard
   - Click on your app
   - Go to **Settings** → **Secrets**
   - Add this to the secrets:
     ```toml
     MODEL_URL = "https://your-cloud-storage-url.com/path/to/model.pth"
     ```
   - Or for Google Drive:
     ```toml
     MODEL_URL = "https://drive.google.com/uc?export=download&id=YOUR_FILE_ID"
     ```

4. **Deploy/Redeploy** - the app will automatically download the model on first load!

### Option 3: Manual Upload (Temporary - Not Recommended)

1. Deploy the app without the model
2. Open the app in Streamlit Cloud
3. Use the sidebar → "Upload Model File" feature
4. Upload your `best_model.pth` file
5. The model will be saved temporarily (will be lost on restart)

## How It Works

The app automatically checks for the model file when it starts:

1. **First**: Checks if `models/best_model.pth` exists locally
2. **If not found**: Checks Streamlit secrets for `MODEL_URL`
3. **If URL found**: Downloads the model automatically
4. **If no URL**: Shows a message to set up the model

## Testing Locally

Before deploying, test the setup locally:

1. **Test with Git LFS**:
   ```bash
   # Make sure model is tracked
   git lfs ls-files
   # Should show: models/best_model.pth
   ```

2. **Test with MODEL_URL**:
   ```bash
   # Set environment variable
   export MODEL_URL="https://your-model-url.com/model.pth"
   # Run the app
   streamlit run app.py
   ```

## Troubleshooting

### Model Not Downloading

- **Check Streamlit Cloud logs** for error messages
- **Verify the URL** is accessible (try opening it in a browser)
- **Check file size** - very large files (>500MB) may timeout
- **Verify secrets** are set correctly in Streamlit Cloud dashboard

### Model Download Fails

- **Google Drive**: Make sure you converted the link format correctly
- **Authentication**: Some cloud storage requires authentication - use public links or direct download URLs
- **File size**: Consider using Git LFS for very large models

### Model Loads But Predictions Fail

- **Check model architecture** matches what you trained with
- **Verify model file** is not corrupted
- **Check logs** for specific error messages

## Best Practices

1. **For models < 100MB**: Use Git LFS
2. **For models > 100MB**: Use cloud storage with MODEL_URL
3. **For production**: Use cloud storage (more reliable)
4. **Always test locally** before deploying

## Next Steps

After setting up the model:

1. Deploy to Streamlit Cloud
2. Check the app logs to verify model download
3. Test the model loading in the sidebar
4. Try making a prediction to verify everything works

