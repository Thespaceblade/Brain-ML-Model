# Quick Deployment Guide - Fix Model Loading Issue

## The Problem
Your model file (281MB) exists locally but is not in the Git repository because `.gitignore` excludes it. Streamlit Cloud can't find the model file, so the app can't load it.

## Quick Fix Options

### Option 1: Use Git LFS (Recommended - Best for Production)

Git LFS allows you to store large files in Git without bloating the repository.

#### Step 1: Install Git LFS
```bash
# On macOS (using Homebrew)
brew install git-lfs

# On Windows (download from https://git-lfs.github.com/)
# Or using Chocolatey
choco install git-lfs

# On Linux
sudo apt-get install git-lfs
```

#### Step 2: Initialize Git LFS in your repository
```bash
cd "/Users/jasoncharwin/Personal Code Projects/Brain ML Model/Brain-ML-Model"
git lfs install
```

#### Step 3: Track model files
```bash
git lfs track "*.pth"
git lfs track "models/*"
```

#### Step 4: Create/Update .gitattributes
```bash
git add .gitattributes
```

#### Step 5: Add and commit model file
```bash
git add models/best_model.pth
git commit -m "Add model file with Git LFS"
git push
```

#### Step 6: Verify
```bash
git lfs ls-files
# Should show: models/best_model.pth
```

### Option 2: Upload to Google Drive (Quick Fix)

#### Step 1: Upload model to Google Drive
1. Go to Google Drive
2. Upload `models/best_model.pth`
3. Right-click → Share → Get shareable link
4. Copy the link

#### Step 2: Convert to direct download link
Replace the link format:
- From: `https://drive.google.com/file/d/FILE_ID/view?usp=sharing`
- To: `https://drive.google.com/uc?export=download&id=FILE_ID`

#### Step 3: Set in Streamlit Cloud
1. Go to Streamlit Cloud dashboard
2. Click on your app
3. Go to "Settings" → "Secrets"
4. Add:
```toml
MODEL_URL = "https://drive.google.com/uc?export=download&id=YOUR_FILE_ID"
```

#### Step 4: Update app.py (already done)
The app will automatically download the model if `MODEL_URL` is set.

### Option 3: Manual Upload in App (Temporary Fix)

1. Deploy the app to Streamlit Cloud (without model file)
2. Open the app
3. Go to sidebar → "Load Model" section
4. If model not found, scroll down to "Upload Model File"
5. Upload your `best_model.pth` file
6. The model will be saved to `models/` directory
7. Enter the path and click "Load Model"

**Note**: This is temporary - the model will be lost if the app restarts or redeploys.

### Option 4: Remove from .gitignore (Not Recommended)

⚠️ **Warning**: This will add a 281MB file directly to Git, which is not recommended.

```bash
# Edit .gitignore and remove these lines:
# models/
# *.pth

# Then add the model
git add models/best_model.pth
git commit -m "Add model file"
git push
```

## Recommended Workflow

1. **Use Git LFS** (Option 1) for version control and deployment
2. **Use Cloud Storage** (Option 2) as a backup download source
3. **Keep manual upload** (Option 3) as a fallback option

## Testing Locally

After setting up Git LFS, test locally:

```bash
# Clone the repository fresh (to test LFS)
cd /tmp
git clone YOUR_REPO_URL test-clone
cd test-clone
# Model should be downloaded automatically via LFS
ls -lh models/best_model.pth
```

## Verifying Deployment

After deploying to Streamlit Cloud:

1. Check the app logs for any errors
2. Try to load the model in the sidebar
3. If it fails, check:
   - Is the model file in the repository? (Check on GitHub)
   - Is Git LFS working? (Check `.gitattributes` file exists)
   - Are there any error messages in the app?

## Troubleshooting

### Git LFS not working
```bash
# Check if LFS is installed
git lfs version

# Re-initialize LFS
git lfs install

# Check tracked files
git lfs ls-files

# If model isn't tracked, re-add it
git lfs track "*.pth"
git add .gitattributes
git add models/best_model.pth
git commit -m "Re-add model with LFS"
git push
```

### Model still not found in Streamlit Cloud
1. Check GitHub - is the file visible in the repository?
2. Check file size - is it showing as a pointer file (small) or actual file?
3. Check Streamlit Cloud logs for errors
4. Verify the path in the app matches the repository structure

### Download from Cloud Storage fails
1. Verify the URL is correct and accessible
2. Check if the file requires authentication
3. Try accessing the URL directly in a browser
4. Check Streamlit Cloud logs for download errors

## Next Steps

1. Choose one of the options above
2. Implement it
3. Test locally first
4. Deploy to Streamlit Cloud
5. Verify the model loads correctly

## Need Help?

- Check `DEPLOYMENT.md` for detailed instructions
- Check Streamlit Cloud logs for errors
- Verify Git LFS is working: `git lfs ls-files`
- Check file paths match in repository and app

