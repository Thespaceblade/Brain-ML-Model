# Setup Instructions

## Git Setup

If git is not installed or not in your PATH, follow these steps:

### 1. Install Git (if not installed)
- Download from: https://git-scm.com/download/win
- Install and make sure to add git to PATH during installation

### 2. Initialize Git Repository

Once git is installed, run these commands in the project directory:

```bash
cd "C:\Users\jason\Personal Code Projects\Brain-ML-Model"

# Initialize git
git init

# Add remote repository
git remote add origin https://github.com/Thespaceblade/Brain-ML-Model.git

# Verify remote
git remote -v

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit: Brain bleeding classification ML model"

# Push to GitHub (you may need to authenticate)
git branch -M main
git push -u origin main
```

Alternatively, you can run the `setup_git.bat` script if you're on Windows.

### 3. GitHub Authentication

If you haven't set up GitHub authentication, you may need to:
- Create a Personal Access Token (PAT) on GitHub
- Use it when prompted for password
- Or set up SSH keys for authentication

## Project Setup

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download Dataset:**
   ```bash
   python scripts/download_dataset.py --output_dir data
   ```

3. **Train Model:**
   ```bash
   python scripts/train.py --data_dir data --epochs 50 --batch_size 32
   ```

## Dataset Information

The project uses the Brain Tumor MRI Dataset from Kaggle:
- Dataset: `masoudnickparvar/brain-tumor-mri-dataset`
- Download script: `scripts/download_dataset.py`
- The script automatically organizes the data into train/val/test splits

## Troubleshooting

### Git not found
- Make sure Git is installed and added to PATH
- Restart your terminal after installing Git
- On Windows, you may need to use Git Bash instead of PowerShell

### Kaggle API issues
- Make sure `kagglehub` is installed: `pip install kagglehub`
- Some datasets may require Kaggle API credentials
- Check Kaggle documentation for authentication setup

### Dataset download issues
- Check your internet connection
- Verify the dataset name is correct
- Some datasets may require accepting terms on Kaggle first



