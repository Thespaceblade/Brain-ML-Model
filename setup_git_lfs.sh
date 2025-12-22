#!/bin/bash
# Setup script for Git LFS to track model files

echo "Setting up Git LFS for model files..."
echo ""

# Check if Git LFS is installed
if ! command -v git-lfs &> /dev/null; then
    echo "❌ Git LFS is not installed!"
    echo "Please install Git LFS first:"
    echo "  macOS: brew install git-lfs"
    echo "  Linux: sudo apt-get install git-lfs"
    echo "  Windows: Download from https://git-lfs.github.com/"
    exit 1
fi

echo "✓ Git LFS is installed"
echo ""

# Initialize Git LFS
echo "Initializing Git LFS..."
git lfs install
echo ""

# Track model files
echo "Tracking model files..."
git lfs track "*.pth"
git lfs track "*.pt"
git lfs track "*.ckpt"
git lfs track "models/*"
echo ""

# Add .gitattributes
echo "Adding .gitattributes..."
git add .gitattributes
echo ""

# Check if model file exists
if [ -f "models/best_model.pth" ]; then
    echo "Found model file: models/best_model.pth"
    echo "Adding model file to Git LFS..."
    git add models/best_model.pth
    echo ""
    
    # Show file status
    echo "Checking Git LFS status..."
    git lfs ls-files
    echo ""
    
    echo "[SUCCESS] Setup complete!"
    echo ""
    echo "Next steps:"
    echo "1. Review changes: git status"
    echo "2. Commit changes: git commit -m 'Add model file with Git LFS'"
    echo "3. Push to repository: git push"
    echo ""
    echo "After pushing, the model file will be stored in Git LFS"
    echo "and will be available in Streamlit Cloud deployment."
else
    echo "[WARNING] Model file not found: models/best_model.pth"
    echo "Please ensure the model file exists before running this script."
    echo ""
    echo "Setup complete, but model file needs to be added manually:"
    echo "  git add models/best_model.pth"
    echo "  git commit -m 'Add model file with Git LFS'"
    echo "  git push"
fi



