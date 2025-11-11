# Visual Interface Guide

## Quick Start

1. **Install Dependencies** (if not already installed):
   ```bash
   pip install -r requirements.txt
   ```

2. **Launch the Interface**:
   ```bash
   streamlit run app.py
   ```

3. **Open in Browser**:
   - The terminal will display a URL (typically `http://localhost:8501`)
   - Open this URL in your web browser

## Features

### 🎯 Single Image Prediction
- Upload a single brain scan image (PNG, JPG, JPEG, BMP, TIFF)
- Get instant prediction with confidence scores
- View detailed probability breakdown
- Interactive probability charts

### 📁 Batch Processing
- Upload multiple images at once
- Process all images with progress tracking
- View results in a sortable table
- Download results as CSV file
- Summary statistics

### ⚙️ Model Configuration
- Select model architecture (ResNet50, EfficientNet)
- Load trained model checkpoints
- Adjust image size settings
- View model status and device information

## Usage Tips

1. **First Time Setup**:
   - Make sure you have a trained model checkpoint (`.pth` file)
   - Default path is `models/best_model.pth`
   - You can specify a custom path in the sidebar

2. **Making Predictions**:
   - Load a model first using the sidebar
   - Upload an image in the "Single Image" tab
   - Click "Predict" to get results
   - Results show prediction, confidence, and probability breakdown

3. **Batch Processing**:
   - Load a model first
   - Go to "Batch Processing" tab
   - Upload multiple images
   - Click "Process All Images"
   - Download results as CSV if needed

4. **Model Selection**:
   - Choose the architecture that matches your trained model
   - If unsure, try ResNet50 first (most common)

## Troubleshooting

- **Model not loading**: Check that the model path is correct and the file exists
- **Import errors**: Make sure all dependencies are installed (`pip install -r requirements.txt`)
- **GPU not detected**: The app will automatically use CPU if GPU is not available
- **Image upload issues**: Ensure images are in supported formats (PNG, JPG, JPEG, BMP, TIFF)

## Technical Details

- **Framework**: Streamlit
- **Visualization**: Plotly for interactive charts
- **Image Processing**: Albumentations (same as training pipeline)
- **Device**: Automatically detects and uses GPU if available


