# Brain Bleeding Classification Model

This project implements a deep learning model to classify brain bleeding from medical images (CT scans/MRI) using Convolutional Neural Networks (CNNs) with transfer learning.

## Project Structure

```
Brain-ML-Model/
├── data/
│   ├── train/
│   │   ├── bleeding/
│   │   └── no_bleeding/
│   ├── val/
│   │   ├── bleeding/
│   │   └── no_bleeding/
│   └── test/
│       ├── bleeding/
│       └── no_bleeding/
├── models/
│   └── (saved model checkpoints)
├── logs/
│   └── (training logs and plots)
├── results/
│   └── (evaluation results)
├── scripts/
│   ├── download_dataset.py
│   ├── preprocess_data.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── app.py
├── src/
│   ├── model.py
│   ├── data_loader.py
│   └── utils.py
├── requirements.txt
├── README.md
├── SETUP.md
└── setup_git.bat
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download the dataset from Kaggle:
```bash
python scripts/download_dataset.py --output_dir data
```
This will download the brain tumor MRI dataset from Kaggle and organize it into the required structure.

Alternatively, you can manually organize your dataset:
   - Place training images in `data/train/bleeding/` and `data/train/no_bleeding/`
   - Place validation images in `data/val/bleeding/` and `data/val/no_bleeding/`
   - Place test images in `data/test/bleeding/` and `data/test/no_bleeding/`

3. (Optional) Preprocess and split data:
```bash
python scripts/preprocess_data.py --input_dir <path_to_data> --output_dir data
```

## Usage

### Download Dataset
```bash
python scripts/download_dataset.py --output_dir data
```

### Preprocess Data
```bash
python scripts/preprocess_data.py --input_dir <path_to_data> --output_dir data
```

### Train Model
```bash
python scripts/train.py --data_dir data --epochs 50 --batch_size 32 --learning_rate 0.001
```

### Evaluate Model
```bash
python scripts/evaluate.py --model_path models/best_model.pth --data_dir data/test
```

### Predict on New Images
```bash
python scripts/predict.py --model_path models/best_model.pth --image_path path/to/image.jpg
```

### Visual Interface (Real-time Testing)
Launch the interactive web interface for real-time testing and visualization:
```bash
streamlit run app.py
```

The visual interface provides:
- **Single Image Prediction**: Upload and analyze individual brain scan images with real-time results
- **Batch Processing**: Process multiple images at once with detailed results table
- **Interactive Visualizations**: View probability charts and confidence scores
- **Model Selection**: Choose between different model architectures (ResNet50, EfficientNet)
- **Real-time Feedback**: Get instant predictions with visual indicators

Open your browser to the URL shown in the terminal (typically `http://localhost:8501`)

## Model Architecture

The model uses a ResNet50 backbone (transfer learning) with custom classification head for binary classification (bleeding vs no bleeding).

## Features

- Data augmentation for better generalization
- Transfer learning with pre-trained ImageNet weights
- Model checkpointing and early stopping
- Comprehensive evaluation metrics
- Support for both CPU and GPU training
- **Interactive web interface** for real-time testing and visualization
- Batch processing capabilities
- Visual feedback with probability charts and confidence scores

