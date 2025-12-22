# Brain Bleeding Classification Model

A deep learning model for classifying brain bleeding from medical MRI images using CNNs with transfer learning. This project uses ResNet50 and EfficientNet architectures to classify brain scans as either showing bleeding/tumors or no bleeding.

## Links

- [GitHub Repository](https://github.com/Thespaceblade/Brain-ML-Model)
- [Portfolio](https://jasonindata.vercel.app)

## Features

- **Transfer Learning**: Pre-trained ImageNet weights for better performance
- **Multiple Architectures**: Support for ResNet50 and EfficientNet variants
- **Data Augmentation**: Albumentations for robust training
- **Interactive Web Interface**: Streamlit-based UI for real-time predictions
- **Model Checkpointing**: Automatic saving of best models during training
- **Early Stopping**: Prevents overfitting
- **Batch Processing**: Support for processing multiple images at once
- **Comprehensive Metrics**: Accuracy, Precision, Recall, F1-score tracking

## Requirements

- Python 3.8+
- PyTorch 2.0+
- See `requirements.txt` for full dependencies

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Thespaceblade/Brain-ML-Model.git
cd Brain-ML-Model
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare Dataset

The model uses the [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) from Kaggle. The dataset is automatically downloaded and prepared using the provided script.

#### Option A: Automatic Download (Recommended)

```bash
python -c "import kagglehub; path = kagglehub.dataset_download('masoudnickparvar/brain-tumor-mri-dataset'); print(f'Dataset downloaded to: {path}')"
```

Then prepare the data:

```bash
python scripts/prepare_kaggle_data.py --kaggle_path ~/.cache/kagglehub/datasets/masoudnickparvar/brain-tumor-mri-dataset/versions/1 --output_dir data
```

#### Option B: Manual Download

1. Download the dataset from Kaggle
2. Extract it to a directory
3. Run the preparation script:

```bash
python scripts/prepare_kaggle_data.py --kaggle_path /path/to/dataset --output_dir data
```

The script will:
- Map tumor classes (glioma, meningioma, pituitary) → "bleeding"
- Map "notumor" → "no_bleeding"
- Split Training data into train/val (70/15)
- Use Testing data as test set

**Expected Data Structure:**
```
data/
├── train/
│   ├── bleeding/
│   └── no_bleeding/
├── val/
│   ├── bleeding/
│   └── no_bleeding/
└── test/
    ├── bleeding/
    └── no_bleeding/
```

## Usage

### Training

Train the model with default settings:

```bash
python scripts/train.py --data_dir data --epochs 50 --batch_size 32
```

**Training Options:**
- `--data_dir`: Path to data directory (default: `data`)
- `--epochs`: Number of training epochs (default: `50`)
- `--batch_size`: Batch size (default: `32`)
- `--model`: Model architecture - `resnet50`, `efficientnet_b0`, `efficientnet_b1` (default: `resnet50`)
- `--learning_rate`: Learning rate (default: `0.001`)
- `--optimizer`: Optimizer - `adam`, `sgd` (default: `adam`)
- `--img_size`: Image size (default: `224`)
- `--use_albumentations`: Use Albumentations for augmentation (default: `True`)

**Example:**
```bash
python scripts/train.py \
    --data_dir data \
    --epochs 100 \
    --batch_size 16 \
    --model resnet50 \
    --learning_rate 0.0001 \
    --img_size 224
```

### Evaluation

Evaluate a trained model:

```bash
python scripts/evaluate.py --model_path models/best_model.pth --data_dir data/test
```

### Prediction on Single Image

```bash
python scripts/predict.py --model_path models/best_model.pth --image_path path/to/image.jpg
```

### Web Interface

Launch the interactive Streamlit web interface:

```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501`

**Web Interface Features:**
- Upload single or multiple images
- Real-time predictions with confidence scores
- Visual probability charts
- Model information and statistics
- Batch processing support

## Model Architecture

The model uses transfer learning with the following architectures:

### ResNet50
- Pre-trained on ImageNet
- Custom classifier head with dropout layers
- Binary classification output (bleeding/no_bleeding)

### EfficientNet
- Support for EfficientNet-B0, B1, B2 variants
- Pre-trained ImageNet weights
- Optimized for efficiency and accuracy

**Model Specifications:**
- Input Size: 224x224 pixels (configurable)
- Output: 2 classes (bleeding, no_bleeding)
- Normalization: ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

## 📁 Project Structure

```
Brain-ML-Model/
├── app.py                      # Streamlit web interface
├── setup_model.py              # Model setup utilities
├── evaluate.py                 # Model evaluation script
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── scripts/
│   ├── train.py               # Training script
│   ├── predict.py              # Single image prediction
│   ├── prepare_kaggle_data.py # Dataset preparation
│   ├── download_dataset.py    # Dataset download utility
│   ├── split_data.py          # Data splitting utility
│   └── preprocess_data.py     # Data preprocessing
├── src/
│   ├── model.py               # Model definitions
│   ├── data_loader.py         # Data loading and transforms
│   └── utils.py               # Utility functions
├── models/                     # Saved model checkpoints
│   ├── best_model.pth         # Best model (by validation accuracy)
│   └── final_model.pth        # Final model after training
├── logs/                       # Training logs and plots
└── data/                       # Dataset (not in repo)
    ├── train/
    ├── val/
    └── test/
```

## Model Performance

The model achieves the following performance metrics:

- **Training Accuracy**: ~85%
- **Validation Accuracy**: ~96%
- **Test Accuracy**: ~91%
- **Precision**: ~95%
- **Recall**: ~92%
- **F1-Score**: ~93%

*Note: Performance may vary based on training configuration and dataset.*

## Configuration

### Image Requirements

- **Formats**: PNG, JPG/JPEG, BMP, TIFF
- **Size**: Any size (automatically resized to 224x224)
- **Color**: RGB (grayscale converted automatically)
- **Content**: Brain MRI/CT scan images

### Training Configuration

Key hyperparameters can be adjusted in `scripts/train.py`:
- Learning rate scheduling (ReduceLROnPlateau or StepLR)
- Early stopping patience
- Weight decay
- Dropout rates

## Deployment

### Streamlit Cloud

The app can be deployed to Streamlit Cloud. See deployment documentation for details.

**Requirements:**
- Model file uploaded to cloud storage
- Streamlit Cloud account
- Repository connected to Streamlit Cloud

### Local Deployment

1. Ensure all dependencies are installed
2. Download or train a model
3. Run `streamlit run app.py`
4. Access via `http://localhost:8501`

## Notes

- The model is trained on brain tumor MRI data, mapped to bleeding/no_bleeding classification
- For production use, ensure proper medical validation and regulatory compliance
- Model performance depends on image quality and similarity to training data
- Large model files are stored using Git LFS

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available for educational and research purposes.

## Acknowledgments

- Dataset: [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) by Masoud Nickparvar
- PyTorch and torchvision for deep learning framework
- Streamlit for web interface
- Albumentations for data augmentation
