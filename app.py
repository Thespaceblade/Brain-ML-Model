"""
Streamlit Web Interface for Brain Bleeding Classification Model
Real-time testing and visualization interface
"""

import streamlit as st
import torch
import numpy as np
from PIL import Image
import os
import sys
import albumentations as A
from albumentations.pytorch import ToTensorV2
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.model import get_model
from src.utils import load_checkpoint, get_device

# Page configuration
st.set_page_config(
    page_title="Brain Bleeding Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .bleeding {
        background-color: #ffebee;
        border: 2px solid #f44336;
    }
    .no-bleeding {
        background-color: #e8f5e9;
        border: 2px solid #4caf50;
    }
    .metric-card {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'device' not in st.session_state:
    st.session_state.device = None
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'model_path' not in st.session_state:
    st.session_state.model_path = None


def inspect_checkpoint(model_path):
    """Inspect checkpoint file to see its structure"""
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        info = {
            'type': type(checkpoint).__name__,
            'keys': None,
            'size_mb': os.path.getsize(model_path) / (1024 * 1024)
        }
        
        if isinstance(checkpoint, dict):
            info['keys'] = list(checkpoint.keys())
            # Get additional info for common keys
            if 'epoch' in checkpoint:
                info['epoch'] = checkpoint['epoch']
            if 'accuracy' in checkpoint:
                info['accuracy'] = checkpoint['accuracy']
            if 'loss' in checkpoint:
                info['loss'] = checkpoint['loss']
        elif isinstance(checkpoint, torch.nn.Module):
            info['type'] = 'Model (direct)'
        else:
            info['type'] = f'{type(checkpoint).__name__} (unexpected)'
        
        return info
    except Exception as e:
        return {'error': str(e)}


def load_model_robust(model_path, model_name='resnet50'):
    """Load model with robust error handling for different checkpoint formats"""
    device = get_device()
    
    # Create model
    model = get_model(
        model_name=model_name,
        num_classes=2,
        pretrained=False
    )
    model = model.to(device)
    
    # Load checkpoint with format detection
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            # Format 1: Standard checkpoint with 'model_state_dict'
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            # Format 2: Direct state_dict
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            # Format 3: Checkpoint is the state_dict itself
            else:
                # Try to load as state_dict directly
                try:
                    model.load_state_dict(checkpoint)
                except Exception as e:
                    # If that fails, try to find any dict that looks like state_dict
                    loaded = False
                    for key in checkpoint.keys():
                        if isinstance(checkpoint[key], dict) and len(checkpoint[key]) > 0:
                            try:
                                model.load_state_dict(checkpoint[key])
                                loaded = True
                                break
                            except:
                                continue
                    if not loaded:
                        raise ValueError(f"Could not find model weights in checkpoint. Available keys: {list(checkpoint.keys())}")
        else:
            # Checkpoint might be state_dict directly
            model.load_state_dict(checkpoint)
        
        model.eval()
        return model, device
        
    except Exception as e:
        # Get checkpoint info for debugging
        checkpoint_info = inspect_checkpoint(model_path)
        error_details = f"Failed to load model checkpoint: {str(e)}\n\n"
        error_details += f"Checkpoint info: {checkpoint_info}\n\n"
        error_details += "Expected format: dict with 'model_state_dict' or 'state_dict' key, or direct state_dict."
        raise Exception(error_details)


def preprocess_image(image, img_size=224):
    """Preprocess image for model input"""
    # Convert PIL to numpy array
    if isinstance(image, Image.Image):
        image = np.array(image.convert('RGB'))
    
    # Apply transforms
    transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])
    
    transformed = transform(image=image)
    image_tensor = transformed['image'].unsqueeze(0)
    return image_tensor


def predict_image(model, image_tensor, device):
    """Make prediction on preprocessed image tensor"""
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    class_names = ['No Bleeding', 'Bleeding']
    prediction = class_names[predicted.item()]
    confidence_score = confidence.item()
    prob_array = probabilities[0].cpu().numpy()
    
    return prediction, confidence_score, prob_array


def create_probability_chart(probabilities):
    """Create a bar chart for class probabilities"""
    classes = ['No Bleeding', 'Bleeding']
    colors = ['#4caf50', '#f44336']
    
    fig = go.Figure(data=[
        go.Bar(
            x=classes,
            y=probabilities * 100,
            marker_color=colors,
            text=[f'{p*100:.2f}%' for p in probabilities],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title='Prediction Probabilities',
        xaxis_title='Class',
        yaxis_title='Probability (%)',
        yaxis=dict(range=[0, 100]),
        height=400,
        template='plotly_white'
    )
    
    return fig


def main():
    # Header
    st.markdown('<h1 class="main-header">🧠 Brain Bleeding Classifier</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar for model configuration
    with st.sidebar:
        st.header("⚙️ Model Configuration")
        
        # Model selection
        model_name = st.selectbox(
            "Model Architecture",
            options=['resnet50', 'efficientnet_b0', 'efficientnet_b1'],
            index=0,
            help="Select the model architecture"
        )
        
        # Model path input
        st.subheader("Load Model")
        model_path_input = st.text_input(
            "Model Path",
            value="models/best_model.pth",
            help="Path to the trained model checkpoint"
        )
        
        # Inspect checkpoint button
        if st.button("🔍 Inspect Checkpoint", use_container_width=True):
            if model_path_input and os.path.exists(model_path_input):
                try:
                    info = inspect_checkpoint(model_path_input)
                    st.json(info)
                except Exception as e:
                    st.error(f"Error inspecting checkpoint: {str(e)}")
            else:
                st.warning("Please enter a valid model path first")
        
        # Load model button
        if st.button("🔄 Load Model", type="primary", use_container_width=True):
            if os.path.exists(model_path_input):
                try:
                    with st.spinner("Loading model..."):
                        # Clear any cached model first
                        if 'model' in st.session_state and st.session_state.model is not None:
                            del st.session_state.model
                        
                        model, device = load_model_robust(model_path_input, model_name)
                        st.session_state.model = model
                        st.session_state.device = device
                        st.session_state.model_loaded = True
                        st.session_state.model_path = model_path_input
                    st.success(f"✅ Model loaded successfully!")
                    device_name = "GPU" if device.type == 'cuda' else "CPU"
                    st.info(f"🖥️ Running on: {device_name}")
                except Exception as e:
                    error_msg = str(e)
                    st.error(f"❌ Error loading model: {error_msg}")
                    with st.expander("🔍 Error Details", expanded=False):
                        st.code(error_msg)
                    st.session_state.model_loaded = False
                    # Clear failed model state
                    st.session_state.model = None
            else:
                st.error(f"❌ Model file not found: {model_path_input}")
                st.info(f"💡 Make sure the path is correct. Current working directory: {os.getcwd()}")
                st.session_state.model_loaded = False
        
        # Model status
        st.markdown("---")
        st.subheader("Model Status")
        if st.session_state.model_loaded:
            st.success("✅ Model Ready")
            if st.session_state.device:
                device_name = "GPU" if st.session_state.device.type == 'cuda' else "CPU"
                st.info(f"🖥️ Running on: {device_name}")
        else:
            st.warning("⚠️ Model Not Loaded")
            st.info("Please load a model to make predictions")
        
        # Image size setting
        st.markdown("---")
        st.subheader("Settings")
        img_size = st.slider(
            "Image Size",
            min_value=128,
            max_value=512,
            value=224,
            step=32,
            help="Input image size for the model"
        )
        
        # Clear cache button
        if st.button("🗑️ Clear Cache", use_container_width=True):
            st.cache_resource.clear()
            st.rerun()
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["📸 Single Image", "📁 Batch Processing", "ℹ️ About"])
    
    # Tab 1: Single Image Prediction
    with tab1:
        st.header("Single Image Prediction")
        
        if not st.session_state.model_loaded:
            st.warning("⚠️ Please load a model in the sidebar before making predictions.")
        else:
            # Image upload
            uploaded_file = st.file_uploader(
                "Upload a brain scan image",
                type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
                help="Upload a medical image (CT scan or MRI) for classification"
            )
            
            if uploaded_file is not None:
                # Display uploaded image
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.subheader("📷 Input Image")
                    image = Image.open(uploaded_file)
                    st.image(image, caption="Uploaded Image", use_container_width=True)
                
                with col2:
                    st.subheader("🔮 Prediction Results")
                    
                    # Predict button
                    if st.button("🔍 Predict", type="primary", use_container_width=True):
                        with st.spinner("Analyzing image..."):
                            # Preprocess image
                            image_tensor = preprocess_image(image, img_size)
                            
                            # Make prediction
                            prediction, confidence, probabilities = predict_image(
                                st.session_state.model,
                                image_tensor,
                                st.session_state.device
                            )
                            
                            # Display results
                            st.session_state.prediction = prediction
                            st.session_state.confidence = confidence
                            st.session_state.probabilities = probabilities
                    
                    # Show results if available
                    if 'prediction' in st.session_state:
                        prediction = st.session_state.prediction
                        confidence = st.session_state.confidence
                        probabilities = st.session_state.probabilities
                        
                        # Prediction box with color coding
                        if prediction == "Bleeding":
                            st.markdown(
                                f'<div class="prediction-box bleeding">'
                                f'<h2>⚠️ {prediction} Detected</h2>'
                                f'<h3>Confidence: {confidence*100:.2f}%</h3>'
                                f'</div>',
                                unsafe_allow_html=True
                            )
                        else:
                            st.markdown(
                                f'<div class="prediction-box no-bleeding">'
                                f'<h2>✅ {prediction}</h2>'
                                f'<h3>Confidence: {confidence*100:.2f}%</h3>'
                                f'</div>',
                                unsafe_allow_html=True
                            )
                        
                        # Probability chart
                        fig = create_probability_chart(probabilities)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Detailed metrics
                        st.subheader("📊 Detailed Probabilities")
                        col3, col4 = st.columns(2)
                        
                        with col3:
                            st.metric(
                                "No Bleeding",
                                f"{probabilities[0]*100:.2f}%",
                                delta=f"{probabilities[0]*100 - 50:.2f}%"
                            )
                        
                        with col4:
                            st.metric(
                                "Bleeding",
                                f"{probabilities[1]*100:.2f}%",
                                delta=f"{probabilities[1]*100 - 50:.2f}%"
                            )
    
    # Tab 2: Batch Processing
    with tab2:
        st.header("Batch Image Processing")
        
        if not st.session_state.model_loaded:
            st.warning("⚠️ Please load a model in the sidebar before processing images.")
        else:
            uploaded_files = st.file_uploader(
                "Upload multiple images",
                type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
                accept_multiple_files=True,
                help="Upload multiple images for batch processing"
            )
            
            if uploaded_files and len(uploaded_files) > 0:
                if st.button("🔍 Process All Images", type="primary", use_container_width=True):
                    results = []
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for idx, uploaded_file in enumerate(uploaded_files):
                        status_text.text(f"Processing {idx + 1}/{len(uploaded_files)}: {uploaded_file.name}")
                        
                        try:
                            # Load and preprocess image
                            image = Image.open(uploaded_file)
                            image_tensor = preprocess_image(image, img_size)
                            
                            # Make prediction
                            prediction, confidence, probabilities = predict_image(
                                st.session_state.model,
                                image_tensor,
                                st.session_state.device
                            )
                            
                            results.append({
                                'Image': uploaded_file.name,
                                'Prediction': prediction,
                                'Confidence': f"{confidence*100:.2f}%",
                                'No Bleeding Prob': f"{probabilities[0]*100:.2f}%",
                                'Bleeding Prob': f"{probabilities[1]*100:.2f}%"
                            })
                        except Exception as e:
                            results.append({
                                'Image': uploaded_file.name,
                                'Prediction': f"Error: {str(e)}",
                                'Confidence': "N/A",
                                'No Bleeding Prob': "N/A",
                                'Bleeding Prob': "N/A"
                            })
                        
                        progress_bar.progress((idx + 1) / len(uploaded_files))
                    
                    status_text.text("✅ Processing complete!")
                    
                    # Display results table
                    st.subheader("📋 Results")
                    import pandas as pd
                    df = pd.DataFrame(results)
                    st.dataframe(df, use_container_width=True)
                    
                    # Download results as CSV
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name="predictions.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                    
                    # Summary statistics
                    st.subheader("📊 Summary Statistics")
                    if len(results) > 0:
                        bleeding_count = sum(1 for r in results if r['Prediction'] == 'Bleeding')
                        no_bleeding_count = sum(1 for r in results if r['Prediction'] == 'No Bleeding')
                        
                        col5, col6 = st.columns(2)
                        with col5:
                            st.metric("Total Images", len(results))
                            st.metric("Bleeding Detected", bleeding_count)
                        with col6:
                            st.metric("No Bleeding", no_bleeding_count)
                            bleeding_percentage = (bleeding_count / len(results)) * 100 if results else 0
                            st.metric("Bleeding Rate", f"{bleeding_percentage:.1f}%")
    
    # Tab 3: About
    with tab3:
        st.header("About This Application")
        
        st.markdown("""
        ### 🧠 Brain Bleeding Classification Model
        
        This application provides a real-time interface for testing a deep learning model 
        that classifies brain medical images (CT scans or MRI) to detect bleeding.
        
        #### Features:
        - **Single Image Prediction**: Upload and analyze individual brain scan images
        - **Batch Processing**: Process multiple images at once
        - **Real-time Results**: Get instant predictions with confidence scores
        - **Visual Feedback**: Interactive charts and probability visualizations
        - **Model Selection**: Choose between different model architectures
        
        #### Model Architecture:
        The model uses transfer learning with:
        - **ResNet50**: Pre-trained on ImageNet, fine-tuned for brain bleeding classification
        - **EfficientNet**: Alternative efficient architecture options
        
        #### Usage:
        1. Load a trained model checkpoint in the sidebar
        2. Upload an image or multiple images
        3. Click "Predict" to get real-time classification results
        4. View detailed probabilities and confidence scores
        
        #### Technical Details:
        - Framework: PyTorch
        - Image Processing: Albumentations
        - Interface: Streamlit
        - Input Size: 224x224 pixels (configurable)
        - Output: Binary classification (Bleeding / No Bleeding)
        
        #### Disclaimer:
        This tool is for research and educational purposes only. 
        It should not be used as a substitute for professional medical diagnosis.
        """)


if __name__ == "__main__":
    main()

