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
from src.utils import get_device

# Try to setup model file if it doesn't exist (for deployment)
try:
    import setup_model
    # Only run setup if model doesn't exist
    if not os.path.exists("models/best_model.pth"):
        setup_model.setup_model()
except Exception as e:
    # Silently fail if setup_model doesn't work - not critical
    pass

# Page configuration
st.set_page_config(
    page_title="Brain Bleeding Classifier",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    /* Global animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    
    @keyframes slideIn {
        from { transform: translateX(-20px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    @keyframes shimmer {
        0% { background-position: 200% 0; }
        100% { background-position: -200% 0; }
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    @keyframes shake {
        0%, 100% { transform: translateX(0); }
        25% { transform: translateX(-5px); }
        75% { transform: translateX(5px); }
    }
    
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Main header with animation */
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(135deg, #1f77b4 0%, #42a5f5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 2rem;
        animation: fadeIn 1s ease-in;
    }
    
    /* Prediction boxes with hover effects */
    .prediction-box {
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        animation: fadeIn 0.6s ease-in;
        cursor: pointer;
        position: relative;
        overflow: hidden;
    }
    
    .prediction-box::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent);
        transition: left 0.5s;
    }
    
    .prediction-box:hover::before {
        left: 100%;
    }
    
    .prediction-box:hover {
        transform: translateY(-5px) scale(1.02);
        box-shadow: 0 12px 24px rgba(0, 0, 0, 0.2);
    }
    
    /* Bleeding prediction box */
    .bleeding {
        background: linear-gradient(135deg, #ffcdd2 0%, #ffb3ba 100%);
        border: 3px solid #d32f2f;
    }
    
    .bleeding:hover {
        background: linear-gradient(135deg, #ffb3ba 0%, #ff9fa6 100%);
        border-color: #b71c1c;
        animation: pulse 0.6s ease-in-out;
    }
    
    .bleeding h2 {
        color: #b71c1c !important;
        font-weight: 800 !important;
        text-shadow: 2px 2px 4px rgba(255, 255, 255, 0.8);
        font-size: 2rem !important;
        margin-bottom: 0.5rem;
        transition: all 0.3s ease;
    }
    
    .bleeding:hover h2 {
        transform: scale(1.05);
        text-shadow: 3px 3px 6px rgba(255, 255, 255, 0.9);
    }
    
    .bleeding h3 {
        color: #c62828 !important;
        font-weight: 700 !important;
        text-shadow: 1px 1px 2px rgba(255, 255, 255, 0.6);
        font-size: 1.5rem !important;
        transition: all 0.3s ease;
    }
    
    .bleeding:hover h3 {
        transform: translateX(5px);
    }
    
    /* No bleeding prediction box */
    .no-bleeding {
        background: linear-gradient(135deg, #c8e6c9 0%, #a5d6a7 100%);
        border: 3px solid #388e3c;
    }
    
    .no-bleeding:hover {
        background: linear-gradient(135deg, #a5d6a7 0%, #81c784 100%);
        border-color: #2e7d32;
        animation: pulse 0.6s ease-in-out;
    }
    
    .no-bleeding h2 {
        color: #1b5e20 !important;
        font-weight: 800 !important;
        text-shadow: 2px 2px 4px rgba(255, 255, 255, 0.8);
        font-size: 2rem !important;
        margin-bottom: 0.5rem;
        transition: all 0.3s ease;
    }
    
    .no-bleeding:hover h2 {
        transform: scale(1.05);
        text-shadow: 3px 3px 6px rgba(255, 255, 255, 0.9);
    }
    
    .no-bleeding h3 {
        color: #2e7d32 !important;
        font-weight: 700 !important;
        text-shadow: 1px 1px 2px rgba(255, 255, 255, 0.6);
        font-size: 1.5rem !important;
        transition: all 0.3s ease;
    }
    
    .no-bleeding:hover h3 {
        transform: translateX(5px);
    }
    
    /* Metric cards with hover effects */
    .metric-card {
        background: linear-gradient(135deg, #f5f5f5 0%, #e8e8e8 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
        border: 2px solid transparent;
    }
    
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.15);
        border-color: #1f77b4;
        background: linear-gradient(135deg, #ffffff 0%, #f0f0f0 100%);
    }
    
    /* Labels with hover effects */
    .bleeding-label {
        color: #b71c1c !important;
        font-weight: 700 !important;
        transition: all 0.3s ease;
        display: inline-block;
        cursor: pointer;
    }
    
    .bleeding-label:hover {
        transform: scale(1.1);
        text-shadow: 2px 2px 4px rgba(183, 28, 28, 0.3);
    }
    
    .no-bleeding-label {
        color: #1b5e20 !important;
        font-weight: 700 !important;
        transition: all 0.3s ease;
        display: inline-block;
        cursor: pointer;
    }
    
    .no-bleeding-label:hover {
        transform: scale(1.1);
        text-shadow: 2px 2px 4px rgba(27, 94, 32, 0.3);
    }
    
    /* Button hover effects */
    .stButton > button {
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        border-radius: 8px;
        font-weight: 600;
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 0;
        height: 0;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.3);
        transform: translate(-50%, -50%);
        transition: width 0.6s, height 0.6s;
    }
    
    .stButton > button:hover::before {
        width: 300px;
        height: 300px;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.2);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* File uploader styling */
    .stFileUploader > div {
        transition: all 0.3s ease;
        border-radius: 10px;
    }
    
    .stFileUploader > div:hover {
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        transform: translateY(-2px);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e1e1e 0%, #2d2d2d 100%);
    }
    
    /* Sidebar text styling */
    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4 {
        color: #ffffff !important;
    }
    
    /* Sidebar input fields */
    [data-testid="stSidebar"] .stTextInput > div > div > input {
        background-color: #3d3d3d !important;
        color: #ffffff !important;
        border-color: #555555 !important;
    }
    
    [data-testid="stSidebar"] .stTextInput > div > div > input:focus {
        border-color: #1f77b4 !important;
        background-color: #4d4d4d !important;
    }
    
    /* Sidebar selectbox */
    [data-testid="stSidebar"] .stSelectbox > div > div {
        background-color: #3d3d3d !important;
        color: #ffffff !important;
    }
    
    /* Sidebar button styling */
    [data-testid="stSidebar"] .stButton > button {
        background-color: #1f77b4 !important;
        color: #ffffff !important;
    }
    
    [data-testid="stSidebar"] .stButton > button:hover {
        background-color: #1565c0 !important;
    }
    
    /* Sidebar info/warning/error boxes */
    [data-testid="stSidebar"] .stSuccess,
    [data-testid="stSidebar"] .stInfo,
    [data-testid="stSidebar"] .stWarning,
    [data-testid="stSidebar"] .stError {
        background-color: rgba(255, 255, 255, 0.1) !important;
        border-color: rgba(255, 255, 255, 0.2) !important;
    }
    
    /* Sidebar markdown text */
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div {
        color: #e0e0e0 !important;
    }
    
    /* Sidebar code blocks */
    [data-testid="stSidebar"] code {
        background-color: #2d2d2d !important;
        color: #ffffff !important;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        transition: all 0.3s ease;
        border-radius: 8px 8px 0 0;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(31, 119, 180, 0.1);
        transform: translateY(-2px);
    }
    
    /* Metric display styling */
    [data-testid="stMetricValue"] {
        transition: all 0.3s ease;
    }
    
    [data-testid="stMetricContainer"] {
        transition: all 0.3s ease;
        border-radius: 8px;
        padding: 0.5rem;
    }
    
    [data-testid="stMetricContainer"]:hover {
        background-color: rgba(31, 119, 180, 0.05);
        transform: translateY(-3px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    
    [data-testid="stMetricContainer"]:hover [data-testid="stMetricValue"] {
        transform: scale(1.05);
    }
    
    /* Image display with hover effect */
    .stImage {
        transition: all 0.3s ease;
        border-radius: 10px;
        overflow: hidden;
    }
    
    .stImage img {
        transition: all 0.3s ease;
    }
    
    .stImage:hover {
        transform: scale(1.02);
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
    }
    
    .stImage:hover img {
        transform: scale(1.05);
    }
    
    /* Dataframe styling */
    .dataframe {
        transition: all 0.3s ease;
        border-radius: 8px;
    }
    
    .dataframe:hover {
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    
    /* Progress bar animation */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #1f77b4, #42a5f5, #1f77b4);
        background-size: 200% 100%;
        animation: shimmer 2s infinite;
    }
    
    /* Spinner animation */
    .stSpinner > div {
        border-color: #1f77b4 transparent #1f77b4 transparent;
        animation: spin 1s linear infinite;
    }
    
    /* Success/Warning/Error message animations */
    .stSuccess {
        animation: slideIn 0.5s ease-out;
        transition: all 0.3s ease;
    }
    
    .stSuccess:hover {
        transform: translateX(5px);
        box-shadow: 0 4px 8px rgba(76, 175, 80, 0.3);
    }
    
    .stWarning {
        animation: shake 0.5s ease-out;
        transition: all 0.3s ease;
    }
    
    .stWarning:hover {
        transform: translateX(5px);
        box-shadow: 0 4px 8px rgba(255, 152, 0, 0.3);
    }
    
    .stError {
        animation: shake 0.5s ease-out;
        transition: all 0.3s ease;
    }
    
    .stError:hover {
        transform: translateX(5px);
        box-shadow: 0 4px 8px rgba(244, 67, 54, 0.3);
    }
    
    /* Info boxes */
    .stInfo {
        transition: all 0.3s ease;
        border-radius: 8px;
    }
    
    .stInfo:hover {
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        transform: translateX(5px);
    }
    
    /* Subheader styling */
    h2, h3 {
        transition: all 0.3s ease;
    }
    
    h2:hover, h3:hover {
        color: #1f77b4;
        transform: translateX(5px);
    }
    
    /* Selectbox and input styling */
    .stSelectbox > div > div {
        transition: all 0.3s ease;
    }
    
    .stSelectbox > div > div:hover {
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        transform: translateY(-2px);
    }
    
    .stTextInput > div > div > input {
        transition: all 0.3s ease;
        border-radius: 6px;
    }
    
    .stTextInput > div > div > input:focus {
        box-shadow: 0 0 0 3px rgba(31, 119, 180, 0.2);
        border-color: #1f77b4;
        transform: scale(1.02);
    }
    
    /* Chart container hover */
    .js-plotly-plot {
        transition: all 0.3s ease;
        border-radius: 10px;
    }
    
    .js-plotly-plot:hover {
        transform: scale(1.01);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    
    /* Header links */
    .header-links {
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .header-links a {
        color: #1f77b4;
        text-decoration: none;
        font-weight: 500;
        font-size: 0.9rem;
        transition: all 0.3s ease;
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 6px;
    }
    
    .header-links a:hover {
        color: #1565c0;
        text-decoration: underline;
        background-color: rgba(31, 119, 180, 0.1);
        transform: translateY(-2px);
    }
    
    /* Smooth scrolling */
    html {
        scroll-behavior: smooth;
    }
    
    /* Loading state */
    .element-container {
        animation: fadeInUp 0.6s ease-out;
    }
    
    /* Download button hover */
    .stDownloadButton > button {
        transition: all 0.3s ease;
    }
    
    .stDownloadButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        transition: all 0.3s ease;
    }
    
    .streamlit-expanderHeader:hover {
        background-color: rgba(31, 119, 180, 0.05);
        transform: translateX(5px);
    }
    
    /* JSON display */
    .stJson {
        transition: all 0.3s ease;
        border-radius: 8px;
    }
    
    .stJson:hover {
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
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
if 'auto_load_attempted' not in st.session_state:
    st.session_state.auto_load_attempted = False


def resolve_model_path(model_path):
    """
    Resolve model path by trying multiple possible locations.
    
    Args:
        model_path: Path to model file (can be relative or absolute)
    
    Returns:
        Resolved absolute path if found, None otherwise
    """
    if not model_path:
        return None
    
    # If absolute path and exists, return it
    if os.path.isabs(model_path) and os.path.exists(model_path):
        return model_path
    
    # Try multiple possible locations
    possible_paths = [
        model_path,  # Original path
        os.path.join(os.path.dirname(os.path.abspath(__file__)), model_path),  # Relative to app.py
        os.path.join(os.getcwd(), model_path),  # Relative to current working directory
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return os.path.abspath(path)
    
    return None


def inspect_checkpoint(model_path):
    """Inspect checkpoint file to see its structure"""
    # Resolve path first
    resolved_path = resolve_model_path(model_path)
    if not resolved_path:
        return {'error': f'Model file not found at: {model_path}'}
    
    try:
        checkpoint = torch.load(resolved_path, map_location='cpu')
        info = {
            'type': type(checkpoint).__name__,
            'keys': None,
            'size_mb': os.path.getsize(resolved_path) / (1024 * 1024),
            'path': resolved_path
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
        return {'error': str(e), 'path': resolved_path}


@st.cache_resource
def load_model_robust(_model_path_abs, model_name='resnet50'):
    """
    Load model with robust error handling for different checkpoint formats.
    Uses Streamlit caching to avoid reloading the model on every rerun.
    
    Note: The model_path parameter is prefixed with _ to indicate it's used for cache invalidation.
    We use the absolute path as the cache key to ensure consistency across environments.
    
    Args:
        _model_path_abs: Absolute path to model checkpoint file (must be resolved before calling)
        model_name: Name of model architecture
    
    Returns:
        Tuple of (model, device)
    """
    # Validate that the path exists (should be resolved before calling this function)
    if not os.path.exists(_model_path_abs):
        raise FileNotFoundError(
            f"Model file not found at: {_model_path_abs}\n\n"
            f"Please check the file path and ensure the model file exists."
        )
    
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
        # Use map_location to ensure compatibility across devices
        checkpoint = torch.load(_model_path_abs, map_location=device)
        
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
        try:
            checkpoint_info = inspect_checkpoint(_model_path_abs)
            error_details = f"Failed to load model checkpoint: {str(e)}\n\n"
            error_details += f"Checkpoint info: {checkpoint_info}\n\n"
            error_details += "Expected format: dict with 'model_state_dict' or 'state_dict' key, or direct state_dict."
        except:
            error_details = f"Failed to load model checkpoint: {str(e)}\n\n"
            error_details += f"Model path: {_model_path_abs}\n"
            error_details += f"File exists: {os.path.exists(_model_path_abs)}\n"
            if os.path.exists(_model_path_abs):
                error_details += f"File size: {os.path.getsize(_model_path_abs) / (1024*1024):.2f} MB\n"
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
    """Create a bar chart for class probabilities with enhanced interactivity"""
    classes = ['No Bleeding', 'Bleeding']
    # Higher contrast colors - darker greens and reds
    colors = ['#2e7d32', '#c62828']  # Dark green and dark red
    hover_colors = ['#1b5e20', '#b71c1c']  # Darker for hover
    
    fig = go.Figure(data=[
        go.Bar(
            x=classes,
            y=probabilities * 100,
            marker_color=colors,
            text=[f'{p*100:.2f}%' for p in probabilities],
            textposition='auto',
            textfont=dict(
                size=16,
                color='white',
                family='Arial Black'
            ),
            marker_line=dict(
                color='white',
                width=2
            ),
            marker_line_color='white',
            hovertemplate='<b>%{x}</b><br>Probability: %{y:.2f}%<extra></extra>',
            hoverlabel=dict(
                bgcolor='rgba(255, 255, 255, 0.9)',
                font_size=14,
                font_family='Arial Black'
            ),
        )
    ])
    
    fig.update_layout(
        title=dict(
            text='Prediction Probabilities',
            font=dict(size=20, color='#333', family='Arial Black')
        ),
        xaxis_title=dict(
            text='Class',
            font=dict(size=14, color='#333', family='Arial')
        ),
        yaxis_title=dict(
            text='Probability (%)',
            font=dict(size=14, color='#333', family='Arial')
        ),
        yaxis=dict(range=[0, 100]),
        height=400,
        template='plotly_white',
        font=dict(family='Arial', size=12),
        xaxis=dict(
            tickfont=dict(size=13, color='#333', family='Arial Black')
        ),
        hovermode='closest',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=False
    )
    
    return fig


def main():
    # Header
    st.markdown('<h1 class="main-header">🧠 Brain Bleeding Classifier</h1>', unsafe_allow_html=True)
    
    # Links subheadings
    col_links1, col_links2 = st.columns(2)
    with col_links1:
        st.markdown('<div class="header-links"><p><a href="https://github.com/Thespaceblade/Brain-ML-Model" target="_blank">GitHub Repository</a></p></div>', unsafe_allow_html=True)
    with col_links2:
        st.markdown('<div class="header-links"><p><a href="https://jasonindata.vercel.app" target="_blank">Portfolio</a></p></div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Sidebar for model configuration
    with st.sidebar:
        st.header("Model Configuration")
        
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
        
        # Try to resolve the path and show status
        resolved_path = resolve_model_path(model_path_input) if model_path_input else None
        if resolved_path:
            st.success(f"✓ Model found: {resolved_path}")
        elif model_path_input:
            st.warning(f"⚠ Model not found at: {model_path_input}")
            st.info(f"Current directory: {os.getcwd()}\nApp directory: {os.path.dirname(os.path.abspath(__file__))}")
        
        # Auto-load model if it exists and hasn't been loaded yet (only once)
        if (resolved_path and 
            not st.session_state.model_loaded and 
            not st.session_state.auto_load_attempted and
            (st.session_state.model_path is None or st.session_state.model_path != resolved_path)):
            st.session_state.auto_load_attempted = True
            try:
                with st.spinner("Auto-loading model..."):
                    # Clear cache to ensure fresh load
                    load_model_robust.clear()
                    # Pass resolved absolute path to cached function
                    model, device = load_model_robust(resolved_path, model_name)
                    st.session_state.model = model
                    st.session_state.device = device
                    st.session_state.model_loaded = True
                    st.session_state.model_path = resolved_path
                st.success("✓ Model auto-loaded successfully!")
                device_name = "GPU" if device.type == 'cuda' else "CPU"
                st.info(f"Running on: {device_name}")
            except Exception as e:
                # Show a warning but don't block the app
                st.warning(f"Auto-load failed. Please load manually. Error: {str(e)}")
                st.session_state.model_loaded = False
        
        # Inspect checkpoint button
        if st.button("Inspect Checkpoint", use_container_width=True):
            if resolved_path:
                try:
                    info = inspect_checkpoint(model_path_input)
                    st.json(info)
                except Exception as e:
                    st.error(f"Error inspecting checkpoint: {str(e)}")
            else:
                st.warning("Please enter a valid model path first")
        
        # Load model button
        if st.button("Load Model", type="primary", use_container_width=True):
            if resolved_path:
                try:
                    with st.spinner("Loading model..."):
                        # Clear cache to force reload
                        load_model_robust.clear()
                        
                        # Pass resolved absolute path to cached function
                        model, device = load_model_robust(resolved_path, model_name)
                        st.session_state.model = model
                        st.session_state.device = device
                        st.session_state.model_loaded = True
                        st.session_state.model_path = resolved_path
                    st.success(f"Model loaded successfully!")
                    device_name = "GPU" if device.type == 'cuda' else "CPU"
                    st.info(f"Running on: {device_name}")
                except Exception as e:
                    error_msg = str(e)
                    st.error(f"Error loading model: {error_msg}")
                    with st.expander("Error Details", expanded=False):
                        st.code(error_msg)
                    st.session_state.model_loaded = False
                    # Clear failed model state
                    st.session_state.model = None
                    st.session_state.device = None
            else:
                st.error(f"Model file not found: {model_path_input}")
                st.info(f"**Current directory**: {os.getcwd()}\n**App directory**: {os.path.dirname(os.path.abspath(__file__))}")
                
                # Show deployment instructions
                with st.expander("📋 How to fix this in Streamlit Cloud", expanded=True):
                    st.markdown("""
                    ### Option 1: Use Git LFS (Recommended for large models)
                    1. Install Git LFS: `git lfs install`
                    2. Track model files: `git lfs track "*.pth"` and `git lfs track "models/*"`
                    3. Add `.gitattributes`: `git add .gitattributes`
                    4. Add model file: `git add models/best_model.pth`
                    5. Commit and push: `git commit -m "Add model with LFS" && git push`
                    
                    ### Option 2: Upload to Cloud Storage
                    1. Upload model to Google Drive, Dropbox, or S3
                    2. Set `MODEL_URL` in Streamlit Cloud secrets
                    3. The app will automatically download the model
                    
                    ### Option 3: Remove from .gitignore (Not recommended for large files)
                    1. Edit `.gitignore` and remove `models/` and `*.pth`
                    2. Add model: `git add models/best_model.pth`
                    3. Commit and push (may take time for large files)
                    
                    ### Option 4: Manual Upload
                    Use the file uploader below to upload your model file directly.
                    """)
                
                # Add file uploader for model file
                st.markdown("---")
                st.subheader("Upload Model File")
                uploaded_model = st.file_uploader(
                    "Upload your model file (.pth)",
                    type=['pth'],
                    help="Upload your trained model checkpoint file"
                )
                
                if uploaded_model is not None:
                    # Save uploaded model
                    try:
                        os.makedirs("models", exist_ok=True)
                        model_save_path = os.path.join("models", uploaded_model.name)
                        with open(model_save_path, "wb") as f:
                            f.write(uploaded_model.getbuffer())
                        st.success(f"Model saved to {model_save_path}")
                        st.info("Please enter the model path above and click 'Load Model'")
                        # Update the default path
                        if model_path_input == "models/best_model.pth":
                            st.info(f"💡 Try using path: {model_save_path}")
                    except Exception as e:
                        st.error(f"Error saving model: {str(e)}")
                
                st.session_state.model_loaded = False
        
        # Model status
        st.markdown("---")
        st.subheader("Model Status")
        if st.session_state.model_loaded:
            st.success("Model Ready")
            if st.session_state.device:
                device_name = "GPU" if st.session_state.device.type == 'cuda' else "CPU"
                st.info(f"Running on: {device_name}")
        else:
            st.warning("Model Not Loaded")
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
        if st.button("Clear Cache", use_container_width=True):
            st.cache_resource.clear()
            st.rerun()
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["Single Image", "Batch Processing", "About"])
    
    # Tab 1: Single Image Prediction
    with tab1:
        st.header("Single Image Prediction")
        
        if not st.session_state.model_loaded:
            st.warning("Please load a model in the sidebar before making predictions.")
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
                    st.subheader("Input Image")
                    image = Image.open(uploaded_file)
                    st.image(image, caption="Uploaded Image", use_container_width=True)
                
                with col2:
                    st.subheader("Prediction Results")
                    
                    # Predict button
                    if st.button("Predict", type="primary", use_container_width=True):
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
                                f'<h2>{prediction} Detected</h2>'
                                f'<h3>Confidence: {confidence*100:.2f}%</h3>'
                                f'</div>',
                                unsafe_allow_html=True
                            )
                        else:
                            st.markdown(
                                f'<div class="prediction-box no-bleeding">'
                                f'<h2>{prediction}</h2>'
                                f'<h3>Confidence: {confidence*100:.2f}%</h3>'
                                f'</div>',
                                unsafe_allow_html=True
                            )
                        
                        # Probability chart
                        fig = create_probability_chart(probabilities)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Detailed metrics
                        st.subheader("Detailed Probabilities")
                        col3, col4 = st.columns(2)
                        
                        with col3:
                            st.markdown(
                                '<p class="no-bleeding-label" style="font-size: 1.2rem; margin-bottom: 0.5rem;">No Bleeding</p>',
                                unsafe_allow_html=True
                            )
                            st.metric(
                                "",
                                f"{probabilities[0]*100:.2f}%",
                                delta=f"{probabilities[0]*100 - 50:.2f}%",
                                delta_color="normal"
                            )
                        
                        with col4:
                            st.markdown(
                                '<p class="bleeding-label" style="font-size: 1.2rem; margin-bottom: 0.5rem;">Bleeding</p>',
                                unsafe_allow_html=True
                            )
                            st.metric(
                                "",
                                f"{probabilities[1]*100:.2f}%",
                                delta=f"{probabilities[1]*100 - 50:.2f}%",
                                delta_color="normal"
                            )
    
    # Tab 2: Batch Processing
    with tab2:
        st.header("Batch Image Processing")
        
        if not st.session_state.model_loaded:
            st.warning("Please load a model in the sidebar before processing images.")
        else:
            uploaded_files = st.file_uploader(
                "Upload multiple images",
                type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
                accept_multiple_files=True,
                help="Upload multiple images for batch processing"
            )
            
            if uploaded_files and len(uploaded_files) > 0:
                if st.button("Process All Images", type="primary", use_container_width=True):
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
                    
                    status_text.text("Processing complete!")
                    
                    # Display results table
                    st.subheader("Results")
                    import pandas as pd
                    df = pd.DataFrame(results)
                    st.dataframe(df, use_container_width=True)
                    
                    # Download results as CSV
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Download Results as CSV",
                        data=csv,
                        file_name="predictions.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                    
                    # Summary statistics
                    st.subheader("Summary Statistics")
                    if len(results) > 0:
                        bleeding_count = sum(1 for r in results if r['Prediction'] == 'Bleeding')
                        no_bleeding_count = sum(1 for r in results if r['Prediction'] == 'No Bleeding')
                        
                        col5, col6 = st.columns(2)
                        with col5:
                            st.metric("Total Images", len(results))
                            st.markdown(
                                '<p class="bleeding-label" style="font-size: 1.1rem; margin-top: 1rem; margin-bottom: 0.3rem;">Bleeding Detected</p>',
                                unsafe_allow_html=True
                            )
                            st.metric("", bleeding_count)
                        with col6:
                            st.markdown(
                                '<p class="no-bleeding-label" style="font-size: 1.1rem; margin-top: 1rem; margin-bottom: 0.3rem;">No Bleeding</p>',
                                unsafe_allow_html=True
                            )
                            st.metric("", no_bleeding_count)
                            bleeding_percentage = (bleeding_count / len(results)) * 100 if results else 0
                            st.markdown('<br>', unsafe_allow_html=True)
                            st.markdown(
                                '<p class="bleeding-label" style="font-size: 1.1rem; margin-bottom: 0.3rem;">Bleeding Rate</p>',
                                unsafe_allow_html=True
                            )
                            st.metric("", f"{bleeding_percentage:.1f}%")
    
    # Tab 3: About
    with tab3:
        st.header("About This Application")
        
        st.markdown("""
        ### Brain Bleeding Classification Model
        
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

