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
import base64

# Page configuration - MUST be first Streamlit call
st.set_page_config(
    page_title="Brain Bleeding Classifier",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from src.model import get_model
    from src.utils import get_device
except ImportError as e:
    # Streamlit is now initialized (st.set_page_config was called)
    st.error(f"Error importing model utilities: {str(e)}")
    st.stop()

# Import setup_model for later use (after Streamlit is initialized)
try:
    import setup_model
except ImportError:
    setup_model = None

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
    
    /* Main header with enhanced animation */
    .main-header {
        font-size: 3.5rem;
        font-weight: 900;
        background: linear-gradient(135deg, #1f77b4 0%, #42a5f5 50%, #7c3aed 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 0.5rem;
        animation: fadeIn 1s ease-in;
        text-shadow: 0 0 30px rgba(31, 119, 180, 0.3);
        letter-spacing: -1px;
        position: relative;
    }
    
    /* Subtitle styling */
    .main-subtitle {
        text-align: center;
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 0.5rem;
        font-weight: 500;
        animation: fadeInUp 0.8s ease-in 0.3s both;
    }
    
    /* Draggable sample images container - compact version */
    .sample-images-container {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 15px;
        padding: 1rem;
        margin: 1rem 0;
        border: 2px dashed #1f77b4;
        position: relative;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.08);
    }
    
    .sample-images-title {
        text-align: center;
        font-size: 1.1rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.5rem;
    }
    
    .sample-images-instruction {
        text-align: center;
        color: #666;
        font-size: 0.85rem;
        margin-bottom: 0.75rem;
        font-style: italic;
    }
    
    /* Drop zone highlight */
    .drop-zone-active {
        background: linear-gradient(135deg, rgba(31, 119, 180, 0.25) 0%, rgba(66, 165, 245, 0.25) 100%) !important;
        border: 4px solid #1f77b4 !important;
        border-style: dashed !important;
        transform: scale(1.03) !important;
        box-shadow: 0 12px 32px rgba(31, 119, 180, 0.4) !important;
        animation: pulse 1s ease-in-out infinite !important;
    }
    
    #drop-zone {
        position: relative;
    }
    
    #drop-zone::after {
        content: '↓ Drop Here ↓';
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        font-size: 1.5rem;
        font-weight: 700;
        color: #1f77b4;
        opacity: 0;
        transition: opacity 0.3s;
        pointer-events: none;
        z-index: 10;
    }
    
    #drop-zone.drop-zone-active::after {
        opacity: 1;
    }
    
    .sample-images-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 1.5rem;
        margin-bottom: 1.5rem;
    }
    
    .sample-image-card {
        background: white;
        border-radius: 10px;
        padding: 0.5rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        cursor: grab;
        position: relative;
        overflow: hidden;
        border: 2px solid transparent;
    }
    
    .sample-image-card::before {
        content: '';
        position: absolute;
        top: 5px;
        right: 5px;
        font-size: 1.2rem;
        opacity: 0;
        transition: opacity 0.3s;
        z-index: 2;
    }
    
    .sample-image-card:hover::before {
        opacity: 1;
    }
    
    .sample-image-card:hover {
        transform: translateY(-8px) scale(1.05);
        box-shadow: 0 8px 24px rgba(31, 119, 180, 0.3);
        border-color: #1f77b4;
    }
    
    .sample-image-card:active {
        cursor: grabbing;
        transform: translateY(-4px) scale(1.02);
    }
    
    .sample-image-wrapper {
        position: relative;
        width: 100%;
        padding-top: 100%;
        overflow: hidden;
        border-radius: 10px;
        background: #f0f0f0;
    }
    
    .sample-image-wrapper img {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        object-fit: cover;
        transition: transform 0.3s;
    }
    
    .sample-image-card:hover .sample-image-wrapper img {
        transform: scale(1.1);
    }
    
    .sample-image-label {
        margin-top: 0.75rem;
        text-align: center;
        font-weight: 700;
        font-size: 0.9rem;
        padding: 0.5rem;
        border-radius: 8px;
    }
    
    .sample-image-label.bleeding {
        background: linear-gradient(135deg, #ffcdd2 0%, #ffb3ba 100%);
        color: #b71c1c;
        border: 2px solid #d32f2f;
    }
    
    .sample-image-label.no-bleeding {
        background: linear-gradient(135deg, #c8e6c9 0%, #a5d6a7 100%);
        color: #1b5e20;
        border: 2px solid #388e3c;
    }
    
    /* Arrow indicator */
    .arrow-indicator {
        text-align: center;
        font-size: 3rem;
        color: #1f77b4;
        margin: 1rem 0;
        animation: bounce 2s ease-in-out infinite;
    }
    
    @keyframes bounce {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-10px); }
    }
    
    /* Drop zone styling */
    .drop-zone {
        border: 3px dashed #1f77b4;
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        background: rgba(31, 119, 180, 0.05);
        transition: all 0.3s;
        min-height: 200px;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-direction: column;
    }
    
    .drop-zone.drag-over {
        background: rgba(31, 119, 180, 0.15);
        border-color: #42a5f5;
        transform: scale(1.02);
    }
    
    .drop-zone-text {
        font-size: 1.2rem;
        color: #1f77b4;
        font-weight: 600;
        margin-top: 1rem;
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
    
    /* Drag and drop visual feedback */
    .dragging {
        opacity: 0.5;
        transform: scale(0.95);
    }
    
    /* Instruction overlay */
    .instruction-overlay {
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        background: rgba(31, 119, 180, 0.95);
        color: white;
        padding: 1rem 2rem;
        border-radius: 15px;
        font-weight: 600;
        font-size: 1.1rem;
        z-index: 10;
        pointer-events: none;
        opacity: 0;
        transition: opacity 0.3s;
    }
    
    .sample-image-card:hover .instruction-overlay {
        opacity: 1;
    }
    
    /* Integrated Header Container */
    .header-container {
        background: linear-gradient(180deg, #ffffff 0%, #f8f9fa 100%);
        padding: 2rem 1rem 1rem 1rem;
        margin-bottom: 2rem;
        border-bottom: 2px solid #e9ecef;
        position: sticky;
        top: 0;
        z-index: 1000;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
    
    /* Navigation Bar - Clean Horizontal Tabs */
    .nav-bar-container {
        margin-top: 1.5rem;
        padding-top: 1rem;
        border-top: 1px solid #e9ecef;
    }
    
    .nav-bar {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 0;
        flex-wrap: wrap;
        max-width: 1000px;
        margin: 0 auto;
    }
    
    /* Navigation Button Styling */
    .nav-button-wrapper {
        position: relative;
        flex: 1;
        min-width: 120px;
        max-width: 200px;
    }
    
    /* Style Streamlit buttons to look like navigation tabs */
    .nav-button-wrapper .stButton > button {
        width: 100%;
        background: transparent !important;
        color: #666 !important;
        border: none !important;
        border-radius: 0 !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        transition: all 0.3s ease !important;
        border-bottom: 3px solid transparent !important;
        margin: 0 !important;
    }
    
    .nav-button-wrapper .stButton > button:hover {
        background: rgba(31, 119, 180, 0.05) !important;
        color: #1f77b4 !important;
        border-bottom-color: rgba(31, 119, 180, 0.3) !important;
        transform: translateY(-2px) !important;
    }
    
    /* Active state */
    .nav-button-wrapper.active .stButton > button {
        color: #1f77b4 !important;
        border-bottom-color: #1f77b4 !important;
        background: rgba(31, 119, 180, 0.08) !important;
        font-weight: 700 !important;
    }
    
    .nav-button-wrapper.active .stButton > button:hover {
        background: rgba(31, 119, 180, 0.12) !important;
    }
    
    /* Header Links Styling - Icon Buttons */
    .header-links-container {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 1.5rem;
        margin: 1rem 0 0.5rem 0;
        flex-wrap: wrap;
    }
    
    .header-link-icon {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 40px;
        height: 40px;
        border-radius: 50%;
        background: rgba(31, 119, 180, 0.1);
        color: #1f77b4;
        text-decoration: none;
        transition: all 0.3s ease;
        border: 2px solid transparent;
    }
    
    .header-link-icon:hover {
        background: rgba(31, 119, 180, 0.2);
        border-color: #1f77b4;
        transform: translateY(-3px) scale(1.1);
        box-shadow: 0 4px 12px rgba(31, 119, 180, 0.3);
    }
    
    .header-link-icon svg {
        width: 20px;
        height: 20px;
        fill: currentColor;
    }
    
    /* Research page specific styles */
    .research-section {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 1.5rem 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        border: 2px solid rgba(31, 119, 180, 0.1);
        transition: all 0.3s ease;
    }
    
    .research-section:hover {
        box-shadow: 0 6px 20px rgba(31, 119, 180, 0.15);
        border-color: rgba(31, 119, 180, 0.2);
    }
    
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1.5rem;
        margin: 1.5rem 0;
    }
    
    .metric-card-research {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 2rem 1.5rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        border: 2px solid transparent;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card-research::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(31, 119, 180, 0.1), transparent);
        transition: left 0.5s;
    }
    
    .metric-card-research:hover::before {
        left: 100%;
    }
    
    .metric-card-research:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 12px 24px rgba(31, 119, 180, 0.25);
        border-color: #1f77b4;
    }
    
    .metric-value {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #1f77b4 0%, #42a5f5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0.5rem 0;
        line-height: 1.2;
    }
    
    .metric-label {
        font-size: 1rem;
        color: #666;
        font-weight: 600;
        margin-top: 0.5rem;
    }
    
    .notebook-viewer {
        background: white;
        border-radius: 15px;
        padding: 2rem;
        margin: 1.5rem 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        max-height: 900px;
        overflow-y: auto;
        border: 2px solid rgba(31, 119, 180, 0.1);
    }
    
    .notebook-viewer::-webkit-scrollbar {
        width: 10px;
    }
    
    .notebook-viewer::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    .notebook-viewer::-webkit-scrollbar-thumb {
        background: #1f77b4;
        border-radius: 10px;
    }
    
    .notebook-viewer::-webkit-scrollbar-thumb:hover {
        background: #1565c0;
    }
    
    .visualization-gallery {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
        gap: 2rem;
        margin: 2rem 0;
    }
    
    .viz-card {
        background: white;
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        border: 2px solid transparent;
        position: relative;
    }
    
    .viz-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(135deg, rgba(31, 119, 180, 0.05) 0%, rgba(66, 165, 245, 0.05) 100%);
        opacity: 0;
        transition: opacity 0.3s;
        z-index: 1;
    }
    
    .viz-card:hover::before {
        opacity: 1;
    }
    
    .viz-card:hover {
        transform: translateY(-10px) scale(1.02);
        box-shadow: 0 12px 30px rgba(31, 119, 180, 0.3);
        border-color: #1f77b4;
    }
    
    .viz-card img {
        width: 100%;
        height: auto;
        display: block;
        transition: transform 0.3s;
    }
    
    .viz-card:hover img {
        transform: scale(1.05);
    }
    
    .viz-card-title {
        padding: 1.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        font-size: 1.1rem;
        background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
        border-top: 2px solid rgba(31, 119, 180, 0.1);
        position: relative;
        z-index: 2;
    }
    
    .experiment-hero {
        background: linear-gradient(135deg, #1f77b4 0%, #42a5f5 50%, #7c3aed 100%);
        padding: 3rem 2rem;
        border-radius: 20px;
        margin: 2rem 0;
        color: white;
        text-align: center;
        box-shadow: 0 8px 30px rgba(31, 119, 180, 0.4);
    }
    
    .experiment-hero h1 {
        color: white;
        font-size: 3rem;
        font-weight: 900;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
    }
    
    .experiment-hero p {
        font-size: 1.3rem;
        opacity: 0.95;
        margin: 0.5rem 0;
    }
    
    .config-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1.5rem;
        margin: 1.5rem 0;
    }
    
    .config-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.08);
        border-left: 4px solid #1f77b4;
        transition: all 0.3s ease;
    }
    
    .config-card:hover {
        transform: translateX(5px);
        box-shadow: 0 4px 15px rgba(31, 119, 180, 0.2);
    }
    
    .config-card h4 {
        color: #1f77b4;
        margin-bottom: 1rem;
        font-size: 1.2rem;
    }
    
    .download-btn {
        background: linear-gradient(135deg, #1f77b4 0%, #42a5f5 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        text-decoration: none;
        display: inline-block;
        margin-top: 0.5rem;
        transition: all 0.3s ease;
        font-weight: 600;
    }
    
    .download-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(31, 119, 180, 0.4);
    }
    </style>
    
    <script>
    // Drag and drop functionality
    document.addEventListener('DOMContentLoaded', function() {
        const sampleCards = document.querySelectorAll('.sample-image-card');
        const fileUploader = document.querySelector('[data-testid="stFileUploader"]');
        
        sampleCards.forEach(card => {
            card.addEventListener('dragstart', function(e) {
                e.dataTransfer.effectAllowed = 'copy';
                e.dataTransfer.setData('text/plain', this.dataset.imagePath);
                this.classList.add('dragging');
            });
            
            card.addEventListener('dragend', function(e) {
                this.classList.remove('dragging');
            });
        });
        
        // Make cards more interactive
        sampleCards.forEach(card => {
            card.addEventListener('mouseenter', function() {
                this.style.cursor = 'grab';
            });
            
            card.addEventListener('mousedown', function() {
                this.style.cursor = 'grabbing';
            });
            
            card.addEventListener('mouseup', function() {
                this.style.cursor = 'grab';
            });
        });
    });
    </script>
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
if 'selected_sample_image' not in st.session_state:
    st.session_state.selected_sample_image = None


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


def get_test_images():
    """Get available test images from the data/test directory"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    bleeding_dir = os.path.join(base_dir, "data", "test", "bleeding")
    no_bleeding_dir = os.path.join(base_dir, "data", "test", "no_bleeding")
    
    test_images = {
        'Bleeding': [],
        'No Bleeding': []
    }
    
    # Get bleeding images
    if os.path.exists(bleeding_dir):
        bleeding_files = sorted([f for f in os.listdir(bleeding_dir) 
                                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])
        test_images['Bleeding'] = bleeding_files[:50]  # Limit to first 50 for performance
    
    # Get no bleeding images
    if os.path.exists(no_bleeding_dir):
        no_bleeding_files = sorted([f for f in os.listdir(no_bleeding_dir) 
                                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])
        test_images['No Bleeding'] = no_bleeding_files[:50]  # Limit to first 50 for performance
    
    return test_images


def get_sample_images(num_samples=4):
    """Get a small sample of test images for the draggable preview section"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    bleeding_dir = os.path.join(base_dir, "data", "test", "bleeding")
    no_bleeding_dir = os.path.join(base_dir, "data", "test", "no_bleeding")
    
    samples = []
    
    # Get bleeding samples
    if os.path.exists(bleeding_dir):
        bleeding_files = sorted([f for f in os.listdir(bleeding_dir) 
                                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])
        for i, filename in enumerate(bleeding_files[:num_samples]):
            if i < num_samples:
                image_path = os.path.join(bleeding_dir, filename)
                if os.path.exists(image_path):
                    samples.append({
                        'filename': filename,
                        'category': 'Bleeding',
                        'path': image_path
                    })
    
    # Get no bleeding samples
    if os.path.exists(no_bleeding_dir):
        no_bleeding_files = sorted([f for f in os.listdir(no_bleeding_dir) 
                                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])
        for i, filename in enumerate(no_bleeding_files[:num_samples]):
            if i < num_samples:
                image_path = os.path.join(no_bleeding_dir, filename)
                if os.path.exists(image_path):
                    samples.append({
                        'filename': filename,
                        'category': 'No Bleeding',
                        'path': image_path
                    })
    
    return samples


def load_test_image(category, filename):
    """Load a test image from the data/test directory"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if category == 'Bleeding':
        image_path = os.path.join(base_dir, "data", "test", "bleeding", filename)
    else:
        image_path = os.path.join(base_dir, "data", "test", "no_bleeding", filename)
    
    if os.path.exists(image_path):
        return Image.open(image_path)
    else:
        raise FileNotFoundError(f"Test image not found: {image_path}")


def image_to_base64(image):
    """Convert PIL Image to base64 string"""
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str


def render_navigation_bar():
    """Render the integrated navigation bar with clean tab-style design"""
    pages = {
        "Home": "Home",
        "Batch": "Batch Processing",
        "Research": "Research & Experiments",
        "About": "About"
    }
    
    # Initialize page if not set
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Home"
    
    # Handle page navigation via query params
    try:
        query_params = st.query_params
        if 'page' in query_params:
            page = query_params['page']
            if page in pages:
                st.session_state.current_page = page
                # Clear query params after setting state
                try:
                    st.query_params.clear()
                except:
                    pass  # Ignore if query params can't be cleared
                st.rerun()
    except Exception:
        pass  # Ignore query param errors
    
    current_page = st.session_state.current_page
    
    # Create clean horizontal navigation bar
    st.markdown('<div class="nav-bar-container">', unsafe_allow_html=True)
    st.markdown('<div class="nav-bar">', unsafe_allow_html=True)
    
    # Create navigation buttons with active state styling
    nav_cols = st.columns(len(pages))
    for idx, (page_key, page_label) in enumerate(pages.items()):
        with nav_cols[idx]:
            is_active = current_page == page_key
            # Add active class wrapper
            wrapper_class = "nav-button-wrapper active" if is_active else "nav-button-wrapper"
            st.markdown(f'<div class="{wrapper_class}">', unsafe_allow_html=True)
            
            if st.button(page_label, key=f"nav_{page_key}", use_container_width=True, type="secondary"):
                st.session_state.current_page = page_key
                st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


def notebook_to_html(notebook_path):
    """Convert Jupyter notebook to HTML for embedding"""
    try:
        from nbconvert import HTMLExporter
        import nbformat
        
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = nbformat.read(f, as_version=4)
        
        html_exporter = HTMLExporter()
        html_exporter.template_name = 'classic'
        (body, resources) = html_exporter.from_notebook_node(notebook)
        
        return body
    except ImportError:
        return None  # nbconvert not installed
    except Exception as e:
        return None  # Error reading notebook


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


def page_home(img_size):
    """Home page - Single Image Prediction"""
    st.header("Single Image Prediction")
    
    if not st.session_state.model_loaded:
        st.warning("Please load a model in the sidebar before making predictions.")
    else:
        # Image source selection - moved to top
        image_source = st.radio(
            "Choose image source:",
            ["Upload Image", "Use Test Image"],
            horizontal=True,
            help="Upload your own image or select from available test images"
        )
        
        image = None
        image_source_name = None
        
        # Check if a sample image was selected
        if st.session_state.selected_sample_image:
            try:
                image = Image.open(st.session_state.selected_sample_image['path'])
                image_source_name = f"{st.session_state.selected_sample_image['category']} - {st.session_state.selected_sample_image['filename']}"
                # Clear the selection after using it
                st.session_state.selected_sample_image = None
            except Exception as e:
                st.error(f"Error loading selected sample image: {str(e)}")
        
        if image_source == "Upload Image":
            # Enhanced drop zone with clear visual feedback
            st.markdown("""
            <div id="drop-zone" style="
                background: linear-gradient(135deg, rgba(31, 119, 180, 0.1) 0%, rgba(66, 165, 245, 0.1) 100%);
                border: 3px dashed #1f77b4;
                border-radius: 15px;
                padding: 2rem;
                text-align: center;
                margin-bottom: 1rem;
                transition: all 0.3s ease;
            ">
                <div style="font-size: 2rem; margin-bottom: 0.5rem; color: #1f77b4;">UPLOAD</div>
                <p style="font-size: 1.3rem; color: #1f77b4; font-weight: 700; margin: 0.5rem 0;">
                    Drag & Drop Your Image Here
                </p>
                <p style="font-size: 1rem; color: #1f77b4; font-weight: 600; margin: 0.5rem 0;">
                    or click to browse
                </p>
                <p style="font-size: 0.85rem; color: #666; margin-top: 0.5rem;">
                    Supported: PNG, JPG, JPEG, BMP, TIFF
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            uploaded_file = st.file_uploader(
                "Upload a brain scan image",
                type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
                help="Upload a medical image (CT scan or MRI) for classification",
                label_visibility="collapsed"
            )
            
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                image_source_name = uploaded_file.name
        
        # Sample images section
        if not image:
            st.markdown("---")
            st.markdown("""
            <div class="sample-images-container">
                <div class="sample-images-title">
                    Or Try Sample Images
                </div>
                <div class="sample-images-instruction">
                    Click any sample below to test instantly
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            sample_images = get_sample_images(num_samples=4)
            
            if len(sample_images) > 0:
                num_cols = min(4, len(sample_images))
                cols = st.columns(num_cols)
                
                for idx, sample in enumerate(sample_images):
                    with cols[idx % num_cols]:
                        try:
                            sample_img = Image.open(sample['path'])
                            display_img = sample_img.copy()
                            display_img.thumbnail((120, 120), Image.Resampling.LANCZOS)
                            
                            label_class = "bleeding" if sample['category'] == 'Bleeding' else "no-bleeding"
                            label_color = "#b71c1c" if sample['category'] == 'Bleeding' else "#1b5e20"
                            bg_color = "#ffcdd2" if sample['category'] == 'Bleeding' else "#c8e6c9"
                            
                            st.markdown(f"""
                            <div style="
                                text-align: center;
                                font-weight: 700;
                                font-size: 0.75rem;
                                color: {label_color};
                                margin-bottom: 0.25rem;
                                padding: 0.25rem;
                                background: {bg_color};
                                border-radius: 5px;
                                border: 1px solid {label_color};
                            ">
                                {sample['category']}
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.image(display_img, use_container_width=True)
                            
                            if st.button(f"Use", key=f"sample_{idx}", use_container_width=True):
                                st.session_state.selected_sample_image = sample
                                st.rerun()
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
        else:
            # Test image selection
            test_images = get_test_images()
            
            if len(test_images['Bleeding']) == 0 and len(test_images['No Bleeding']) == 0:
                st.warning("No test images found in data/test directory.")
            else:
                available_categories = []
                if len(test_images['Bleeding']) > 0:
                    available_categories.append('Bleeding')
                if len(test_images['No Bleeding']) > 0:
                    available_categories.append('No Bleeding')
                
                if len(available_categories) > 0:
                    selected_category = st.selectbox(
                        "Select image category:",
                        available_categories,
                        help="Choose whether to test with bleeding or no bleeding images"
                    )
                    
                    if selected_category:
                        available_images = test_images[selected_category]
                        
                        if len(available_images) > 0:
                            selected_image = st.selectbox(
                                f"Select a {selected_category.lower()} test image:",
                                available_images,
                                help="Choose a test image to analyze"
                            )
                            
                            if selected_image:
                                try:
                                    image = load_test_image(selected_category, selected_image)
                                    image_source_name = f"{selected_category} - {selected_image}"
                                except Exception as e:
                                    st.error(f"Error loading test image: {str(e)}")
        
        # Display image and prediction interface
        if image is not None:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("Input Image")
                caption = image_source_name if image_source_name else "Selected Image"
                st.image(image, caption=caption, use_container_width=True)
            
            with col2:
                st.subheader("Prediction Results")
                
                predict_clicked = st.button("Predict", type="primary", use_container_width=True, key="predict_button")
                
                if predict_clicked:
                    with st.spinner("Analyzing image..."):
                        image_tensor = preprocess_image(image, img_size)
                        prediction, confidence, probabilities = predict_image(
                            st.session_state.model,
                            image_tensor,
                            st.session_state.device
                        )
                        
                        st.session_state.prediction = prediction
                        st.session_state.confidence = confidence
                        st.session_state.probabilities = probabilities
                        st.session_state.image_source_name = image_source_name
                
                # Show results if available
                if 'prediction' in st.session_state:
                    prediction = st.session_state.prediction
                    confidence = st.session_state.confidence
                    probabilities = st.session_state.probabilities
                    
                    # Prediction box
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
                        st.metric("No Bleeding", f"{probabilities[0]*100:.2f}%")
                    with col4:
                        st.metric("Bleeding", f"{probabilities[1]*100:.2f}%")


def page_batch(img_size):
    """Batch Processing page"""
    st.header("Batch Processing")
    
    if not st.session_state.model_loaded:
        st.warning("Please load a model in the sidebar before processing images.")
    else:
        st.markdown("Upload multiple images to process them all at once.")
        
        uploaded_files = st.file_uploader(
            "Upload multiple images",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
            accept_multiple_files=True,
            help="Select multiple brain scan images for batch processing"
        )
        
        if uploaded_files:
            if st.button("Process All Images", type="primary"):
                results = []
                progress_bar = st.progress(0)
                
                for idx, uploaded_file in enumerate(uploaded_files):
                    try:
                        image = Image.open(uploaded_file)
                        image_tensor = preprocess_image(image, img_size)
                        prediction, confidence, probabilities = predict_image(
                            st.session_state.model,
                            image_tensor,
                            st.session_state.device
                        )
                        
                        results.append({
                            'Filename': uploaded_file.name,
                            'Prediction': prediction,
                            'Confidence': f"{confidence*100:.2f}%",
                            'No Bleeding Prob': f"{probabilities[0]*100:.2f}%",
                            'Bleeding Prob': f"{probabilities[1]*100:.2f}%"
                        })
                    except Exception as e:
                        results.append({
                            'Filename': uploaded_file.name,
                            'Prediction': 'Error',
                            'Confidence': str(e),
                            'No Bleeding Prob': 'N/A',
                            'Bleeding Prob': 'N/A'
                        })
                    
                    progress_bar.progress((idx + 1) / len(uploaded_files))
                
                st.success(f"Processed {len(results)} images!")
                
                # Display results
                import pandas as pd
                df = pd.DataFrame(results)
                st.dataframe(df, use_container_width=True)
                
                # Summary statistics
                if len(results) > 0:
                    st.subheader("Summary Statistics")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        bleeding_count = sum(1 for r in results if r['Prediction'] == 'Bleeding')
                        st.metric("Bleeding Detected", bleeding_count)
                    with col2:
                        no_bleeding_count = sum(1 for r in results if r['Prediction'] == 'No Bleeding')
                        st.metric("No Bleeding", no_bleeding_count)
                    with col3:
                        st.metric("Total Processed", len(results))
                
                # Download results
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download Results as CSV",
                    data=csv,
                    file_name="batch_predictions.csv",
                    mime="text/csv"
                )


def page_research():
    """Research & Experiments page - Comprehensive experiment documentation"""
    
    # Hero Section
    st.markdown("""
    <div class="experiment-hero">
        <h1>Complete Model Evaluation</h1>
        <p>Comprehensive experimental evaluation of the Brain Bleeding Classification model</p>
        <p style="font-size: 1rem; opacity: 0.9;">ResNet50 Transfer Learning | 99.57% Test Accuracy</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for better organization
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Overview & Results", 
        "Model Architecture", 
        "Training Configuration", 
        "Visualizations", 
        "Complete Notebook"
    ])
    
    with tab1:
        # Overview Section
        st.markdown('<div class="research-section">', unsafe_allow_html=True)
        st.subheader("Experiment Overview")
        st.markdown("""
        This experiment presents a comprehensive evaluation of a deep learning model for binary classification 
        of brain bleeding in medical CT/MRI images. The model uses ResNet50 with transfer learning, trained on 
        a stratified dataset split with rigorous evaluation protocols.
        """)
        
        # Key Highlights
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div class="config-card">
                <h4>Objective</h4>
                <p>Binary classification of brain bleeding from medical images using deep learning</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="config-card">
                <h4>Methodology</h4>
                <p>Transfer learning with ResNet50, fine-tuned on medical imaging dataset</p>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown("""
            <div class="config-card">
                <h4>Results</h4>
                <p>99.57% test accuracy with comprehensive evaluation metrics</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Performance Results Section
        st.markdown('<div class="research-section">', unsafe_allow_html=True)
        st.subheader("Performance Results")
        
        # Load results from CSV if available
        results_csv_path = "research/visualizations/output/research_evaluation/results_summary.csv"
        if os.path.exists(results_csv_path):
            import pandas as pd
            try:
                df_results = pd.read_csv(results_csv_path)
                st.markdown("### Results Summary Table")
                st.dataframe(df_results, use_container_width=True, hide_index=True)
            except:
                pass
        
        # Metrics Grid
        st.markdown("### Test Set Performance Metrics")
        st.markdown('<div class="metric-grid">', unsafe_allow_html=True)
        
        test_metrics = {
            "Test Accuracy": "99.57%",
            "Precision": "99.55%",
            "Recall (Sensitivity)": "99.85%",
            "Specificity": "98.86%",
            "F1-Score": "99.70%",
            "AUC-ROC": "99.88%",
            "AUC-PR": "99.96%",
            "False Positive Rate": "1.14%"
        }
        
        cols = st.columns(4)
        for idx, (metric, value) in enumerate(test_metrics.items()):
            with cols[idx % 4]:
                st.markdown(f"""
                <div class="metric-card-research">
                    <div class="metric-value">{value}</div>
                    <div class="metric-label">{metric}</div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Confusion Matrix
        st.markdown("### Confusion Matrix (Test Set)")
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("""
            **Test Set Confusion Matrix:**
            
            ```
                        Predicted
                      No Bleeding  Bleeding
            Actual No Bleeding      261          3
            Actual Bleeding           1        660
            ```
            
            **Detailed Counts:**
            - **True Negatives (TN):** 261
            - **False Positives (FP):** 3
            - **False Negatives (FN):** 1
            - **True Positives (TP):** 660
            
            **Total Test Samples:** 925
            """)
        
        with col2:
            cm_path = "research/visualizations/output/research_evaluation/confusion_matrix_resnet50_test.png"
            if os.path.exists(cm_path):
                st.image(cm_path, use_container_width=True)
            else:
                st.info("Confusion matrix visualization not found")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        # Model Architecture Section
        st.markdown('<div class="research-section">', unsafe_allow_html=True)
        st.subheader("Model Architecture")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### Base Architecture: ResNet50
            
            **Transfer Learning Approach:**
            - Pre-trained on ImageNet dataset
            - All layers trainable (fine-tuning)
            - Feature extraction backbone: ResNet50
            - 2048-dimensional feature vector output
            
            **Why ResNet50?**
            - Proven architecture for image classification
            - Residual connections prevent degradation
            - Pre-trained weights provide strong initialization
            - Suitable for medical image analysis
            """)
            
            st.markdown("""
            ### Custom Classifier Head
            
            The final classification layers:
            ```
            Input: 2048 features (from ResNet50)
            ↓
            Dropout (0.5)
            ↓
            Linear: 2048 → 512
            ↓
            ReLU activation
            ↓
            Dropout (0.3)
            ↓
            Linear: 512 → 2 (binary classification)
            ```
            """)
        
        with col2:
            st.markdown("""
            ### Model Specifications
            
            **Architecture Details:**
            - **Total Parameters:** 24,558,146
            - **Trainable Parameters:** 24,558,146
            - **Input Size:** 224 × 224 pixels (RGB)
            - **Output Classes:** 2 (No Bleeding, Bleeding)
            - **Normalization:** ImageNet statistics
              - Mean: [0.485, 0.456, 0.406]
              - Std: [0.229, 0.224, 0.225]
            
            **Model Components:**
            - ResNet50 backbone (convolutional layers)
            - Global Average Pooling
            - Custom fully connected layers
            - Dropout for regularization
            """)
            
            st.markdown("""
            ### Training Strategy
            
            **Fine-tuning Approach:**
            - All layers updated during training
            - Learning rate: 0.001 (initial)
            - Adaptive learning rate scheduling
            - Weight decay for regularization
            """)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab3:
        # Training Configuration Section
        st.markdown('<div class="research-section">', unsafe_allow_html=True)
        st.subheader("Training Configuration")
        
        st.markdown('<div class="config-grid">', unsafe_allow_html=True)
        
        config_sections = [
            {
                "title": "Hyperparameters",
                "content": """
                - **Epochs:** 50 (maximum)
                - **Learning Rate:** 0.001
                - **Optimizer:** Adam
                - **Weight Decay:** 1e-4
                - **Loss Function:** CrossEntropyLoss
                """
            },
            {
                "title": "Training Strategy",
                "content": """
                - **Batch Size:** 32
                - **Image Size:** 224×224
                - **Data Augmentation:** Yes
                - **Early Stopping:** Patience=10
                - **LR Scheduler:** ReduceLROnPlateau
                """
            },
            {
                "title": "Data Split",
                "content": """
                - **Training:** 70% (4,314 images)
                - **Validation:** 15% (925 images)
                - **Test:** 15% (925 images)
                - **Stratified by class**
                """
            },
            {
                "title": "Training Duration",
                "content": """
                - **Actual Epochs:** 48
                - **Early Stopping:** Triggered
                - **Best Val Accuracy:** 99.89%
                - **Training Time:** ~12 hours (CPU)
                """
            }
        ]
        
        cols = st.columns(2)
        for idx, config in enumerate(config_sections):
            with cols[idx % 2]:
                st.markdown(f"""
                <div class="config-card">
                    <h4>{config['title']}</h4>
                    {config['content']}
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Data Augmentation Details
        st.markdown("### Data Augmentation")
        st.markdown("""
        The following augmentations were applied during training (with probabilities):
        - **Horizontal Flip:** 50%
        - **Random Rotation 90°:** 50%
        - **Random Brightness/Contrast:** 50% (±20%)
        - **Gaussian Noise:** 30% (std: 0.1-0.5)
        - **Gaussian Blur:** 30% (kernel: 3)
        
        Validation and test sets used only resizing and normalization (no augmentation).
        """)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab4:
        # Visualizations Gallery
        st.markdown('<div class="research-section">', unsafe_allow_html=True)
        st.subheader("Visualization Gallery")
        st.markdown("All visualizations generated from the complete evaluation experiment.")
        
        viz_dir = "research/visualizations/output/research_evaluation"
        if os.path.exists(viz_dir):
            viz_files = {
                "ROC Curve": {
                    "file": "roc_curve_resnet50_test.png",
                    "desc": "Receiver Operating Characteristic curve showing model's ability to distinguish between classes. AUC-ROC: 99.88%"
                },
                "PR Curve": {
                    "file": "pr_curve_resnet50_test.png",
                    "desc": "Precision-Recall curve demonstrating model performance across different thresholds. AUC-PR: 99.96%"
                },
                "Calibration Curve": {
                    "file": "calibration_curve_resnet50_test.png",
                    "desc": "Reliability diagram showing how well-calibrated the model's probability predictions are"
                },
                "Training Curves": {
                    "file": "training_curves_resnet50_test.png",
                    "desc": "Training and validation loss/accuracy over 48 epochs, showing convergence and early stopping"
                },
                "Confusion Matrix": {
                    "file": "confusion_matrix_resnet50_test.png",
                    "desc": "Raw confusion matrix showing true vs predicted classifications on test set"
                },
                "Normalized Confusion Matrix": {
                    "file": "confusion_matrix_normalized_resnet50_test.png",
                    "desc": "Normalized confusion matrix showing classification percentages"
                }
            }
            
            # Display in grid with expandable details
            for idx, (viz_name, viz_info) in enumerate(viz_files.items()):
                viz_path = os.path.join(viz_dir, viz_info["file"])
                if os.path.exists(viz_path):
                    with st.expander(f"{viz_name}", expanded=(idx < 2)):
                        st.markdown(f'<div class="viz-card">', unsafe_allow_html=True)
                        st.image(viz_path, use_container_width=True)
                        st.markdown(f'<div class="viz-card-title">{viz_name}</div>', unsafe_allow_html=True)
                        st.markdown(f'<p style="padding: 1rem; color: #666;">{viz_info["desc"]}</p>', unsafe_allow_html=True)
                        
                        # Download button
                        with open(viz_path, "rb") as f:
                            st.download_button(
                                label=f"Download {viz_name}",
                                data=f.read(),
                                file_name=viz_info["file"],
                                mime="image/png",
                                key=f"download_{idx}"
                            )
                        st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("Visualizations directory not found")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab5:
        # Jupyter Notebook Viewer
        st.markdown('<div class="research-section">', unsafe_allow_html=True)
        st.subheader("Complete Evaluation Notebook")
        st.markdown("""
        The complete evaluation notebook contains the full experimental pipeline including data splitting, 
        model training, evaluation, and visualization generation. View it interactively below or download 
        the notebook file.
        """)
        
        notebook_path = "research/complete_evaluation.ipynb"
        if os.path.exists(notebook_path):
            # Try to convert and display notebook
            notebook_html = notebook_to_html(notebook_path)
            if notebook_html:
                st.markdown("### Interactive Notebook Viewer")
                st.markdown('<div class="notebook-viewer">', unsafe_allow_html=True)
                st.markdown(notebook_html, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("To view the notebook interactively, install nbconvert: `pip install nbconvert`")
                
                # Show notebook summary
                st.markdown("### Notebook Summary")
                st.markdown("""
                The complete evaluation notebook (`research/complete_evaluation.ipynb`) contains:
                
                **1. Data Preparation**
                - Stratified train/val/test split (70/15/15)
                - Data loading and preprocessing
                - Image augmentation setup
                
                **2. Model Training**
                - ResNet50 model initialization
                - Training loop with early stopping
                - Learning rate scheduling
                - Model checkpointing
                
                **3. Evaluation**
                - Validation set evaluation
                - Test set evaluation (final)
                - Comprehensive metrics calculation
                - Confusion matrix generation
                
                **4. Visualization Generation**
                - ROC and PR curves
                - Calibration curves
                - Training history plots
                - Confusion matrices
                
                **5. Results Summary**
                - Performance metrics table
                - Model saving
                - Results export to CSV
                """)
                
                # Show notebook file info
                with st.expander("Notebook File Information", expanded=False):
                    import json
                    try:
                        with open(notebook_path, 'r') as f:
                            nb_data = json.load(f)
                        st.json({
                            "Total Cells": len(nb_data.get('cells', [])),
                            "Notebook Format": nb_data.get('nbformat', 'Unknown'),
                            "Notebook Format Minor": nb_data.get('nbformat_minor', 'Unknown'),
                            "File Size": f"{os.path.getsize(notebook_path) / 1024:.2f} KB"
                        })
                    except Exception as e:
                        st.info(f"Could not read notebook metadata: {str(e)}")
                
                # Download notebook button
                with open(notebook_path, "rb") as f:
                    st.download_button(
                        label="Download Complete Notebook (.ipynb)",
                        data=f.read(),
                        file_name="complete_evaluation.ipynb",
                        mime="application/json",
                        key="download_notebook"
                    )
        else:
            st.warning(f"Notebook file not found at: {notebook_path}")
        
        st.markdown('</div>', unsafe_allow_html=True)


def page_about():
    """About page"""
    st.header("About This Application")
    
    st.markdown("""
    ### Overview
    
    This application provides an interactive interface for brain bleeding classification using deep learning.
    The model uses transfer learning with ResNet50 and EfficientNet architectures to classify brain scan images.
    
    ### Features
    
    - **Transfer Learning**: Pre-trained ImageNet weights for better performance
    - **Multiple Architectures**: Support for ResNet50 and EfficientNet variants
    - **Data Augmentation**: Albumentations for robust training
    - **Interactive Web Interface**: Real-time predictions with confidence scores
    - **Batch Processing**: Process multiple images at once
    - **Comprehensive Metrics**: Detailed performance analysis
    
    ### Technical Details
    
    - **Framework**: PyTorch
    - **Web Framework**: Streamlit
    - **Model**: ResNet50 / EfficientNet
    - **Input**: 224×224 RGB images
    - **Output**: Binary classification (Bleeding / No Bleeding)
    
    ### Important Notes
    
    **[WARNING]** This tool is for research and educational purposes only.
    
    - It should **not be used as a substitute for professional medical diagnosis**
    - Always consult with medical professionals for actual diagnosis
    - Model predictions are based on training data and may not be 100% accurate
    
    ### Links
    
    - [GitHub Repository](https://github.com/Thespaceblade/Brain-ML-Model)
    - [Portfolio](https://jasonindata.vercel.app)
    """)


def main():
    # Display a simple message first to ensure app is running
    # This helps with health checks
    try:
        # Try to setup model file if it doesn't exist (for deployment)
        # This runs after Streamlit is initialized, so secrets are available
        if setup_model is not None:
            try:
                if not os.path.exists("models/best_model.pth"):
                    try:
                        setup_model.setup_model()
                    except Exception as e:
                        # Silently fail if setup_model doesn't work - not critical
                        pass
            except Exception as e:
                # Silently fail if setup_model doesn't work - not critical
                pass
    except Exception as e:
        # If anything fails during initialization, continue anyway
        pass
    
    # Integrated Header Container
    st.markdown('<div class="header-container">', unsafe_allow_html=True)
    
    # Main Header
    st.markdown('<h1 class="main-header">Brain Bleeding Classifier</h1>', unsafe_allow_html=True)
    st.markdown('<p class="main-subtitle">ResNet50 & EfficientNet CNN Models | Transfer Learning Classification</p>', unsafe_allow_html=True)
    
    # Header Links - Icon Buttons
    st.markdown("""
    <div class="header-links-container">
        <a href="https://github.com/Thespaceblade/Brain-ML-Model" target="_blank" class="header-link-icon" title="GitHub Repository">
            <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
            </svg>
        </a>
        <a href="https://jasonindata.vercel.app" target="_blank" class="header-link-icon" title="Portfolio">
            <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path d="M20 6h-4V4c0-1.11-.89-2-2-2h-4c-1.11 0-2 .89-2 2v2H4c-1.11 0-2 .89-2 2v11c0 1.11.89 2 2 2h16c1.11 0 2-.89 2-2V8c0-1.11-.89-2-2-2zm-6 0h-4V4h4v2z"/>
            </svg>
        </a>
    </div>
    """, unsafe_allow_html=True)
    
    # Navigation Bar (integrated in header)
    render_navigation_bar()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Instructions Section (only on Home page)
    if st.session_state.get('current_page', 'Home') == 'Home':
        with st.expander("How to Use This Application", expanded=False):
            st.markdown("""
        ### Quick Start Guide
        
        **Step 1: Load Your Model**
        - In the sidebar, enter the path to your trained model (default: `models/best_model.pth`)
        - Select the model architecture (ResNet50, EfficientNet-B0, or EfficientNet-B1)
        - Click **"Load Model"** to load the model into memory
        - Wait for the success message indicating the model is ready
        
        **Step 2: Select or Upload an Image**
        - Navigate to the **"Single Image"** tab for individual predictions
        - Choose **"Upload Image"** to use your own brain scan image, or **"Use Test Image"** to select from available test images
        - Test images are available for both bleeding and no bleeding cases - perfect for trying out the model without your own images
        - Or use the **"Batch Processing"** tab for multiple images
        - Supported formats: PNG, JPG, JPEG, BMP, TIFF
        
        **Step 3: Get Predictions**
        - Click the **"Predict"** button (for single images)
        - Or click **"Process All Images"** (for batch processing)
        - View the results with confidence scores and probability breakdown
        
        ---
        
        ### Detailed Instructions
        
        #### Single Image Prediction
        1. **Load Model**: Ensure a model is loaded in the sidebar (check "Model Status")
        2. **Select Image Source**: 
           - Choose **"Upload Image"** to upload your own brain scan image
           - Or choose **"Use Test Image"** to select from available test images (both bleeding and no bleeding examples are available)
        3. **Select Test Image** (if using test images): Choose the category (Bleeding or No Bleeding) and then select a specific test image
        4. **Predict**: Click the "Predict" button to analyze the image (test images are processed through the model just like uploaded images)
        5. **View Results**: 
           - Prediction result (Bleeding / No Bleeding)
           - Confidence score (0-100%)
           - Detailed probability breakdown
           - Interactive probability chart
        
        #### Batch Processing
        1. **Load Model**: Ensure a model is loaded in the sidebar
        2. **Upload Multiple Images**: Select multiple images using the file uploader
        3. **Process**: Click "Process All Images" to analyze all uploaded images
        4. **Review Results**:
           - View results in a sortable table
           - Check summary statistics
           - Download results as CSV file
        
        #### Model Configuration (Sidebar)
        - **Model Architecture**: Select the architecture that matches your trained model
        - **Model Path**: Enter the path to your `.pth` model checkpoint file
        - **Image Size**: Adjust the input image size (default: 224x224)
        - **Model Status**: Check if the model is loaded and ready
        - **Device Info**: See whether the model is running on GPU or CPU
        
        ---
        
        ### Tips & Best Practices
        
        - **Image Quality**: Use high-quality images for better predictions
        - **Image Format**: Ensure images are in supported formats (PNG, JPG, JPEG, BMP, TIFF)
        - **Model Matching**: Make sure the selected architecture matches your trained model
        - **File Paths**: Use relative paths (e.g., `models/best_model.pth`) for easier portability
        - **GPU Usage**: The app automatically uses GPU if available, otherwise falls back to CPU
        
        ---
        
        ### Important Notes
        
        - This tool is for **research and educational purposes only**
        - It should **not be used as a substitute for professional medical diagnosis**
        - Always consult with medical professionals for actual diagnosis
        - Model predictions are based on training data and may not be 100% accurate
        
        ---
        
        ### Troubleshooting
        
        **Model Not Loading?**
        - Check that the model file path is correct
        - Ensure the file exists in the specified location
        - Verify the model architecture matches your trained model
        - Check the error message in the sidebar for details
        
        **Images Not Uploading?**
        - Ensure images are in supported formats (PNG, JPG, JPEG, BMP, TIFF)
        - Check file size (very large files may take longer to process)
        - Try converting images to PNG or JPG format
        
        **Slow Performance?**
        - Use GPU if available (check device info in sidebar)
        - Reduce image size using the slider in the sidebar
        - Process fewer images at once in batch mode
        
        **Need Help?**
        - Check the "About" tab for technical details
        - Review the GitHub repository for documentation
        - Check error messages for specific issues
        """)
    
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
            st.warning(f"[WARNING] Model not found at: {model_path_input}")
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
                with st.expander("How to fix this in Streamlit Cloud", expanded=True):
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
                            st.info(f"Try using path: {model_save_path}")
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
            value=st.session_state.get('img_size', 224),
            step=32,
            help="Input image size for the model"
        )
        st.session_state.img_size = img_size
        
        # Clear cache button
        if st.button("Clear Cache", use_container_width=True):
            st.cache_resource.clear()
            st.rerun()
    
    # Main content area - Route to appropriate page
    current_page = st.session_state.get('current_page', 'Home')
    
    # Get image size from sidebar (will be set in sidebar section)
    img_size = st.session_state.get('img_size', 224)
    
    # Route to page
    if current_page == 'Home':
        page_home(img_size)
    elif current_page == 'Batch':
        page_batch(img_size)
    elif current_page == 'Research':
        page_research()
    elif current_page == 'About':
        page_about()
    else:
        page_home(img_size)
    
    # Store img_size in session state for use in pages
    if 'img_size' not in st.session_state:
        st.session_state.img_size = 224
    # Old tab code removed - now using page functions above


# Streamlit automatically runs the script
# Call main() to start the app with error handling
if __name__ == "__main__" or True:  # Always run for Streamlit
    try:
        main()
    except Exception as e:
        # Display error to user if app fails to start
        st.error(f"Application Error: {str(e)}")
        st.exception(e)
        st.info("Please check the logs for more details or contact support.")

