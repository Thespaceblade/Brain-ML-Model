"""Check if all required dependencies are installed."""

import sys

dependencies = {
    'torch': 'PyTorch',
    'torchvision': 'Torchvision',
    'kagglehub': 'Kagglehub',
    'albumentations': 'Albumentations',
    'sklearn': 'Scikit-learn',
    'pandas': 'Pandas',
    'numpy': 'NumPy',
    'PIL': 'Pillow',
    'matplotlib': 'Matplotlib',
    'seaborn': 'Seaborn',
    'cv2': 'OpenCV',
    'tqdm': 'tqdm'
}

missing = []
installed = []

for module, name in dependencies.items():
    try:
        if module == 'cv2':
            import cv2
        elif module == 'PIL':
            from PIL import Image
        elif module == 'sklearn':
            import sklearn
        else:
            __import__(module)
        installed.append(name)
        print(f"[OK] {name} is installed")
    except ImportError:
        missing.append(name)
        print(f"[MISSING] {name} is NOT installed")

print("\n" + "="*50)
if missing:
    print(f"Missing dependencies: {', '.join(missing)}")
    print("\nPlease install them using:")
    print("pip install -r requirements.txt")
    sys.exit(1)
else:
    print("All dependencies are installed!")
    print("="*50)
    sys.exit(0)

