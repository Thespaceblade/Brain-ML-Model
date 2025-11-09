"""
Prediction script for brain bleeding classification model.
"""

import argparse
import os
import sys
import torch
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import get_model
from src.utils import load_checkpoint, get_device


def predict_image(model, image_path, device, img_size=224):
    """
    Predict on a single image.
    
    Args:
        model: Trained model
        image_path: Path to image file
        device: Device to run on
        img_size: Image size for preprocessing
    
    Returns:
        Tuple of (prediction, confidence)
    """
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image = np.array(image)
    
    # Transform
    transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])
    
    transformed = transform(image=image)
    image_tensor = transformed['image'].unsqueeze(0).to(device)
    
    # Predict
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    class_names = ['No Bleeding', 'Bleeding']
    prediction = class_names[predicted.item()]
    confidence = confidence.item()
    
    return prediction, confidence, probabilities[0].cpu().numpy()


def predict(args):
    """
    Main prediction function.
    
    Args:
        args: Command line arguments
    """
    # Set device
    device = get_device()
    
    # Load model
    print(f"Loading model from {args.model_path}")
    model = get_model(
        model_name=args.model_name,
        num_classes=2,
        pretrained=False
    )
    model = model.to(device)
    
    # Load checkpoint
    load_checkpoint(args.model_path, model)
    
    # Predict on single image
    if args.image_path:
        if not os.path.exists(args.image_path):
            print(f"Error: Image not found: {args.image_path}")
            return
        
        print(f"\nPredicting on: {args.image_path}")
        print("-" * 60)
        
        prediction, confidence, probabilities = predict_image(
            model, args.image_path, device, args.img_size
        )
        
        print(f"Prediction: {prediction}")
        print(f"Confidence: {confidence:.4f} ({confidence * 100:.2f}%)")
        print(f"\nProbabilities:")
        print(f"  No Bleeding: {probabilities[0]:.4f} ({probabilities[0] * 100:.2f}%)")
        print(f"  Bleeding: {probabilities[1]:.4f} ({probabilities[1] * 100:.2f}%)")
    
    # Predict on directory
    elif args.image_dir:
        if not os.path.exists(args.image_dir):
            print(f"Error: Directory not found: {args.image_dir}")
            return
        
        print(f"\nPredicting on images in: {args.image_dir}")
        print("-" * 60)
        
        # Get all image files
        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
        image_files = [
            f for f in os.listdir(args.image_dir)
            if f.lower().endswith(image_extensions)
        ]
        
        if len(image_files) == 0:
            print("No image files found in directory")
            return
        
        results = []
        for img_file in image_files:
            img_path = os.path.join(args.image_dir, img_file)
            prediction, confidence, probabilities = predict_image(
                model, img_path, device, args.img_size
            )
            results.append({
                'image': img_file,
                'prediction': prediction,
                'confidence': confidence,
                'prob_no_bleeding': probabilities[0],
                'prob_bleeding': probabilities[1]
            })
            print(f"{img_file}: {prediction} ({confidence * 100:.2f}%)")
        
        # Save results to CSV if requested
        if args.save_results:
            import pandas as pd
            df = pd.DataFrame(results)
            output_path = os.path.join(args.image_dir, 'predictions.csv')
            df.to_csv(output_path, index=False)
            print(f"\nResults saved to: {output_path}")
    else:
        print("Error: Please provide either --image_path or --image_dir")
        return


def main():
    parser = argparse.ArgumentParser(description='Predict brain bleeding from images')
    
    # Model arguments
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--model_name', type=str, default='resnet50',
                        choices=['resnet50', 'efficientnet_b0', 'efficientnet_b1'],
                        help='Model architecture')
    
    # Input arguments
    parser.add_argument('--image_path', type=str, default=None,
                        help='Path to single image file')
    parser.add_argument('--image_dir', type=str, default=None,
                        help='Path to directory containing images')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Image size for preprocessing')
    
    # Output arguments
    parser.add_argument('--save_results', action='store_true', default=False,
                        help='Save results to CSV file')
    
    args = parser.parse_args()
    
    predict(args)


if __name__ == '__main__':
    main()



