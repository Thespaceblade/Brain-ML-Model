"""
Brain Bleeding Classification Model
Uses transfer learning with ResNet50 backbone
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights, EfficientNet_B0_Weights


class BrainBleedingClassifier(nn.Module):
    """
    CNN model for binary classification of brain bleeding.
    Uses ResNet50 as backbone with transfer learning.
    """
    
    def __init__(self, num_classes=2, pretrained=True, freeze_backbone=False):
        """
        Initialize the model.
        
        Args:
            num_classes: Number of classes (2 for binary classification)
            pretrained: Whether to use pretrained ImageNet weights
            freeze_backbone: Whether to freeze the backbone for fine-tuning
        """
        super(BrainBleedingClassifier, self).__init__()
        
        # Load ResNet50 backbone
        if pretrained:
            self.backbone = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        else:
            self.backbone = models.resnet50(weights=None)
        
        # Freeze backbone if specified (for fine-tuning)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Get the number of features from the last layer
        num_features = self.backbone.fc.in_features
        
        # Replace the final fully connected layer
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        """Forward pass through the network."""
        return self.backbone(x)


class EfficientNetClassifier(nn.Module):
    """
    Alternative model using EfficientNet for brain bleeding classification.
    """
    
    def __init__(self, num_classes=2, model_name='efficientnet_b0', pretrained=True):
        """
        Initialize EfficientNet model.
        
        Args:
            num_classes: Number of classes
            model_name: EfficientNet variant (b0-b7)
            pretrained: Whether to use pretrained weights
        """
        super(EfficientNetClassifier, self).__init__()
        
        try:
            import torchvision.models as models
            # Load EfficientNet
            weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
            if model_name == 'efficientnet_b0':
                self.backbone = models.efficientnet_b0(weights=weights)
            elif model_name == 'efficientnet_b1':
                self.backbone = models.efficientnet_b1(weights=weights)
            elif model_name == 'efficientnet_b2':
                self.backbone = models.efficientnet_b2(weights=weights)
            else:
                self.backbone = models.efficientnet_b0(weights=weights)
            
            # Replace classifier
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_features, num_classes)
            )
        except AttributeError:
            # Fallback to ResNet if EfficientNet not available
            print("EfficientNet not available, using ResNet50 instead")
            if pretrained:
                self.backbone = models.resnet50(weights=ResNet50_Weights.DEFAULT)
            else:
                self.backbone = models.resnet50(weights=None)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_features, num_classes)
            )
    
    def forward(self, x):
        """Forward pass through the network."""
        return self.backbone(x)


def get_model(model_name='resnet50', num_classes=2, pretrained=True, freeze_backbone=False):
    """
    Factory function to get a model.
    
    Args:
        model_name: Name of the model ('resnet50' or 'efficientnet')
        num_classes: Number of classes
        pretrained: Whether to use pretrained weights
        freeze_backbone: Whether to freeze backbone (for fine-tuning)
    
    Returns:
        Initialized model
    """
    if model_name == 'resnet50':
        return BrainBleedingClassifier(
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone
        )
    elif model_name.startswith('efficientnet'):
        return EfficientNetClassifier(
            num_classes=num_classes,
            model_name=model_name,
            pretrained=pretrained
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")

