# ~/bird-detection/scripts/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class BirdClassifierCNN(nn.Module):
    """
    Compact CNN for bird audio classification.
    Designed for edge deployment on ESP32-S3.
    
    Input: Mel spectrogram (1, n_mels, time_frames)
    Output: Class logits (num_classes,)
    """
    
    def __init__(self, num_classes=6, n_mels=40):
        super().__init__()
        
        # Convolutional blocks
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.classifier(x)
        return x
    
    def count_parameters(self):
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class BirdClassifierSmall(nn.Module):
    """
    Even smaller model for very constrained deployment.
    ~10KB weights when quantized.
    """
    
    def __init__(self, num_classes=6, n_mels=40):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def get_model(model_name="standard", num_classes=6, n_mels=40):
    """Factory function to get model by name."""
    if model_name == "standard":
        return BirdClassifierCNN(num_classes, n_mels)
    elif model_name == "small":
        return BirdClassifierSmall(num_classes, n_mels)
    else:
        raise ValueError(f"Unknown model: {model_name}")


if __name__ == "__main__":
    # Test models
    for name in ["standard", "small"]:
        model = get_model(name, num_classes=6)
        x = torch.randn(1, 1, 40, 188)  # Batch, Channel, Mels, Time
        y = model(x)
        print(f"\n{name.upper()} Model:")
        print(f"  Parameters: {model.count_parameters():,}")
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {y.shape}")
