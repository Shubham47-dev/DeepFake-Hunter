# models/rgb_stream.py
import torch
import torch.nn as nn
import timm

class RGBStream(nn.Module):
    def __init__(self):
        super().__init__()
        # Load Xception pre-trained on ImageNet
        # num_classes=0 means "Give me the features, not the classification"
        self.backbone = timm.create_model('xception', pretrained=True, num_classes=0)
        
    def forward(self, x):
        # x shape: (Batch, 3, 299, 299)
        features = self.backbone(x)
        return features # Output shape: (Batch, 2048)