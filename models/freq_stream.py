# models/freq_stream.py
import torch
import torch.nn as nn

class FrequencyStream(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: 1 Channel (Grayscale DCT map), 299x299 resolution
        self.conv_block = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2), # 299 -> 149
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2), # 149 -> 74
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2), # 74 -> 37
            
            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)) # Crush everything to 1x1 pixel
        )
        
    def forward(self, x):
        # x shape: (Batch, 1, 299, 299)
        x = self.conv_block(x)
        x = x.flatten(1) # Flatten to vector
        return x # Output shape: (Batch, 256)