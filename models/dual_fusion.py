# models/dual_fusion.py
import torch
import torch.nn as nn
from .rgb_stream import RGBStream
from .freq_stream import FrequencyStream

class DualBranchDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.rgb_stream = RGBStream()       # Output: 2048
        self.freq_stream = FrequencyStream() # Output: 256
        
        # Combined size = 2048 + 256 = 2304
        self.classifier = nn.Sequential(
            nn.Linear(2304, 512),
            nn.ReLU(),
            nn.Dropout(0.5), # Prevent overfitting
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 1) # Final output (logit)
        )
        
    def forward(self, rgb_img, freq_img):
        # 1. Get features from both eyes
        rgb_feat = self.rgb_stream(rgb_img)
        freq_feat = self.freq_stream(freq_img)
        
        # 2. Fuse them (Concatenate)
        combined = torch.cat((rgb_feat, freq_feat), dim=1)
        
        # 3. Classify
        output = self.classifier(combined)
        return output