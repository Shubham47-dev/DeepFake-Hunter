import torch.nn as nn

class DeepFakeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # Combine Sigmoid + BCELoss for numerical stability
        self.criterion = nn.BCEWithLogitsLoss()
        
    def forward(self, predictions, targets):
        # Flatten targets to match output shape
        return self.criterion(predictions.view(-1), targets)