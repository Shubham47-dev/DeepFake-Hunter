import torch
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import cv2

def generate_heatmap(model, input_tensor, original_image):
    
    
    target_layers = [model.rgb_branch.features[-1]]

  
    cam = GradCAM(model=model, target_layers=target_layers)

    targets = [ClassifierOutputTarget(1)]
    
    
    dummy_freq = torch.zeros_like(input_tensor) 
    
    
    
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
    
   
    rgb_img = np.float32(original_image) / 255.0
    rgb_img = cv2.resize(rgb_img, (256, 256))
    
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    
    return visualization
    class RGBWrapper(torch.nn.Module):
        def __init__(self, full_model):
            super().__init__()
            self.main = full_model.rgb_branch 
        def forward(self, x):
            return self.main(x)

cam = GradCAM(model=RGBWrapper(model), target_layers=[model.rgb_branch.features[-1]])