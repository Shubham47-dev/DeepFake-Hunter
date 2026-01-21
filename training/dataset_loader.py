import os
import cv2
import torch
import numpy as np
import scipy.fftpack as fft
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class DualStreamDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        root_dir: Path to folder containing 'Real' and 'Fake' subfolders.
        Example: data/train/
        """
        self.root_dir = root_dir
        self.image_paths = []
        self.labels = []
        
        # 1. Auto-scan for Real (Label 0) and Fake (Label 1)
        classes = {'Real': 0, 'Fake': 1}
        for cls_name, cls_idx in classes.items():
            cls_folder = os.path.join(root_dir, cls_name)
            if not os.path.exists(cls_folder):
                continue
                
            for img_name in os.listdir(cls_folder):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(cls_folder, img_name))
                    self.labels.append(cls_idx)
        
        # 2. Define Transforms (Resize to Xception input size)
        self.transform = transform
        if self.transform is None:
            self.transform = A.Compose([
                A.Resize(299, 299),
                A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                ToTensorV2()
            ])
            
        # Transform specifically for Frequency (Grayscale -> Tensor)
        self.freq_transform = A.Compose([
            A.Resize(299, 299),
            ToTensorV2()
        ])

    def generate_dct(self, image):
        """
        Converts RGB Image -> Grayscale -> Discrete Cosine Transform (DCT)
        This reveals the invisible 'grid' artifacts of Deepfakes.
        """
        # Convert to Grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply 2D DCT (Discrete Cosine Transform)
        dct_obj = fft.dct(fft.dct(gray.T, norm='ortho').T, norm='ortho')
        
        # Use Log Scale because values vary wildly
        dct_log = np.log(np.abs(dct_obj) + 1e-12) # +epsilon to avoid log(0)
        
        # Normalize to 0-1 range for the neural net
        dct_norm = (dct_log - dct_log.min()) / (dct_log.max() - dct_log.min())
        
        # Add channel dimension (H, W) -> (H, W, 1)
        return dct_norm.astype(np.float32)[:, :, np.newaxis]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load Image (RGB)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Generate Frequency Map (The "Hard Mode" Feature)
        dct_map = self.generate_dct(image)
        
        # Apply Transforms
        if self.transform:
            image_tensor = self.transform(image=image)['image']
            freq_tensor = self.freq_transform(image=dct_map)['image']
            
        return image_tensor, freq_tensor, torch.tensor(label, dtype=torch.float32)