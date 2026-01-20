import numpy as np
import cv2
import matplotlib.pyplot as plt

def create_spectrum_plot(image_tensor):
    
   
    img_np = image_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    
   
    f = np.fft.fft2(img_gray)
    fshift = np.fft.fftshift(f)
    
   
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-9)
    
   
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(magnitude_spectrum, cmap='inferno')
    ax.set_title("Frequency Analysis (DFT)")
    ax.axis('off')
    
    return fig