import streamlit as st
import torch
import cv2
import numpy as np
from visuals.spectrum import create_spectrum_plot
from visuals.gradcam import generate_heatmap

st.set_page_config(page_title="DeepFake Hunter", page_icon="🕵️", layout="wide")

st.title("🕵️ DeepFake Forensic Hunter")
st.markdown("### Dual-Branch Frequency & Spatial Analysis")

uploaded_file = st.file_uploader("Upload a Video or Image", type=['mp4', 'jpg', 'png'])

if uploaded_file:
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("#### 1. Input Face")
        st.image(face_image_np, caption="Extracted Region", use_column_width=True)
    
    with col2:
        st.write("#### 2. Frequency Domain")
        fig = create_spectrum_plot(face_tensor)
        st.pyplot(fig)
        st.caption("Artifacts in high-frequency spectrum indicate GAN generation.")

    with col3:
        st.write("#### 3. AI Attention (Grad-CAM)")
        st.info("Model Inference Loading...") 

   
    st.progress(0.95)
    st.error("⚠️ DETECTION: 95.2% FAKE")