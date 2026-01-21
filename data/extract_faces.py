import os
import cv2
import torch
from facenet_pytorch import MTCNN
from tqdm import tqdm

# --- CONFIGURATION ---
VIDEO_DIR = "data/videos"
OUTPUT_DIR = "data/train"
FRAMES_PER_VIDEO = 15  # Extract 15 faces per video (Total ~3000 images)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def extract_faces():
    print(f"🕵️ Initializing MTCNN Face Detector on {DEVICE}...")
    mtcnn = MTCNN(keep_all=False, select_largest=True, device=DEVICE)
    
    for category in ["Real", "Fake"]:
        input_path = os.path.join(VIDEO_DIR, category)
        output_path = os.path.join(OUTPUT_DIR, category)
        os.makedirs(output_path, exist_ok=True)
        
        videos = os.listdir(input_path)
        print(f"\nProcessing {len(videos)} {category} videos...")
        
        for video_name in tqdm(videos):
            if not video_name.endswith(('.mp4', '.avi', '.mov')):
                continue
                
            vid_path = os.path.join(input_path, video_name)
            cap = cv2.VideoCapture(vid_path)
            
            # Get video length to skip frames evenly
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0: continue
            
            step = max(1, total_frames // FRAMES_PER_VIDEO)
            
            frame_count = 0
            saved_count = 0
            
            while cap.isOpened() and saved_count < FRAMES_PER_VIDEO:
                ret, frame = cap.read()
                if not ret: break
                
                # Only process every Nth frame
                if frame_count % step == 0:
                    # Convert BGR (OpenCV) to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Detect and Crop (MTCNN does this in one line!)
                    # save_path automatically saves the file
                    save_name = f"{video_name.split('.')[0]}_frame{frame_count}.jpg"
                    save_path = os.path.join(output_path, save_name)
                    
                    try:
                        mtcnn(frame_rgb, save_path=save_path)
                        saved_count += 1
                    except Exception:
                        pass # Face detection failed for this frame, skip it
                
                frame_count += 1
                
            cap.release()

    print("\n✨ Face Extraction Complete!")
    print(f"Check your training data at: {OUTPUT_DIR}")

if __name__ == "__main__":
    extract_faces()