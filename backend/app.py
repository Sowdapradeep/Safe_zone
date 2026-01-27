import cv2
import numpy as np
import os
import tempfile
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
import sys

# Add current directory to path so we can import from local modules if run directly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from inference import AnomalyDetector
from utils.video_utils import resize_frame_smart, get_video_properties, create_video_writer
import base64
import time

# --- Configuration ---
# ROI will cover the lower half of the frame (like seats area in the image)
ROI_HEIGHT_RATIO = 0.5  # Use bottom 50% of frame height
ANOMALY_THRESHOLD = -0.05
PATCH_SIZE = 224
STRIDE = 224  # No overlap for faster processing
FRAME_SKIP = 60  # Process every 60th frame (approx every 2 seconds at 30fps)
TARGET_RESIZE_WIDTH = 800 # Increased to ensure ROI > 224px height

# Global detector instance
detector = AnomalyDetector(model_dir=os.path.join(os.path.dirname(__file__), "model"))

# Persistent storage for analyzed videos (for local/render serving)
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "analyzed_videos")
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models at startup
    try:
        detector.load_models()
    except Exception as e:
        print(f"Error loading models: {e}")
    yield
    # Cleanup if needed

app = FastAPI(title="Anomaly Detection API", lifespan=lifespan)

# Add CORS Middleware to allow requests from Vercel/Render
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, replace with specific Vercel URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Surveillance Anomaly Detection API is running (Cyber Blue Edition)"}

@app.get("/api/video/{filename}")
async def get_video(filename: str):
    file_path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Video not found")
    return FileResponse(file_path, media_type="video/webm" if filename.endswith('.webm') else "video/mp4")

@app.post("/analyze-video")
async def analyze_video(file: UploadFile = File(...)):
    print(f"DEBUG: analyze_video called with {file.filename}")
    if detector.vit_model is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    # 1. Save uploaded file to temp
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, file.filename)
    
    # Analyze and save to persistent output dir
    output_filename = f"analyzed_{int(time.time())}_{file.filename}"
    if output_filename.endswith('.mp4'):
        output_filename = output_filename.replace('.mp4', '.webm')
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    
    cap = None
    out = None
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 2. Open Video
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Could not open video file")

        # Get video properties
        fps, orig_width, orig_height, _ = get_video_properties(cap)
        
        # Determine processing resolution (Optimization)
        dummy_frame = np.zeros((orig_height, orig_width, 3), dtype=np.uint8)
        resized_dummy = resize_frame_smart(dummy_frame, target_width=TARGET_RESIZE_WIDTH)
        height, width = resized_dummy.shape[:2]
        
        print(f"DEBUG: Processing at resolution: {width}x{height} (Original: {orig_width}x{orig_height})")

        # Create video writer
        out, actual_output_path = create_video_writer(output_path, fps, width, height)
        output_path = actual_output_path # Update to the actual path used (might be .webm)

        # Calculate ROI: Full width, lower portion of frame
        # User requested increased size, so we increase ratio from 0.5 to 0.75
        ROI_HEIGHT_RATIO = 0.75 
        roi_x = 0
        roi_y = int(height * (1 - ROI_HEIGHT_RATIO))
        roi_w = width
        roi_h = int(height * ROI_HEIGHT_RATIO)

        # MOG2 Background Subtractor for Motion Detection
        fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=True)

        anomaly_detected = False
        anomaly_frames = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"DEBUG: End of video at frame {frame_idx}")
                break
            
            # Use the calculated ROI position
            x, y, w, h = roi_x, roi_y, roi_w, roi_h
            
            # Resize frame for speed
            frame = resize_frame_smart(frame, target_width=TARGET_RESIZE_WIDTH)
            
            # Draw ROI border
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, "RESTRICTED ZONE (Motion-Active)", (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            frame_has_anomaly = False
            
            # Log progress
            if frame_idx % 100 == 0:
                print(f"Processing frame {frame_idx}...")
            
            # Process frames
            roi_frame = frame[y:y+h, x:x+w]
            
            # 1. Apply Background Subtraction
            fgmask = fgbg.apply(roi_frame)
            _, fgmask = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
            
            # 2. Find Contours (Moving Objects)
            contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            candidates = []
            candidate_boxes = []
            
            for contour in contours:
                if cv2.contourArea(contour) > 800: # Min area threshold
                    cx, cy, cw, ch = cv2.boundingRect(contour)
                    
                    # Expand box slightly
                    pad = 10
                    cx1, cy1 = max(0, cx - pad), max(0, cy - pad)
                    cx2, cy2 = min(w, cx + cw + pad), min(h, cy + ch + pad)
                    
                    # Ensure patch is valid size (closest to PATCH_SIZE if possible, or just crop)
                    # For ViT, we resize in preprocess, so any size crop works as long as it's not empty
                    crop = roi_frame[cy1:cy2, cx1:cx2]
                    
                    if crop.size > 0:
                        candidates.append(detector.preprocess_patch(crop))
                        candidate_boxes.append((x + cx1, y + cy1, cx2 - cx1, cy2 - cy1))

            # 3. Analyze Candidates
            # Only run inference if we have candidates and we are on a check frame
            # (or run every frame if 'candidates' exists, but let's stick to skip for speed if many objects)
            if candidates and (frame_idx % (FRAME_SKIP // 2) == 0): # Check more often than blind grid (30 frames)
                 print(f"DEBUG: Running inference on {len(candidates)} candidates at frame {frame_idx}")
                 features = detector.extract_features(candidates)
                 scores = detector.score_anomalies(features)
                 
                 for score, (bx, by, bw, bh) in zip(scores, candidate_boxes):
                     # Visual feedback for detected objects
                     if score < ANOMALY_THRESHOLD:
                         frame_has_anomaly = True
                         cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (0, 0, 255), 2)
                         cv2.putText(frame, f"ANOMALY {score:.2f}", (bx, by - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                     else:
                         # Valid object detected but not anomalous enough
                         cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (0, 255, 0), 1)
            
            if frame_has_anomaly:
                anomaly_detected = True
                anomaly_frames.append(frame_idx)
                cv2.putText(frame, "!!! ANOMALY DETECTED !!!", (20, 40),
                           cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255), 2)

            # Write frame
            out.write(frame)
            frame_idx += 1
            
            # Periodically force garbage collection
            if frame_idx % 300 == 0:
                import gc
                gc.collect()

        # Encode frames to base64 for reliable transport
        # ... (rest of the logic remains) ...
        return {
            "status": "success",
            "anomaly_detected": anomaly_detected,
            "anomaly_frames": anomaly_frames,
            "total_frames": frame_idx,
            "video_url": f"/api/video/{os.path.basename(output_path)}",
            "filename": os.path.basename(output_path)
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"DEBUG: Exception in analyze_video: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    
    finally:
        if cap is not None:
            cap.release()
        if out is not None:
            out.release()
            
        # Clean up temp file
        if os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
            except Exception as cleanup_img:
                print(f"DEBUG: Failed to cleanup temp dir: {cleanup_img}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
