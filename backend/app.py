import cv2
import numpy as np
import os
import tempfile
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
import sys

# Add current directory to path so we can import from local modules if run directly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from inference import AnomalyDetector
from utils.video_utils import resize_frame_smart, get_video_properties, create_video_writer

# --- Configuration ---
# ROI will cover the lower half of the frame (like seats area in the image)
ROI_HEIGHT_RATIO = 0.5  # Use bottom 50% of frame height
ANOMALY_THRESHOLD = -0.05
PATCH_SIZE = 224
STRIDE = 224  # No overlap for faster processing
FRAME_SKIP = 30  # Process every 30th frame for speed
TARGET_RESIZE_WIDTH = 480 # Reduced from 640 to save memory

# Global detector instance
detector = AnomalyDetector(model_dir=os.path.join(os.path.dirname(__file__), "model"))

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

@app.get("/")
async def root():
    return {"message": "Surveillance Anomaly Detection API is running (Backend Structure)"}

@app.post("/analyze-video")
async def analyze_video(file: UploadFile = File(...)):
    if detector.vit_model is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    # 1. Save uploaded file to temp
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, file.filename)
    output_path = os.path.join(temp_dir, f"analyzed_{file.filename}")
    
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
        
        print(f"Processing at resolution: {width}x{height} (Original: {orig_width}x{orig_height})")

        # Create video writer
        out, actual_output_path = create_video_writer(output_path, fps, width, height)
        output_path = actual_output_path # Update to the actual path used (might be .webm)

        # Calculate ROI: Full width, lower half of frame
        roi_x = 0
        roi_y = int(height * (1 - ROI_HEIGHT_RATIO))
        roi_w = width
        roi_h = int(height * ROI_HEIGHT_RATIO)

        anomaly_detected = False
        anomaly_frames = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Resize frame for speed
            frame = resize_frame_smart(frame, target_width=TARGET_RESIZE_WIDTH)
            
            # Use the calculated ROI position
            x, y, w, h = roi_x, roi_y, roi_w, roi_h
            
            # Draw ROI border
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, "RESTRICTED ZONE", (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            frame_has_anomaly = False
            
            # Log progress
            if frame_idx % 100 == 0:
                print(f"Processing frame {frame_idx}...")
            
            # Process every Nth frame
            if frame_idx % FRAME_SKIP == 0 and w >= PATCH_SIZE and h >= PATCH_SIZE:
                roi_frame = frame[y:y+h, x:x+w]
                
                # Extract patches
                patches = []
                patch_coords = []
                for i in range(0, h - PATCH_SIZE + 1, STRIDE):
                    for j in range(0, w - PATCH_SIZE + 1, STRIDE):
                        patch = roi_frame[i:i+PATCH_SIZE, j:j+PATCH_SIZE]
                        patches.append(detector.preprocess_patch(patch))
                        patch_coords.append((x + j, y + i))
                
                if patches:
                    features = detector.extract_features(patches)
                    scores = detector.score_anomalies(features)
                    
                    # Mark anomalies
                    for score, (px, py) in zip(scores, patch_coords):
                        if score < ANOMALY_THRESHOLD:
                            frame_has_anomaly = True
                            cv2.rectangle(frame, (px, py), (px + PATCH_SIZE, py + PATCH_SIZE), (0, 0, 255), 2)
                            cv2.putText(frame, f"{score:.2f}", (px + 5, py + 15),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                    
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

        cap.release()
        out.release()

        # Determine media type based on extension
        res_media_type = "video/webm" if output_path.endswith('.webm') else "video/mp4"

        return FileResponse(
            path=output_path,
            media_type=res_media_type,
            filename=os.path.basename(output_path),
            headers={
                "X-Anomaly-Detected": str(anomaly_detected),
                "X-Anomaly-Frames": ",".join(map(str, anomaly_frames)),
                "X-Alert-Message": "Restricted Zone Anomaly Detected" if anomaly_detected else "No Anomaly Detected"
            }
        )

    except Exception as e:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
