import torch
import torchvision
from torchvision import transforms
import cv2
import joblib
import numpy as np
import os
import tempfile
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager

# --- Configuration ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
VIT_MODEL_PATH = "vit_feature_extractor.pth"
ANOMALY_MODEL_PATH = "isolation_forest_model.joblib"
# ROI will cover the lower half of the frame (like seats area in the image)
ROI_HEIGHT_RATIO = 0.5  # Use bottom 50% of frame height
ANOMALY_THRESHOLD = -0.05
PATCH_SIZE = 224
STRIDE = 224  # No overlap for faster processing
FRAME_SKIP = 20  # Process every 20th frame for speed

# Global model containers
models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models at startup
    try:
        print(f"Loading models onto {DEVICE}...")
        
        # 1. Feature Extractor
        model_vit = torchvision.models.vit_b_16(pretrained=False)
        state_dict = torch.load(VIT_MODEL_PATH, map_location=DEVICE)
        model_vit.load_state_dict(state_dict, strict=False)
        model_vit.heads = torch.nn.Identity()
        model_vit.to(DEVICE)
        model_vit.eval()
        models["vit"] = model_vit

        # 2. Anomaly Detector
        models["anomaly"] = joblib.load(ANOMALY_MODEL_PATH)
        
        print("Models loaded successfully.")
    except Exception as e:
        print(f"Error loading models: {e}")
        # We don't exit here so the process can be debugged, but API calls will fail
    
    yield
    # Clean up on shutdown
    models.clear()

app = FastAPI(title="Anomaly Detection API", lifespan=lifespan)

# --- Preprocessing ---
vit_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@app.get("/")
async def root():
    return {"message": "Surveillance Anomaly Detection API is running"}

@app.post("/analyze-video")
async def analyze_video(file: UploadFile = File(...)):
    if "vit" not in models or "anomaly" not in models:
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
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # --- OPTIMIZATION: Resize heavy 4K/2K videos ---
        TARGET_WIDTH = 640
        if orig_width > TARGET_WIDTH:
            scale_ratio = TARGET_WIDTH / orig_width
            width = TARGET_WIDTH
            height = int(orig_height * scale_ratio)
        else:
            width = orig_width
            height = orig_height
            
        print(f"Processing at resolution: {width}x{height} (Original: {orig_width}x{orig_height})")

        # Create video writer with NEW dimensions
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Calculate ROI: Full width, lower half of frame (like seats in the image)
        roi_x = 0  # Start from left edge
        roi_y = int(height * (1 - ROI_HEIGHT_RATIO))  # Start from middle
        roi_w = width  # Full width
        roi_h = int(height * ROI_HEIGHT_RATIO)  # Lower half

        anomaly_detected = False
        anomaly_frames = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Resize frame for speed optimization
            if width != orig_width:
                frame = cv2.resize(frame, (width, height))
            
            # Use the calculated ROI position
            x, y, w, h = roi_x, roi_y, roi_w, roi_h
            
            # Draw ROI border on every frame
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, "RESTRICTED ZONE", (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            frame_has_anomaly = False
            
            # Log progress every 100 frames
            if frame_idx % 100 == 0:
                print(f"Processing frame {frame_idx}...")
            
            # Process every Nth frame for speed
            if frame_idx % FRAME_SKIP == 0 and w >= PATCH_SIZE and h >= PATCH_SIZE:
                roi_frame = frame[y:y+h, x:x+w]
                
                # Extract patches for this frame
                patches = []
                patch_coords = []
                for i in range(0, h - PATCH_SIZE + 1, STRIDE):
                    for j in range(0, w - PATCH_SIZE + 1, STRIDE):
                        patch = roi_frame[i:i+PATCH_SIZE, j:j+PATCH_SIZE]
                        patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
                        patches.append(vit_transform(patch_rgb))
                        patch_coords.append((x + j, y + i))
                
                if patches:
                    batch_tensors = torch.stack(patches).to(DEVICE)
                    with torch.no_grad():
                        features = models["vit"](batch_tensors).cpu().numpy()
                    
                    # Check for anomalies in the batch
                    scores = models["anomaly"].decision_function(features)
                    
                    # Mark anomalous patches
                    for score, (px, py) in zip(scores, patch_coords):
                        if score < ANOMALY_THRESHOLD:
                            frame_has_anomaly = True
                            cv2.rectangle(frame, (px, py), (px + PATCH_SIZE, py + PATCH_SIZE), (0, 0, 255), 2)
                            cv2.putText(frame, f"{score:.2f}", (px + 5, py + 15),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                    
                    if frame_has_anomaly:
                        anomaly_detected = True
                        anomaly_frames.append(frame_idx)
                        # Add alert text
                        cv2.putText(frame, "!!! ANOMALY DETECTED !!!", (20, 40),
                                   cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255), 2)

            # Write frame to output video
            out.write(frame)
            frame_idx += 1

        cap.release()
        out.release()

        # Return the annotated video file
        return FileResponse(
            path=output_path,
            media_type="video/mp4",
            filename=f"analyzed_{file.filename}",
            headers={
                "X-Anomaly-Detected": str(anomaly_detected),
                "X-Anomaly-Frames": ",".join(map(str, anomaly_frames)),
                "X-Alert-Message": "Restricted Zone Anomaly Detected" if anomaly_detected else "No Anomaly Detected"
            }
        )

    except Exception as e:
        # Cleanup on error
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
