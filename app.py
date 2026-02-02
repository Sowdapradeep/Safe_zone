import torch
import torchvision
from torchvision import transforms
import cv2
import joblib
import numpy as np
import os
import tempfile
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.responses import FileResponse, StreamingResponse
from contextlib import asynccontextmanager
import asyncio
import json
import time
import uuid

connected_clients = set()
jobs = {} # Global job store: { job_id: { status: 'processing', progress: 0, results: None, timestamp: time.time() } }

def cleanup_temp(path: str, retries: int = 5, delay: float = 1.0):
    """Cleanup temporary files with retry logic for Windows file locking."""
    if not os.path.exists(path):
        return
        
    for i in range(retries):
        try:
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
            else:
                os.remove(path)
            # Check if it's actually gone
            if not os.path.exists(path):
                print(f"Successfully cleaned up: {path}")
                return
            time.sleep(delay)
        except Exception as e:
            if i < retries - 1:
                time.sleep(delay)
            else:
                print(f"Failed to cleanup {path} after {retries} attempts: {e}")

# --- Configuration ---

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DETECTOR_MODEL_PATH = "backend/model/object_detection_model (1).pth"
VIT_MODEL_PATH = "backend/model/vit_feature_extractor.pth"
ANOMALY_MODEL_PATH = "backend/model/isolation_forest_model.joblib"

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Surveillance-relevant classes (COCO indices)
# 1=person, 2=bicycle, 3=car, 4=motorcycle, 6=bus, 8=truck
RELEVANT_CLASS_IDS = {1, 2, 3, 4, 6, 8}

# Global model container
models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models at startup
    try:
        print(f"Loading models onto {DEVICE}...")
        
        # 1. Object Detector
        detector = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False)
        detector.load_state_dict(torch.load(DETECTOR_MODEL_PATH, map_location=DEVICE))
        detector.to(DEVICE)
        detector.eval()
        models["detector"] = detector

        # 2. Feature Extractor
        model_vit = torchvision.models.vit_b_16(pretrained=False)
        state_dict = torch.load(VIT_MODEL_PATH, map_location=DEVICE)
        model_vit.load_state_dict(state_dict, strict=False)
        model_vit.heads = torch.nn.Identity()
        model_vit.to(DEVICE)
        model_vit.eval()
        models["vit"] = model_vit

        # 3. Anomaly Detector
        models["anomaly"] = joblib.load(ANOMALY_MODEL_PATH)
        
        print("All models loaded successfully.")
    except Exception as e:
        print(f"Error loading models: {e}")
    
    yield
    models.clear()

app = FastAPI(title="SafeZone Surveillance API", lifespan=lifespan)

from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Anomaly-Detected"]
)

# --- Preprocessing ---
def frame_to_tensor(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return torchvision.transforms.functional.to_tensor(img).to(DEVICE)

vit_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@app.get("/")
async def root():
    return {"message": "SafeZone Surveillance AI API is running"}

@app.get("/api/check-status/{job_id}")
async def check_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    return {
        "job_id": job_id,
        "status": job["status"],
        "progress": job["progress"],
        **(job["results"] or {})
    }

@app.get("/api/video/{filename}")
async def get_video(filename: str):
    # This serves analyzed videos from the job store's temporary directories
    # We need a way to map filename back to the temp dir or keep them in a central place
    # For simplicity, we'll look in the uploads/analyzed folder if we move them there
    # OR we store the output path in the job results
    
    # Let's find the job that produced this file
    target_job = None
    for jid, data in jobs.items():
        if data.get("results") and data["results"].get("filename") == filename:
            target_job = data
            break
            
    if not target_job or not target_job["results"].get("output_path"):
        # Fallback to a central analyzed folder if it exists
        central_path = os.path.join("backend", "analyzed", filename)
        if os.path.exists(central_path):
            return FileResponse(central_path, media_type="video/mp4")
        raise HTTPException(status_code=404, detail="Video not found")
        
    output_path = target_job["results"]["output_path"]
    if not os.path.exists(output_path):
         raise HTTPException(status_code=404, detail="Video file missing from server")
         
    return FileResponse(output_path, media_type="video/mp4", filename=filename)

async def run_analysis(job_id: str, file_path: str, output_path: str, filename: str, temp_dir: str):
    cap = None
    out = None
    try:
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            jobs[job_id] = {"status": "error", "message": "Could not open video file", "progress": 0}
            return

        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"[{job_id}] Starting analysis: {filename} ({orig_width}x{orig_height} @ {fps}fps, {total_frames} frames)")
        
        # Optimization resolution
        TARGET_WIDTH = 480
        scale = 1.0
        if orig_width > TARGET_WIDTH:
            scale = TARGET_WIDTH / orig_width
            width = TARGET_WIDTH
            height = int(orig_height * scale)
        else:
            width = orig_width
            height = orig_height
            
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        ZONE_POINTS = [
            (0, int(height * 0.4)),
            (width, int(height * 0.4)),
            (width, height),
            (0, height)
        ]
        zone_array = np.array(ZONE_POINTS, dtype=np.int32)

        SKIP_FRAMES = 25
        anomaly_detected = False
        anomaly_frames = []
        frame_count = 0
        last_detections = []
        fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if scale != 1.0:
                frame = cv2.resize(frame, (width, height))
            
            start_time = time.time()
            if frame_count % SKIP_FRAMES == 0:
                # Update progress
                if total_frames > 0:
                    progress = int((frame_count / total_frames) * 100)
                    jobs[job_id]["progress"] = progress

                last_detections = [] 
                input_tensor = frame_to_tensor(frame)
                with torch.no_grad():
                    detections = models["detector"]([input_tensor])[0]

                for i in range(len(detections['boxes'])):
                    box = detections['boxes'][i].cpu().numpy().astype(int)
                    score = detections['scores'][i].item()
                    label_id = detections['labels'][i].item()
                    
                    if label_id not in RELEVANT_CLASS_IDS or score < 0.4:
                        continue

                    x1, y1, x2, y2 = box
                    h, w, _ = frame.shape
                    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
                    if x2 - x1 < 10 or y2 - y1 < 10: continue

                    foot_x = int((x1 + x2) / 2)
                    foot_y = int(y2)
                    is_in_zone = cv2.pointPolygonTest(zone_array, (float(foot_x), float(foot_y)), False) >= 0

                    crop = frame[y1:y2, x1:x2]
                    try:
                        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                        vit_input = vit_transform(crop_rgb).unsqueeze(0).to(DEVICE)
                        with torch.no_grad():
                            features = models["vit"](vit_input).cpu().numpy()
                        
                        prediction = models["anomaly"].predict(features)[0]
                        anomaly_score = models["anomaly"].decision_function(features)[0]
                        
                        last_detections.append({
                            'box': (x1, y1, x2, y2),
                            'prediction': prediction,
                            'anomaly_score': anomaly_score,
                            'is_in_zone': is_in_zone,
                            'foot_pt': (foot_x, foot_y),
                            'category': "OBJECT"
                        })
                    except Exception: pass

            # Draw & Write
            cv2.polylines(frame, [zone_array], True, (255, 0, 0), 2)
            for det in last_detections:
                x1, y1, x2, y2 = det['box']
                if det['prediction'] == -1:
                    color, text = (0, 0, 255), "ANOMALY DETECTED"
                    anomaly_detected = True
                    if frame_count not in anomaly_frames: anomaly_frames.append(frame_count)
                elif det['is_in_zone']:
                    color, text = (0, 165, 255), "INTRUDER DETECTED"
                else:
                    color, text = (0, 255, 0), "TRACKING"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            out.write(frame)
            frame_count += 1

        # Completed
        jobs[job_id].update({
            "status": "completed",
            "progress": 100,
            "results": {
                "filename": f"analyzed_{filename}",
                "output_path": output_path,
                "anomaly_detected": anomaly_detected,
                "anomaly_frames": anomaly_frames[:20],
                "total_frames": total_frames,
                "fps": fps
            }
        })
        print(f"[{job_id}] Analysis complete for {filename}")

    except Exception as e:
        print(f"[{job_id}] Analysis Error: {e}")
        jobs[job_id] = {"status": "error", "message": str(e), "progress": 0}
    finally:
        if cap: cap.release()
        if out: out.release()
        # Keep results for 30 mins, then cleanup
        asyncio.get_event_loop().call_later(1800, cleanup_temp, temp_dir)

@app.post("/analyze-video")
async def analyze_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if "detector" not in models:
        raise HTTPException(status_code=503, detail="Models not loaded")

    job_id = str(uuid.uuid4())
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, file.filename)
    output_path = os.path.join(temp_dir, f"analyzed_{file.filename}")
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        jobs[job_id] = {
            "status": "processing",
            "progress": 0,
            "results": None,
            "timestamp": time.time()
        }
        
        background_tasks.add_task(run_analysis, job_id, file_path, output_path, file.filename, temp_dir)
        
        return {"job_id": job_id}

    except Exception as e:
        cleanup_temp(temp_dir)
        raise HTTPException(status_code=500, detail=f"Failed to start analysis: {str(e)}")

# Remove old analyze_video implementation from previous line range if needed

async def gen_live_frames():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam")
        return

    # Default Monitoring Zone (Lower 60% of the screen as a polygon)
    # We estimate dimensions if not known yet
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    
    ZONE_POINTS = [
        (0, int(height * 0.4)),
        (width, int(height * 0.4)),
        (width, height),
        (0, height)
    ]
    zone_array = np.array(ZONE_POINTS, dtype=np.int32)
    
    SKIP_FRAMES = 15 # Increased for CPU stability
    frame_count = 0
    last_detections = []
    fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break
            
            start_time = time.time()
            
            if frame_count % SKIP_FRAMES == 0:
                last_detections = []
                input_tensor = frame_to_tensor(frame)
                with torch.no_grad():
                    detections = models["detector"]([input_tensor])[0]

                for i in range(len(detections['boxes'])):
                    box = detections['boxes'][i].cpu().numpy().astype(int)
                    score = detections['scores'][i].item()
                    label_id = detections['labels'][i].item()
                    
                    if score > 0.4: # Slightly higher for live for stability
                        x1, y1, x2, y2 = box
                        h, w, _ = frame.shape
                        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
                        
                        foot_x = int((x1 + x2) / 2)
                        foot_y = y2
                        is_in_zone = cv2.pointPolygonTest(zone_array, (foot_x, foot_y), False) >= 0

                        crop = frame[y1:y2, x1:x2]
                        try:
                            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                            vit_input = vit_transform(crop_rgb).unsqueeze(0).to(DEVICE)
                            with torch.no_grad():
                                features = models["vit"](vit_input).cpu().numpy()
                            
                            prediction = models["anomaly"].predict(features)[0]
                            anomaly_score = models["anomaly"].decision_function(features)[0]
                            category = COCO_INSTANCE_CATEGORY_NAMES[label_id] if label_id < len(COCO_INSTANCE_CATEGORY_NAMES) else "Item"
                            
                            det = {
                                'box': (x1, y1, x2, y2),
                                'prediction': int(prediction),
                                'anomaly_score': float(anomaly_score),
                                'is_in_zone': bool(is_in_zone),
                                'foot_pt': (foot_x, foot_y),
                                'category': category
                            }
                            last_detections.append(det)

                            # Logic alert
                            if prediction == -1:
                                alert_data = {
                                    "timestamp": time.strftime("%H:%M:%S"),
                                    "confidence": float(1.0 - (anomaly_score + 1)/2), # Rough confidence
                                    "zone": "Restricted Zone",
                                    "type": "Anomaly Detected"
                                }
                                for client in connected_clients:
                                    try:
                                        await client.send_text(json.dumps(alert_data))
                                    except Exception:
                                        pass

                        except Exception as e:
                            pass

            # Visualization
            cv2.polylines(frame, [zone_array], True, (255, 0, 0), 2)
            for det in last_detections:
                x1, y1, x2, y2 = det['box']
                if det['prediction'] == -1:
                    color = (0, 0, 255) # Red
                    text = f"ANOMALY: {det['category']}"
                elif det['is_in_zone']:
                    color = (0, 165, 255) # Orange
                    text = f"INTRUSION: {det['category']}"
                else:
                    color = (0, 255, 0) # Green
                    text = f"{det['category']} ({det['anomaly_score']:.2f})"
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Encode frame
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            frame_count += 1
            # Small sleep to yield to event loop
            await asyncio.sleep(0.01)
    finally:
        cap.release()


@app.get("/live-feed")
async def live_feed():
    return StreamingResponse(gen_live_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.websocket("/ws/alerts")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.add(websocket)
    try:
        while True:
            await websocket.receive_text() # Keep alive
    except WebSocketDisconnect:
        connected_clients.remove(websocket)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

