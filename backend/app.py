import cv2
import numpy as np
import os
import tempfile
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from contextlib import asynccontextmanager
import sys
import asyncio
import json
import time

# Add current directory to path so we can import from local modules if run directly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from inference import AnomalyDetector
from utils.video_utils import resize_frame_smart, get_video_properties, create_video_writer
from utils.tracker import CentroidTracker
import base64

# --- Configuration ---
ROI_HEIGHT_RATIO = 0.5
ANOMALY_THRESHOLD = -0.05
PATCH_SIZE = 224
STRIDE = 224
FRAME_SKIP = 90
TARGET_RESIZE_WIDTH = 640

# Global detector instance
detector = AnomalyDetector(model_dir=os.path.join(os.path.dirname(__file__), "model"))

# Persistent storage for analyzed videos
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "analyzed_videos")
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

# In-memory job store
jobs = {}
connected_clients = set()

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        detector.load_models()
        print("✅ Models ready and active.")
    except Exception as e:
        print(f"❌ Error loading models: {e}")
    yield

app = FastAPI(title="Anomaly Detection API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "message": "Surveillance Anomaly Detection API is running",
        "models_loaded": detector.vit_model is not None,
        "device": str(detector.device),
        "timestamp": time.time()
    }

@app.get("/api/video/{filename}")
async def get_video(filename: str):
    file_path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Video not found")
    return FileResponse(file_path, media_type="video/webm" if filename.endswith('.webm') else "video/mp4")

@app.get("/api/check-status/{job_id}")
async def check_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]

def get_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return None
    return cap

async def gen_live_frames():
    cap = get_camera()
    if not cap:
        # Fallback image if camera fails
        return
    try:
        while True:
            success, frame = cap.read()
            if not success: break
            ret, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            await asyncio.sleep(0.01)
    finally:
        if cap: cap.release()

@app.get("/live-feed")
async def live_feed():
    return StreamingResponse(gen_live_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.websocket("/ws/alerts")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.add(websocket)
    try:
        while True: await websocket.receive_text()
    except WebSocketDisconnect:
        connected_clients.remove(websocket)

def process_video_sync(job_id, file_path, output_path, output_filename):
    jobs[job_id] = {"status": "processing", "progress": 0}
    cap = None
    out = None
    try:
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened(): return {"status": "error", "message": "Could not open video file"}
        fps, orig_width, orig_height, _ = get_video_properties(cap)
        dummy_frame = np.zeros((orig_height, orig_width, 3), dtype=np.uint8)
        resized_dummy = resize_frame_smart(dummy_frame, target_width=TARGET_RESIZE_WIDTH)
        height, width = resized_dummy.shape[:2]
        out, actual_output_path = create_video_writer(output_path, fps, width, height)
        roi_x, roi_y, roi_w, roi_h = 0, int(height * 0.25), width, int(height * 0.75)
        fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=True)
        ct = CentroidTracker(maxDisappeared=30)
        object_anomalies = {}
        anomaly_detected, anomaly_frames, frame_idx = False, [], 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame = resize_frame_smart(frame, target_width=TARGET_RESIZE_WIDTH)
            cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (255, 0, 0), 2)
            if frame_idx % 50 == 0:
                total_frames_estimate = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                if total_frames_estimate > 0:
                    jobs[job_id]["progress"] = min(int((frame_idx / total_frames_estimate) * 100), 99)
            roi_frame = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
            fgmask = fgbg.apply(roi_frame)
            _, fgmask = cv2.threshold(fgmask, 50, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            rects = []
            for contour in contours:
                if cv2.contourArea(contour) > 300:
                    bx, by, bw, bh = cv2.boundingRect(contour)
                    rects.append((roi_x + bx, roi_y + by, roi_x + bx + bw, roi_y + by + bh))
            tracked_objects = ct.update(rects)
            frame_has_anomaly = False
            for (objectID, rect) in tracked_objects.items():
                tx1, ty1, tx2, ty2 = rect
                if frame_idx % (FRAME_SKIP // 2) == 0:
                    crop = frame[ty1:ty2, tx1:tx2]
                    if crop.size > 0:
                        patch = detector.preprocess_patch(crop)
                        score = detector.score_anomalies(detector.extract_features([patch]))[0]
                        object_anomalies[objectID] = (score < ANOMALY_THRESHOLD, score)
                if objectID in object_anomalies:
                    is_anomalous, score = object_anomalies[objectID]
                    if is_anomalous:
                        frame_has_anomaly = True
                        cv2.rectangle(frame, (tx1, ty1), (tx2, ty2), (0, 0, 255), 2)
            if frame_has_anomaly:
                anomaly_detected = True
                anomaly_frames.append(frame_idx)
                # Broadcast alert if WS clients are connected
                alert_data = {"timestamp": time.strftime("%H:%M:%S"), "confidence": 0.85, "zone": "Restricted Zone"}
                for client in connected_clients: asyncio.run_coroutine_threadsafe(client.send_text(json.dumps(alert_data)), asyncio.get_event_loop())
            out.write(frame)
            frame_idx += 1
        return {"status": "success", "anomaly_detected": anomaly_detected, "anomaly_frames": anomaly_frames, "total_frames": frame_idx, "fps": fps, "filename": output_filename}
    except Exception as e: return {"status": "error", "message": str(e)}
    finally:
        if cap: cap.release()
        if out: out.release()

def run_analysis_task(job_id, file_path, output_path, output_filename, temp_dir):
    try:
        result = process_video_sync(job_id, file_path, output_path, output_filename)
        jobs[job_id] = {**result, "progress": 100, "status": "completed"}
    except Exception as e:
        jobs[job_id] = {"status": "error", "message": str(e)}
    finally:
        if os.path.exists(temp_dir): shutil.rmtree(temp_dir)

@app.post("/analyze-video")
async def analyze_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if detector.vit_model is None: raise HTTPException(status_code=503, detail="Models not loaded")
    job_id = f"job_{int(time.time())}"
    jobs[job_id] = {"status": "starting", "progress": 0}
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, file.filename)
    output_filename = f"analyzed_{int(time.time())}_{file.filename}"
    if output_filename.endswith('.mp4'): output_filename = output_filename.replace('.mp4', '.webm')
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    try:
        with open(file_path, "wb") as buffer: shutil.copyfileobj(file.file, buffer)
        background_tasks.add_task(run_analysis_task, job_id, file_path, output_path, output_filename, temp_dir)
        return {"status": "queued", "job_id": job_id}
    except Exception as e:
        if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
        raise HTTPException(status_code=500, detail=f"Failed to queue job: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
