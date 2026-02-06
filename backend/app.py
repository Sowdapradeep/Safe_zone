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
ANOMALY_THRESHOLD = 0.0
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
    # On Windows, CAP_DSHOW is often much faster and more reliable
    for index in [0, 1, 2]:
        try:
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW) if os.name == 'nt' else cv2.VideoCapture(index)
            if cap.isOpened():
                # Test read a frame
                ret, _ = cap.read()
                if ret:
                    print(f"✅ Camera found at index {index}")
                    return cap
                cap.release()
        except Exception as e:
            print(f"⚠️ Error opening camera at index {index}: {e}")
            continue
    return None

async def gen_live_frames():
    cap = get_camera()
    
    if not cap:
        print("❌ Final Camera Error: No camera source found.")
        # Generate a black frame with an error message instead of an empty stream
        black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(black_frame, "CAMERA CONNECT ERROR", (100, 240), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        ret, buffer = cv2.imencode('.jpg', black_frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        return

    # Initialize detection utilities
    fgbg = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=15, detectShadows=True)
    ct = CentroidTracker(maxDisappeared=30)
    object_anomalies = {}
    frame_idx = 0
    last_alert_sent = 0
    
    # ROI setup (centered restricted zone)
    target_width = TARGET_RESIZE_WIDTH
    target_height = 480 # Default estimate
    roi_x, roi_y, roi_w, roi_h = 0, 0, 0, 0

    try:
        while True:
            success, frame = cap.read()
            if not success: 
                print("⚠️ Camera frame read error.")
                break
            
            frame_resized = resize_frame_smart(frame, target_width=target_width)
            h, w = frame_resized.shape[:2]
            
            # Update ROI coordinates if they haven't been set
            if roi_w == 0:
                roi_x, roi_y, roi_w, roi_h = 0, int(h * 0.25), w, int(h * 0.75)

            # Draw Restricted Zone ROI
            cv2.rectangle(frame_resized, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (255, 0, 0), 2)
            cv2.putText(frame_resized, "RESTRICTED ZONE", (roi_x + 10, roi_y + 25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            # Motion Detection (Lowered threshold for extreme sensitivity)
            roi_frame = frame_resized[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
            fgmask = fgbg.apply(roi_frame)
            _, fgmask = cv2.threshold(fgmask, 50, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            rects = []
            for contour in contours:
                if cv2.contourArea(contour) > 50:
                    bx, by, bw, bh = cv2.boundingRect(contour)
                    rects.append((roi_x + bx, roi_y + by, roi_x + bx + bw, roi_y + by + bh))
            
            if rects:
                print(f"[{frame_idx}] Motion detected! {len(rects)} objects found in ROI.")

            tracked_objects = ct.update(rects)
            frame_has_anomaly = False
            top_confidence = 0
            
            # --- AI Logic ---
            
            # 1. Periodic background scan (even if no motion)
            if frame_idx % 45 == 0:
                # Scan a center patch of the ROI
                cx, cy = roi_x + roi_w // 2, roi_y + roi_h // 2
                pw, ph = 224, 224
                x1, y1 = max(0, cx - pw // 2), max(0, cy - ph // 2)
                x2, y2 = min(w, x1 + pw), min(h, y1 + ph)
                
                try:
                    crop = frame_resized[y1:y2, x1:x2]
                    if crop.size > 0:
                        patch = detector.preprocess_patch(crop)
                        features = detector.extract_features([patch])
                        score = detector.score_anomalies(features)[0]
                        is_anomalous = score < ANOMALY_THRESHOLD
                        if is_anomalous:
                            frame_has_anomaly = True
                            top_confidence = max(top_confidence, min(0.99, abs(score) + 0.75))
                            print(f"[{frame_idx}] 🛡️ BACKGROUND SCAN DETECTED ANOMALY: Score {score:.4f}")
                except: pass

            # 2. Motion-triggered detailed scan
            for (objectID, rect) in tracked_objects.items():
                tx1, ty1, tx2, ty2 = rect
                
                if frame_idx % 10 == 0:
                    crop = frame_resized[ty1:ty2, tx1:tx2]
                    if crop.size > 0:
                        try:
                            patch = detector.preprocess_patch(crop)
                            features = detector.extract_features([patch])
                            score = detector.score_anomalies(features)[0]
                            is_anomalous = score < ANOMALY_THRESHOLD
                            object_anomalies[objectID] = (is_anomalous, score)
                            print(f"[{frame_idx}] ⚡ MOTION AI Check - Object {objectID}: Score {score:.4f} {'🚨' if is_anomalous else 'OK'}")
                        except: pass

                if objectID in object_anomalies:
                    is_anomalous, score = object_anomalies[objectID]
                    if is_anomalous:
                        frame_has_anomaly = True
                        confidence = min(0.99, abs(score) + 0.75)
                        top_confidence = max(top_confidence, confidence)
                        cv2.rectangle(frame_resized, (tx1, ty1), (tx2, ty2), (0, 0, 255), 2)

            # Broadcast WebSocket Alerts
            if frame_has_anomaly and (time.time() - last_alert_sent > 3):
                last_alert_sent = time.time()
                alert_data = {
                    "timestamp": time.strftime("%H:%M:%S"),
                    "confidence": top_confidence,
                    "zone": "Restricted Zone"
                }
                print(f"🚨 LIVE ALERT SENT: Confidence {top_confidence:.2f}")
                for client in connected_clients:
                    try:
                        # Use await directly since this is an async function
                        await client.send_text(json.dumps(alert_data))
                    except:
                        pass

            # Burn in HUD
            cv2.putText(frame_resized, f"LIVE: {time.strftime('%H:%M:%S')}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if frame_has_anomaly:
                cv2.putText(frame_resized, "MOTION ANOMALY DETECTED", (10, h - 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            ret, buffer = cv2.imencode('.jpg', frame_resized)
            if not ret: continue
            
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
            frame_idx += 1
            await asyncio.sleep(0.04)
    finally:
        if cap:
            cap.release()
            print("🛑 Live feed camera released.")

@app.get("/api/live-feed")
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
            
            frame_resized = resize_frame_smart(frame, target_width=TARGET_RESIZE_WIDTH)
            
            # Burn in the restricted zone box
            cv2.rectangle(frame_resized, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (255, 0, 0), 2)
            
            if frame_idx % 30 == 0:
                total_frames_estimate = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                if total_frames_estimate > 0:
                    progress = min(int((frame_idx / total_frames_estimate) * 100), 99)
                    jobs[job_id]["progress"] = progress
                    print(f"[{job_id}] Progress: {progress}% (Frame {frame_idx})")
            roi_frame = frame_resized[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
            fgmask = fgbg.apply(roi_frame)
            _, fgmask = cv2.threshold(fgmask, 50, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            rects = []
            for contour in contours:
                # Lower threshold for sensitivity
                if cv2.contourArea(contour) > 100:
                    bx, by, bw, bh = cv2.boundingRect(contour)
                    rects.append((roi_x + bx, roi_y + by, roi_x + bx + bw, roi_y + by + bh))
            tracked_objects = ct.update(rects)
            frame_has_anomaly = False
            for (objectID, rect) in tracked_objects.items():
                tx1, ty1, tx2, ty2 = rect
                if frame_idx % 15 == 0: # Check AI more frequently
                    crop = frame_resized[ty1:ty2, tx1:tx2]
                    if crop.size > 0:
                        try:
                            patch = detector.preprocess_patch(crop)
                            features = detector.extract_features([patch])
                            score = detector.score_anomalies(features)[0]
                            is_anomalous = score < ANOMALY_THRESHOLD
                            object_anomalies[objectID] = (is_anomalous, score)
                            if is_anomalous:
                                print(f"[{job_id}] 🚨 AI detected anomaly at frame {frame_idx}! Score: {score:.4f}")
                        except Exception as e:
                            print(f"[{job_id}] AI Error on frame {frame_idx}: {e}")

                if objectID in object_anomalies:
                    is_anomalous, score = object_anomalies[objectID]
                    if is_anomalous:
                        frame_has_anomaly = True
                        cv2.rectangle(frame_resized, (tx1, ty1), (tx2, ty2), (0, 0, 255), 2)
                        cv2.putText(frame_resized, f"ANOMALY: {score:.2f}", (tx1, ty1-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
            if frame_has_anomaly:
                anomaly_detected = True
                if frame_idx not in anomaly_frames:
                    anomaly_frames.append(frame_idx)
                
                # Broadcast alert if WS clients are connected
                alert_data = {"timestamp": time.strftime("%H:%M:%S"), "confidence": 0.85, "zone": "Restricted Zone"}
                for client in connected_clients: 
                    asyncio.run_coroutine_threadsafe(client.send_text(json.dumps(alert_data)), asyncio.get_event_loop())
            
            out.write(frame_resized)
            frame_idx += 1
        return {"status": "success", "anomaly_detected": anomaly_detected, "anomaly_frames": anomaly_frames, "total_frames": frame_idx, "fps": fps, "filename": output_filename}
    except Exception as e: return {"status": "error", "message": str(e)}
    finally:
        if cap: cap.release()
        if out: out.release()

def run_analysis_task(job_id, file_path, output_path, output_filename, temp_dir):
    try:
        result = process_video_sync(job_id, file_path, output_path, output_filename)
        # Only mark as completed if the processing was actually successful
        final_status = "completed" if result.get("status") == "success" else "error"
        jobs[job_id] = {**result, "progress": 100, "status": final_status}
    except Exception as e:
        jobs[job_id] = {"status": "error", "message": str(e)}
    finally:
        if os.path.exists(temp_dir): shutil.rmtree(temp_dir)

# In-memory incident store
incidents = []

@app.get("/api/incidents")
async def get_incidents():
    return incidents

@app.post("/api/incidents")
async def create_incident(incident: dict):
    # Assign a simple ID if one isn't provided (though frontend seems to handle IDs)
    if "id" not in incident:
        incident["id"] = f"inc_{int(time.time())}_{len(incidents)}"
    incidents.insert(0, incident) # Add to beginning
    return incident

@app.post("/api/analyze-video")
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
