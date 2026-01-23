import torch
import torchvision
from torchvision import transforms
import cv2
import joblib
import numpy as np
import time
import sys

# --- Configuration ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
VIT_MODEL_PATH = "vit_feature_extractor.pth"
ANOMALY_MODEL_PATH = "isolation_forest_model.joblib"

# ROI will cover the lower half of the frame (like seats area in the image)
ROI_HEIGHT_RATIO = 0.5  # Use bottom 50% of frame height
ANOMALY_THRESHOLD = -0.05       # Lower decision score = more anomalous
PATCH_SIZE = 224                # Vision Transformer standard input
STRIDE = 112                    # 50% overlap for detailed scanning

# --- Model Loading ---

def load_feature_extractor(path):
    print(f"Using device: {DEVICE}")
    print("Loading VLM-based Feature Extractor (ViT)...")
    model = torchvision.models.vit_b_16(pretrained=False)
    state_dict = torch.load(path, map_location=DEVICE)
    # strict=False is used because we might be using a partial state_dict or a different head
    model.load_state_dict(state_dict, strict=False)
    model.heads = torch.nn.Identity() # Remove classification head to get embeddings
    model.to(DEVICE)
    model.eval()
    return model

def load_anomaly_detector(path):
    print("Loading Unsupervised Anomaly Detector (Isolation Forest)...")
    return joblib.load(path)

# --- Preprocessing ---

vit_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def main():
    # 1. Handle Input (Video File or Webcam)
    video_source = 0
    if len(sys.argv) > 1:
        video_source = sys.argv[1]
        print(f"Using video file: {video_source}")
    else:
        print("Using webcam (Default). Pass a file path as an argument to use a video file.")

    # 2. Load Models
    try:
        feature_extractor = load_feature_extractor(VIT_MODEL_PATH)
        anomaly_model = load_anomaly_detector(ANOMALY_MODEL_PATH)
    except Exception as e:
        print(f"Critical Error loading models: {e}")
        return

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Error: Could not open video source {video_source}")
        return

    print("\n--- ROI SURVEILLANCE RUNNING ---")
    print(f"ROI: Full width, lower {int(ROI_HEIGHT_RATIO*100)}% of frame")
    # 3. Add global threshold access
    global ANOMALY_THRESHOLD
    
    # 3. Motion Detection Setup (MOG2)
    fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=True)
    
    # Get first frame to calculate ROI
    ret, first_frame = cap.read()
    if not ret:
        print("Error: Could not read first frame")
        return
    
    frame_height, frame_width = first_frame.shape[:2]
    
    # Calculate ROI: Full width, lower half of frame
    roi_x = 0
    roi_y = int(frame_height * (1 - ROI_HEIGHT_RATIO))
    roi_w = frame_width
    roi_h = int(frame_height * ROI_HEIGHT_RATIO)
    
    # Reset video capture to beginning
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    cnt = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cnt += 1
        # 4. Use calculated ROI position
        x, y, w, h = roi_x, roi_y, roi_w, roi_h
        
        roi_frame = frame[y:y+h, x:x+w]
        
        # Draw ROI border
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, "RESTRICTED ZONE (Motion-Guided)", (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # 5. Find Motion Candidates in ROI
        fgmask = fgbg.apply(roi_frame)
        # Clean up mask
        _, fgmask = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        candidates = []
        candidate_boxes = []
        
        for contour in contours:
            if cv2.contourArea(contour) > 1000: # Min area for an 'object'
                cx, cy, cw, ch = cv2.boundingRect(contour)
                # Expand box slightly for ViT context
                pad = 10
                cx1, cy1 = max(0, cx - pad), max(0, cy - pad)
                cx2, cy2 = min(w, cx + cw + pad), min(h, cy + ch + pad)
                
                crop = roi_frame[cy1:cy2, cx1:cx2]
                if crop.size > 0:
                    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    candidates.append(vit_transform(crop_rgb))
                    # Store global coordinates for drawing
                    candidate_boxes.append((x + cx1, y + cy1, cx2 - cx1, cy2 - cy1))

        anomaly_detected = False
        if candidates:
            batch_tensors = torch.stack(candidates).to(DEVICE)
            with torch.no_grad():
                features = feature_extractor(batch_tensors).cpu().numpy()
            
            scores = anomaly_model.decision_function(features)
            
            min_score = np.min(scores)
            if cnt % 15 == 0:
                print(f"Motion in Zone! Detected {len(candidates)} objects. Min Anomaly Score: {min_score:.4f}")

            for score, (bx, by, bw, bh) in zip(scores, candidate_boxes):
                # Using a slightly more sensitive threshold for motion crops
                if score < ANOMALY_THRESHOLD or score < -0.02: 
                    anomaly_detected = True
                    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (0, 0, 255), 2)
                    cv2.putText(frame, f"ANOMALY: {score:.3f}", (bx, by - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                else:
                    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (0, 255, 0), 1)
                    cv2.putText(frame, f"Safe: {score:.3f}", (bx, by - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # 6. Threshold and Alert
        if anomaly_detected:
            cv2.putText(frame, "!!! Restricted Zone Anomaly Detected !!!", (20, 40),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255), 2)
            if int(time.time() * 5) % 2 == 0:
                cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 10)

        # 7. Show Output
        cv2.putText(frame, f"Threshold: {ANOMALY_THRESHOLD:.2f} | Controls: +/-", 
                    (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow("Restricted Area - VLM Anomaly Detection", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('+') or key == ord('='):
            ANOMALY_THRESHOLD += 0.01
        elif key == ord('-') or key == ord('_'):
            ANOMALY_THRESHOLD -= 0.01

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
