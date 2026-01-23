import torch
import torchvision
from torchvision import transforms
import cv2
import joblib
import numpy as np
import time

# --- Configuration ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

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

# --- Model Loading ---

def load_object_detector(path):
    print("Loading Object Detector...")
    # Standard Faster R-CNN with ResNet-50 backbone
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False)
    state_dict = torch.load(path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model

def load_feature_extractor(path):
    print("Loading ViT Feature Extractor...")
    # ViT-B/16
    model = torchvision.models.vit_b_16(pretrained=False)
    
    # We need to handle the missing head or modify the model to output features
    state_dict = torch.load(path, map_location=DEVICE)
    
    # If the state dict is missing the head weights (which we suspect), loose loading might work
    # BUT we also want to remove the head so we get the 768-dim vector, not class logits.
    # Standard ViT output is (batch, num_classes). We want representations.
    
    # Load weights with strict=False to ignore missing head keys if any
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"ViT Load Stats - Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    
    # Replace the classification head with Identity to get embeddings
    model.heads = torch.nn.Identity()
    
    model.to(DEVICE)
    model.eval()
    return model

def load_anomaly_detector(path):
    print("Loading Isolation Forest...")
    model = joblib.load(path)
    return model

# --- Preprocessing ---

# Transforms for Video Frames (for Object Detector)
# FasterRCNN expects 0-1 tensors. 
def frame_to_tensor(frame):
    # Convert BGR to RGB
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Convert to Tensor and normalize to [0, 1]
    return torchvision.transforms.functional.to_tensor(img).to(DEVICE)

# Transforms for ViT Crops
vit_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# --- Main Pipeline ---

# --- Global Zone State ---
ZONE_POINTS = []
def mouse_callback(event, x, y, flags, param):
    global ZONE_POINTS
    if event == cv2.EVENT_LBUTTONDOWN:
        ZONE_POINTS.append((x, y))
        print(f"Added point: ({x}, {y})")

def main():
    global ZONE_POINTS
    # 1. Load Models
    try:
        detector = load_object_detector("object_detection_model (1).pth")
        feature_extractor = load_feature_extractor("vit_feature_extractor.pth")
        anomaly_model = load_anomaly_detector("isolation_forest_model.joblib")
    except Exception as e:
        print(f"Failed to load models: {e}")
        return

    # 2. Open Video Source
    import sys
    import os
    
    video_source = 0 # Default to webcam
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        if os.path.exists(video_path):
            video_source = video_path
            print(f"Using video file: {video_source}")
        else:
            print(f"Warning: File {video_path} not found. Using webcam.")
            
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Error: Could not open video source {video_source}.")
        return
    else:
        # Debug info
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"Video opened successfully. Total frames: {length}, FPS: {fps}")
        if length == 0:
            print("Warning: Video has 0 frames reported. It might be corrupt or codec is unsupported.")

    # --- Setup Mode: Draw Zone ---
    cv2.namedWindow("Setup: Draw Zone")
    cv2.setMouseCallback("Setup: Draw Zone", mouse_callback)
    
    print("\n--- SETUP MODE ---")
    print("1. Click at least 3 points on the screen to define the 'Surveillance Zone'.")
    print("2. Press 'd' when Done.")
    print("3. Press 'r' to Reset points.")
    print("4. Press 'q' to Quit setup.")

    while True:
        ret, frame = cap.read()
        if not ret:
            # Loop video during setup so user can draw
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        
        # Draw points and lines
        for pt in ZONE_POINTS:
            cv2.circle(frame, pt, 5, (255, 0, 0), -1)
        if len(ZONE_POINTS) >= 2:
            cv2.polylines(frame, [np.array(ZONE_POINTS)], False, (255, 0, 0), 2)
        
        cv2.putText(frame, "SETUP: Click to draw zone. 'd' when done.", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Setup: Draw Zone", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('d'):
            if len(ZONE_POINTS) < 3:
                print("Please select at least 3 points.")
            else:
                break
        elif key == ord('r'):
            ZONE_POINTS = []
        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return

    cv2.destroyWindow("Setup: Draw Zone")
    zone_array = np.array(ZONE_POINTS, dtype=np.int32)

    # Optimization Config
    SKIP_FRAMES = 5  # Run detection every N frames
    RESIZE_WIDTH = 640 # Downscale frame for speed
    
    frame_count = 0
    last_detections = [] # Store results for skipped frames

    print(f"Starting Surveillance System (Optimized: Skip {SKIP_FRAMES} frames)... Press 'q' to quit.")

    # Background subtractor for motion fallback
    fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
    
    # To calculate real FPS
    loop_start_time = time.time()
    
    # Reset video for main processing
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            # Loop video indefinitely so user can take screenshots/monitor
            print("End of video. Looping...")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        start_time = time.time()
        
        # Resize frame for processing speed (keep aspect ratio)
        height, width = frame.shape[:2]
        if width > RESIZE_WIDTH:
            scale = RESIZE_WIDTH / width
            new_height = int(height * scale)
            frame_resized = cv2.resize(frame, (RESIZE_WIDTH, new_height))
        else:
            frame_resized = frame
            scale = 1.0

        max_score_this_frame = 0.0
        
        # Only process every Nth frame
        if frame_count % SKIP_FRAMES == 0:
            last_detections = [] 
            
            input_tensor = frame_to_tensor(frame_resized)
            with torch.no_grad():
                detections = detector([input_tensor])[0]

            if len(detections['scores']) > 0:
                max_score_this_frame = torch.max(detections['scores']).item()

            for i in range(len(detections['boxes'])):
                box = detections['boxes'][i].cpu().numpy().astype(int)
                score = detections['scores'][i].item()
                label_id = detections['labels'][i].item()
                
                # Removed score > 0.4 and label_id == 1 to detect ALL objects 
                if score > 0.2: # Lowered threshold further
                    x1, y1, x2, y2 = box
                    h, w, _ = frame_resized.shape
                    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
                    
                    if x2 - x1 < 10 or y2 - y1 < 10: continue

                    # Check Zone Intrusion (using foot position)
                    orig_x_center = int((x1 + x2) / 2 / scale)
                    orig_y_bottom = int(y2 / scale)
                    is_in_zone = cv2.pointPolygonTest(zone_array, (orig_x_center, orig_y_bottom), False) >= 0

                    person_crop = frame_resized[y1:y2, x1:x2]
                    person_crop_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
                    
                    try:
                        vit_input = vit_transform(person_crop_rgb).unsqueeze(0).to(DEVICE)
                        with torch.no_grad():
                            features = feature_extractor(vit_input)
                        
                        features_np = features.cpu().numpy()
                        prediction = anomaly_model.predict(features_np)[0]
                        anomaly_score = anomaly_model.decision_function(features_np)[0]

                        # Get human readable category
                        category = COCO_INSTANCE_CATEGORY_NAMES[label_id] if label_id < len(COCO_INSTANCE_CATEGORY_NAMES) else "Item"
                        
                        # Store result with foot point and category
                        orig_box = (int(x1/scale), int(y1/scale), int(x2/scale), int(y2/scale))
                        foot_point = (orig_x_center, orig_y_bottom)
                        last_detections.append((orig_box, prediction, anomaly_score, is_in_zone, foot_point, category))
                        
                    except Exception as e:
                        print(f"Error processing crop: {e}")

        # --- Motion Fallback ---
        # Mask for the zone
        mask = np.zeros(frame_resized.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [ (np.array(ZONE_POINTS) * scale).astype(np.int32) ], 255)
        
        # Apply motion detection
        fgmask = fgbg.apply(frame_resized)
        # Bitwise AND to only look at motion inside the zone
        motion_in_zone = cv2.bitwise_and(fgmask, fgmask, mask=mask)
        motion_score = np.sum(motion_in_zone > 0)
        
        # If significant motion in zone, trigger alert even if AI fails
        motion_alert = motion_score > 500 # Threshold for pixels moving

        # Draw cached detections and the zone
        cv2.polylines(frame, [zone_array], True, (255, 0, 0), 3) # Thicker blue line
        
        any_intrusion = motion_alert
        for (box, prediction, anomaly_score, is_in_zone, foot_pt, category) in last_detections:
            x1, y1, x2, y2 = box
            
            # Draw the 'test point' used for zone checking (foot point)
            cv2.circle(frame, foot_pt, 4, (0, 255, 255), -1)

            # Color logic: Red for Anomaly, Orange for Zone Intrusion, Green for Safe
            if prediction == -1:
                color = (0, 0, 255) # Red
                text = f"ANOMALY: {category}"
            elif is_in_zone:
                color = (0, 165, 255) # Orange
                text = f"INTRUSION: {category}"
                any_intrusion = True
            else:
                color = (0, 255, 0) # Green
                text = f"{category} ({anomaly_score:.2f})"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if any_intrusion:
             msg = "WARNING: Motion in Zone!" if motion_alert and not any_intrusion else "WARNING: Object in Zone!"
             cv2.putText(frame, msg, (frame.shape[1]//2 - 200, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 3)

        # Compute display FPS (average)
        fps = 1.0 / (time.time() - start_time + 0.0001)
        fps = min(fps, 60.0)
        
        status_text = "Processing" if frame_count % SKIP_FRAMES == 0 else "Skipping"
        cv2.putText(frame, f"FPS: {fps:.1f} ({status_text})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, f"Max Confidence: {max_score_this_frame:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Safe Zone Surveillance", frame)
        
        frame_count += 1
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"screenshot_{int(time.time())}.png"
            cv2.imwrite(filename, frame)
            print(f"Screenshot saved to {filename}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
