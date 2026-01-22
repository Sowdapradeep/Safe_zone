import torch
import joblib
import os
import sys

def log(message):
    with open("model_info.log", "a") as f:
        f.write(message + "\n")
    print(message)

def inspect_pth(path, name):
    log(f"--- Inspecting {name} ---")
    try:
        content = torch.load(path, map_location=torch.device('cpu'))
        
        if isinstance(content, dict):
            # Check Faster R-CNN Head
            if 'roi_heads.box_predictor.cls_score.weight' in content:
                shape = content['roi_heads.box_predictor.cls_score.weight'].shape
                log(f"Faster R-CNN cls_score shape: {shape} (Num classes = {shape[0]})")
            
            # Check ViT Head/Patch Size
            if 'conv_proj.weight' in content:
                shape = content['conv_proj.weight'].shape
                log(f"ViT conv_proj shape: {shape} (Patch size = {shape[2]}x{shape[3]})")
            if 'heads.head.weight' in content:
                 shape = content['heads.head.weight'].shape
                 log(f"ViT head shape: {shape} (Num classes = {shape[0]})")
                    
        elif isinstance(content, torch.nn.Module):
            log("File contains a full pickled Model Object.")
            
    except Exception as e:
        log(f"Error loading {name}: {e}")
    log("\n")

def inspect_joblib(path, name):
    log(f"--- Inspecting {name} ---")
    try:
        model = joblib.load(path)
        log(f"Type: {type(model)}")
    except Exception as e:
        log(f"Error loading {name}: {e}")
    log("\n")

if __name__ == "__main__":
    with open("model_info.log", "w") as f:
        f.write("Starting shape inspection...\n")

    base_path = os.getcwd()
    
    od_path = os.path.join(base_path, "object_detection_model (1).pth")
    if os.path.exists(od_path):
        inspect_pth(od_path, "Object Detection Model")

    vit_path = os.path.join(base_path, "vit_feature_extractor.pth")
    if os.path.exists(vit_path):
        inspect_pth(vit_path, "ViT Feature Extractor")
