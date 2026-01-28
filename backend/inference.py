import torch
import torchvision
from torchvision import transforms
import joblib
import numpy as np
import os
import cv2
import gc

# Configuration defaults
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
VIT_MODEL_FILENAME = "vit_feature_extractor.pth"
ANOMALY_MODEL_FILENAME = "isolation_forest_model.joblib"

class AnomalyDetector:
    def __init__(self, model_dir="model"):
        self.device = DEVICE
        self.model_dir = model_dir
        self.vit_model = None
        self.anomaly_model = None
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.is_half = False

    def load_models(self):
        # Extreme RAM Optimization for 512MB limits
        torch.set_num_threads(1) 
        
        print(f"Loading models onto {self.device} (mmap mode)...")
        
        vit_path = os.path.join(self.model_dir, VIT_MODEL_FILENAME)
        anomaly_path = os.path.join(self.model_dir, ANOMALY_MODEL_FILENAME)

        if not os.path.exists(vit_path):
            raise FileNotFoundError(f"ViT model not found at {vit_path}")
        if not os.path.exists(anomaly_path):
            raise FileNotFoundError(f"Anomaly model not found at {anomaly_path}")

        # 1. Feature Extractor - MobileNetV3 Small (RAM Optimized)
        # We use pretrained=True here but torchvision will cache it.
        # This model is ~15MB vs ViT which is ~350MB.
        self.vit_model = torchvision.models.mobilenet_v3_small(weights='DEFAULT')
        self.vit_model.classifier = torch.nn.Identity() # Remove the classifier head
        self.vit_model.to(self.device)
        self.vit_model.eval()
        
        # Immediate cleanup
        gc.collect()

        # 2. Anomaly Detector
        self.anomaly_model = joblib.load(anomaly_path)
        gc.collect()
        
        print("Models loaded successfully (MobileNet Optimized).")

    def extract_features(self, patches):
        """
        Extract features from a list of image patches (tensors).
        """
        if not patches:
            return None
        
        batch_tensors = torch.stack(patches).to(self.device)
        if self.is_half:
            batch_tensors = batch_tensors.half()
            
        with torch.no_grad():
            features = self.vit_model(batch_tensors).cpu().float().numpy()
        return features

    def score_anomalies(self, features):
        """
        Returns anomaly scores for the given features.
        """
        if self.anomaly_model is None:
            raise RuntimeError("Anomaly model not loaded")
        return self.anomaly_model.decision_function(features)

    def preprocess_patch(self, patch_bgr):
        """
        Convert BGR patch to tensor ready for ViT.
        """
        patch_rgb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB)
        return self.transform(patch_rgb)

