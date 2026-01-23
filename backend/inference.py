import torch
import torchvision
from torchvision import transforms
import joblib
import numpy as np
import os

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

    def load_models(self):
        print(f"Loading models onto {self.device}...")
        
        vit_path = os.path.join(self.model_dir, VIT_MODEL_FILENAME)
        anomaly_path = os.path.join(self.model_dir, ANOMALY_MODEL_FILENAME)

        if not os.path.exists(vit_path):
            raise FileNotFoundError(f"ViT model not found at {vit_path}")
        if not os.path.exists(anomaly_path):
            raise FileNotFoundError(f"Anomaly model not found at {anomaly_path}")

        # 1. Feature Extractor
        self.vit_model = torchvision.models.vit_b_16(pretrained=False)
        state_dict = torch.load(vit_path, map_location=self.device)
        self.vit_model.load_state_dict(state_dict, strict=False)
        self.vit_model.heads = torch.nn.Identity()
        self.vit_model.to(self.device)
        self.vit_model.eval()

        # 2. Anomaly Detector
        self.anomaly_model = joblib.load(anomaly_path)
        print("Models loaded successfully.")

    def extract_features(self, patches):
        """
        Extract features from a list of image patches (tensors).
        """
        if not patches:
            return None
        
        batch_tensors = torch.stack(patches).to(self.device)
        with torch.no_grad():
            features = self.vit_model(batch_tensors).cpu().numpy()
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
    
    import cv2 # Local import for checking types if needed, or just rely on passed objects
