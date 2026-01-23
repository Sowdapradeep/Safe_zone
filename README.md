# Safe Zone Surveillance System 🛡️

An AI-powered video surveillance system designed to detect intrusions and anomalies within a user-defined safety zone using Computer Vision and Machine Learning.

## 🚀 Overview

This system combines three powerful models to provide robust surveillance:
1.  **Object Detection (Faster R-CNN):** Identifies people and objects in the video stream.
2.  **Feature Extraction (ViT):** Uses a Vision Transformer to extract deep visual features from detected objects.
3.  **Anomaly Detection (Isolation Forest):** Analyzes extracted features to identify unusual behavior or unauthorized items.

## ✨ Key Features

*   **Customizable Safety Zone:** Interactively define your surveillance area by clicking points on the screen.
*   **Intrusion Alerts:** Real-time visual warnings when objects enter the restricted zone.
*   **Anomaly Detection:** Flags objects that the system hasn't seen before or that behave unusually.
*   **Motion Fallback:** Includes a background subtraction-based motion detector to trigger alerts even if object detection fails.
*   **Performance Optimized:** Implements frame skipping and resizing to maintain high processing speeds on various hardware.

## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Sowdapradeep/Safe_zone.git
    cd Safe_zone
    ```

2.  **Install dependencies:**
    ```bash
    pip install torch torchvision opencv-python joblib numpy
    ```
    *Note: A GPU with CUDA support is recommended for optimal performance.*

3.  **Download Models:**
    Ensure the following model files are in the project root:
    *   `object_detection_model (1).pth`
    *   `vit_feature_extractor.pth`
    *   `isolation_forest_model.joblib`

## 🚦 How to Use

1.  Run the main script:
    ```bash
    python main.py
    ```

2.  **Setup Mode:**
    *   Click at least 3 points on the video feed to draw your **Surveillance Zone**.
    *   Press `d` when done.
    *   Press `r` to reset the points.
    *   Press `q` to quit.

3.  **Surveillance Mode:**
    *   The system will start monitoring the defined zone.
    *   **Green boxes:** Safe/Normal objects.
    *   **Orange boxes:** Intrusion detected in the zone.
    *   **Red boxes:** Anomaly detected.
    *   Press `q` to stop monitoring.

## ⚙️ Technical Details

*   **Backend:** Python
*   **Computer Vision:** OpenCV, PyTorch (Torchvision)
*   **Deep Learning:** Faster R-CNN (ResNet-50), Vision Transformer (ViT-B/16)
*   **Machine Learning:** Scikit-learn (Isolation Forest via Joblib)

## 📄 License

This project is specialized for security and surveillance applications. 
