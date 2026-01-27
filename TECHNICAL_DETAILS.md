# SafeZone Technical Deep-Dive

This document provides a detailed breakdown of the artificial intelligence models, architectural decisions, and technology stack powering the SafeZone Surveillance system.

## 🧠 1. AI Models Used

The system utilizes a **three-stage AI pipeline** to ensure accuracy and computational efficiency:

1.  **Object Detection: Faster R-CNN (ResNet-50)**
    *   **Role**: Acts as the "primary detector" to identify and locate high-value objects (people, vehicles) within the video frame.
    *   **Implementation**: Uses a ResNet-50 backbone for robust feature extraction in varied lighting conditions.

2.  **Feature Extraction: Vision Transformer (ViT-B/16)**
    *   **Role**: The "behavioral analyzer." It processes crops of detected objects to generate high-dimensional embeddings (vectors) that represent the object's visual and behavioral signature.
    *   **Benefit**: Unlike standard CNNs, the ViT architecture captures global dependencies, which is crucial for understanding movement patterns.

3.  **Anomaly Detection: Isolation Forest**
    *   **Role**: The "unsupervised decision-maker." It analyzes the embeddings from the ViT and assigns an anomaly score based on how much the current behavior deviates from the learned "normal" baseline.
    *   **Method**: Unsupervised learning via outlier detection.

4.  **Optimization: MOG2 (Background Subtraction)**
    *   **Role**: A lightweight motion detection algorithm that acts as a "trigger."
    *   **Efficiency**: Saves ~70% of CPU/GPU resources by only activating the heavy-lifting AI pipeline when movement is detected in a Restricted Zone.

## 🔄 2. Why We Chose This Architecture

### Unsupervised vs. Supervised Learning
*   **The Problem with Supervised**: Traditional models trained on "labeled crimes" are limited. They can only see what they've seen before.
*   **The SafeZone Approach**: By using **Isolation Forest**, we focus on learning what is "normal" for a specific camera view. Anything outside that norm—regardless of whether it's a known threat—is flagged. This allows the system to detect "zero-day" security breaches.

### Computational Efficiency
*   **The Challenge**: Running a Vision Transformer on 30FPS live video is extremely hardware-intensive.
*   **The Fix**: We implemented **frame-skipping** (processing every 15-25 frames) and **ROI (Region of Interest) Masking**. This allows the stack to run smoothly on cloud platforms like Render without requiring expensive GPU clusters.

## 💻 3. The Tech Stack

The project is built using a modern, decoupled architecture:

*   **Frontend**: `React` (Vite) + `Tailwind CSS`. We used `Shadcn UI` for a premium, high-contrast "Control Room" aesthetic.
*   **AI Engine**: `Python` + `FastAPI`. We leveraged `PyTorch` and `OpenCV` for the core machine learning and vision tasks.
*   **Application Server**: `Node.js` (Express). Manages the business logic, incident persistence, and system health status.
*   **Database**: `MongoDB`. A schema-less approach allows us to store varied incident metadata (timestamps, confidence levels, zone IDs).
*   **Real-time Communication**: `WebSockets (WS)`. Essential for pushing "Anomaly Detected" alerts to the operator with zero page refresh.
*   **Deployment**: 
    *   **Frontend**: `Vercel` for global edge delivery.
    *   **Backend**: `Render` for high-performance Python environment hosting.
