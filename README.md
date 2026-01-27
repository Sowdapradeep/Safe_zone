# CCTV Maintenance & Monitoring System with Real-Time Anomaly Detection (MVP)

## 1. Project Overview
The **CCTV Maintenance & Monitoring System** is a professional-grade control-room simulation designed to enhance security oversight through AI-driven automation. Unlike standard video viewing applications, this system integrates **asset health tracking**, **maintenance scheduling awareness**, and **real-time anomaly detection**. 

This MVP (Minimum Viable Product) simulates a high-security environment (e.g., restricted facility, secure zone) where the system identifies unauthorized movement or unusual behavior within a designated Restricted Zone (ROI), allowing security operators to focus on verified threats rather than monitoring dozens of screens simultaneously.

## 2. Problem Statement
Continuous human monitoring of CCTV feeds is subject to fatigue, leading to missed incidents in critical environments like jails, army camps, or industrial secure zones. 
- **Fatigue & Overload**: Operators monitoring multiple feeds often miss subtle or brief anomalies.
- **Latency in Response**: Without automated alerts, identifying the exact moment of a security breach relies on manual discovery.
- **Maintenance Gaps**: Security is often compromised by offline cameras or overdue maintenance that goes unnoticed until an incident occurs.

## 3. Solution Overview
The system provides an intelligent layer over the raw video feed:
- **Continuous Analysis**: AI models scan every frame for patterns that deviate from the "normal" baseline.
- **Unsupervised Learning**: Uses an unsupervised approach to identify anomalies without requiring a pre-labeled dataset of "crimes" or "breaches."
- **Restricted Zone Monitoring**: Dynamic Region of Interest (ROI) logic focuses the AI's attention on critical entry/exit points.
- **Incident Escalation**: Detected anomalies are automatically logged as incidents, requiring operator acknowledgement and classification.

## 4. Key Features
### Monitoring & Control Room UI
- Dashboard-style interface with real-time status indicators.
- Live system health telemetry (Camera status, Threat levels, AI Confidence).

### AI Anomaly Detection
- **Visual Transformer (ViT)** based feature extraction.
- **Isolation Forest** unsupervised anomaly scoring.
- Consistent object tracking across frames.

### Restricted Zone Monitoring
- Visual ROI (Region of Interest) overlays.
- Motion-guided detection to reduce false positives in non-critical areas.

### Incident & Alert Management
- Automated incident logging with confidence scores and timestamps.
- Real-time alerts via WebSockets.
- Status tracking (Open, In Progress, Resolved, False Alarm).

### Maintenance Awareness
- Simulated asset management (Uptime tracking, Next maintenance scheduling).
- Online/Offline status reporting for the camera network.

### Dual Input Modes
- Supports both recorded CCTV footage analysis and live webcam-based real-time demos.

## 5. Input Modes Explanation
To ensure versatility in evaluation, the system supports two distinct input modes:

### Recorded Video (Playback Analysis)
- User uploads standard CCTV footage (MP4/WebM).
- The system processes the video frame-by-frame, simulating a live feed monitoring scenario.
- AI logic draws tracking IDs and anomaly boxes directly onto the analyzed playback.

### Local Camera (Real-Time Demo)
- The system uses the local computer's webcam to demonstrate real-time analysis capabilities.
- This mode is clearly labeled as a **Demo Mode** intended to showcase the latency and responsiveness of the AI pipeline.

> [!IMPORTANT]
> **Disclaimer**: This MVP does not directly integrate with physical CCTV hardware protocols (RTSP/ONVIF). It is a proof-of-concept simulation using file uploads and local camera sources.

## 6. Phase-by-Phase Project Development
### Phase 1: AI Model Development (Unsupervised Anomaly Detection)
- **Goal**: Create a model capable of identifying "unusual" behavior without supervised labels.
- **Implementation**: Leveraged a pre-trained ViT for feature extraction and an Isolation Forest for outlier detection.
- **Outcome**: A robust backend capable of assigning an "anomaly score" to any given image patch.
- *See Screenshot – Phase 1*

### Phase 2: Restricted Zone (ROI) Logic & Visual Marking
- **Goal**: Narrow the AI's focus to specific areas.
- **Implementation**: Developed ROI masks and integrated motion detection (MOG2) to trigger analysis only when movement is detected in the zone.
- **Outcome**: Massive reduction in CPU overhead and false positives from background movement.
- *See Screenshot – Phase 2*

### Phase 3: Backend API for AI Inference
- **Goal**: Build a scalable bridge between the AI logic and the UI.
- **Implementation**: Created a FastAPI (Python) service for AI tasks and a Node.js/Express service for system management and incident persistence.
- **Outcome**: A non-blocking, multi-service architecture that handles long-running video analysis.
- *See Screenshot – Phase 3*

### Phase 4: CCTV Monitoring Dashboard (MVP UI)
- **Goal**: Design a professional, control-room style user interface.
- **Implementation**: Built a React-based SPA with high-contrast, premium aesthetics (glassmorphism/dark mode).
- **Outcome**: A functional dashboard that displays feed, telemetry, and incident logs simultaneously.
- *See Screenshot – Phase 4*

### Phase 5: Real-Time Playback & Demo Integration
- **Goal**: Combine the UI with the backend for end-to-end monitoring.
- **Implementation**: Integrated WebSockets for real-time alerts and a proxy-based video delivery system for processed feeds.
- **Outcome**: A complete lifecycle from video upload to anomaly detection and incident resolution.
- *See Screenshot – Phase 5*

## 7. System Architecture
1. **Input Layer**: Video File Upload or Webcam Stream.
2. **AI Inference Engine (Python)**: Handles frame resizing, ROI masking, feature extraction, and anomaly scoring.
3. **Application Server (Node.js)**: Manages state, persists incidents to MongoDB, and proxies analyzed feeds.
4. **Monitoring Frontend (React)**: Displays the active feed, real-time telemetry, and the Incident Management console.

## 8. Tech Stack
- **Frontend**: React (Vite), Tailwind CSS, Lucide Icons, Shadcn UI (components).
- **Backend (API & Logic)**: Node.js (Express), Python (FastAPI).
- **AI / ML**: PyTorch, TorchVision (ViT), Scikit-learn (Isolation Forest), OpenCV.
- **Storage**: MongoDB.
- **Real-time**: WebSockets (WS).

## 9. MVP Scope & Limitations
- **Included**: One-camera analysis, incident logging, ROI monitoring, asset health visualization, recorded video/webcam input.
- **Excluded**: Multi-camera grid view (NVR style), complex user authentication, direct RTSP/ONVIF hardware integration, and historical analytics reporting.
- **Rationale**: The core focus was to demonstrate a functional AI security pipeline and a professional UI within the constraints of the hiring challenge.

## 10. Future Enhancements
- **Hardware Integration**: Implementing RTSP/ONVIF support for real-time physical camera connectivity.
- **Multi-Camera Grid**: Expanding the UI to support 4, 9, or 16-camera simultaneous monitoring.
- **Mobile Technician App**: A companion app for on-site maintenance technicians to receive and resolve tickets.
- **Voice Alerts**: Integration of automated vocal warnings for control room staff during high-threat events.

## 11. How to Run the Project
1. **Clone the Repository**: `git clone [repository-url]`
2. **Setup Backend (AI)**: Navigate to `/backend`, install `requirements.txt`, and run `python app.py`.
3. **Setup Server (Node)**: Navigate to `/server`, install dependencies, and run `npm start`.
4. **Setup Frontend**: In the root directory, install npm packages and run `npm run dev`.
5. **Access**: Open [http://localhost:5173](http://localhost:5173) in your browser.

## 12. Conclusion
This project demonstrates the potential of combining modern web aesthetics with powerful AI-driven security logic. It bridges the gap between simple video playback and an intelligent monitoring system, providing a strong foundation for future CCTV control-room innovations.
