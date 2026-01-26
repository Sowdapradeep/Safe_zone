# SafeZone Surveillance - AI-Powered CCTV Monitoring System

An intelligent surveillance system with ML-based anomaly detection for restricted zones. Features real-time video analysis, incident tracking, and a modern monitoring dashboard.

## Architecture

- **Frontend**: React + TypeScript + Vite (Deployed on Vercel)
- **Node.js Backend**: Express.js server handling API requests and WebSocket connections
- **Python AI Service**: FastAPI service with PyTorch-based anomaly detection
- **Database**: MongoDB for incident management

## Local Development Setup

### Prerequisites
- Node.js 18+ and npm
- Python 3.11+
- MongoDB (local or Atlas)

### 1. Clone and Install

```bash
git clone https://github.com/Sowdapradeep/Safe_zone.git
cd "safe zone survillence"

# Install frontend dependencies
npm install

# Install backend dependencies
cd server
npm install
cd ..

# Install Python dependencies
cd backend
pip install -r requirements.txt
cd ..
```

### 2. Environment Configuration

**Frontend** (root directory):
```bash
# Create .env.local
echo "VITE_API_URL=http://localhost:5000" > .env.local
```

**Backend** (server directory):
```bash
cd server
# Create .env file
cat > .env << EOF
MONGO_URI=mongodb://127.0.0.1:27017/safezone
PY_SERVICE_URL=http://127.0.0.1:8000
PORT=5000
NODE_ENV=development
EOF
cd ..
```

### 3. Run the Application

You need to run all three services:

```bash
# Terminal 1: Frontend
npm run dev

# Terminal 2: Node.js Backend
cd server
npm start

# Terminal 3: Python AI Service
cd backend
python app.py
```

Access the application at `http://localhost:5173`

## Production Deployment

### Deploy to Render.com

1. **Prerequisites**:
   - GitHub account with your code pushed
   - Render.com account (free tier available)
   - MongoDB Atlas account (free tier)

2. **Deploy Using render.yaml**:
   - Push your code to GitHub
   - Go to [Render Dashboard](https://dashboard.render.com/)
   - Click "New" → "Blueprint"
   - Connect your GitHub repository
   - Render will automatically detect `render.yaml` and create both services

3. **Configure Environment Variables in Render**:
   
   **For safezone-backend service:**
   - `MONGO_URI`: Your MongoDB Atlas connection string
   - `PY_SERVICE_URL`: URL of your Python service (e.g., `https://safezone-ai-service.onrender.com`)
   
   **For safezone-ai-service:**
   - No additional variables needed

4. **Configure Frontend on Vercel**:
   - Go to Vercel project settings
   - Add environment variable:
     - `VITE_API_URL`: Your Render backend URL (e.g., `https://safezone-backend.onrender.com`)
   - Redeploy the frontend

### MongoDB Atlas Setup

1. Go to [MongoDB Atlas](https://www.mongodb.com/cloud/atlas)
2. Create a free cluster
3. Create a database user
4. Whitelist Render's IP addresses (or use `0.0.0.0/0` for all IPs)
5. Get your connection string and add it to Render environment variables

## Features

- **AI Anomaly Detection**: Real-time video analysis using Vision Transformer models
- **ROI-Based Monitoring**: Focus on restricted zones
- **Incident Management**: Track, acknowledge, and resolve security incidents
- **Live Dashboard**: Real-time system status and camera feeds
- **WebSocket Alerts**: Instant notifications for detected anomalies

## Technology Stack

**Frontend:**
- React 18 with TypeScript
- Tailwind CSS & Radix UI components
- Vite for build tooling
- Sonner for notifications

**Backend:**
- Node.js + Express
- Socket.IO for real-time communication
- Multer for file uploads
- Mongoose for MongoDB ODM

**AI Service:**
- FastAPI
- PyTorch + TorchVision (ViT models)
- OpenCV for video processing
- Scikit-learn (Isolation Forest)

## Model Files

The AI models are stored in `backend/model/`:
- `vit_feature_extractor.pth` - Vision Transformer for feature extraction
- `isolation_forest_model.joblib` - Anomaly detection model

These files are tracked with Git LFS and will be deployed with the Python service.

## Support

For issues or questions, please open an issue on the GitHub repository.

## License

This project was built with Figma integration. Original design: [CCTV Dashboard UI](https://www.figma.com/design/4TpyWociWrvZNvu8UEub6J/CCTV-Maintenance-Dashboard-UI)
