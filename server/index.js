const express = require('express');
const mongoose = require('mongoose');
const cors = require('cors');
const http = require('http');
const { Server } = require('socket.io');
const multer = require('multer');
const axios = require('axios');
const FormData = require('form-data');
const fs = require('fs');
const path = require('path');
const Incident = require('./models/Incident');

const app = express();
const server = http.createServer(app);
const io = new Server(server, {
    cors: {
        origin: "*",
        methods: ["GET", "POST"]
    }
});

// Middleware
app.use(cors());
app.use(express.json());

// MongoDB Connection
// Replace with your actual Mongo URI or use localhost
const MONGO_URI = process.env.MONGO_URI || 'mongodb://127.0.0.1:27017/safezone';

mongoose.connect(MONGO_URI)
    .then(() => console.log('MongoDB Connected'))
    .catch(err => console.error('MongoDB Connection Error:', err));

// Multer Setup for File Uploads
const upload = multer({ dest: 'uploads/' });

// --- AI Service Integration (Python Microservice) ---
// We assume the existing Python FastAPI is running on port 8000
const PY_SERVICE_URL = 'http://127.0.0.1:8000';

// API Routes

// 1. Upload Video for Analysis
app.post('/api/analyze-video', upload.single('file'), async (req, res) => {
    if (!req.file) {
        return res.status(400).send('No file uploaded');
    }

    try {
        const filePath = req.file.path;
        const fileName = req.file.originalname;

        // Forward to Python Service
        const form = new FormData();
        form.append('file', fs.createReadStream(filePath), fileName);

        console.log(`Forwarding ${fileName} to Python AI Service...`);

        const response = await axios.post(`${PY_SERVICE_URL}/analyze-video`, form, {
            headers: {
                ...form.getHeaders()
            },
            responseType: 'stream', // We want the video stream back
            timeout: 300000 // 5 minutes timeout to prevent hanging
        });

        // Pipe the response directly back to the client
        // We also need to forward the custom headers for anomalies
        if (response.headers['x-anomaly-detected']) {
            res.set('X-Anomaly-Detected', response.headers['x-anomaly-detected']);
        }
        if (response.headers['x-anomaly-frames']) {
            res.set('X-Anomaly-Frames', response.headers['x-anomaly-frames']);
        }

        res.set('Content-Type', 'video/mp4');
        response.data.pipe(res);

        // Clean up temp file after a delay
        setTimeout(() => {
            fs.unlink(filePath, () => { });
        }, 60000);

    } catch (error) {
        console.error('AI Service Error:', error.message);
        res.status(502).json({ error: 'Failed to process video via AI service' });
    }
});

// 2. Incident Management
app.get('/api/incidents', async (req, res) => {
    try {
        const incidents = await Incident.find().sort({ timestamp: -1 }).limit(50);
        res.json(incidents);
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

app.post('/api/incidents', async (req, res) => {
    try {
        const newIncident = new Incident(req.body);
        const saved = await newIncident.save();

        // Broadcast to connected clients
        io.emit('new-incident', saved);

        res.status(201).json(saved);
    } catch (err) {
        res.status(400).json({ error: err.message });
    }
});

app.patch('/api/incidents/:id', async (req, res) => {
    try {
        const updated = await Incident.findByIdAndUpdate(req.params.id, req.body, { new: true });
        res.json(updated);
    } catch (err) {
        res.status(400).json({ error: err.message });
    }
});

// 3. Live Feed Proxy (Optional - simpler to let frontend hit Python directly for streams)
// We provide the URL where the frontend can find the stream
app.get('/api/live-feed-url', (req, res) => {
    res.json({ url: `${PY_SERVICE_URL}/live-feed` });
});


// Socket.IO for Real-time Alerts
// We can listen to the Python service (via a separate channel or webhook) 
// OR the Python service pushes to *us*.
// For now, let's assume the Python service speaks WS to port 8000 clients.
// Ideally, we'd migrate Python to POST alerts to this Node server.

// Listen
const PORT = process.env.PORT || 5000;
server.listen(PORT, () => {
    console.log(`Node.js Backend running on http://localhost:${PORT}`);
});
