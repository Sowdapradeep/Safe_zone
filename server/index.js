require('dotenv').config();
const express = require('express');
const mongoose = require('mongoose');
const cors = require('cors');
const http = require('http');
const { Server: SocketIOServer } = require('socket.io');
const { WebSocketServer } = require('ws');
const multer = require('multer');
const axios = require('axios');
const FormData = require('form-data');
const fs = require('fs');
const path = require('path');
const Incident = require('./models/Incident');

const app = express();
const server = http.createServer(app);

// 1. Socket.IO (for legacy support or other clients)
const io = new SocketIOServer(server, {
    cors: {
        origin: "*",
        methods: ["GET", "POST"]
    }
});

// 2. Raw WebSocket Server (for frontend App.tsx)
const wss = new WebSocketServer({ noServer: true });

wss.on('connection', (ws) => {
    console.log('New WS Client connected');
    ws.on('close', () => console.log('WS Client disconnected'));
});

// Handle upgrade from HTTP to WS
server.on('upgrade', (request, socket, head) => {
    wss.handleUpgrade(request, socket, head, (ws) => {
        wss.emit('connection', ws, request);
    });
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
const PY_SERVICE_URL = process.env.PY_SERVICE_URL || 'http://127.0.0.1:8000';

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
        form.append('file', fs.createReadStream(filePath), {
            filename: fileName,
            contentType: 'video/mp4'
        });

        console.log(`Forwarding ${fileName} to Python AI Service at ${PY_SERVICE_URL}...`);

        const response = await axios.post(`${PY_SERVICE_URL}/analyze-video`, form, {
            headers: {
                ...form.getHeaders()
            },
            responseType: 'stream',
            timeout: 600000, // 10 minutes for large videos
            maxContentLength: Infinity,
            maxBodyLength: Infinity
        });

        // Forward headers
        if (response.headers['x-anomaly-detected']) {
            res.set('X-Anomaly-Detected', response.headers['x-anomaly-detected']);
        }
        if (response.headers['x-anomaly-frames']) {
            res.set('X-Anomaly-Frames', response.headers['x-anomaly-frames']);
        }

        res.set('Content-Type', response.headers['content-type'] || 'video/mp4');

        // Handle stream errors
        response.data.on('error', (err) => {
            console.error('Download Stream Error:', err.message);
            if (!res.headersSent) {
                res.status(502).json({ error: 'Stream interrupted' });
            }
        });

        response.data.pipe(res);

        // Clean up temp file after a delay
        setTimeout(() => {
            if (fs.existsSync(filePath)) {
                fs.unlink(filePath, (err) => {
                    if (err) console.error('Cleanup Error:', err.message);
                });
            }
        }, 120000); // 2 minutes

    } catch (error) {
        let errorMsg = error.message;
        if (error.response) {
            errorMsg = `AI Service Error (${error.response.status})`;
            // Can't easily log data here as it's a stream
        }

        console.error('Backend Error:', errorMsg);
        if (!res.headersSent) {
            res.status(502).json({
                error: 'Failed to process video via AI service',
                details: errorMsg
            });
        }
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
        broadcastAnomaly(saved);

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

// Health check endpoint for Render
app.get('/', (req, res) => {
    res.json({
        status: 'ok',
        service: 'SafeZone Backend',
        timestamp: new Date().toISOString()
    });
});

// Socket.IO and WS for Real-time Alerts
function broadcastAnomaly(data) {
    // Broadcast via Socket.IO
    io.emit('new-incident', data);

    // Broadcast via Raw WebSocket
    wss.clients.forEach((client) => {
        if (client.readyState === 1) { // 1 = OPEN
            client.send(JSON.stringify(data));
        }
    });
}

// Listen
const PORT = process.env.PORT || 5000;
server.listen(PORT, '0.0.0.0', () => {
    console.log(`Node.js Backend running on port ${PORT}`);
    console.log(`MongoDB URI: ${MONGO_URI ? 'Configured' : 'Not configured'}`);
    console.log(`Python Service URL: ${PY_SERVICE_URL}`);
});
