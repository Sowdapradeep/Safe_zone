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
app.use('/uploads', express.static(path.join(__dirname, 'uploads')));

// MongoDB Connection
// Replace with your actual Mongo URI or use localhost
const MONGO_URI = process.env.MONGO_URI || 'mongodb://127.0.0.1:27017/safezone';

// Multer Setup
const upload = multer({ dest: 'uploads/' });

// In-memory fallback for production if MongoDB is dead
let incidentFallback = [];

mongoose.connect(MONGO_URI, { serverSelectionTimeoutMS: 5000 })
    .then(() => {
        console.log('✅ MongoDB Connected successfully');
    })
    .catch(err => {
        console.error('❌ MongoDB Connection Error:', err.message);
        console.log('⚠️ Falling back to in-memory incident storage (Temporary).');
    });

// --- AI Service Integration (Python Microservice) ---
// We assume the existing Python FastAPI is running on port 8000
const PY_SERVICE_URL = process.env.PY_SERVICE_URL || 'http://127.0.0.1:8000';

// API Routes

// 1. Upload Video for Analysis
app.post('/api/analyze-video', upload.single('file'), async (req, res) => {
    const startTime = Date.now();
    console.log(`[PROD-LOG] Received upload request at ${new Date().toISOString()}`);

    if (!req.file) {
        console.error('[PROD-LOG] No file received in request');
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

        console.log(`Forwarding ${fileName} to AI Service: ${PY_SERVICE_URL}`);

        if (PY_SERVICE_URL.includes('127.0.0.1') && process.env.NODE_ENV === 'production') {
            console.warn('[WARNING] PY_SERVICE_URL is localhost in production environment!');
        }

        let response;
        let retries = 5; // Increased retries for model pre-warming
        while (retries > 0) {
            try {
                response = await axios.post(`${PY_SERVICE_URL}/analyze-video`, form, {
                    headers: { ...form.getHeaders() },
                    timeout: 1800000,
                    maxContentLength: Infinity,
                    maxBodyLength: Infinity
                });
                break;
            } catch (err) {
                retries--;
                const status = err.response ? err.response.status : 'No Response';
                console.log(`[PROD-LOG] AI Service attempt failed (Status: ${status}, Error: ${err.code}). Retrying... (${retries} left)`);

                if (retries === 0) throw err;
                await new Promise(resolve => setTimeout(resolve, 3000));
            }
        }

        console.log(`[PROD-LOG] Analysis complete for ${fileName} in ${Date.now() - startTime}ms`);

        // Update: The response now contains a job_id for async processing
        return res.json(response.data);

    } catch (error) {
        let errorMsg = error.message;
        if (error.response && error.response.data) {
            errorMsg = error.response.data.detail || error.message;
        }

        console.error('Backend Error details:', {
            message: errorMsg,
            code: error.code
        });

        if (!res.headersSent) {
            res.status(502).json({
                error: 'Failed to queue video analysis',
                details: errorMsg,
                code: error.code
            });
        }
    }
});

// 1b. Polling Endpoint Proxy
app.get('/api/check-status/:jobId', async (req, res) => {
    try {
        const response = await axios.get(`${PY_SERVICE_URL}/api/check-status/${req.params.jobId}`);
        return res.json(response.data);
    } catch (error) {
        console.error('Status Check Error:', error.message);
        res.status(502).json({ error: 'Failed to check analysis status' });
    }
});

// 2. Proxy for Analyzed Video Files
app.get('/api/video/:filename', async (req, res) => {
    try {
        console.log(`Proxying video request: ${req.params.filename}`);
        const response = await axios.get(`${PY_SERVICE_URL}/api/video/${req.params.filename}`, {
            responseType: 'stream'
        });
        res.set('Content-Type', response.headers['content-type']);
        response.data.pipe(res);
    } catch (error) {
        console.error('Video Proxy Error:', error.message);
        res.status(404).send('Video not found');
    }
});


// 2. Incident Management
app.get('/api/incidents', async (req, res) => {
    try {
        if (mongoose.connection.readyState === 1) {
            const incidents = await Incident.find().sort({ timestamp: -1 }).limit(50);
            return res.json(incidents);
        } else {
            console.log('[LOG] DB disconnected, serving from fallback memory');
            return res.json(incidentFallback);
        }
    } catch (err) {
        console.error('[ERROR] Failed to fetch incidents:', err.message);
        res.status(200).json(incidentFallback); // Avoid 500 for UI stability
    }
});

app.post('/api/incidents', async (req, res) => {
    try {
        if (mongoose.connection.readyState === 1) {
            const newIncident = new Incident(req.body);
            const saved = await newIncident.save();
            broadcastAnomaly(saved);
            return res.status(201).json(saved);
        } else {
            const tempIncident = { ...req.body, _id: Date.now().toString(), timestamp: new Date() };
            incidentFallback.unshift(tempIncident);
            if (incidentFallback.length > 50) incidentFallback.pop();
            broadcastAnomaly(tempIncident);
            return res.status(201).json(tempIncident);
        }
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
// Health check endpoint
app.get('/api/health', (req, res) => {
    res.json({
        status: 'ok',
        database: mongoose.connection.readyState === 1 ? 'connected' : 'disconnected',
        ai_service: PY_SERVICE_URL,
        timestamp: new Date().toISOString(),
        env: process.env.NODE_ENV || 'development'
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

// Production Timeout Fixes
server.timeout = 1800000; // 30 minutes
server.keepAliveTimeout = 65000; // Slightly higher than common ELB timeouts
server.headersTimeout = 66000;
