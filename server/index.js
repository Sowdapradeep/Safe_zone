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

mongoose.connect(MONGO_URI)
    .then(() => {
        console.log('✅ MongoDB Connected successfully');
        console.log(`Connection string: ${MONGO_URI.replace(/:([^:@]+)@/, ':****@')}`);
    })
    .catch(err => {
        console.error('❌ MongoDB Connection Error:', err.message);
        console.error('Ensure MONGO_URI is set correctly in environment variables.');
    });

// Multer Setup for File Uploads
const upload = multer({ dest: 'uploads/' });

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

        console.log(`Forwarding ${fileName} to Python AI Service at ${PY_SERVICE_URL}...`);

        // Simple retry logic for production stability
        let response;
        let retries = 2;
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
                if (retries === 0) throw err;
                console.log(`[PROD-LOG] Backend busy or restarting, retrying... (${retries} left)`);
                await new Promise(resolve => setTimeout(resolve, 5000));
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
        console.log(`[LOG] Fetching incidents... DB State: ${mongoose.connection.readyState}`);
        const incidents = await Incident.find().sort({ timestamp: -1 }).limit(50);
        res.json(incidents);
    } catch (err) {
        console.error('[ERROR] Failed to fetch incidents:', err.message);
        res.status(500).json({
            error: 'Failed to retrieve incidents from database',
            details: err.message,
            dbState: mongoose.connection.readyState
        });
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

// Production Timeout Fixes
server.timeout = 1800000; // 30 minutes
server.keepAliveTimeout = 65000; // Slightly higher than common ELB timeouts
server.headersTimeout = 66000;
