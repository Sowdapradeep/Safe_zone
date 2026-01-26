const mongoose = require('mongoose');

const IncidentSchema = new mongoose.Schema({
    cameraId: { type: String, required: true },
    location: { type: String, required: true },
    zone: { type: String, required: true },
    incidentType: { type: String, default: 'Anomaly Detected' },
    severity: { type: String, enum: ['low', 'medium', 'high'], default: 'medium' },
    status: { type: String, enum: ['open', 'in-progress', 'false-alarm', 'closed'], default: 'open' },
    confidence: { type: Number, required: true },
    description: { type: String },
    timestamp: { type: Date, default: Date.now },
    videoUrl: { type: String } // Path to the clip if saved
});

module.exports = mongoose.model('Incident', IncidentSchema);
