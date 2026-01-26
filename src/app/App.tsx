import { useState, useCallback, useEffect } from 'react';
import { VideoMonitor } from './components/video-monitor';
import { SystemStatusPanel } from './components/system-status-panel';
import { OperationsPanel } from './components/operations-panel';
import { IncidentLog, Incident } from './components/incident-log';
import { CameraAssetInfo } from './components/camera-asset-info';
import { Shield, Activity } from 'lucide-react';
import { Toaster } from './components/ui/sonner';
import { toast } from 'sonner';

// Get API URL from environment variable, fallback to localhost for development
const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000';

export default function App() {
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [isMonitoring, setIsMonitoring] = useState(false);
  const [showROI, setShowROI] = useState(true);

  const [systemState, setSystemState] = useState<'idle' | 'monitoring' | 'alert' | 'maintenance'>('idle');
  const [threatLevel, setThreatLevel] = useState<'low' | 'medium' | 'high'>('low');
  const [lastAnomalyConfidence, setLastAnomalyConfidence] = useState<number | null>(null);
  const [incidents, setIncidents] = useState<Incident[]>([]);





  const handleVideoUpload = useCallback((file: File) => {

    setVideoFile(file);
    toast.success('Video feed loaded successfully', {
      description: `Recording: ${file.name}`
    });
  }, []);

  const [isProcessing, setIsProcessing] = useState(false);

  const handleStartMonitoring = useCallback(async () => {

    if (!videoFile) return;

    setIsProcessing(true);
    toast.info('Initiating AI Analysis...', {
      description: 'Uploading and processing video stream. This may take a moment based on file size.',
      duration: Infinity, // Keep open until done
      id: 'processing-toast'
    });

    try {
      const formData = new FormData();
      formData.append('file', videoFile);

      const response = await fetch(`${API_URL}/api/analyze-video`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        let errorMessage = `Server responded with ${response.status}`;
        try {
          const errorData = await response.json();
          if (errorData.details) errorMessage = errorData.details;
          else if (errorData.error) errorMessage = errorData.error;
        } catch (e) {
          // If not JSON, use default status message
        }
        throw new Error(errorMessage);
      }

      // Get the processed video blob
      const blob = await response.blob();
      const processedFile = new File([blob], `analyzed_${videoFile.name}`, { type: 'video/mp4' });

      // Update UI with processed video
      setVideoFile(processedFile);
      setIsMonitoring(true);
      setSystemState('monitoring');

      // Parse headers for anomalies
      const anomalyDetected = response.headers.get('x-anomaly-detected') === 'True';
      if (anomalyDetected) {
        // Trigger an initial alert if immediate anomalies were found
        const anomalyFrames = response.headers.get('x-anomaly-frames');
        toast.error('Security Alert', {
          description: `Anomalies detected in processed feed. Frames: ${anomalyFrames}`,
          duration: 5000
        });
        setThreatLevel('high');
        setSystemState('alert');
      }

      toast.dismiss('processing-toast');
      toast.success('Monitoring Active', {
        description: 'AI anomaly detection stream is live',
        icon: <Activity className="w-4 h-4" />
      });

    } catch (error: any) {
      console.error('Analysis failed:', error);
      toast.dismiss('processing-toast');
      toast.error('System Error', {
        description: error.message || 'Failed to connect to analysis server. Ensure backend is running.',
      });
      setIsMonitoring(false);
    } finally {
      setIsProcessing(false);
    }
  }, [videoFile]);

  const handleStopMonitoring = useCallback(() => {
    setIsMonitoring(false);
    setSystemState('idle');
    setThreatLevel('low');
    toast.info('Monitoring stopped', {
      description: 'System returned to idle state'
    });
  }, []);

  const handleAnomalyDetected = useCallback((anomaly: {
    timestamp: string;
    confidence: number;
    zone: string;
  }) => {
    setLastAnomalyConfidence(anomaly.confidence);
    setSystemState('alert');

    // Determine threat level based on confidence
    if (anomaly.confidence > 0.9) {
      setThreatLevel('high');
    } else if (anomaly.confidence > 0.75) {
      setThreatLevel('medium');
    } else {
      setThreatLevel('low');
    }

    // Create new incident
    const newIncident: Incident = {
      id: `incident-${Date.now()}`,
      cameraId: 'CAM-001-NE',
      location: 'Building A / Floor 2 / Zone North-East',
      zone: anomaly.zone,
      incidentType: 'Anomaly Detected',
      severity: anomaly.confidence > 0.9 ? 'high' : anomaly.confidence > 0.75 ? 'medium' : 'low',
      status: 'open',
      timestamp: anomaly.timestamp,
      confidence: anomaly.confidence,
      description: `AI-detected anomalous activity in ${anomaly.zone}. Confidence: ${(anomaly.confidence * 100).toFixed(1)}%. Requires operator review.`
    };

    setIncidents(prev => [newIncident, ...prev]);

    toast.error('ANOMALY DETECTED', {
      description: `${anomaly.zone} - Confidence: ${(anomaly.confidence * 100).toFixed(1)}%`,
      duration: 5000
    });

    // Return to monitoring state after alert
    setTimeout(() => {
      setSystemState('monitoring');
    }, 3000);
  }, []);

  // WebSocket for Live Alerts (Real-time events from backend)
  useEffect(() => {
    if (!isMonitoring) return;

    // Note: Node.js backend might require socket.io client lib, but for raw ws if configured:
    const wsUrl = API_URL.replace('http', 'ws'); // Convert http:// to ws:// or https:// to wss://
    const socket = new WebSocket(wsUrl);

    socket.onmessage = (event) => {
      const data = JSON.parse(event.data);
      handleAnomalyDetected(data);
    };

    return () => socket.close();
  }, [isMonitoring, handleAnomalyDetected]);


  const handleScanIncidents = useCallback(() => {
    toast.info('Manual incident scan initiated', {
      description: 'Analyzing current feed for anomalies...'
    });
  }, []);

  const handleAcknowledgeIncident = useCallback((id: string) => {
    setIncidents(prev =>
      prev.map(incident =>
        incident.id === id ? { ...incident, status: 'in-progress' as const } : incident
      )
    );
    toast.success('Incident acknowledged', {
      description: 'Status updated to In Progress'
    });
  }, []);

  const handleMarkFalseAlarm = useCallback((id: string) => {
    setIncidents(prev =>
      prev.map(incident =>
        incident.id === id ? { ...incident, status: 'false-alarm' as const } : incident
      )
    );
    toast.info('Marked as false alarm', {
      description: 'Incident removed from active queue'
    });
  }, []);

  const handleResolveIncident = useCallback((id: string) => {
    setIncidents(prev =>
      prev.map(incident =>
        incident.id === id ? { ...incident, status: 'closed' as const } : incident
      )
    );
    toast.success('Incident resolved', {
      description: 'Incident has been closed'
    });
  }, []);

  return (
    <div className="h-screen bg-zinc-950 text-white flex flex-col overflow-hidden">
      {/* Header */}
      <header className="bg-zinc-900 border-b border-yellow-900/30 px-6 py-4 flex items-center justify-between">
        <div className="flex items-center gap-4">
          <div className="bg-yellow-600 p-2.5 rounded">
            <Shield className="w-7 h-7 text-black" />
          </div>
          <div>
            <h1 className="text-2xl font-bold text-yellow-400">SafeZone Surveillance</h1>
            <p className="text-sm text-yellow-700">CCTV Maintenance & Monitoring System</p>
          </div>
        </div>
        <div className="flex items-center gap-6">
          <div className="text-right">
            <div className="text-xs text-yellow-700 mb-1">Control Room Operator</div>
            <div className="text-base font-semibold text-yellow-300">Station Alpha-01</div>
          </div>
          <div className="text-right">
            <div className="text-xs text-yellow-700 mb-1">System Time</div>
            <div className="text-base font-mono font-semibold text-lime-400">
              {new Date().toLocaleTimeString('en-US', { hour12: false })}
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <div className="flex-1 grid grid-cols-12 gap-5 p-5 overflow-hidden">
        {/* Left Panel - System Status & Camera Info */}
        <div className="col-span-3 space-y-5 overflow-y-auto">
          <SystemStatusPanel
            systemState={systemState}
            cameraHealth="online"
            threatLevel={threatLevel}
            anomalyConfidence={lastAnomalyConfidence}
            totalCameras={12}
            onlineCameras={11}
          />

          <CameraAssetInfo
            cameraId="CAM-001-NE"
            location="Building A / Floor 2 / Zone North-East"
            building="Building A - Main Facility"
            floor="Floor 2"
            zone="North-East Sector"
            status="online"
            lastMaintenance="2025-11-15"
            nextMaintenance="2026-02-15"
            uptime="45 days 12 hrs"
            resolution="1920x1080"
            fps={30}
          />
        </div>

        {/* Center Panel - Video Monitor */}
        <div className="col-span-6 flex flex-col">
          <div className="bg-zinc-900 border border-yellow-900/30 rounded-lg p-4 mb-3">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-xs text-yellow-700 mb-1">Active Camera</div>
                <div className="text-base font-bold font-mono text-yellow-400">CAM-001-NE</div>
              </div>
              <div>
                <div className="text-xs text-yellow-700 mb-1">Location</div>
                <div className="text-base font-bold text-yellow-200">Building A / Floor 2 / North-East</div>
              </div>
              <div>
                <div className="text-xs text-yellow-700 mb-1">Feed Status</div>
                <div className="text-base font-bold text-lime-400">
                  {isMonitoring ? 'LIVE' : 'STANDBY'}
                </div>
              </div>
            </div>
          </div>

          <div className="flex-1 bg-zinc-900 border border-yellow-900/30 rounded-lg overflow-hidden">
            <VideoMonitor
              videoFile={videoFile}
              isMonitoring={isMonitoring}
              onAnomalyDetected={handleAnomalyDetected}
              showROI={showROI}
            />
          </div>
        </div>

        {/* Right Panel - Operations & Incidents */}
        <div className="col-span-3 space-y-5 overflow-y-auto">
          <OperationsPanel
            onVideoUpload={handleVideoUpload}
            onStartMonitoring={handleStartMonitoring}
            onStopMonitoring={handleStopMonitoring}
            onScanIncidents={handleScanIncidents}
            isMonitoring={isMonitoring}
            hasVideo={videoFile !== null}
            showROI={showROI}
            onToggleROI={setShowROI}
            isProcessing={isProcessing}
          />


          <div className="h-[calc(100vh-650px)] min-h-[400px]">
            <IncidentLog
              incidents={incidents}
              onAcknowledge={handleAcknowledgeIncident}
              onMarkFalseAlarm={handleMarkFalseAlarm}
              onResolve={handleResolveIncident}
            />
          </div>
        </div>
      </div>

      {/* Footer */}
      <footer className="bg-zinc-900 border-t border-yellow-900/30 px-6 py-3 flex items-center justify-between text-sm text-yellow-700">
        <div className="font-medium">
          SafeZone Surveillance v2.4.1 | AI Anomaly Detection Active | Updated: {new Date().toLocaleDateString()}
        </div>
        <div className="flex items-center gap-6 font-medium">
          <span>System: Operational</span>
          <span>Network: Connected</span>
          <span>Storage: 78%</span>
        </div>
      </footer>

      <Toaster position="top-right" theme="dark" />
    </div>
  );
}