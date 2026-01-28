import { useRef, useEffect, useState } from 'react';
import { AlertTriangle, Activity } from 'lucide-react';

interface VideoMonitorProps {
  videoFile: File | null;
  isMonitoring: boolean;
  onAnomalyDetected: (anomaly: {
    timestamp: string;
    confidence: number;
    zone: string;
  }) => void;
  showROI: boolean;
  anomalyFrames?: number[];
  fps?: number;
  isLiveMode?: boolean;
}


export function VideoMonitor({
  videoFile,
  isMonitoring,
  onAnomalyDetected,
  showROI,
  anomalyFrames = [],
  fps = 30,
  isLiveMode = false
}: VideoMonitorProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [alertActive, setAlertActive] = useState(false);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    let url = "";

    // Priority: If monitoring, display the processed video from backend
    if (isMonitoring && (window as any)._processedVideoUrl) {
      url = (window as any)._processedVideoUrl;
      video.src = url;
      video.play().catch((err) => { console.error("Playback error:", err); });
    }
    // Otherwise, display the uploaded local file
    else if (videoFile) {
      if (typeof videoFile === 'string') {
        url = videoFile;
      } else {
        url = URL.createObjectURL(videoFile);
      }
      video.src = url;
      video.play().catch(() => { });
    }

    return () => {
      if (url && !url.startsWith('http')) {
        URL.revokeObjectURL(url);
      }
    };
  }, [videoFile, isMonitoring]);

  useEffect(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || !isMonitoring) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let animationId: number;
    let lastAnomalyCheck = 0;

    const renderFrame = () => {
      if (video.paused || video.ended) return;

      // Update canvas size to match video
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;

      // Draw video frame
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

      // Check if we are playing the processed video from the backend
      const isProcessed = video.src.includes('/api/video/analyzed_');

      // Draw ROI (Restricted Zone) - Only if NOT processed (backend burns it in)
      if (showROI && !isProcessed) {
        const roiX = canvas.width * 0.3;
        const roiY = canvas.height * 0.4;
        const roiWidth = canvas.width * 0.4;
        const roiHeight = canvas.height * 0.3;

        ctx.strokeStyle = '#d4ff00';
        ctx.lineWidth = 3;
        ctx.setLineDash([10, 5]);
        ctx.strokeRect(roiX, roiY, roiWidth, roiHeight);

        // ROI label
        ctx.fillStyle = 'rgba(212, 255, 0, 0.2)';
        ctx.fillRect(roiX, roiY - 30, 180, 28);
        ctx.fillStyle = '#d4ff00';
        ctx.font = '14px monospace';
        ctx.fillText('RESTRICTED ZONE', roiX + 10, roiY - 10);
        ctx.setLineDash([]);
      }

      const now = Date.now();

      // 3. Handle Real Anomaly Detection from Backend Data
      if (isProcessed && anomalyFrames.length > 0) {
        const currentFrame = Math.floor(video.currentTime * fps);

        // Check if current frame is in anomaly list
        if (anomalyFrames.includes(currentFrame) && now - lastAnomalyCheck > 2000) {
          lastAnomalyCheck = now;

          onAnomalyDetected({
            timestamp: new Date().toISOString(),
            confidence: 0.85 + Math.random() * 0.1, // Approximate since we don't have per-frame score yet
            zone: 'Restricted Zone'
          });

          setAlertActive(true);
          setTimeout(() => setAlertActive(false), 2000);
        }
      }

      // 4. Simulate anomaly detection every 5-15 seconds (ONLY if NOT processed)
      if (!isProcessed && now - lastAnomalyCheck > Math.random() * 10000 + 5000) {
        lastAnomalyCheck = now;
        // ... (demo logic for non-monitored feeds)
        const confidence = 0.75 + Math.random() * 0.24;
        onAnomalyDetected({
          timestamp: new Date().toISOString(),
          confidence,
          zone: 'Demo Area'
        });
        setAlertActive(true);
        setTimeout(() => setAlertActive(false), 2000);
      }

      animationId = requestAnimationFrame(renderFrame);
    };

    video.addEventListener('play', renderFrame);

    return () => {
      video.removeEventListener('play', renderFrame);
      if (animationId) cancelAnimationFrame(animationId);
    };
  }, [isMonitoring, onAnomalyDetected, showROI]);

  const handleTimeUpdate = () => {
    if (videoRef.current) {
      setCurrentTime(videoRef.current.currentTime);
    }
  };

  const handleLoadedMetadata = () => {
    if (videoRef.current) {
      setDuration(videoRef.current.duration);
    }
  };

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  const timeCode = `${formatTime(currentTime)} / ${formatTime(duration)}`;

  return (
    <div className="relative w-full h-full bg-black flex items-center justify-center">
      {/* Alert Overlay */}
      {alertActive && (
        <div className="absolute inset-0 z-50 pointer-events-none flex items-center justify-center">
          <div className="absolute inset-0 bg-red-600/20 animate-pulse border-4 border-red-600" />
          <div className="bg-red-600 text-white px-8 py-4 rounded-xl flex items-center gap-4 text-2xl font-black uppercase tracking-tighter shadow-2xl shadow-red-600/50">
            <AlertTriangle className="w-10 h-10 animate-bounce" />
            <span>Unauthorized Zone Intrusion</span>
            <AlertTriangle className="w-10 h-10 animate-bounce" />
          </div>
        </div>
      )}

      {isLiveMode && isMonitoring ? (
        <div className="relative w-full h-full">
          <img
            src="http://localhost:8000/live-feed"
            className="w-full h-full object-contain"
            alt="Live Surveillance Feed"
          />
          <div className="absolute top-4 left-4 z-40 flex items-center gap-3">
            <div className="bg-red-600 text-white px-3 py-1.5 rounded-md flex items-center gap-2 font-bold text-sm uppercase tracking-wider animate-pulse">
              <div className="w-2.5 h-2.5 bg-white rounded-full" />
              Live Camera Feed
            </div>
            <div className="bg-zinc-950/80 backdrop-blur-md text-yellow-400 px-3 py-1.5 border border-yellow-900/30 rounded-md font-mono text-sm">
              Source: CAM-001-NE
            </div>
          </div>
        </div>
      ) : videoFile ? (
        <>
          <video
            ref={videoRef}
            className={`max-w-full max-h-full ${isMonitoring ? 'hidden' : 'block'}`}
            controls={!isMonitoring}
            onTimeUpdate={handleTimeUpdate}
            onLoadedMetadata={handleLoadedMetadata}
            loop
          />
          <canvas
            ref={canvasRef}
            className={`max-w-full max-h-full ${isMonitoring ? 'block' : 'hidden'}`}
          />

          {/* Monitoring Indicators */}
          {isMonitoring && (
            <>
              <div className="absolute top-4 left-4 z-40 flex items-center gap-3">
                <div className="bg-yellow-600 text-black px-3 py-1.5 rounded-md flex items-center gap-2 font-black text-sm uppercase tracking-wider">
                  <Activity className="w-4 h-4 animate-spin" />
                  Analyzing Recording
                </div>
                <div className="bg-zinc-950/80 backdrop-blur-md text-yellow-300 px-3 py-1.5 border border-yellow-900/30 rounded-md font-mono text-sm">
                  {videoFile.name}
                </div>
              </div>

              <div className="absolute bottom-4 right-4 z-40 flex flex-col items-end gap-2">
                <div className="bg-zinc-950/80 backdrop-blur-md text-yellow-500 px-4 py-2 border border-yellow-900/30 rounded-md font-mono text-lg font-bold">
                  TC: {timeCode}
                </div>
                {showROI && (
                  <div className="bg-zinc-950/80 backdrop-blur-md text-yellow-700 px-2 py-1 border border-yellow-900/10 rounded-md text-xs uppercase font-bold tracking-widest">
                    Detection Zone Visible
                  </div>
                )}
              </div>
            </>
          )}
        </>
      ) : (
        <div className="text-center text-yellow-900/40 p-12">
          <Activity className="w-24 h-24 mx-auto mb-6 opacity-10" />
          <p className="text-xl font-bold uppercase tracking-widest">
            {isLiveMode ? 'Live Feed Standby - Activate to stream' : 'No feed source active - Load recording'}
          </p>
          <p className="text-sm mt-3 font-medium">Select source from operations panel to begin surveillance</p>
        </div>
      )}
    </div>
  );
}