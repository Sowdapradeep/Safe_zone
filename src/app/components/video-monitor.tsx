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
}


export function VideoMonitor({
  videoFile,
  isMonitoring,
  onAnomalyDetected,
  showROI
}: VideoMonitorProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [alertActive, setAlertActive] = useState(false);

  useEffect(() => {
    if (videoFile && videoRef.current) {
      const url = URL.createObjectURL(videoFile);
      videoRef.current.src = url;

      // Auto-play when a new file (especially analyzed one) is loaded
      videoRef.current.play().catch(err => {
        console.warn("Autoplay failed, user interaction might be needed:", err);
      });

      return () => {
        URL.revokeObjectURL(url);
      };
    }
  }, [videoFile]);

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

      const isProcessed = videoFile?.name.startsWith('analyzed_');

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

      // Simulate anomaly detection every 5-15 seconds
      const now = Date.now();
      if (!isProcessed && now - lastAnomalyCheck > Math.random() * 10000 + 5000) {
        lastAnomalyCheck = now;

        // Random anomaly detection (20% chance)
        if (Math.random() < 0.2) {
          const confidence = 0.75 + Math.random() * 0.24; // 75-99%
          const zones = ['Restricted Zone', 'Perimeter', 'Entry Point'];
          const zone = zones[Math.floor(Math.random() * zones.length)];

          onAnomalyDetected({
            timestamp: new Date().toISOString(),
            confidence,
            zone
          });

          setAlertActive(true);
          setTimeout(() => setAlertActive(false), 3000);

          // Draw alert indicator
          ctx.strokeStyle = '#ef4444';
          ctx.lineWidth = 4;
          ctx.setLineDash([]);
          const alertX = canvas.width * 0.5;
          const alertY = canvas.height * 0.5;
          const alertSize = 100;
          ctx.strokeRect(alertX - alertSize / 2, alertY - alertSize / 2, alertSize, alertSize);
        }
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

  return (
    <div className="relative w-full h-full bg-black flex items-center justify-center">
      {alertActive && (
        <div className="absolute top-0 left-0 right-0 z-20 bg-red-600 text-white py-2 px-4 flex items-center gap-2 animate-pulse">
          <AlertTriangle className="w-5 h-5" />
          <span className="font-semibold">ANOMALY DETECTED - UNAUTHORIZED ACTIVITY</span>
        </div>
      )}

      {videoFile ? (
        <>
          <video
            ref={videoRef}
            className={`max-w-full max-h-full ${isMonitoring ? 'hidden' : 'block'}`}
            controls={!isMonitoring}
            onTimeUpdate={handleTimeUpdate}
            onLoadedMetadata={handleLoadedMetadata}
            loop
            autoPlay
            muted
            playsInline
          />
          <canvas
            ref={canvasRef}
            className={`max-w-full max-h-full ${isMonitoring ? 'block' : 'hidden'}`}
          />

          {isMonitoring && (
            <>
              {/* Recording indicator */}
              <div className="absolute top-4 left-4 flex items-center gap-2 bg-red-600 text-white px-3 py-1.5 rounded">
                <Activity className="w-4 h-4 animate-pulse" />
                <span className="text-sm font-semibold">ANALYZING RECORDING</span>
              </div>

              {/* Timecode */}
              <div className="absolute bottom-4 left-4 bg-black/80 text-lime-400 px-3 py-1.5 rounded font-mono text-sm">
                {formatTime(currentTime)} / {formatTime(duration)}
              </div>
            </>
          )}
        </>
      ) : (
        <div className="text-gray-500 text-center p-8">
          <Activity className="w-16 h-16 mx-auto mb-4 opacity-30" />
          <p className="text-lg">No feed source active</p>
          <p className="text-sm mt-2">
            Upload a recording to begin monitoring
          </p>
        </div>
      )}
    </div>
  );
}