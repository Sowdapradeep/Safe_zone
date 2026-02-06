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

      // 4. Simulated anomaly detection removed by user request
      /*
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
      */

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
    <div className="relative w-full h-full bg-black flex items-center justify-center overflow-hidden">
      {/* Alert Overlay */}
      {alertActive && (
        <div className="absolute inset-0 z-50 pointer-events-none flex items-center justify-center">
          <div className="absolute inset-0 bg-red-600/10 animate-pulse border-[12px] border-red-600/30" />
          <div className="bg-red-600/90 backdrop-blur-md text-white px-10 py-5 rounded-2xl flex items-center gap-6 text-3xl font-black uppercase tracking-tighter shadow-[0_0_50px_rgba(220,38,38,0.5)] border border-white/20">
            <AlertTriangle className="w-12 h-12 animate-bounce" />
            <div className="flex flex-col items-center">
              <span>Security Breach</span>
              <span className="text-xs font-mono tracking-widest opacity-80 mt-1">Unauthorized Zone Entry</span>
            </div>
            <AlertTriangle className="w-12 h-12 animate-bounce" />
          </div>
        </div>
      )}

      {/* Global CRT Overlay */}
      <div className="absolute inset-0 pointer-events-none z-30 bg-[linear-gradient(rgba(18,16,16,0)_50%,rgba(0,0,0,0.1)_50%),linear-gradient(90deg,rgba(255,0,0,0.03),rgba(0,255,0,0.01),rgba(0,0,255,0.03))] bg-[length:100%_3px,4px_100%] opacity-40 mix-blend-overlay" />
      <div className="absolute inset-0 pointer-events-none z-30 shadow-[inset_0_0_150px_rgba(0,0,0,0.8)]" />

      {isLiveMode && isMonitoring ? (
        <div className="relative w-full h-full">
          <img
            src={import.meta.env.VITE_LIVE_FEED_URL || `${import.meta.env.VITE_API_URL || 'http://localhost:8000'}/api/live-feed`}
            className="w-full h-full object-contain grayscale-[0.2] contrast-125 brightness-110"
            alt="Live Surveillance Feed"
          />
          <div className="absolute top-6 left-6 z-40 flex items-center gap-4">
            <div className="bg-red-600 text-white px-4 py-1.5 rounded-lg flex items-center gap-2.5 font-black text-sm uppercase tracking-widest shadow-lg shadow-red-600/30 animate-pulse">
              <div className="w-2.5 h-2.5 bg-white rounded-full" />
              REC
            </div>
            <div className="bg-black/60 backdrop-blur-md text-zinc-300 px-4 py-1.5 border border-white/10 rounded-lg font-mono text-sm shadow-xl tracking-tight">
              SOURCE: <span className="text-white font-bold">CAM-001-NE</span>
            </div>
          </div>

          <div className="absolute bottom-6 right-6 z-40 bg-black/60 backdrop-blur-md text-emerald-500 px-5 py-2 border border-white/10 rounded-lg font-mono text-lg font-black shadow-2xl tracking-widest">
            {new Date().toLocaleTimeString('en-US', { hour12: false })}
          </div>
        </div>
      ) : videoFile ? (
        <>
          <video
            ref={videoRef}
            className={`max-w-full max-h-full grayscale-[0.2] contrast-110 brightness-110 ${isMonitoring ? 'hidden' : 'block'}`}
            controls={!isMonitoring}
            onTimeUpdate={handleTimeUpdate}
            onLoadedMetadata={handleLoadedMetadata}
            loop
          />
          <canvas
            ref={canvasRef}
            className={`max-w-full max-h-full grayscale-[0.2] contrast-110 brightness-110 ${isMonitoring ? 'block' : 'hidden'}`}
          />

          {/* Monitoring Indicators */}
          {isMonitoring && (
            <>
              <div className="absolute top-6 left-6 z-40 flex items-center gap-4">
                <div className="bg-yellow-500 text-black px-4 py-1.5 rounded-lg flex items-center gap-2.5 font-black text-sm uppercase tracking-widest shadow-lg shadow-yellow-500/30 transition-all">
                  <Activity className="w-4 h-4 animate-spin" />
                  AI Analysis Active
                </div>
                <div className="bg-black/60 backdrop-blur-md text-zinc-300 px-4 py-1.5 border border-white/10 rounded-lg font-mono text-sm shadow-xl">
                  {videoFile.name}
                </div>
              </div>

              <div className="absolute bottom-6 right-6 z-40 flex flex-col items-end gap-3">
                <div className="bg-black/70 backdrop-blur-md text-zinc-200 px-6 py-3 border border-white/10 rounded-xl font-mono text-xl font-black shadow-2xl tracking-tighter">
                  TC: <span className="text-yellow-500">{timeCode}</span>
                </div>
                {showROI && (
                  <div className="bg-black/50 backdrop-blur-md text-zinc-500 px-3 py-1 border border-white/5 rounded-lg text-[10px] uppercase font-black tracking-widest">
                    Detection Mesh Active
                  </div>
                )}
              </div>
            </>
          )}
        </>
      ) : (
        <div className="text-center text-zinc-800 p-12">
          <Activity className="w-24 h-24 mx-auto mb-6 opacity-5" />
          <p className="text-xl font-black uppercase tracking-widest opacity-20">
            {isLiveMode ? 'Waiting for Feed' : 'No Data Stream'}
          </p>
          <p className="text-xs mt-3 font-mono uppercase tracking-[0.3em] opacity-30">Connect source to initialize</p>
        </div>
      )}
    </div>
  );
}