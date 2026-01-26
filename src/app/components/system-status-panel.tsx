import { Activity, Camera, Shield, AlertCircle } from 'lucide-react';
import { Badge } from './ui/badge';

interface SystemStatusPanelProps {
  systemState: 'idle' | 'monitoring' | 'alert' | 'maintenance';
  cameraHealth: 'online' | 'offline' | 'degraded';
  threatLevel: 'low' | 'medium' | 'high';
  anomalyConfidence: number | null;
  totalCameras: number;
  onlineCameras: number;
}

export function SystemStatusPanel({
  systemState,
  cameraHealth,
  threatLevel,
  anomalyConfidence,
  totalCameras,
  onlineCameras
}: SystemStatusPanelProps) {
  const getSystemStateColor = () => {
    switch (systemState) {
      case 'monitoring': return 'bg-cyan-600 shadow-[0_0_15px_rgba(8,145,178,0.4)]';
      case 'alert': return 'bg-red-600 shadow-[0_0_15px_rgba(220,38,38,0.4)]';
      case 'maintenance': return 'bg-blue-800';
      default: return 'bg-slate-700';
    }
  };

  const getSystemStateText = () => {
    switch (systemState) {
      case 'monitoring': return 'MONITORING ACTIVE';
      case 'alert': return 'ALERT CONDITION';
      case 'maintenance': return 'MAINTENANCE REQUIRED';
      default: return 'SYSTEM IDLE';
    }
  };

  const getThreatLevelColor = () => {
    switch (threatLevel) {
      case 'high': return 'text-red-400 bg-red-950/50 border border-red-900/50';
      case 'medium': return 'text-orange-400 bg-orange-950/50 border border-orange-900/50';
      default: return 'text-cyan-400 bg-cyan-950/50 border border-cyan-900/50';
    }
  };

  const getCameraHealthColor = () => {
    switch (cameraHealth) {
      case 'online': return 'text-cyan-400';
      case 'degraded': return 'text-orange-500';
      default: return 'text-red-500';
    }
  };

  return (
    <div className="bg-slate-950/50 border border-cyan-900/30 rounded-lg p-5 space-y-5 backdrop-blur-md">
      <div className="flex items-center justify-between">
        <h3 className="text-base font-bold text-cyan-400 uppercase tracking-wider">
          System Status
        </h3>
        <div className={`${getSystemStateColor()} px-3 py-1.5 rounded text-white text-sm font-bold`}>
          {getSystemStateText()}
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4">
        {/* Camera Health */}
        <div className="bg-slate-900/40 border border-cyan-900/20 rounded-lg p-4 transition-all hover:border-cyan-500/30">
          <div className="flex items-center gap-2 mb-3">
            <Camera className={`w-5 h-5 ${getCameraHealthColor()}`} />
            <span className="text-sm font-medium text-slate-400">Camera Status</span>
          </div>
          <div className={`text-2xl font-bold ${getCameraHealthColor()} mb-2`}>
            {cameraHealth.toUpperCase()}
          </div>
          <div className="text-sm text-gray-400">
            {onlineCameras} / {totalCameras} Online
          </div>
        </div>

        {/* Threat Level */}
        <div className="bg-slate-900/40 border border-cyan-900/20 rounded-lg p-4 transition-all hover:border-cyan-500/30">
          <div className="flex items-center gap-2 mb-3">
            <Shield className="w-5 h-5 text-cyan-500" />
            <span className="text-sm font-medium text-slate-400">Threat Level</span>
          </div>
          <Badge className={`${getThreatLevelColor()} border-0 text-lg font-bold px-3 py-1`}>
            {threatLevel.toUpperCase()}
          </Badge>
        </div>

        {/* System Activity */}
        <div className="bg-slate-900/40 border border-cyan-900/20 rounded-lg p-4 transition-all hover:border-cyan-500/30">
          <div className="flex items-center gap-2 mb-3">
            <Activity className="w-5 h-5 text-cyan-400" />
            <span className="text-sm font-medium text-slate-400">Activity</span>
          </div>
          <div className="text-2xl font-bold text-cyan-400">
            {systemState === 'monitoring' ? 'ACTIVE' : 'STANDBY'}
          </div>
        </div>

        {/* AI Confidence */}
        <div className="bg-slate-900/40 border border-cyan-900/20 rounded-lg p-4 transition-all hover:border-cyan-500/30">
          <div className="flex items-center gap-2 mb-3">
            <AlertCircle className="w-5 h-5 text-cyan-400" />
            <span className="text-sm font-medium text-slate-400">AI Confidence</span>
          </div>
          <div className="text-2xl font-bold text-cyan-400">
            {anomalyConfidence ? `${(anomalyConfidence * 100).toFixed(1)}%` : '--'}
          </div>
        </div>
      </div>

      {/* AI Detection Info */}
      <div className="bg-cyan-950/20 border border-cyan-800/40 rounded-lg p-4">
        <p className="text-sm leading-relaxed text-slate-300">
          <strong className="text-cyan-400">AI Anomaly Detection:</strong> Behavior analysis system. No facial recognition.
        </p>
      </div>
    </div>
  );
}