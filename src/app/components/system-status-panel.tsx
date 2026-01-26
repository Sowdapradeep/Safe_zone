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
      case 'monitoring': return 'bg-lime-600';
      case 'alert': return 'bg-red-600';
      case 'maintenance': return 'bg-yellow-600';
      default: return 'bg-gray-600';
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
      case 'high': return 'text-red-500 bg-red-950';
      case 'medium': return 'text-yellow-500 bg-yellow-950';
      default: return 'text-green-500 bg-green-950';
    }
  };

  const getCameraHealthColor = () => {
    switch (cameraHealth) {
      case 'online': return 'text-lime-400';
      case 'degraded': return 'text-yellow-500';
      default: return 'text-red-500';
    }
  };

  return (
    <div className="bg-zinc-950 border border-yellow-900/30 rounded-lg p-5 space-y-5">
      <div className="flex items-center justify-between">
        <h3 className="text-base font-bold text-yellow-400 uppercase tracking-wider">
          System Status
        </h3>
        <div className={`${getSystemStateColor()} px-3 py-1.5 rounded text-white text-sm font-bold`}>
          {getSystemStateText()}
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4">
        {/* Camera Health */}
        <div className="bg-zinc-900 border border-yellow-900/30 rounded-lg p-4">
          <div className="flex items-center gap-2 mb-3">
            <Camera className={`w-5 h-5 ${getCameraHealthColor()}`} />
            <span className="text-sm font-medium text-yellow-600">Camera Status</span>
          </div>
          <div className={`text-2xl font-bold ${getCameraHealthColor()} mb-2`}>
            {cameraHealth.toUpperCase()}
          </div>
          <div className="text-sm text-gray-400">
            {onlineCameras} / {totalCameras} Online
          </div>
        </div>

        {/* Threat Level */}
        <div className="bg-zinc-900 border border-yellow-900/30 rounded-lg p-4">
          <div className="flex items-center gap-2 mb-3">
            <Shield className="w-5 h-5 text-yellow-600" />
            <span className="text-sm font-medium text-yellow-600">Threat Level</span>
          </div>
          <Badge className={`${getThreatLevelColor()} border-0 text-lg font-bold px-3 py-1`}>
            {threatLevel.toUpperCase()}
          </Badge>
        </div>

        {/* System Activity */}
        <div className="bg-zinc-900 border border-yellow-900/30 rounded-lg p-4">
          <div className="flex items-center gap-2 mb-3">
            <Activity className="w-5 h-5 text-lime-400" />
            <span className="text-sm font-medium text-yellow-600">Activity</span>
          </div>
          <div className="text-2xl font-bold text-lime-400">
            {systemState === 'monitoring' ? 'ACTIVE' : 'STANDBY'}
          </div>
        </div>

        {/* AI Confidence */}
        <div className="bg-zinc-900 border border-yellow-900/30 rounded-lg p-4">
          <div className="flex items-center gap-2 mb-3">
            <AlertCircle className="w-5 h-5 text-yellow-400" />
            <span className="text-sm font-medium text-yellow-600">AI Confidence</span>
          </div>
          <div className="text-2xl font-bold text-yellow-400">
            {anomalyConfidence ? `${(anomalyConfidence * 100).toFixed(1)}%` : '--'}
          </div>
        </div>
      </div>

      {/* AI Detection Info */}
      <div className="bg-yellow-950/20 border border-yellow-800/40 rounded-lg p-4">
        <p className="text-sm leading-relaxed text-yellow-200">
          <strong className="text-yellow-300">AI Anomaly Detection:</strong> Behavior analysis system. No facial recognition.
        </p>
      </div>
    </div>
  );
}