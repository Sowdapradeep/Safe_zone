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
    <div className="bg-zinc-900/60 border border-zinc-800/50 rounded-xl p-5 space-y-5 backdrop-blur-md shadow-xl">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-bold text-zinc-400 uppercase tracking-widest">
          System Status
        </h3>
        <div className={`px-3 py-1.5 rounded-full text-white text-[10px] font-bold tracking-widest uppercase shadow-lg ${systemState === 'monitoring' ? 'bg-emerald-500 shadow-emerald-500/20' :
            systemState === 'alert' ? 'bg-red-500 shadow-red-500/20 animate-pulse' :
              'bg-zinc-700'
          }`}>
          {getSystemStateText()}
        </div>
      </div>

      <div className="grid grid-cols-2 gap-3">
        {/* Camera Health */}
        <div className="bg-black/40 border border-zinc-800/50 rounded-lg p-3 transition-all hover:border-emerald-500/30 group">
          <div className="flex items-center gap-2 mb-2">
            <Camera className={`w-4 h-4 ${getCameraHealthColor()} group-hover:scale-110 transition-transform`} />
            <span className="text-xs font-medium text-zinc-500">Camera Status</span>
          </div>
          <div className={`text-xl font-bold ${getCameraHealthColor()} mb-1`}>
            {cameraHealth.toUpperCase()}
          </div>
          <div className="text-[10px] text-zinc-600 font-mono">
            {onlineCameras} / {totalCameras} Online
          </div>
        </div>

        {/* Threat Level */}
        <div className="bg-black/40 border border-zinc-800/50 rounded-lg p-3 transition-all hover:border-red-500/30 group">
          <div className="flex items-center gap-2 mb-2">
            <Shield className="w-4 h-4 text-zinc-400 group-hover:scale-110 transition-transform" />
            <span className="text-xs font-medium text-zinc-500">Threat Level</span>
          </div>
          <Badge className={`${getThreatLevelColor()} border-0 text-sm font-bold px-2 py-0.5`}>
            {threatLevel.toUpperCase()}
          </Badge>
        </div>

        {/* System Activity */}
        <div className="bg-black/40 border border-zinc-800/50 rounded-lg p-3 transition-all hover:border-cyan-500/30 group">
          <div className="flex items-center gap-2 mb-2">
            <Activity className="w-4 h-4 text-zinc-400 group-hover:scale-110 transition-transform" />
            <span className="text-xs font-medium text-zinc-500">Activity</span>
          </div>
          <div className="text-lg font-bold text-zinc-200">
            {systemState === 'monitoring' ? 'ACTIVE' : 'STANDBY'}
          </div>
        </div>

        {/* AI Confidence */}
        <div className="bg-black/40 border border-zinc-800/50 rounded-lg p-3 transition-all hover:border-violet-500/30 group">
          <div className="flex items-center gap-2 mb-2">
            <AlertCircle className="w-4 h-4 text-zinc-400 group-hover:scale-110 transition-transform" />
            <span className="text-xs font-medium text-zinc-500">AI Confidence</span>
          </div>
          <div className="text-xl font-bold text-white font-mono">
            {anomalyConfidence ? `${(anomalyConfidence * 100).toFixed(1)}%` : '--'}
          </div>
        </div>
      </div>

      {/* AI Detection Info */}
      <div className="bg-emerald-950/20 border border-emerald-900/30 rounded-lg p-3">
        <p className="text-xs leading-relaxed text-zinc-400">
          <strong className="text-emerald-500">AI Active:</strong> Behavior analysis system online.
        </p>
      </div>
    </div>
  );
}