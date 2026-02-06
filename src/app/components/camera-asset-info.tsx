import { Camera, MapPin, Wrench, Calendar, Activity } from 'lucide-react';
import { Badge } from './ui/badge';

interface CameraAssetInfoProps {
  cameraId: string;
  location: string;
  building: string;
  floor: string;
  zone: string;
  status: 'online' | 'offline' | 'maintenance';
  lastMaintenance: string;
  nextMaintenance: string;
  uptime: string;
  resolution: string;
  fps: number;
}

export function CameraAssetInfo({
  cameraId,
  location,
  building,
  floor,
  zone,
  status,
  lastMaintenance,
  nextMaintenance,
  uptime,
  resolution,
  fps
}: CameraAssetInfoProps) {
  const getStatusColor = () => {
    switch (status) {
      case 'online': return 'bg-green-900 text-green-300 border-green-700';
      case 'offline': return 'bg-red-900 text-red-300 border-red-700';
      case 'maintenance': return 'bg-yellow-900 text-yellow-300 border-yellow-700';
    }
  };

  const getStatusDot = () => {
    switch (status) {
      case 'online': return 'bg-green-500';
      case 'offline': return 'bg-red-500';
      case 'maintenance': return 'bg-yellow-500';
    }
  };

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
  };

  return (
    <div className="bg-zinc-900/60 border border-zinc-800/50 rounded-xl p-5 backdrop-blur-md shadow-xl">
      <div className="flex items-center justify-between mb-5">
        <h3 className="text-sm font-bold text-zinc-400 uppercase tracking-widest">
          Camera Asset
        </h3>
        <Badge className={`${getStatusColor()} border-0 flex items-center gap-2 px-2.5 py-1 text-xs font-bold tracking-wide`}>
          <span className={`w-2 h-2 rounded-full ${getStatusDot()} animate-pulse shadow-lg`}></span>
          {status.toUpperCase()}
        </Badge>
      </div>

      <div className="space-y-3">
        {/* Camera ID */}
        <div className="bg-black/40 border border-zinc-800/50 rounded-lg p-3 group hover:border-zinc-700/50 transition-colors">
          <div className="flex items-center gap-2 mb-2">
            <Camera className="w-4 h-4 text-zinc-500 group-hover:text-yellow-500 transition-colors" />
            <span className="text-xs font-medium text-zinc-500">Camera ID</span>
          </div>
          <div className="text-lg font-bold text-white font-mono tracking-tight">{cameraId}</div>
        </div>

        {/* Location Hierarchy */}
        <div className="bg-black/40 border border-zinc-800/50 rounded-lg p-3 group hover:border-zinc-700/50 transition-colors">
          <div className="flex items-center gap-2 mb-3">
            <MapPin className="w-4 h-4 text-zinc-500 group-hover:text-emerald-500 transition-colors" />
            <span className="text-xs font-medium text-zinc-500">Location</span>
          </div>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-zinc-600 text-xs">Building</span>
              <span className="text-zinc-300 font-medium text-xs">{building}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-zinc-600 text-xs">Floor</span>
              <span className="text-zinc-300 font-medium text-xs">{floor}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-zinc-600 text-xs">Zone</span>
              <span className="text-zinc-300 font-medium text-xs">{zone}</span>
            </div>
          </div>
        </div>

        {/* Maintenance Schedule */}
        <div className="bg-black/40 border border-zinc-800/50 rounded-lg p-3 group hover:border-zinc-700/50 transition-colors">
          <div className="flex items-center gap-2 mb-3">
            <Wrench className="w-4 h-4 text-zinc-500 group-hover:text-amber-500 transition-colors" />
            <span className="text-xs font-medium text-zinc-500">Maintenance</span>
          </div>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-zinc-600 text-xs">Last Service</span>
              <span className="text-zinc-300 font-medium text-xs font-mono">{formatDate(lastMaintenance)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-zinc-600 text-xs">Next Service</span>
              <span className="text-amber-500 font-medium text-xs font-mono">{formatDate(nextMaintenance)}</span>
            </div>
          </div>
        </div>

        {/* Technical Specs */}
        <div className="bg-black/40 border border-zinc-800/50 rounded-lg p-3 group hover:border-zinc-700/50 transition-colors">
          <div className="flex items-center gap-2 mb-3">
            <Activity className="w-4 h-4 text-zinc-500 group-hover:text-emerald-500 transition-colors" />
            <span className="text-xs font-medium text-zinc-500">Specifications</span>
          </div>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-zinc-600 text-xs">Resolution</span>
              <span className="text-zinc-300 font-medium text-xs font-mono">{resolution}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-zinc-600 text-xs">Frame Rate</span>
              <span className="text-zinc-300 font-medium text-xs font-mono">{fps} FPS</span>
            </div>
            <div className="flex justify-between">
              <span className="text-zinc-600 text-xs">Uptime</span>
              <span className="text-emerald-500 font-medium text-xs font-mono">{uptime}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}