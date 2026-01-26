import { Camera, MapPin, Wrench, Calendar, Activity } from 'lucide-react';
import { Badge } from '@/app/components/ui/badge';

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
    <div className="bg-zinc-950 border border-yellow-900/30 rounded-lg p-5">
      <div className="flex items-center justify-between mb-5">
        <h3 className="text-base font-bold text-yellow-400 uppercase tracking-wider">
          Camera Asset
        </h3>
        <Badge className={`${getStatusColor()} border flex items-center gap-2 px-3 py-1.5 text-sm`}>
          <span className={`w-2.5 h-2.5 rounded-full ${getStatusDot()} animate-pulse`}></span>
          {status.toUpperCase()}
        </Badge>
      </div>

      <div className="space-y-4">
        {/* Camera ID */}
        <div className="bg-zinc-900 border border-yellow-900/30 rounded-lg p-4">
          <div className="flex items-center gap-2 mb-3">
            <Camera className="w-5 h-5 text-yellow-400" />
            <span className="text-sm font-medium text-yellow-600">Camera ID</span>
          </div>
          <div className="text-xl font-bold text-white font-mono">{cameraId}</div>
        </div>

        {/* Location Hierarchy */}
        <div className="bg-zinc-900 border border-yellow-900/30 rounded-lg p-4">
          <div className="flex items-center gap-2 mb-3">
            <MapPin className="w-5 h-5 text-lime-400" />
            <span className="text-sm font-medium text-yellow-600">Location</span>
          </div>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-gray-500">Building:</span>
              <span className="text-white font-semibold">{building}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-500">Floor:</span>
              <span className="text-white font-semibold">{floor}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-500">Zone:</span>
              <span className="text-white font-semibold">{zone}</span>
            </div>
          </div>
        </div>

        {/* Maintenance Schedule */}
        <div className="bg-zinc-900 border border-yellow-900/30 rounded-lg p-4">
          <div className="flex items-center gap-2 mb-3">
            <Wrench className="w-5 h-5 text-amber-400" />
            <span className="text-sm font-medium text-yellow-600">Maintenance</span>
          </div>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-gray-500">Last Service:</span>
              <span className="text-white font-medium">{formatDate(lastMaintenance)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-500">Next Service:</span>
              <span className="text-amber-400 font-medium">{formatDate(nextMaintenance)}</span>
            </div>
          </div>
        </div>

        {/* Technical Specs */}
        <div className="bg-zinc-900 border border-yellow-900/30 rounded-lg p-4">
          <div className="flex items-center gap-2 mb-3">
            <Activity className="w-5 h-5 text-lime-400" />
            <span className="text-sm font-medium text-yellow-600">Specifications</span>
          </div>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-gray-500">Resolution:</span>
              <span className="text-white font-medium">{resolution}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-500">Frame Rate:</span>
              <span className="text-white font-medium">{fps} FPS</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-500">Uptime:</span>
              <span className="text-lime-400 font-medium">{uptime}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}