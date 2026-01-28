import { useState } from 'react';
import { Upload, Play, Square, Search, RotateCcw } from 'lucide-react';
import { Button } from './ui/button';
import { Label } from './ui/label';
import { Switch } from './ui/switch';

interface OperationsPanelProps {
  onVideoUpload: (file: File) => void;
  onStartMonitoring: () => void;
  onStopMonitoring: () => void;
  onScanIncidents: () => void;
  isMonitoring: boolean;
  hasVideo: boolean;
  showROI: boolean;
  onToggleROI: (show: boolean) => void;
  isProcessing: boolean;
  isLiveMode: boolean;
  onToggleLiveMode: () => void;
}

export function OperationsPanel({
  onVideoUpload,
  onStartMonitoring,
  onStopMonitoring,
  onScanIncidents,
  isMonitoring,
  hasVideo,
  showROI,
  onToggleROI,
  isProcessing,
  isLiveMode,
  onToggleLiveMode
}: OperationsPanelProps) {
  const [uploadedFileName, setUploadedFileName] = useState<string>('');

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setUploadedFileName(file.name);
      onVideoUpload(file);
    }
  };

  return (
    <div className="bg-zinc-950/50 border border-yellow-900/30 rounded-lg backdrop-blur-md overflow-hidden">
      <div className="p-6 border-b border-yellow-900/30 bg-zinc-900/50">
        <h3 className="text-base font-black text-yellow-400 uppercase tracking-widest flex items-center gap-3">
          <div className="w-2 h-6 bg-yellow-600 rounded-full" />
          Operations Control
        </h3>
      </div>

      <div className="p-6 space-y-6">
        {/* Toggle Section */}
        <div className="bg-zinc-900/50 border border-yellow-900/10 rounded-lg p-5 flex items-center justify-between">
          <div className="space-y-1">
            <Label className="text-sm font-bold text-yellow-500 uppercase tracking-wider">Live Camera Mode</Label>
            <p className="text-xs text-yellow-700">Stream directly from CAM-001-NE</p>
          </div>
          <Switch
            checked={isLiveMode}
            onCheckedChange={onToggleLiveMode}
            disabled={isMonitoring}
            className="data-[state=checked]:bg-yellow-600"
          />
        </div>

        {/* Source Section */}
        <div className="space-y-4">
          <Label className="text-sm font-bold text-yellow-600 uppercase tracking-widest">Surveillance Source</Label>
          <div className="grid grid-cols-1 gap-4">
            <Button
              variant="outline"
              disabled={isMonitoring || isLiveMode || isProcessing}
              className={`h-24 border-dashed border-2 flex flex-col gap-2 transition-all duration-300 ${isLiveMode ? 'opacity-40' : 'hover:border-yellow-500 hover:bg-yellow-900/10'}`}
              onClick={() => document.getElementById('video-upload')?.click()}
            >
              <Upload className={`w-8 h-8 ${isLiveMode || isMonitoring ? 'text-zinc-700' : 'text-yellow-600 animate-bounce'}`} />
              <div className="text-center">
                <span className="block text-sm font-bold text-yellow-400">
                  {isLiveMode ? 'Recording Disabled' : uploadedFileName || 'Load Stream Recording'}
                </span>
                <span className="text-[10px] text-yellow-800 uppercase tracking-widest">MP4 / MKV / AVI</span>
              </div>
            </Button>
            <input
              id="video-upload"
              type="file"
              accept="video/*"
              className="hidden"
              onChange={handleFileChange}
            />
          </div>
        </div>

        {/* Action Button */}
        <div className="pt-2">
          {!isMonitoring ? (
            <Button
              className="w-full h-16 bg-yellow-600 hover:bg-yellow-500 text-black font-black text-lg uppercase tracking-widest shadow-lg shadow-yellow-600/20"
              onClick={onStartMonitoring}
              disabled={isProcessing || (!isLiveMode && !hasVideo)}
            >
              {isProcessing ? (
                <>
                  <RotateCcw className="w-6 h-6 mr-3 animate-spin" />
                  Processing Stream...
                </>
              ) : (
                <>
                  <Play className="w-6 h-6 mr-3 fill-current" />
                  {isLiveMode ? 'Start Live Feed' : 'Initiate Monitoring'}
                </>
              )}
            </Button>
          ) : (
            <Button
              className="w-full h-16 bg-red-600 hover:bg-red-500 text-white font-black text-lg uppercase tracking-widest shadow-lg shadow-red-600/20"
              onClick={onStopMonitoring}
            >
              <Square className="w-6 h-6 mr-3 fill-current" />
              Cease Surveillance
            </Button>
          )}
        </div>

        {/* ROI Toggle */}
        <div className="flex items-center justify-between bg-zinc-900/40 border border-yellow-900/20 rounded-lg p-5">
          <div className="flex flex-col gap-1">
            <Label htmlFor="roi-toggle" className="text-sm font-bold text-yellow-200 cursor-pointer">
              Restricted Zone Overlay
            </Label>
            <span className="text-[10px] text-yellow-900 font-bold uppercase tracking-wider">Detection Boundary Visualization</span>
          </div>
          <Switch
            id="roi-toggle"
            checked={showROI}
            onCheckedChange={onToggleROI}
            disabled={!isMonitoring}
          />
        </div>

        {/* Manual Scan */}
        <Button
          onClick={onScanIncidents}
          disabled={!isMonitoring}
          variant="outline"
          className="w-full border-yellow-900/30 hover:bg-yellow-900/10 text-yellow-600 h-12 text-sm font-bold uppercase tracking-widest"
        >
          <Search className="w-5 h-5 mr-3" />
          Manual Incident Scan
        </Button>
      </div>

      {/* Footer Status */}
      <div className="bg-zinc-900 p-4 flex items-center justify-between border-t border-yellow-900/30">
        <div className="flex items-center gap-2">
          <div className={`w-2 h-2 rounded-full ${isMonitoring ? 'bg-lime-500 animate-pulse' : 'bg-yellow-900'}`} />
          <span className="text-[10px] font-black text-yellow-800 uppercase tracking-[0.2em]">System Status</span>
        </div>
        <span className={`text-[10px] font-black uppercase tracking-[0.2em] ${isMonitoring ? 'text-lime-400' : 'text-yellow-900'}`}>
          {isMonitoring ? 'Operational' : 'Standby'}
        </span>
      </div>
    </div>
  );
}
