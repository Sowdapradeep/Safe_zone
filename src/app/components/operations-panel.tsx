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
    <div className="bg-zinc-900/60 border border-zinc-800/50 rounded-xl backdrop-blur-md overflow-hidden shadow-xl">
      <div className="p-5 border-b border-zinc-800/50 bg-black/20">
        <h3 className="text-sm font-black text-zinc-400 uppercase tracking-widest flex items-center gap-3">
          <div className="w-1.5 h-4 bg-yellow-500 rounded-full shadow-[0_0_10px_rgba(234,179,8,0.5)]" />
          Operations Control
        </h3>
      </div>

      <div className="p-5 space-y-5">
        {/* Toggle Section */}
        <div className="bg-black/40 border border-zinc-800/50 rounded-lg p-4 flex items-center justify-between group hover:border-zinc-700/50 transition-colors">
          <div className="space-y-1">
            <Label className="text-xs font-bold text-zinc-300 uppercase tracking-wider group-hover:text-yellow-500 transition-colors">Live Camera Mode</Label>
            <p className="text-[10px] text-zinc-500 font-mono">Stream directly from CAM-001-NE</p>
          </div>
          <Switch
            checked={isLiveMode}
            onCheckedChange={onToggleLiveMode}
            disabled={isMonitoring}
            className="data-[state=checked]:bg-yellow-500"
          />
        </div>

        {/* Source Section */}
        <div className="space-y-3">
          <Label className="text-xs font-bold text-zinc-500 uppercase tracking-widest">Surveillance Source</Label>
          <div className="grid grid-cols-1 gap-4">
            <Button
              variant="outline"
              disabled={isMonitoring || isLiveMode || isProcessing}
              className={`h-24 border-dashed border-2 flex flex-col gap-2 transition-all duration-300 rounded-xl ${isLiveMode
                ? 'opacity-40 bg-zinc-900/50 border-zinc-800'
                : 'bg-zinc-900/50 border-zinc-800 hover:border-yellow-500/50 hover:bg-yellow-500/5'
                }`}
              onClick={() => document.getElementById('video-upload')?.click()}
            >
              <Upload className={`w-6 h-6 ${isLiveMode || isMonitoring ? 'text-zinc-600' : 'text-yellow-500 group-hover:scale-110 transition-transform'}`} />
              <div className="text-center">
                <span className={`block text-xs font-bold ${isLiveMode ? 'text-zinc-600' : 'text-zinc-300'}`}>
                  {isLiveMode ? 'Recording Disabled' : uploadedFileName || 'Load Stream Recording'}
                </span>
                <span className="text-[10px] text-zinc-600 uppercase tracking-widest mt-1">MP4 / MKV / AVI</span>
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
        <div className="pt-1">
          {!isMonitoring ? (
            <Button
              className="w-full h-14 bg-gradient-to-r from-yellow-600 to-yellow-500 hover:from-yellow-500 hover:to-yellow-400 text-black font-black text-sm uppercase tracking-widest shadow-lg shadow-yellow-500/20 rounded-xl transition-all hover:scale-[1.02] active:scale-[0.98]"
              onClick={onStartMonitoring}
              disabled={isProcessing || (!isLiveMode && !hasVideo)}
            >
              {isProcessing ? (
                <>
                  <RotateCcw className="w-5 h-5 mr-3 animate-spin" />
                  Processing Stream...
                </>
              ) : (
                <>
                  <Play className="w-5 h-5 mr-3 fill-current" />
                  {isLiveMode ? 'Start Live Feed' : 'Initiate Monitoring'}
                </>
              )}
            </Button>
          ) : (
            <Button
              className="w-full h-14 bg-gradient-to-r from-red-600 to-red-500 hover:from-red-500 hover:to-red-400 text-white font-black text-sm uppercase tracking-widest shadow-lg shadow-red-500/20 rounded-xl transition-all hover:scale-[1.02] active:scale-[0.98]"
              onClick={onStopMonitoring}
            >
              <Square className="w-5 h-5 mr-3 fill-current" />
              Cease Surveillance
            </Button>
          )}
        </div>

        {/* ROI Toggle */}
        <div className="flex items-center justify-between bg-black/40 border border-zinc-800/50 rounded-lg p-4 group hover:border-zinc-700/50 transition-colors">
          <div className="flex flex-col gap-1">
            <Label htmlFor="roi-toggle" className="text-xs font-bold text-zinc-300 cursor-pointer group-hover:text-yellow-500 transition-colors">
              Restricted Zone Overlay
            </Label>
            <span className="text-[10px] text-zinc-600 font-bold uppercase tracking-wider">Detection Boundary</span>
          </div>
          <Switch
            id="roi-toggle"
            checked={showROI}
            onCheckedChange={onToggleROI}
            disabled={!isMonitoring}
            className="data-[state=checked]:bg-yellow-500"
          />
        </div>

        {/* Manual Scan */}
        <Button
          onClick={onScanIncidents}
          disabled={!isMonitoring}
          variant="outline"
          className="w-full border-zinc-800 bg-zinc-900/50 hover:bg-zinc-800 text-zinc-400 hover:text-white h-10 text-xs font-bold uppercase tracking-widest transition-colors"
        >
          <Search className="w-4 h-4 mr-2" />
          Manual Incident Scan
        </Button>
      </div>

      {/* Footer Status */}
      <div className="bg-black/20 p-4 flex items-center justify-between border-t border-zinc-800/50">
        <div className="flex items-center gap-2">
          <div className={`w-1.5 h-1.5 rounded-full ${isMonitoring ? 'bg-emerald-500 animate-pulse' : 'bg-zinc-700'}`} />
          <span className="text-[10px] font-bold text-zinc-500 uppercase tracking-widest">System Status</span>
        </div>
        <span className={`text-[10px] font-bold uppercase tracking-widest ${isMonitoring ? 'text-emerald-500' : 'text-zinc-600'}`}>
          {isMonitoring ? 'Operational' : 'Standby'}
        </span>
      </div>
    </div>
  );
}
