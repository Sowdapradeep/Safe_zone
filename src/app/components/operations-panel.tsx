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
  isProcessing
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
    <div className="bg-slate-950/50 border border-cyan-900/30 rounded-lg p-5 space-y-5 backdrop-blur-md">
      <h3 className="text-base font-bold text-cyan-400 uppercase tracking-wider border-b border-cyan-900/30 pb-3">
        Operations Control
      </h3>

      {/* Video Upload */}
      <div className="space-y-3">
        <Label className="text-sm font-medium text-slate-500">Feed Source</Label>
        <div className="relative">
          <input
            type="file"
            accept="video/*"
            onChange={handleFileChange}
            className="hidden"
            id="video-upload"
            disabled={isMonitoring || isProcessing}
          />
          <label htmlFor="video-upload">
            <Button
              type="button"
              variant="outline"
              className="w-full justify-start bg-slate-900/50 border-cyan-900/30 hover:bg-slate-800 hover:border-cyan-500 text-cyan-100 h-12 text-base"
              disabled={isMonitoring || isProcessing}
              asChild
            >
              <span>
                <Upload className="w-5 h-5 mr-3" />
                {uploadedFileName || 'Upload Recording'}
              </span>
            </Button>
          </label>
        </div>
        {uploadedFileName && (
          <p className="text-sm text-cyan-400 font-medium">✓ {uploadedFileName}</p>
        )}
      </div>


      {/* Monitoring Control */}
      <div className="space-y-3">
        <Label className="text-sm font-medium text-slate-500">Monitoring Control</Label>
        {!isMonitoring ? (
          <Button
            onClick={onStartMonitoring}
            disabled={!hasVideo || isProcessing}
            className="w-full bg-cyan-600 hover:bg-cyan-500 text-slate-950 font-bold h-12 text-base shadow-[0_0_20px_rgba(8,145,178,0.3)] transition-all"
          >
            {isProcessing ? (
              <>
                <RotateCcw className="w-5 h-5 mr-2 animate-spin" />
                Processing Feed...
              </>
            ) : (
              <>
                <Play className="w-5 h-5 mr-2" />
                Start Monitoring
              </>
            )}
          </Button>
        ) : (
          <Button
            onClick={onStopMonitoring}
            className="w-full bg-red-600 hover:bg-red-700 text-white font-bold h-12 text-base"
          >
            <Square className="w-5 h-5 mr-2" />
            Stop Monitoring
          </Button>
        )}
      </div>

      {/* ROI Toggle */}
      <div className="flex items-center justify-between bg-slate-900/40 border border-cyan-900/20 rounded-lg p-4">
        <div className="flex flex-col gap-1">
          <Label htmlFor="roi-toggle" className="text-sm font-medium text-cyan-100 cursor-pointer">
            Restricted Zone Overlay
          </Label>
          <span className="text-xs text-gray-500">Show ROI boundaries</span>
        </div>
        <Switch
          id="roi-toggle"
          checked={showROI}
          onCheckedChange={onToggleROI}
          disabled={!isMonitoring}
        />
      </div>

      {/* Incident Scan */}
      <div className="space-y-3">
        <Button
          onClick={onScanIncidents}
          disabled={!isMonitoring}
          variant="outline"
          className="w-full bg-slate-900/50 border-cyan-700 hover:bg-cyan-900/20 text-cyan-300 h-11 text-base"
        >
          <Search className="w-5 h-5 mr-2" />
          Manual Incident Scan
        </Button>
      </div>

      {/* System Status Indicator */}
      <div className="bg-slate-900/40 border border-cyan-900/20 rounded-lg p-4">
        <div className="flex items-center justify-between">
          <span className="text-sm font-medium text-slate-500">System Mode</span>
          <span className={`text-sm font-bold ${isMonitoring ? 'text-cyan-400' : 'text-slate-600'}`}>
            {isMonitoring ? 'ACTIVE' : 'STANDBY'}
          </span>
        </div>
      </div>
    </div>
  );
}