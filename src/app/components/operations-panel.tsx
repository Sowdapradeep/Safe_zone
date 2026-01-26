import { useState } from 'react';
import { Upload, Play, Square, Search, RotateCcw } from 'lucide-react';
import { Button } from '@/app/components/ui/button';
import { Label } from '@/app/components/ui/label';
import { Switch } from '@/app/components/ui/switch';

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
    <div className="bg-zinc-950 border border-yellow-900/30 rounded-lg p-5 space-y-5">
      <h3 className="text-base font-bold text-yellow-400 uppercase tracking-wider border-b border-yellow-900/30 pb-3">
        Operations Control
      </h3>

      {/* Video Upload */}
      <div className="space-y-3">
        <Label className="text-sm font-medium text-yellow-600">Feed Source</Label>
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
              className="w-full justify-start bg-zinc-900 border-yellow-900/30 hover:bg-zinc-800 hover:border-yellow-700 text-yellow-100 h-12 text-base"
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
          <p className="text-sm text-lime-400 font-medium">✓ {uploadedFileName}</p>
        )}
      </div>


      {/* Monitoring Control */}
      <div className="space-y-3">
        <Label className="text-sm font-medium text-yellow-600">Monitoring Control</Label>
        {!isMonitoring ? (
          <Button
            onClick={onStartMonitoring}
            disabled={!hasVideo || isProcessing}
            className="w-full bg-lime-600 hover:bg-lime-700 text-black font-bold h-12 text-base"
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
      <div className="flex items-center justify-between bg-zinc-900 border border-yellow-900/30 rounded-lg p-4">
        <div className="flex flex-col gap-1">
          <Label htmlFor="roi-toggle" className="text-sm font-medium text-yellow-200 cursor-pointer">
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
          className="w-full bg-yellow-900/30 border-yellow-700 hover:bg-yellow-900/50 text-yellow-300 h-11 text-base"
        >
          <Search className="w-5 h-5 mr-2" />
          Manual Incident Scan
        </Button>
      </div>

      {/* System Status Indicator */}
      <div className="bg-zinc-900 border border-yellow-900/30 rounded-lg p-4">
        <div className="flex items-center justify-between">
          <span className="text-sm font-medium text-yellow-600">System Mode</span>
          <span className={`text-sm font-bold ${isMonitoring ? 'text-lime-400' : 'text-gray-500'}`}>
            {isMonitoring ? 'ACTIVE' : 'STANDBY'}
          </span>
        </div>
      </div>
    </div>
  );
}