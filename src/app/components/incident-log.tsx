import { useState } from 'react';
import { AlertTriangle, CheckCircle, Clock, XCircle } from 'lucide-react';
import { Badge } from '@/app/components/ui/badge';
import { Button } from '@/app/components/ui/button';
import { ScrollArea } from '@/app/components/ui/scroll-area';

export interface Incident {
  id: string;
  cameraId: string;
  location: string;
  zone: string;
  incidentType: string;
  severity: 'low' | 'medium' | 'high';
  status: 'open' | 'in-progress' | 'closed' | 'false-alarm';
  timestamp: string;
  confidence: number;
  description: string;
}

interface IncidentLogProps {
  incidents: Incident[];
  onAcknowledge: (id: string) => void;
  onMarkFalseAlarm: (id: string) => void;
  onResolve: (id: string) => void;
}

export function IncidentLog({
  incidents,
  onAcknowledge,
  onMarkFalseAlarm,
  onResolve
}: IncidentLogProps) {
  const [filter, setFilter] = useState<'all' | 'open' | 'in-progress' | 'closed'>('all');

  const filteredIncidents = incidents.filter(incident => {
    if (filter === 'all') return true;
    return incident.status === filter;
  });

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'high': return 'bg-red-900 text-red-300 border-red-700';
      case 'medium': return 'bg-yellow-900 text-yellow-300 border-yellow-700';
      default: return 'bg-blue-900 text-blue-300 border-blue-700';
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'open': return 'bg-red-950 text-red-400 border-red-800';
      case 'in-progress': return 'bg-yellow-950 text-yellow-400 border-yellow-800';
      case 'closed': return 'bg-green-950 text-green-400 border-green-800';
      case 'false-alarm': return 'bg-gray-900 text-gray-400 border-gray-700';
      default: return 'bg-gray-900 text-gray-400 border-gray-700';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'open': return <AlertTriangle className="w-3 h-3" />;
      case 'in-progress': return <Clock className="w-3 h-3" />;
      case 'closed': return <CheckCircle className="w-3 h-3" />;
      case 'false-alarm': return <XCircle className="w-3 h-3" />;
    }
  };

  const formatTimestamp = (timestamp: string) => {
    const date = new Date(timestamp);
    return date.toLocaleString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit'
    });
  };

  return (
    <div className="bg-zinc-950 border border-yellow-900/30 rounded-lg h-full flex flex-col">
      <div className="p-5 border-b border-yellow-900/30">
        <h3 className="text-base font-bold text-yellow-400 uppercase tracking-wider mb-4">
          Incident Management
        </h3>
        
        {/* Filter Tabs */}
        <div className="flex gap-2">
          {['all', 'open', 'in-progress', 'closed'].map((f) => (
            <button
              key={f}
              onClick={() => setFilter(f as any)}
              className={`px-4 py-2 text-sm font-medium rounded ${
                filter === f
                  ? 'bg-yellow-600 text-black font-semibold'
                  : 'bg-zinc-900 border border-yellow-900/30 text-yellow-700 hover:bg-zinc-800'
              }`}
            >
              {f === 'all' ? 'All' : f.split('-').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ')}
              {f !== 'all' && (
                <span className="ml-1.5">
                  ({incidents.filter(i => i.status === f).length})
                </span>
              )}
            </button>
          ))}
        </div>
      </div>

      <ScrollArea className="flex-1 p-4">
        <div className="space-y-4">
          {filteredIncidents.length === 0 ? (
            <div className="text-center py-12 text-gray-500">
              <AlertTriangle className="w-16 h-16 mx-auto mb-4 opacity-30" />
              <p className="text-base">No incidents to display</p>
            </div>
          ) : (
            filteredIncidents.map((incident) => (
              <div
                key={incident.id}
                className="bg-zinc-900 border border-yellow-900/30 rounded-lg p-4 space-y-3"
              >
                {/* Header */}
                <div className="flex items-start justify-between gap-3">
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-2">
                      <Badge className={`${getSeverityColor(incident.severity)} border text-xs font-semibold`}>
                        {incident.severity.toUpperCase()}
                      </Badge>
                      <Badge className={`${getStatusColor(incident.status)} border text-xs flex items-center gap-1.5 font-semibold`}>
                        {getStatusIcon(incident.status)}
                        {incident.status.toUpperCase().replace('-', ' ')}
                      </Badge>
                    </div>
                    <h4 className="text-base font-semibold text-white leading-tight">
                      {incident.incidentType}
                    </h4>
                  </div>
                  <div className="text-xs text-gray-500 whitespace-nowrap">
                    {formatTimestamp(incident.timestamp)}
                  </div>
                </div>

                {/* Details */}
                <div className="grid grid-cols-2 gap-3 text-sm">
                  <div>
                    <span className="text-gray-500">Camera:</span>
                    <span className="text-white ml-2 font-mono font-medium">{incident.cameraId}</span>
                  </div>
                  <div>
                    <span className="text-gray-500">Zone:</span>
                    <span className="text-white ml-2 font-medium">{incident.zone}</span>
                  </div>
                  <div className="col-span-2">
                    <span className="text-gray-500">Location:</span>
                    <span className="text-white ml-2 font-medium">{incident.location}</span>
                  </div>
                  <div>
                    <span className="text-gray-500">Confidence:</span>
                    <span className="text-yellow-400 ml-2 font-mono font-medium">
                      {(incident.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>

                <p className="text-sm text-yellow-600 border-t border-yellow-900/30 pt-3 leading-relaxed">
                  {incident.description}
                </p>

                {/* Actions */}
                {incident.status === 'open' && (
                  <div className="flex gap-3 pt-2">
                    <Button
                      size="sm"
                      onClick={() => onAcknowledge(incident.id)}
                      className="flex-1 bg-yellow-600 hover:bg-yellow-700 text-black font-semibold text-sm h-10"
                    >
                      Acknowledge
                    </Button>
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => onMarkFalseAlarm(incident.id)}
                      className="flex-1 border-yellow-700 text-yellow-300 hover:bg-zinc-800 text-sm h-10"
                    >
                      False Alarm
                    </Button>
                  </div>
                )}

                {incident.status === 'in-progress' && (
                  <Button
                    size="sm"
                    onClick={() => onResolve(incident.id)}
                    className="w-full bg-lime-600 hover:bg-lime-700 text-black font-semibold text-sm h-10"
                  >
                    Mark Resolved
                  </Button>
                )}
              </div>
            ))
          )}
        </div>
      </ScrollArea>
    </div>
  );
}