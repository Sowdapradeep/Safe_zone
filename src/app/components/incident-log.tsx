import { useState } from 'react';
import { AlertTriangle, CheckCircle, Clock, XCircle } from 'lucide-react';
import { Badge } from './ui/badge';
import { Button } from './ui/button';
import { ScrollArea } from './ui/scroll-area';

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
    <div className="bg-zinc-900/60 border border-zinc-800/50 rounded-xl h-full flex flex-col backdrop-blur-md shadow-xl overflow-hidden">
      <div className="p-5 border-b border-zinc-800/50 bg-black/20">
        <h3 className="text-sm font-bold text-zinc-400 uppercase tracking-widest mb-4 flex items-center gap-2">
          <div className="w-1.5 h-4 bg-yellow-500 rounded-full shadow-[0_0_10px_rgba(234,179,8,0.3)]" />
          Incident Management
        </h3>

        {/* Filter Tabs */}
        <div className="flex gap-1.5">
          {['all', 'open', 'in-progress', 'closed'].map((f) => (
            <button
              key={f}
              onClick={() => setFilter(f as any)}
              className={`px-3 py-1.5 text-[10px] font-bold uppercase tracking-wider rounded-lg transition-all ${filter === f
                ? 'bg-yellow-500 text-black shadow-lg shadow-yellow-500/20'
                : 'bg-zinc-800/50 border border-zinc-700/50 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300'
                }`}
            >
              {f === 'all' ? 'All' : f.replace('-', ' ')}
              {f !== 'all' && (
                <span className={`ml-1.5 opacity-60 ${filter === f ? 'text-black' : 'text-zinc-400'}`}>
                  {incidents.filter(i => i.status === f).length}
                </span>
              )}
            </button>
          ))}
        </div>
      </div>

      <ScrollArea className="flex-1 p-4">
        <div className="space-y-3">
          {filteredIncidents.length === 0 ? (
            <div className="text-center py-16 text-zinc-600">
              <AlertTriangle className="w-12 h-12 mx-auto mb-4 opacity-20" />
              <p className="text-xs font-mono uppercase tracking-widest">No active incidents</p>
            </div>
          ) : (
            filteredIncidents.map((incident, index) => (
              <div
                key={incident.id || (incident as any)._id || index}
                className="bg-black/40 border border-zinc-800/50 rounded-xl p-4 space-y-4 group hover:border-zinc-700/50 transition-all hover:shadow-lg"
              >
                {/* Header */}
                <div className="flex items-start justify-between gap-3">
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-2">
                      <Badge className={`${getSeverityColor(incident.severity)} border-0 text-[10px] font-bold px-2 py-0.5 rounded-full`}>
                        {incident.severity.toUpperCase()}
                      </Badge>
                      <Badge className={`${getStatusColor(incident.status)} border-0 text-[10px] flex items-center gap-1.5 font-bold px-2 py-0.5 rounded-full`}>
                        {getStatusIcon(incident.status)}
                        {incident.status.toUpperCase().replace('-', ' ')}
                      </Badge>
                    </div>
                    <h4 className="text-sm font-bold text-zinc-100 leading-tight">
                      {incident.incidentType}
                    </h4>
                  </div>
                  <div className="text-[10px] text-zinc-600 font-mono bg-zinc-900/50 px-2 py-1 rounded border border-zinc-800/50">
                    {formatTimestamp(incident.timestamp).split(',')[1]}
                  </div>
                </div>

                {/* Details */}
                <div className="grid grid-cols-2 gap-x-4 gap-y-2 text-[10px] font-mono">
                  <div className="flex flex-col">
                    <span className="text-zinc-600 uppercase tracking-tighter">Source</span>
                    <span className="text-zinc-300 font-bold">{incident.cameraId}</span>
                  </div>
                  <div className="flex flex-col">
                    <span className="text-zinc-600 uppercase tracking-tighter">Confidence</span>
                    <span className="text-emerald-500 font-bold">{(incident.confidence * 100).toFixed(1)}%</span>
                  </div>
                  <div className="col-span-2 flex flex-col">
                    <span className="text-zinc-600 uppercase tracking-tighter">Zone</span>
                    <span className="text-zinc-300">{incident.zone}</span>
                  </div>
                </div>

                <div className="relative">
                  <p className="text-[11px] text-zinc-400 border-l-2 border-yellow-500/30 pl-3 leading-relaxed italic">
                    {incident.description}
                  </p>
                </div>

                {/* Actions */}
                {incident.status === 'open' && (
                  <div className="flex gap-2 pt-1">
                    <Button
                      size="sm"
                      onClick={() => onAcknowledge(incident.id)}
                      className="flex-1 bg-yellow-600 hover:bg-yellow-500 text-black font-bold text-[11px] py-4 rounded-lg shadow-lg shadow-yellow-500/10"
                    >
                      Acknowledge
                    </Button>
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => onMarkFalseAlarm(incident.id)}
                      className="flex-1 border-zinc-700 text-zinc-400 hover:bg-zinc-800 hover:text-white text-[11px] py-4 rounded-lg"
                    >
                      False Alarm
                    </Button>
                  </div>
                )}

                {incident.status === 'in-progress' && (
                  <Button
                    size="sm"
                    onClick={() => onResolve(incident.id)}
                    className="w-full bg-emerald-600 hover:bg-emerald-500 text-black font-bold text-[11px] py-4 rounded-lg shadow-lg shadow-emerald-500/10"
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