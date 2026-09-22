import React from 'react';
import { Layers, AlertTriangle, CheckCircle, Percent, Clock, BarChart3 } from 'lucide-react';

export default function MetricsOverview({ summary }) {
  if (!summary) return null;

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4 mb-6">
      {/* Total Sessions */}
      <div className="glass-panel p-4 rounded-xl relative overflow-hidden">
        <div className="flex items-center justify-between mb-2">
          <span className="text-xs font-medium text-gray-400">Total Block Sessions</span>
          <div className="p-2 bg-blue-500/10 rounded-lg text-blue-400">
            <Layers className="w-4 h-4" />
          </div>
        </div>
        <div className="text-2xl font-bold text-white tracking-tight">{summary.total_sessions.toLocaleString()}</div>
        <div className="text-[11px] text-gray-500 mt-1">Block sequences evaluated</div>
      </div>

      {/* Anomalies Detected */}
      <div className="glass-panel p-4 rounded-xl border-rose-500/20 relative overflow-hidden">
        <div className="flex items-center justify-between mb-2">
          <span className="text-xs font-medium text-rose-300">Anomalous Blocks</span>
          <div className="p-2 bg-rose-500/10 rounded-lg text-rose-400">
            <AlertTriangle className="w-4 h-4" />
          </div>
        </div>
        <div className="text-2xl font-bold text-rose-400 tracking-tight">{summary.anomaly_count.toLocaleString()}</div>
        <div className="text-[11px] text-rose-300/70 mt-1">Requires system investigation</div>
      </div>

      {/* Normal Sessions */}
      <div className="glass-panel p-4 rounded-xl relative overflow-hidden">
        <div className="flex items-center justify-between mb-2">
          <span className="text-xs font-medium text-emerald-400">Normal Blocks</span>
          <div className="p-2 bg-emerald-500/10 rounded-lg text-emerald-400">
            <CheckCircle className="w-4 h-4" />
          </div>
        </div>
        <div className="text-2xl font-bold text-emerald-400 tracking-tight">{summary.normal_count.toLocaleString()}</div>
        <div className="text-[11px] text-gray-500 mt-1">Standard sequence patterns</div>
      </div>

      {/* Anomaly Rate */}
      <div className="glass-panel p-4 rounded-xl relative overflow-hidden">
        <div className="flex items-center justify-between mb-2">
          <span className="text-xs font-medium text-gray-400">Anomaly Rate</span>
          <div className="p-2 bg-purple-500/10 rounded-lg text-purple-400">
            <Percent className="w-4 h-4" />
          </div>
        </div>
        <div className="text-2xl font-bold text-purple-400 tracking-tight">{summary.anomaly_rate}%</div>
        <div className="w-full bg-gray-800 rounded-full h-1.5 mt-2">
          <div
            className="bg-purple-500 h-1.5 rounded-full"
            style={{ width: `${Math.min(summary.anomaly_rate, 100)}%` }}
          ></div>
        </div>
      </div>

      {/* Latency / Speed */}
      <div className="glass-panel p-4 rounded-xl relative overflow-hidden">
        <div className="flex items-center justify-between mb-2">
          <span className="text-xs font-medium text-gray-400">Pipeline Latency</span>
          <div className="p-2 bg-amber-500/10 rounded-lg text-amber-400">
            <Clock className="w-4 h-4" />
          </div>
        </div>
        <div className="text-2xl font-bold text-amber-400 tracking-tight">{summary.processing_time_seconds}s</div>
        <div className="text-[11px] text-gray-500 mt-1">Drain parsing + LSTM inference</div>
      </div>
    </div>
  );
}
