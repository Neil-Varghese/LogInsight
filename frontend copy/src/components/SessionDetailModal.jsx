import React from 'react';
import { X, AlertTriangle, CheckCircle, Code, Layers, FileCode } from 'lucide-react';

export default function SessionDetailModal({ session, onClose }) {
  if (!session) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-end bg-black/60 backdrop-blur-sm transition-opacity">
      <div className="w-full max-w-3xl h-full bg-[#0d1322] border-l border-gray-800 p-6 overflow-y-auto flex flex-col shadow-2xl animate-in slide-in-from-right duration-200">
        {/* Header */}
        <div className="flex items-start justify-between pb-4 border-b border-gray-800 mb-6">
          <div>
            <div className="flex items-center gap-3">
              <h2 className="text-xl font-bold font-mono text-white">{session.block_id}</h2>
              {session.is_anomalous ? (
                <span className="px-2.5 py-1 rounded-full bg-rose-500/10 text-rose-400 border border-rose-500/20 text-xs font-semibold flex items-center gap-1.5">
                  <AlertTriangle className="w-3.5 h-3.5" />
                  Anomalous Block
                </span>
              ) : (
                <span className="px-2.5 py-1 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 text-xs font-medium flex items-center gap-1.5">
                  <CheckCircle className="w-3.5 h-3.5" />
                  Normal Block
                </span>
              )}
            </div>
            <p className="text-xs text-gray-400 mt-1">
              Sequence of {session.sequence_len} log events parsed via Drain
            </p>
          </div>

          <button
            onClick={onClose}
            className="p-2 text-gray-400 hover:text-white bg-gray-900 rounded-lg border border-gray-800 hover:border-gray-700"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Anomaly Gauge Banner */}
        <div className={`p-4 rounded-xl mb-6 border ${
          session.is_anomalous ? 'bg-rose-950/30 border-rose-500/30' : 'bg-gray-900/60 border-gray-800'
        }`}>
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-medium text-gray-300">LSTM Classifier Confidence</span>
            <span className="font-mono text-sm font-bold text-white">
              {(session.anomaly_score * 100).toFixed(2)}%
            </span>
          </div>
          <div className="w-full bg-gray-800 rounded-full h-2 overflow-hidden">
            <div
              className={`h-2 rounded-full transition-all duration-500 ${
                session.is_anomalous ? 'bg-rose-500' : 'bg-emerald-500'
              }`}
              style={{ width: `${session.anomaly_score * 100}%` }}
            ></div>
          </div>
        </div>

        {/* Drain Template Sequence Pill Bar */}
        <div className="mb-6">
          <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2 flex items-center gap-1.5">
            <Layers className="w-4 h-4 text-blue-400" />
            Drain EventId Sequence ({session.event_ids.length})
          </h3>
          <div className="flex flex-wrap gap-1.5 bg-gray-950 p-3 rounded-xl border border-gray-800 font-mono text-xs">
            {session.event_ids.map((eid, idx) => (
              <span
                key={idx}
                className="px-2 py-1 bg-blue-950/40 border border-blue-500/20 text-blue-300 rounded"
              >
                E{idx + 1}: <span className="font-bold text-blue-200">{eid}</span>
              </span>
            ))}
          </div>
        </div>

        {/* Timeline of Raw Log Lines */}
        <div className="flex-1">
          <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3 flex items-center gap-1.5">
            <FileCode className="w-4 h-4 text-blue-400" />
            Raw Log Sequence Timeline
          </h3>

          <div className="space-y-3">
            {session.raw_logs.map((log, idx) => (
              <div
                key={idx}
                className="p-3.5 bg-gray-900/60 rounded-xl border border-gray-800/80 hover:border-gray-700 transition-all text-xs"
              >
                <div className="flex items-center justify-between text-gray-400 font-mono mb-1.5 text-[11px]">
                  <div className="flex items-center gap-2">
                    <span className="text-blue-400 font-bold">Line {log.LineId || idx + 1}</span>
                    <span>•</span>
                    <span>{log.Date} {log.Time}</span>
                    <span>•</span>
                    <span className="px-1.5 py-0.5 bg-gray-800 rounded text-gray-300">{log.Level}</span>
                  </div>
                  <span className="text-gray-500 font-mono">EventId: {log.EventId}</span>
                </div>

                <div className="font-mono text-gray-200 bg-gray-950 p-2.5 rounded-lg border border-gray-800/60 text-xs overflow-x-auto whitespace-pre-wrap">
                  {log.Content}
                </div>

                {log.EventTemplate && (
                  <div className="mt-2 text-[11px] text-gray-400 flex items-center gap-1.5">
                    <Code className="w-3.5 h-3.5 text-gray-500 shrink-0" />
                    <span className="font-mono text-gray-400 truncate">
                      Template: <span className="text-gray-300">{log.EventTemplate}</span>
                    </span>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
