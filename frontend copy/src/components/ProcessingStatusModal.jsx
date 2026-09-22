import React from 'react';
import { Cpu, FileText, Layers, CheckCircle2, Loader2, Zap, Terminal } from 'lucide-react';

export default function ProcessingStatusModal({ isLoading, filename, streamData }) {
  if (!isLoading) return null;

  const percent = streamData?.percent ?? 0;
  const message = streamData?.message || 'Initiating classification pipeline...';
  const stage = streamData?.stage || 'preprocessing';

  const steps = [
    { label: '1. Drain Log Parsing', icon: FileText, stage: 'preprocessing' },
    { label: '2. LSTM Inference', icon: Zap, stage: 'classification' },
  ];

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-md animate-in fade-in duration-200">
      <div className="w-full max-w-lg bg-[#0f172a] border border-gray-800 rounded-2xl p-6 shadow-2xl space-y-6">
        {/* Header */}
        <div className="flex items-center gap-3">
          <div className="p-3 bg-blue-500/10 text-blue-400 rounded-xl border border-blue-500/20 shadow-inner">
            <Cpu className="w-6 h-6 animate-pulse" />
          </div>
          <div>
            <h3 className="text-base font-bold text-white tracking-tight">Log Anomaly Detection Pipeline</h3>
            <p className="text-xs text-gray-400 font-mono truncate max-w-[300px]">
              Processing Log Stream: <span className="text-blue-400 font-bold">{filename || 'HDFS Log'}</span>
            </p>
          </div>
        </div>

        {/* Live Percentage & Session Counter Card */}
        <div className="bg-gray-900/90 border border-gray-800 rounded-xl p-4 space-y-3 shadow-sm">
          <div className="flex items-center justify-between text-xs font-mono">
            <span className="text-gray-200 font-semibold flex items-center gap-2">
              <Loader2 className="w-3.5 h-3.5 text-blue-400 animate-spin" />
              {message}
            </span>
            <span className="text-blue-400 font-bold text-base font-mono">{percent}%</span>
          </div>

          {/* Full Pipeline Progress Bar (Parsing -> Sessionization -> LSTM) */}
          <div className="w-full bg-gray-950 rounded-full h-3.5 overflow-hidden border border-gray-800 p-0.5 relative">
            <div
              className="bg-gradient-to-r from-blue-600 via-indigo-500 to-emerald-400 h-2.5 rounded-full transition-all duration-150 ease-out shadow-lg shadow-blue-500/30"
              style={{ width: `${percent}%` }}
            ></div>
          </div>

        </div>

        {/* Sub-steps Phase Indicator Cards */}
        <div className="grid grid-cols-2 gap-2.5">
          {steps.map((st, i) => {
            const isCompleted = stage === 'complete' || (stage === 'classification' && st.stage === 'preprocessing');
            const isActive = stage === st.stage;
            const Icon = st.icon;

            return (
              <div
                key={i}
                className={`p-3 rounded-xl border text-center transition-all ${
                  isActive
                    ? 'bg-blue-600/20 border-blue-500/50 text-blue-300 shadow-md shadow-blue-500/10'
                    : isCompleted
                    ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400'
                    : 'bg-gray-950/60 border-gray-900 text-gray-600'
                }`}
              >
                <div className="flex justify-center mb-1">
                  {isCompleted ? (
                    <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                  ) : isActive ? (
                    <Loader2 className="w-4 h-4 text-blue-400 animate-spin" />
                  ) : (
                    <Icon className="w-4 h-4 text-gray-600" />
                  )}
                </div>
                <div className="text-[11px] font-semibold leading-snug">{st.label}</div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
