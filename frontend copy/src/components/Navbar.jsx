import React from 'react';
import { Activity, ShieldAlert, Cpu, CheckCircle2 } from 'lucide-react';

export default function Navbar({ isModelLoaded }) {
  return (
    <header className="sticky top-0 z-30 glass-panel border-b border-gray-800 px-6 py-4">
      <div className="max-w-7xl mx-auto flex flex-col md:flex-row md:items-center justify-between gap-4">
        {/* Brand */}
        <div className="flex items-center space-x-3">
          <div className="p-2.5 bg-gradient-to-tr from-blue-600 to-indigo-600 rounded-xl shadow-lg shadow-blue-500/20 text-white">
            <Activity className="w-6 h-6 animate-pulse" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-xl font-bold text-white tracking-tight">LogInsight</h1>
              <span className="text-xs px-2 py-0.5 rounded-full font-medium bg-blue-500/10 text-blue-400 border border-blue-500/20">
                LSTM Anomaly Detector
              </span>
            </div>
            <p className="text-xs text-gray-400">Sequence-based log parsing & real-time classification</p>
          </div>
        </div>

        {/* Controls & Status */}
        <div className="flex items-center gap-6">
          <div className="flex items-center gap-3 bg-gray-900/60 border border-gray-800 px-3.5 py-1.5 rounded-lg">
            <span className="text-xs text-gray-400 font-medium">Anomaly Threshold:</span>
            <span className="text-xs font-mono font-bold text-blue-400">50%</span>
          </div>

          {/* Model Status Pill */}
          <div className="flex items-center gap-2 text-xs px-3 py-1.5 rounded-lg bg-emerald-500/10 border border-emerald-500/20 text-emerald-400">
            <Cpu className="w-4 h-4 text-emerald-400" />
            <span className="font-medium">LSTM Keras Model Online</span>
            <CheckCircle2 className="w-3.5 h-3.5" />
          </div>
        </div>
      </div>
    </header>
  );
}
