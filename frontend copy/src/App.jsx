import React, { useState } from 'react';
import Navbar from './components/Navbar';
import LogUploader from './components/LogUploader';
import MetricsOverview from './components/MetricsOverview';
import ProcessingStatusModal from './components/ProcessingStatusModal';
import SessionTable from './components/SessionTable';
import SessionDetailModal from './components/SessionDetailModal';
import { AlertCircle, Terminal } from 'lucide-react';

const pause = (milliseconds) => new Promise((resolve) => setTimeout(resolve, milliseconds));

export default function App() {
  const [summary, setSummary] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [currentFilename, setCurrentFilename] = useState('');
  const [streamData, setStreamData] = useState(null);
  const [selectedSession, setSelectedSession] = useState(null);

  const waitForJob = async (jobId) => {
    while (true) {
      const response = await fetch(`/api/jobs/${jobId}`);
      const job = await response.json();
      if (!response.ok) throw new Error(job.error || 'Unable to read prediction progress.');
      if (job.stage === 'error') throw new Error(job.error || 'Prediction failed.');
      if (job.stage === 'complete') return job.result;

      const percent = Number.isFinite(job.progress) ? job.progress : 0;
      setStreamData({
        stage: job.stage,
        percent,
        message: job.stage === 'preprocessing'
          ? `Preprocessing log file: ${percent.toFixed(1)}%`
          : `Classifying sequences: ${percent.toFixed(0)}%`,
      });
      await pause(500);
    }
  };

  const handleAnalyzeSample = async (filename) => {
    setIsLoading(true);
    setError(null);
    setCurrentFilename(filename);
    setStreamData({ stage: 'preprocessing', percent: 0, message: 'Starting preprocessing…' });
    const startedAt = performance.now();

    try {
      const response = await fetch('/api/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ filename }),
      });
      const startData = await response.json();
      if (!response.ok) throw new Error(startData.error || 'Failed to start prediction.');

      const result = await waitForJob(startData.job_id);
      const total = result.normal_logs + result.anomalous_logs;
      setSummary({
        filename,
        normal_count: result.normal_logs,
        anomaly_count: result.anomalous_logs,
        total_sessions: total,
        total_lines: total,
        anomaly_rate: total ? ((result.anomalous_logs / total) * 100).toFixed(2) : '0.00',
        processing_time_seconds: ((performance.now() - startedAt) / 1000).toFixed(1),
        job_id: startData.job_id,
      });
      setStreamData({ stage: 'complete', percent: 100, message: 'Classification complete.' });
    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex flex-col bg-[#0b0f19]">
      <Navbar isModelLoaded />
      <main className="flex-1 max-w-7xl w-full mx-auto px-6 py-8">
        {error && (
          <div className="mb-6 p-4 rounded-xl bg-rose-500/10 border border-rose-500/20 text-rose-300 flex items-center gap-3 text-xs">
            <AlertCircle className="w-5 h-5 shrink-0 text-rose-400" />
            <span>{error}</span>
          </div>
        )}

        <LogUploader onAnalyzeSample={handleAnalyzeSample} isLoading={isLoading} />

        {summary ? <>
          <div className="flex items-center gap-2 mb-4">
            <span className="text-xs font-semibold text-gray-400 uppercase tracking-wider">Active Dataset:</span>
            <span className="text-xs font-mono font-bold text-blue-400 bg-blue-500/10 border border-blue-500/20 px-2.5 py-1 rounded-md">{summary.filename}</span>
          </div>
          <MetricsOverview summary={summary} />
          <SessionTable jobId={summary.job_id} onSelectSession={setSelectedSession} />
        </> : !isLoading && (
          <div className="glass-panel p-12 rounded-2xl text-center">
            <div className="p-4 bg-gray-900 rounded-full w-16 h-16 mx-auto mb-4 flex items-center justify-center text-blue-400 border border-gray-800"><Terminal className="w-8 h-8" /></div>
            <h3 className="text-lg font-bold text-white mb-2">No Log Data Analyzed Yet</h3>
            <p className="text-xs text-gray-400 max-w-md mx-auto">Select a file from <code>test sets</code> to run Drain parsing and LSTM classification.</p>
          </div>
        )}
      </main>
      <footer className="border-t border-gray-900 py-6 text-center text-xs text-gray-600"><p>LogInsight • HDFS Sequence Anomaly Detection with LSTM & Drain Log Parser</p></footer>
      <ProcessingStatusModal isLoading={isLoading} filename={currentFilename} streamData={streamData} />
      {selectedSession && <SessionDetailModal session={selectedSession} onClose={() => setSelectedSession(null)} />}
    </div>
  );
}
