import React, { useEffect, useState } from 'react';
import { FileText, Play, Loader2, Sparkles } from 'lucide-react';

export default function LogUploader({ onAnalyzeSample, isLoading }) {
  const [testSets, setTestSets] = useState([]);
  const [selectedFile, setSelectedFile] = useState('');
  const [error, setError] = useState('');

  useEffect(() => {
    fetch('/api/test-sets')
      .then((response) => {
        if (!response.ok) throw new Error('Could not load test sets.');
        return response.json();
      })
      .then((files) => {
        setTestSets(files);
        if (files.length) setSelectedFile(files[0].path);
      })
      .catch((loadError) => setError(loadError.message));
  }, []);

  return (
    <div className="glass-panel rounded-2xl p-6 mb-6">
      <div className="flex flex-col gap-4">
        <div className="flex items-center gap-2">
          <Sparkles className="w-4 h-4 text-amber-400" />
          <h3 className="text-sm font-semibold text-gray-200">Available Test Sets</h3>
        </div>
        <p className="text-xs text-gray-400">Files are loaded directly from the backend’s <code>test sets</code> directory.</p>
        {error ? <p className="text-xs text-rose-400">{error}</p> : (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
            {testSets.map((file) => (
              <button key={file.path} onClick={() => setSelectedFile(file.path)} disabled={isLoading}
                className={`flex items-center gap-2 p-3 rounded-lg text-left border text-xs transition-all ${selectedFile === file.path ? 'bg-blue-600/20 border-blue-500/50 text-blue-300' : 'bg-gray-800/40 border-gray-700/50 text-gray-400 hover:bg-gray-800'}`}>
                <FileText className="w-4 h-4 text-blue-400" /><span>{file.name}</span>
              </button>
            ))}
          </div>
        )}
        <button onClick={() => selectedFile && onAnalyzeSample(selectedFile)} disabled={isLoading || !selectedFile}
          className="w-full flex items-center justify-center gap-2 py-2.5 px-4 bg-gradient-to-r from-blue-600 to-indigo-600 text-white font-medium rounded-lg text-xs shadow-lg shadow-blue-500/20 disabled:opacity-50 disabled:cursor-not-allowed">
          {isLoading ? <><Loader2 className="w-4 h-4 animate-spin" /><span>Parsing & Classifying…</span></> : <><Play className="w-4 h-4 fill-current" /><span>Run Anomaly Detection</span></>}
        </button>
      </div>
    </div>
  );
}
