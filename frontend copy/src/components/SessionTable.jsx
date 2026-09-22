import React, { useEffect, useState } from 'react';
import { AlertCircle, CheckCircle2, ChevronRight, Search, Terminal } from 'lucide-react';

export default function SessionTable({ jobId, onSelectSession }) {
  const [data, setData] = useState({ blocks: [], total: 0, page: 1, limit: 25 });
  const [filter, setFilter] = useState('all');
  const [search, setSearch] = useState('');
  const [page, setPage] = useState(1);
  const [error, setError] = useState('');

  useEffect(() => {
    const query = new URLSearchParams({ page, limit: 25, filter, search });
    fetch(`/api/jobs/${jobId}/blocks?${query}`)
      .then((response) => response.ok ? response.json() : response.json().then((item) => Promise.reject(new Error(item.error))))
      .then(setData).catch((requestError) => setError(requestError.message));
  }, [jobId, page, filter, search]);

  const inspect = async (block) => {
    const response = await fetch(`/api/jobs/${jobId}/blocks/${encodeURIComponent(block.block_id)}`);
    const detail = await response.json();
    if (response.ok) onSelectSession(detail); else setError(detail.error || 'Could not load block details.');
  };
  const pages = Math.max(1, Math.ceil(data.total / data.limit));

  return <div className="glass-panel rounded-2xl p-6">
    <div className="flex flex-col md:flex-row justify-between gap-3 mb-5">
      <div><h2 className="text-lg font-bold text-white flex items-center gap-2"><Terminal className="w-5 h-5 text-blue-400" />Classified Block Sessions</h2><p className="text-xs text-gray-400">{data.total.toLocaleString()} blocks stored in this run’s results database</p></div>
      <div className="flex gap-2"><div className="relative"><Search className="w-4 h-4 absolute left-3 top-2 text-gray-500" /><input value={search} onChange={(e) => { setSearch(e.target.value); setPage(1); }} placeholder="Block ID" className="bg-gray-900 border border-gray-800 rounded-lg pl-9 py-1.5 text-xs text-white" /></div><select value={filter} onChange={(e) => { setFilter(e.target.value); setPage(1); }} className="bg-gray-900 border border-gray-800 rounded-lg px-2 text-xs text-gray-300"><option value="all">All</option><option value="anomalous">Anomalous</option><option value="normal">Normal</option></select></div>
    </div>
    {error && <p className="mb-3 text-xs text-rose-400">{error}</p>}
    <div className="overflow-x-auto rounded-xl border border-gray-800"><table className="w-full text-left text-xs text-gray-300"><thead className="bg-gray-900 text-gray-400"><tr><th className="px-4 py-3">Block ID</th><th className="px-4 py-3">Status</th><th className="px-4 py-3">Anomaly confidence</th><th className="px-4 py-3">Events</th><th className="px-4 py-3"></th></tr></thead><tbody>{data.blocks.map((block) => <tr key={block.block_id} className="border-t border-gray-800"><td className="px-4 py-3 font-mono text-white">{block.block_id}</td><td className="px-4 py-3">{block.is_anomalous ? <span className="text-rose-400"><AlertCircle className="w-3.5 h-3.5 inline mr-1" />Anomalous</span> : <span className="text-emerald-400"><CheckCircle2 className="w-3.5 h-3.5 inline mr-1" />Normal</span>}</td><td className="px-4 py-3">{(block.anomaly_score * 100).toFixed(1)}%</td><td className="px-4 py-3">{block.sequence_len}</td><td className="px-4 py-3 text-right"><button onClick={() => inspect(block)} className="text-blue-400">Inspect <ChevronRight className="w-4 h-4 inline" /></button></td></tr>)}</tbody></table></div>
    <div className="flex justify-between mt-4 text-xs text-gray-400"><span>Page {data.page} of {pages}</span><div className="flex gap-2"><button disabled={page === 1} onClick={() => setPage(page - 1)} className="px-3 py-1 bg-gray-900 rounded disabled:opacity-40">Previous</button><button disabled={page === pages} onClick={() => setPage(page + 1)} className="px-3 py-1 bg-gray-900 rounded disabled:opacity-40">Next</button></div></div>
  </div>;
}
