"use client";

import { useEffect, useState } from "react";
import { getIps, type Ips } from "@/lib/api";
import { utc } from "@/lib/utils";

const POLL_MS = 3000;
const size = (bytes: number) => (bytes < 1024 ? `${bytes} B` : bytes < 1048576 ? `${(bytes / 1024).toFixed(1)} KB` : `${(bytes / 1048576).toFixed(1)} MB`);

// Every machine named in the log. "Active" = seen within the last minute of log time.
export function ActiveIps({ onSearch }: { onSearch: (ip: string) => void }) {
  const [data, setData] = useState<Ips | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout>;
    async function refresh() {
      try {
        const next = await getIps();
        if (!cancelled) { setData(next); setError(""); }
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : "Could not reach the server.");
      }
      if (!cancelled) timer = setTimeout(refresh, POLL_MS);
    }
    refresh();
    return () => { cancelled = true; clearTimeout(timer); };
  }, []);

  return (
    <section className="rounded-lg border border-slate-800 bg-slate-900 shadow-sm">
      <div className="flex items-baseline justify-between border-b border-slate-800 px-3 py-2">
        <h2 className="text-xs font-semibold uppercase tracking-wide text-slate-400">Active IP addresses</h2>
        <span className="text-[11px] text-slate-500">
          {data ? `${data.active_count} active of ${data.ips.length} seen · active = in the last ${data.window}s of log time` : "loading…"}
        </span>
      </div>
      {error && <p className="px-3 py-2 text-xs text-red-400">{error}</p>}
      {data && data.ips.length === 0 && <p className="px-3 py-6 text-center text-xs text-slate-500">Waiting for log lines…</p>}
      {data && data.ips.length > 0 && (
        <table className="w-full text-xs">
          <thead className="text-left text-[11px] uppercase text-slate-500">
            <tr>
              <th className="px-3 py-1.5 font-medium">IP address</th>
              <th className="px-3 py-1.5 font-medium">Status</th>
              <th className="px-3 py-1.5 text-right font-medium">Log lines</th>
              <th className="px-3 py-1.5 text-right font-medium">Data sent</th>
              <th className="px-3 py-1.5 font-medium">First seen (UTC)</th>
              <th className="px-3 py-1.5 font-medium">Last seen (UTC)</th>
              <th className="px-3 py-1.5" />
            </tr>
          </thead>
          <tbody>
            {data.ips.map((row) => (
              <tr key={row.ip} className="border-t border-slate-800">
                <td className="px-3 py-1.5 font-mono text-slate-200">{row.ip}</td>
                <td className="px-3 py-1.5">
                  <span className={`inline-flex items-center gap-1.5 ${row.active ? "text-emerald-400" : "text-slate-500"}`}>
                    <span className={`size-1.5 rounded-full ${row.active ? "animate-pulse bg-emerald-400" : "bg-slate-600"}`} />
                    {row.active ? "active" : "idle"}
                  </span>
                </td>
                <td className="px-3 py-1.5 text-right font-mono text-slate-300">{row.lines.toLocaleString()}</td>
                <td className="px-3 py-1.5 text-right font-mono text-slate-300">{row.bytes_sent ? size(row.bytes_sent) : "–"}</td>
                <td className="px-3 py-1.5 font-mono text-slate-400">{utc(row.first)}</td>
                <td className="px-3 py-1.5 font-mono text-slate-400">{utc(row.last)}</td>
                <td className="px-3 py-1.5 text-right">
                  <button
                    type="button" onClick={() => onSearch(row.ip)}
                    className="rounded-sm border border-slate-700 px-2 py-0.5 text-[11px] text-slate-300 hover:border-indigo-500"
                  >
                    See its logs
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </section>
  );
}
