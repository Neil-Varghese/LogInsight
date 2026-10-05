"use client";

import { ChevronDown, ChevronUp, Search as SearchIcon } from "lucide-react";
import { useEffect, useState, type FormEvent } from "react";
import { useInspector } from "@/components/dashboard/inspect-drawer";
import { searchLogs, type SearchResult } from "@/lib/api";

// datetime-local gives "2008-11-09T20:35:18" with no zone; log time is UTC, so read it as UTC.
const toEpoch = (value: string) => (value ? Date.parse(`${value}Z`) / 1000 : null);
const FIELD = "rounded-sm border border-slate-700 bg-slate-950 px-2 py-1 font-mono text-xs text-slate-200 outline-none focus:border-indigo-500";

// Advanced search: every raw log line for one IP address, one time window, or both, plus the blocks those lines belong to.
// Closed it is just one search box; clicking it opens the whole search.
export function LogSearch({ initialIp, threshold }: { initialIp: string; threshold: number }) {
  const [open, setOpen] = useState(Boolean(initialIp)); // arriving from the Active IPs tab opens it
  const [ip, setIp] = useState(initialIp);
  const [from, setFrom] = useState("");
  const [to, setTo] = useState("");
  const [result, setResult] = useState<SearchResult | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const { open: inspect, drawer } = useInspector();

  async function run() {
    setBusy(true); setError("");
    try {
      setResult(await searchLogs(ip.trim(), toEpoch(from), toEpoch(to)));
    } catch (e) {
      setResult(null); setError(e instanceof Error ? e.message : "Search failed.");
    } finally {
      setBusy(false);
    }
  }

  // Arriving with an address already chosen: search straight away.
  useEffect(() => {
    if (!initialIp) return;
    searchLogs(initialIp, null, null).then(setResult).catch((e) => setError(e instanceof Error ? e.message : "Search failed."));
  }, [initialIp]);

  function submit(event: FormEvent) { event.preventDefault(); run(); }

  return (
    <section className="rounded-lg border border-slate-800 bg-slate-900 shadow-sm">
      <button
        type="button" aria-expanded={open} aria-controls="log-search-panel" onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center gap-2 px-3 py-2.5 text-left text-xs text-slate-400 hover:text-slate-200"
      >
        <SearchIcon className="size-3.5 shrink-0" />
        <span className="flex-1">{open ? "Search logs" : "Search logs by IP address or time window…"}</span>
        {open ? <ChevronUp className="size-3.5 shrink-0" /> : <ChevronDown className="size-3.5 shrink-0" />}
      </button>

      {open && (
        <div id="log-search-panel" className="space-y-2 border-t border-slate-800 p-3">
          <form onSubmit={submit} className="flex flex-wrap items-end gap-3 text-xs">
            <label className="flex flex-col gap-1 text-slate-400">
              IP address
              <input value={ip} onChange={(e) => setIp(e.target.value)} placeholder="10.250.19.102" className={`${FIELD} w-40`} />
            </label>
            <label className="flex flex-col gap-1 text-slate-400">
              From (UTC)
              <input type="datetime-local" step={1} value={from} onChange={(e) => setFrom(e.target.value)} className={FIELD} />
            </label>
            <label className="flex flex-col gap-1 text-slate-400">
              To (UTC)
              <input type="datetime-local" step={1} value={to} onChange={(e) => setTo(e.target.value)} className={FIELD} />
            </label>
            <button type="submit" disabled={busy} className="rounded-sm border border-indigo-500 bg-indigo-950 px-3 py-1 text-slate-100 disabled:opacity-50">
              {busy ? "Searching…" : "Search"}
            </button>
            <button
              type="button" onClick={() => { setIp(""); setFrom(""); setTo(""); setResult(null); setError(""); }}
              className="rounded-sm border border-slate-700 px-3 py-1 text-slate-300 hover:border-slate-500"
            >
              Clear
            </button>
            <span className="text-[11px] text-slate-500">Fill in an address, a time window, or both. Leave a time empty for no limit.</span>
          </form>

          {error && <p className="border border-red-900 bg-red-950 px-3 py-2 text-xs text-red-400">{error}</p>}

          {result && (
            <>
              <div className="border border-slate-800 bg-slate-950">
                <div className="flex flex-wrap items-baseline justify-between gap-2 border-b border-slate-800 px-3 py-2">
                  <h3 className="text-xs font-semibold uppercase tracking-wide text-slate-400">Blocks involved</h3>
                  <span className="text-[11px] text-slate-500">
                    {result.total_blocks.toLocaleString()} blocks, {result.total_lines.toLocaleString()} log lines
                    {result.total_blocks > result.blocks.length && ` · showing the ${result.blocks.length} highest scores`}
                  </span>
                </div>
                {result.blocks.length === 0 ? (
                  <p className="px-3 py-4 text-xs text-slate-500">No block-related lines matched.</p>
                ) : (
                  <div className="max-h-64 overflow-y-auto">
                    <table className="w-full text-xs">
                      <tbody>
                        {result.blocks.map((b) => {
                          const flagged = b.score !== null && b.score >= threshold;
                          return (
                            <tr key={b.block_id} className="border-t border-slate-800 first:border-t-0">
                              <td className="px-3 py-1 font-mono text-slate-200">{b.block_id}</td>
                              <td className="px-3 py-1 text-right font-mono text-slate-400">{b.lines} lines</td>
                              <td className={`px-3 py-1 text-right font-mono ${flagged ? "text-red-400" : "text-slate-400"}`}>
                                {b.score === null ? "not scored" : `${(b.score * 100).toFixed(1)}%`}
                              </td>
                              <td className="px-3 py-1 text-right">
                                <button type="button" onClick={() => inspect("monitor", b.block_id)} className="rounded-sm border border-slate-700 px-2 py-0.5 text-[11px] text-slate-300 hover:border-indigo-500">
                                  Inspect
                                </button>
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>

              <div className="border border-slate-800 bg-slate-950">
                <div className="flex items-baseline justify-between border-b border-slate-800 px-3 py-2">
                  <h3 className="text-xs font-semibold uppercase tracking-wide text-slate-400">Raw log lines</h3>
                  <span className="text-[11px] text-slate-500">
                    {result.truncated ? `first ${result.lines.length.toLocaleString()} of ${result.total_lines.toLocaleString()}` : `${result.total_lines.toLocaleString()} lines`}
                  </span>
                </div>
                <pre className="max-h-80 overflow-auto p-3 font-mono text-[11px] leading-5 text-slate-300">
                  {result.lines.length ? result.lines.join("\n") : "No lines matched."}
                </pre>
              </div>
            </>
          )}
        </div>
      )}
      {drawer}
    </section>
  );
}
