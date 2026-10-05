"use client";

import { useEffect, useState } from "react";
import { Charts } from "@/components/dashboard/charts";
import { useInspector } from "@/components/dashboard/inspect-drawer";
import { KpiCards } from "@/components/dashboard/kpi-cards";
import { SessionsTable, type Session } from "@/components/dashboard/sessions-table";
import { getLiveBlocks, getSummary, type Status, type Summary } from "@/lib/api";

const POLL_MS = 3000;

const kb = (bytes: number) => (bytes < 1024 ? `${bytes} B` : bytes < 1048576 ? `${(bytes / 1024).toFixed(1)} KB` : `${(bytes / 1048576).toFixed(1)} MB`);
const ago = (seconds: number) => (seconds < 5 ? "just now" : seconds < 90 ? `${Math.round(seconds)}s ago` : `${Math.round(seconds / 60)}m ago`);

export function Monitor({ status, threshold, onThreshold }: {
  status: Status | null; threshold: number; onThreshold: (percent: number) => void;
}) {
  const [summary, setSummary] = useState<Summary | null>(null);
  const [rows, setRows] = useState<Session[]>([]);
  const [total, setTotal] = useState(0);
  const [filter, setFilter] = useState<"anomalous" | "all">("anomalous");
  const [error, setError] = useState("");
  const { open, drawer } = useInspector();

  // Re-fetch every few seconds, and immediately whenever the threshold or filter changes.
  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout>;
    async function refresh() {
      try {
        const [s, blocks] = await Promise.all([getSummary(threshold / 100), getLiveBlocks(threshold / 100, filter)]);
        if (cancelled) return;
        setSummary(s); setRows(blocks.rows); setTotal(blocks.total); setError("");
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : "Could not reach the server.");
      }
      if (!cancelled) timer = setTimeout(refresh, POLL_MS);
    }
    refresh();
    return () => { cancelled = true; clearTimeout(timer); };
  }, [threshold, filter]);

  const idleFor = status?.last_ingest ? status.now - status.last_ingest : null;
  const live = status?.model_loaded && !status.error && !error;

  const filterButton = (id: "anomalous" | "all", label: string) => (
    <button
      type="button"
      onClick={() => setFilter(id)}
      className={`rounded-sm border px-2 py-0.5 text-[11px] ${filter === id ? "border-indigo-500 text-slate-100" : "border-slate-700 text-slate-400 hover:text-slate-200"}`}
    >
      {label}
    </button>
  );

  return (
    <div className="space-y-2">
      <section className="flex flex-wrap items-center gap-x-6 gap-y-2 rounded-lg border border-slate-800 bg-slate-900 shadow-sm px-3 py-2 text-xs">
        <span className="flex items-center gap-2 font-medium text-slate-200">
          <span className={`size-2 rounded-full ${live ? "animate-pulse bg-emerald-400" : "bg-amber-400"}`} />
          {live ? "LIVE" : "STARTING"}
          <span className="font-mono font-normal text-slate-400">{status?.file ?? "…"}</span>
        </span>
        <Stat label="Lines ingested" value={status ? status.lines.toLocaleString() : "–"} />
        <Stat label="Backlog" value={status ? kb(status.backlog_bytes) : "–"} />
        <Stat label="Last line" value={idleFor === null ? "none yet" : ago(idleFor)} />
        <Stat label="Unmatched lines" value={status ? status.unmatched.toLocaleString() : "–"} />
        <label className="ml-auto flex items-center gap-2 text-slate-400">
          Anomaly threshold
          <input
            type="range" min={1} max={99} value={threshold}
            onChange={(e) => onThreshold(Number(e.target.value))}
            className="w-36 accent-indigo-500"
          />
          <span className="w-9 font-mono text-slate-200">{threshold}%</span>
        </label>
      </section>

      {(error || status?.error) && (
        <p className="border border-red-900 bg-red-950 px-3 py-2 text-xs text-red-400">{error || status?.error}</p>
      )}

      <KpiCards
        k={{ total: summary?.total ?? 0, anomalous: summary?.anomalous ?? 0, normal: summary?.normal ?? 0, rate: summary?.rate ?? 0 }}
        last={{ label: "Blocks Tracked", value: status ? status.blocks.toLocaleString() : "–", sub: "distinct HDFS blocks seen so far" }}
      />
      <Charts summary={summary} threshold={threshold / 100} />
      <SessionsTable
        title="Live Block Sessions"
        rows={rows}
        onInspect={(blockId) => open("monitor", blockId)}
        toolbar={
          <div className="flex items-center gap-2">
            <span className="text-[11px] text-slate-500">top {rows.length} of {total.toLocaleString()} by score</span>
            {filterButton("anomalous", "Anomalous")}
            {filterButton("all", "All")}
          </div>
        }
      />
      {drawer}
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <span className="text-slate-500">
      {label} <span className="font-mono text-slate-200">{value}</span>
    </span>
  );
}
