"use client";

import { Pause, Play, Search } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { Charts } from "@/components/dashboard/charts";
import { useInspector } from "@/components/dashboard/inspect-drawer";
import { KpiCards } from "@/components/dashboard/kpi-cards";
import { SessionsTable, type Session } from "@/components/dashboard/sessions-table";
import { getScores, getSummary, type Score, type Status, type Summary } from "@/lib/api";
import { kb, ago } from "@/lib/utils";

const POLL_MS = 3000;
const SUMMARY_DEBOUNCE_MS = 400;
const CRITICAL = 0.9; // the "Critical" chip: score of 90% or more

type Filter = "all" | "anomalous" | "normal" | "critical";
const FILTERS: { id: Filter; label: string }[] = [
  { id: "all", label: "All" },
  { id: "anomalous", label: "Anomalous" },
  { id: "normal", label: "Normal" },
  { id: "critical", label: "Critical ≥90%" },
];

// Repeats `load` every POLL_MS until the effect is cleaned up (or `delay` ms before the first call).
function poll(load: () => Promise<void>, delay = 0) {
  let timer: ReturnType<typeof setTimeout>;
  let stopped = false;
  const tick = async () => {
    await load();
    if (!stopped) timer = setTimeout(tick, POLL_MS);
  };
  timer = setTimeout(tick, delay);
  return () => { stopped = true; clearTimeout(timer); };
}

export function Monitor({ status, threshold, onThreshold, paused, onPause }: {
  status: Status | null; threshold: number; onThreshold: (percent: number) => void; paused: boolean; onPause: () => void;
}) {
  const [scores, setScores] = useState<Score[] | null>(null); // every block; the threshold is applied to these in the browser
  const [serverSummary, setServerSummary] = useState<Summary | null>(null); // only used for "events that stand out", which needs the event lists
  const [filter, setFilter] = useState<Filter>("anomalous");
  const [search, setSearch] = useState("");
  const [error, setError] = useState("");
  const { open, drawer } = useInspector();
  const cut = threshold / 100;

  // Scores: re-fetched every few seconds. Moving the slider does NOT touch the network for these.
  useEffect(() => {
    if (paused) return;
    return poll(async () => {
      try { setScores((await getScores()).scores); setError(""); }
      catch (e) { setError(e instanceof Error ? e.message : "Could not reach the server."); }
    });
  }, [paused]);

  // The events chart is the one thing the server still computes per threshold, so it waits for the slider to settle.
  useEffect(() => {
    if (paused) return;
    return poll(async () => {
      try { setServerSummary(await getSummary(cut)); } catch { /* the scores poll above already reports connection errors */ }
    }, SUMMARY_DEBOUNCE_MS);
  }, [cut, paused]);

  // Everything below recomputes instantly from `scores` + the slider.
  const { rows, summary, kpis } = useMemo(() => {
    const all = scores ?? [];
    const bucket = serverSummary?.bucket_seconds ?? 60;
    const histogram = Array.from({ length: 10 }, (_, bin) => ({ bin, count: 0 }));
    const series = new Map<number, { t: number; normal: number; anomalous: number }>();
    const rows: Session[] = [];
    let anomalous = 0;
    for (const [blockId, score, events, lastSeen] of all) {
      const flagged = score >= cut;
      if (flagged) anomalous++;
      histogram[Math.min(9, Math.floor(score * 10))].count++;
      if (lastSeen !== null) {
        const t = Math.floor(lastSeen / bucket) * bucket;
        const point = series.get(t) ?? { t, normal: 0, anomalous: 0 };
        point[flagged ? "anomalous" : "normal"]++;
        series.set(t, point);
      }
      rows.push({ blockId, anomalous: flagged, confidence: score * 100, events });
    }
    const total = all.length;
    const kpis = { total, anomalous, normal: total - anomalous, rate: total ? (100 * anomalous) / total : 0 };
    const summary: Summary = {
      ...kpis, bucket_seconds: bucket, histogram, top_events: serverSummary?.top_events ?? [],
      series: [...series.values()].sort((a, b) => a.t - b.t).map((p) => ({ ...p, rate: (100 * p.anomalous) / (p.normal + p.anomalous) })),
    };
    return { rows, summary, kpis };
  }, [scores, cut, serverSummary]);

  const visible = useMemo(() => {
    const needle = search.trim().toLowerCase();
    return rows.filter((r) =>
      (!needle || r.blockId.toLowerCase().includes(needle)) &&
      (filter === "all" || (filter === "anomalous" ? r.anomalous : filter === "normal" ? !r.anomalous : r.confidence >= CRITICAL * 100)));
  }, [rows, search, filter]);

  const idleFor = status?.last_ingest ? status.now - status.last_ingest : null;
  const live = !paused && status?.model_loaded && !status.error && !error;

  return (
    <div className="space-y-2">
      {/* Command bar: live state + pause, then search, severity filters and the threshold slider. */}
      <section className="space-y-2 rounded-lg border border-slate-800 bg-slate-900 px-3 py-2 text-xs shadow-sm">
        <div className="flex flex-wrap items-center gap-x-6 gap-y-2">
          <span className="flex items-center gap-2 font-medium text-slate-200">
            <span className={`size-2 rounded-full ${live ? "animate-pulse bg-emerald-400" : "bg-amber-400"}`} />
            {paused ? "PAUSED" : live ? "LIVE" : "STARTING"}
            <span className="font-mono font-normal text-slate-400">{status?.file ?? "…"}</span>
          </span>
          <Stat label="Lines ingested" value={status ? status.lines.toLocaleString() : "–"} />
          <Stat label="Backlog" value={status ? kb(status.backlog_bytes) : "–"} />
          <Stat label="Last line" value={idleFor === null ? "none yet" : ago(idleFor)} />
          <Stat label="Unmatched lines" value={status ? status.unmatched.toLocaleString() : "–"} />
          <button
            type="button" onClick={onPause}
            className="ml-auto inline-flex items-center gap-1 rounded-sm border border-slate-700 px-2 py-0.5 text-slate-300 hover:border-indigo-500 hover:text-slate-100"
          >
            {paused ? <><Play className="size-3" /> Resume</> : <><Pause className="size-3" /> Pause</>}
          </button>
        </div>
        <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
          <label className="relative min-w-40 flex-1 sm:max-w-xs">
            <Search className="pointer-events-none absolute left-2 top-1/2 size-3.5 -translate-y-1/2 text-slate-500" />
            <input
              type="search" value={search} onChange={(e) => setSearch(e.target.value)} placeholder="Search block ID…"
              className="h-7 w-full rounded-sm border border-slate-700 bg-slate-950 pl-7 pr-2 text-slate-200 placeholder:text-slate-500 focus:border-indigo-500 focus:outline-none"
            />
          </label>
          <div className="flex gap-1" role="group" aria-label="Severity filter">
            {FILTERS.map(({ id, label }) => (
              <button
                key={id} type="button" onClick={() => setFilter(id)} aria-pressed={filter === id}
                className={`rounded-sm border px-2 py-0.5 ${filter === id ? "border-indigo-500 text-slate-100" : "border-slate-700 text-slate-400 hover:text-slate-200"}`}
              >
                {label}
              </button>
            ))}
          </div>
          <label className="ml-auto flex items-center gap-2 text-slate-400">
            Anomaly threshold
            <input type="range" min={0} max={100} value={threshold} onChange={(e) => onThreshold(Number(e.target.value))} className="w-36 accent-indigo-500" />
            <span className="w-9 font-mono text-slate-200">{threshold}%</span>
          </label>
        </div>
      </section>

      {(error || status?.error) && (
        <p className="rounded-sm border border-red-900 bg-red-950 px-3 py-2 text-xs text-red-400">{error || status?.error}</p>
      )}

      {/* Until the first scores arrive, the panels pulse as a skeleton. */}
      <div className={`space-y-2 ${scores === null ? "animate-pulse" : ""}`}>
        <KpiCards
          k={kpis}
          last={{ label: "Blocks Tracked", value: status ? status.blocks.toLocaleString() : "–", sub: "distinct HDFS blocks seen so far" }}
        />
        <Charts summary={summary} threshold={cut} />
        <SessionsTable
          title="Flagged Block Sessions"
          rows={visible}
          onInspect={(blockId) => open("monitor", blockId)}
          toolbar={<span className="text-[11px] text-slate-500">showing {visible.length.toLocaleString()} of {rows.length.toLocaleString()} blocks</span>}
        />
      </div>
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
