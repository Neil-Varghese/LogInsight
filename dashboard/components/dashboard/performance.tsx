"use client";

import { Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { AXIS, ChartCard, GRID, INDIGO, TOOLTIP } from "@/components/dashboard/charts";
import type { Status } from "@/lib/api";

const size = (bytes: number) => (bytes < 1048576 ? `${(bytes / 1024).toFixed(0)} KB` : `${(bytes / 1048576).toFixed(1)} MB`);
const duration = (s: number) => (s < 90 ? `${Math.round(s)}s` : s < 5400 ? `${Math.round(s / 60)}m` : `${(s / 3600).toFixed(1)}h`);

// How fast the monitor is working, measured over its last ~200 busy ticks (a tick = one read-match-score-save pass).
export function Performance({ status }: { status: Status | null }) {
  const perf = status?.perf;
  const metric = (label: string, value: string, hint: string) => (
    <div key={label} className="rounded-lg border border-slate-800 bg-slate-900 shadow-sm px-3 py-3">
      <div className="text-[11px] uppercase tracking-wide text-slate-500">{label}</div>
      <div className="font-mono text-lg text-slate-100">{value}</div>
      <div className="text-[11px] text-slate-500">{hint}</div>
    </div>
  );
  return (
    <div className="space-y-2">
      {!perf || perf.ticks === 0 ? (
        <p className="rounded-lg border border-slate-800 bg-slate-900 shadow-sm px-3 py-6 text-center text-xs text-slate-500">
          No lines processed yet. Run the simulation (Simulation tab) and the numbers appear here.
        </p>
      ) : (
        <div className="grid grid-cols-2 gap-2 md:grid-cols-4">
          {metric("Capacity", `${Math.round(perf.capacity_lines_per_s).toLocaleString()} lines/s`, "most it can keep up with")}
          {metric("Tick time (avg)", `${perf.tick_avg_ms.toFixed(0)} ms`, "read + match + score + save")}
          {metric("Tick time (p95)", `${perf.tick_p95_ms.toFixed(0)} ms`, "the slowest 1 in 20")}
          {metric("Scoring", `${perf.score_ms_per_block.toFixed(1)} ms/block`, "LSTM time per block")}
          {metric("Model load", perf.model_load_s === null ? "–" : `${perf.model_load_s.toFixed(1)} s`, "one-off at start-up")}
          {metric("Database", size(perf.db_bytes), "monitor.sqlite3")}
          {metric("Uptime", duration(perf.uptime_s), "since the server started")}
          {metric("Ticks sampled", String(perf.ticks), "busy ticks, last 200")}
        </div>
      )}
      <ChartCard title="Ingest throughput" note="lines per second" empty={!status || status.rates.length === 0}>
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={(status?.rates ?? []).map((r) => ({ t: r.t, rate: r.rate }))} margin={{ top: 4, right: 8, left: -12, bottom: 0 }}>
            <CartesianGrid stroke={GRID} vertical={false} />
            <XAxis dataKey="t" tickFormatter={(t) => new Date(t * 1000).toLocaleTimeString([], { minute: "2-digit", second: "2-digit" })} tick={AXIS} stroke={GRID} minTickGap={32} />
            <YAxis tick={AXIS} stroke={GRID} allowDecimals={false} />
            <Tooltip {...TOOLTIP} labelFormatter={(t) => new Date(Number(t) * 1000).toLocaleTimeString()} />
            <Area type="monotone" dataKey="rate" name="Lines/s" stroke={INDIGO} fill={INDIGO} fillOpacity={0.15} isAnimationActive={false} />
          </AreaChart>
        </ResponsiveContainer>
      </ChartCard>
      <p className="text-[11px] text-slate-500">
        If the ingest throughput stays below Capacity, the monitor keeps up with the log. If it gets close, the Backlog on the Monitor tab starts to grow.
      </p>
    </div>
  );
}
