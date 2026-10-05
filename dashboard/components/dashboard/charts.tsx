"use client";

import type { ReactNode } from "react";
import { Area, AreaChart, Bar, BarChart, CartesianGrid, Cell, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { Status, Summary } from "@/lib/api";

const RED = "#f87171";
const GREEN = "#34d399";
const INDIGO = "#818cf8";
const GRID = "#1e293b";
const AXIS = { fontSize: 11, fill: "#94a3b8" };
const TOOLTIP = {
  contentStyle: { background: "#0f172a", border: "1px solid #1e293b", borderRadius: 2, fontSize: 12 },
  labelStyle: { color: "#cbd5e1" },
  cursor: { fill: "rgba(148,163,184,0.08)" },
};

function ChartCard({ title, note, className = "", empty, children }: {
  title: string; note?: string; className?: string; empty?: boolean; children: ReactNode;
}) {
  return (
    <section className={`border border-slate-800 bg-slate-900 ${className}`}>
      <div className="flex items-baseline justify-between border-b border-slate-800 px-3 py-2">
        <h2 className="text-xs font-semibold uppercase tracking-wide text-slate-400">{title}</h2>
        {note && <span className="text-[11px] text-slate-500">{note}</span>}
      </div>
      <div className="h-52 p-2">
        {empty ? <div className="flex h-full items-center justify-center text-xs text-slate-500">Waiting for log lines…</div> : children}
      </div>
    </section>
  );
}

// Log timestamps are UTC (HDFS logs carry no timezone).
function logTime(t: number, bucket: number) {
  const iso = new Date(t * 1000).toISOString();
  if (bucket < 60) return iso.slice(11, 19);
  if (bucket < 3600) return iso.slice(11, 16);
  return bucket < 86400 ? `${iso.slice(5, 10)} ${iso.slice(11, 16)}` : iso.slice(5, 10);
}

const short = (text: string, max: number) => (text.length > max ? `${text.slice(0, max - 1)}…` : text);

export function Charts({ summary, status, threshold }: { summary: Summary | null; status: Status | null; threshold: number }) {
  const empty = !summary || summary.total === 0;
  const bucket = summary?.bucket_seconds ?? 60;
  const rates = (status?.rates ?? []).map((r) => ({ t: r.t, rate: r.rate }));
  return (
    <div className="grid gap-2 lg:grid-cols-3">
      <ChartCard title="Blocks over time" note={`log time (UTC), ${bucket}s buckets`} empty={empty} className="lg:col-span-2">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={summary?.series ?? []} margin={{ top: 4, right: 8, left: -12, bottom: 0 }}>
            <CartesianGrid stroke={GRID} vertical={false} />
            <XAxis dataKey="t" tickFormatter={(t) => logTime(t, bucket)} tick={AXIS} stroke={GRID} minTickGap={24} />
            <YAxis tick={AXIS} stroke={GRID} allowDecimals={false} />
            <Tooltip {...TOOLTIP} labelFormatter={(t) => logTime(Number(t), bucket)} />
            <Legend wrapperStyle={{ fontSize: 11 }} />
            <Bar dataKey="normal" name="Normal" stackId="a" fill={GREEN} fillOpacity={0.7} isAnimationActive={false} />
            <Bar dataKey="anomalous" name="Anomalous" stackId="a" fill={RED} isAnimationActive={false} />
          </BarChart>
        </ResponsiveContainer>
      </ChartCard>

      <ChartCard title="Anomaly score distribution" note="sqrt scale" empty={empty}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={(summary?.histogram ?? []).map((h) => ({ ...h, label: `${h.bin * 10}-${h.bin * 10 + 10}%` }))}
            margin={{ top: 4, right: 8, left: -12, bottom: 0 }}
          >
            <CartesianGrid stroke={GRID} vertical={false} />
            <XAxis dataKey="label" tick={{ ...AXIS, fontSize: 9 }} stroke={GRID} interval={0} angle={-35} textAnchor="end" height={44} />
            <YAxis tick={AXIS} stroke={GRID} scale="sqrt" allowDecimals={false} />
            <Tooltip {...TOOLTIP} />
            <Bar dataKey="count" name="Blocks" isAnimationActive={false}>
              {(summary?.histogram ?? []).map((h) => <Cell key={h.bin} fill={h.bin / 10 >= threshold ? RED : "#64748b"} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </ChartCard>

      <ChartCard title="Ingest throughput" note="lines per second">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={rates} margin={{ top: 4, right: 8, left: -12, bottom: 0 }}>
            <CartesianGrid stroke={GRID} vertical={false} />
            <XAxis dataKey="t" tickFormatter={(t) => new Date(t * 1000).toLocaleTimeString([], { minute: "2-digit", second: "2-digit" })} tick={AXIS} stroke={GRID} minTickGap={32} />
            <YAxis tick={AXIS} stroke={GRID} allowDecimals={false} />
            <Tooltip {...TOOLTIP} labelFormatter={(t) => new Date(Number(t) * 1000).toLocaleTimeString()} />
            <Area type="monotone" dataKey="rate" name="Lines/s" stroke={INDIGO} fill={INDIGO} fillOpacity={0.15} isAnimationActive={false} />
          </AreaChart>
        </ResponsiveContainer>
      </ChartCard>

      <ChartCard title="Events that stand out in flagged blocks" note="% of blocks containing the event" empty={!summary || summary.top_events.length === 0} className="lg:col-span-2">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={summary?.top_events ?? []} layout="vertical" margin={{ top: 4, right: 16, left: 0, bottom: 0 }}>
            <CartesianGrid stroke={GRID} horizontal={false} />
            <XAxis type="number" domain={[0, 100]} tick={AXIS} stroke={GRID} unit="%" />
            <YAxis type="category" dataKey="template" width={210} tick={{ ...AXIS, fontSize: 10 }} stroke={GRID} tickFormatter={(v) => short(String(v), 36)} />
            <Tooltip {...TOOLTIP} formatter={(v) => `${Number(v).toFixed(1)}%`} />
            <Legend wrapperStyle={{ fontSize: 11 }} />
            <Bar dataKey="anomalous_pct" name="Flagged blocks" fill={RED} isAnimationActive={false} />
            <Bar dataKey="normal_pct" name="Normal blocks" fill="#64748b" isAnimationActive={false} />
          </BarChart>
        </ResponsiveContainer>
      </ChartCard>
    </div>
  );
}
