"use client";

import type { ReactNode } from "react";
import { Bar, BarChart, Brush, CartesianGrid, Cell, ComposedChart, Legend, Line, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { Summary } from "@/lib/api";

const RED = "#d9534f";
const GREEN = "#5aa469";
export const INDIGO = "#2f8ac6";
// Theme-dependent colours come from CSS variables in globals.css, so charts follow the dark/light toggle.
export const GRID = "var(--chart-grid)";
export const AXIS = { fontSize: 11, fill: "var(--chart-axis)" };
export const TOOLTIP = {
  contentStyle: { background: "var(--chart-tip-bg)", border: "1px solid var(--chart-tip-border)", borderRadius: 6, fontSize: 12, color: "var(--chart-tip-text)" },
  labelStyle: { color: "var(--chart-tip-text)" },
  cursor: { fill: "rgba(47,138,198,0.08)" },
};

export function ChartCard({ title, note, className = "", empty, children }: {
  title: string; note?: string; className?: string; empty?: boolean; children: ReactNode;
}) {
  return (
    <section className={`rounded-lg border border-slate-800 bg-slate-900 shadow-sm ${className}`}>
      <div className="flex items-baseline justify-between border-b border-slate-800 px-3 py-2">
        <h2 className="text-sm font-semibold text-slate-100">{title}</h2>
        {note && <span className="text-[11px] text-slate-500">{note}</span>}
      </div>
      <div className="h-56 p-2">
        {empty ? <div className="flex h-full items-center justify-center text-xs text-slate-500">Waiting for log lines…</div> : children}
      </div>
    </section>
  );
}

// Log timestamps are UTC (HDFS logs carry no timezone).
export function logTime(t: number, bucket: number) {
  const iso = new Date(t * 1000).toISOString();
  if (bucket < 60) return iso.slice(11, 19);
  if (bucket < 3600) return iso.slice(11, 16);
  return bucket < 86400 ? `${iso.slice(5, 10)} ${iso.slice(11, 16)}` : iso.slice(5, 10);
}

const short = (text: string, max: number) => (text.length > max ? `${text.slice(0, max - 1)}…` : text);

export function Charts({ summary, threshold }: { summary: Summary | null; threshold: number }) {
  const empty = !summary || summary.total === 0;
  const bucket = summary?.bucket_seconds ?? 60;
  return (
    <div className="grid gap-2 lg:grid-cols-3">
      <ChartCard title="Blocks over time" note={`log time (UTC), ${bucket}s buckets`} empty={empty} className="lg:col-span-2">
        <ResponsiveContainer width="100%" height="100%">
          {/* Left axis = how many blocks arrived, right axis = what share of them were anomalous. Drag the strip at the bottom to zoom. */}
          <ComposedChart data={summary?.series ?? []} margin={{ top: 4, right: 0, left: -12, bottom: 0 }}>
            <CartesianGrid stroke={GRID} vertical={false} />
            <XAxis dataKey="t" tickFormatter={(t) => logTime(t, bucket)} tick={AXIS} stroke={GRID} minTickGap={24} />
            <YAxis yAxisId="vol" tick={AXIS} stroke={GRID} allowDecimals={false} />
            <YAxis yAxisId="pct" orientation="right" domain={[0, 100]} unit="%" tick={AXIS} stroke={GRID} width={40} />
            <Tooltip {...TOOLTIP} labelFormatter={(t) => logTime(Number(t), bucket)} formatter={(v, name) => (name === "Anomaly %" ? `${Number(v).toFixed(1)}%` : v)} />
            <Legend wrapperStyle={{ fontSize: 11 }} />
            <Bar yAxisId="vol" dataKey="normal" name="Normal" stackId="a" fill={GREEN} fillOpacity={0.7} isAnimationActive={false} />
            <Bar yAxisId="vol" dataKey="anomalous" name="Anomalous" stackId="a" fill={RED} isAnimationActive={false} />
            <Line yAxisId="pct" dataKey="rate" name="Anomaly %" stroke="#f5b94a" strokeWidth={1.5} dot={false} isAnimationActive={false} />
            <Brush dataKey="t" height={18} stroke={GRID} fill="transparent" tickFormatter={(t) => logTime(t, bucket)} />
          </ComposedChart>
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
              {(summary?.histogram ?? []).map((h) => <Cell key={h.bin} fill={h.bin / 10 >= threshold ? RED : "#9aa7b3"} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </ChartCard>

      <ChartCard title="Events that stand out in flagged blocks" note="% of blocks containing the event" empty={!summary || summary.top_events.length === 0} className="lg:col-span-3">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={summary?.top_events ?? []} layout="vertical" margin={{ top: 4, right: 16, left: 0, bottom: 0 }}>
            <CartesianGrid stroke={GRID} horizontal={false} />
            <XAxis type="number" domain={[0, 100]} tick={AXIS} stroke={GRID} unit="%" />
            <YAxis type="category" dataKey="template" width={210} tick={{ ...AXIS, fontSize: 10 }} stroke={GRID} tickFormatter={(v) => short(String(v), 36)} />
            <Tooltip {...TOOLTIP} formatter={(v) => `${Number(v).toFixed(1)}%`} />
            <Legend wrapperStyle={{ fontSize: 11 }} />
            <Bar dataKey="anomalous_pct" name="Flagged blocks" fill={RED} isAnimationActive={false} />
            <Bar dataKey="normal_pct" name="Normal blocks" fill="#9aa7b3" isAnimationActive={false} />
          </BarChart>
        </ResponsiveContainer>
      </ChartCard>
    </div>
  );
}
