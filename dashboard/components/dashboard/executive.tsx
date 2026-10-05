"use client";

import { useEffect, useState, type ReactNode } from "react";
import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { AXIS, GRID, INDIGO, TOOLTIP } from "@/components/dashboard/charts";
import { getFeed, getIps, getSummary, getTraffic, type Feed, type Ips, type Summary, type Traffic } from "@/lib/api";
import { TIER_ORDER, tierOf } from "@/lib/severity";

// Goals the "Target" and "Gap" lines compare against. Edit these to your own service levels.
const TARGETS = { anomalyRatePct: 5, criticalLines: 0 };
const TIER_COLOR: Record<number, string> = { 2: "#c62d2d", 3: "#e07a2f", 4: "#f5b94a", 5: "#7e57c2", 6: "#2f8ac6", 7: "#9aa7b3" };
const POLL_MS = 3000;
// Two short lines per priority: what it means, and what to do.
const TIER_HELP: Record<number, string> = {
  2: "The system is failing or about to. Act immediately.",
  3: "An operation failed, such as a block write. Fix within the day.",
  4: "Something odd that has not broken anything yet. Check soon.",
  5: "A log level we don't recognise. Never ignored: have a look.",
  6: "Normal activity, such as a block received. No action needed.",
  7: "Developer detail for troubleshooting. Usually safe to ignore.",
};

const mb = (bytes: number) => Math.round((bytes / 1048576) * 100) / 100;
const size = (bytes: number) => (bytes >= 1073741824 ? `${(bytes / 1073741824).toFixed(2)} GB` : `${mb(bytes).toLocaleString()} MB`);

// One column, CISO-dashboard style: blue title band with its question, two headline numbers, then charts.
function Column({ title, question, children }: { title: string; question: string; children: ReactNode }) {
  return (
    <section className="overflow-hidden rounded-lg border border-slate-800 bg-slate-900 shadow-sm">
      <div className="bg-[#1e2d4d] [html.light_&]:bg-[#bcd8ee] px-3 py-2">
        <h2 className="border-b border-slate-700 pb-1 text-[clamp(0.9rem,1vw,1rem)] font-semibold text-slate-100">{title}</h2>
        <p className="pt-1 text-xs text-slate-200">{question}</p>
      </div>
      {children}
    </section>
  );
}

// Headline number; bad = true paints it red. `target` adds the "Target / Gap" lines (gap = value - target).
function Kpi({ label, value, bad, target, gap, note }: {
  label: string; value: string; bad?: boolean; target?: string; gap?: string; note?: string;
}) {
  return (
    <div className="min-w-0">
      <div className="min-h-7 text-xs font-medium text-slate-200">{label}</div>
      <div className={`mt-1 text-[clamp(1.25rem,1.5vw,1.65rem)] ${bad ? "text-red-400" : "text-slate-100"}`}>{value}</div>
      {note && <div className="mt-1 text-[11px] text-slate-500">{note}</div>}
      {target && (
        <div className="mt-2 text-[11px] text-slate-400">
          <div>Target {target}</div>
          <div>Gap {gap}</div>
        </div>
      )}
    </div>
  );
}

const Kpis = ({ children }: { children: ReactNode }) => <div className="grid min-h-[clamp(6rem,11vh,8.5rem)] grid-cols-2 gap-2.5 p-2.5">{children}</div>;

function Sub({ title, empty, children }: { title: string; empty?: boolean; children: ReactNode }) {
  return (
    <div className="border-t border-slate-800 p-3">
      <h3 className="text-xs font-semibold text-slate-200">{title}</h3>
      <div className="mt-2 h-[clamp(9rem,17vh,12rem)]">
        {empty ? <div className="flex h-full items-center justify-center text-xs text-slate-500">Waiting for log lines…</div> : children}
      </div>
    </div>
  );
}

const signed = (n: number, digits = 0) => `${n > 0 ? "+" : ""}${n.toFixed(digits)}`;

export function Executive({ threshold }: { threshold: number }) {
  const [summary, setSummary] = useState<Summary | null>(null);
  const [traffic, setTraffic] = useState<Traffic | null>(null);
  const [ips, setIps] = useState<Ips | null>(null);
  const [feed, setFeed] = useState<Feed | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout>;
    async function refresh() {
      try {
        // after = a huge number asks the feed for its counts only, no lines
        const [s, t, i, f] = await Promise.all([getSummary(threshold / 100), getTraffic(), getIps(), getFeed(1e12)]);
        if (!cancelled) { setSummary(s); setTraffic(t); setIps(i); setFeed(f); setError(""); }
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : "Could not reach the server.");
      }
      if (!cancelled) timer = setTimeout(refresh, POLL_MS);
    }
    refresh();
    return () => { cancelled = true; clearTimeout(timer); };
  }, [threshold]);

  const count = (sev: number) => feed?.counts[String(sev)] ?? 0;
  const urgent = count(2) + count(3); // Critical + Error
  const rate = summary?.rate ?? 0;
  const noTraffic = !traffic || traffic.transfers === 0;
  const topIps = (ips?.ips ?? []).slice(0, 6).map((r) => ({ ip: r.ip, lines: r.lines }));

  return (
    <div className="space-y-2">
      {error && <p className="rounded-lg border border-red-900 bg-red-950 px-3 py-2 text-xs text-red-400">{error}</p>}
      <div className="grid gap-2 md:grid-cols-2 xl:grid-cols-4">
        <Column title="Detection" question="Is the model catching problems?">
          <Kpis>
            <Kpi label="Anomalous blocks" value={(summary?.anomalous ?? 0).toLocaleString()} bad={(summary?.anomalous ?? 0) > 0} note={`of ${(summary?.total ?? 0).toLocaleString()} blocks`} />
            <Kpi
              label="Anomaly rate" value={`${rate.toFixed(2)}%`} bad={rate > TARGETS.anomalyRatePct}
              target={`${TARGETS.anomalyRatePct}%`} gap={`${signed(rate - TARGETS.anomalyRatePct, 2)}%`}
            />
          </Kpis>
          <Sub title="Blocks by outcome" empty={!summary || summary.total === 0}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={[{ name: "Normal", n: summary?.normal ?? 0 }, { name: "Anomalous", n: summary?.anomalous ?? 0 }]} layout="vertical" margin={{ top: 4, right: 16, left: 0, bottom: 0 }}>
                <CartesianGrid stroke={GRID} horizontal={false} />
                <XAxis type="number" tick={AXIS} stroke={GRID} allowDecimals={false} />
                <YAxis type="category" dataKey="name" width={70} tick={AXIS} stroke={GRID} />
                <Tooltip {...TOOLTIP} />
                <Bar dataKey="n" name="Blocks" isAnimationActive={false}>
                  <Cell fill="#5aa469" /><Cell fill="#d9534f" />
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Sub>
        </Column>

        <Column title="Threat Levels" question="How serious are the log lines?">
          <Kpis>
            <Kpi label="Critical + Error lines" value={urgent.toLocaleString()} bad={urgent > TARGETS.criticalLines} target={String(TARGETS.criticalLines)} gap={signed(urgent - TARGETS.criticalLines)} />
            <Kpi label="Warnings" value={count(4).toLocaleString()} note="P3 lines" />
          </Kpis>
          <Sub title="Lines by priority" empty={!feed || TIER_ORDER.every((s) => count(s) === 0)}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={TIER_ORDER.map((s) => ({ name: `${tierOf(s).priority} ${tierOf(s).label}`, n: count(s), sev: s }))} layout="vertical" margin={{ top: 4, right: 16, left: 0, bottom: 0 }}>
                <CartesianGrid stroke={GRID} horizontal={false} />
                <XAxis type="number" tick={AXIS} stroke={GRID} allowDecimals={false} />
                <YAxis type="category" dataKey="name" width={96} tick={{ ...AXIS, fontSize: 10 }} stroke={GRID} />
                <Tooltip {...TOOLTIP} />
                <Bar dataKey="n" name="Lines" isAnimationActive={false}>
                  {TIER_ORDER.map((s) => <Cell key={s} fill={TIER_COLOR[s]} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Sub>
        </Column>

        <Column title="Traffic" question="Is data moving normally?">
          <Kpis>
            <Kpi label="Data transferred" value={size(traffic?.total_bytes ?? 0)} note="between machines" />
            <Kpi label="Block transfers" value={(traffic?.transfers ?? 0).toLocaleString()} />
          </Kpis>
          <Sub title="Top sending machines (MB)" empty={noTraffic}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={(traffic?.top_sources ?? []).map((s) => ({ ip: s.ip, mb: mb(s.bytes) }))} layout="vertical" margin={{ top: 4, right: 16, left: 0, bottom: 0 }}>
                <CartesianGrid stroke={GRID} horizontal={false} />
                <XAxis type="number" tick={AXIS} stroke={GRID} />
                <YAxis type="category" dataKey="ip" width={92} tick={{ ...AXIS, fontSize: 10 }} stroke={GRID} />
                <Tooltip {...TOOLTIP} />
                <Bar dataKey="mb" name="Sent (MB)" fill={INDIGO} isAnimationActive={false} />
              </BarChart>
            </ResponsiveContainer>
          </Sub>
        </Column>

        <Column title="Hosts" question="Who is talking right now?">
          <Kpis>
            <Kpi label="Active IPs" value={(ips?.active_count ?? 0).toLocaleString()} note={ips ? `in the last ${Math.round(ips.window / 60)} min` : undefined} />
            <Kpi label="IPs seen" value={(ips?.ips.length ?? 0).toLocaleString()} note="since start" />
          </Kpis>
          <Sub title="Busiest IPs (lines)" empty={topIps.length === 0}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={topIps} layout="vertical" margin={{ top: 4, right: 16, left: 0, bottom: 0 }}>
                <CartesianGrid stroke={GRID} horizontal={false} />
                <XAxis type="number" tick={AXIS} stroke={GRID} allowDecimals={false} />
                <YAxis type="category" dataKey="ip" width={92} tick={{ ...AXIS, fontSize: 10 }} stroke={GRID} />
                <Tooltip {...TOOLTIP} />
                <Bar dataKey="lines" name="Lines" fill="#a62a6c" isAnimationActive={false} />
              </BarChart>
            </ResponsiveContainer>
          </Sub>
        </Column>

      </div>
      <section className="rounded-lg border border-slate-800 bg-slate-900 p-3 shadow-sm">
        <h2 className="text-sm font-semibold text-slate-100">Priority levels</h2>
        <p className="text-xs text-slate-500">Log levels ranked by threat to the system (RFC 5424 syslog severity). P1 is the most urgent.</p>
        <div className="mt-3 grid gap-2 sm:grid-cols-2 xl:grid-cols-6">
          {TIER_ORDER.map((s) => (
            <div key={s} className="rounded-md border border-slate-800 p-2.5">
              <span className={`inline-block rounded-sm border px-1.5 py-0.5 text-[11px] font-semibold ${tierOf(s).badge}`}>{tierOf(s).priority} {tierOf(s).label}</span>
              <p className="mt-1.5 text-xs text-slate-300">{TIER_HELP[s]}</p>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
