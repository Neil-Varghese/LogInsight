"use client";

import { useEffect, useState } from "react";
import { Area, AreaChart, Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { AXIS, ChartCard, GRID, INDIGO, TOOLTIP, logTime } from "@/components/dashboard/charts";
import { LiveFeed } from "@/components/dashboard/live-feed";
import { LogSearch } from "@/components/dashboard/search";
import { getTraffic,type Traffic as TrafficData } from "@/lib/api";

const mb = (bytes: number) => Math.round((bytes / 1048576) * 100) / 100; // real HDFS samples are ~90 KB a block, so keep 2 decimals
const size = (bytes: number) => (bytes >= 1073741824 ? `${(bytes / 1073741824).toFixed(2)} GB` : `${mb(bytes).toLocaleString()} MB`);

// Network traffic = HDFS "Received block ... of size N from /ip" lines: data copied between machines.
function TrafficCharts({ traffic }: { traffic: TrafficData | null }) {
  const empty = !traffic || traffic.transfers === 0;
  const bucket = traffic?.bucket_seconds ?? 60;
  return (
    <div className="grid gap-2 lg:grid-cols-3">
      <ChartCard
        title="Network traffic"
        note={empty ? "block copies between machines" : `${size(traffic.total_bytes)} in ${traffic.transfers.toLocaleString()} block transfers, ${bucket}s buckets`}
        empty={empty} className="lg:col-span-2"
      >
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={(traffic?.series ?? []).map((p) => ({ t: p.t, mb: mb(p.bytes) }))} margin={{ top: 4, right: 8, left: -12, bottom: 0 }}>
            <CartesianGrid stroke={GRID} vertical={false} />
            <XAxis dataKey="t" tickFormatter={(t) => logTime(t, bucket)} tick={AXIS} stroke={GRID} minTickGap={24} />
            <YAxis tick={AXIS} stroke={GRID} unit=" MB" />
            <Tooltip {...TOOLTIP} labelFormatter={(t) => logTime(Number(t), bucket)} formatter={(v) => `${Number(v).toLocaleString()} MB`} />
            <Area type="monotone" dataKey="mb" name="Transferred" stroke={INDIGO} fill={INDIGO} fillOpacity={0.2} isAnimationActive={false} />
          </AreaChart>
        </ResponsiveContainer>
      </ChartCard>

      <ChartCard title="Top sending machines" note="MB sent" empty={empty}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={(traffic?.top_sources ?? []).map((s) => ({ ip: s.ip, mb: mb(s.bytes) }))} layout="vertical" margin={{ top: 4, right: 16, left: 0, bottom: 0 }}>
            <CartesianGrid stroke={GRID} horizontal={false} />
            <XAxis type="number" tick={AXIS} stroke={GRID} />
            <YAxis type="category" dataKey="ip" width={92} tick={{ ...AXIS, fontSize: 10 }} stroke={GRID} />
            <Tooltip {...TOOLTIP} formatter={(v) => `${Number(v).toLocaleString()} MB`} />
            <Bar dataKey="mb" name="Sent" fill={INDIGO} isAnimationActive={false} />
          </BarChart>
        </ResponsiveContainer>
      </ChartCard>
    </div>
  );
}

const POLL_MS = 3000;

// The Traffic tab: search box (closed until clicked), the live log stream, then the network charts, which refresh every few seconds.
export function Traffic({ initialIp, threshold }: { initialIp: string; threshold: number }) {
  const [traffic, setTraffic] = useState<TrafficData | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout>;
    async function refresh() {
      try {
        const next = await getTraffic();
        if (!cancelled) { setTraffic(next); setError(""); }
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : "Could not reach the server.");
      }
      if (!cancelled) timer = setTimeout(refresh, POLL_MS);
    }
    refresh();
    return () => { cancelled = true; clearTimeout(timer); };
  }, []);

  return (
    <div className="space-y-2">
      {error && <p className="border border-red-900 bg-red-950 px-3 py-2 text-xs text-red-400">{error}</p>}
      <LogSearch initialIp={initialIp} threshold={threshold} />
      <LiveFeed />
      <TrafficCharts traffic={traffic} />
    </div>
  );
}
