import { AlertTriangle, CheckCircle2, Gauge, Layers, Timer } from "lucide-react";
import type { ReactNode } from "react";

export type Kpis = { total: number; anomalous: number; normal: number; rate: number };
export type LastCard = { label: string; value: string; sub?: string };

function Card({ label, value, icon, tone = "text-slate-100", sub }: {
  label: string; value: string; icon: ReactNode; tone?: string; sub?: string;
}) {
  return (
    <div className="border border-slate-800 bg-slate-900 px-3 py-2.5">
      <div className="flex items-center justify-between text-xs text-slate-400">
        <span>{label}</span>
        {icon}
      </div>
      <div className={`mt-1 font-mono text-2xl font-semibold tabular-nums ${tone}`}>{value}</div>
      {sub && <div className="mt-0.5 text-[11px] text-slate-500">{sub}</div>}
    </div>
  );
}

export function KpiCards({ k, last }: { k: Kpis; last: LastCard }) {
  return (
    <div className="grid grid-cols-2 gap-2 lg:grid-cols-5">
      <Card label="Total Block Sessions" value={k.total.toLocaleString()} icon={<Layers className="size-3.5" />} />
      <Card label="Anomalous Blocks" value={k.anomalous.toLocaleString()} tone="text-red-400" icon={<AlertTriangle className="size-3.5 text-red-400" />} />
      <Card label="Normal Blocks" value={k.normal.toLocaleString()} tone="text-emerald-400" icon={<CheckCircle2 className="size-3.5 text-emerald-400" />} />
      <Card label="Anomaly Rate" value={`${k.rate.toFixed(2)}%`} icon={<Gauge className="size-3.5" />} />
      <Card label={last.label} value={last.value} icon={<Timer className="size-3.5" />} sub={last.sub} />
    </div>
  );
}
