import { Activity } from "lucide-react";

export function TopNav({ threshold, online }: { threshold: number; online: boolean | null }) {
  return (
    <header className="flex h-11 items-center justify-between border-b border-slate-800 bg-slate-900 px-4">
      <div className="flex items-center gap-2">
        <Activity className="size-4 text-indigo-400" />
        <span className="text-sm font-semibold text-slate-100">LogInsight</span>
        <span className="text-xs text-slate-500">: LSTM Anomaly Detector</span>
      </div>
      <div className="flex items-center gap-4 text-xs">
        <span className="text-slate-400">
          Anomaly Threshold: <span className="font-mono text-slate-200">{threshold}%</span>
        </span>
        {/* ponytail: "online" = the backend answered; no model health endpoint yet */}
        <span className={`flex items-center gap-1.5 rounded-sm border px-2 py-0.5 font-medium ${online === false ? "border-red-900 bg-red-950 text-red-400" : "border-emerald-800 bg-emerald-950 text-emerald-400"}`}>
          <span className={`size-1.5 rounded-full ${online === false ? "bg-red-400" : "bg-emerald-400"}`} />
          {online === false ? "Backend Offline" : "LSTM Keras Model Online"}
        </span>
      </div>
    </header>
  );
}
