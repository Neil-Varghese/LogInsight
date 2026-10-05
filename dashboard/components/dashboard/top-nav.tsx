import { Activity } from "lucide-react";

export type View = "monitor" | "settings";
export type Health = "online" | "loading" | "error" | "offline" | null;

const BADGE: Record<Exclude<Health, null>, { text: string; box: string; dot: string }> = {
  online: { text: "LSTM Keras Model Online", box: "border-emerald-800 bg-emerald-950 text-emerald-400", dot: "bg-emerald-400" },
  loading: { text: "LSTM Model Loading…", box: "border-amber-800 bg-amber-950 text-amber-400", dot: "bg-amber-400" },
  error: { text: "Monitor Error", box: "border-red-900 bg-red-950 text-red-400", dot: "bg-red-400" },
  offline: { text: "Backend Offline", box: "border-red-900 bg-red-950 text-red-400", dot: "bg-red-400" },
};

export function TopNav({ threshold, health, view, onView }: {
  threshold: number; health: Health; view: View; onView: (view: View) => void;
}) {
  const badge = BADGE[health ?? "loading"];
  const tab = (id: View, label: string) => (
    <button
      type="button"
      onClick={() => onView(id)}
      className={`h-11 border-b-2 px-3 text-xs font-medium transition-colors ${
        view === id ? "border-indigo-500 text-slate-100" : "border-transparent text-slate-400 hover:text-slate-200"
      }`}
    >
      {label}
    </button>
  );
  return (
    <header className="flex h-11 items-center justify-between border-b border-slate-800 bg-slate-900 px-4">
      <div className="flex items-center gap-6">
        <div className="flex items-center gap-2">
          <Activity className="size-4 text-indigo-400" />
          <span className="text-sm font-semibold text-slate-100">LogInsight</span>
          <span className="hidden text-xs text-slate-500 sm:inline">: LSTM Anomaly Detector</span>
        </div>
        <nav className="flex">{tab("monitor", "Monitor")}{tab("settings", "Settings")}</nav>
      </div>
      <div className="flex items-center gap-4 text-xs">
        <span className="hidden text-slate-400 sm:inline">
          Anomaly Threshold: <span className="font-mono text-slate-200">{threshold}%</span>
        </span>
        {/* "online" = the monitor thread loaded the model and is running; the badge follows the real status */}
        <span className={`flex items-center gap-1.5 rounded-sm border px-2 py-0.5 font-medium ${badge.box}`}>
          <span className={`size-1.5 rounded-full ${badge.dot}`} />
          {badge.text}
        </span>
      </div>
    </header>
  );
}
