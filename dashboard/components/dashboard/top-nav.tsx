import { Activity, Globe, Network, Play, Settings, ShieldCheck, Star } from "lucide-react";
import type { ComponentType } from "react";

export type View = "executive" | "monitor" | "simulation" | "traffic" | "ips" | "settings";
export type Health = "online" | "loading" | "error" | "offline" | null;

// Each section: its icon on the left rail and the name shown on the tab above the page.
const SECTIONS: { id: View; label: string; Icon: ComponentType<{ className?: string }> }[] = [
  { id: "executive", label: "Executive", Icon: ShieldCheck },
  { id: "monitor", label: "Monitor", Icon: Activity },
  { id: "simulation", label: "Simulation", Icon: Play },
  { id: "traffic", label: "Traffic", Icon: Network },
  { id: "ips", label: "Active IPs", Icon: Globe },
  { id: "settings", label: "Settings", Icon: Settings },
];

const BADGE: Record<Exclude<Health, null>, { text: string; box: string; dot: string }> = {
  online: { text: "LSTM Keras Model Online", box: "border-emerald-800 bg-emerald-950 text-emerald-400", dot: "bg-emerald-400" },
  loading: { text: "LSTM Model Loading…", box: "border-amber-800 bg-amber-950 text-amber-400", dot: "bg-amber-400" },
  error: { text: "Monitor Error", box: "border-red-900 bg-red-950 text-red-400", dot: "bg-red-400" },
  offline: { text: "Backend Offline", box: "border-red-900 bg-red-950 text-red-400", dot: "bg-red-400" },
};

// Dark top bar (ServiceNow style): name, workspace pill, threshold and model status.
export function TopNav({ threshold, health }: { threshold: number; health: Health }) {
  const badge = BADGE[health ?? "loading"];
  return (
    <header className="flex h-11 items-center justify-between gap-3 bg-[#161d27] px-4 text-slate-300">
      <span className="text-lg font-semibold tracking-tight text-white">LogInsight</span>
      <span className="hidden items-center gap-1.5 rounded-full border border-slate-600 px-4 py-1 text-xs font-semibold text-slate-200 md:flex">
        LSTM Anomaly Workspace <Star className="size-3" />
      </span>
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

// Left icon rail: one button per section; the open one gets a ring, like the ServiceNow workspace.
export function SideRail({ view, onView }: { view: View; onView: (view: View) => void }) {
  return (
    <nav aria-label="Sections" className="flex w-12 shrink-0 flex-col items-center gap-2 bg-gradient-to-b from-[#2b2f8f] to-[#14121f] py-3">
      {SECTIONS.map(({ id, label, Icon }) => (
        <button
          key={id}
          type="button"
          title={label}
          aria-label={label}
          aria-current={view === id ? "page" : undefined}
          onClick={() => onView(id)}
          className={`grid size-9 place-items-center rounded-lg border transition-colors ${
            view === id ? "border-white bg-white text-[#2b2f8f] shadow-md" : "border-white/30 bg-white/15 text-white hover:bg-white/30"
          }`}
        >
          <Icon className="size-[18px]" />
        </button>
      ))}
    </nav>
  );
}

// Grey strip above the page holding the open section's name, like the "BOM Queue" tab.
export function TabStrip({ view }: { view: View }) {
  const { label, Icon } = SECTIONS.find((s) => s.id === view)!;
  return (
    <div className="flex h-9 items-end bg-slate-800 px-2">
      <span className="flex h-8 items-center gap-1.5 rounded-t-md bg-slate-950 px-3 text-xs font-semibold text-slate-100">
        <Icon className="size-3.5" /> {label}
      </span>
    </div>
  );
}
