import { Activity, Globe, LogOut, Moon, Network, Pause, Play, Settings, ShieldCheck, Sun } from "lucide-react";
import type { ComponentType } from "react";

export type View = "executive" | "monitor" | "simulation" | "traffic" | "ips" | "settings";
export type Health = "online" | "loading" | "error" | "offline" | null;

// Each section: its icon on the left rail and the name shown on the tab above the page.
const SECTIONS: { id: View; label: string; Icon: ComponentType<{ className?: string }> }[] = [
  { id: "monitor", label: "Monitor", Icon: Activity },
  { id: "executive", label: "Executive", Icon: ShieldCheck },
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

export type Telemetry = { linesPerSec: number | null; backlog: string; lastLine: string; latencyMs: number | null };

function Chip({ label, value, tone = "text-slate-200" }: { label: string; value: string; tone?: string }) {
  return (
    <span className="flex items-baseline gap-1 text-slate-500">
      {label} <span className={`font-mono tabular-nums ${tone}`}>{value}</span>
    </span>
  );
}

// Top bar: name, live telemetry chips (ingest rate, backlog, last line, API latency), model status, pause/resume and theme toggle.
export function TopNav({ threshold, health, telemetry, paused, onPause, light, onTheme, onLogout }: {
  threshold: number; health: Health; telemetry: Telemetry; paused: boolean; onPause: () => void; light: boolean; onTheme: () => void; onLogout: () => void;
}) {
  const badge = BADGE[health ?? "loading"];
  const ms = telemetry.latencyMs;
  const latencyTone = ms === null ? undefined : ms < 150 ? "text-emerald-400" : ms < 500 ? "text-amber-400" : "text-red-400";
  const iconButton = "grid size-8 place-items-center rounded-md border border-white/30 bg-white/15 text-white transition-colors hover:bg-white/30";
  return (
    <header className="flex h-11 shrink-0 items-center justify-between gap-3 bg-[#161d27] px-3 text-slate-300 sm:px-4">
      <span className="text-lg font-semibold tracking-tight text-white">LogInsight</span>
      <div className="flex min-w-0 items-center gap-3 text-xs sm:gap-4">
        {/* Telemetry chips drop away on narrow screens so the status badge always fits. */}
        <div className="hidden items-center gap-4 xl:flex">
          <Chip label="Ingest" value={telemetry.linesPerSec === null ? "–" : `${telemetry.linesPerSec.toLocaleString()} lines/s`} />
          <Chip label="Backlog" value={telemetry.backlog} />
          <Chip label="Last line" value={telemetry.lastLine} />
          <Chip label="API" value={ms === null ? "–" : `${ms} ms`} tone={latencyTone} />
          <Chip label="Threshold" value={`${threshold}%`} />
        </div>
        {/* "online" = the monitor thread loaded the model and is running; the badge follows the real status */}
        <span className={`flex items-center gap-1.5 whitespace-nowrap rounded-sm border px-2 py-0.5 font-medium ${badge.box}`}>
          <span className={`size-1.5 rounded-full ${badge.dot} ${health === "online" && !paused ? "animate-pulse" : ""}`} />
          <span className="hidden sm:inline">{badge.text}</span>
          <span className="sm:hidden">{badge.text.split(" ").pop()}</span>
        </span>
        <button type="button" onClick={onPause} aria-label={paused ? "Resume live polling" : "Pause live polling"} title={paused ? "Resume live polling" : "Pause live polling"} className={iconButton}>
          {paused ? <Play className="size-3.5" /> : <Pause className="size-3.5" />}
        </button>
        <button type="button" onClick={onTheme} aria-label="Toggle light/dark theme" title="Toggle light/dark theme" className={iconButton}>
          {light ? <Moon className="size-3.5" /> : <Sun className="size-3.5" />}
        </button>
        <button type="button" onClick={onLogout} aria-label="Log out" title="Log out" className={iconButton}>
          <LogOut className="size-3.5" />
        </button>
      </div>
    </header>
  );
}

// Left icon rail: one button per section; the open one gets a ring, like the ServiceNow workspace.
export function SideRail({ view, onView }: { view: View; onView: (view: View) => void }) {
  return (
    <nav aria-label="Sections" className="flex h-12 shrink-0 items-center justify-around bg-gradient-to-r from-[#2b2f8f] to-[#14121f] md:h-auto md:w-12 md:flex-col md:justify-start md:gap-2 md:bg-gradient-to-b md:py-3">
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
