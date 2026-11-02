"use client";

import { useEffect, useState } from "react";
import { getSimInfo, startSim, stopSim, resetMonitor, type SimInfo, type SimMode } from "@/lib/api";

const POLL_MS = 2000;
const clock = (s: number) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;

const SOURCES: { id: SimMode; title: string; text: string }[] = [
  { id: "replay", title: "Replay (no Docker needed)", text: "Streams endless HDFS-style logs at 20 lines per second, about 5% of blocks taken from the anomalous sample. Same generator as Docker, run locally." },
  { id: "docker", title: "Docker cluster", text: "Starts 3 fake datanode containers that print an endless HDFS-style stream. Needs Docker Desktop running." },
];

// Run / stop / reset controls for the log source, shown at the top of the Monitor tab.
export function SimControls() {
  const [sim, setSim] = useState<SimInfo | null>(null);
  const [mode, setMode] = useState<SimMode>("replay");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [confirmReset, setConfirmReset] = useState(false);

  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout>;
    async function refresh() {
      try {
        const info = await getSimInfo();
        if (!cancelled) setSim(info);
      } catch { /* keep the last data; the top bar already shows when the backend is offline */ }
      if (!cancelled) timer = setTimeout(refresh, POLL_MS);
    }
    refresh();
    return () => { cancelled = true; clearTimeout(timer); };
  }, []);

  async function toggle() {
    setBusy(true); setError("");
    try {
      setSim(await (sim?.running ? stopSim() : startSim(mode)));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not reach the server.");
    } finally {
      setBusy(false);
    }
  }

  async function reset() {
    if (!confirmReset) { setConfirmReset(true); return; } // two clicks: this wipes the monitor's results
    setConfirmReset(false); setBusy(true); setError("");
    try { await resetMonitor(); }
    catch (e) { setError(e instanceof Error ? e.message : "Could not reach the server."); }
    finally { setBusy(false); }
  }

  const running = sim?.running ?? false;
  const finished = !running && sim?.mode && sim.exit_code !== null;
  return (
    <div className="space-y-2">
      <section className="space-y-3 rounded-lg border border-slate-800 bg-slate-900 shadow-sm p-4">
        <div className="grid gap-2 md:grid-cols-2">
          {SOURCES.map((s) => (
            <label
              key={s.id}
              className={`block cursor-pointer border p-3 text-xs ${(running ? sim?.mode : mode) === s.id ? "border-indigo-500 bg-slate-950" : "border-slate-800 hover:border-slate-600"} ${running ? "pointer-events-none opacity-70" : ""}`}
            >
              <input type="radio" name="source" className="mr-2 accent-indigo-500" checked={(running ? sim?.mode : mode) === s.id} onChange={() => setMode(s.id)} />
              <span className="font-semibold text-slate-200">{s.title}</span>
              <span className="mt-1 block text-slate-400">{s.text}</span>
            </label>
          ))}
        </div>
        <div className="flex flex-wrap items-center gap-4">
          <button
            type="button" onClick={toggle} disabled={busy}
            className={`h-12 min-w-56 rounded-sm border px-8 text-sm font-semibold disabled:opacity-60 ${running ? "border-red-700 bg-red-950 text-red-200 hover:bg-red-900" : "border-emerald-600 bg-emerald-700 text-white hover:bg-emerald-600"}`}
          >
            {busy ? "Working…" : running ? "■  Stop simulation" : "▶  Run simulation"}
          </button>
          <button
            type="button" onClick={reset} disabled={busy || running} onBlur={() => setConfirmReset(false)}
            className="h-12 rounded-sm border border-slate-700 px-6 text-sm font-semibold text-slate-300 hover:bg-slate-800 disabled:opacity-40"
          >
            {confirmReset ? "Click again to wipe all results" : "Reset to 0"}
          </button>
          <span className="text-xs text-slate-400">
            {running && <span className="flex items-center gap-2 text-emerald-400"><span className="size-2 animate-pulse rounded-full bg-emerald-400" />Running {sim?.mode} · {clock(sim?.elapsed_s ?? 0)} elapsed</span>}
            {finished && `Finished after ${clock(sim?.elapsed_s ?? 0)}. The results below are what the monitor made of it.`}
            {!running && !finished && "Press Run: the monitor starts from zero and the results fill in below as lines arrive."}
          </span>
        </div>
        {error && <p className="border border-red-900 bg-red-950 px-3 py-2 text-xs text-red-400">{error}</p>}
      </section>
    </div>
  );
}
