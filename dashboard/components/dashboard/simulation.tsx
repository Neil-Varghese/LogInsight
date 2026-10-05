"use client";

import { useEffect, useState } from "react";
import { Charts } from "@/components/dashboard/charts";
import { KpiCards } from "@/components/dashboard/kpi-cards";
import { getSimInfo, getSummary, startSim, stopSim, type SimInfo, type SimMode, type Status, type Summary } from "@/lib/api";

const POLL_MS = 2000;
const clock = (s: number) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;

const SOURCES: { id: SimMode; title: string; text: string }[] = [
  { id: "replay", title: "Replay (no Docker needed)", text: "Plays a real HDFS sample log into the monitor at 20 lines per second, about 90 seconds in total." },
  { id: "docker", title: "Docker cluster", text: "Starts 3 fake datanode containers that print an endless HDFS-style stream. Needs Docker Desktop running." },
];

const STEPS = [
  "Reads real HDFS block sessions from the sample files (normal and anomalous) and groups the lines by block id.",
  "Starts a session: picks one at random (an anomalous one about 5% of the time), gives it a brand-new random blk_ id and the current time.",
  "Keeps 8 sessions open at once and prints one random line from a random session, so lines from different blocks interleave like on a busy node.",
  "Waits a random gap (average 1 / lines-per-second) between lines. 3 containers do this at once, and collect.py merges them into sim/cluster.log.",
];

export function Simulation({ status, threshold }: { status: Status | null; threshold: number }) {
  const [sim, setSim] = useState<SimInfo | null>(null);
  const [summary, setSummary] = useState<Summary | null>(null);
  const [mode, setMode] = useState<SimMode>("replay");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout>;
    async function refresh() {
      try {
        const [info, results] = await Promise.all([getSimInfo(), getSummary(threshold / 100)]);
        if (!cancelled) { setSim(info); setSummary(results); }
      } catch { /* keep the last data; the top bar already shows when the backend is offline */ }
      if (!cancelled) timer = setTimeout(refresh, POLL_MS);
    }
    refresh();
    return () => { cancelled = true; clearTimeout(timer); };
  }, [threshold]);

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
          <span className="text-xs text-slate-400">
            {running && <span className="flex items-center gap-2 text-emerald-400"><span className="size-2 animate-pulse rounded-full bg-emerald-400" />Running {sim?.mode} · {clock(sim?.elapsed_s ?? 0)} elapsed</span>}
            {finished && `Finished after ${clock(sim?.elapsed_s ?? 0)}. The results below are what the monitor made of it.`}
            {!running && !finished && "Press Run: the monitor starts from zero and the results fill in below as lines arrive."}
          </span>
        </div>
        {error && <p className="border border-red-900 bg-red-950 px-3 py-2 text-xs text-red-400">{error}</p>}
      </section>

      <KpiCards
        k={{ total: summary?.total ?? 0, anomalous: summary?.anomalous ?? 0, normal: summary?.normal ?? 0, rate: summary?.rate ?? 0 }}
        last={{ label: "Lines Ingested", value: status ? status.lines.toLocaleString() : "–", sub: "since the monitor last reset" }}
      />
      <Charts summary={summary} threshold={threshold / 100} />

      <section className="rounded-lg border border-slate-800 bg-slate-900 shadow-sm p-3">
        <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-400">How the Docker simulator generates data</h3>
        <ol className="list-decimal space-y-1 pl-5 text-xs text-slate-400">{STEPS.map((step) => <li key={step}>{step}</li>)}</ol>
        <div className="mt-3 grid gap-2 text-xs md:grid-cols-2">
          <div>
            <div className="mb-1 text-[11px] uppercase text-slate-500">Real sample line (2008)</div>
            <pre className="overflow-x-auto border border-slate-800 bg-slate-950 p-2 font-mono text-[11px] text-slate-400">081109 203518 143 INFO dfs.DataNode$DataXceiver: Receiving block blk_-1608999687919862906 src: /10.250.19.102:54106 dest: /10.250.19.102:50010</pre>
          </div>
          <div>
            <div className="mb-1 text-[11px] uppercase text-slate-500">What the simulator prints (new id + current time, rest unchanged)</div>
            <pre className="overflow-x-auto border border-slate-800 bg-slate-950 p-2 font-mono text-[11px] text-slate-300">261004 142233 143 INFO dfs.DataNode$DataXceiver: Receiving block blk_2931840031120442871 src: /10.250.19.102:54106 dest: /10.250.19.102:50010</pre>
          </div>
        </div>
        {sim && (
          <p className="mt-3 text-xs text-slate-400">
            Docker settings (docker-compose.yml): <span className="font-mono text-slate-200">{sim.containers ?? "?"}</span> containers ×{" "}
            <span className="font-mono text-slate-200">{sim.lines_per_second ?? "?"}</span> lines/s,{" "}
            <span className="font-mono text-slate-200">{sim.anomaly_rate ? `${Number(sim.anomaly_rate) * 100}%` : "?"}</span> anomalous sessions.
          </p>
        )}
        <div className="mt-2 text-[11px] uppercase text-slate-500">Newest lines in {sim?.file ?? "the watched file"} (what the monitor is reading)</div>
        <pre className="mt-1 max-h-48 overflow-auto border border-slate-800 bg-slate-950 p-2 font-mono text-[11px] leading-5 text-slate-300">
          {sim?.tail.length ? sim.tail.join("\n") : "Empty so far. Press Run simulation."}
        </pre>
      </section>
    </div>
  );
}
