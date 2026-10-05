"use client";

import { ChevronDown, ChevronRight } from "lucide-react";
import { useEffect, useState } from "react";
import { useInspector } from "@/components/dashboard/inspect-drawer";
import { KpiCards, type Kpis } from "@/components/dashboard/kpi-cards";
import { SessionsTable, type Session } from "@/components/dashboard/sessions-table";
import { TestSetSelector } from "@/components/dashboard/test-set-selector";
import { getTestSets, resetMonitor, runDetection, type Status } from "@/lib/api";

const EMPTY: Kpis = { total: 0, anomalous: 0, normal: 0, rate: 0 };

export function Settings({ status }: { status: Status | null }) {
  const [developer, setDeveloper] = useState(false);
  return (
    <div className="space-y-2">
      <section className="border border-slate-800 bg-slate-900 p-3">
        <h2 className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-400">Log source</h2>
        <dl className="grid grid-cols-[10rem_1fr] gap-y-1 text-xs">
          <dt className="text-slate-500">Watched file</dt>
          <dd className="font-mono text-slate-200">{status?.file ?? "–"}</dd>
          <dt className="text-slate-500">Change it</dt>
          <dd className="text-slate-400">
            start the server with <span className="font-mono text-slate-300">LOGINSIGHT_WATCH=path/to/hdfs.log</span> (relative to the project folder)
          </dd>
          <dt className="text-slate-500">Format</dt>
          <dd className="text-slate-400">HDFS only for now</dd>
        </dl>
      </section>

      <section className="border border-slate-800 bg-slate-900">
        <button
          type="button"
          aria-expanded={developer}
          onClick={() => setDeveloper((open) => !open)}
          className="flex w-full items-center gap-2 px-3 py-2 text-left text-xs font-semibold uppercase tracking-wide text-slate-400 hover:text-slate-200"
        >
          {developer ? <ChevronDown className="size-3.5" /> : <ChevronRight className="size-3.5" />}
          Developer settings
        </button>
        {developer && <DeveloperSettings />}
      </section>
    </div>
  );
}

// Test sets: one-off batch runs over a file (parse everything, score everything), separate from the live monitor.
function DeveloperSettings() {
  const [files, setFiles] = useState<string[]>([]);
  const [selected, setSelected] = useState("");
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const [kpis, setKpis] = useState<Kpis>(EMPTY);
  const [latency, setLatency] = useState(0);
  const [rows, setRows] = useState<Session[]>([]);
  const [jobId, setJobId] = useState("");
  const [confirmReset, setConfirmReset] = useState(false);
  const { open, drawer } = useInspector();

  useEffect(() => {
    getTestSets()
      .then((sets) => { setFiles(sets.map((s) => s.name)); setSelected(sets[0]?.name ?? ""); })
      .catch((e) => setMessage(e instanceof Error ? e.message : "Could not load test sets."));
  }, []);

  async function run() {
    if (!selected || busy) return;
    setBusy(true);
    setMessage("Starting…");
    try {
      const result = await runDetection(selected, setMessage);
      setKpis(result.kpis); setLatency(result.latencySeconds); setRows(result.rows); setJobId(result.jobId); setMessage("");
    } catch (e) {
      setMessage(e instanceof Error ? e.message : "Run failed.");
    } finally {
      setBusy(false);
    }
  }

  async function reset() {
    if (!confirmReset) { setConfirmReset(true); return; } // two clicks: this wipes the live monitor's results
    setConfirmReset(false);
    try { await resetMonitor(); setMessage("Live monitor reset; it is re-reading its file from the top."); }
    catch (e) { setMessage(e instanceof Error ? e.message : "Reset failed."); }
  }

  return (
    <div className="space-y-2 border-t border-slate-800 p-3">
      <div className="flex items-center justify-between border border-slate-800 bg-slate-950 px-3 py-2 text-xs">
        <span className="text-slate-400">Live monitor data: clear all results and re-read the watched file from the top.</span>
        <button
          type="button"
          onClick={reset}
          onBlur={() => setConfirmReset(false)}
          className={`rounded-sm border px-2 py-0.5 ${confirmReset ? "border-red-700 bg-red-950 text-red-300" : "border-slate-700 text-slate-300 hover:border-slate-500"}`}
        >
          {confirmReset ? "Click again to confirm" : "Reset & reprocess"}
        </button>
      </div>

      <TestSetSelector files={files} selected={selected} busy={busy} onSelect={setSelected} onRun={run} />
      {message && <p className="text-xs text-slate-400">{message}</p>}
      <KpiCards k={kpis} last={{ label: "Pipeline Latency", value: `${latency.toFixed(1)}s`, sub: "Drain parsing + LSTM inference" }} />
      <SessionsTable rows={rows} onInspect={(blockId) => open(jobId, blockId)} />
      {drawer}
    </div>
  );
}
