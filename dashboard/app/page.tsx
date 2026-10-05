"use client";

import { useEffect, useState } from "react";
import { TopNav } from "@/components/dashboard/top-nav";
import { TestSetSelector } from "@/components/dashboard/test-set-selector";
import { KpiCards, type Kpis } from "@/components/dashboard/kpi-cards";
import { SessionsTable, type Session } from "@/components/dashboard/sessions-table";
import { InspectDrawer, type Inspection } from "@/components/dashboard/inspect-drawer";
import { getTestSets, inspectBlock, runDetection } from "@/lib/api";

const EMPTY: Kpis = { total: 0, anomalous: 0, normal: 0, rate: 0, latencySeconds: 0 };

export default function Home() {
  const [files, setFiles] = useState<string[]>([]);
  const [online, setOnline] = useState<boolean | null>(null);
  const [selected, setSelected] = useState("");
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState("");
  const [kpis, setKpis] = useState<Kpis>(EMPTY);
  const [rows, setRows] = useState<Session[]>([]);
  const [jobId, setJobId] = useState("");
  const [inspection, setInspection] = useState<Inspection | null>(null);

  useEffect(() => {
    getTestSets()
      .then((sets) => { setFiles(sets.map((s) => s.name)); setSelected(sets[0]?.name ?? ""); setOnline(true); })
      .catch(() => setOnline(false));
  }, []);

  async function run() {
    if (!selected || busy) return;
    setBusy(true);
    setStatus("Starting…");
    try {
      const result = await runDetection(selected, setStatus);
      setKpis(result.kpis);
      setRows(result.rows);
      setJobId(result.jobId);
      setStatus("");
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "Run failed.");
    } finally {
      setBusy(false);
    }
  }

  async function inspect(blockId: string) {
    setInspection({ blockId, loading: true, explanation: null, error: null, rawLogs: [] });
    try {
      const result = await inspectBlock(jobId, blockId);
      setInspection({ blockId, loading: false, ...result });
    } catch (error) {
      setInspection({ blockId, loading: false, explanation: null, rawLogs: [], error: error instanceof Error ? error.message : "Inspect failed." });
    }
  }

  return (
    <div className="flex min-h-screen flex-col">
      <TopNav threshold={50} online={online} />
      <main className="mx-auto w-full max-w-7xl space-y-2 p-3">
        <TestSetSelector files={files} selected={selected} busy={busy} onSelect={setSelected} onRun={run} />
        {status && <p className="text-xs text-slate-400">{status}</p>}
        <KpiCards k={kpis} />
        <SessionsTable rows={rows} onInspect={inspect} />
      </main>
      {inspection && <InspectDrawer data={inspection} onClose={() => setInspection(null)} />}
    </div>
  );
}
