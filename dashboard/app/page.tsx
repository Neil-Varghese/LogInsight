"use client";

import { useEffect, useState } from "react";
import { Monitor } from "@/components/dashboard/monitor";
import { Settings } from "@/components/dashboard/settings";
import { TopNav, type Health, type View } from "@/components/dashboard/top-nav";
import { getStatus, type Status } from "@/lib/api";

const POLL_MS = 3000;

export default function Home() {
  const [view, setView] = useState<View>("monitor");
  const [threshold, setThreshold] = useState(50); // percent; the server scores once, the threshold only re-colours results
  const [status, setStatus] = useState<Status | null>(null);
  const [reachable, setReachable] = useState<boolean | null>(null);

  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout>;
    async function poll() {
      try {
        const next = await getStatus();
        if (!cancelled) { setStatus(next); setReachable(true); }
      } catch {
        if (!cancelled) setReachable(false);
      }
      if (!cancelled) timer = setTimeout(poll, POLL_MS);
    }
    poll();
    return () => { cancelled = true; clearTimeout(timer); };
  }, []);

  const health: Health = reachable === null ? null : !reachable ? "offline" : status?.error ? "error" : status?.model_loaded ? "online" : "loading";

  return (
    <div className="flex min-h-screen flex-col">
      <TopNav threshold={threshold} health={health} view={view} onView={setView} />
      <main className="mx-auto w-full max-w-7xl p-3">
        {view === "monitor" ? <Monitor status={status} threshold={threshold} onThreshold={setThreshold} /> : <Settings status={status} />}
      </main>
    </div>
  );
}
