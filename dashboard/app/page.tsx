"use client";

import { useEffect, useState } from "react";
import { ActiveIps } from "@/components/dashboard/active-ips";
import { Executive } from "@/components/dashboard/executive";
import { Monitor } from "@/components/dashboard/monitor";
import { Settings } from "@/components/dashboard/settings";
import { Simulation } from "@/components/dashboard/simulation";
import { Traffic } from "@/components/dashboard/traffic";
import { SideRail, TabStrip, TopNav, type Health, type View } from "@/components/dashboard/top-nav";
import { getStatus, type Status } from "@/lib/api";

const POLL_MS = 3000;

export default function Home() {
  const [view, setView] = useState<View>("monitor");
  const [threshold, setThreshold] = useState(50); // percent; the server scores once, the threshold only re-colours results
  const [searchIp, setSearchIp] = useState(""); // set by clicking an IP on the Active IPs tab
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
    <div className="flex h-screen flex-col">
      <TopNav threshold={threshold} health={health} />
      <div className="flex min-h-0 flex-1">
        <SideRail view={view} onView={(next) => { if (next !== "traffic") setSearchIp(""); setView(next); }} />
        <div className="flex min-w-0 flex-1 flex-col">
          <TabStrip view={view} />
          <main className="min-h-0 flex-1 overflow-y-auto p-3">
            {view === "executive" && <Executive status={status} threshold={threshold} />}
            {view === "monitor" && <Monitor status={status} threshold={threshold} onThreshold={setThreshold} />}
            {view === "simulation" && <Simulation status={status} threshold={threshold} />}
            {view === "traffic" && <Traffic initialIp={searchIp} threshold={threshold / 100} />}
            {view === "ips" && <ActiveIps onSearch={(ip) => { setSearchIp(ip); setView("traffic"); }} />}
            {view === "settings" && <Settings status={status} />}
          </main>
        </div>
      </div>
    </div>
  );
}
