"use client";

import { useEffect, useRef, useState } from "react";
import { ActiveIps } from "@/components/dashboard/active-ips";
import { Executive } from "@/components/dashboard/executive";
import { Monitor } from "@/components/dashboard/monitor";
import { Settings } from "@/components/dashboard/settings";
import { Simulation } from "@/components/dashboard/simulation";
import { Traffic } from "@/components/dashboard/traffic";
import { SideRail, TabStrip, TopNav, type Health, type View } from "@/components/dashboard/top-nav";
import { getStatus, type Status } from "@/lib/api";
import { getMe, logout } from "@/lib/auth";
import { useTheme } from "@/lib/theme";
import { ago, kb } from "@/lib/utils";

const POLL_MS = 3000;

export default function DashboardPage() {
  const [view, setView] = useState<View>("monitor");
  const [threshold, setThreshold] = useState(50); // percent; the server scores once, the threshold only re-colours results
  const [searchIp, setSearchIp] = useState(""); // set by clicking an IP on the Active IPs tab
  const [status, setStatus] = useState<Status | null>(null);
  const [reachable, setReachable] = useState<boolean | null>(null);
  const [paused, setPaused] = useState(false); // pauses every live poll below it (status here, scores and charts in Monitor)
  const [latencyMs, setLatencyMs] = useState<number | null>(null); // round trip of the last status poll
  const [light, toggleTheme] = useTheme();
  const [signedIn, setSignedIn] = useState(false); // nothing is drawn (and no API call made) until the server confirms the session
  const mainRef = useRef<HTMLElement>(null);

  // Every tab shares one scrolling <main>, so it keeps its scroll position when the tab changes; start each tab at the top.
  useEffect(() => { mainRef.current?.scrollTo(0, 0); }, [view]);

  useEffect(() => {
    getMe().then((user) => { if (user) setSignedIn(true); else window.location.replace("/login/"); });
  }, []);

  useEffect(() => {
    if (paused || !signedIn) return;
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout>;
    async function poll() {
      try {
        const began = performance.now();
        const next = await getStatus();
        if (!cancelled) { setStatus(next); setReachable(true); setLatencyMs(Math.round(performance.now() - began)); }
      } catch {
        if (!cancelled) setReachable(false);
      }
      if (!cancelled) timer = setTimeout(poll, POLL_MS);
    }
    poll();
    return () => { cancelled = true; clearTimeout(timer); };
  }, [paused, signedIn]);

  const health: Health = reachable === null ? null : !reachable ? "offline" : status?.error ? "error" : status?.model_loaded ? "online" : "loading";

  const idle = status?.last_ingest ? status.now - status.last_ingest : null;
  const telemetry = {
    linesPerSec: status?.rates.length ? status.rates[status.rates.length - 1].rate : null,
    backlog: status ? kb(status.backlog_bytes) : "–",
    lastLine: idle === null ? "none yet" : ago(idle),
    latencyMs,
  };

  if (!signedIn) return <div className="grid h-screen place-items-center text-slate-400">Loading…</div>;

  return (
    <div className="flex h-screen flex-col">
      <TopNav threshold={threshold} health={health} telemetry={telemetry} paused={paused} onPause={() => setPaused((p) => !p)} light={light} onTheme={toggleTheme} onLogout={() => logout().then(() => window.location.replace("/login/"))} />
      {/* On phones the section rail sits at the bottom (flex-col-reverse); from md up it is the left rail. */}
      <div className="flex min-h-0 flex-1 flex-col-reverse md:flex-row">
        <SideRail view={view} onView={(next) => { if (next !== "traffic") setSearchIp(""); setView(next); }} />
        <div className="flex min-h-0 min-w-0 flex-1 flex-col">
          <TabStrip view={view} />
          <main ref={mainRef} className="min-h-0 flex-1 overflow-y-auto p-3">
            {view === "executive" && <Executive threshold={threshold} />}
            {view === "monitor" && <Monitor status={status} threshold={threshold} onThreshold={setThreshold} paused={paused} onPause={() => setPaused((p) => !p)} />}
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
