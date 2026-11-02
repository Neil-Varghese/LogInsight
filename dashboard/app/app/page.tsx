"use client";

import { useEffect, useRef, useState } from "react";
import { ActiveIps } from "@/components/dashboard/active-ips";
import { Executive } from "@/components/dashboard/executive";
import { Monitor } from "@/components/dashboard/monitor";
import { Settings } from "@/components/dashboard/settings";
import { Traffic } from "@/components/dashboard/traffic";
import { SideRail, TabStrip, TopNav, type Health, type View } from "@/components/dashboard/top-nav";
import { getStatus, type Status } from "@/lib/api";
import { getMe, logout } from "@/lib/auth";
import { useTheme } from "@/lib/theme";

const POLL_MS = 3000;

export default function DashboardPage() {
  const [view, setView] = useState<View>("monitor");
  const threshold = 50; // percent; the fixed cut-off: a block scoring at or above it is anomalous
  const [searchIp, setSearchIp] = useState(""); // set by clicking an IP on the Active IPs tab
  const [status, setStatus] = useState<Status | null>(null);
  const [reachable, setReachable] = useState<boolean | null>(null);
  const [light, toggleTheme] = useTheme();
  const [signedIn, setSignedIn] = useState(false); // nothing is drawn (and no API call made) until the server confirms the session
  const mainRef = useRef<HTMLElement>(null);

  // Every tab shares one scrolling <main>, so it keeps its scroll position when the tab changes; start each tab at the top.
  useEffect(() => { mainRef.current?.scrollTo(0, 0); }, [view]);

  useEffect(() => {
    getMe().then((user) => { if (user) setSignedIn(true); else window.location.replace("/login/"); });
  }, []);

  useEffect(() => {
    if (!signedIn) return;
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
  }, [signedIn]);

  const health: Health = reachable === null ? null : !reachable ? "offline" : status?.error ? "error" : status?.model_loaded ? "online" : "loading";

  if (!signedIn) return <div className="grid h-screen place-items-center text-slate-400">Loading…</div>;

  return (
    <div className="flex h-screen flex-col">
      <TopNav health={health} light={light} onTheme={toggleTheme} onLogout={() => logout().then(() => window.location.replace("/login/"))} />
      {/* On phones the section rail sits at the bottom (flex-col-reverse); from md up it is the left rail. */}
      <div className="flex min-h-0 flex-1 flex-col-reverse md:flex-row">
        <SideRail view={view} onView={(next) => { if (next !== "traffic") setSearchIp(""); setView(next); }} />
        <div className="flex min-h-0 min-w-0 flex-1 flex-col">
          <TabStrip view={view} />
          <main ref={mainRef} className="min-h-0 flex-1 overflow-y-auto p-3">
            {view === "executive" && <Executive threshold={threshold} />}
            {view === "monitor" && <Monitor status={status} threshold={threshold} />}
            {view === "traffic" && <Traffic initialIp={searchIp} threshold={threshold / 100} />}
            {view === "ips" && <ActiveIps onSearch={(ip) => { setSearchIp(ip); setView("traffic"); }} />}
            {view === "settings" && <Settings status={status} />}
          </main>
        </div>
      </div>
    </div>
  );
}
