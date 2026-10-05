"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { Activity, Globe, Moon, Network, Play, Search, ShieldCheck, Sun, type LucideIcon } from "lucide-react";
import { useTheme } from "@/lib/theme";

const FEATURES: { Icon: LucideIcon; title: string; text: string }[] = [
  { Icon: ShieldCheck, title: "LSTM anomaly detection", text: "A Keras LSTM learns what normal HDFS block activity looks like and scores every block that does not fit." },
  { Icon: Activity, title: "Live monitor", text: "Watches a log file as it grows and scores new blocks within seconds, with charts that update on their own." },
  { Icon: ShieldCheck, title: "Threat-ranked logs", text: "Anomalies are sorted into severity tiers so you read the worst ones first, and a slider sets how strict to be." },
  { Icon: Network, title: "Traffic and IPs", text: "See which hosts are busiest and which are involved in anomalies, then jump straight to their log lines." },
  { Icon: Search, title: "Search", text: "Look up a block ID, an IP address or a keyword and read the raw lines behind any alert." },
  { Icon: Play, title: "Simulator and explanations", text: "Replay normal or attack traffic on demand, and get a plain-English explanation for any flagged block." },
];

const STEPS = [
  ["Point it at a log", "Choose a log file to watch, or upload a test set to analyse."],
  ["Blocks get scored", "Each block's event sequence goes through the LSTM and receives an anomaly score."],
  ["Act on what stands out", "Review the ranked list, read the raw lines and get an explanation."],
];

const TECH = ["Keras LSTM", "Python", "SQLite", "Next.js", "React", "Tailwind"];

type Stats = { total: number; anomalous: number; rate: number; model_loaded: boolean };

// The biggest analysis so far, from the server (public totals only). Re-checked every 15 s, so it follows when a bigger run appears.
function useStats() {
  const [stats, setStats] = useState<Stats | null>(null);
  useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        const response = await fetch("/api/public/stats", { cache: "no-store" });
        if (response.ok && !cancelled) setStats(await response.json());
      } catch {} // server down: keep the dashes / last numbers
    }
    load();
    const timer = setInterval(load, 15000);
    return () => { cancelled = true; clearInterval(timer); };
  }, []);
  return stats;
}

export default function Landing() {
  const [light, toggleTheme] = useTheme();
  const stats = useStats();
  const kpis = [
    ["Blocks scored", stats ? stats.total.toLocaleString() : "–"],
    ["Anomalies", stats ? stats.anomalous.toLocaleString() : "–"],
    ["Anomaly rate", stats ? `${stats.rate.toFixed(1)}%` : "–"],
    ["Model", stats ? (stats.model_loaded ? "Online" : "Loading") : "–"],
  ];
  const button = "rounded-md px-4 py-2 text-sm font-semibold transition-colors";
  const navLink = "rounded-md border border-white/25 bg-white/10 px-3 py-1.5 font-medium text-white transition-colors hover:bg-white/25";
  return (
    <div className="min-h-screen bg-slate-950 text-slate-200">
      <header className="flex h-12 items-center justify-between bg-[#161d27] px-4 sm:px-8">
        <span className="text-lg font-semibold tracking-tight text-white">LogInsight</span>
        <nav className="flex items-center gap-4 text-sm">
          <a href="#features" className={`${navLink} hidden sm:inline`}>Features</a>
          <a href="#how" className={`${navLink} hidden sm:inline`}>How it works</a>
          <Link href="/login/" className={navLink}>Log in</Link>
          <Link href="/login/?mode=signup" className={`${button} bg-white text-[#2b2f8f] hover:bg-slate-200`}>Get started</Link>
          <button type="button" onClick={toggleTheme} aria-label="Toggle light/dark theme" title="Toggle light/dark theme" className="grid size-8 place-items-center rounded-md border border-white/30 bg-white/15 text-white transition-colors hover:bg-white/30">
            {light ? <Moon className="size-3.5" /> : <Sun className="size-3.5" />}
          </button>
        </nav>
      </header>

      <section className="bg-gradient-to-br from-[#2b2f8f] to-[#14121f] px-4 py-16 text-white sm:px-8 sm:py-24">
        <div className="mx-auto grid max-w-6xl items-center gap-10 lg:grid-cols-2">
          <div>
            <h1 className="text-3xl font-bold tracking-tight sm:text-5xl">Catch bad logs before they become outages.</h1>
            <p className="mt-4 max-w-xl text-base text-white/80">LogInsight reads your logs as they come in, picks out the ones that look wrong, and tells you in plain words what may be going on.</p>
            <div className="mt-8 flex flex-wrap gap-3">
              <Link href="/login/?mode=signup" className={`${button} bg-white text-[#2b2f8f] hover:bg-slate-200`}>Get started</Link>
              <Link href="/login/" className={`${button} border border-white/40 bg-white/10 text-white hover:bg-white/25`}>Log in</Link>
            </div>
          </div>
          {/* Real numbers from the biggest analysis so far (the bar splits its blocks into normal and anomalous). */}
          <div className="rounded-xl border border-white/20 bg-white/10 p-4 shadow-xl backdrop-blur">
            <div className="grid grid-cols-2 gap-3">
              {kpis.map(([label, value]) => (
                <div key={label} className="rounded-lg bg-black/25 p-4">
                  <div className="text-xs text-white/60">{label}</div>
                  <div className="mt-1 text-2xl font-semibold tabular-nums">{value}</div>
                </div>
              ))}
            </div>
            <div className="mt-3 rounded-lg bg-black/25 p-3">
              <div className="flex justify-between text-xs text-white/70">
                <span>Normal blocks</span><span>Anomalous blocks</span>
              </div>
              <div className="mt-2 flex h-3 overflow-hidden rounded-full bg-emerald-400">
                <div className="bg-red-400" style={{ width: `${Math.max(stats?.rate ?? 0, stats?.anomalous ? 2 : 0)}%`, marginLeft: "auto" }} />
              </div>
            </div>
          </div>
        </div>
      </section>

      <section id="features" className="mx-auto max-w-6xl px-4 py-16 sm:px-8">
        <h2 className="text-2xl font-semibold text-slate-100">What it does</h2>
        <div className="mt-8 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {FEATURES.map(({ Icon, title, text }) => (
            <div key={title} className="rounded-lg border border-slate-800 bg-slate-900 p-5">
              <span className="grid size-9 place-items-center rounded-lg bg-indigo-950 text-indigo-400"><Icon className="size-[18px]" /></span>
              <h3 className="mt-3 font-semibold text-slate-100">{title}</h3>
              <p className="mt-1 text-slate-400">{text}</p>
            </div>
          ))}
        </div>
      </section>

      <section id="how" className="bg-slate-900 px-4 py-16 sm:px-8">
        <div className="mx-auto max-w-6xl">
          <h2 className="text-2xl font-semibold text-slate-100">How it works</h2>
          <ol className="mt-8 grid gap-6 md:grid-cols-3">
            {STEPS.map(([title, text], i) => (
              <li key={title} className="flex gap-4">
                <span className="grid size-9 shrink-0 place-items-center rounded-full bg-[#2b2f8f] font-semibold text-white">{i + 1}</span>
                <div><h3 className="font-semibold text-slate-100">{title}</h3><p className="mt-1 text-slate-400">{text}</p></div>
              </li>
            ))}
          </ol>
        </div>
      </section>

      <section className="mx-auto flex max-w-6xl flex-wrap items-center gap-2 px-4 py-10 sm:px-8">
        <Globe className="size-4 text-slate-500" /><span className="mr-2 text-slate-500">Built with</span>
        {TECH.map((t) => <span key={t} className="rounded-full border border-slate-700 px-3 py-1 text-xs text-slate-300">{t}</span>)}
      </section>

      <footer className="border-t border-slate-800 px-4 py-6 text-center text-xs text-slate-500">LogInsight · LSTM anomaly detection for HDFS logs</footer>
    </div>
  );
}
