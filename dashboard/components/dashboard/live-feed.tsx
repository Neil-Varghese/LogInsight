"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { getFeed, type FeedLine } from "@/lib/api";
import { TIERS, TIER_ORDER, tierOf } from "@/lib/severity";

const POLL_MS = 1000; // how often new lines are fetched
const TICK_MS = 60; // how often one queued line (or a few, if many are waiting) is shown: this is what makes them pop in one by one
const KEEP = 300; // rows kept on screen
const QUEUE_MAX = 2000; // lines waiting to be shown (only grows while paused)

const clock = (t: number | null) => (t === null ? "--:--:--" : new Date(t * 1000).toISOString().slice(11, 19));

// Every log line the monitor processes, newest first, ranked by threat level.
export function LiveFeed() {
  const [lines, setLines] = useState<FeedLine[]>([]);
  const [counts, setCounts] = useState<Record<string, number>>({});
  const [off, setOff] = useState<number[]>([]); // severities switched off by the filter chips
  const [threatFirst, setThreatFirst] = useState(false);
  const [paused, setPaused] = useState(false);
  const [error, setError] = useState("");
  const queue = useRef<FeedLine[]>([]);
  const cursor = useRef(-1); // sequence number of the last line fetched
  const generation = useRef<number | null>(null);
  const pausedRef = useRef(false);

  useEffect(() => { pausedRef.current = paused; }, [paused]);

  // Fetch whatever is new once a second and park it in the queue.
  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout>;
    async function poll() {
      try {
        const feed = await getFeed(cursor.current);
        if (cancelled) return;
        const reset = (generation.current !== null && generation.current !== feed.generation) || feed.latest < cursor.current;
        if (reset) { queue.current = []; setLines([]); } // the monitor was reset or the server restarted: start the view over
        generation.current = feed.generation;
        const last = feed.lines[feed.lines.length - 1];
        cursor.current = feed.latest < cursor.current ? -1 : last ? last.seq : cursor.current;
        queue.current = queue.current.concat(feed.lines).slice(-QUEUE_MAX);
        setCounts(feed.counts); setError("");
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : "Could not reach the server.");
      }
      if (!cancelled) timer = setTimeout(poll, POLL_MS);
    }
    poll();
    return () => { cancelled = true; clearTimeout(timer); };
  }, []);

  // Show queued lines at a steady pace; the more that are waiting, the more per step, so the view never falls far behind.
  useEffect(() => {
    const id = setInterval(() => {
      if (pausedRef.current || queue.current.length === 0) return;
      const batch = queue.current.splice(0, Math.max(1, Math.ceil(queue.current.length / 20)));
      setLines((prev) => [...batch.reverse(), ...prev].slice(0, KEEP));
    }, TICK_MS);
    return () => clearInterval(id);
  }, []);

  const shown = useMemo(() => {
    const visible = lines.filter((l) => !off.includes(l.sev));
    return threatFirst ? [...visible].sort((a, b) => a.sev - b.sev || b.seq - a.seq) : visible;
  }, [lines, off, threatFirst]);

  const button = "rounded-sm border px-2 py-0.5 text-[11px] border-slate-700 text-slate-300 hover:border-slate-500";
  return (
    <section className="rounded-lg border border-slate-800 bg-slate-900 shadow-sm">
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-800 px-3 py-2">
        <h2 className="text-xs font-semibold uppercase tracking-wide text-slate-400">Live log stream</h2>
        <div className="flex items-center gap-2">
          <button type="button" aria-pressed={threatFirst} onClick={() => setThreatFirst((v) => !v)} className={`${button} ${threatFirst ? "!border-indigo-500 text-slate-100" : ""}`}>
            {threatFirst ? "Sorted: threat first" : "Sorted: newest first"}
          </button>
          <button type="button" onClick={() => setPaused((v) => !v)} className={`${button} ${paused ? "!border-amber-600 text-amber-300" : ""}`}>
            {paused ? "▶ Resume" : "❚❚ Pause"}
          </button>
        </div>
      </div>

      <div className="flex flex-wrap gap-1.5 border-b border-slate-800 px-3 py-2">
        {TIER_ORDER.map((sev) => {
          const tier = TIERS[sev];
          const hidden = off.includes(sev);
          return (
            <button
              key={sev} type="button" aria-pressed={!hidden} title={`${tier.levels}${hidden ? " (hidden, click to show)" : " (click to hide)"}`}
              onClick={() => setOff((cur) => (hidden ? cur.filter((s) => s !== sev) : [...cur, sev]))}
              className={`flex items-center gap-1.5 rounded-sm border px-2 py-0.5 text-[11px] ${tier.badge} ${hidden ? "opacity-40 line-through" : ""}`}
            >
              <span className="font-semibold">{tier.priority}</span> {tier.label}
              <span className="font-mono">{(counts[String(sev)] ?? 0).toLocaleString()}</span>
            </button>
          );
        })}
      </div>

      {error && <p className="border-b border-red-900 bg-red-950 px-3 py-1.5 text-xs text-red-400">{error}</p>}
      <div role="log" aria-live="off" className="h-80 overflow-y-auto">
        {shown.length === 0 ? (
          <div className="flex h-full items-center justify-center px-3 text-center text-xs text-slate-500">
            {lines.length === 0 ? "Waiting for log lines… press Run on the Simulation tab." : "Every priority is switched off, or none of the lines on screen match."}
          </div>
        ) : (
          shown.map((l) => (
            <div key={l.seq} className="flex animate-in items-baseline gap-3 border-b border-slate-800/60 px-3 py-1 font-mono text-[11px] duration-300 fade-in slide-in-from-top-1 motion-reduce:animate-none">
              <span className="shrink-0 text-slate-500">{clock(l.t)}</span>
              <span className={`w-20 shrink-0 rounded-sm border px-1 text-center ${tierOf(l.sev).badge}`}>{tierOf(l.sev).priority} {l.level}</span>
              <span className="hidden w-32 shrink-0 truncate text-slate-500 md:inline">{l.src}</span>
              <span className="min-w-0 break-all text-slate-300">{l.text}</span>
            </div>
          ))
        )}
      </div>
      <p className="border-t border-slate-800 px-3 py-1.5 text-[11px] text-slate-500">
        Threat ranking follows the RFC 5424 syslog severity scale: P1 Critical = FATAL, P2 Error, P3 Warning, P4 Unclassified (unknown level, never ignored), P5 Info, P6 Debug.
        Counts cover every line processed since the last reset; the list keeps the newest {KEEP}.
      </p>
    </section>
  );
}
