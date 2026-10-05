"use client";

import { ArrowDown, ArrowUp, ChevronRight } from "lucide-react";
import { useMemo, useState, type ReactNode } from "react";

export type Session = { blockId: string; anomalous: boolean; confidence: number; events: number };

type SortKey = "blockId" | "anomalous" | "confidence" | "events";
const COLUMNS: { key: SortKey; label: string; right?: boolean }[] = [
  { key: "blockId", label: "Block ID" },
  { key: "anomalous", label: "Status" },
  { key: "confidence", label: "Anomaly Confidence (%)", right: true },
  { key: "events", label: "Events", right: true },
];
const GRID = "grid grid-cols-[minmax(220px,1fr)_110px_170px_80px_90px] items-center"; // one template for header and rows so columns line up
const ROW = 32; // px; fixed so the window maths is a multiplication
const VIEW = 420; // px height of the scrolling body
const OVERSCAN = 8; // extra rows above/below so fast scrolling never shows a gap

// Sortable grid that only renders the rows in view, so thousands of blocks stay smooth.
export function SessionsTable({ rows, onInspect, title = "Classified Block Sessions", toolbar }: {
  rows: Session[]; onInspect: (blockId: string) => void; title?: string; toolbar?: ReactNode;
}) {
  const [sort, setSort] = useState<{ key: SortKey; desc: boolean }>({ key: "confidence", desc: true });
  const [top, setTop] = useState(0);

  const sorted = useMemo(() => {
    const sign = sort.desc ? -1 : 1;
    return [...rows].sort((a, b) => {
      const x = a[sort.key], y = b[sort.key];
      return sign * (typeof x === "string" ? x.localeCompare(y as string) : Number(x) - Number(y));
    });
  }, [rows, sort]);

  const first = Math.max(0, Math.floor(top / ROW) - OVERSCAN);
  const last = Math.min(sorted.length, Math.ceil((top + VIEW) / ROW) + OVERSCAN);

  return (
    <section className="rounded-lg border border-slate-800 bg-slate-900 shadow-sm">
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-800 px-3 py-2">
        <h2 className="text-xs font-semibold uppercase tracking-wide text-slate-400">{title}</h2>
        {toolbar}
      </div>
      <div className="overflow-x-auto">
        <div className="min-w-[640px]">
          <div role="row" className={`${GRID} h-8 border-b border-slate-800 text-xs font-semibold text-slate-200`}>
            {COLUMNS.map(({ key, label, right }) => (
              <button
                key={key} type="button" onClick={() => setSort((s) => ({ key, desc: s.key === key ? !s.desc : key !== "blockId" }))}
                className={`flex h-8 items-center gap-1 px-3 hover:text-white ${right ? "justify-end" : ""}`}
              >
                {label}
                {sort.key === key && (sort.desc ? <ArrowDown className="size-3" /> : <ArrowUp className="size-3" />)}
              </button>
            ))}
            <span />
          </div>
          <div style={{ height: VIEW }} className="overflow-y-auto" onScroll={(e) => setTop(e.currentTarget.scrollTop)}>
            {sorted.length === 0 && <p className="px-3 py-6 text-center text-xs text-slate-500">No block sessions match.</p>}
            <div style={{ paddingTop: first * ROW, paddingBottom: (sorted.length - last) * ROW }}>
              {sorted.slice(first, last).map((r) => (
                <div key={r.blockId} role="row" style={{ height: ROW }} className={`${GRID} border-b border-slate-800/60 hover:bg-slate-800/50`}>
                  <span className="truncate px-3 font-mono text-xs text-indigo-400">{r.blockId}</span>
                  <span className="px-3">
                    <span
                      className={
                        r.anomalous
                          ? "rounded-sm border border-red-900 bg-red-950 px-1.5 py-0.5 text-[11px] font-medium text-red-400"
                          : "rounded-sm border border-emerald-900 bg-emerald-950 px-1.5 py-0.5 text-[11px] font-medium text-emerald-400"
                      }
                    >
                      {r.anomalous ? "Anomalous" : "Normal"}
                    </span>
                  </span>
                  <span className="px-3 text-right font-mono text-xs tabular-nums">{r.confidence.toFixed(1)}%</span>
                  <span className="px-3 text-right font-mono text-xs tabular-nums">{r.events}</span>
                  <span className="px-3 text-right">
                    <button
                      type="button" onClick={() => onInspect(r.blockId)}
                      className="inline-flex items-center gap-0.5 rounded-sm border border-slate-700 px-2 py-0.5 text-xs text-slate-300 hover:border-indigo-500 hover:text-slate-100"
                    >
                      Inspect <ChevronRight className="size-3" />
                    </button>
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
