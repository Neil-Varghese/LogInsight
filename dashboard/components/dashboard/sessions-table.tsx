import { ChevronRight } from "lucide-react";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";

export type Session = { blockId: string; anomalous: boolean; confidence: number; events: number };

export function SessionsTable({ rows, onInspect }: { rows: Session[]; onInspect: (blockId: string) => void }) {
  return (
    <section className="border border-slate-800 bg-slate-900">
      <h2 className="border-b border-slate-800 px-3 py-2 text-xs font-semibold uppercase tracking-wide text-slate-400">
        Classified Block Sessions
      </h2>
      <Table>
        <TableHeader>
          <TableRow className="border-slate-800 hover:bg-transparent">
            <TableHead className="h-8 px-3 text-xs text-slate-400">Block ID</TableHead>
            <TableHead className="h-8 px-3 text-xs text-slate-400">Status</TableHead>
            <TableHead className="h-8 px-3 text-right text-xs text-slate-400">Anomaly Confidence (%)</TableHead>
            <TableHead className="h-8 px-3 text-right text-xs text-slate-400">Events</TableHead>
            <TableHead className="h-8 px-3" />
          </TableRow>
        </TableHeader>
        <TableBody>
          {rows.map((r) => (
            <TableRow key={r.blockId} className="border-slate-800 hover:bg-slate-800/50">
              <TableCell className="px-3 py-1.5 font-mono text-xs">{r.blockId}</TableCell>
              <TableCell className="px-3 py-1.5">
                <span
                  className={
                    r.anomalous
                      ? "rounded-sm border border-red-900 bg-red-950 px-1.5 py-0.5 text-[11px] font-medium text-red-400"
                      : "rounded-sm border border-emerald-900 bg-emerald-950 px-1.5 py-0.5 text-[11px] font-medium text-emerald-400"
                  }
                >
                  {r.anomalous ? "Anomalous" : "Normal"}
                </span>
              </TableCell>
              <TableCell className="px-3 py-1.5 text-right font-mono text-xs tabular-nums">{r.confidence.toFixed(1)}%</TableCell>
              <TableCell className="px-3 py-1.5 text-right font-mono text-xs tabular-nums">{r.events}</TableCell>
              <TableCell className="px-3 py-1.5 text-right">
                <button
                  type="button"
                  onClick={() => onInspect(r.blockId)}
                  className="inline-flex items-center gap-0.5 rounded-sm border border-slate-700 px-2 py-0.5 text-xs text-slate-300 hover:border-indigo-500 hover:text-white"
                >
                  Inspect <ChevronRight className="size-3" />
                </button>
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </section>
  );
}
