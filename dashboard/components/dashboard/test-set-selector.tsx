import { FileText, Play } from "lucide-react";
import { cn } from "@/lib/utils";

export function TestSetSelector({
  files, selected, busy, onSelect, onRun,
}: { files: string[]; selected: string; busy: boolean; onSelect: (name: string) => void; onRun: () => void }) {
  return (
    <section className="rounded-lg border border-slate-800 bg-slate-900 shadow-sm p-3">
      <h2 className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-400">Test Set</h2>
      <div className="grid grid-cols-2 gap-2 lg:grid-cols-4">
        {files.map((name) => (
          <button
            key={name}
            type="button"
            onClick={() => onSelect(name)}
            className={cn(
              "flex items-center gap-2 rounded-sm border px-2.5 py-2 text-left font-mono text-xs transition-colors",
              name === selected
                ? "border-indigo-500 bg-indigo-950 text-slate-100"
                : "border-slate-800 bg-slate-950 text-slate-400 hover:border-slate-600",
            )}
          >
            <FileText className="size-3.5 shrink-0" />
            <span className="truncate">{name}</span>
          </button>
        ))}
      </div>
      <button
        type="button"
        onClick={onRun}
        disabled={busy || !selected}
        className="mt-2 flex h-9 w-full items-center justify-center gap-2 rounded-sm bg-indigo-600 text-sm font-medium text-white transition-colors hover:bg-indigo-500 disabled:opacity-50"
      >
        <Play className="size-3.5 fill-current" />
        {busy ? "Running…" : "Run Anomaly Detection"}
      </button>
    </section>
  );
}
