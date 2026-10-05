import { X } from "lucide-react";
import { useRef, useState } from "react";
import { inspectBlock, type Explanation } from "@/lib/api";

export type Inspection = { blockId: string; loading: boolean; explanation: Explanation | null; error: string | null; rawLogs: string[] };

// React escapes text by default, so LLM output and log lines are safe to render here.
export function InspectDrawer({ data, onClose }: { data: Inspection; onClose: () => void }) {
  const e = data.explanation;
  return (
    <>
      <div className="fixed inset-0 bg-black/50" onClick={onClose} />
      <aside className="fixed inset-y-0 right-0 flex w-full max-w-lg flex-col border-l border-slate-800 bg-slate-900">
        <div className="flex h-11 items-center justify-between border-b border-slate-800 px-4">
          <span className="truncate font-mono text-xs text-slate-200">{data.blockId}</span>
          <button type="button" onClick={onClose} aria-label="Close" className="text-slate-400 hover:text-white">
            <X className="size-4" />
          </button>
        </div>
        <div className="flex-1 space-y-4 overflow-y-auto p-4 text-xs">
          {data.loading && <p className="text-slate-400">Asking the model…</p>}
          {data.error && <p className="border border-red-900 bg-red-950 p-2 text-red-400">{data.error}</p>}
          {e && (
            <>
              <Field title="What happened">{e.summary}</Field>
              <Field title="Likely cause">{e.likely_cause} <span className="text-slate-500">(explanation confidence: {e.confidence})</span></Field>
              {e.incomplete_sequence && (
                <p className="border border-amber-900 bg-amber-950 p-2 text-amber-400">This block looks incomplete, so treat the cause as a guess.</p>
              )}
              <Field title="Check next">
                <ul className="list-disc space-y-1 pl-4">{e.suggested_checks.map((c, i) => <li key={i}>{c}</li>)}</ul>
              </Field>
            </>
          )}
          {data.rawLogs.length > 0 && (
            <Field title={`Log lines (${data.rawLogs.length})`}>
              <pre className="max-h-72 overflow-auto border border-slate-800 bg-slate-950 p-2 font-mono text-[11px] text-slate-400">{data.rawLogs.join("\n")}</pre>
            </Field>
          )}
        </div>
      </aside>
    </>
  );
}

function Field({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div>
      <h3 className="mb-1 text-[11px] font-semibold uppercase tracking-wide text-slate-500">{title}</h3>
      <div className="text-slate-200">{children}</div>
    </div>
  );
}

// Shared by the live view and the batch view: open(jobId, blockId) loads the explanation and shows the drawer.
export function useInspector() {
  const [data, setData] = useState<Inspection | null>(null);
  const latest = useRef(0);
  async function open(jobId: string, blockId: string) {
    const mine = ++latest.current; // a slow earlier click must not overwrite a newer one
    setData({ blockId, loading: true, explanation: null, error: null, rawLogs: [] });
    try {
      const result = await inspectBlock(jobId, blockId);
      if (mine === latest.current) setData({ blockId, loading: false, ...result });
    } catch (error) {
      if (mine === latest.current) setData({ blockId, loading: false, explanation: null, rawLogs: [], error: error instanceof Error ? error.message : "Inspect failed." });
    }
  }
  return { open, drawer: data && <InspectDrawer data={data} onClose={() => { latest.current++; setData(null); }} /> };
}
