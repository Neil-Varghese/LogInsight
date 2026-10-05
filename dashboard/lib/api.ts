import type { Session } from "@/components/dashboard/sessions-table";

export type TestSet = { name: string; path: string };
export type Explanation = {
  summary: string; likely_cause: string; confidence: string;
  incomplete_sequence: boolean; suggested_checks: string[];
};

async function json(response: Response) {
  // Session ended (or never started): go to the sign-in page. The server enforces this; the redirect is just the friendly part.
  if (response.status === 401 && typeof window !== "undefined") window.location.replace("/login/");
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `Request failed (${response.status})`);
  return data;
}

export const getTestSets = (): Promise<TestSet[]> => fetch("/api/test-sets").then(json);

// Start a run, poll until it finishes, then fetch every block (the API caps a page at 100).
export async function runDetection(filename: string, onProgress: (text: string) => void) {
  const started = Date.now();
  const { job_id } = await json(await fetch("/api/run", {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ filename }),
  }));
  let result;
  while (!result) {
    const job = await json(await fetch(`/api/jobs/${job_id}`));
    if (job.stage === "error") throw new Error(job.error || "Prediction failed.");
    if (job.stage === "complete") result = job.result;
    else { onProgress(`${job.stage === "preprocessing" ? "Parsing logs" : "Classifying"}: ${Math.round(job.progress ?? 0)}%`); await new Promise((r) => setTimeout(r, 500)); }
  }
  const rows: Session[] = [];
  for (let page = 1; ; page++) {
    const data = await json(await fetch(`/api/jobs/${job_id}/blocks?limit=100&page=${page}`));
    for (const b of data.blocks) rows.push({ blockId: b.block_id, anomalous: b.is_anomalous, confidence: b.anomaly_score * 100, events: b.sequence_len });
    if (rows.length >= data.total || data.blocks.length === 0) break;
  }
  const total = result.normal_logs + result.anomalous_logs;
  return {
    jobId: job_id as string, rows,
    latencySeconds: (Date.now() - started) / 1000, // ponytail: client-side wall clock incl. polling (~0.5s granularity); add server timing if exact numbers matter
    kpis: {
      total, anomalous: result.anomalous_logs, normal: result.normal_logs,
      rate: total ? (result.anomalous_logs / total) * 100 : 0,
    },
  };
}

export async function inspectBlock(jobId: string, blockId: string) {
  const base = `/api/jobs/${jobId}/blocks/${encodeURIComponent(blockId)}`;
  const [detail, explained] = await Promise.all([fetch(base), fetch(`${base}/explain`)]);
  const rawLogs: string[] = detail.ok ? (await detail.json()).raw_logs : [];
  const data = await explained.json();
  return { rawLogs, explanation: (data.explanation ?? null) as Explanation | null, error: (data.error ?? null) as string | null };
}

// ---- live monitor (the always-on watcher in the Python server) ----
export type Perf = {
  model_load_s: number | null; ticks: number; tick_avg_ms: number; tick_p95_ms: number;
  capacity_lines_per_s: number; score_ms_per_block: number; db_bytes: number; uptime_s: number;
};
export type Status = {
  model_loaded: boolean; error: string | null; last_ingest: number | null; started: number; file: string;
  lines: number; unmatched: number; blocks: number; backlog_bytes: number; now: number;
  rates: { t: number; rate: number }[]; perf: Perf;
};
export type Summary = {
  total: number; anomalous: number; normal: number; rate: number; bucket_seconds: number;
  series: { t: number; normal: number; anomalous: number; rate?: number }[];
  histogram: { bin: number; count: number }[];
  top_events: { event_id: string; template: string; anomalous_pct: number; normal_pct: number }[];
};

export type Traffic = {
  total_bytes: number; transfers: number; bucket_seconds: number;
  series: { t: number; bytes: number }[];
  top_sources: { ip: string; bytes: number; transfers: number }[];
};

export const getTraffic = (): Promise<Traffic> => fetch("/api/monitor/traffic", { cache: "no-store" }).then(json);
export const getStatus = (): Promise<Status> => fetch("/api/monitor/status", { cache: "no-store" }).then(json);
export const getSummary = (threshold: number): Promise<Summary> =>
  fetch(`/api/monitor/summary?threshold=${threshold}`, { cache: "no-store" }).then(json);
// Every block: [id, score 0-1, events, last_seen]. The Monitor tab applies the threshold to these itself.
export type Score = [string, number, number, number | null];
export const getScores = (): Promise<{ scores: Score[] }> => fetch("/api/monitor/scores", { cache: "no-store" }).then(json);
export const resetMonitor = () => fetch("/api/monitor/reset", { method: "POST" }).then(json);

// Top 100 blocks by score at the given cut-off (the API caps a page at 100).
export async function getLiveBlocks(threshold: number, filter: "anomalous" | "all") {
  const data = await json(await fetch(`/api/jobs/monitor/blocks?limit=100&threshold=${threshold}&filter=${filter}`, { cache: "no-store" }));
  const rows: Session[] = data.blocks.map((b: { block_id: string; is_anomalous: boolean; anomaly_score: number; sequence_len: number }) => (
    { blockId: b.block_id, anomalous: b.is_anomalous, confidence: b.anomaly_score * 100, events: b.sequence_len }));
  return { rows, total: data.total as number };
}

export type IpRow = { ip: string; lines: number; first: number; last: number; bytes_sent: number; active: boolean };
export type Ips = { newest: number | null; window: number; active_count: number; ips: IpRow[] };
export type SearchResult = {
  total_lines: number; truncated: boolean; lines: string[]; total_blocks: number;
  blocks: { block_id: string; lines: number; score: number | null }[];
};
export type SimMode = "replay" | "docker";
export type SimInfo = {
  containers: string | null; lines_per_second: string | null; anomaly_rate: string | null;
  running: boolean; mode: SimMode | null; elapsed_s: number | null; exit_code: number | null;
  file: string; tail: string[];
};

export type FeedLine = { seq: number; t: number | null; level: string; sev: number; src: string; text: string; block: string | null };
export type Feed = { generation: number; latest: number; counts: Record<string, number>; lines: FeedLine[] };
export const getFeed = (after: number): Promise<Feed> => fetch(`/api/monitor/feed?after=${after}`, { cache: "no-store" }).then(json);
export const getIps =(): Promise<Ips> => fetch("/api/monitor/ips", { cache: "no-store" }).then(json);
export const startSim = (mode: SimMode): Promise<SimInfo> =>
  fetch("/api/sim/start", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ mode }) }).then(json);
export const stopSim = (): Promise<SimInfo> => fetch("/api/sim/stop", { method: "POST" }).then(json);
export const getSimInfo = (): Promise<SimInfo> => fetch("/api/sim/info", { cache: "no-store" }).then(json);
export const searchLogs = (ip: string, from: number | null, to: number | null): Promise<SearchResult> => {
  const query = new URLSearchParams({ ip, from: from === null ? "" : String(from), to: to === null ? "" : String(to) });
  return fetch(`/api/monitor/search?${query}`, { cache: "no-store" }).then(json);
};
