// Threat ranking of log levels. HDFS writes Log4j levels, which sit on the RFC 5424 syslog severity scale (0-7, lower = worse).
// The backend turns each line's level into one of these numbers; this table says how to show it.
export type Tier = { priority: string; label: string; levels: string; badge: string };

export const TIERS: Record<number, Tier> = {
  2: { priority: "P1", label: "Critical", levels: "FATAL", badge: "border-red-700 bg-red-950 text-red-300" },
  3: { priority: "P2", label: "Error", levels: "ERROR", badge: "border-orange-700 bg-orange-950 text-orange-300" },
  4: { priority: "P3", label: "Warning", levels: "WARN", badge: "border-amber-700 bg-amber-950 text-amber-300" },
  5: { priority: "P4", label: "Unclassified", levels: "unknown level, needs a look", badge: "border-violet-700 bg-violet-950 text-violet-300" },
  6: { priority: "P5", label: "Info", levels: "INFO", badge: "border-slate-600 bg-slate-800 text-slate-300" },
  7: { priority: "P6", label: "Debug", levels: "DEBUG, TRACE", badge: "border-slate-800 bg-slate-900 text-slate-500" },
};

export const TIER_ORDER = [2, 3, 4, 5, 6, 7]; // most to least threatening
export const tierOf = (severity: number) => TIERS[severity] ?? TIERS[5];
