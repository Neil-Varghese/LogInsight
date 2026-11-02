// THE FILE THE TEAM EDITS. Everything on Settings > About comes from here; no other code needs to change.
//
// To add yourself:
//   1. Put your photo in dashboard/public/team/ (e.g. alex.jpg), square works best.
//   2. Replace your placeholder entry below: name, role, a short bio, and photo: "/team/alex.jpg".
//   3. Run `npm run build` in dashboard/ so port 8000 shows the change.

export type Developer = { name: string; role: string; bio: string; photo: string };

// What led to building LogInsight. Each string is one paragraph.
export const STORY: string[] = [
  "Modern systems produce billions of log events a day, far too many to read by hand when diagnosing faults or auditing activity.",
  "Existing anomaly detectors act as black boxes: they raise alerts without a cause, which leads to alert fatigue and slow response.",
  "LogInsight uses an LSTM to flag anomalous HDFS blocks and an LLM to explain them, with a Docker log simulator and a live dashboard.",
  "The goal is to help SOC analysts and DevOps engineers investigate anomalies with more confidence and less domain expertise.",
];

export const DEVELOPERS: Developer[] = [
  { name: "Neil Varghese Abraham", role: "1BY23CS144", bio: "", photo: "/team/dev1.svg" },
  { name: "Nikitha R Shenoy", role: "1BY23CS148", bio: "", photo: "/team/dev2.svg" },
  { name: "Sai Mansi Maddali", role: "1BY23CS196", bio: "", photo: "/team/dev3.svg" },
  { name: "Trinabh Chadha", role: "1BY23CS275", bio: "", photo: "/team/dev4.svg" },
];
