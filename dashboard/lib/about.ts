// THE FILE THE TEAM EDITS. Everything on Settings > About comes from here; no other code needs to change.
//
// To add yourself:
//   1. Put your photo in dashboard/public/team/ (e.g. alex.jpg), square works best.
//   2. Replace your placeholder entry below: name, role, a short bio, and photo: "/team/alex.jpg".
//   3. Run `npm run build` in dashboard/ so port 8000 shows the change.

export type Developer = { name: string; role: string; bio: string; photo: string };

// What led to building LogInsight. Each string is one paragraph.
export const STORY: string[] = [
  "Placeholder: describe the problem that started this project, for example how much log data a Hadoop (HDFS) cluster produces and how hard it is to spot trouble by reading it by hand.",
  "Placeholder: describe the idea, for example teaching an LSTM model what normal block activity looks like so that unusual behaviour is flagged automatically, in real time.",
  "Placeholder: describe the goal, for example a dashboard that a person on call can open and understand in seconds.",
];

export const DEVELOPERS: Developer[] = [
  { name: "Developer One", role: "Role / focus area", bio: "Placeholder: two or three sentences about yourself and what you built.", photo: "/team/dev1.svg" },
  { name: "Developer Two", role: "Role / focus area", bio: "Placeholder: two or three sentences about yourself and what you built.", photo: "/team/dev2.svg" },
  { name: "Developer Three", role: "Role / focus area", bio: "Placeholder: two or three sentences about yourself and what you built.", photo: "/team/dev3.svg" },
  { name: "Developer Four", role: "Role / focus area", bio: "Placeholder: two or three sentences about yourself and what you built.", photo: "/team/dev4.svg" },
];
