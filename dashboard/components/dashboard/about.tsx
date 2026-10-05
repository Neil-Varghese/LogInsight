import { DEVELOPERS, STORY } from "@/lib/about";

// Settings > About: why the app exists and who built it. The text lives in lib/about.ts.
export function About() {
  return (
    <div className="space-y-4">
      <div>
        <h3 className="text-sm font-semibold text-slate-100">Why we built LogInsight</h3>
        <div className="mt-1 space-y-2 text-xs text-slate-300">
          {STORY.map((p, i) => <p key={i}>{p}</p>)}
        </div>
      </div>
      <div>
        <h3 className="text-sm font-semibold text-slate-100">The team</h3>
        <div className="mt-2 grid gap-2 sm:grid-cols-2 xl:grid-cols-4">
          {DEVELOPERS.map((d) => (
            <div key={d.name} className="rounded-md border border-slate-800 p-3">
              {/* eslint-disable-next-line @next/next/no-img-element -- static export, plain <img> is fine */}
              <img src={d.photo} alt={d.name} className="size-20 rounded-full border border-slate-800 object-cover" />
              <div className="mt-2 text-sm font-semibold text-slate-100">{d.name}</div>
              <div className="text-xs text-indigo-400">{d.role}</div>
              <p className="mt-1 text-xs text-slate-300">{d.bio}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
