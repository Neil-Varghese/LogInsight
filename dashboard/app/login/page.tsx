"use client";

import Link from "next/link";
import { useEffect, useState, type FormEvent } from "react";
import { ArrowLeft, Moon, Sun } from "lucide-react";
import { getMe, sendAuth } from "@/lib/auth";
import { useTheme } from "@/lib/theme";

export default function LoginPage() {
  const [light, toggleTheme] = useTheme();
  const [signup, setSignup] = useState(false);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    // read the ?mode=signup link from the landing page after mount (a static export has no query string at build time)
    // eslint-disable-next-line react-hooks/set-state-in-effect
    if (new URLSearchParams(window.location.search).get("mode") === "signup") setSignup(true);
    getMe().then((user) => { if (user) window.location.replace("/app/"); }); // already signed in
  }, []);

  async function submit(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError("");
    try {
      await sendAuth(signup ? "signup" : "login", email, password);
      window.location.replace("/app/");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Something went wrong.");
      setBusy(false);
    }
  }

  const field = "mt-1 w-full rounded-md border border-slate-700 bg-slate-950 px-3 py-2 text-slate-100 outline-none focus:border-indigo-400";
  return (
    <div className="grid min-h-screen place-items-center bg-gradient-to-br from-[#2b2f8f] to-[#14121f] px-4">
      <Link href="/" className="absolute left-4 top-4 flex h-8 items-center gap-1.5 rounded-md border border-white/30 bg-white/15 px-3 text-sm font-medium text-white transition-colors hover:bg-white/30">
        <ArrowLeft className="size-3.5" /> Back to home
      </Link>
      <button type="button" onClick={toggleTheme} aria-label="Toggle light/dark theme" title="Toggle light/dark theme" className="absolute right-4 top-4 grid size-8 place-items-center rounded-md border border-white/30 bg-white/15 text-white transition-colors hover:bg-white/30">
        {light ? <Moon className="size-3.5" /> : <Sun className="size-3.5" />}
      </button>
      <form onSubmit={submit} className="w-full max-w-sm rounded-xl border border-slate-800 bg-slate-900 p-6 shadow-xl">
        <Link href="/" className="text-lg font-semibold tracking-tight text-slate-100">LogInsight</Link>
        <h1 className="mt-4 text-xl font-semibold text-slate-100">{signup ? "Create your account" : "Log in"}</h1>
        <label className="mt-5 block text-slate-400">Email
          <input type="email" required autoComplete="email" value={email} onChange={(e) => setEmail(e.target.value)} className={field} />
        </label>
        <label className="mt-4 block text-slate-400">Password
          <input type="password" required minLength={signup ? 8 : undefined} autoComplete={signup ? "new-password" : "current-password"} value={password} onChange={(e) => setPassword(e.target.value)} className={field} />
        </label>
        {signup && <p className="mt-1 text-xs text-slate-500">At least 8 characters.</p>}
        {error && <p role="alert" className="mt-4 rounded-md border border-red-900 bg-red-950 px-3 py-2 text-red-400">{error}</p>}
        <button type="submit" disabled={busy} className="mt-5 w-full rounded-md bg-[#2b2f8f] py-2 font-semibold text-white transition-colors hover:bg-[#3a3fb5] disabled:opacity-60">
          {busy ? "Please wait…" : signup ? "Sign up" : "Log in"}
        </button>
        <p className="mt-4 text-center text-slate-400">
          {signup ? "Already have an account?" : "New here?"}{" "}
          <button type="button" onClick={() => { setSignup(!signup); setError(""); }} className="font-semibold text-indigo-400 hover:underline">{signup ? "Log in" : "Create an account"}</button>
        </p>
      </form>
    </div>
  );
}
