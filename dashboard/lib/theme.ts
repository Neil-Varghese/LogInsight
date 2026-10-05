"use client";

import { useEffect, useState } from "react";

// Dark is the default; the choice is remembered per browser and shared by the landing, login and dashboard pages.
export function useTheme(): [boolean, () => void] {
  const [light, setLight] = useState(false);
  useEffect(() => {
    // one-time read of browser storage after mount (reading it during render would not match the server-built HTML)
    // eslint-disable-next-line react-hooks/set-state-in-effect
    try { if (localStorage.getItem("theme") === "light") setLight(true); } catch {}
  }, []);
  useEffect(() => { document.documentElement.classList.toggle("light", light); }, [light]);
  function toggle() {
    const next = !light;
    setLight(next);
    try { localStorage.setItem("theme", next ? "light" : "dark"); } catch {}
  }
  return [light, toggle];
}
