// Talks to the Python server's /api/auth/* routes. The session lives in an HttpOnly cookie, so this code never sees the token.
export async function getMe(): Promise<{ email: string } | null> {
  const response = await fetch("/api/auth/me", { cache: "no-store" });
  return response.ok ? response.json() : null;
}

export async function sendAuth(action: "login" | "signup", email: string, password: string): Promise<void> {
  const response = await fetch(`/api/auth/${action}`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ email, password }) });
  if (!response.ok) throw new Error((await response.json().catch(() => ({}))).error || `Request failed (${response.status})`);
}

export const logout = () => fetch("/api/auth/logout", { method: "POST" });
