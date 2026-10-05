import type { NextConfig } from "next";

const dev = process.env.NODE_ENV === "development";

const nextConfig: NextConfig = {
  // Production: a static export that the Python server serves, so /api/* is already same-origin.
  // Dev (port 3000): forward /api/* to the Python server instead (rewrites don't exist in an export).
  ...(dev
    ? { async rewrites() { return [{ source: "/api/:path*", destination: "http://127.0.0.1:8000/api/:path*" }]; } }
    : { output: "export" as const }),
};

export default nextConfig;
