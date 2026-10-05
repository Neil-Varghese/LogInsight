import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

// Log timestamps are UTC (HDFS logs carry no timezone): "2008-11-09 20:35:18".
export const utc = (epochSeconds: number) => new Date(epochSeconds * 1000).toISOString().slice(0, 19).replace("T", " ");

export const kb = (bytes: number) => (bytes < 1024 ? `${bytes} B` : bytes < 1048576 ? `${(bytes / 1024).toFixed(1)} KB` : `${(bytes / 1048576).toFixed(1)} MB`);
export const ago = (seconds: number) => (seconds < 5 ? "just now" : seconds < 90 ? `${Math.round(seconds)}s ago` : `${Math.round(seconds / 60)}m ago`);
