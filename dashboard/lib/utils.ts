import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

// Log timestamps are UTC (HDFS logs carry no timezone): "2008-11-09 20:35:18".
export const utc = (epochSeconds: number) => new Date(epochSeconds * 1000).toISOString().slice(0, 19).replace("T", " ");
