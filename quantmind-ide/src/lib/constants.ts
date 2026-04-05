// Shared API configuration constants

function normalizeBase(url: string): string {
  return url.replace(/\/+$/, '');
}

function normalizeLoopback(url: string): string {
  if (typeof window === 'undefined') {
    return url;
  }
  try {
    const parsed = new URL(url);
    if (parsed.hostname === 'localhost' || parsed.hostname === '127.0.0.1') {
      parsed.protocol = window.location.protocol;
      parsed.hostname = window.location.hostname;
      return parsed.toString().replace(/\/$/, '');
    }
  } catch {
    return url;
  }
  return url;
}

function defaultApiBase(): string {
  if (typeof window !== 'undefined') {
    // Browser default: same-origin API so dev proxy and production reverse proxy
    // work without hardcoded localhost ports.
    return window.location.origin;
  }
  return 'http://127.0.0.1:8000';
}

const envApiBase = typeof import.meta !== 'undefined' ? import.meta.env?.VITE_API_URL as string | undefined : undefined;
export const API_BASE = normalizeBase(normalizeLoopback(envApiBase || defaultApiBase()));
export const WS_BASE = normalizeBase(
  API_BASE.replace(/^http:/, 'ws:').replace(/^https:/, 'wss:')
);
