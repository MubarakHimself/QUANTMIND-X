/**
 * API Endpoint Configuration
 *
 * Configuration for local/Cloudzy architecture split:
 * - UI runs locally
 * - Data services run on Cloudzy
 * - HMM API runs on Contabo
 */

// API endpoint configuration
function normalizeLoopbackUrl(url: string): string {
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

export const API_CONFIG = {
  // Local development
  get LOCAL_API_URL() {
    if (typeof window !== 'undefined') {
      return `${window.location.protocol}//${window.location.hostname}:8000`;
    }
    return 'http://127.0.0.1:8000';
  },

  // Cloudzy production (for data access) - use env var override
  get CLOUDZY_API_URL() {
    return normalizeLoopbackUrl(import.meta.env.VITE_CLOUDZY_API_URL || this.LOCAL_API_URL);
  },

  // Contabo HMM API - use env var override
  get CONTABO_HMM_API() {
    if (import.meta.env.VITE_CONTABO_HMM_API) {
      return normalizeLoopbackUrl(import.meta.env.VITE_CONTABO_HMM_API);
    }
    if (typeof window !== 'undefined') {
      return `${window.location.protocol}//${window.location.hostname}:8001`;
    }
    return 'http://127.0.0.1:8001';
  },

  // T1 API (Cloudzy — trading backend)
  get T1_API_URL() {
    if (import.meta.env.VITE_T1_API_URL) {
      return normalizeLoopbackUrl(import.meta.env.VITE_T1_API_URL);
    }
    if (typeof window !== 'undefined') {
      return window.location.origin;
    }
    return this.LOCAL_API_URL;
  },

  // T2 API (Contabo — HMM/research backend)
  get T2_API_URL() {
    if (import.meta.env.VITE_T2_API_URL) {
      return normalizeLoopbackUrl(import.meta.env.VITE_T2_API_URL);
    }
    if (typeof window !== 'undefined') {
      return window.location.origin;
    }
    return this.LOCAL_API_URL;
  },

  // Which API to use (VITE_API_URL override > build mode)
  get API_URL() {
    if (import.meta.env.VITE_API_URL) {
      return normalizeLoopbackUrl(import.meta.env.VITE_API_URL);
    }
    // Browser default: same-origin API routing.
    // In development this uses Vite's /api proxy, in production it supports
    // reverse-proxy deployments without hard-coded localhost ports.
    if (typeof window !== 'undefined') {
      return window.location.origin;
    }
    return import.meta.env.PROD
      ? this.CLOUDZY_API_URL
      : this.LOCAL_API_URL;
  },

  // API base path for backend endpoints (includes /api prefix)
  get API_BASE() {
    return `${this.API_URL}/api`;
  }
};

// Cloudzy prefixes — routes to T1
const CLOUDZY_PREFIXES = [
  '/api/trading', '/api/kill-switch', '/api/mt5',
  '/api/broker', '/api/paper-trading', '/api/sessions',
  '/api/router', '/api/metrics', '/api/settings/risk'
];

/**
 * Get base URL for an endpoint based on endpoint prefix routing.
 * Trading-related endpoints go to T1 (Cloudzy), others go to T2 (Contabo).
 * Non-CLOUDZY: in dev returns origin+/api (Vite proxy), in prod returns VITE_T2_API_URL.
 */
export function getBaseUrl(endpoint: string): string {
  if (CLOUDZY_PREFIXES.some(p => endpoint.startsWith(p))) {
    return API_CONFIG.T1_API_URL;
  }
  if (import.meta.env.VITE_T2_API_URL) {
    return API_CONFIG.T2_API_URL;
  }
  return API_CONFIG.API_BASE;
}

// Agent execution configuration
export const AGENT_CONFIG = {
  // Local agent endpoint (default)
  get LOCAL_AGENT_URL() {
    return `${API_CONFIG.API_URL}/api/v2/agents`;
  },

  // Remote agent endpoint (Cloudzy) - use env var override
  get REMOTE_AGENT_URL() {
    return import.meta.env.VITE_REMOTE_AGENT_URL || `${API_CONFIG.LOCAL_API_URL}/api/v2/agents`;
  },

  // Which agent URL to use (supports VITE_AGENT_URL override)
  get AGENT_URL() {
    // Allow environment variable override
    if (import.meta.env.VITE_AGENT_URL) {
      return import.meta.env.VITE_AGENT_URL;
    }
    return import.meta.env.PROD
      ? this.REMOTE_AGENT_URL
      : this.LOCAL_AGENT_URL;
  }
};
