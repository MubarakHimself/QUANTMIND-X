/**
 * API Endpoint Configuration
 *
 * Configuration for local/Cloudzy architecture split:
 * - UI runs locally
 * - Data services run on Cloudzy
 * - HMM API runs on Contabo
 */

// API endpoint configuration
export const API_CONFIG = {
  // Local development
  LOCAL_API_URL: 'http://localhost:8000',

  // Cloudzy production (for data access)
  CLOUDZY_API_URL: 'http://<cloudzy-ip>:8000',

  // Contabo HMM API
  CONTABO_HMM_API: 'http://155.133.27.86:8001',

  // Which API to use (determined by build mode)
  get API_URL() {
    return import.meta.env.PROD
      ? this.CLOUDZY_API_URL
      : this.LOCAL_API_URL;
  },

  // API base path for backend endpoints (includes /api prefix)
  get API_BASE() {
    return `${this.API_URL}/api`;
  }
};

// Agent execution configuration
export const AGENT_CONFIG = {
  // Local agent endpoint (default)
  LOCAL_AGENT_URL: '/api/v2/agents',

  // Remote agent endpoint (Cloudzy)
  REMOTE_AGENT_URL: 'http://<cloudzy-ip>:8000/api/v2/agents',

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
