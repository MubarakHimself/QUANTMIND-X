/**
 * WFA Store - Walk-Forward Analysis Calibrator State Management
 *
 * Handles WFA window calibration data, history, and Sharpe ratio analysis.
 */

import { writable, derived } from 'svelte/store';

const API_BASE = '/api';

// ============================================================================
// Types
// ============================================================================

export interface WfaCalibration {
  optimal_window: number;
  window_candidates: number[];
  sharpe_ratios: number[];
  current_window: number;
  window_type?: string;
  last_calibration: string;
  regime: string;
}

export interface WfaHistoryEntry {
  timestamp: string;
  optimal_window: number;
  window_type: string;
  avg_regime_interval: number;
  sharpe_ratio: number;
  regime: string;
}

export interface WfaHistory {
  calibrations: WfaHistoryEntry[];
  total: number;
}

interface WfaState {
  calibration: WfaCalibration | null;
  history: WfaHistoryEntry[];
  isLoading: boolean;
  error: string | null;
  lastUpdate: number;
}

const initialState: WfaState = {
  calibration: null,
  history: [],
  isLoading: false,
  error: null,
  lastUpdate: 0
};

function createWfaStore() {
  const { subscribe, set, update } = writable<WfaState>(initialState);

  // ============================================================================
  // API Calls
  // ============================================================================

  async function fetchCalibration(): Promise<WfaCalibration> {
    const response = await fetch(`${API_BASE}/hmm/wfa/calibration`);
    if (!response.ok) {
      throw new Error(`Failed to fetch WFA calibration: ${response.statusText}`);
    }
    return await response.json();
  }

  async function fetchHistory(limit: number = 20): Promise<WfaHistory> {
    const response = await fetch(`${API_BASE}/hmm/wfa/history?limit=${limit}`);
    if (!response.ok) {
      throw new Error(`Failed to fetch WFA history: ${response.statusText}`);
    }
    return await response.json();
  }

  // ============================================================================
  // Actions
  // ============================================================================

  async function loadCalibration() {
    update(state => ({ ...state, isLoading: true, error: null }));
    try {
      const calibration = await fetchCalibration();
      update(state => ({
        ...state,
        calibration,
        isLoading: false,
        lastUpdate: Date.now()
      }));
    } catch (error) {
      update(state => ({
        ...state,
        isLoading: false,
        error: error instanceof Error ? error.message : 'Failed to load WFA calibration'
      }));
    }
  }

  async function loadHistory(limit: number = 20) {
    update(state => ({ ...state, isLoading: true, error: null }));
    try {
      const history = await fetchHistory(limit);
      update(state => ({
        ...state,
        history: history.calibrations,
        isLoading: false,
        lastUpdate: Date.now()
      }));
    } catch (error) {
      update(state => ({
        ...state,
        isLoading: false,
        error: error instanceof Error ? error.message : 'Failed to load WFA history'
      }));
    }
  }

  async function refresh() {
    await Promise.all([loadCalibration(), loadHistory()]);
  }

  function reset() {
    set(initialState);
  }

  // ============================================================================
  // Derived Stores
  // ============================================================================

  /**
   * Best Sharpe ratio from current calibration
   */
  const bestSharpe = derived(
    { subscribe },
    ($state: WfaState) => {
      if (!$state.calibration) return null;
      const ratios = $state.calibration.sharpe_ratios;
      return ratios.length > 0 ? Math.max(...ratios) : null;
    }
  );

  /**
   * Current window deviation from optimal
   */
  const windowDeviation = derived(
    { subscribe },
    ($state: WfaState) => {
      if (!$state.calibration) return null;
      return $state.calibration.current_window - $state.calibration.optimal_window;
    }
  );

  /**
   * Whether current window matches optimal
   */
  const isWindowOptimal = derived(
    { subscribe },
    ($state: WfaState) => {
      if (!$state.calibration) return false;
      return $state.calibration.current_window === $state.calibration.optimal_window;
    }
  );

  /**
   * Formatted last calibration timestamp
   */
  const lastCalibrationFormatted = derived(
    { subscribe },
    ($state: WfaState) => {
      if (!$state.calibration?.last_calibration) return null;
      const date = new Date($state.calibration.last_calibration);
      return date.toLocaleString();
    }
  );

  /**
   * Regime color for display
   */
  const regimeColor = derived(
    { subscribe },
    ($state: WfaState) => {
      const regime = $state.calibration?.regime || 'UNKNOWN';
      const colors: Record<string, string> = {
        'TRENDING': '#10b981',
        'RANGING': '#f59e0b',
        'BREAKOUT': '#8b5cf6',
        'VOLATILE': '#ef4444',
        'UNKNOWN': '#6b7280'
      };
      return colors[regime] || colors['UNKNOWN'];
    }
  );

  // Auto-refresh every 60 seconds
  let refreshInterval: ReturnType<typeof setInterval> | null = null;

  function startAutoRefresh(intervalMs: number = 60000) {
    if (refreshInterval) clearInterval(refreshInterval);
    refreshInterval = setInterval(() => {
      refresh();
    }, intervalMs);
  }

  function stopAutoRefresh() {
    if (refreshInterval) {
      clearInterval(refreshInterval);
      refreshInterval = null;
    }
  }

  return {
    subscribe,
    loadCalibration,
    loadHistory,
    refresh,
    reset,
    startAutoRefresh,
    stopAutoRefresh,
    bestSharpe,
    windowDeviation,
    isWindowOptimal,
    lastCalibrationFormatted,
    regimeColor
  };
}

export const wfaStore = createWfaStore();
export default wfaStore;
