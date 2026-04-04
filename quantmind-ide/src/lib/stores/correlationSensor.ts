/**
 * Correlation Sensor Store
 *
 * Manages polling for correlation sensor data from the risk API:
 * - max_eigenvalue and RMT threshold analysis
 * - Correlation matrices for M5 and H1 timeframes
 * - Regime classification
 *
 * Polls every 5 seconds for real-time regime detection.
 */

import { writable, derived } from 'svelte/store';

// Types for correlation sensor data
export interface CorrelationSensorData {
  max_eigenvalue: number;
  rmt_threshold: number;
  is_correlated: boolean;
  regime: 'CORRELATED' | 'UNCORRELATED' | 'NEUTRAL';
  m5_matrix: number[][];
  h1_matrix: number[][];
  eigenvalues: number[];
  symbols: string[];
  timestamp: number;
}

export interface CorrelationMatrixData {
  timeframe: 'M5' | 'H1';
  matrix: number[][];
  symbols: string[];
  timestamp: number;
}

export interface CorrelationSensorState {
  data: CorrelationSensorData | null;
  loading: boolean;
  error: string | null;
  lastUpdated: Date | null;
}

// Initial state
const initialState: CorrelationSensorState = {
  data: null,
  loading: false,
  error: null,
  lastUpdated: null
};

// Create the store
function createCorrelationSensorStore() {
  const { subscribe, set, update } = writable<CorrelationSensorState>(initialState);

  let pollingInterval: ReturnType<typeof setInterval> | null = null;

  async function fetchData() {
    update(state => ({ ...state, loading: true, error: null }));

    try {
      const response = await fetch('/api/risk/correlation/sensor');

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data: CorrelationSensorData = await response.json();

      update(state => ({
        ...state,
        data,
        loading: false,
        lastUpdated: new Date()
      }));
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to fetch correlation sensor data';
      update(state => ({
        ...state,
        loading: false,
        error: errorMessage
      }));
    }
  }

  async function fetchMatrix(timeframe: 'M5' | 'H1'): Promise<CorrelationMatrixData | null> {
    try {
      const response = await fetch(`/api/risk/correlation/matrix/${timeframe}`);

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data: CorrelationMatrixData = await response.json();
      return data;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to fetch correlation matrix';
      update(state => ({
        ...state,
        error: errorMessage
      }));
      return null;
    }
  }

  function startPolling(intervalMs: number = 5000) {
    // Fetch immediately
    fetchData();

    // Then poll at interval
    if (pollingInterval) {
      clearInterval(pollingInterval);
    }
    pollingInterval = setInterval(fetchData, intervalMs);
  }

  function stopPolling() {
    if (pollingInterval) {
      clearInterval(pollingInterval);
      pollingInterval = null;
    }
  }

  return {
    subscribe,
    fetch: fetchData,
    fetchMatrix,
    startPolling,
    stopPolling,
    reset: () => set(initialState)
  };
}

// Export the store
export const correlationSensorStore = createCorrelationSensorStore();

// Derived stores for individual data points
export const correlationData = derived(
  correlationSensorStore,
  $store => $store.data
);

export const correlationLoading = derived(
  correlationSensorStore,
  $store => $store.loading
);

export const correlationError = derived(
  correlationSensorStore,
  $store => $store.error
);

export const correlationLastUpdated = derived(
  correlationSensorStore,
  $store => $store.lastUpdated
);

// Derived store for regime status
export const correlationRegime = derived(
  correlationSensorStore,
  $store => $store.data?.regime ?? null
);

// Derived store for max eigenvalue status
export const eigenvalueStatus = derived(
  correlationSensorStore,
  $store => {
    const data = $store.data;
    if (!data) return null;

    const ratio = data.max_eigenvalue / data.rmt_threshold;

    if (ratio > 1.2) {
      return 'critical'; // Well above threshold - correlated
    } else if (ratio > 1.0) {
      return 'warning'; // Near threshold - neutral
    } else {
      return 'normal'; // Below threshold - uncorrelated
    }
  }
);

// Color helpers for eigenvalue status
export const eigenvalueColor = derived(
  eigenvalueStatus,
  $status => {
    switch ($status) {
      case 'critical':
        return '#ff3b3b'; // Red - correlated
      case 'warning':
        return '#ffb700'; // Amber - neutral
      case 'normal':
        return '#00d4ff'; // Cyan - uncorrelated
      default:
        return 'rgba(255, 255, 255, 0.5)';
    }
  }
);

// Regime color helper
export const regimeColor = derived(
  correlationRegime,
  $regime => {
    switch ($regime) {
      case 'CORRELATED':
        return '#ff3b3b';
      case 'UNCORRELATED':
        return '#00d4ff';
      case 'NEUTRAL':
        return '#ffb700';
      default:
        return 'rgba(255, 255, 255, 0.5)';
    }
  }
);
