/**
 * HMM Store
 *
 * Svelte store for Hidden Macro Model (HMM) state management.
 * Handles regime detection, predictions, and sync status.
 */

import { writable, derived, type Readable } from 'svelte/store';
import { PUBLIC_API_BASE } from '$env/static/public';

const apiBase = PUBLIC_API_BASE || '';

// ============================================================================
// Types
// ============================================================================

export interface HMMStatus {
  model_loaded: boolean;
  model_version: string | null;
  deployment_mode: string;
  hmm_weight: number;
  shadow_mode_active: boolean;
  contabo_version: string | null;
  cloudzy_version: string | null;
  version_mismatch: boolean;
  agreement_metrics: Record<string, any>;
  last_sync: string | null;
  sync_status: string | null;
}

export interface HMMRegime {
  regime: string;
  confidence: number;
  state: number;
  timestamp?: string;
}

export interface HMMPrediction {
  symbol: string;
  timeframe: string;
  regime: string;
  confidence: number;
  model_version: string;
  timestamp: string;
}

export interface HMMTrainingStatus {
  job_id: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  progress: number;
  message: string;
  started_at: string;
  completed_at?: string;
}

export interface RegimeTransition {
  from_regime: string;
  to_regime: string;
  timestamp: string;
  symbol: string;
  confidence: number;
}

// ============================================================================
// State
// ============================================================================

interface HMMState {
  status: HMMStatus | null;
  currentRegime: HMMRegime | null;
  predictions: HMMPrediction[];
  trainingStatus: HMMTrainingStatus | null;
  regimeHistory: RegimeTransition[];
  isLoading: boolean;
  error: string | null;
  lastUpdate: number;
}

const initialState: HMMState = {
  status: null,
  currentRegime: null,
  predictions: [],
  trainingStatus: null,
  regimeHistory: [],
  isLoading: false,
  error: null,
  lastUpdate: 0
};

function createHMMStore() {
  const { subscribe, set, update } = writable<HMMState>(initialState);

  // ============================================================================
  // API Calls
  // ============================================================================

  async function fetchStatus(): Promise<HMMStatus> {
    const response = await fetch(`${apiBase}/api/hmm/status`);
    if (!response.ok) {
      throw new Error(`Failed to fetch HMM status: ${response.statusText}`);
    }
    return await response.json();
  }

  async function fetchPrediction(symbol: string, timeframe: string): Promise<HMMRegime> {
    const response = await fetch(`${apiBase}/api/hmm/predict/${symbol}/${timeframe}`);
    if (!response.ok) {
      throw new Error(`Failed to fetch HMM prediction: ${response.statusText}`);
    }
    const data = await response.json();
    return {
      regime: data.reading?.regime || data.regime || 'UNKNOWN',
      confidence: data.reading?.confidence || data.confidence || 0,
      state: data.reading?.state || 0,
      timestamp: new Date().toISOString()
    };
  }

  async function fetchPredictions(limit: number = 50, symbol?: string): Promise<{predictions: HMMPrediction[], total: number}> {
    const params = new URLSearchParams({ limit: limit.toString(), offset: '0' });
    if (symbol) params.append('symbol', symbol);
    const response = await fetch(`${apiBase}/api/hmm/predictions?${params}`);
    if (!response.ok) {
      throw new Error(`Failed to fetch HMM predictions: ${response.statusText}`);
    }
    return await response.json();
  }

  async function startTraining(symbol?: string): Promise<HMMTrainingStatus> {
    const response = await fetch(`${apiBase}/api/hmm/train`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ symbol })
    });
    if (!response.ok) {
      throw new Error(`Failed to start training: ${response.statusText}`);
    }
    return await response.json();
  }

  async function triggerSync(): Promise<{ success: boolean; message: string }> {
    const response = await fetch(`${apiBase}/api/hmm/sync`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ verify_checksum: true })
    });
    if (!response.ok) {
      throw new Error(`Failed to trigger sync: ${response.statusText}`);
    }
    return await response.json();
  }

  async function setMode(mode: string): Promise<{ success: boolean; message: string }> {
    const response = await fetch(`${apiBase}/api/hmm/mode`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ mode })
    });
    if (!response.ok) {
      throw new Error(`Failed to set mode: ${response.statusText}`);
    }
    return await response.json();
  }

  // ============================================================================
  // Actions
  // ============================================================================

  async function loadStatus() {
    update(state => ({ ...state, isLoading: true, error: null }));
    try {
      const status = await fetchStatus();
      update(state => ({
        ...state,
        status,
        isLoading: false,
        lastUpdate: Date.now()
      }));
    } catch (error) {
      update(state => ({
        ...state,
        isLoading: false,
        error: error instanceof Error ? error.message : 'Failed to load status'
      }));
    }
  }

  async function loadCurrentRegime(symbol: string = 'EURUSD', timeframe: string = 'H1') {
    update(state => ({ ...state, isLoading: true, error: null }));
    try {
      const currentRegime = await fetchPrediction(symbol, timeframe);
      update(state => ({
        ...state,
        currentRegime,
        isLoading: false,
        lastUpdate: Date.now()
      }));
    } catch (error) {
      update(state => ({
        ...state,
        isLoading: false,
        error: error instanceof Error ? error.message : 'Failed to load regime'
      }));
    }
  }

  async function loadPredictions(symbol?: string) {
    update(state => ({ ...state, isLoading: true, error: null }));
    try {
      const result = await fetchPredictions(50, symbol);
      update(state => ({
        ...state,
        predictions: result.predictions,
        isLoading: false,
        lastUpdate: Date.now()
      }));
    } catch (error) {
      update(state => ({
        ...state,
        isLoading: false,
        error: error instanceof Error ? error.message : 'Failed to load predictions'
      }));
    }
  }

  async function train(symbol?: string) {
    update(state => ({ ...state, isLoading: true, error: null }));
    try {
      const trainingStatus = await startTraining(symbol);
      update(state => ({
        ...state,
        trainingStatus,
        isLoading: false,
        lastUpdate: Date.now()
      }));
    } catch (error) {
      update(state => ({
        ...state,
        isLoading: false,
        error: error instanceof Error ? error.message : 'Failed to start training'
      }));
    }
  }

  async function sync() {
    update(state => ({ ...state, isLoading: true, error: null }));
    try {
      const result = await triggerSync();
      await loadStatus(); // Reload status after sync
      update(state => ({ ...state, isLoading: false }));
      return result;
    } catch (error) {
      update(state => ({
        ...state,
        isLoading: false,
        error: error instanceof Error ? error.message : 'Failed to sync'
      }));
      throw error;
    }
  }

  async function changeMode(mode: string) {
    update(state => ({ ...state, isLoading: true, error: null }));
    try {
      const result = await setMode(mode);
      await loadStatus(); // Reload status after mode change
      update(state => ({ ...state, isLoading: false }));
      return result;
    } catch (error) {
      update(state => ({
        ...state,
        isLoading: false,
        error: error instanceof Error ? error.message : 'Failed to change mode'
      }));
      throw error;
    }
  }

  function reset() {
    set(initialState);
  }

  // ============================================================================
  // Derived Stores
  // ============================================================================

  const regimeColor = derived(
    { subscribe },
    ($state: HMMState) => {
      const regime = $state.currentRegime?.regime || 'UNKNOWN';
      const colors: Record<string, string> = {
        'TRENDING_LOW_VOL': '#10b981',  // green
        'TRENDING_HIGH_VOL': '#3b82f6', // blue
        'RANGING_LOW_VOL': '#f59e0b',   // amber
        'RANGING_HIGH_VOL': '#ef4444',  // red
        'BREAKOUT': '#8b5cf6',          // purple
        'UNKNOWN': '#6b7280'            // gray
      };
      return colors[regime] || colors['UNKNOWN'];
    }
  );

  const isSynced = derived(
    { subscribe },
    ($state: HMMState) => !$state.status?.version_mismatch && $state.status?.model_loaded
  );

  const canTrain = derived(
    { subscribe },
    ($state: HMMState) => $state.status?.model_loaded && !$state.trainingStatus?.status
  );

  // Auto-refresh every 30 seconds
  let refreshInterval: ReturnType<typeof setInterval> | null = null;

  function startAutoRefresh(intervalMs: number = 30000) {
    if (refreshInterval) clearInterval(refreshInterval);
    refreshInterval = setInterval(() => {
      loadStatus();
      loadCurrentRegime();
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
    loadStatus,
    loadCurrentRegime,
    loadPredictions,
    train,
    sync,
    changeMode,
    reset,
    startAutoRefresh,
    stopAutoRefresh,
    regimeColor,
    isSynced,
    canTrain
  };
}

export const hmmStore = createHMMStore();
export default hmmStore;
