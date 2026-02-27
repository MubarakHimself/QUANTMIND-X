// EA Store - Manages Expert Advisor lifecycle state
import { writable, derived, get } from 'svelte/store';
import type { Writable } from 'svelte/store';

// Types
export interface ExpertAdvisor {
  name: string;
  file_path: string;
  size: number;
  modified: number;
  status?: 'created' | 'validating' | 'valid' | 'invalid' | 'backtesting' | 'deployed' | 'stopped';
  validation_errors?: string[];
  validation_warnings?: string[];
}

export interface EACreateParams {
  strategy_name: string;
  ea_name?: string;
  parameters?: Record<string, any>;
}

export interface EABacktestParams {
  ea_name: string;
  symbol?: string;
  timeframe?: string;
  date_from?: string;
  date_to?: string;
  deposit?: number;
}

export interface EADeployParams {
  ea_name: string;
  account_id?: string;
}

export interface EAStoreState {
  eas: ExpertAdvisor[];
  loading: boolean;
  error: string | null;
  selectedEA: string | null;
}

// API base URL
const API_BASE = 'http://localhost:8000';

// Initial state
const initialState: EAStoreState = {
  eas: [],
  loading: false,
  error: null,
  selectedEA: null
};

// Create writable store
function createEAStore(): Writable<EAStoreState> & {
  listEAs: () => Promise<void>;
  createEA: (params: EACreateParams) => Promise<void>;
  validateEA: (eaName: string) => Promise<void>;
  backtestEA: (params: EABacktestParams) => Promise<void>;
  deployPaper: (params: EADeployParams) => Promise<void>;
  stopEA: (eaName: string, environment?: string) => Promise<void>;
  selectEA: (eaName: string | null) => void;
  clearError: () => void;
} {
  const { subscribe, set, update } = writable<EAStoreState>(initialState);

  return {
    subscribe,
    set,

    // List all EAs
    listEAs: async () => {
      update(state => ({ ...state, loading: true, error: null }));
      try {
        const response = await fetch(`${API_BASE}/api/ea/list`);
        if (!response.ok) throw new Error('Failed to list EAs');
        const data = await response.json();
        update(state => ({
          ...state,
          eas: data.eas || [],
          loading: false
        }));
      } catch (error) {
        update(state => ({
          ...state,
          loading: false,
          error: error instanceof Error ? error.message : 'Failed to list EAs'
        }));
      }
    },

    // Create new EA
    createEA: async (params: EACreateParams) => {
      update(state => ({ ...state, loading: true, error: null }));
      try {
        const response = await fetch(`${API_BASE}/api/ea/create`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(params)
        });
        if (!response.ok) {
          const error = await response.json();
          throw new Error(error.detail || 'Failed to create EA');
        }
        const data = await response.json();
        update(state => ({
          ...state,
          eas: [...state.eas, {
            name: data.ea_name,
            file_path: data.file_path,
            size: 0,
            modified: Date.now() / 1000,
            status: 'created'
          }],
          loading: false
        }));
      } catch (error) {
        update(state => ({
          ...state,
          loading: false,
          error: error instanceof Error ? error.message : 'Failed to create EA'
        }));
        throw error;
      }
    },

    // Validate EA
    validateEA: async (eaName: string) => {
      update(state => ({
        ...state,
        eas: state.eas.map(ea =>
          ea.name === eaName ? { ...ea, status: 'validating' } : ea
        ),
        error: null
      }));
      try {
        const response = await fetch(`${API_BASE}/api/ea/validate`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ ea_name: eaName })
        });
        if (!response.ok) {
          const error = await response.json();
          throw new Error(error.detail || 'Validation failed');
        }
        const data = await response.json();
        update(state => ({
          ...state,
          eas: state.eas.map(ea =>
            ea.name === eaName ? {
              ...ea,
              status: data.valid ? 'valid' : 'invalid',
              validation_errors: data.errors || [],
              validation_warnings: data.warnings || []
            } : ea
          )
        }));
      } catch (error) {
        update(state => ({
          ...state,
          eas: state.eas.map(ea =>
            ea.name === eaName ? { ...ea, status: 'invalid' } : ea
          ),
          error: error instanceof Error ? error.message : 'Validation failed'
        }));
        throw error;
      }
    },

    // Backtest EA
    backtestEA: async (params: EABacktestParams) => {
      update(state => ({
        ...state,
        eas: state.eas.map(ea =>
          ea.name === params.ea_name ? { ...ea, status: 'backtesting' } : ea
        ),
        error: null
      }));
      try {
        const response = await fetch(`${API_BASE}/api/ea/backtest`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(params)
        });
        if (!response.ok) {
          const error = await response.json();
          throw new Error(error.detail || 'Backtest failed');
        }
        update(state => ({
          ...state,
          eas: state.eas.map(ea =>
            ea.name === params.ea_name ? { ...ea, status: 'valid' } : ea
          )
        }));
      } catch (error) {
        update(state => ({
          ...state,
          eas: state.eas.map(ea =>
            ea.name === params.ea_name ? { ...ea, status: 'valid' } : ea
          ),
          error: error instanceof Error ? error.message : 'Backtest failed'
        }));
        throw error;
      }
    },

    // Deploy to paper trading
    deployPaper: async (params: EADeployParams) => {
      update(state => ({
        ...state,
        error: null
      }));
      try {
        const response = await fetch(`${API_BASE}/api/ea/deploy-paper`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(params)
        });
        if (!response.ok) {
          const error = await response.json();
          throw new Error(error.detail || 'Deployment failed');
        }
        update(state => ({
          ...state,
          eas: state.eas.map(ea =>
            ea.name === params.ea_name ? { ...ea, status: 'deployed' } : ea
          )
        }));
      } catch (error) {
        update(state => ({
          ...state,
          error: error instanceof Error ? error.message : 'Deployment failed'
        }));
        throw error;
      }
    },

    // Stop EA
    stopEA: async (eaName: string, environment = 'paper') => {
      update(state => ({
        ...state,
        error: null
      }));
      try {
        const response = await fetch(`${API_BASE}/api/ea/stop`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ ea_name: eaName, environment })
        });
        if (!response.ok) {
          const error = await response.json();
          throw new Error(error.detail || 'Failed to stop EA');
        }
        update(state => ({
          ...state,
          eas: state.eas.map(ea =>
            ea.name === eaName ? { ...ea, status: 'stopped' } : ea
          )
        }));
      } catch (error) {
        update(state => ({
          ...state,
          error: error instanceof Error ? error.message : 'Failed to stop EA'
        }));
        throw error;
      }
    },

    // Select EA
    selectEA: (eaName: string | null) => {
      update(state => ({ ...state, selectedEA: eaName }));
    },

    // Clear error
    clearError: () => {
      update(state => ({ ...state, error: null }));
    }
  };
}

// Export store instance
export const eaStore = createEAStore();

// Derived stores
export const eas: Readable<ExpertAdvisor[]> = derived(eaStore, $store => $store.eas);
export const eaLoading: Readable<boolean> = derived(eaStore, $store => $store.loading);
export const eaError: Readable<string | null> = derived(eaStore, $store => $store.error);
export const selectedEA: Readable<string | null> = derived(eaStore, $store => $store.selectedEA);

export default eaStore;
