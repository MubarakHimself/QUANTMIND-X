/**
 * A/B Race Board Store
 *
 * State management for A/B variant comparison.
 * Implements 5-second polling for real-time updates.
 */
import { writable, derived } from 'svelte/store';

export interface VariantMetrics {
  pnl: number;
  trade_count: number;
  drawdown: number;
  sharpe: number;
  win_rate: number;
  avg_profit: number;
  avg_loss: number;
  profit_factor: number;
  max_consecutive_wins: number;
  max_consecutive_losses: number;
}

export interface StatisticalSignificance {
  p_value: number;
  is_significant: boolean;
  winner: string | null;
  confidence_level: number;
  sample_size_a: number;
  sample_size_b: number;
}

export interface ABComparison {
  strategy_id: string;
  variant_a: string;
  variant_b: string;
  metrics_a: VariantMetrics;
  metrics_b: VariantMetrics;
  statistical_significance: StatisticalSignificance | null;
  timestamp: string;
}

export interface ABComparisonState {
  comparison: ABComparison | null;
  loading: boolean;
  error: string | null;
  polling: boolean;
}

const initialState: ABComparisonState = {
  comparison: null,
  loading: false,
  error: null,
  polling: false,
};

function createABRaceStore() {
  const { subscribe, set, update } = writable<ABComparisonState>(initialState);

  let pollInterval: ReturnType<typeof setInterval> | null = null;

  async function loadComparison(strategyId: string, variantA: string, variantB: string) {
    update(state => ({ ...state, loading: true, error: null }));

    try {
      const response = await fetch(
        `/api/strategies/variants/${strategyId}/compare?variant_a=${encodeURIComponent(variantA)}&variant_b=${encodeURIComponent(variantB)}`
      );

      if (!response.ok) {
        throw new Error(`Failed to load comparison: ${response.statusText}`);
      }

      const data = await response.json();

      update(state => ({
        ...state,
        comparison: data,
        loading: false,
      }));
    } catch (error) {
      update(state => ({
        ...state,
        error: error instanceof Error ? error.message : 'Unknown error',
        loading: false,
      }));
    }
  }

  function startPolling(strategyId: string, variantA: string, variantB: string) {
    if (pollInterval) {
      clearInterval(pollInterval);
    }

    update(state => ({ ...state, polling: true }));

    // Poll every 5 seconds
    pollInterval = setInterval(() => {
      loadComparison(strategyId, variantA, variantB);
    }, 5000);
  }

  function stopPolling() {
    if (pollInterval) {
      clearInterval(pollInterval);
      pollInterval = null;
    }

    update(state => ({ ...state, polling: false }));
  }

  function clearComparison() {
    stopPolling();
    set(initialState);
  }

  return {
    subscribe,
    loadComparison,
    startPolling,
    stopPolling,
    clearComparison,
  };
}

export const abRaceStore = createABRaceStore();

// Derived store for winner detection
export const winningVariant = derived(abRaceStore, $store => {
  if (!$store.comparison?.statistical_significance) {
    return null;
  }
  const sig = $store.comparison.statistical_significance;
  if (sig.is_significant && sig.winner) {
    return sig.winner;
  }
  return null;
});