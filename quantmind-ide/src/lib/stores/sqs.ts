/**
 * SQS Store - Spread Quality Score Polling
 *
 * Manages polling for SQS data from the risk API.
 * Polls every 5 seconds per story requirements.
 *
 * Story: 4-7-spread-quality-score-sqs-system
 */

import { writable, derived } from 'svelte/store';

// Types for SQS data
export interface SQSResponse {
  symbol: string;
  sqs: number;
  threshold: number;
  allowed: boolean;
  is_hard_block: boolean;
  reason: string;
  strategy_type: string;
  current_spread: number;
  historical_avg_spread: number;
  bucket_sample_count: number;
  news_override_active: boolean;
  weekend_guard_active: boolean;
  warmup_active: boolean;
  evaluated_at_utc: string;
}

export interface SQSSummary {
  symbols: string[];
  evaluations: Record<string, SQSResponse>;
  evaluated_at_utc: string;
}

export interface SQSState {
  data: SQSSummary | null;
  loading: boolean;
  error: string | null;
  lastUpdated: Date | null;
}

// Initial state
const initialState: SQSState = {
  data: null,
  loading: false,
  error: null,
  lastUpdated: null
};

// Create the store
function createSQSStore() {
  const { subscribe, set, update } = writable<SQSState>(initialState);

  let pollingInterval: ReturnType<typeof setInterval> | null = null;

  async function fetchData() {
    update(state => ({ ...state, loading: true, error: null }));

    try {
      const response = await fetch('/api/risk/sqs/all');

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data: SQSSummary = await response.json();

      update(state => ({
        ...state,
        data,
        loading: false,
        lastUpdated: new Date()
      }));
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to fetch SQS data';
      update(state => ({
        ...state,
        loading: false,
        error: errorMessage
      }));
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
    startPolling,
    stopPolling,
    reset: () => set(initialState)
  };
}

// Export the store
export const sqsStore = createSQSStore();

// Derived stores for individual symbol data
export const sqsData = derived(sqsStore, $store => $store.data?.evaluations ?? null);
export const sqsSymbols = derived(sqsStore, $store => $store.data?.symbols ?? []);
export const sqsLoading = derived(sqsStore, $store => $store.loading);
export const sqsError = derived(sqsStore, $store => $store.error);
export const sqsLastUpdated = derived(sqsStore, $store => $store.lastUpdated);

// Helper to get alert level from SQS value
export function getSQSAlertLevel(sqs: number): 'normal' | 'warning' | 'critical' {
  if (sqs >= 0.80) return 'normal';
  if (sqs >= 0.50) return 'warning';
  return 'critical';
}

// Helper to get color for SQS value
export function getSQSColor(sqs: number): string {
  if (sqs >= 0.80) return '#22c55e';  // green
  if (sqs >= 0.50) return '#f0a500'; // yellow
  return '#ff3b3b';                   // red
}
