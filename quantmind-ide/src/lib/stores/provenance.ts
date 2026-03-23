/**
 * Provenance Chain Store
 *
 * State management for EA provenance tracking.
 */
import { writable } from 'svelte/store';

export interface ProvenanceNode {
  stage: string;
  timestamp: string;
  actor: string;
  status: string;
  details: Record<string, unknown>;
}

export interface ProvenanceChain {
  strategy_id: string;
  version_tag: string;
  chain: ProvenanceNode[];
  source_url: string | null;
  total_stages: number;
}

export interface ProvenanceState {
  chain: ProvenanceChain | null;
  loading: boolean;
  error: string | null;
}

const initialState: ProvenanceState = {
  chain: null,
  loading: false,
  error: null,
};

function createProvenanceStore() {
  const { subscribe, set, update } = writable<ProvenanceState>(initialState);

  async function loadProvenance(strategyId: string, versionTag?: string) {
    update(state => ({ ...state, loading: true, error: null }));

    try {
      const url = versionTag
        ? `/api/strategies/${strategyId}/versions/${versionTag}/provenance`
        : `/api/strategies/${strategyId}/provenance`;

      const response = await fetch(url);

      if (!response.ok) {
        throw new Error(`Failed to load provenance: ${response.statusText}`);
      }

      const data = await response.json();

      update(state => ({
        ...state,
        chain: data,
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

  async function queryProvenance(strategyId: string, query: string) {
    update(state => ({ ...state, loading: true, error: null }));

    try {
      const response = await fetch('/api/strategies/provenance/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ strategy_id: strategyId, query }),
      });

      if (!response.ok) {
        throw new Error(`Failed to query provenance: ${response.statusText}`);
      }

      const data = await response.json();

      update(state => ({
        ...state,
        chain: data.chain,
        loading: false,
      }));

      return data.answer;
    } catch (error) {
      update(state => ({
        ...state,
        error: error instanceof Error ? error.message : 'Unknown error',
        loading: false,
      }));
      return null;
    }
  }

  function clearProvenance() {
    set(initialState);
  }

  return {
    subscribe,
    loadProvenance,
    queryProvenance,
    clearProvenance,
  };
}

export const provenanceStore = createProvenanceStore();