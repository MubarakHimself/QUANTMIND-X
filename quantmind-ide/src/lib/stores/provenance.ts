/**
 * Provenance Chain Store
 *
 * State management for EA provenance tracking.
 */
import { writable } from 'svelte/store';
import { apiFetch } from '$lib/api';

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
      const endpoint = versionTag
        ? `/strategies/${strategyId}/versions/${versionTag}/provenance`
        : `/strategies/${strategyId}/provenance`;

      const data = await apiFetch<ProvenanceChain>(endpoint);

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
      const data = await apiFetch<{ chain: ProvenanceChain; answer: string | null }>('/strategies/provenance/query', {
        method: 'POST',
        body: JSON.stringify({ strategy_id: strategyId, query }),
      });

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
