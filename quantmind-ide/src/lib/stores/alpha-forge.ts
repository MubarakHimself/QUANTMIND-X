/**
 * Alpha Forge Pipeline Store
 *
 * Manages pipeline state for the Alpha Forge Pipeline Status Board.
 * Provides real-time polling for active pipeline runs.
 */

import { writable, derived, get } from 'svelte/store';
import type { Readable, Writable } from 'svelte/store';

// =============================================================================
// Types
// =============================================================================

export type PipelineStage =
  | 'VIDEO_INGEST'
  | 'RESEARCH'
  | 'TRD'
  | 'DEVELOPMENT'
  | 'COMPILE'
  | 'BACKTEST'
  | 'VALIDATION'
  | 'EA_LIFECYCLE'
  | 'APPROVAL';

export type StageStatus = 'waiting' | 'running' | 'passed' | 'failed';

export type ApprovalStatus = 'pending_review' | 'approved' | 'rejected' | 'none';

export interface PipelineStageInfo {
  stage: PipelineStage;
  status: StageStatus;
  started_at?: string;
  completed_at?: string;
  error?: string;
}

export interface PipelineRun {
  strategy_id: string;
  strategy_name: string;
  current_stage: PipelineStage;
  stage_status: StageStatus;
  stages: PipelineStageInfo[];
  approval_status: ApprovalStatus;
  started_at: string;
  updated_at: string;
  metadata?: Record<string, unknown>;
}

export interface PipelineStatusResponse {
  runs: PipelineRun[];
  total: number;
  active_count: number;
}

export interface PendingApproval {
  pending_count: number;
  strategies: Array<{
    strategy_id: string;
    strategy_name: string;
    current_stage: PipelineStage;
    updated_at: string;
  }>;
}

// =============================================================================
// Constants
// =============================================================================

const POLLING_INTERVAL_MS = 5000; // 5 seconds as per story spec

// =============================================================================
// State
// =============================================================================

interface PipelineState {
  runs: PipelineRun[];
  selectedRun: PipelineRun | null;
  pendingApprovals: PendingApproval['strategies'];
  loading: boolean;
  error: string | null;
  pollingEnabled: boolean;
  lastUpdated: string | null;
}

const initialState: PipelineState = {
  runs: [],
  selectedRun: null,
  pendingApprovals: [],
  loading: false,
  error: null,
  pollingEnabled: true,
  lastUpdated: null,
};

// Create main store
const pipelineState = writable<PipelineState>(initialState);

// =============================================================================
// Derived Stores
// =============================================================================

export const pipelineRuns: Readable<PipelineRun[]> = derived(
  pipelineState,
  ($state) => $state.runs
);

export const activeRuns: Readable<PipelineRun[]> = derived(
  pipelineState,
  ($state) => $state.runs.filter((r) => r.stage_status === 'running')
);

export const pendingApprovalStrategies: Readable<PendingApproval['strategies']> = derived(
  pipelineState,
  ($state) => $state.pendingApprovals
);

export const pendingApprovalCount: Readable<number> = derived(
  pipelineState,
  ($state) => $state.pendingApprovals.length
);

export const selectedPipelineRun: Readable<PipelineRun | null> = derived(
  pipelineState,
  ($state) => $state.selectedRun
);

export const pipelineLoading: Readable<boolean> = derived(
  pipelineState,
  ($state) => $state.loading
);

export const pipelineError: Readable<string | null> = derived(
  pipelineState,
  ($state) => $state.error
);

export const pipelineLastUpdated: Readable<string | null> = derived(
  pipelineState,
  ($state) => $state.lastUpdated
);

export const activeRunCount: Readable<number> = derived(
  pipelineState,
  ($state) => $state.runs.filter((r) => r.stage_status === 'running').length
);

// =============================================================================
// Polling
// =============================================================================

let pollingInterval: ReturnType<typeof setInterval> | null = null;

// =============================================================================
// Store Actions
// =============================================================================

export const alphaForgeStore = {
  subscribe: pipelineState.subscribe,

  /**
   * Start polling for pipeline status (5s interval)
   */
  startPolling(intervalMs: number = POLLING_INTERVAL_MS): void {
    if (pollingInterval) {
      clearInterval(pollingInterval);
    }

    pipelineState.update((s) => ({ ...s, pollingEnabled: true }));

    // Fetch immediately
    this.fetchPipelineStatus();

    // Set up polling
    pollingInterval = setInterval(() => {
      const state = get(pipelineState);
      if (state.pollingEnabled) {
        this.fetchPipelineStatus();
      }
    }, intervalMs);
  },

  /**
   * Stop polling for pipeline status
   */
  stopPolling(): void {
    if (pollingInterval) {
      clearInterval(pollingInterval);
      pollingInterval = null;
    }
    pipelineState.update((s) => ({ ...s, pollingEnabled: false }));
  },

  /**
   * Fetch pipeline status from API
   */
  async fetchPipelineStatus(activeOnly: boolean = false): Promise<void> {
    pipelineState.update((s) => ({ ...s, loading: true }));

    try {
      const params = new URLSearchParams();
      if (activeOnly) params.set('active_only', 'true');
      params.set('limit', '50');

      const response = await fetch(`/api/pipeline/status?${params}`);
      if (!response.ok) throw new Error('Failed to fetch pipeline status');

      const data: PipelineStatusResponse = await response.json();

      pipelineState.update((s) => ({
        ...s,
        runs: data.runs || [],
        loading: false,
        lastUpdated: new Date().toISOString(),
      }));
    } catch (err) {
      pipelineState.update((s) => ({
        ...s,
        loading: false,
        error: err instanceof Error ? err.message : 'Failed to fetch pipeline status',
      }));
    }
  },

  /**
   * Fetch pending approvals for the approval badge
   */
  async fetchPendingApprovals(): Promise<void> {
    try {
      const response = await fetch('/api/pipeline/pending-approvals');
      if (!response.ok) throw new Error('Failed to fetch pending approvals');

      const data: PendingApproval = await response.json();

      pipelineState.update((s) => ({
        ...s,
        pendingApprovals: data.strategies || [],
      }));
    } catch (err) {
      pipelineState.update((s) => ({
        ...s,
        error: err instanceof Error ? err.message : 'Failed to fetch pending approvals',
      }));
    }
  },

  /**
   * Fetch a specific pipeline run by strategy ID
   */
  async fetchPipelineRun(strategyId: string): Promise<PipelineRun | null> {
    pipelineState.update((s) => ({ ...s, loading: true }));

    try {
      const response = await fetch(`/api/pipeline/status/${strategyId}`);
      if (!response.ok) throw new Error('Failed to fetch pipeline run');

      const data = await response.json();
      const run = data.run as PipelineRun;

      pipelineState.update((s) => ({
        ...s,
        selectedRun: run,
        loading: false,
      }));

      return run;
    } catch (err) {
      pipelineState.update((s) => ({
        ...s,
        loading: false,
        error: err instanceof Error ? err.message : 'Failed to fetch pipeline run',
      }));
      return null;
    }
  },

  /**
   * Select a pipeline run
   */
  selectRun(run: PipelineRun | null): void {
    pipelineState.update((s) => ({ ...s, selectedRun: run }));
  },

  /**
   * Clear error
   */
  clearError(): void {
    pipelineState.update((s) => ({ ...s, error: null }));
  },

  /**
   * Reset store
   */
  reset(): void {
    this.stopPolling();
    pipelineState.set(initialState);
  },
};

export default alphaForgeStore;