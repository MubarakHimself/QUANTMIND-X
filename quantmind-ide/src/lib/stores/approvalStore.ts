/**
 * Approval Gate Store
 *
 * Manages approval gate state for workflow approvals.
 */

import { writable, derived, get } from 'svelte/store';
import type { Readable, Writable } from 'svelte/store';

// Types
export type ApprovalStatus = 'pending' | 'approved' | 'rejected';
export type GateType = 'stage_transition' | 'deployment' | 'risk_check' | 'manual_review';

export interface ApprovalGate {
  gate_id: string;
  workflow_id: string;
  workflow_type?: string;
  from_stage: string;
  to_stage: string;
  gate_type: GateType;
  status: ApprovalStatus;
  requester?: string;
  approver?: string;
  reason?: string;
  notes?: string;
  extra_data?: Record<string, any>;
  created_at: string;
  updated_at: string;
  approved_at?: string;
  rejected_at?: string;
}

export interface ApprovalActionRequest {
  approver: string;
  notes?: string;
}

export interface ApprovalActionResponse {
  success: boolean;
  gate_id: string;
  status: ApprovalStatus;
  message: string;
  approver: string;
  timestamp: string;
}

// State
interface ApprovalState {
  pendingGates: ApprovalGate[];
  gateHistory: ApprovalGate[];
  selectedGate: ApprovalGate | null;
  loading: boolean;
  error: string | null;
  pollingEnabled: boolean;
}

const initialState: ApprovalState = {
  pendingGates: [],
  gateHistory: [],
  selectedGate: null,
  loading: false,
  error: null,
  pollingEnabled: true,
};

// Create stores
const approvalState = writable<ApprovalState>(initialState);

// Derived stores
export const pendingApprovals: Readable<ApprovalGate[]> = derived(
  approvalState,
  ($state) => $state.pendingGates
);

export const approvalHistory: Readable<ApprovalGate[]> = derived(
  approvalState,
  ($state) => $state.gateHistory
);

export const selectedApproval: Readable<ApprovalGate | null> = derived(
  approvalState,
  ($state) => $state.selectedGate
);

export const hasPendingApprovals: Readable<boolean> = derived(
  approvalState,
  ($state) => $state.pendingGates.length > 0
);

export const pendingCount: Readable<number> = derived(
  approvalState,
  ($state) => $state.pendingGates.length
);

export const approvalLoading: Readable<boolean> = derived(
  approvalState,
  ($state) => $state.loading
);

export const approvalError: Readable<string | null> = derived(
  approvalState,
  ($state) => $state.error
);

// Polling interval reference
let pollingInterval: ReturnType<typeof setInterval> | null = null;

// Actions
export const approvalStore = {
  subscribe: approvalState.subscribe,

  /**
   * Start polling for pending approvals
   */
  startPolling(intervalMs: number = 10000): void {
    if (pollingInterval) {
      clearInterval(pollingInterval);
    }

    approvalState.update((s) => ({ ...s, pollingEnabled: true }));

    // Fetch immediately
    this.fetchPendingGates();

    // Set up polling
    pollingInterval = setInterval(() => {
      const state = get(approvalState);
      if (state.pollingEnabled) {
        this.fetchPendingGates();
      }
    }, intervalMs);
  },

  /**
   * Stop polling for pending approvals
   */
  stopPolling(): void {
    if (pollingInterval) {
      clearInterval(pollingInterval);
      pollingInterval = null;
    }
    approvalState.update((s) => ({ ...s, pollingEnabled: false }));
  },

  /**
   * Fetch pending approval gates
   */
  async fetchPendingGates(workflowId?: string): Promise<void> {
    try {
      const params = new URLSearchParams();
      if (workflowId) params.set('workflow_id', workflowId);
      params.set('limit', '50');

      const response = await fetch(`/api/approval-gates/pending?${params}`);
      if (!response.ok) throw new Error('Failed to fetch pending approvals');

      const data = await response.json();
      const gates: ApprovalGate[] = data.gates || [];

      approvalState.update((s) => ({
        ...s,
        pendingGates: gates,
        loading: false,
      }));
    } catch (err) {
      approvalState.update((s) => ({
        ...s,
        error: err instanceof Error ? err.message : 'Failed to fetch pending approvals',
        loading: false,
      }));
    }
  },

  /**
   * Fetch all gates for a specific workflow
   */
  async fetchWorkflowGates(workflowId: string): Promise<ApprovalGate[]> {
    approvalState.update((s) => ({ ...s, loading: true }));

    try {
      const response = await fetch(`/api/approval-gates/workflow/${workflowId}`);
      if (!response.ok) throw new Error('Failed to fetch workflow approvals');

      const data = await response.json();
      const gates: ApprovalGate[] = data.gates || [];

      approvalState.update((s) => ({
        ...s,
        gateHistory: gates,
        loading: false,
      }));

      return gates;
    } catch (err) {
      approvalState.update((s) => ({
        ...s,
        error: err instanceof Error ? err.message : 'Failed to fetch workflow approvals',
        loading: false,
      }));
      return [];
    }
  },

  /**
   * Fetch a specific gate by ID
   */
  async fetchGate(gateId: string): Promise<ApprovalGate | null> {
    try {
      const response = await fetch(`/api/approval-gates/${gateId}`);
      if (!response.ok) throw new Error('Failed to fetch approval gate');

      const gate: ApprovalGate = await response.json();

      approvalState.update((s) => ({
        ...s,
        selectedGate: gate,
      }));

      return gate;
    } catch (err) {
      approvalState.update((s) => ({
        ...s,
        error: err instanceof Error ? err.message : 'Failed to fetch approval gate',
      }));
      return null;
    }
  },

  /**
   * Approve an approval gate
   */
  async approveGate(gateId: string, request: ApprovalActionRequest): Promise<ApprovalActionResponse | null> {
    approvalState.update((s) => ({ ...s, loading: true }));

    try {
      const response = await fetch(`/api/approval-gates/${gateId}/approve`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(request),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Failed to approve gate');
      }

      const result: ApprovalActionResponse = await response.json();

      // Update local state
      approvalState.update((s) => ({
        ...s,
        loading: false,
        pendingGates: s.pendingGates.filter((g) => g.gate_id !== gateId),
        gateHistory: [
          ...s.gateHistory,
          { ...s.selectedGate, status: 'approved' } as ApprovalGate,
        ].filter(Boolean),
      }));

      return result;
    } catch (err) {
      approvalState.update((s) => ({
        ...s,
        loading: false,
        error: err instanceof Error ? err.message : 'Failed to approve gate',
      }));
      return null;
    }
  },

  /**
   * Reject an approval gate
   */
  async rejectGate(gateId: string, request: ApprovalActionRequest): Promise<ApprovalActionResponse | null> {
    approvalState.update((s) => ({ ...s, loading: true }));

    try {
      const response = await fetch(`/api/approval-gates/${gateId}/reject`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(request),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Failed to reject gate');
      }

      const result: ApprovalActionResponse = await response.json();

      // Update local state
      approvalState.update((s) => ({
        ...s,
        loading: false,
        pendingGates: s.pendingGates.filter((g) => g.gate_id !== gateId),
        gateHistory: [
          ...s.gateHistory,
          { ...s.selectedGate, status: 'rejected' } as ApprovalGate,
        ].filter(Boolean),
      }));

      return result;
    } catch (err) {
      approvalState.update((s) => ({
        ...s,
        loading: false,
        error: err instanceof Error ? err.message : 'Failed to reject gate',
      }));
      return null;
    }
  },

  /**
   * Select an approval gate
   */
  selectGate(gate: ApprovalGate | null): void {
    approvalState.update((s) => ({ ...s, selectedGate: gate }));
  },

  /**
   * Clear error
   */
  clearError(): void {
    approvalState.update((s) => ({ ...s, error: null }));
  },

  /**
   * Reset store
   */
  reset(): void {
    this.stopPolling();
    approvalState.set(initialState);
  },
};

export default approvalStore;
