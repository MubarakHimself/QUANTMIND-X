/**
 * FlowForge Store
 *
 * Manages Prefect workflow state for FlowForge canvas.
 * Story 11.5: FlowForge Canvas — Prefect Kanban & Node Graph
 */

import { writable, derived, get } from 'svelte/store';
import type { Readable } from 'svelte/store';

// Types matching the API response
export interface PrefectTask {
  id: string;
  name: string;
  state: 'PENDING' | 'RUNNING' | 'COMPLETED' | 'CANCELLED' | 'FAILED';
  x: number;
  y: number;
}

export interface PrefectWorkflow {
  id: string;
  flow_id: string;
  name: string;
  department: string;
  state: WorkflowState;
  started_at: string | null;
  duration_seconds: number;
  completed_steps: number;
  total_steps: number;
  next_step: string;
  tasks: PrefectTask[];
  dependencies: { from: string; to: string }[];
}

export type WorkflowState = 'PENDING' | 'RUNNING' | 'PENDING_REVIEW' | 'DONE' | 'CANCELLED' | 'EXPIRED_REVIEW';

export interface WorkflowsByState {
  PENDING: PrefectWorkflow[];
  RUNNING: PrefectWorkflow[];
  PENDING_REVIEW: PrefectWorkflow[];
  DONE: PrefectWorkflow[];
  CANCELLED: PrefectWorkflow[];
  EXPIRED_REVIEW: PrefectWorkflow[];
}

// Kanban columns
export const KANBAN_COLUMNS: { id: WorkflowState; label: string }[] = [
  { id: 'PENDING', label: 'Pending' },
  { id: 'RUNNING', label: 'Running' },
  { id: 'PENDING_REVIEW', label: 'Pending Review' },
  { id: 'DONE', label: 'Done' },
  { id: 'CANCELLED', label: 'Cancelled' },
  { id: 'EXPIRED_REVIEW', label: 'Expired Review' },
];

// State
interface FlowForgeState {
  workflows: PrefectWorkflow[];
  workflowsByState: WorkflowsByState;
  selectedWorkflow: PrefectWorkflow | null;
  selectedWorkflowForNodeGraph: PrefectWorkflow | null;
  loading: boolean;
  error: string | null;
  showNodeGraph: boolean;
}

const initialState: FlowForgeState = {
  workflows: [],
  workflowsByState: {
    PENDING: [],
    RUNNING: [],
    PENDING_REVIEW: [],
    DONE: [],
    CANCELLED: [],
    EXPIRED_REVIEW: [],
  },
  selectedWorkflow: null,
  selectedWorkflowForNodeGraph: null,
  loading: false,
  error: null,
  showNodeGraph: false,
};

// Create store
const flowForgeState = writable<FlowForgeState>(initialState);

// Derived stores
export const workflows: Readable<PrefectWorkflow[]> = derived(
  flowForgeState,
  ($state) => $state.workflows
);

export const workflowsByState: Readable<WorkflowsByState> = derived(
  flowForgeState,
  ($state) => $state.workflowsByState
);

export const flowforgeSelectedWorkflow: Readable<PrefectWorkflow | null> = derived(
  flowForgeState,
  ($state) => $state.selectedWorkflow
);

export const selectedWorkflowForNodeGraph: Readable<PrefectWorkflow | null> = derived(
  flowForgeState,
  ($state) => $state.selectedWorkflowForNodeGraph
);

export const showNodeGraph: Readable<boolean> = derived(
  flowForgeState,
  ($state) => $state.showNodeGraph
);

export const flowforgeLoading: Readable<boolean> = derived(flowForgeState, ($state) => $state.loading);
export const flowforgeError: Readable<string | null> = derived(flowForgeState, ($state) => $state.error);

// Actions
export const flowForgeStore = {
  subscribe: flowForgeState.subscribe,

  /**
   * Fetch all Prefect workflows
   */
  async fetchWorkflows(): Promise<void> {
    flowForgeState.update((s) => ({ ...s, loading: true, error: null }));

    try {
      const response = await fetch('/api/prefect/workflows');
      if (!response.ok) throw new Error('Failed to fetch workflows');

      const data = await response.json();
      const workflows: PrefectWorkflow[] = data.workflows || [];
      const byState: WorkflowsByState = data.by_state || initialState.workflowsByState;

      flowForgeState.update((s) => ({
        ...s,
        loading: false,
        workflows,
        workflowsByState: byState,
      }));
    } catch (err) {
      flowForgeState.update((s) => ({
        ...s,
        loading: false,
        error: err instanceof Error ? err.message : 'Failed to fetch workflows',
      }));
    }
  },

  /**
   * Fetch single workflow details (including task graph)
   */
  async fetchWorkflowDetails(workflowId: string): Promise<PrefectWorkflow | null> {
    try {
      const response = await fetch(`/api/prefect/workflows/${workflowId}`);
      if (!response.ok) throw new Error('Failed to fetch workflow details');

      const workflow: PrefectWorkflow = await response.json();
      return workflow;
    } catch (err) {
      console.error('Failed to fetch workflow details:', err);
      return null;
    }
  },

  /**
   * Select a workflow for details
   */
  selectWorkflow(workflow: PrefectWorkflow | null): void {
    flowForgeState.update((s) => ({ ...s, selectedWorkflow: workflow }));
  },

  /**
   * Open node graph for a workflow
   */
  openNodeGraph(workflow: PrefectWorkflow): void {
    flowForgeState.update((s) => ({
      ...s,
      selectedWorkflowForNodeGraph: workflow,
      showNodeGraph: true,
    }));
  },

  /**
   * Close node graph
   */
  closeNodeGraph(): void {
    flowForgeState.update((s) => ({
      ...s,
      showNodeGraph: false,
      selectedWorkflowForNodeGraph: null,
    }));
  },

  /**
   * Cancel a specific workflow (per-card kill switch)
   */
  async cancelWorkflow(workflowId: string): Promise<boolean> {
    try {
      const response = await fetch(`/api/prefect/workflows/${workflowId}/cancel`, {
        method: 'POST',
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Failed to cancel workflow');
      }

      const result = await response.json();

      // Update local state to reflect the change
      flowForgeState.update((s) => {
        const updatedWorkflows = s.workflows.map((w) =>
          w.id === workflowId ? { ...w, state: 'CANCELLED' as WorkflowState } : w
        );

        const updatedByState = { ...s.workflowsByState };
        // Move from RUNNING to CANCELLED
        updatedByState.RUNNING = updatedByState.RUNNING.filter((w) => w.id !== workflowId);
        updatedByState.CANCELLED = [
          ...updatedByState.CANCELLED,
          { ...updatedWorkflows.find((w) => w.id === workflowId)!, state: 'CANCELLED' as WorkflowState },
        ];

        return {
          ...s,
          workflows: updatedWorkflows,
          workflowsByState: updatedByState,
        };
      });

      return true;
    } catch (err) {
      flowForgeState.update((s) => ({
        ...s,
        error: err instanceof Error ? err.message : 'Failed to cancel workflow',
      }));
      return false;
    }
  },

  /**
   * Resume a cancelled workflow
   */
  async resumeWorkflow(workflowId: string): Promise<boolean> {
    try {
      const response = await fetch(`/api/prefect/workflows/${workflowId}/resume`, {
        method: 'POST',
      });

      if (!response.ok) throw new Error('Failed to resume workflow');

      // Update local state
      flowForgeState.update((s) => {
        const updatedWorkflows = s.workflows.map((w) =>
          w.id === workflowId ? { ...w, state: 'RUNNING' as WorkflowState } : w
        );

        const updatedByState = { ...s.workflowsByState };
        updatedByState.CANCELLED = updatedByState.CANCELLED.filter((w) => w.id !== workflowId);
        updatedByState.RUNNING = [
          ...updatedByState.RUNNING,
          { ...updatedWorkflows.find((w) => w.id === workflowId)!, state: 'RUNNING' as WorkflowState },
        ];

        return {
          ...s,
          workflows: updatedWorkflows,
          workflowsByState: updatedByState,
        };
      });

      return true;
    } catch (err) {
      flowForgeState.update((s) => ({
        ...s,
        error: err instanceof Error ? err.message : 'Failed to resume workflow',
      }));
      return false;
    }
  },

  /**
   * Clear error
   */
  clearError(): void {
    flowForgeState.update((s) => ({ ...s, error: null }));
  },

  /**
   * Reset store
   */
  reset(): void {
    flowForgeState.set(initialState);
  },
};

export default flowForgeStore;