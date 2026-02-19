/**
 * Workflow Store
 *
 * Manages workflow state and history.
 */

import { writable, derived, get } from 'svelte/store';
import type { Readable, Writable } from 'svelte/store';

// Types
export type WorkflowStatus = 'pending' | 'running' | 'paused' | 'completed' | 'failed' | 'cancelled';
export type StepStatus = 'pending' | 'running' | 'completed' | 'failed' | 'skipped';

export interface WorkflowStep {
  step_id: string;
  name: string;
  description: string;
  agent_type: string;
  status: StepStatus;
  started_at?: string;
  completed_at?: string;
  input_data: Record<string, any>;
  output_data: Record<string, any>;
  error?: string;
  retries: number;
  duration_seconds?: number;
}

export interface Workflow {
  workflow_id: string;
  workflow_type: string;
  status: WorkflowStatus;
  current_step_index: number;
  progress_percent: number;
  steps: WorkflowStep[];
  input_data: Record<string, any>;
  intermediate_results: Record<string, any>;
  final_result: Record<string, any>;
  created_at: string;
  started_at?: string;
  completed_at?: string;
  duration_seconds?: number;
  error?: string;
}

export interface WorkflowSummary {
  workflow_id: string;
  workflow_type: string;
  status: WorkflowStatus;
  progress_percent: number;
  created_at: string;
  error?: string;
}

// State
interface WorkflowState {
  activeWorkflows: Workflow[];
  workflowHistory: WorkflowSummary[];
  selectedWorkflow: Workflow | null;
  loading: boolean;
  error: string | null;
}

const initialState: WorkflowState = {
  activeWorkflows: [],
  workflowHistory: [],
  selectedWorkflow: null,
  loading: false,
  error: null,
};

// Create stores
const workflowState = writable<WorkflowState>(initialState);

// Derived stores
export const activeWorkflows: Readable<Workflow[]> = derived(
  workflowState,
  ($state) => $state.activeWorkflows
);
export const workflowHistory: Readable<WorkflowSummary[]> = derived(
  workflowState,
  ($state) => $state.workflowHistory
);
export const selectedWorkflow: Readable<Workflow | null> = derived(
  workflowState,
  ($state) => $state.selectedWorkflow
);
export const hasActiveWorkflows: Readable<boolean> = derived(
  workflowState,
  ($state) => $state.activeWorkflows.length > 0
);
export const loading: Readable<boolean> = derived(workflowState, ($state) => $state.loading);
export const error: Readable<string | null> = derived(workflowState, ($state) => $state.error);

// Actions
export const workflowStore = {
  subscribe: workflowState.subscribe,

  /**
   * Start NPRD to EA workflow
   */
  async startNPRDToEA(nprdContent: string, metadata?: Record<string, any>): Promise<string | null> {
    workflowState.update((s) => ({ ...s, loading: true, error: null }));

    try {
      const response = await fetch('/api/workflows/nprd-to-ea', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          nprd_content: nprdContent,
          metadata,
        }),
      });

      if (!response.ok) throw new Error('Failed to start workflow');

      const data = await response.json();

      // Add to active workflows
      workflowState.update((s) => ({
        ...s,
        loading: false,
        activeWorkflows: [
          ...s.activeWorkflows,
          {
            workflow_id: data.workflow_id,
            workflow_type: 'nprd_to_ea',
            status: data.status,
            current_step_index: 0,
            progress_percent: 0,
            steps: [],
            input_data: { nprd_content: nprdContent },
            intermediate_results: {},
            final_result: {},
            created_at: new Date().toISOString(),
          } as Workflow,
        ],
      }));

      return data.workflow_id;
    } catch (err) {
      workflowState.update((s) => ({
        ...s,
        loading: false,
        error: err instanceof Error ? err.message : 'Failed to start workflow',
      }));
      return null;
    }
  },

  /**
   * Fetch workflow status
   */
  async fetchWorkflow(workflowId: string): Promise<Workflow | null> {
    try {
      const response = await fetch(`/api/workflows/${workflowId}`);
      if (!response.ok) throw new Error('Failed to fetch workflow');

      const workflow: Workflow = await response.json();

      // Update in active or history
      workflowState.update((s) => {
        const isActive = s.activeWorkflows.some((w) => w.workflow_id === workflowId);

        if (isActive) {
          // Update active workflow
          if (workflow.status === 'completed' || workflow.status === 'failed' || workflow.status === 'cancelled') {
            // Move to history
            return {
              ...s,
              activeWorkflows: s.activeWorkflows.filter((w) => w.workflow_id !== workflowId),
              workflowHistory: [
                {
                  workflow_id: workflow.workflow_id,
                  workflow_type: workflow.workflow_type,
                  status: workflow.status,
                  progress_percent: workflow.progress_percent,
                  created_at: workflow.created_at,
                  error: workflow.error,
                },
                ...s.workflowHistory,
              ],
              selectedWorkflow: s.selectedWorkflow?.workflow_id === workflowId ? workflow : s.selectedWorkflow,
            };
          } else {
            // Update in place
            return {
              ...s,
              activeWorkflows: s.activeWorkflows.map((w) =>
                w.workflow_id === workflowId ? workflow : w
              ),
              selectedWorkflow: s.selectedWorkflow?.workflow_id === workflowId ? workflow : s.selectedWorkflow,
            };
          }
        }

        return { ...s, selectedWorkflow: workflow };
      });

      return workflow;
    } catch (err) {
      console.error('Failed to fetch workflow:', err);
      return null;
    }
  },

  /**
   * Cancel workflow
   */
  async cancelWorkflow(workflowId: string): Promise<boolean> {
    try {
      const response = await fetch(`/api/workflows/${workflowId}/cancel`, {
        method: 'POST',
      });

      if (!response.ok) throw new Error('Failed to cancel workflow');

      // Update local state
      workflowState.update((s) => ({
        ...s,
        activeWorkflows: s.activeWorkflows.map((w) =>
          w.workflow_id === workflowId ? { ...w, status: 'cancelled' as WorkflowStatus } : w
        ),
      }));

      return true;
    } catch (err) {
      workflowState.update((s) => ({
        ...s,
        error: err instanceof Error ? err.message : 'Failed to cancel workflow',
      }));
      return false;
    }
  },

  /**
   * Pause workflow
   */
  async pauseWorkflow(workflowId: string): Promise<boolean> {
    try {
      const response = await fetch(`/api/workflows/${workflowId}/pause`, {
        method: 'POST',
      });

      if (!response.ok) throw new Error('Failed to pause workflow');

      workflowState.update((s) => ({
        ...s,
        activeWorkflows: s.activeWorkflows.map((w) =>
          w.workflow_id === workflowId ? { ...w, status: 'paused' as WorkflowStatus } : w
        ),
      }));

      return true;
    } catch (err) {
      return false;
    }
  },

  /**
   * Resume workflow
   */
  async resumeWorkflow(workflowId: string): Promise<boolean> {
    try {
      const response = await fetch(`/api/workflows/${workflowId}/resume`, {
        method: 'POST',
      });

      if (!response.ok) throw new Error('Failed to resume workflow');

      workflowState.update((s) => ({
        ...s,
        activeWorkflows: s.activeWorkflows.map((w) =>
          w.workflow_id === workflowId ? { ...w, status: 'running' as WorkflowStatus } : w
        ),
      }));

      return true;
    } catch (err) {
      return false;
    }
  },

  /**
   * Fetch all workflows
   */
  async fetchWorkflows(status?: WorkflowStatus): Promise<void> {
    workflowState.update((s) => ({ ...s, loading: true }));

    try {
      const params = new URLSearchParams();
      if (status) params.set('status', status);

      const response = await fetch(`/api/workflows?${params}`);
      if (!response.ok) throw new Error('Failed to fetch workflows');

      const data = await response.json();
      const workflows: Workflow[] = data.workflows || [];

      // Split workflows into active and history based on status
      const activeStatuses: WorkflowStatus[] = ['pending', 'running', 'paused'];
      const terminalStatuses: WorkflowStatus[] = ['completed', 'failed', 'cancelled'];

      const activeWorkflowsList = workflows.filter(
        (w) => activeStatuses.includes(w.status)
      );
      const historyList = workflows
        .filter((w) => terminalStatuses.includes(w.status))
        .map((w) => ({
          workflow_id: w.workflow_id,
          workflow_type: w.workflow_type,
          status: w.status,
          progress_percent: w.progress_percent,
          created_at: w.created_at,
          error: w.error,
        }));

      workflowState.update((s) => {
        // Update selectedWorkflow if it's in the active list and was refreshed
        let updatedSelected = s.selectedWorkflow;
        if (s.selectedWorkflow) {
          const refreshedActive = activeWorkflowsList.find(
            (w) => w.workflow_id === s.selectedWorkflow?.workflow_id
          );
          if (refreshedActive) {
            updatedSelected = refreshedActive;
          }
        }

        return {
          ...s,
          loading: false,
          activeWorkflows: activeWorkflowsList,
          workflowHistory: historyList,
          selectedWorkflow: updatedSelected,
        };
      });
    } catch (err) {
      workflowState.update((s) => ({
        ...s,
        loading: false,
        error: err instanceof Error ? err.message : 'Failed to fetch workflows',
      }));
    }
  },

  /**
   * Select workflow
   */
  selectWorkflow(workflow: Workflow | null): void {
    workflowState.update((s) => ({ ...s, selectedWorkflow: workflow }));
  },

  /**
   * Clear error
   */
  clearError(): void {
    workflowState.update((s) => ({ ...s, error: null }));
  },

  /**
   * Reset store
   */
  reset(): void {
    workflowState.set(initialState);
  },
};

export default workflowStore;
