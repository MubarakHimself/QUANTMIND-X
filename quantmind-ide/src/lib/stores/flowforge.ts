/**
 * FlowForge Store
 *
 * Manages Prefect workflow state for FlowForge canvas.
 * Story 11.5: FlowForge Canvas — Prefect Kanban & Node Graph
 * Story 11.8: FlowForge ↔ Prefect API Contract
 */

import { writable, derived, get } from 'svelte/store';
import type { Readable } from 'svelte/store';
import { addNotification } from './notifications';
import { API_CONFIG } from '$lib/config/api';

// ============================================================================
// Types for FlowForge ↔ Prefect API Contract (Story 11.8)
// ============================================================================

/**
 * A node in the FlowForge canvas workflow graph.
 * Used when deploying workflows to Prefect.
 */
export interface FlowForgeNode {
  id: string;
  type: string;
  config: Record<string, unknown>;
  depends_on?: string[];
}

/**
 * Request to deploy a workflow from FlowForge canvas to Prefect.
 */
export interface DeployWorkflowRequest {
  canvas_workflow_uuid: string;
  name: string;
  nodes: FlowForgeNode[];
  department?: string;
}

/**
 * Response from deploying a workflow.
 */
export interface DeployWorkflowResponse {
  workflow_id: string;
  deployment_id: string;
  deployment_name?: string;
}

/**
 * Request to trigger a workflow run.
 */
export interface RunWorkflowRequest {
  canvas_id?: string;
  operator_id?: string;
  run_reason?: string;
}

/**
 * Response from triggering a workflow run.
 */
export interface RunWorkflowResponse {
  run_id: string;
  deployment_id: string;
  state: string;
  created_at?: string;
}

/**
 * SSE event types for workflow stage transitions.
 */
export interface StageEvent {
  type: 'stage';
  run_id: string;
  stage: string;
  status: 'running' | 'completed' | 'failed' | 'cancelled';
  elapsed_s: number;
}

export interface ErrorEvent {
  type: 'error';
  run_id: string;
  task_name: string;
  error_type: string;
  error_message: string;
  retry_count: number;
}

export interface CompletionEvent {
  type: 'completion';
  run_id: string;
  status: 'completed' | 'failed' | 'cancelled';
  final_state: string;
  elapsed_s: number;
}

export type WorkflowEvent = StageEvent | ErrorEvent | CompletionEvent;

/**
 * Error detail for failed tasks (displayed in error panel).
 */
export interface TaskErrorDetail {
  task_name: string;
  error_type: string;
  error_message: string;
  retry_count: number;
}

/**
 * A deployed FlowForge workflow (from the proxy API).
 */
export interface DeployedWorkflow {
  id: string;
  name: string;
  flow_name?: string;
  deployment_id: string;
  canvas_workflow_uuid: string;
  state?: WorkflowState;
  department?: string;
}

/**
 * Error info for display in node graph.
 */
export interface NodeErrorInfo {
  taskName: string;
  errorType: string;
  errorMessage: string;
  retryCount: number;
}

// ============================================================================
// Original Prefect Types (Story 11.5)
// ============================================================================

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
  // Story 11.8: API Contract state
  deployedWorkflows: DeployedWorkflow[];
  deploymentMapping: Record<string, string>; // canvas_uuid -> deployment_id
  activeEventSource: EventSource | null;
  currentRunId: string | null;
  currentNodeErrors: NodeErrorInfo[];
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
  // Story 11.8
  deployedWorkflows: [],
  deploymentMapping: {},
  activeEventSource: null,
  currentRunId: null,
  currentNodeErrors: [],
};

// Create store
const flowForgeState = writable<FlowForgeState>(initialState);

function getFlowForgeApiUrl(path: string): string {
  return `${API_CONFIG.API_URL}${path}`;
}

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
      const response = await fetch(getFlowForgeApiUrl('/api/prefect/workflows'));
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
      const response = await fetch(getFlowForgeApiUrl(`/api/prefect/workflows/${workflowId}`));
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
      const response = await fetch(getFlowForgeApiUrl(`/api/prefect/workflows/${workflowId}/cancel`), {
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
      const response = await fetch(getFlowForgeApiUrl(`/api/prefect/workflows/${workflowId}/resume`), {
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

  // =========================================================================
  // Story 11.8: FlowForge ↔ Prefect API Contract
  // =========================================================================

  /**
   * Deploy a workflow from FlowForge canvas to Prefect.
   * POST /api/workflows
   *
   * AC1: Workflow Deployment
   */
  async deployWorkflow(request: DeployWorkflowRequest): Promise<DeployWorkflowResponse | null> {
    flowForgeState.update((s) => ({ ...s, loading: true, error: null }));

    try {
      const response = await fetch(getFlowForgeApiUrl('/api/workflows'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(request),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Failed to deploy workflow');
      }

      const result: DeployWorkflowResponse = await response.json();

      // Store the mapping
      flowForgeState.update((s) => ({
        ...s,
        loading: false,
        deploymentMapping: {
          ...s.deploymentMapping,
          [request.canvas_workflow_uuid]: result.deployment_id,
        },
      }));

      return result;
    } catch (err) {
      flowForgeState.update((s) => ({
        ...s,
        loading: false,
        error: err instanceof Error ? err.message : 'Failed to deploy workflow',
      }));
      return null;
    }
  },

  /**
   * List all deployed workflows from the proxy API.
   * GET /api/workflows
   *
   * AC6: Workflow List with Status Badges
   */
  async listDeployedWorkflows(): Promise<void> {
    flowForgeState.update((s) => ({ ...s, loading: true, error: null }));

    try {
      const response = await fetch(getFlowForgeApiUrl('/api/workflows'));
      if (!response.ok) throw new Error('Failed to fetch workflows');

      const data = await response.json();
      const workflows: DeployedWorkflow[] = data.workflows || [];

      flowForgeState.update((s) => ({
        ...s,
        loading: false,
        deployedWorkflows: workflows,
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
   * Trigger a workflow run and subscribe to SSE events.
   * POST /api/workflows/{workflow_id}/run
   * GET /api/workflows/{workflow_id}/events?run_id={run_id}
   *
   * AC2: Workflow Run Trigger
   * AC3: Real-time SSE Stage Updates
   * AC4: Workflow Completion/Failure Notification
   */
  async triggerWorkflowRun(
    workflowId: string,
    request: RunWorkflowRequest,
    onStageEvent?: (event: StageEvent) => void,
    onErrorEvent?: (event: ErrorEvent) => void,
    onCompletionEvent?: (event: CompletionEvent) => void
  ): Promise<RunWorkflowResponse | null> {
    flowForgeState.update((s) => ({ ...s, loading: true, error: null }));

    try {
      // Trigger the run
      const runResponse = await fetch(getFlowForgeApiUrl(`/api/workflows/${workflowId}/run`), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(request),
      });

      if (!runResponse.ok) {
        const error = await runResponse.json();
        throw new Error(error.detail || 'Failed to trigger workflow run');
      }

      const runResult: RunWorkflowResponse = await runResponse.json();

      flowForgeState.update((s) => ({
        ...s,
        loading: false,
        currentRunId: runResult.run_id,
        currentNodeErrors: [],
      }));

      // Subscribe to SSE events
      const eventSource = new EventSource(
        getFlowForgeApiUrl(`/api/workflows/${workflowId}/events?run_id=${runResult.run_id}`)
      );

      eventSource.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data) as WorkflowEvent;

          switch (data.type) {
            case 'stage':
              onStageEvent?.(data as StageEvent);
              break;
            case 'error':
              // Add to node errors for display
              flowForgeState.update((s) => ({
                ...s,
                currentNodeErrors: [
                  ...s.currentNodeErrors,
                  {
                    taskName: (data as ErrorEvent).task_name,
                    errorType: (data as ErrorEvent).error_type,
                    errorMessage: (data as ErrorEvent).error_message,
                    retryCount: (data as ErrorEvent).retry_count,
                  },
                ],
              }));
              onErrorEvent?.(data as ErrorEvent);
              break;
            case 'completion':
              // Close the event source on completion
              eventSource.close();
              flowForgeState.update((s) => ({
                ...s,
                activeEventSource: null,
                currentRunId: null,
              }));

              // AC4: Copilot notification for department canvas workflows
              if (request.canvas_id) {
                const completionData = data as CompletionEvent;
                const statusIcon = completionData.status === 'completed' ? 'success' : 'error';
                const statusText = completionData.status === 'completed' ? 'completed successfully' : `failed (${completionData.final_state})`;
                addNotification({
                  type: 'agent',
                  title: 'FlowForge Workflow',
                  body: `Workflow run ${statusText} in ${completionData.elapsed_s}s`,
                  canvasLink: `flowforge`,
                });
              }

              onCompletionEvent?.(data as CompletionEvent);
              break;
          }
        } catch (err) {
          console.error('Failed to parse SSE event:', err);
        }
      };

      eventSource.onerror = () => {
        eventSource.close();
        flowForgeState.update((s) => ({
          ...s,
          activeEventSource: null,
          currentRunId: null,
        }));
      };

      // Store the event source
      flowForgeState.update((s) => ({
        ...s,
        activeEventSource: eventSource,
      }));

      return runResult;
    } catch (err) {
      flowForgeState.update((s) => ({
        ...s,
        loading: false,
        error: err instanceof Error ? err.message : 'Failed to trigger workflow run',
      }));
      return null;
    }
  },

  /**
   * Cancel a running workflow.
   * DELETE /api/workflows/{workflow_id}/run/{run_id}
   *
   * AC5: Workflow Cancellation (Kill)
   */
  async cancelWorkflowRun(workflowId: string, runId: string): Promise<boolean> {
    flowForgeState.update((s) => ({ ...s, loading: true, error: null }));

    try {
      // Close any active SSE connection
      const state = get(flowForgeState);
      if (state.activeEventSource) {
        state.activeEventSource.close();
      }

      const response = await fetch(getFlowForgeApiUrl(`/api/workflows/${workflowId}/run/${runId}`), {
        method: 'DELETE',
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Failed to cancel workflow');
      }

      flowForgeState.update((s) => ({
        ...s,
        loading: false,
        activeEventSource: null,
        currentRunId: null,
      }));

      return true;
    } catch (err) {
      flowForgeState.update((s) => ({
        ...s,
        loading: false,
        error: err instanceof Error ? err.message : 'Failed to cancel workflow',
      }));
      return false;
    }
  },

  /**
   * Delete a deployed workflow.
   * DELETE /api/workflows/{workflow_id}
   *
   * AC9: Workflow Deletion
   */
  async deleteWorkflow(workflowId: string): Promise<boolean> {
    flowForgeState.update((s) => ({ ...s, loading: true, error: null }));

    try {
      const response = await fetch(getFlowForgeApiUrl(`/api/workflows/${workflowId}`), {
        method: 'DELETE',
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Failed to delete workflow');
      }

      // Remove from deployed workflows
      flowForgeState.update((s) => {
        const updated = { ...s.deploymentMapping };
        delete updated[workflowId];
        return {
          ...s,
          loading: false,
          deployedWorkflows: s.deployedWorkflows.filter((w) => w.id !== workflowId),
          deploymentMapping: updated,
        };
      });

      return true;
    } catch (err) {
      flowForgeState.update((s) => ({
        ...s,
        loading: false,
        error: err instanceof Error ? err.message : 'Failed to delete workflow',
      }));
      return false;
    }
  },

  /**
   * Clear node errors (after they've been displayed).
   */
  clearNodeErrors(): void {
    flowForgeState.update((s) => ({ ...s, currentNodeErrors: [] }));
  },

  /**
   * Get deployment ID for a canvas workflow UUID.
   */
  getDeploymentId(canvasUuid: string): string | undefined {
    const state = get(flowForgeState);
    return state.deploymentMapping[canvasUuid];
  },

  // =========================================================================
  // Story 11.8: Draft localStorage (AC8)
  // =========================================================================

  /**
   * Save workflow draft to localStorage.
   * Key: flowforge_draft_{workflow_uuid}
   * NO API call is made for draft saves.
   *
   * AC8: Draft Save (No API Call)
   */
  saveDraft(workflowUuid: string, nodes: FlowForgeNode[]): void {
    try {
      const key = `flowforge_draft_${workflowUuid}`;
      localStorage.setItem(key, JSON.stringify({ nodes, savedAt: new Date().toISOString() }));
    } catch (err) {
      console.error('Failed to save draft:', err);
    }
  },

  /**
   * Load workflow draft from localStorage.
   */
  loadDraft(workflowUuid: string): FlowForgeNode[] | null {
    try {
      const key = `flowforge_draft_${workflowUuid}`;
      const data = localStorage.getItem(key);
      if (data) {
        const parsed = JSON.parse(data);
        return parsed.nodes || null;
      }
    } catch (err) {
      console.error('Failed to load draft:', err);
    }
    return null;
  },

  /**
   * Delete workflow draft from localStorage.
   */
  deleteDraft(workflowUuid: string): void {
    try {
      const key = `flowforge_draft_${workflowUuid}`;
      localStorage.removeItem(key);
    } catch (err) {
      console.error('Failed to delete draft:', err);
    }
  },

  /**
   * Check if a draft exists for a workflow.
   */
  hasDraft(workflowUuid: string): boolean {
    const key = `flowforge_draft_${workflowUuid}`;
    return localStorage.getItem(key) !== null;
  },
};

export default flowForgeStore;
