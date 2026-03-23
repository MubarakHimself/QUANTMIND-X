<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { fade, slide, fly } from 'svelte/transition';

  // Types
  interface WorkflowStep {
    stage: string;
    status: 'pending' | 'running' | 'completed' | 'failed';
    started_at: string | null;
    completed_at: string | null;
    error: string | null;
    output: Record<string, any> | null;
    retry_count: number;
  }

  interface Workflow {
    workflow_id: string;
    status: 'pending' | 'running' | 'paused' | 'completed' | 'failed' | 'cancelled';
    created_at: string;
    updated_at: string;
    steps: Record<string, WorkflowStep>;
    input_file: string | null;
    output_files: Record<string, string>;
    metadata: Record<string, any>;
    error: string | null;
  }

  // State
  let workflows: Workflow[] = $state([]);
  let selectedWorkflow: Workflow | null = $state(null);
  let loading = $state(true);
  let error: string | null = $state(null);
  let autoRefresh = $state(true);
  let refreshInterval: number | null = null;

  // Stage display info
  const stageInfo: Record<string, { label: string; icon: string; description: string }> = {
    video_ingest_processing: { label: 'Video Ingest Processing', icon: '📹', description: 'Processing video content' },
    analyst: { label: 'Analyst', icon: '🔍', description: 'Analyzing strategy requirements' },
    trd_generation: { label: 'TRD Generation', icon: '📝', description: 'Creating Trading Requirements Document' },
    quantcode: { label: 'QuantCode', icon: '💻', description: 'Generating MQL5 code' },
    compilation: { label: 'Compilation', icon: '⚙️', description: 'Compiling Expert Advisor' },
    backtest: { label: 'Backtest', icon: '📈', description: 'Running strategy backtest' },
    validation: { label: 'Validation', icon: '✅', description: 'Validating results' }
  };

  // Status colors
  const statusColors: Record<string, string> = {
    pending: 'text-gray-400',
    running: 'text-blue-400',
    completed: 'text-green-400',
    failed: 'text-red-400',
    paused: 'text-yellow-400',
    cancelled: 'text-gray-500'
  };

  // Fetch workflows
  async function fetchWorkflows() {
    try {
      const response = await fetch('/api/workflows');
      if (!response.ok) throw new Error('Failed to fetch workflows');
      const data = await response.json();
      workflows = data.workflows || [];
      
      // Update selected workflow if it exists
      const currentWorkflow = selectedWorkflow;
      if (currentWorkflow) {
        const updated = workflows.find(w => w.workflow_id === currentWorkflow.workflow_id);
        if (updated) {
          selectedWorkflow = updated;
        }
      }
      
      error = null;
    } catch (e) {
      error = e instanceof Error ? e.message : 'Failed to fetch workflows';
    } finally {
      loading = false;
    }
  }

  // Select workflow
  function selectWorkflow(workflow: Workflow) {
    selectedWorkflow = workflow;
  }

  // Get progress percentage
  function getProgress(workflow: Workflow): number {
    if (!workflow.steps) return 0;
    const steps = Object.values(workflow.steps);
    const completed = steps.filter(s => s.status === 'completed').length;
    return Math.round((completed / steps.length) * 100);
  }

  // Get current stage
  function getCurrentStage(workflow: Workflow): string {
    if (!workflow.steps) return 'Unknown';
    for (const [name, step] of Object.entries(workflow.steps)) {
      if (step.status === 'running') {
        return stageInfo[name]?.label || name;
      }
    }
    return 'Completed';
  }

  // Format time
  function formatTime(isoString: string | null): string {
    if (!isoString) return '-';
    const date = new Date(isoString);
    return date.toLocaleTimeString();
  }

  // Format duration
  function formatDuration(start: string | null, end: string | null): string {
    if (!start) return '-';
    const startDate = new Date(start);
    const endDate = end ? new Date(end) : new Date();
    const diff = Math.floor((endDate.getTime() - startDate.getTime()) / 1000);
    
    if (diff < 60) return `${diff}s`;
    if (diff < 3600) return `${Math.floor(diff / 60)}m ${diff % 60}s`;
    return `${Math.floor(diff / 3600)}h ${Math.floor((diff % 3600) / 60)}m`;
  }

  // Cancel workflow
  async function cancelWorkflow(workflowId: string) {
    if (!confirm('Are you sure you want to cancel this workflow?')) return;
    
    try {
      const response = await fetch(`/api/workflows/${workflowId}/cancel`, {
        method: 'POST'
      });
      if (!response.ok) throw new Error('Failed to cancel workflow');
      await fetchWorkflows();
    } catch (e) {
      error = e instanceof Error ? e.message : 'Failed to cancel workflow';
    }
  }

  // Retry workflow
  async function retryWorkflow(workflowId: string) {
    try {
      const response = await fetch(`/api/workflows/${workflowId}/retry`, {
        method: 'POST'
      });
      if (!response.ok) throw new Error('Failed to retry workflow');
      await fetchWorkflows();
    } catch (e) {
      error = e instanceof Error ? e.message : 'Failed to retry workflow';
    }
  }

  // Lifecycle
  onMount(() => {
    fetchWorkflows();
    
    if (autoRefresh) {
      refreshInterval = window.setInterval(fetchWorkflows, 5000);
    }
  });

  onDestroy(() => {
    if (refreshInterval) {
      clearInterval(refreshInterval);
    }
  });
</script>

<div class="workflow-status-panel">
  <!-- Header -->
  <div class="panel-header">
    <h3>Workflow Status</h3>
    <div class="header-actions">
      <button class="refresh-btn" onclick={fetchWorkflows} title="Refresh">
        🔄
      </button>
      <label class="auto-refresh-toggle">
        <input type="checkbox" bind:checked={autoRefresh} />
        <span>Auto-refresh</span>
      </label>
    </div>
  </div>

  <!-- Error Display -->
  {#if error}
    <div class="error-banner" in:fly={{ y: -20 }}>
      <span>⚠️</span>
      <span>{error}</span>
      <button onclick={() => error = null}>×</button>
    </div>
  {/if}

  <!-- Loading State -->
  {#if loading && workflows.length === 0}
    <div class="loading-state" in:fade>
      <div class="spinner"></div>
      <p>Loading workflows...</p>
    </div>
  {:else}
    <div class="content-area">
      <!-- Workflow List -->
      <div class="workflow-list">
        <h4>Active Workflows ({workflows.length})</h4>
        
        {#if workflows.length === 0}
          <div class="empty-state">
            <span class="empty-icon">📋</span>
            <p>No active workflows</p>
            <p class="hint">Upload an Video Ingest video to start a new workflow</p>
          </div>
        {:else}
          <div class="workflow-cards">
            {#each workflows as workflow (workflow.workflow_id)}
              <div 
                class="workflow-card"
                class:selected={selectedWorkflow?.workflow_id === workflow.workflow_id}
                onclick={() => selectWorkflow(workflow)}
                in:fly={{ y: 20 }}
              >
                <div class="card-header">
                  <span class="workflow-id">{workflow.workflow_id.slice(0, 16)}...</span>
                  <span class="status-badge {workflow.status}">{workflow.status}</span>
                </div>
                
                <div class="progress-section">
                  <div class="progress-bar">
                    <div 
                      class="progress-fill"
                      style="width: {getProgress(workflow)}%"
                    ></div>
                  </div>
                  <span class="progress-text">{getProgress(workflow)}%</span>
                </div>
                
                <div class="card-footer">
                  <span class="current-stage">{getCurrentStage(workflow)}</span>
                  <span class="duration">{formatDuration(workflow.created_at, workflow.updated_at)}</span>
                </div>
              </div>
            {/each}
          </div>
        {/if}
      </div>

      <!-- Workflow Details -->
      {#if selectedWorkflow}
        <div class="workflow-details" in:slide>
          <div class="details-header">
            <h4>Workflow Details</h4>
            <div class="details-actions">
              {#if selectedWorkflow.status === 'running'}
                <button
                  class="cancel-btn"
                  onclick={() => selectedWorkflow && cancelWorkflow(selectedWorkflow.workflow_id)}
                >
                  Cancel
                </button>
              {/if}
              {#if selectedWorkflow.status === 'failed'}
                <button
                  class="retry-btn"
                  onclick={() => selectedWorkflow && retryWorkflow(selectedWorkflow.workflow_id)}
                >
                  Retry
                </button>
              {/if}
              <button class="close-btn" onclick={() => selectedWorkflow = null}>×</button>
            </div>
          </div>

          <div class="details-content">
            <!-- Pipeline Visualization -->
            <div class="pipeline">
              {#each Object.entries(selectedWorkflow.steps || {}) as [name, step]}
                <div class="pipeline-stage {step.status}">
                  <div class="stage-icon">
                    {stageInfo[name]?.icon || '📌'}
                  </div>
                  <div class="stage-info">
                    <span class="stage-name">{stageInfo[name]?.label || name}</span>
                    <span class="stage-status {step.status}">{step.status}</span>
                  </div>
                  {#if step.status === 'running'}
                    <div class="stage-spinner"></div>
                  {:else if step.status === 'completed'}
                    <span class="stage-check">✓</span>
                  {:else if step.status === 'failed'}
                    <span class="stage-error">✗</span>
                  {/if}
                </div>
              {/each}
            </div>

            <!-- Error Display -->
            {#if selectedWorkflow.error}
              <div class="workflow-error">
                <h5>Error</h5>
                <p>{selectedWorkflow.error}</p>
              </div>
            {/if}

            <!-- Output Files -->
            {#if Object.keys(selectedWorkflow.output_files || {}).length > 0}
              <div class="output-files">
                <h5>Output Files</h5>
                <ul>
                  {#each Object.entries(selectedWorkflow.output_files) as [name, path]}
                    <li>
                      <span class="file-name">{name}</span>
                      <span class="file-path">{path}</span>
                    </li>
                  {/each}
                </ul>
              </div>
            {/if}

            <!-- Metadata -->
            <div class="metadata">
              <h5>Metadata</h5>
              <div class="meta-grid">
                <div class="meta-item">
                  <span class="meta-label">Created</span>
                  <span class="meta-value">{new Date(selectedWorkflow.created_at).toLocaleString()}</span>
                </div>
                <div class="meta-item">
                  <span class="meta-label">Updated</span>
                  <span class="meta-value">{new Date(selectedWorkflow.updated_at).toLocaleString()}</span>
                </div>
                <div class="meta-item">
                  <span class="meta-label">Input</span>
                  <span class="meta-value">{selectedWorkflow.input_file || '-'}</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      {/if}
    </div>
  {/if}
</div>

<style>
  .workflow-status-panel {
    background: var(--bg-secondary, #1e1e2e);
    border-radius: 8px;
    padding: 16px;
    color: var(--text-primary, #cdd6f4);
    height: 100%;
    display: flex;
    flex-direction: column;
  }

  .panel-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 16px;
    padding-bottom: 12px;
    border-bottom: 1px solid var(--border-color, #313244);
  }

  .panel-header h3 {
    margin: 0;
    font-size: 1.125rem;
  }

  .header-actions {
    display: flex;
    align-items: center;
    gap: 12px;
  }

  .refresh-btn {
    background: transparent;
    border: none;
    cursor: pointer;
    padding: 4px;
    font-size: 1rem;
  }

  .auto-refresh-toggle {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 0.75rem;
    color: var(--text-secondary, #a6adc8);
  }

  .error-banner {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 12px;
    background: rgba(243, 139, 168, 0.1);
    border-radius: 6px;
    color: #f38ba8;
    font-size: 0.875rem;
    margin-bottom: 12px;
  }

  .error-banner button {
    margin-left: auto;
    background: transparent;
    border: none;
    color: inherit;
    cursor: pointer;
  }

  .loading-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 48px;
    color: var(--text-secondary, #a6adc8);
  }

  .spinner {
    width: 32px;
    height: 32px;
    border: 3px solid var(--border-color, #45475a);
    border-top-color: var(--accent, #89b4fa);
    border-radius: 50%;
    animation: spin 1s linear infinite;
    margin-bottom: 12px;
  }

  @keyframes spin {
    to { transform: rotate(360deg); }
  }

  .content-area {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
    flex: 1;
    overflow: hidden;
  }

  .workflow-list {
    overflow-y: auto;
  }

  .workflow-list h4 {
    margin: 0 0 12px;
    font-size: 0.875rem;
    color: var(--text-secondary, #a6adc8);
  }

  .empty-state {
    text-align: center;
    padding: 32px;
    color: var(--text-secondary, #a6adc8);
  }

  .empty-icon {
    font-size: 2rem;
    display: block;
    margin-bottom: 8px;
  }

  .hint {
    font-size: 0.75rem;
    margin-top: 8px;
  }

  .workflow-cards {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .workflow-card {
    background: var(--bg-tertiary, #313244);
    border: 1px solid var(--border-color, #45475a);
    border-radius: 8px;
    padding: 12px;
    cursor: pointer;
    transition: all 0.2s ease;
  }

  .workflow-card:hover {
    border-color: var(--accent, #89b4fa);
  }

  .workflow-card.selected {
    border-color: var(--accent, #89b4fa);
    background: rgba(137, 180, 250, 0.1);
  }

  .card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
  }

  .workflow-id {
    font-size: 0.75rem;
    color: var(--text-secondary, #a6adc8);
    font-family: monospace;
  }

  .status-badge {
    font-size: 0.625rem;
    padding: 2px 6px;
    border-radius: 4px;
    text-transform: uppercase;
    font-weight: 600;
  }

  .status-badge.running {
    background: rgba(137, 180, 250, 0.2);
    color: #89b4fa;
  }

  .status-badge.completed {
    background: rgba(166, 227, 161, 0.2);
    color: #a6e3a1;
  }

  .status-badge.failed {
    background: rgba(243, 139, 168, 0.2);
    color: #f38ba8;
  }

  .status-badge.pending {
    background: rgba(166, 173, 200, 0.2);
    color: #a6adc8;
  }

  .progress-section {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 8px;
  }

  .progress-bar {
    flex: 1;
    height: 4px;
    background: var(--bg-secondary, #1e1e2e);
    border-radius: 2px;
    overflow: hidden;
  }

  .progress-fill {
    height: 100%;
    background: var(--accent, #89b4fa);
    transition: width 0.3s ease;
  }

  .progress-text {
    font-size: 0.75rem;
    color: var(--text-secondary, #a6adc8);
    min-width: 36px;
    text-align: right;
  }

  .card-footer {
    display: flex;
    justify-content: space-between;
    font-size: 0.75rem;
    color: var(--text-secondary, #a6adc8);
  }

  .workflow-details {
    background: var(--bg-tertiary, #313244);
    border-radius: 8px;
    overflow: hidden;
    display: flex;
    flex-direction: column;
  }

  .details-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 16px;
    border-bottom: 1px solid var(--border-color, #45475a);
  }

  .details-header h4 {
    margin: 0;
    font-size: 0.875rem;
  }

  .details-actions {
    display: flex;
    gap: 8px;
  }

  .cancel-btn, .retry-btn {
    padding: 4px 12px;
    border-radius: 4px;
    font-size: 0.75rem;
    cursor: pointer;
  }

  .cancel-btn {
    background: rgba(243, 139, 168, 0.2);
    border: 1px solid #f38ba8;
    color: #f38ba8;
  }

  .retry-btn {
    background: rgba(166, 227, 161, 0.2);
    border: 1px solid #a6e3a1;
    color: #a6e3a1;
  }

  .close-btn {
    background: transparent;
    border: none;
    color: var(--text-secondary, #a6adc8);
    font-size: 1.25rem;
    cursor: pointer;
  }

  .details-content {
    padding: 16px;
    overflow-y: auto;
    flex: 1;
  }

  .pipeline {
    display: flex;
    flex-direction: column;
    gap: 8px;
    margin-bottom: 16px;
  }

  .pipeline-stage {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 8px 12px;
    background: var(--bg-secondary, #1e1e2e);
    border-radius: 6px;
    border-left: 3px solid transparent;
  }

  .pipeline-stage.running {
    border-left-color: #89b4fa;
  }

  .pipeline-stage.completed {
    border-left-color: #a6e3a1;
  }

  .pipeline-stage.failed {
    border-left-color: #f38ba8;
  }

  .stage-icon {
    font-size: 1.25rem;
  }

  .stage-info {
    flex: 1;
  }

  .stage-name {
    display: block;
    font-size: 0.875rem;
  }

  .stage-status {
    font-size: 0.75rem;
    color: var(--text-secondary, #a6adc8);
  }

  .stage-status.running {
    color: #89b4fa;
  }

  .stage-status.completed {
    color: #a6e3a1;
  }

  .stage-status.failed {
    color: #f38ba8;
  }

  .stage-spinner {
    width: 16px;
    height: 16px;
    border: 2px solid var(--border-color, #45475a);
    border-top-color: var(--accent, #89b4fa);
    border-radius: 50%;
    animation: spin 1s linear infinite;
  }

  .stage-check {
    color: #a6e3a1;
    font-weight: bold;
  }

  .stage-error {
    color: #f38ba8;
    font-weight: bold;
  }

  .workflow-error {
    padding: 12px;
    background: rgba(243, 139, 168, 0.1);
    border-radius: 6px;
    margin-bottom: 16px;
  }

  .workflow-error h5 {
    margin: 0 0 8px;
    font-size: 0.875rem;
    color: #f38ba8;
  }

  .workflow-error p {
    margin: 0;
    font-size: 0.75rem;
    color: var(--text-secondary, #a6adc8);
  }

  .output-files, .metadata {
    margin-bottom: 16px;
  }

  .output-files h5, .metadata h5 {
    margin: 0 0 8px;
    font-size: 0.875rem;
    color: var(--text-secondary, #a6adc8);
  }

  .output-files ul {
    list-style: none;
    padding: 0;
    margin: 0;
  }

  .output-files li {
    display: flex;
    justify-content: space-between;
    padding: 6px 0;
    font-size: 0.75rem;
    border-bottom: 1px solid var(--border-color, #313244);
  }

  .file-name {
    color: var(--text-primary, #cdd6f4);
  }

  .file-path {
    color: var(--text-secondary, #a6adc8);
    font-family: monospace;
  }

  .meta-grid {
    display: grid;
    gap: 8px;
  }

  .meta-item {
    display: flex;
    justify-content: space-between;
    font-size: 0.75rem;
  }

  .meta-label {
    color: var(--text-secondary, #a6adc8);
  }

  .meta-value {
    color: var(--text-primary, #cdd6f4);
  }
</style>
