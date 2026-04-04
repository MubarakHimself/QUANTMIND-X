<script lang="ts">
  import { onMount } from 'svelte';
  import {
    Clock,
    CheckCircle,
    AlertCircle,
    RefreshCw,
    Play,
    Pause,
    FileText,
    Bot,
    TrendingUp,
    ShieldAlert,
    Award,
    BarChart3,
    X
  } from 'lucide-svelte';
  import { API_BASE } from '$lib/constants';
  import EodReportViewer from './EodReportViewer.svelte';

  const apiBase = API_BASE || '';

  // Modal state
  let showEodReportModal = $state(false);

  // Workflow state
  let workflowStatus = $state<{
    run_id: string | null;
    status: string;
    started_at: string | null;
    completed_at: string | null;
    steps: Record<string, {
      step_name: string;
      scheduled_time: string;
      deadline_offset_minutes: number;
      status: string;
      started_at: string | null;
      completed_at: string | null;
      error: string | null;
    }>;
    error: string | null;
  }>({
    run_id: null,
    status: 'not_started',
    started_at: null,
    completed_at: null,
    steps: {},
    error: null,
  });

  let loading = $state(true);
  let error = $state<string | null>(null);
  let lastUpdate = $state<string | null>(null);
  let schedulerStatus = $state({ running: false, workflow_3_start: '16:15' });

  // Step definitions
  const steps = [
    { key: 'eod_report', label: 'EOD Report', time: '16:15 GMT', icon: FileText },
    { key: 'session_performer_id', label: 'Session Performer', time: '16:45 GMT', icon: Bot },
    { key: 'dpr_update', label: 'DPR Update', time: '17:00 GMT', icon: TrendingUp },
    { key: 'queue_rerank', label: 'Queue Re-rank', time: '17:30 GMT', icon: ShieldAlert },
    { key: 'fortnight_accumulation', label: 'Fortnight Accum', time: '18:00 GMT', icon: Award },
  ];

  async function fetchWorkflowStatus() {
    try {
      const baseUrl = apiBase || window.location.origin;
      const response = await fetch(`${baseUrl}/api/dead-zone/workflow/status`);

      if (!response.ok) {
        throw new Error(`Failed to fetch: ${response.status}`);
      }

      const data = await response.json();
      workflowStatus = data;
      lastUpdate = new Date().toLocaleTimeString();
      error = null;
    } catch (e) {
      console.error('Failed to fetch workflow status:', e);
      error = e instanceof Error ? e.message : 'Unknown error';
    } finally {
      loading = false;
    }
  }

  async function fetchSchedulerStatus() {
    try {
      const baseUrl = apiBase || window.location.origin;
      const response = await fetch(`${baseUrl}/api/dead-zone/scheduler/status`);

      if (!response.ok) {
        throw new Error(`Failed to fetch: ${response.status}`);
      }

      schedulerStatus = await response.json();
    } catch (e) {
      console.error('Failed to fetch scheduler status:', e);
    }
  }

  async function triggerWorkflow() {
    try {
      const baseUrl = apiBase || window.location.origin;
      const response = await fetch(`${baseUrl}/api/dead-zone/workflow/trigger`, {
        method: 'POST',
      });

      if (!response.ok) {
        throw new Error(`Failed to trigger: ${response.status}`);
      }

      const result = await response.json();
      console.log('Workflow triggered:', result);
      await fetchWorkflowStatus();
    } catch (e) {
      console.error('Failed to trigger workflow:', e);
      error = e instanceof Error ? e.message : 'Unknown error';
    }
  }

  function getStepStatus(key: string): string {
    return workflowStatus.steps[key]?.status || 'pending';
  }

  function getStepIcon(key: string) {
    const step = steps.find(s => s.key === key);
    return step?.icon || Clock;
  }

  function getStatusColor(status: string): string {
    switch (status) {
      case 'completed': return '#10b981';
      case 'running': return '#f59e0b';
      case 'failed': return '#ef4444';
      default: return '#6b7280';
    }
  }

  function getStatusBgColor(status: string): string {
    switch (status) {
      case 'completed': return 'rgba(16, 185, 129, 0.15)';
      case 'running': return 'rgba(245, 158, 11, 0.15)';
      case 'failed': return 'rgba(239, 68, 68, 0.15)';
      default: return 'rgba(107, 114, 128, 0.1)';
    }
  }

  onMount(() => {
    fetchWorkflowStatus();
    fetchSchedulerStatus();
    const interval = setInterval(fetchWorkflowStatus, 30000);
    return () => clearInterval(interval);
  });
</script>

<div class="dead-zone-panel">
  <div class="panel-header">
    <Clock size={18} />
    <h3>Dead Zone Workflow 3</h3>
    <span class="workflow-status" class:running={workflowStatus.status === 'running'}>
      {workflowStatus.status.toUpperCase()}
    </span>
    {#if lastUpdate}
      <span class="last-update">Updated: {lastUpdate}</span>
    {/if}
    <div class="header-actions">
      <button
        class="eod-report-btn"
        onclick={() => showEodReportModal = true}
        title="View EOD Report"
      >
        <BarChart3 size={14} />
        <span>EOD Report</span>
      </button>
      <button class="refresh-btn" onclick={fetchWorkflowStatus} title="Refresh">
        <RefreshCw size={14} />
      </button>
      <button
        class="trigger-btn"
        onclick={triggerWorkflow}
        disabled={workflowStatus.status === 'running'}
        title="Manually trigger workflow"
      >
        <Play size={14} />
      </button>
    </div>
  </div>

  {#if error}
    <div class="error-banner">
      <AlertCircle size={16} />
      <span>{error}</span>
    </div>
  {/if}

  {#if loading}
    <div class="loading-state">
      <div class="spinner"></div>
      <span>Loading workflow status...</span>
    </div>
  {:else}
    <!-- Scheduler Status -->
    <div class="scheduler-status">
      <span class="scheduler-label">Scheduler:</span>
      <span class="scheduler-value" class:active={schedulerStatus.running}>
        {schedulerStatus.running ? 'ACTIVE' : 'INACTIVE'}
      </span>
      <span class="workflow-time">Workflow starts at {schedulerStatus.workflow_3_start}</span>
    </div>

    <!-- Steps Timeline -->
    <div class="steps-timeline">
      {#each steps as step, index}
        {@const status = getStepStatus(step.key)}
        {@const Icon = step.icon}
        <div class="step-item" style="--status-color: {getStatusColor(status)}">
          <div class="step-connector" class:first={index === 0}>
            {#if index > 0}
              <div class="connector-line" class:completed={getStepStatus(steps[index - 1].key) === 'completed'}></div>
            {/if}
          </div>
          <div class="step-icon" style="background: {getStatusBgColor(status)}">
            <Icon size={16} color={getStatusColor(status)} />
          </div>
          <div class="step-content">
            <div class="step-label">{step.label}</div>
            <div class="step-time">{step.time}</div>
            <div class="step-status-badge" style="background: {getStatusBgColor(status)}; color: {getStatusColor(status)}">
              {#if status === 'completed'}
                <CheckCircle size={12} />
              {:else if status === 'running'}
                <RefreshCw size={12} class="spinning" />
              {:else if status === 'failed'}
                <AlertCircle size={12} />
              {/if}
              <span>{status.toUpperCase()}</span>
            </div>
          </div>
        </div>
      {/each}
    </div>

    <!-- Current Run Info -->
    {#if workflowStatus.run_id}
      <div class="run-info">
        <div class="run-id">Run ID: {workflowStatus.run_id.slice(0, 8)}...</div>
        {#if workflowStatus.started_at}
          <div class="run-time">Started: {new Date(workflowStatus.started_at).toLocaleString()}</div>
        {/if}
        {#if workflowStatus.error}
          <div class="run-error">Error: {workflowStatus.error}</div>
        {/if}
      </div>
    {/if}
  {/if}
</div>

<!-- EOD Report Modal -->
{#if showEodReportModal}
  <div class="modal-overlay" onclick={() => showEodReportModal = false}>
    <div class="modal-content" onclick={(e) => e.stopPropagation()}>
      <button class="modal-close" onclick={() => showEodReportModal = false}>
        <X size={18} />
      </button>
      <EodReportViewer />
    </div>
  </div>
{/if}

<style>
  .dead-zone-panel {
    background: rgba(8, 8, 12, 0.75);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 16px;
    color: #e4e4e7;
  }

  .panel-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 16px;
    flex-wrap: wrap;
  }

  .panel-header h3 {
    font-size: 14px;
    font-weight: 600;
    margin: 0;
    flex: 1;
  }

  .workflow-status {
    font-size: 10px;
    padding: 2px 8px;
    border-radius: 10px;
    background: rgba(107, 114, 128, 0.2);
    color: #9ca3af;
  }

  .workflow-status.running {
    background: rgba(245, 158, 11, 0.2);
    color: #f59e0b;
  }

  .last-update {
    font-size: 11px;
    color: #6b7280;
  }

  .header-actions {
    display: flex;
    gap: 6px;
  }

  .eod-report-btn {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 6px 12px;
    background: rgba(16, 185, 129, 0.15);
    border: 1px solid rgba(16, 185, 129, 0.3);
    border-radius: 6px;
    color: #10b981;
    font-size: 12px;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.2s;
  }

  .eod-report-btn:hover {
    background: rgba(16, 185, 129, 0.25);
    border-color: rgba(16, 185, 129, 0.5);
    color: #10b981;
  }

  .refresh-btn, .trigger-btn {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 6px;
    padding: 6px;
    cursor: pointer;
    color: #9ca3af;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: all 0.2s;
  }

  .refresh-btn:hover, .trigger-btn:hover:not(:disabled) {
    background: rgba(255, 255, 255, 0.1);
    color: #e4e4e7;
  }

  .trigger-btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .error-banner {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 10px;
    background: rgba(239, 68, 68, 0.15);
    border: 1px solid rgba(239, 68, 68, 0.3);
    border-radius: 8px;
    margin-bottom: 16px;
    color: #ef4444;
    font-size: 13px;
  }

  .loading-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 12px;
    padding: 32px;
    color: #6b7280;
  }

  .spinner {
    width: 24px;
    height: 24px;
    border: 2px solid rgba(255, 255, 255, 0.1);
    border-top-color: #10b981;
    border-radius: 50%;
    animation: spin 1s linear infinite;
  }

  @keyframes spin {
    to { transform: rotate(360deg); }
  }

  .scheduler-status {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 10px 12px;
    background: rgba(255, 255, 255, 0.03);
    border-radius: 8px;
    margin-bottom: 20px;
    font-size: 12px;
  }

  .scheduler-label {
    color: #6b7280;
  }

  .scheduler-value {
    padding: 2px 8px;
    border-radius: 4px;
    background: rgba(107, 114, 128, 0.2);
    color: #9ca3af;
  }

  .scheduler-value.active {
    background: rgba(16, 185, 129, 0.2);
    color: #10b981;
  }

  .workflow-time {
    margin-left: auto;
    color: #6b7280;
  }

  .steps-timeline {
    display: flex;
    flex-direction: column;
    gap: 0;
  }

  .step-item {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    position: relative;
  }

  .step-connector {
    display: flex;
    flex-direction: column;
    align-items: center;
    width: 20px;
  }

  .step-connector.first {
    display: none;
  }

  .connector-line {
    width: 2px;
    height: 24px;
    background: rgba(107, 114, 128, 0.3);
  }

  .connector-line.completed {
    background: rgba(16, 185, 129, 0.5);
  }

  .step-icon {
    width: 36px;
    height: 36px;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
  }

  .step-content {
    flex: 1;
    padding-bottom: 16px;
  }

  .step-label {
    font-size: 13px;
    font-weight: 500;
    color: #e4e4e7;
    margin-bottom: 2px;
  }

  .step-time {
    font-size: 11px;
    color: #6b7280;
    margin-bottom: 6px;
  }

  .step-status-badge {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 10px;
    font-weight: 500;
  }

  .step-status-badge :global(.spinning) {
    animation: spin 1s linear infinite;
  }

  .run-info {
    margin-top: 16px;
    padding: 12px;
    background: rgba(255, 255, 255, 0.03);
    border-radius: 8px;
    font-size: 12px;
  }

  .run-id {
    font-family: monospace;
    color: #9ca3af;
    margin-bottom: 4px;
  }

  .run-time {
    color: #6b7280;
  }

  .run-error {
    color: #ef4444;
    margin-top: 4px;
  }

  .modal-overlay {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(0, 0, 0, 0.75);
    backdrop-filter: blur(4px);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 1000;
    padding: 20px;
  }

  .modal-content {
    position: relative;
    width: 100%;
    max-width: 600px;
    max-height: 90vh;
    overflow-y: auto;
    animation: slideIn 0.2s ease-out;
  }

  @keyframes slideIn {
    from {
      opacity: 0;
      transform: translateY(-20px);
    }
    to {
      opacity: 1;
      transform: translateY(0);
    }
  }

  .modal-close {
    position: absolute;
    top: 12px;
    right: 12px;
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 6px;
    padding: 6px;
    cursor: pointer;
    color: #9ca3af;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: all 0.2s;
    z-index: 10;
  }

  .modal-close:hover {
    background: rgba(255, 255, 255, 0.1);
    color: #e4e4e7;
  }
</style>
