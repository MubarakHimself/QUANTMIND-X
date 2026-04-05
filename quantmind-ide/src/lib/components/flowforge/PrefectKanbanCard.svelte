<script lang="ts">
  /**
   * PrefectKanbanCard Component
   *
   * Displays a Prefect workflow card with:
   * - Workflow name, department
   * - State badge with cyan pulse border for RUNNING
   * - Duration, step progress (X/Y), next step
   * - Per-card workflow kill switch (only on RUNNING cards)
   */

  import type { PrefectWorkflow } from '$lib/stores/flowforge';
  import {
    Square,
    Clock,
    BarChart3,
    ArrowRight,
    AlertTriangle,
    FolderTree,
    Pause,
    Play,
    RotateCcw,
  } from 'lucide-svelte';

  interface Props {
    workflow: PrefectWorkflow;
    onKillSwitch?: (workflow: PrefectWorkflow) => void;
    onPause?: (workflow: PrefectWorkflow) => void;
    onResume?: (workflow: PrefectWorkflow) => void;
    onRetry?: (workflow: PrefectWorkflow) => void;
    onClick?: (workflow: PrefectWorkflow) => void;
  }

  let { workflow, onKillSwitch, onPause, onResume, onRetry, onClick }: Props = $props();

  // Format duration
  function formatDuration(seconds: number): string {
    if (seconds === 0) return '-';
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    if (hours > 0) return `${hours}h ${minutes}m`;
    return `${minutes}m`;
  }

  // Handle kill switch click
  function handleKillSwitch(e: MouseEvent) {
    e.stopPropagation();
    if (onKillSwitch) {
      onKillSwitch(workflow);
    }
  }

  function handlePause(e: MouseEvent) {
    e.stopPropagation();
    onPause?.(workflow);
  }

  function handleResume(e: MouseEvent) {
    e.stopPropagation();
    onResume?.(workflow);
  }

  function handleRetry(e: MouseEvent) {
    e.stopPropagation();
    onRetry?.(workflow);
  }

  // Handle card click
  function handleClick() {
    if (onClick) {
      onClick(workflow);
    }
  }

  // State colors
  const stateColors: Record<string, string> = {
    PENDING: '#94a3b8',
    RUNNING: '#06b6d4',
    PENDING_REVIEW: '#f59e0b',
    DONE: '#22c55e',
    CANCELLED: '#ef4444',
    EXPIRED_REVIEW: '#dc2626',
  };

  const isRunning = $derived(workflow.state === 'RUNNING');
  const hasBlockingError = $derived(!!workflow.blocking_error);
  const canPause = $derived(Boolean(workflow.can_pause && onPause));
  const canResume = $derived(Boolean(workflow.can_resume && onResume));
  const canRetry = $derived(Boolean(workflow.can_retry && onRetry));
</script>

<div
  class="kanban-card {workflow.state.toLowerCase()}"
  class:pulse-border={isRunning}
  class:cancelled-state={workflow.state === 'CANCELLED'}
  onclick={handleClick}
  role="button"
  tabindex="0"
  onkeydown={(e) => e.key === 'Enter' && handleClick()}
>
  <!-- Card Header -->
  <div class="card-header">
    <!-- AC5: Strikethrough visual for cancelled state -->
    <span class="workflow-name" class:strikethrough={workflow.state === 'CANCELLED'} title={workflow.name}>{workflow.name}</span>
    <span class="department">{workflow.department}</span>
  </div>

  <!-- State Badge -->
  <div class="state-badge" style="--state-color: {stateColors[workflow.state]}">
    <span class="state-dot"></span>
    <span class="state-label">{workflow.state.replace('_', ' ')}</span>
  </div>

  <!-- Card Details -->
  <div class="card-details">
    {#if workflow.state === 'RUNNING'}
      <div class="detail-row">
        <Clock size={14} />
        <span class="detail-value">{formatDuration(workflow.duration_seconds)}</span>
      </div>
    {/if}

    <div class="detail-row">
      <BarChart3 size={14} />
      <span class="detail-value">
        {workflow.completed_steps}/{workflow.total_steps}
      </span>
    </div>

    <div class="detail-row next-step">
      <ArrowRight size={14} />
      <span class="detail-value" title={workflow.next_step}>{workflow.next_step}</span>
    </div>

    {#if workflow.current_stage}
      <div class="detail-row">
        <span class="detail-label">Stage</span>
        <span class="detail-value" title={workflow.current_stage}>{workflow.current_stage}</span>
      </div>
    {/if}

    {#if workflow.waiting_reason}
      <div class="detail-row waiting">
        <Clock size={14} />
        <span class="detail-value" title={workflow.waiting_reason}>Waiting: {workflow.waiting_reason}</span>
      </div>
    {/if}

    {#if workflow.latest_artifact}
      <div class="detail-row artifact">
        <FolderTree size={14} />
        <span class="detail-value" title={workflow.latest_artifact.path}>
          {workflow.latest_artifact.name}
        </span>
      </div>
    {/if}

    {#if hasBlockingError}
      <div class="detail-row error">
        <AlertTriangle size={14} />
        <span class="detail-value" title={workflow.blocking_error ?? undefined}>
          {workflow.blocking_error}
        </span>
      </div>
    {/if}
  </div>

  <div class="card-actions">
    {#if canPause}
      <button class="action-button pause" onclick={handlePause} title="Pause Workflow">
        <Pause size={14} />
      </button>
    {/if}

    {#if canResume}
      <button class="action-button resume" onclick={handleResume} title="Resume Workflow">
        <Play size={14} />
      </button>
    {/if}

    {#if canRetry}
      <button class="action-button retry" onclick={handleRetry} title="Retry Workflow">
        <RotateCcw size={14} />
      </button>
    {/if}

    {#if workflow.state === 'RUNNING' && onKillSwitch}
      <button class="action-button kill-switch" onclick={handleKillSwitch} title="Stop Workflow">
        <Square size={16} />
      </button>
    {/if}
  </div>
</div>

<style>
  .kanban-card {
    position: relative;
    padding: 12px;
    background: rgba(30, 32, 40, 0.85);
    border-radius: 8px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    cursor: pointer;
    transition: all 0.2s ease;
    display: flex;
    flex-direction: column;
    gap: 8px;
    padding-bottom: 44px;
  }

  .kanban-card:hover {
    background: rgba(40, 44, 55, 0.9);
    border-color: rgba(255, 255, 255, 0.12);
    transform: translateY(-2px);
  }

  /* Pulse border for RUNNING state */
  .pulse-border {
    animation: pulse 2s ease-in-out infinite;
  }

  @keyframes pulse {
    0%, 100% {
      border-color: rgba(6, 182, 212, 0.4);
      box-shadow: 0 0 0 0 rgba(6, 182, 212, 0);
    }
    50% {
      border-color: rgba(6, 182, 212, 0.8);
      box-shadow: 0 0 12px 2px rgba(6, 182, 212, 0.3);
    }
  }

  /* Card Header */
  .card-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 8px;
  }

  .workflow-name {
    font-weight: 600;
    font-size: 14px;
    color: #f1f5f9;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    flex: 1;
  }

  .department {
    font-size: 11px;
    color: #94a3b8;
    background: rgba(148, 163, 184, 0.15);
    padding: 2px 6px;
    border-radius: 4px;
    white-space: nowrap;
  }

  /* State Badge */
  .state-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 8px;
    border-radius: 4px;
    background: rgba(255, 255, 255, 0.05);
    width: fit-content;
  }

  .state-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--state-color);
  }

  .state-label {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    color: var(--state-color);
    letter-spacing: 0.5px;
  }

  /* Card Details */
  .card-details {
    display: flex;
    flex-direction: column;
    gap: 4px;
    margin-top: 4px;
  }

  .detail-row {
    display: flex;
    align-items: center;
    gap: 6px;
    color: #94a3b8;
    font-size: 12px;
  }

  .detail-label {
    min-width: 36px;
    color: #64748b;
    text-transform: uppercase;
    font-size: 10px;
    letter-spacing: 0.05em;
  }

  .detail-value {
    color: #cbd5e1;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    max-width: 140px;
  }

  .card-actions {
    position: absolute;
    right: 10px;
    bottom: 10px;
    display: inline-flex;
    align-items: center;
    gap: 6px;
  }

  .action-button {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 30px;
    height: 30px;
    border-radius: 7px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    background: rgba(15, 23, 42, 0.9);
    color: #cbd5e1;
    transition: all 0.16s ease;
  }

  .action-button:hover {
    transform: translateY(-1px);
    border-color: rgba(255, 255, 255, 0.16);
  }

  .action-button.pause {
    color: #f8fafc;
    background: rgba(148, 163, 184, 0.14);
  }

  .action-button.resume {
    color: #86efac;
    background: rgba(34, 197, 94, 0.12);
  }

  .action-button.retry {
    color: #fbbf24;
    background: rgba(245, 158, 11, 0.12);
  }

  .next-step .detail-value {
    color: #64748b;
    font-style: italic;
  }

  .detail-row.waiting .detail-value {
    color: #f59e0b;
  }

  .detail-row.artifact .detail-value {
    color: #7dd3fc;
  }

  .detail-row.error .detail-value {
    color: #fca5a5;
  }

  .action-button.kill-switch {
    color: #fca5a5;
    background: rgba(239, 68, 68, 0.15);
    border-color: rgba(239, 68, 68, 0.22);
  }

  .action-button.kill-switch:hover,
  .action-button.kill-switch:focus {
    background: rgba(239, 68, 68, 0.22);
    border-color: rgba(239, 68, 68, 0.7);
    outline: none;
  }

  .action-button.kill-switch:focus-visible {
    box-shadow: 0 0 0 2px rgba(239, 68, 68, 0.5);
  }

  /* State-specific border colors */
  .kanban-card.running {
    border-left: 3px solid #06b6d4;
  }

  .kanban-card.pending {
    border-left: 3px solid #94a3b8;
  }

  .kanban-card.pending_review {
    border-left: 3px solid #f59e0b;
  }

  .kanban-card.done {
    border-left: 3px solid #22c55e;
  }

  .kanban-card.cancelled {
    border-left: 3px solid #ef4444;
    opacity: 0.7;
  }

  /* AC5: Cancelled state visual with strikethrough */
  .kanban-card.cancelled-state {
    background: rgba(30, 32, 40, 0.6);
  }

  .workflow-name.strikethrough {
    text-decoration: line-through;
    text-decoration-color: #ef4444;
    text-decoration-thickness: 2px;
    color: #94a3b8;
  }

  .kanban-card.expired_review {
    border-left: 3px solid #dc2626;
  }
</style>
