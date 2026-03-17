<!-- @migration-task Error while migrating Svelte code: This type of directive is not valid on components
https://svelte.dev/e/component_invalid_directive -->
<!-- @migration-task Error while migrating Svelte code: This type of directive is not valid on components
https://svelte.dev/e/component_invalid_directive -->
<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import { fade, slide } from 'svelte/transition';
  import { ChevronDown, ChevronUp, Clock, CheckCircle, XCircle, X, Loader2 } from 'lucide-svelte';

  const dispatch = createEventDispatcher();

  // Session interface
  export interface Session {
    id: string;
    name: string;
    status: 'running' | 'completed' | 'failed' | 'cancelled';
    started_at: string;
    completed_at?: string;
    strategies_total: number;
    strategies_completed: number;
    strategies: Array<{
      id: string;
      name: string;
      status: string;
    }>;
  }

  export let session!: Session;
  export let expanded = false;

  // Computed values
  $: progress = session.strategies_total > 0
    ? Math.round((session.strategies_completed / session.strategies_total) * 100)
    : 0;

  $: isRunning = session.status === 'running';
  $: isCompleted = session.status === 'completed';
  $: isFailed = session.status === 'failed';
  $: isCancelled = session.status === 'cancelled';

  $: statusColor = isRunning ? 'var(--accent-primary)' :
    isCompleted ? 'var(--accent-success)' :
    isFailed ? 'var(--accent-danger)' :
    'var(--text-muted)';

  $: statusBg = isRunning ? 'oklch(65% 0.18 250 / 0.15)' :
    isCompleted ? 'oklch(70% 0.18 145 / 0.15)' :
    isFailed ? 'oklch(65% 0.20 25 / 0.15)' :
    'oklch(50% 0.03 260 / 0.15)';

  // Format timestamps
  function formatTime(isoString: string): string {
    const date = new Date(isoString);
    const now = new Date();
    const diff = now.getTime() - date.getTime();

    if (diff < 60000) return 'Just now';
    if (diff < 3600000) return `${Math.floor(diff / 60000)}m ago`;
    if (diff < 86400000) return `${Math.floor(diff / 3600000)}h ago`;
    return date.toLocaleDateString();
  }

  function formatDuration(start: string, end?: string): string {
    const startDate = new Date(start);
    const endDate = end ? new Date(end) : new Date();
    const diff = Math.floor((endDate.getTime() - startDate.getTime()) / 1000);

    if (diff < 60) return `${diff}s`;
    if (diff < 3600) return `${Math.floor(diff / 60)}m ${diff % 60}s`;
    return `${Math.floor(diff / 3600)}h ${Math.floor((diff % 3600) / 60)}m`;
  }

  function getStatusIcon() {
    if (isRunning) return Loader2;
    if (isCompleted) return CheckCircle;
    if (isFailed) return XCircle;
    return X;
  }

  function toggleExpand() {
    expanded = !expanded;
    dispatch('select', { session: session });
  }

  function handleCardClick(e: MouseEvent) {
    // Don't toggle if clicking on a strategy item
    if ((e.target as HTMLElement).closest('.strategy-item')) return;
    toggleExpand();
  }
</script>

<div
  class="session-card"
  class:running={isRunning}
  class:completed={isCompleted}
  class:failed={isFailed}
  class:cancelled={isCancelled}
  class:expanded
  on:click={handleCardClick}
  role="button"
  tabindex="0"
  on:keydown={(e) => e.key === 'Enter' && toggleExpand()}
>
  <!-- Card Header -->
  <div class="card-header">
    <div class="header-left">
      <span class="session-name">{session.name}</span>
      <span class="session-id">{session.id.slice(0, 8)}...</span>
    </div>
    <div class="header-right">
      <span class="status-badge" style="background-color: {statusBg}; color: {statusColor}">
        {#if isRunning}
          <svelte:component this={Loader2} size={12} class:spin />
        {:else}
          <svelte:component this={getStatusIcon()} size={12} />
        {/if}
        <span>{session.status}</span>
      </span>
    </div>
  </div>

  <!-- Progress Section (always visible) -->
  <div class="progress-section">
    <div class="progress-info">
      <span class="strategy-count">{session.strategies_completed} / {session.strategies_total}</span>
      <span class="time-ago">
        <Clock size={12} />
        {formatTime(session.started_at)}
      </span>
    </div>
    <div class="progress-bar">
      <div
        class="progress-fill"
        class:running={isRunning}
        class:completed={isCompleted}
        class:failed={isFailed}
        style="width: {progress}%"
      ></div>
    </div>
  </div>

  <!-- Expanded Details -->
  {#if expanded}
    <div class="card-details" in:slide={{ duration: 200 }}>
      <div class="details-meta">
        <div class="meta-item">
          <span class="meta-label">Started</span>
          <span class="meta-value">{new Date(session.started_at).toLocaleString()}</span>
        </div>
        {#if session.completed_at}
          <div class="meta-item">
            <span class="meta-label">Completed</span>
            <span class="meta-value">{new Date(session.completed_at).toLocaleString()}</span>
          </div>
        {/if}
        <div class="meta-item">
          <span class="meta-label">Duration</span>
          <span class="meta-value">{formatDuration(session.started_at, session.completed_at)}</span>
        </div>
      </div>

      {#if session.strategies && session.strategies.length > 0}
        <div class="strategies-list">
          <div class="strategies-header">Strategies ({session.strategies.length})</div>
          <div class="strategies-items">
            {#each session.strategies as strategy}
              <div class="strategy-item" class:completed={strategy.status === 'completed'} class:failed={strategy.status === 'failed'}>
                <span class="strategy-name">{strategy.name}</span>
                <span class="strategy-status">{strategy.status}</span>
              </div>
            {/each}
          </div>
        </div>
      {/if}
    </div>
  {/if}
</div>

<style>
  .session-card {
    background: var(--bg-tertiary);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
    padding: 12px;
    cursor: pointer;
    transition: all 0.2s ease;
    user-select: none;
  }

  .session-card:hover {
    border-color: var(--border-strong);
    box-shadow: 0 2px 8px oklch(0% 0 0 / 0.2);
  }

  .session-card:focus-visible {
    outline: 2px solid var(--accent-primary);
    outline-offset: 2px;
  }

  .session-card.running {
    border-left: 3px solid var(--accent-primary);
  }

  .session-card.completed {
    border-left: 3px solid var(--accent-success);
  }

  .session-card.failed {
    border-left: 3px solid var(--accent-danger);
  }

  .session-card.cancelled {
    border-left: 3px solid var(--text-muted);
  }

  .card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 10px;
  }

  .header-left {
    display: flex;
    align-items: center;
    gap: 8px;
    flex: 1;
    min-width: 0;
  }

  .session-name {
    font-weight: 600;
    font-size: 0.875rem;
    color: var(--text-primary);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .session-id {
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    font-size: 0.75rem;
    color: var(--text-muted);
  }

  .header-right {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .status-badge {
    display: flex;
    align-items: center;
    gap: 4px;
    padding: 3px 8px;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: capitalize;
    white-space: nowrap;
  }

  .status-badge :global(.spin) {
    animation: spin 1s linear infinite;
  }

  @keyframes spin {
    to {
      transform: rotate(360deg);
    }
  }

  .progress-section {
    display: flex;
    flex-direction: column;
    gap: 6px;
  }

  .progress-info {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.75rem;
  }

  .strategy-count {
    color: var(--text-secondary);
    font-weight: 500;
  }

  .time-ago {
    display: flex;
    align-items: center;
    gap: 4px;
    color: var(--text-muted);
  }

  .progress-bar {
    height: 4px;
    background: var(--bg-secondary);
    border-radius: 2px;
    overflow: hidden;
  }

  .progress-fill {
    height: 100%;
    border-radius: 2px;
    transition: width 0.3s ease, background-color 0.3s ease;
  }

  .progress-fill.running {
    background: linear-gradient(90deg, var(--accent-primary), oklch(75% 0.18 250));
    animation: pulse-glow 2s ease-in-out infinite;
  }

  .progress-fill.completed {
    background: var(--accent-success);
  }

  .progress-fill.failed {
    background: var(--accent-danger);
  }

  @keyframes pulse-glow {
    0%, 100% {
      opacity: 1;
    }
    50% {
      opacity: 0.8;
    }
  }

  .card-details {
    margin-top: 12px;
    padding-top: 12px;
    border-top: 1px solid var(--border-subtle);
  }

  .details-meta {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    margin-bottom: 12px;
  }

  .meta-item {
    display: flex;
    flex-direction: column;
    gap: 2px;
  }

  .meta-label {
    font-size: 0.75rem;
    color: var(--text-muted);
  }

  .meta-value {
    font-size: 0.875rem;
    color: var(--text-secondary);
  }

  .strategies-list {
    display: flex;
    flex-direction: column;
    gap: 6px;
  }

  .strategies-header {
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--text-secondary);
    margin-bottom: 4px;
  }

  .strategies-items {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .strategy-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 6px 8px;
    background: var(--bg-secondary);
    border-radius: 4px;
    font-size: 0.875rem;
    transition: background-color 0.15s ease;
  }

  .strategy-item:hover {
    background: var(--bg-tertiary);
  }

  .strategy-item.completed {
    border-left: 2px solid var(--accent-success);
  }

  .strategy-item.failed {
    border-left: 2px solid var(--accent-danger);
  }

  .strategy-name {
    color: var(--text-primary);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .strategy-status {
    font-size: 0.75rem;
    color: var(--text-muted);
    text-transform: capitalize;
  }
</style>
