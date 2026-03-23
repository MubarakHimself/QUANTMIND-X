<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import { fade, slide } from 'svelte/transition';
  import { ChevronDown, ChevronUp, Clock, CheckCircle, XCircle, X, Loader2 } from 'lucide-svelte';

  const dispatch = createEventDispatcher();

  // Session interface
  interface Session {
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

  interface Props {
    session: Session;
    expanded?: boolean;
  }

  let { session, expanded = false }: Props = $props();

  // Computed values
  let progress = $derived(session.strategies_total > 0
    ? Math.round((session.strategies_completed / session.strategies_total) * 100)
    : 0);

  let isRunning = $derived(session.status === 'running');
  let isCompleted = $derived(session.status === 'completed');
  let isFailed = $derived(session.status === 'failed');
  let isCancelled = $derived(session.status === 'cancelled');

  let statusColor = $derived(isRunning ? 'var(--color-accent-cyan)' :
    isCompleted ? 'var(--color-accent-green)' :
    isFailed ? 'var(--color-accent-red)' :
    'var(--color-text-muted)');

  // Status background color
  let statusBg = $derived(isRunning ? 'oklch(65% 0.18 250 / 0.15)' :
    isCompleted ? 'oklch(70% 0.18 145 / 0.15)' :
    isFailed ? 'oklch(65% 0.20 25 / 0.15)' :
    'oklch(50% 0.03 260 / 0.15)');

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
          <span class="spin"><svelte:component this={Loader2} size={12} /></span>
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
    background: var(--color-bg-elevated);
    border: 1px solid var(--color-border-subtle);
    border-radius: 8px;
    padding: 12px;
    cursor: pointer;
    transition: all 0.2s ease;
    user-select: none;
  }

  .session-card:hover {
    border-color: var(--color-border-medium);
    box-shadow: 0 2px 8px oklch(0% 0 0 / 0.2);
  }

  .session-card:focus-visible {
    outline: 2px solid var(--color-accent-cyan);
    outline-offset: 2px;
  }

  .session-card.running {
    border-left: 3px solid var(--color-accent-cyan);
  }

  .session-card.completed {
    border-left: 3px solid var(--color-accent-green);
  }

  .session-card.failed {
    border-left: 3px solid var(--color-accent-red);
  }

  .session-card.cancelled {
    border-left: 3px solid var(--color-text-muted);
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
    color: var(--color-text-primary);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .session-id {
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    font-size: 0.75rem;
    color: var(--color-text-muted);
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
    color: var(--color-text-secondary);
    font-weight: 500;
  }

  .time-ago {
    display: flex;
    align-items: center;
    gap: 4px;
    color: var(--color-text-muted);
  }

  .progress-bar {
    height: 4px;
    background: var(--color-bg-surface);
    border-radius: 2px;
    overflow: hidden;
  }

  .progress-fill {
    height: 100%;
    border-radius: 2px;
    transition: width 0.3s ease, background-color 0.3s ease;
  }

  .progress-fill.running {
    background: linear-gradient(90deg, var(--color-accent-cyan), oklch(75% 0.18 250));
    animation: pulse-glow 2s ease-in-out infinite;
  }

  .progress-fill.completed {
    background: var(--color-accent-green);
  }

  .progress-fill.failed {
    background: var(--color-accent-red);
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
    border-top: 1px solid var(--color-border-subtle);
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
    color: var(--color-text-muted);
  }

  .meta-value {
    font-size: 0.875rem;
    color: var(--color-text-secondary);
  }

  .strategies-list {
    display: flex;
    flex-direction: column;
    gap: 6px;
  }

  .strategies-header {
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--color-text-secondary);
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
    background: var(--color-bg-surface);
    border-radius: 4px;
    font-size: 0.875rem;
    transition: background-color 0.15s ease;
  }

  .strategy-item:hover {
    background: var(--color-bg-elevated);
  }

  .strategy-item.completed {
    border-left: 2px solid var(--color-accent-green);
  }

  .strategy-item.failed {
    border-left: 2px solid var(--color-accent-red);
  }

  .strategy-name {
    color: var(--color-text-primary);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .strategy-status {
    font-size: 0.75rem;
    color: var(--color-text-muted);
    text-transform: capitalize;
  }
</style>
