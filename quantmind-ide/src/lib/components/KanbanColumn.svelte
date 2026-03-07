<script lang="ts">
  import {
    Inbox,
    Cpu,
    FileCode,
    CheckCircle,
    Clock,
    AlertCircle
  } from 'lucide-svelte';

  export let title = '';
  export let status: 'inbox' | 'processing' | 'extracting' | 'done' = 'inbox';
  export let strategies: any[] = [];
  export let isLoading = false;

  $: columnIcon = getColumnIcon();
  $: count = strategies.length;

  function getColumnIcon() {
    switch (status) {
      case 'inbox': return Inbox;
      case 'processing': return Cpu;
      case 'extracting': return FileCode;
      case 'done': return CheckCircle;
      default: return Inbox;
    }
  }

  function getStatusColor(): string {
    switch (status) {
      case 'inbox': return 'var(--tag-inbox, #6366f1)';
      case 'processing': return 'var(--tag-processing, #f59e0b)';
      case 'extracting': return 'var(--tag-extracting, #3b82f6)';
      case 'done': return 'var(--tag-done, #10b981)';
      default: return 'var(--tag-default, #9ca3af)';
    }
  }

  function formatDate(dateStr: string): string {
    const date = new Date(dateStr);
    return date.toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric'
    });
  }

  function getStrategyStatusBadge(strategy: any): { label: string; color: string } | null {
    if (strategy.status === 'quarantined') {
      return { label: 'Quarantined', color: '#ef4444' };
    }
    return null;
  }
</script>

<div class="kanban-column" class:loading={isLoading}>
  <!-- Column Header -->
  <div class="column-header">
    <div class="header-left">
      <svelte:component this={columnIcon} size={18} />
      <h3>{title}</h3>
    </div>
    <div class="count-badge" style="background-color: {getStatusColor()}">
      {count}
    </div>
  </div>

  <!-- Strategy Cards -->
  <div class="column-content">
    {#if isLoading}
      <div class="loading-skeletons">
        {#each Array(3) as _}
          <div class="card-skeleton"></div>
        {/each}
      </div>
    {:else if strategies.length === 0}
      <div class="empty-state">
        <AlertCircle size={24} />
        <span>No strategies</span>
      </div>
    {:else}
      {#each strategies as strategy (strategy.id)}
        <div class="strategy-card">
          <!-- Card Header -->
          <div class="card-header">
            <span class="strategy-name">{strategy.name}</span>
            {#if getStrategyStatusBadge(strategy)}
              {@const badge = getStrategyStatusBadge(strategy)}
              <span class="status-badge" style="background-color: {badge.color}">
                {badge.label}
              </span>
            {/if}
          </div>

          <!-- Card Body -->
          <div class="card-body">
            <!-- Created Date -->
            <div class="card-meta">
              <Clock size={12} />
              <span>{formatDate(strategy.created_at)}</span>
            </div>

            <!-- File Indicators -->
            <div class="file-indicators">
              {#if strategy.has_video_ingest}
                <span class="file-tag video_ingest">Video Ingest</span>
              {/if}
              {#if strategy.has_trd}
                <span class="file-tag trd">TRD</span>
              {/if}
              {#if strategy.has_ea}
                <span class="file-tag ea">EA</span>
              {/if}
              {#if strategy.has_backtest}
                <span class="file-tag backtest">Backtest</span>
              {/if}
            </div>
          </div>

          <!-- Card Footer -->
          <div class="card-footer">
            <span class="strategy-id">ID: {strategy.id.slice(0, 8)}...</span>
          </div>
        </div>
      {/each}
    {/if}
  </div>
</div>

<style>
  .kanban-column {
    display: flex;
    flex-direction: column;
    background: var(--bg-secondary, #1e293b);
    border-radius: 8px;
    height: 100%;
    overflow: hidden;
    transition: background 0.2s ease;
  }

  .kanban-column:hover {
    background: var(--bg-secondary-hover, #263344);
  }

  .column-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 16px;
    border-bottom: 1px solid var(--border-color, #334155);
  }

  .header-left {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .header-left h3 {
    margin: 0;
    font-size: 14px;
    font-weight: 600;
    color: var(--text-primary, #f1f5f9);
  }

  .count-badge {
    min-width: 24px;
    height: 24px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 12px;
    font-size: 12px;
    font-weight: 700;
    color: white;
  }

  .column-content {
    flex: 1;
    overflow-y: auto;
    padding: 12px;
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .loading-skeletons {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .card-skeleton {
    height: 100px;
    background: linear-gradient(90deg, var(--bg-tertiary, #334155) 25%, var(--bg-secondary, #1e293b) 50%, var(--bg-tertiary, #334155) 75%);
    background-size: 200% 100%;
    animation: skeleton-loading 1.5s infinite;
    border-radius: 6px;
  }

  @keyframes skeleton-loading {
    0% { background-position: 200% 0; }
    100% { background-position: -200% 0; }
  }

  .empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 32px 16px;
    gap: 8px;
    color: var(--text-muted, #64748b);
    text-align: center;
  }

  .empty-state span {
    font-size: 13px;
  }

  .strategy-card {
    background: var(--bg-tertiary, #334155);
    border-radius: 6px;
    padding: 12px;
    cursor: pointer;
    transition: all 0.2s ease;
    border: 1px solid transparent;
  }

  .strategy-card:hover {
    border-color: var(--accent-primary, #3b82f6);
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
  }

  .card-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 8px;
    gap: 8px;
  }

  .strategy-name {
    font-weight: 600;
    font-size: 13px;
    color: var(--text-primary, #f1f5f9);
    word-break: break-word;
  }

  .status-badge {
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 10px;
    font-weight: 600;
    color: white;
    white-space: nowrap;
    flex-shrink: 0;
  }

  .card-body {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .card-meta {
    display: flex;
    align-items: center;
    gap: 4px;
    color: var(--text-secondary, #94a3b8);
    font-size: 11px;
  }

  .file-indicators {
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
  }

  .file-tag {
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 9px;
    font-weight: 600;
    text-transform: uppercase;
    color: white;
  }

  .file-tag.video_ingest {
    background-color: var(--file-video-ingest, #8b5cf6);
  }

  .file-tag.trd {
    background-color: var(--file-trd, #06b6d4);
  }

  .file-tag.ea {
    background-color: var(--file-ea, #f97316);
  }

  .file-tag.backtest {
    background-color: var(--file-backtest, #10b981);
  }

  .card-footer {
    margin-top: 8px;
    padding-top: 8px;
    border-top: 1px solid var(--border-color, #475569);
  }

  .strategy-id {
    font-size: 10px;
    color: var(--text-muted, #64748b);
    font-family: 'Courier New', monospace;
  }

  /* Scrollbar styling */
  .column-content::-webkit-scrollbar {
    width: 6px;
  }

  .column-content::-webkit-scrollbar-track {
    background: var(--bg-secondary, #1e293b);
  }

  .column-content::-webkit-scrollbar-thumb {
    background: var(--border-color, #475569);
    border-radius: 3px;
  }

  .column-content::-webkit-scrollbar-thumb:hover {
    background: var(--text-muted, #64748b);
  }
</style>
