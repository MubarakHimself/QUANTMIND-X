<!-- @migration-task Error while migrating Svelte code: This type of directive is not valid on components
https://svelte.dev/e/component_invalid_directive -->
<!-- @migration-task Error while migrating Svelte code: This type of directive is not valid on components
https://svelte.dev/e/component_invalid_directive -->
<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { createEventDispatcher } from 'svelte';
  import { 
    Activity, Cpu, HardDrive, Wifi, Zap, Bot, TrendingUp,
    Database, RefreshCw, Settings, Maximize2, Minimize2, Clock,
    ArrowUp, ArrowDown, AlertTriangle, CheckCircle, XCircle
  } from 'lucide-svelte';

  import MetricCard from './MetricCard.svelte';

  const dispatch = createEventDispatcher();

  // Lifecycle stats
  interface LifecycleStats {
    promotionsToday: number;
    quarantinedToday: number;
    activeBots: number;
    nextCheck: string;
  }

  interface LifecycleEvent {
    id: number;
    bot_id: string;
    from_tag: string;
    to_tag: string;
    reason: string;
    timestamp: string;
    triggered_by: string;
  }

  let stats: LifecycleStats = {
    promotionsToday: 0,
    quarantinedToday: 0,
    activeBots: 0,
    nextCheck: '03:00 UTC'
  };

  let recentEvents: LifecycleEvent[] = [];
  let isLoading = false;
  let error: string | null = null;
  let isFullscreen = false;
  let refreshInterval: ReturnType<typeof setInterval> | null = null;

  // WebSocket connection for real-time updates
  let ws: WebSocket | null = null;

  onMount(() => {
    fetchLifecycleData();
    connectWebSocket();
    
    // Auto-refresh every 30 seconds
    refreshInterval = setInterval(fetchLifecycleData, 30000);
  });

  onDestroy(() => {
    if (ws) {
      ws.close();
    }
    if (refreshInterval) {
      clearInterval(refreshInterval);
    }
  });

  async function fetchLifecycleData() {
    isLoading = true;
    error = null;
    
    try {
      const response = await fetch('/api/lifecycle/stats');
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      stats = data.stats || stats;
      recentEvents = data.recent_events || [];
    } catch (e) {
      console.error('Failed to fetch lifecycle data:', e);
      error = e instanceof Error ? e.message : 'Failed to fetch data';
    } finally {
      isLoading = false;
    }
  }

  function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;
    
    ws = new WebSocket(wsUrl);
    
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === 'lifecycle_event') {
          // Add new event to the top of the list
          recentEvents = [data.event, ...recentEvents].slice(0, 50);
          
          // Update stats
          if (data.event.to_tag === '@quarantine') {
            stats.quarantinedToday++;
          } else if (['@pending', '@perfect', '@live'].includes(data.event.to_tag)) {
            stats.promotionsToday++;
          }
        }
      } catch (e) {
        console.error('WebSocket message parse error:', e);
      }
    };
    
    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };
  }

  function getTagColor(tag: string): string {
    switch (tag) {
      case '@primal': return 'var(--tag-primal, #6366f1)';
      case '@pending': return 'var(--tag-pending, #f59e0b)';
      case '@perfect': return 'var(--tag-perfect, #10b981)';
      case '@live': return 'var(--tag-live, #3b82f6)';
      case '@quarantine': return 'var(--tag-quarantine, #ef4444)';
      case '@dead': return 'var(--tag-dead, #6b7280)';
      default: return 'var(--tag-default, #9ca3af)';
    }
  }

  function getTagIcon(tag: string) {
    switch (tag) {
      case '@quarantine': return AlertTriangle;
      case '@dead': return XCircle;
      default: return CheckCircle;
    }
  }

  function formatTimestamp(timestamp: string): string {
    const date = new Date(timestamp);
    return date.toLocaleString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  }

  function toggleFullscreen() {
    isFullscreen = !isFullscreen;
    dispatch('fullscreen', { isFullscreen });
  }
</script>

<div class="lifecycle-panel" class:fullscreen={isFullscreen}>
  <div class="panel-header">
    <div class="header-left">
      <Activity size={20} />
      <h3>Bot Lifecycle Management</h3>
    </div>
    <div class="header-right">
      <span class="next-check">
        <Clock size={14} />
        Next check: {stats.nextCheck}
      </span>
      <button class="icon-btn" on:click={fetchLifecycleData} title="Refresh">
        <RefreshCw size={16} class:spin={isLoading} />
      </button>
      <button class="icon-btn" on:click={toggleFullscreen} title={isFullscreen ? 'Minimize' : 'Maximize'}>
        {#if isFullscreen}
          <Minimize2 size={16} />
        {:else}
          <Maximize2 size={16} />
        {/if}
      </button>
    </div>
  </div>

  {#if error}
    <div class="error-banner">
      {error}
    </div>
  {/if}

  <div class="stats-grid">
    <div class="stat-card promotion">
      <div class="stat-icon">
        <ArrowUp size={24} />
      </div>
      <div class="stat-content">
        <span class="stat-value">{stats.promotionsToday}</span>
        <span class="stat-label">Promotions Today</span>
      </div>
    </div>
    
    <div class="stat-card quarantine">
      <div class="stat-icon">
        <ArrowDown size={24} />
      </div>
      <div class="stat-content">
        <span class="stat-value">{stats.quarantinedToday}</span>
        <span class="stat-label">Quarantined Today</span>
      </div>
    </div>
    
    <div class="stat-card active">
      <div class="stat-icon">
        <Bot size={24} />
      </div>
      <div class="stat-content">
        <span class="stat-value">{stats.activeBots}</span>
        <span class="stat-label">Active Bots</span>
      </div>
    </div>
    
    <div class="stat-card next">
      <div class="stat-icon">
        <Clock size={24} />
      </div>
      <div class="stat-content">
        <span class="stat-value">{stats.nextCheck}</span>
        <span class="stat-label">Next Check</span>
      </div>
    </div>
  </div>

  <div class="timeline-section">
    <h4>Recent Lifecycle Events</h4>
    <div class="timeline">
      {#if isLoading && recentEvents.length === 0}
        <div class="loading">Loading...</div>
      {:else if recentEvents.length === 0}
        <div class="empty">No lifecycle events yet</div>
      {:else}
        {#each recentEvents as event (event.id)}
          <div class="timeline-item">
            <div class="timeline-marker" style="background-color: {getTagColor(event.to_tag)}">
              {#if event.to_tag === '@quarantine'}
                <AlertTriangle size={12} />
              {:else if event.to_tag === '@dead'}
                <XCircle size={12} />
              {:else}
                <CheckCircle size={12} />
              {/if}
            </div>
            <div class="timeline-content">
              <div class="event-header">
                <span class="bot-id">{event.bot_id}</span>
                <span class="tag-change">
                  <span class="tag from" style="background-color: {getTagColor(event.from_tag)}">{event.from_tag}</span>
                  <span class="arrow">→</span>
                  <span class="tag to" style="background-color: {getTagColor(event.to_tag)}">{event.to_tag}</span>
                </span>
              </div>
              <div class="event-reason">{event.reason}</div>
              <div class="event-meta">
                <span class="timestamp">{formatTimestamp(event.timestamp)}</span>
                <span class="triggered-by">by {event.triggered_by}</span>
              </div>
            </div>
          </div>
        {/each}
      {/if}
    </div>
  </div>
</div>

<style>
  .lifecycle-panel {
    background: var(--bg-secondary, #1e293b);
    border-radius: 8px;
    padding: 16px;
    display: flex;
    flex-direction: column;
    gap: 16px;
    height: 100%;
    overflow: hidden;
  }

  .lifecycle-panel.fullscreen {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    z-index: 1000;
    border-radius: 0;
  }

  .panel-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .header-left {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .header-left h3 {
    margin: 0;
    font-size: 16px;
    font-weight: 600;
  }

  .header-right {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .next-check {
    display: flex;
    align-items: center;
    gap: 4px;
    font-size: 12px;
    color: var(--text-secondary, #94a3b8);
  }

  .icon-btn {
    background: none;
    border: none;
    color: var(--text-secondary, #94a3b8);
    cursor: pointer;
    padding: 4px;
    border-radius: 4px;
    transition: all 0.2s;
  }

  .icon-btn:hover {
    background: var(--bg-hover, #334155);
    color: var(--text-primary, #f1f5f9);
  }

  :global(.spin) {
    animation: spin 1s linear infinite;
  }

  @keyframes spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
  }

  .error-banner {
    background: var(--accent-danger, #ef4444);
    color: white;
    padding: 8px 12px;
    border-radius: 4px;
    font-size: 14px;
  }

  .stats-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
  }

  .stat-card {
    background: var(--bg-tertiary, #334155);
    border-radius: 8px;
    padding: 12px;
    display: flex;
    align-items: center;
    gap: 12px;
  }

  .stat-card.promotion .stat-icon {
    color: var(--accent-success, #10b981);
  }

  .stat-card.quarantine .stat-icon {
    color: var(--accent-danger, #ef4444);
  }

  .stat-card.active .stat-icon {
    color: var(--accent-primary, #3b82f6);
  }

  .stat-card.next .stat-icon {
    color: var(--accent-warning, #f59e0b);
  }

  .stat-content {
    display: flex;
    flex-direction: column;
  }

  .stat-value {
    font-size: 20px;
    font-weight: 700;
    color: var(--text-primary, #f1f5f9);
  }

  .stat-label {
    font-size: 11px;
    color: var(--text-secondary, #94a3b8);
    text-transform: uppercase;
  }

  .timeline-section {
    flex: 1;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }

  .timeline-section h4 {
    margin: 0 0 12px 0;
    font-size: 14px;
    font-weight: 600;
    color: var(--text-primary, #f1f5f9);
  }

  .timeline {
    flex: 1;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .loading, .empty {
    text-align: center;
    color: var(--text-secondary, #94a3b8);
    padding: 24px;
  }

  .timeline-item {
    display: flex;
    gap: 12px;
    padding: 8px;
    background: var(--bg-tertiary, #334155);
    border-radius: 6px;
  }

  .timeline-marker {
    width: 24px;
    height: 24px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    flex-shrink: 0;
  }

  .timeline-content {
    flex: 1;
    min-width: 0;
  }

  .event-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 4px;
  }

  .bot-id {
    font-weight: 600;
    font-size: 13px;
    color: var(--text-primary, #f1f5f9);
  }

  .tag-change {
    display: flex;
    align-items: center;
    gap: 4px;
  }

  .tag {
    font-size: 10px;
    padding: 2px 6px;
    border-radius: 4px;
    color: white;
    font-weight: 600;
  }

  .arrow {
    color: var(--text-secondary, #94a3b8);
    font-size: 10px;
  }

  .event-reason {
    font-size: 12px;
    color: var(--text-secondary, #94a3b8);
    margin-bottom: 4px;
  }

  .event-meta {
    display: flex;
    justify-content: space-between;
    font-size: 11px;
    color: var(--text-muted, #64748b);
  }

  @media (max-width: 768px) {
    .stats-grid {
      grid-template-columns: repeat(2, 1fr);
    }
  }
</style>
