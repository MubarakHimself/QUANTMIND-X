<script lang="ts">
  import { onMount, onDestroy } from 'svelte';

  // =============================================================================
  // Types
  // =============================================================================

  interface ActivityEvent {
    id: string;
    agent_id: string;
    agent_type: string;
    agent_name: string;
    event_type: 'action' | 'decision' | 'tool_call' | 'tool_result';
    action: string;
    timestamp: string;
    details?: Record<string, unknown>;
    reasoning?: string;
    tool_name?: string;
    tool_result?: Record<string, unknown>;
    status: 'pending' | 'running' | 'completed' | 'failed';
  }

  interface ActivityStats {
    total_events: number;
    events_last_hour: number;
    active_agents: number;
  }

  // =============================================================================
  // Props
  // =============================================================================

  export let agentId: string | null = null;
  export let maxEvents: number = 50;

  // =============================================================================
  // State
  // =============================================================================

  let events: ActivityEvent[] = [];
  let stats: ActivityStats | null = null;
  let ws: WebSocket | null = null;
  let connected = false;
  let loading = true;
  let error: string | null = null;

  // =============================================================================
  // Lifecycle
  // =============================================================================

  onMount(async () => {
    await loadInitialData();
    connectWebSocket();
  });

  onDestroy(() => { if (ws) ws.close(); });

  // =============================================================================
  // Data Loading
  // =============================================================================

  async function loadInitialData() {
    loading = true;
    error = null;
    try {
      const params = new URLSearchParams();
      if (agentId) params.set('agent_id', agentId);
      params.set('limit', String(maxEvents));

      const [eventsRes, statsRes] = await Promise.all([
        fetch(`/api/agents/activity?${params}`),
        fetch('/api/agents/activity/stats')
      ]);

      const eventsData = await eventsRes.json();
      if (eventsData.success) events = eventsData.data.events;

      const statsData = await statsRes.json();
      if (statsData.success) stats = statsData.data;
    } catch (e) {
      error = e instanceof Error ? e.message : 'Failed to load activity';
    } finally {
      loading = false;
    }
  }

  // =============================================================================
  // WebSocket
  // =============================================================================

  function connectWebSocket() {
    const wsUrl = `${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}/api/agents/activity/stream`;
    ws = new WebSocket(wsUrl);
    ws.onopen = () => { connected = true; };
    ws.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data);
        if (data.id) events = [data, ...events].slice(0, maxEvents);
        else if (Array.isArray(data)) events = [...data, ...events].slice(0, maxEvents);
      } catch {}
    };
    ws.onclose = () => { connected = false; setTimeout(connectWebSocket, 3000); };
  }

  // =============================================================================
  // Helpers
  // =============================================================================

  const icons: Record<string, string> = { action: '⚡', decision: '🎯', tool_call: '🔧', tool_result: '✅' };
  const colors: Record<string, string> = { action: '#3b82f6', decision: '#8b5cf6', tool_call: '#f59e0b', tool_result: '#10b981' };
  const statusColors: Record<string, string> = { pending: '#9ca3af', running: '#3b82f6', completed: '#10b981', failed: '#ef4444' };

  function fmtTime(ts: string): string { return new Date(ts).toLocaleTimeString(); }
  function truncate(str: string, len: number): string { return str.length > len ? str.slice(0, len) + '...' : str; }
</script>

<div class="feed-container">
  <header class="feed-header">
    <div class="header-left">
      <h2>Live Agent Activity</h2>
      <span class="status" class:connected>{connected ? 'Connected' : 'Disconnected'}</span>
    </div>
    <button class="btn" on:click={loadInitialData}>Refresh</button>
  </header>

  {#if stats}
    <div class="stats-bar">
      <div class="stat"><span class="val">{stats.total_events}</span><span class="lbl">Total</span></div>
      <div class="stat"><span class="val">{stats.events_last_hour}</span><span class="lbl">Last Hour</span></div>
      <div class="stat"><span class="val">{stats.active_agents}</span><span class="lbl">Active</span></div>
    </div>
  {/if}

  {#if loading && !events.length}
    <div class="msg">Loading...</div>
  {:else if error}
    <div class="msg error">{error}</div>
  {:else}
    <div class="events-list">
      {#each events as ev (ev.id)}
        <div class="event" style="--c: {colors[ev.event_type] || '#6b7280'}">
          <div class="icon">{icons[ev.event_type] || '📋'}</div>
          <div class="content">
            <div class="header">
              <span class="name">{ev.agent_name}</span>
              <span class="type">{ev.agent_type}</span>
              <span class="time">{fmtTime(ev.timestamp)}</span>
            </div>
            <div class="action-row">
              <span class="badge" style="background: {colors[ev.event_type]}">{ev.event_type}</span>
              <span class="text">{ev.action}</span>
              {#if ev.status !== 'completed'}
                <span class="stat" style="color: {statusColors[ev.status]}">{ev.status}</span>
              {/if}
            </div>
            {#if ev.reasoning}
              <div class="reasoning"><span class="lbl">Reasoning:</span> {truncate(ev.reasoning, 100)}</div>
            {/if}
            {#if ev.tool_name}
              <div class="tool"><span class="lbl">Tool:</span> <code>{ev.tool_name}</code></div>
            {/if}
            {#if ev.tool_result}
              <details class="result"><summary>Result</summary><pre>{JSON.stringify(ev.tool_result, null, 2)}</pre></details>
            {/if}
          </div>
        </div>
      {:else}
        <div class="msg">No activity</div>
      {/each}
    </div>
  {/if}
</div>

<style>
  .feed-container { display: flex; flex-direction: column; height: 100%; background: #1f2937; border-radius: 0.5rem; overflow: hidden; }
  .feed-header { display: flex; justify-content: space-between; align-items: center; padding: 0.75rem 1rem; background: #111827; border-bottom: 1px solid #374151; }
  .header-left { display: flex; align-items: center; gap: 0.75rem; }
  .feed-header h2 { margin: 0; font-size: 1rem; font-weight: 600; color: #f9fafb; }
  .status { font-size: 0.75rem; padding: 0.125rem 0.5rem; border-radius: 9999px; background: #374151; color: #9ca3af; }
  .status.connected { background: #065f46; color: #34d399; }
  .btn { padding: 0.375rem 0.75rem; background: #374151; border: none; border-radius: 0.25rem; color: #9ca3af; cursor: pointer; font-size: 0.75rem; }
  .btn:hover { background: #4b5563; color: #f9fafb; }
  .stats-bar { display: flex; gap: 1.5rem; padding: 0.75rem 1rem; background: #111827; border-bottom: 1px solid #374151; }
  .stat { display: flex; flex-direction: column; align-items: center; }
  .val { font-size: 1.25rem; font-weight: 700; color: #f9fafb; }
  .lbl { font-size: 0.625rem; text-transform: uppercase; color: #9ca3af; }
  .events-list { flex: 1; overflow-y: auto; padding: 0.5rem; }
  .event { display: flex; gap: 0.75rem; padding: 0.75rem; margin-bottom: 0.5rem; background: #111827; border-radius: 0.375rem; border-left: 3px solid var(--c); }
  .icon { font-size: 1.25rem; flex-shrink: 0; }
  .content { flex: 1; min-width: 0; }
  .header { display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.375rem; flex-wrap: wrap; }
  .name { font-weight: 600; color: #f9fafb; font-size: 0.875rem; }
  .type { font-size: 0.75rem; padding: 0.125rem 0.375rem; background: #374151; border-radius: 0.25rem; color: #9ca3af; }
  .time { font-size: 0.75rem; color: #6b7280; margin-left: auto; }
  .action-row { display: flex; align-items: center; gap: 0.5rem; flex-wrap: wrap; }
  .badge { font-size: 0.625rem; padding: 0.125rem 0.375rem; border-radius: 0.25rem; color: white; text-transform: uppercase; font-weight: 600; }
  .text { font-size: 0.875rem; color: #e5e7eb; }
  .stat { font-size: 0.75rem; font-weight: 500; }
  .reasoning, .tool { margin-top: 0.5rem; font-size: 0.75rem; }
  .lbl { color: #9ca3af; }
  .reasoning { color: #d1d5db; font-style: italic; }
  code { font-family: monospace; background: #374151; padding: 0.125rem 0.25rem; border-radius: 0.125rem; color: #f9fafb; }
  .result { margin-top: 0.5rem; }
  .result summary { font-size: 0.75rem; color: #9ca3af; cursor: pointer; }
  .result pre { margin-top: 0.5rem; padding: 0.5rem; background: #1f2937; border-radius: 0.25rem; font-size: 0.75rem; color: #d1d5db; overflow-x: auto; max-height: 150px; }
  .msg { padding: 2rem; text-align: center; color: #9ca3af; }
  .error { color: #ef4444; }
</style>
