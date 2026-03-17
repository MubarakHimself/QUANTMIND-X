<script lang="ts">
  import { onMount } from 'svelte';
  import { page } from '$app/state';
  
  // Get agent_id from URL
  let agentId = $derived($page.params.id);
  
  // Types
  interface LifecycleEvent {
    id: string;
    event_type: string;
    timestamp: string;
    details: {
      duration?: number;
      error?: string;
      success?: boolean;
      [key: string]: unknown;
    };
  }
  
  interface AgentInfo {
    agent_id: string;
    agent_type: string;
    name: string;
    registered_at: string;
    last_invoked: string | null;
  }
  
  // State
  let agent: AgentInfo | null = $state(null);
  let events: LifecycleEvent[] = $state([]);
  let loading = $state(true);
  let error: string | null = $state(null);
  
  // Load agent and lifecycle events
  async function loadData() {
    loading = true;
    error = null;
    
    try {
      // Load agent info
      const agentRes = await fetch(`/api/agents/${agentId}`);
      const agentData = await agentRes.json();
      
      if (agentData.success) {
        agent = {
          agent_id: agentData.data.agent_id,
          agent_type: agentData.data.agent_type,
          name: agentData.data.name,
          registered_at: agentData.data.config?.created_at || new Date().toISOString(),
          last_invoked: null
        };
      }
      
      // Load lifecycle events (from a hypothetical endpoint or use metrics)
      const metricsRes = await fetch(`/api/agents/${agentId}/metrics`);
      const metricsData = await metricsRes.json();
      
      if (metricsData.success && metricsData.data.invocation_timeline) {
        // Transform metrics to lifecycle events
        events = metricsData.data.invocation_timeline.map((inv: { timestamp: string; duration: number; success: boolean }, index: number) => ({
          id: `inv-${index}`,
          event_type: inv.success ? 'invocation_complete' : 'invocation_error',
          timestamp: inv.timestamp,
          details: { duration: inv.duration }
        }));
      }
      
    } catch (e) {
      error = e instanceof Error ? e.message : 'Failed to load timeline';
    } finally {
      loading = false;
    }
  }
  
  onMount(() => {
    loadData();
    
    // Poll for updates
    const interval = setInterval(loadData, 10000);
    return () => clearInterval(interval);
  });
  
  function getEventIcon(eventType: string): string {
    switch (eventType) {
      case 'agent_created':
        return '✨';
      case 'invocation_complete':
        return '✅';
      case 'invocation_error':
        return '❌';
      case 'config_updated':
        return '🔧';
      case 'health_check':
        return '💚';
      default:
        return '📌';
    }
  }
  
  function getEventLabel(eventType: string): string {
    switch (eventType) {
      case 'agent_created':
        return 'Agent Created';
      case 'invocation_complete':
        return 'Invocation Complete';
      case 'invocation_error':
        return 'Invocation Failed';
      case 'config_updated':
        return 'Config Updated';
      case 'health_check':
        return 'Health Check';
      default:
        return eventType;
    }
  }
  
  function formatDuration(ms: number): string {
    if (ms < 1000) return `${ms.toFixed(0)}ms`;
    return `${(ms / 1000).toFixed(2)}s`;
  }
</script>

<div class="timeline-page">
  <header class="page-header">
    <div class="header-content">
      <a href="/agents" class="back-link">← Back to Agents</a>
      <h1>📜 Agent Timeline</h1>
      {#if agent}
        <p class="agent-info">
          <span class="agent-name">{agent.name}</span>
          <span class="agent-id">({agent.agent_id})</span>
        </p>
      {/if}
    </div>
    <button class="refresh-btn" onclick={loadData}>Refresh</button>
  </header>
  
  {#if loading && !agent}
    <div class="loading">Loading timeline...</div>
  {:else if error}
    <div class="error">{error}</div>
  {:else}
    <div class="timeline-container">
      <!-- Agent Created Event -->
      {#if agent}
        <div class="timeline-item created">
          <div class="timeline-marker">
            <span class="marker-icon">✨</span>
            <div class="marker-line"></div>
          </div>
          <div class="timeline-content">
            <div class="event-header">
              <span class="event-type">Agent Created</span>
              <span class="event-time">{new Date(agent.registered_at).toLocaleString()}</span>
            </div>
            <div class="event-details">
              <p>Agent <strong>{agent.name}</strong> was registered in the system.</p>
              <p class="detail-meta">Type: {agent.agent_type}</p>
            </div>
          </div>
        </div>
      {/if}
      
      <!-- Lifecycle Events -->
      {#each events as event}
        <div class="timeline-item" class:success={event.event_type === 'invocation_complete'} class:error={event.event_type === 'invocation_error'}>
          <div class="timeline-marker">
            <span class="marker-icon">{getEventIcon(event.event_type)}</span>
            <div class="marker-line"></div>
          </div>
          <div class="timeline-content">
            <div class="event-header">
              <span class="event-type">{getEventLabel(event.event_type)}</span>
              <span class="event-time">{new Date(event.timestamp).toLocaleString()}</span>
            </div>
            <div class="event-details">
              {#if event.details.duration}
                <p>Duration: <code>{formatDuration(event.details.duration)}</code></p>
              {/if}
              {#if event.details.error}
                <p class="error-detail">Error: {event.details.error}</p>
              {/if}
            </div>
          </div>
        </div>
      {/each}
      
      {#if events.length === 0}
        <div class="empty-state">
          <p>No lifecycle events recorded yet.</p>
          <p>Invoke the agent to see timeline events.</p>
        </div>
      {/if}
    </div>
  {/if}
</div>

<style>
  .timeline-page {
    padding: 1.5rem;
    max-width: 900px;
    margin: 0 auto;
  }
  
  .page-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 2rem;
  }
  
  .header-content {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }
  
  .back-link {
    color: #3b82f6;
    text-decoration: none;
    font-size: 0.875rem;
  }
  
  .back-link:hover {
    text-decoration: underline;
  }
  
  h1 {
    font-size: 1.5rem;
    font-weight: 600;
    margin: 0;
  }
  
  .agent-info {
    margin: 0;
    font-size: 0.875rem;
    color: #9ca3af;
  }
  
  .agent-name {
    color: #f9fafb;
    font-weight: 500;
  }
  
  .agent-id {
    font-family: monospace;
  }
  
  .refresh-btn {
    padding: 0.5rem 1rem;
    background: #3b82f6;
    color: white;
    border: none;
    border-radius: 0.375rem;
    cursor: pointer;
  }
  
  .loading, .error, .empty-state {
    text-align: center;
    padding: 3rem;
    color: #6b7280;
  }
  
  .error {
    color: #ef4444;
  }
  
  .timeline-container {
    position: relative;
    padding-left: 2rem;
  }
  
  .timeline-item {
    position: relative;
    padding-bottom: 2rem;
  }
  
  .timeline-marker {
    position: absolute;
    left: -2rem;
    display: flex;
    flex-direction: column;
    align-items: center;
  }
  
  .marker-icon {
    width: 2rem;
    height: 2rem;
    display: flex;
    align-items: center;
    justify-content: center;
    background: #1f2937;
    border-radius: 50%;
    font-size: 1rem;
    z-index: 1;
  }
  
  .marker-line {
    width: 2px;
    flex: 1;
    background: #374151;
    margin-top: 0.25rem;
  }
  
  .timeline-item:last-child .marker-line {
    display: none;
  }
  
  .timeline-item.created .marker-icon {
    background: #3b82f6;
  }
  
  .timeline-item.success .marker-icon {
    background: #22c55e;
  }
  
  .timeline-item.error .marker-icon {
    background: #ef4444;
  }
  
  .timeline-content {
    background: #1f2937;
    border-radius: 0.5rem;
    padding: 1rem;
    margin-left: 0.5rem;
  }
  
  .event-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.5rem;
  }
  
  .event-type {
    font-weight: 600;
    color: #f9fafb;
  }
  
  .event-time {
    font-size: 0.75rem;
    color: #9ca3af;
  }
  
  .event-details {
    font-size: 0.875rem;
    color: #d1d5db;
  }
  
  .event-details p {
    margin: 0.25rem 0;
  }
  
  .event-details code {
    background: #374151;
    padding: 0.125rem 0.375rem;
    border-radius: 0.25rem;
    font-family: monospace;
  }
  
  .detail-meta {
    color: #9ca3af;
    font-size: 0.75rem;
  }
  
  .error-detail {
    color: #ef4444;
  }
</style>
