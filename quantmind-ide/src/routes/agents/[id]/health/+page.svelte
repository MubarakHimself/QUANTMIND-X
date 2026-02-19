<script lang="ts">
  import { onMount } from 'svelte';
  import { page } from '$app/stores';
  
  // Get agent_id from URL
  $: agentId = $page.params.id;
  
  // Types
  interface HealthStatus {
    status: 'healthy' | 'degraded' | 'unhealthy' | 'not_found';
    checks: Record<string, {
      status: string;
      message?: string;
      timestamp?: string;
    }>;
    last_check?: string;
    uptime_seconds?: number;
  }
  
  // State
  let health: HealthStatus | null = null;
  let loading = true;
  let error: string | null = null;
  let autoRefresh = true;
  
  // Load health status
  async function loadHealth() {
    loading = true;
    error = null;
    
    try {
      const response = await fetch(`/api/agents/${agentId}/health`);
      const data = await response.json();
      
      if (data.success) {
        health = data.data;
      } else {
        error = data.detail || 'Failed to load health status';
      }
    } catch (e) {
      error = e instanceof Error ? e.message : 'Failed to load health status';
    } finally {
      loading = false;
    }
  }
  
  onMount(() => {
    loadHealth();
    
    let interval: ReturnType<typeof setInterval>;
    if (autoRefresh) {
      interval = setInterval(loadHealth, 10000);
    }
    
    return () => {
      if (interval) clearInterval(interval);
    };
  });
  
  function getStatusColor(status: string): string {
    switch (status) {
      case 'healthy':
        return '#22c55e';
      case 'degraded':
        return '#eab308';
      case 'unhealthy':
        return '#ef4444';
      case 'not_found':
        return '#6b7280';
      default:
        return '#6b7280';
    }
  }
  
  function getStatusLabel(status: string): string {
    switch (status) {
      case 'healthy':
        return 'Healthy';
      case 'degraded':
        return 'Degraded';
      case 'unhealthy':
        return 'Unhealthy';
      case 'not_found':
        return 'Not Found';
      default:
        return status;
    }
  }
  
  function formatUptime(seconds: number): string {
    const days = Math.floor(seconds / 86400);
    const hours = Math.floor((seconds % 86400) / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    
    if (days > 0) return `${days}d ${hours}h ${minutes}m`;
    if (hours > 0) return `${hours}h ${minutes}m`;
    return `${minutes}m`;
  }
</script>

<div class="health-page">
  <header class="page-header">
    <div class="header-content">
      <a href="/agents" class="back-link">← Back to Agents</a>
      <h1>💚 Agent Health Monitor</h1>
      <p class="agent-id">Agent: <code>{agentId}</code></p>
    </div>
    <div class="header-actions">
      <label class="auto-refresh">
        <input type="checkbox" bind:checked={autoRefresh} on:change={loadHealth} />
        Auto-refresh
      </label>
      <button class="refresh-btn" on:click={loadHealth}>Refresh</button>
    </div>
  </header>
  
  {#if loading && !health}
    <div class="loading">Loading health status...</div>
  {:else if error}
    <div class="error">{error}</div>
  {:else if health}
    <!-- Overall Status -->
    <div class="status-card" style="border-color: {getStatusColor(health.status)}">
      <div class="status-indicator" style="background: {getStatusColor(health.status)}"></div>
      <div class="status-info">
        <h2>Overall Status: {getStatusLabel(health.status)}</h2>
        {#if health.last_check}
          <p class="last-check">Last checked: {new Date(health.last_check).toLocaleString()}</p>
        {/if}
        {#if health.uptime_seconds !== undefined}
          <p class="uptime">Uptime: {formatUptime(health.uptime_seconds)}</p>
        {/if}
      </div>
    </div>
    
    <!-- Health Checks -->
    {#if health.checks && Object.keys(health.checks).length > 0}
      <div class="section">
        <h2>Health Checks</h2>
        <div class="checks-grid">
          {#each Object.entries(health.checks) as [checkName, checkResult]}
            <div class="check-card" class:healthy={checkResult.status === 'healthy'} class:degraded={checkResult.status === 'degraded'} class:unhealthy={checkResult.status === 'unhealthy'}>
              <div class="check-header">
                <span class="check-name">{checkName}</span>
                <span class="check-status" style="color: {getStatusColor(checkResult.status)}">
                  {getStatusLabel(checkResult.status)}
                </span>
              </div>
              {#if checkResult.message}
                <p class="check-message">{checkResult.message}</p>
              {/if}
              {#if checkResult.timestamp}
                <p class="check-timestamp">{new Date(checkResult.timestamp).toLocaleString()}</p>
              {/if}
            </div>
          {/each}
        </div>
      </div>
    {:else}
      <div class="empty-state">
        <p>No health checks available.</p>
      </div>
    {/if}
    
    {#if health.status === 'not_found'}
      <div class="warning-card">
        <p>⚠️ Agent not found in the registry.</p>
        <p>The agent may have been deleted or not registered yet.</p>
      </div>
    {/if}
  {/if}
</div>

<style>
  .health-page {
    padding: 1.5rem;
    max-width: 1000px;
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
  
  .agent-id {
    color: #9ca3af;
    font-size: 0.875rem;
    margin: 0;
  }
  
  .agent-id code {
    background: #374151;
    padding: 0.125rem 0.5rem;
    border-radius: 0.25rem;
  }
  
  .header-actions {
    display: flex;
    align-items: center;
    gap: 1rem;
  }
  
  .auto-refresh {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.875rem;
    color: #9ca3af;
    cursor: pointer;
  }
  
  .auto-refresh input {
    cursor: pointer;
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
  
  .status-card {
    display: flex;
    align-items: center;
    gap: 1.5rem;
    background: #1f2937;
    border-radius: 0.5rem;
    padding: 1.5rem;
    margin-bottom: 2rem;
    border-left: 4px solid;
  }
  
  .status-indicator {
    width: 1rem;
    height: 1rem;
    border-radius: 50%;
    flex-shrink: 0;
  }
  
  .status-info h2 {
    font-size: 1.25rem;
    font-weight: 600;
    margin: 0 0 0.5rem 0;
    color: #f9fafb;
  }
  
  .status-info p {
    margin: 0;
    font-size: 0.875rem;
    color: #9ca3af;
  }
  
  .section {
    margin-bottom: 2rem;
  }
  
  h2 {
    font-size: 1rem;
    font-weight: 600;
    margin: 0 0 1rem 0;
    color: #f9fafb;
  }
  
  .checks-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 1rem;
  }
  
  .check-card {
    background: #1f2937;
    border-radius: 0.5rem;
    padding: 1rem;
    border-left: 3px solid #6b7280;
  }
  
  .check-card.healthy {
    border-left-color: #22c55e;
  }
  
  .check-card.degraded {
    border-left-color: #eab308;
  }
  
  .check-card.unhealthy {
    border-left-color: #ef4444;
  }
  
  .check-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.5rem;
  }
  
  .check-name {
    font-weight: 600;
    color: #f9fafb;
    font-size: 0.875rem;
  }
  
  .check-status {
    font-size: 0.75rem;
    font-weight: 500;
  }
  
  .check-message {
    font-size: 0.875rem;
    color: #d1d5db;
    margin: 0.5rem 0;
  }
  
  .check-timestamp {
    font-size: 0.75rem;
    color: #6b7280;
    margin: 0;
  }
  
  .warning-card {
    background: rgba(234, 179, 8, 0.1);
    border: 1px solid #eab308;
    border-radius: 0.5rem;
    padding: 1.5rem;
    text-align: center;
  }
  
  .warning-card p {
    margin: 0.25rem 0;
    color: #eab308;
  }
</style>
