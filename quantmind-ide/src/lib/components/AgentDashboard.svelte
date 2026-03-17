<script lang="ts">
  import { onMount } from 'svelte';
  
  // Types
  interface AgentInfo {
    agent_id: string;
    agent_type: string;
    name: string;
    registered_at: string;
    last_invoked: string | null;
    invocation_count: number;
  }
  
  interface AgentSummary {
    total_agents: number;
    total_invocations: number;
    agent_types: Record<string, number>;
    agents: AgentInfo[];
  }
  
  interface HealthStatus {
    overall_status: string;
    total_agents: number;
    healthy_agents: number;
    degraded_agents: number;
    unhealthy_agents: number;
  }
  
  // State
  let summary: AgentSummary | null = $state(null);
  let health: HealthStatus | null = $state(null);
  let loading = $state(true);
  let error: string | null = $state(null);
  
  // Load data
  async function loadData() {
    loading = true;
    error = null;
    
    try {
      const [summaryRes, healthRes] = await Promise.all([
        fetch('/api/agents'),
        fetch('/api/agents/health')
      ]);
      
      const summaryData = await summaryRes.json();
      const healthData = await healthRes.json();
      
      if (summaryData.success) {
        summary = summaryData.data;
      }
      
      if (healthData.success) {
        health = {
          overall_status: healthData.data.overall_status,
          total_agents: healthData.data.total_agents,
          healthy_agents: healthData.data.healthy_agents,
          degraded_agents: healthData.data.degraded_agents,
          unhealthy_agents: healthData.data.unhealthy_agents
        };
      }
    } catch (e) {
      error = e instanceof Error ? e.message : 'Failed to load data';
    } finally {
      loading = false;
    }
  }
  
  onMount(() => {
    loadData();
    
    // Poll for updates
    const interval = setInterval(loadData, 30000);
    return () => clearInterval(interval);
  });
  
  function getStatusColor(status: string): string {
    switch (status) {
      case 'healthy': return '#22c55e';
      case 'degraded': return '#eab308';
      case 'unhealthy': return '#ef4444';
      default: return '#6b7280';
    }
  }
  
  function getAgentTypeIcon(type: string): string {
    switch (type) {
      case 'analyst': return '📊';
      case 'quantcode': return '💻';
      case 'copilot': return '🚀';
      case 'router': return '🔀';
      default: return '🤖';
    }
  }
</script>

<div class="agent-dashboard">
  <header class="dashboard-header">
    <h1>🤖 Agent Management</h1>
    <button class="refresh-btn" onclick={loadData}>
      Refresh
    </button>
  </header>
  
  {#if loading && !summary}
    <div class="loading">Loading agents...</div>
  {:else if error}
    <div class="error">{error}</div>
  {:else}
    <!-- Stats Grid -->
    <div class="stats-grid">
      <div class="stat-card">
        <div class="stat-value">{summary?.total_agents ?? 0}</div>
        <div class="stat-label">Total Agents</div>
      </div>
      
      <div class="stat-card">
        <div class="stat-value">{summary?.total_invocations ?? 0}</div>
        <div class="stat-label">Total Invocations</div>
      </div>
      
      <div class="stat-card">
        <div class="stat-value" style="color: {getStatusColor(health?.overall_status ?? 'unknown')}">
          {health?.healthy_agents ?? 0}
        </div>
        <div class="stat-label">Healthy</div>
      </div>
      
      <div class="stat-card">
        <div class="stat-value" style="color: {getStatusColor('unhealthy')}">
          {health?.unhealthy_agents ?? 0}
        </div>
        <div class="stat-label">Unhealthy</div>
      </div>
    </div>
    
    <!-- Agent Types -->
    <div class="agent-types">
      <h2>Agent Types</h2>
      <div class="type-grid">
        {#if summary?.agent_types}
          {#each Object.entries(summary.agent_types) as [type, count]}
            <div class="type-card">
              <span class="type-icon">{getAgentTypeIcon(type)}</span>
              <span class="type-name">{type}</span>
              <span class="type-count">{count}</span>
            </div>
          {/each}
        {/if}
      </div>
    </div>
    
    <!-- Agents List -->
    <div class="agents-section">
      <h2>Registered Agents</h2>
      <div class="agents-grid">
        {#if summary?.agents}
          {#each summary.agents as agent}
            <div class="agent-card">
              <div class="agent-header">
                <span class="agent-icon">{getAgentTypeIcon(agent.agent_type)}</span>
                <span class="agent-name">{agent.name}</span>
              </div>
              <div class="agent-details">
                <div class="detail-row">
                  <span class="detail-label">ID:</span>
                  <span class="detail-value">{agent.agent_id}</span>
                </div>
                <div class="detail-row">
                  <span class="detail-label">Type:</span>
                  <span class="detail-value">{agent.agent_type}</span>
                </div>
                <div class="detail-row">
                  <span class="detail-label">Invocations:</span>
                  <span class="detail-value">{agent.invocation_count}</span>
                </div>
                <div class="detail-row">
                  <span class="detail-label">Registered:</span>
                  <span class="detail-value">{new Date(agent.registered_at).toLocaleDateString()}</span>
                </div>
              </div>
              <div class="agent-actions">
                <a href="/agents/{agent.agent_id}/metrics" class="action-btn">Metrics</a>
                <a href="/agents/{agent.agent_id}/config" class="action-btn">Config</a>
                <a href="/agents/{agent.agent_id}/health" class="action-btn">Health</a>
                <a href="/agents/{agent.agent_id}/timeline" class="action-btn">Timeline</a>
              </div>
            </div>
          {/each}
        {:else}
          <p class="no-agents">No agents registered</p>
        {/if}
      </div>
    </div>
  {/if}
</div>

<style>
  .agent-dashboard {
    padding: 1.5rem;
    max-width: 1400px;
    margin: 0 auto;
  }
  
  .dashboard-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 2rem;
  }
  
  .dashboard-header h1 {
    font-size: 1.75rem;
    font-weight: 600;
    margin: 0;
  }
  
  .refresh-btn {
    padding: 0.5rem 1rem;
    background: #3b82f6;
    color: white;
    border: none;
    border-radius: 0.375rem;
    cursor: pointer;
    font-size: 0.875rem;
  }
  
  .refresh-btn:hover {
    background: #2563eb;
  }
  
  .loading, .error {
    text-align: center;
    padding: 2rem;
    color: #6b7280;
  }
  
  .error {
    color: #ef4444;
  }
  
  .stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
    margin-bottom: 2rem;
  }
  
  .stat-card {
    background: #1f2937;
    border-radius: 0.5rem;
    padding: 1.25rem;
    text-align: center;
  }
  
  .stat-value {
    font-size: 2rem;
    font-weight: 700;
    color: #f9fafb;
  }
  
  .stat-label {
    font-size: 0.875rem;
    color: #9ca3af;
    margin-top: 0.25rem;
  }
  
  .agent-types {
    margin-bottom: 2rem;
  }
  
  .agent-types h2 {
    font-size: 1.25rem;
    margin-bottom: 1rem;
  }
  
  .type-grid {
    display: flex;
    gap: 1rem;
    flex-wrap: wrap;
  }
  
  .type-card {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    background: #374151;
    padding: 0.75rem 1rem;
    border-radius: 0.375rem;
  }
  
  .type-icon {
    font-size: 1.25rem;
  }
  
  .type-name {
    text-transform: capitalize;
    font-weight: 500;
  }
  
  .type-count {
    background: #4b5563;
    padding: 0.125rem 0.5rem;
    border-radius: 9999px;
    font-size: 0.75rem;
  }
  
  .agents-section h2 {
    font-size: 1.25rem;
    margin-bottom: 1rem;
  }
  
  .agents-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
    gap: 1rem;
  }
  
  .agent-card {
    background: #1f2937;
    border-radius: 0.5rem;
    padding: 1.25rem;
  }
  
  .agent-header {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 1rem;
    padding-bottom: 0.75rem;
    border-bottom: 1px solid #374151;
  }
  
  .agent-icon {
    font-size: 1.5rem;
  }
  
  .agent-name {
    font-size: 1.125rem;
    font-weight: 600;
  }
  
  .agent-details {
    margin-bottom: 1rem;
  }
  
  .detail-row {
    display: flex;
    justify-content: space-between;
    padding: 0.25rem 0;
    font-size: 0.875rem;
  }
  
  .detail-label {
    color: #9ca3af;
  }
  
  .detail-value {
    color: #f9fafb;
    font-family: monospace;
  }
  
  .agent-actions {
    display: flex;
    gap: 0.5rem;
  }
  
  .action-btn {
    flex: 1;
    padding: 0.5rem;
    background: #374151;
    color: #f9fafb;
    text-align: center;
    border-radius: 0.375rem;
    text-decoration: none;
    font-size: 0.875rem;
    transition: background 0.2s;
  }
  
  .action-btn:hover {
    background: #4b5563;
  }
  
  .no-agents {
    grid-column: 1 / -1;
    text-align: center;
    color: #6b7280;
    padding: 2rem;
  }
</style>
