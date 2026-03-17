<script lang="ts">
  import { onMount } from 'svelte';
  import { page } from '$app/state';
  
  // Get agent_id from URL
  let agentId = $derived($page.params.id);
  
  // Types
  interface MetricsData {
    total_invocations: number;
    successful_invocations: number;
    failed_invocations: number;
    average_duration: number;
    total_tool_calls: number;
    tool_call_counts: Record<string, number>;
    invocation_timeline: Array<{
      timestamp: string;
      duration: number;
      success: boolean;
    }>;
  }
  
  // State
  let metrics: MetricsData | null = $state(null);
  let loading = $state(true);
  let error: string | null = $state(null);
  
  // Load metrics
  async function loadMetrics() {
    loading = true;
    error = null;
    
    try {
      const response = await fetch(`/api/agents/${agentId}/metrics`);
      const data = await response.json();
      
      if (data.success) {
        metrics = data.data;
      } else {
        error = 'Failed to load metrics';
      }
    } catch (e) {
      error = e instanceof Error ? e.message : 'Failed to load metrics';
    } finally {
      loading = false;
    }
  }
  
  onMount(() => {
    loadMetrics();
    
    // Poll for updates
    const interval = setInterval(loadMetrics, 15000);
    return () => clearInterval(interval);
  });
  
  function formatDuration(ms: number): string {
    if (ms < 1000) return `${ms.toFixed(0)}ms`;
    return `${(ms / 1000).toFixed(2)}s`;
  }
  
  function getSuccessRate(): string {
    if (!metrics || metrics.total_invocations === 0) return '0%';
    return ((metrics.successful_invocations / metrics.total_invocations) * 100).toFixed(1) + '%';
  }
</script>

<div class="metrics-page">
  <header class="page-header">
    <div class="header-content">
      <a href="/agents" class="back-link">← Back to Agents</a>
      <h1>📊 Agent Metrics</h1>
      <p class="agent-id">Agent: <code>{agentId}</code></p>
    </div>
    <button class="refresh-btn" onclick={loadMetrics}>Refresh</button>
  </header>
  
  {#if loading && !metrics}
    <div class="loading">Loading metrics...</div>
  {:else if error}
    <div class="error">{error}</div>
  {:else if metrics}
    <!-- Overview Stats -->
    <div class="stats-grid">
      <div class="stat-card">
        <div class="stat-value">{metrics.total_invocations}</div>
        <div class="stat-label">Total Invocations</div>
      </div>
      
      <div class="stat-card success">
        <div class="stat-value">{metrics.successful_invocations}</div>
        <div class="stat-label">Successful</div>
      </div>
      
      <div class="stat-card error">
        <div class="stat-value">{metrics.failed_invocations}</div>
        <div class="stat-label">Failed</div>
      </div>
      
      <div class="stat-card">
        <div class="stat-value">{getSuccessRate()}</div>
        <div class="stat-label">Success Rate</div>
      </div>
      
      <div class="stat-card">
        <div class="stat-value">{formatDuration(metrics.average_duration)}</div>
        <div class="stat-label">Avg Duration</div>
      </div>
      
      <div class="stat-card">
        <div class="stat-value">{metrics.total_tool_calls}</div>
        <div class="stat-label">Tool Calls</div>
      </div>
    </div>
    
    <!-- Tool Calls Breakdown -->
    {#if metrics.tool_call_counts && Object.keys(metrics.tool_call_counts).length > 0}
      <div class="section">
        <h2>Tool Calls Breakdown</h2>
        <div class="tool-grid">
          {#each Object.entries(metrics.tool_call_counts) as [tool, count]}
            <div class="tool-card">
              <span class="tool-name">{tool}</span>
              <span class="tool-count">{count}</span>
            </div>
          {/each}
        </div>
      </div>
    {/if}
    
    <!-- Invocation Timeline -->
    {#if metrics.invocation_timeline && metrics.invocation_timeline.length > 0}
      <div class="section">
        <h2>Recent Invocations</h2>
        <div class="timeline">
          {#each metrics.invocation_timeline as invocation}
            <div class="timeline-item" class:success={invocation.success} class:error={!invocation.success}>
              <div class="timeline-dot"></div>
              <div class="timeline-content">
                <span class="timeline-time">{new Date(invocation.timestamp).toLocaleString()}</span>
                <span class="timeline-duration">{formatDuration(invocation.duration)}</span>
                <span class="timeline-status">{invocation.success ? 'Success' : 'Failed'}</span>
              </div>
            </div>
          {/each}
        </div>
      </div>
    {/if}
    
    {#if !metrics.invocation_timeline || metrics.invocation_timeline.length === 0}
      <div class="empty-state">
        <p>No invocation data available yet.</p>
        <p>Invoke the agent to see metrics.</p>
      </div>
    {/if}
  {/if}
</div>

<style>
  .metrics-page {
    padding: 1.5rem;
    max-width: 1200px;
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
  
  .stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 1rem;
    margin-bottom: 2rem;
  }
  
  .stat-card {
    background: #1f2937;
    border-radius: 0.5rem;
    padding: 1.25rem;
    text-align: center;
  }
  
  .stat-card.success .stat-value {
    color: #22c55e;
  }
  
  .stat-card.error .stat-value {
    color: #ef4444;
  }
  
  .stat-value {
    font-size: 1.75rem;
    font-weight: 700;
    color: #f9fafb;
  }
  
  .stat-label {
    font-size: 0.75rem;
    color: #9ca3af;
    margin-top: 0.25rem;
    text-transform: uppercase;
  }
  
  .section {
    margin-bottom: 2rem;
  }
  
  h2 {
    font-size: 1.125rem;
    margin-bottom: 1rem;
    color: #f9fafb;
  }
  
  .tool-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 0.75rem;
  }
  
  .tool-card {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    background: #1f2937;
    padding: 0.75rem 1rem;
    border-radius: 0.375rem;
  }
  
  .tool-name {
    color: #f9fafb;
    font-family: monospace;
  }
  
  .tool-count {
    background: #374151;
    padding: 0.125rem 0.5rem;
    border-radius: 9999px;
    font-size: 0.75rem;
    color: #9ca3af;
  }
  
  .timeline {
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
  }
  
  .timeline-item {
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 0.75rem;
    background: #1f2937;
    border-radius: 0.375rem;
  }
  
  .timeline-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #6b7280;
  }
  
  .timeline-item.success .timeline-dot {
    background: #22c55e;
  }
  
  .timeline-item.error .timeline-dot {
    background: #ef4444;
  }
  
  .timeline-content {
    display: flex;
    gap: 1rem;
    flex: 1;
    font-size: 0.875rem;
  }
  
  .timeline-time {
    color: #9ca3af;
  }
  
  .timeline-duration {
    color: #f9fafb;
    font-family: monospace;
  }
  
  .timeline-status {
    margin-left: auto;
  }
  
  .timeline-item.success .timeline-status {
    color: #22c55e;
  }
  
  .timeline-item.error .timeline-status {
    color: #ef4444;
  }
</style>
