<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { fade, slide } from 'svelte/transition';

  // Types
  interface MCPServerStatus {
    server_id: string;
    name: string;
    type: string;
    description: string;
    url: string;
    icon: string;
    status: 'online' | 'offline' | 'error';
    healthy: boolean;
    latency_ms: number | null;
    last_check: string | null;
    error: string | null;
    tools: string[];
    tools_available: number;
  }

  interface MCPStatusResponse {
    healthy: boolean;
    total_servers: number;
    online_servers: number;
    offline_servers: number;
    servers: MCPServerStatus[];
    last_updated: string;
  }

  // State
  let status: MCPStatusResponse | null = null;
  let loading = true;
  let error: string | null = null;
  let autoRefresh = true;
  let refreshInterval: number | null = null;

  // Server icons mapping
  const serverIcons: Record<string, string> = {
    context7: '📚',
    sequential_thinking: '🧠',
    pageindex: '📄',
    backtest: '📈',
    mt5_compiler: '⚙️'
  };

  // Status colors
  const statusColors: Record<string, string> = {
    online: 'bg-green-500',
    offline: 'bg-gray-400',
    error: 'bg-red-500'
  };

  // Fetch status from API
  async function fetchStatus() {
    try {
      const response = await fetch('/api/mcp/status');
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      status = await response.json();
      error = null;
    } catch (e) {
      error = e instanceof Error ? e.message : 'Failed to fetch MCP status';
      console.error('Failed to fetch MCP status:', e);
    } finally {
      loading = false;
    }
  }

  // Retry connection to a server
  async function retryConnection(serverId: string) {
    try {
      const response = await fetch(`/api/mcp/status/retry/${serverId}`, {
        method: 'POST'
      });
      if (response.ok) {
        // Refresh status after retry
        await fetchStatus();
      }
    } catch (e) {
      console.error('Failed to retry connection:', e);
    }
  }

  // Format latency
  function formatLatency(ms: number | null): string {
    if (ms === null) return '-';
    return `${ms}ms`;
  }

  // Format last check time
  function formatLastCheck(isoString: string | null): string {
    if (!isoString) return 'Never';
    const date = new Date(isoString);
    return date.toLocaleTimeString();
  }

  // Lifecycle
  onMount(() => {
    fetchStatus();
    
    // Set up auto-refresh
    if (autoRefresh) {
      refreshInterval = window.setInterval(fetchStatus, 60000); // Refresh every minute
    }
  });

  onDestroy(() => {
    if (refreshInterval) {
      clearInterval(refreshInterval);
    }
  });
</script>

<div class="mcp-status-panel">
  <!-- Header -->
  <div class="panel-header">
    <h3 class="text-lg font-semibold">MCP Servers</h3>
    <div class="header-actions">
      <button 
        class="refresh-btn" 
        on:click={fetchStatus}
        disabled={loading}
        title="Refresh status"
      >
        <span class="icon" class:spinning={loading}>🔄</span>
      </button>
      <label class="auto-refresh-toggle">
        <input type="checkbox" bind:checked={autoRefresh} />
        <span class="toggle-label">Auto-refresh</span>
      </label>
    </div>
  </div>

  <!-- Loading State -->
  {#if loading && !status}
    <div class="loading-state" in:fade>
      <div class="spinner"></div>
      <p>Loading MCP status...</p>
    </div>
  {/if}

  <!-- Error State -->
  {#if error}
    <div class="error-state" in:fade>
      <span class="error-icon">⚠️</span>
      <p>{error}</p>
      <button class="retry-btn" on:click={fetchStatus}>Retry</button>
    </div>
  {/if}

  <!-- Status Overview -->
  {#if status}
    <div class="status-overview" in:slide>
      <div class="overview-item">
        <span class="overview-label">Total</span>
        <span class="overview-value">{status.total_servers}</span>
      </div>
      <div class="overview-item online">
        <span class="overview-label">Online</span>
        <span class="overview-value">{status.online_servers}</span>
      </div>
      <div class="overview-item offline">
        <span class="overview-label">Offline</span>
        <span class="overview-value">{status.offline_servers}</span>
      </div>
      <div class="health-indicator" class:healthy={status.healthy}>
        {#if status.healthy}
          <span class="health-icon">✓</span>
          <span>All systems operational</span>
        {:else}
          <span class="health-icon">!</span>
          <span>Some systems offline</span>
        {/if}
      </div>
    </div>

    <!-- Server List -->
    <div class="server-list">
      {#each status.servers as server (server.server_id)}
        <div class="server-card" class:offline={server.status !== 'online'}>
          <div class="server-header">
            <div class="server-icon">
              {serverIcons[server.type] || '🔌'}
            </div>
            <div class="server-info">
              <h4 class="server-name">{server.name}</h4>
              <p class="server-description">{server.description}</p>
            </div>
            <div class="server-status">
              <span 
                class="status-dot {statusColors[server.status]}"
                title={server.status}
              ></span>
              <span class="status-text">{server.status}</span>
            </div>
          </div>

          <div class="server-details">
            <div class="detail-row">
              <span class="detail-label">Latency:</span>
              <span class="detail-value">{formatLatency(server.latency_ms)}</span>
            </div>
            <div class="detail-row">
              <span class="detail-label">Last Check:</span>
              <span class="detail-value">{formatLastCheck(server.last_check)}</span>
            </div>
            <div class="detail-row">
              <span class="detail-label">Tools:</span>
              <span class="detail-value">{server.tools_available} available</span>
            </div>
          </div>

          {#if server.error}
            <div class="server-error">
              <span class="error-icon">⚠️</span>
              <span>{server.error}</span>
            </div>
          {/if}

          {#if server.status !== 'online'}
            <div class="server-actions">
              <button 
                class="retry-btn" 
                on:click={() => retryConnection(server.server_id)}
              >
                Retry Connection
              </button>
            </div>
          {/if}

          <!-- Tools List (expandable) -->
          {#if server.tools.length > 0}
            <details class="tools-list">
              <summary>Available Tools ({server.tools.length})</summary>
              <ul>
                {#each server.tools as tool}
                  <li>{tool}</li>
                {/each}
              </ul>
            </details>
          {/if}
        </div>
      {/each}
    </div>

    <!-- Last Updated -->
    <div class="last-updated">
      Last updated: {new Date(status.last_updated).toLocaleString()}
    </div>
  {/if}
</div>

<style>
  .mcp-status-panel {
    background: var(--bg-secondary, #1e1e2e);
    border-radius: 8px;
    padding: 16px;
    color: var(--text-primary, #cdd6f4);
  }

  .panel-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 16px;
    padding-bottom: 12px;
    border-bottom: 1px solid var(--border-color, #313244);
  }

  .panel-header h3 {
    margin: 0;
    font-size: 1.125rem;
  }

  .header-actions {
    display: flex;
    align-items: center;
    gap: 12px;
  }

  .refresh-btn {
    background: transparent;
    border: none;
    cursor: pointer;
    padding: 4px;
    border-radius: 4px;
    color: var(--text-secondary, #a6adc8);
  }

  .refresh-btn:hover {
    background: var(--bg-hover, #313244);
  }

  .refresh-btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .icon.spinning {
    animation: spin 1s linear infinite;
  }

  @keyframes spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
  }

  .auto-refresh-toggle {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 0.875rem;
    color: var(--text-secondary, #a6adc8);
  }

  .status-overview {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
    margin-bottom: 16px;
  }

  .overview-item {
    background: var(--bg-tertiary, #313244);
    padding: 12px;
    border-radius: 6px;
    text-align: center;
  }

  .overview-item.online {
    border-left: 3px solid #a6e3a1;
  }

  .overview-item.offline {
    border-left: 3px solid #f38ba8;
  }

  .overview-label {
    display: block;
    font-size: 0.75rem;
    color: var(--text-secondary, #a6adc8);
    margin-bottom: 4px;
  }

  .overview-value {
    font-size: 1.5rem;
    font-weight: 600;
  }

  .health-indicator {
    grid-column: span 3;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    padding: 12px;
    border-radius: 6px;
    background: var(--bg-tertiary, #313244);
  }

  .health-indicator.healthy {
    background: rgba(166, 227, 161, 0.1);
    color: #a6e3a1;
  }

  .health-indicator:not(.healthy) {
    background: rgba(243, 139, 168, 0.1);
    color: #f38ba8;
  }

  .server-list {
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  .server-card {
    background: var(--bg-tertiary, #313244);
    border-radius: 8px;
    padding: 12px;
    border: 1px solid var(--border-color, #45475a);
  }

  .server-card.offline {
    border-color: #f38ba8;
  }

  .server-header {
    display: flex;
    align-items: flex-start;
    gap: 12px;
  }

  .server-icon {
    font-size: 1.5rem;
    width: 40px;
    height: 40px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: var(--bg-secondary, #1e1e2e);
    border-radius: 8px;
  }

  .server-info {
    flex: 1;
  }

  .server-name {
    margin: 0;
    font-size: 1rem;
    font-weight: 600;
  }

  .server-description {
    margin: 4px 0 0;
    font-size: 0.75rem;
    color: var(--text-secondary, #a6adc8);
  }

  .server-status {
    display: flex;
    align-items: center;
    gap: 6px;
  }

  .status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
  }

  .status-text {
    font-size: 0.75rem;
    text-transform: capitalize;
  }

  .server-details {
    margin-top: 12px;
    padding-top: 12px;
    border-top: 1px solid var(--border-color, #45475a);
  }

  .detail-row {
    display: flex;
    justify-content: space-between;
    font-size: 0.875rem;
    margin-bottom: 4px;
  }

  .detail-label {
    color: var(--text-secondary, #a6adc8);
  }

  .server-error {
    margin-top: 12px;
    padding: 8px;
    background: rgba(243, 139, 168, 0.1);
    border-radius: 4px;
    font-size: 0.875rem;
    color: #f38ba8;
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .server-actions {
    margin-top: 12px;
  }

  .retry-btn {
    background: var(--accent, #89b4fa);
    color: var(--bg-primary, #1e1e2e);
    border: none;
    padding: 6px 12px;
    border-radius: 4px;
    font-size: 0.875rem;
    cursor: pointer;
  }

  .retry-btn:hover {
    opacity: 0.9;
  }

  .tools-list {
    margin-top: 12px;
    font-size: 0.875rem;
  }

  .tools-list summary {
    cursor: pointer;
    color: var(--text-secondary, #a6adc8);
  }

  .tools-list ul {
    margin: 8px 0 0;
    padding-left: 20px;
    color: var(--text-primary, #cdd6f4);
  }

  .tools-list li {
    margin-bottom: 4px;
  }

  .loading-state,
  .error-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 32px;
    text-align: center;
  }

  .spinner {
    width: 32px;
    height: 32px;
    border: 3px solid var(--border-color, #45475a);
    border-top-color: var(--accent, #89b4fa);
    border-radius: 50%;
    animation: spin 1s linear infinite;
    margin-bottom: 12px;
  }

  .last-updated {
    margin-top: 16px;
    padding-top: 12px;
    border-top: 1px solid var(--border-color, #313244);
    font-size: 0.75rem;
    color: var(--text-secondary, #a6adc8);
    text-align: center;
  }
</style>
