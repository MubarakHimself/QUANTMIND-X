<script lang="ts">
  import { createBubbler, stopPropagation } from 'svelte/legacy';

  const bubble = createBubbler();
  import { fade, slide } from 'svelte/transition';
  import { onMount } from 'svelte';
  import { Plus, Trash2, RefreshCw, Server, Check, X, AlertCircle, ExternalLink, Loader2, Terminal, Globe, Zap, AlertTriangle, Wifi, WifiOff } from 'lucide-svelte';
  import { settingsStore } from '../../../stores/settingsStore';
  import type { MCPServer } from '../../../stores/settingsStore';
  import { API_CONFIG } from '$lib/config/api';

  const API_BASE = API_CONFIG.API_BASE;

  // Error state for each server
  let serverErrors: Record<string, { message: string; timestamp: Date } | null> = $state({});
  let connectingServers: Set<string> = new Set();
  let showErrorDetails: Record<string, boolean> = {};

  // Default server templates
  const DEFAULT_SERVERS = [
    {
      id: 'context7',
      name: 'Context7 MCP',
      description: 'MQL5 documentation retrieval',
      type: 'stdio' as const,
      command: 'npx',
      args: ['-y', '@context7/mcp-server'],
      autoConnect: false
    },
    {
      id: 'filesystem',
      name: 'Filesystem MCP',
      description: 'Local filesystem access',
      type: 'stdio' as const,
      command: 'npx',
      args: ['-y', '@anthropic-ai/mcp-server-filesystem', '--root', './workspace'],
      autoConnect: false
    },
    {
      id: 'metatrader5',
      name: 'MetaTrader 5 MCP',
      description: 'MetaTrader 5 trading platform integration',
      type: 'stdio' as const,
      command: 'npx',
      args: ['-y', '@anthropic-ai/mcp-server-mt5'],
      autoConnect: false
    },
    {
      id: 'sequential_thinking',
      name: 'Sequential Thinking MCP',
      description: 'Task decomposition and reasoning',
      type: 'stdio' as const,
      command: 'npx',
      args: ['-y', '@anthropic-ai/mcp-server-sequential-thinking'],
      autoConnect: false
    }
  ];

  // State
  let showAddModal = $state(false);
  let serverType: 'http' | 'stdio' = $state('stdio');
  let newServer = $state({
    name: '',
    description: '',
    url: '',
    command: '',
    args: '',
    autoConnect: false
  });
  let editingServer: MCPServer | null = null;
  let loading = $state(false);
  let backendAvailable = $state(false);

  // Reactive state
  let mcpServers = $derived($settingsStore.mcpServers);
  let connectedCount = $derived(mcpServers.filter(s => s.status === 'connected').length);
  let errorCount = $derived(Object.values(serverErrors).filter(Boolean).length);

  // Fetch servers from backend on mount
  onMount(async () => {
    try {
      loading = true;
      const response = await fetch(`${API_BASE}/mcp/servers`);
      if (response.ok) {
        backendAvailable = true;
        const data = await response.json();
        console.log('MCP servers from backend:', data);
        // Could merge with local servers here if needed
      }
    } catch (e) {
      console.log('MCP backend not available, using local store');
      backendAvailable = false;
    } finally {
      loading = false;
    }
  });

  // Connect to server via API with error handling
  async function connectServer(server: MCPServer) {
    connectingServers.add(server.id);
    connectingServers = connectingServers;
    serverErrors[server.id] = null;

    if (backendAvailable) {
      try {
        const response = await fetch(`${API_BASE}/mcp/servers/${server.id}/connect`, {
          method: 'POST'
        });
        if (response.ok) {
          settingsStore.updateMCPServer(server.id, { status: 'connected' });
        } else {
          const error = await response.json().catch(() => ({}));
          throw new Error(error.message || `HTTP ${response.status}: Connection failed`);
        }
      } catch (e: any) {
        console.error('Failed to connect via API:', e);
        serverErrors[server.id] = {
          message: e.message || 'Failed to connect to server',
          timestamp: new Date()
        };
        settingsStore.updateMCPServer(server.id, { status: 'error' });
      } finally {
        connectingServers.delete(server.id);
        connectingServers = connectingServers;
      }
    } else {
      // Local mode - simulate connection
      await new Promise(resolve => setTimeout(resolve, 500));
      settingsStore.updateMCPServer(server.id, { status: 'connected' });
      connectingServers.delete(server.id);
      connectingServers = connectingServers;
    }
  }

  // Disconnect from server via API
  async function disconnectServer(server: MCPServer) {
    connectingServers.add(server.id);
    connectingServers = connectingServers;

    if (backendAvailable) {
      try {
        const response = await fetch(`${API_BASE}/mcp/servers/${server.id}/disconnect`, {
          method: 'POST'
        });
        if (response.ok) {
          settingsStore.updateMCPServer(server.id, { status: 'disconnected' });
        }
      } catch (e) {
        console.error('Failed to disconnect via API:', e);
        settingsStore.updateMCPServer(server.id, { status: 'disconnected' });
      } finally {
        connectingServers.delete(server.id);
        connectingServers = connectingServers;
      }
    } else {
      settingsStore.updateMCPServer(server.id, { status: 'disconnected' });
      connectingServers.delete(server.id);
      connectingServers = connectingServers;
    }
  }

  // Retry connection
  function retryConnection(server: MCPServer) {
    serverErrors[server.id] = null;
    connectServer(server);
  }

  // Dismiss error
  function dismissError(serverId: string) {
    serverErrors[serverId] = null;
  }

  // Toggle error details
  function toggleErrorDetails(serverId: string) {
    showErrorDetails[serverId] = !showErrorDetails[serverId];
  }
  
  // Add new server
  function handleAddServer() {
    if (!newServer.name) return;
    if (serverType === 'http' && !newServer.url) return;
    if (serverType === 'stdio' && !newServer.command) return;

    const serverData: Omit<MCPServer, 'id'> = {
      name: newServer.name,
      description: newServer.description,
      type: serverType,
      status: 'disconnected',
      capabilities: [],
      autoConnect: newServer.autoConnect
    };

    if (serverType === 'http') {
      serverData.url = newServer.url;
    } else {
      serverData.command = newServer.command;
      serverData.args = newServer.args ? newServer.args.split(' ').filter(a => a) : [];
    }

    settingsStore.addMCPServer(serverData);

    newServer = { name: '', description: '', url: '', command: '', args: '', autoConnect: false };
    showAddModal = false;
  }

  // Add from template
  function addFromTemplate(template: typeof DEFAULT_SERVERS[0]) {
    settingsStore.addMCPServer({
      name: template.name,
      description: template.description,
      type: template.type,
      command: template.command,
      args: template.args,
      status: 'disconnected',
      capabilities: [],
      autoConnect: template.autoConnect
    });
  }
  
  // Remove server
  function handleRemoveServer(serverId: string) {
    if (confirm('Are you sure you want to remove this server?')) {
      settingsStore.removeMCPServer(serverId);
    }
  }
  
  // Toggle server connection
  function handleToggleConnection(server: MCPServer) {
    if (server.status === 'connected') {
      disconnectServer(server);
    } else {
      connectServer(server);
    }
  }
  
  // Refresh server capabilities
  function handleRefreshServer(server: MCPServer) {
    // Simulate refreshing capabilities
    settingsStore.updateMCPServer(server.id, {
      capabilities: ['tools', 'resources', 'prompts'].filter(() => Math.random() > 0.3)
    });
  }
  
  // Get status color
  function getStatusColor(status: string): string {
    switch (status) {
      case 'connected': return 'var(--accent-success)';
      case 'error': return 'var(--accent-danger)';
      case 'connecting': return 'var(--accent-warning)';
      default: return 'var(--text-muted)';
    }
  }

  // Get status icon
  function getStatusIcon(status: string) {
    switch (status) {
      case 'connected': return Wifi;
      case 'error': return AlertTriangle;
      case 'connecting': return Loader2;
      default: return WifiOff;
    }
  }

  // Check if server is connecting
  function isConnecting(serverId: string): boolean {
    return connectingServers.has(serverId);
  }

  // Format error timestamp
  function formatErrorTime(date?: Date): string {
    if (!date) return '';
    return new Date(date).toLocaleTimeString();
  }
  
  // Format last connected
  function formatLastConnected(date?: Date): string {
    if (!date) return 'Never';
    return new Date(date).toLocaleString();
  }

  // Get backend status
  function getBackendStatus(): string {
    if (loading) return 'Checking...';
    return backendAvailable ? 'API Connected' : 'Local Mode';
  }
</script>

<div class="mcp-settings">
  <div class="header">
    <div class="header-info">
      <h3>MCP Servers</h3>
      <p class="description">Manage Model Context Protocol servers for extended agent capabilities.</p>
    </div>
    <div class="header-stats">
      <span class="stat">
        <Check size={12} />
        {connectedCount} connected
      </span>
      <span class="stat">
        <Server size={12} />
        {mcpServers.length} total
      </span>
      {#if errorCount > 0}
        <span class="stat error">
          <AlertCircle size={12} />
          {errorCount} error{errorCount > 1 ? 's' : ''}
        </span>
      {/if}
      <span class="stat" class:success={backendAvailable}>
        {#if loading}
          <Loader2 size={10} class="spin" />
        {/if}
        {getBackendStatus()}
      </span>
    </div>
  </div>
  
  <!-- Server List -->
  <div class="server-list">
    {#if mcpServers.length === 0}
      <div class="empty-state">
        <Server size={32} />
        <h4>No MCP Servers</h4>
        <p>Add an MCP server to extend agent capabilities with tools, resources, and prompts.</p>
        <button class="btn primary" onclick={() => showAddModal = true}>
          <Plus size={14} />
          Add Server
        </button>
      </div>
    {:else}
      {#each mcpServers as server (server.id)}
        <div class="server-card" class:connected={server.status === 'connected'} class:error={server.status === 'error'}>
          <div class="server-header">
            <div class="server-info">
              <div class="server-name">
                <Server size={14} />
                {server.name}
              </div>
              <span class="server-url">{server.url}</span>
            </div>
            <div class="server-status" style="color: {getStatusColor(server.status)}">
              {#if isConnecting(server.id)}
                <Loader2 size={12} class="spin" />
              {:else}
                {@const SvelteComponent = getStatusIcon(server.status)}
                <SvelteComponent size={12} />
              {/if}
              <span class="status-text">{isConnecting(server.id) ? 'Connecting...' : server.status}</span>
            </div>
          </div>

          <!-- Error Display -->
          {#if serverErrors[server.id]}
            <div class="server-error" transition:slide>
              <div class="error-header">
                <AlertTriangle size={12} />
                <span>Connection Error</span>
                <button class="error-dismiss" onclick={() => dismissError(server.id)}>x</button>
              </div>
              <div class="error-message">{serverErrors[server.id]?.message}</div>
              <div class="error-time">{formatErrorTime(serverErrors[server.id]?.timestamp)}</div>
              <button class="btn small retry-btn" onclick={() => retryConnection(server)}>
                <Zap size={10} /> Retry
              </button>
            </div>
          {/if}

          {#if server.capabilities.length > 0}
            <div class="server-capabilities">
              {#each server.capabilities as cap}
                <span class="capability-tag">{cap}</span>
              {/each}
            </div>
          {/if}

          <div class="server-meta">
            <span class="meta-item">Last connected: {formatLastConnected(server.lastConnected)}</span>
            <label class="auto-connect">
              <input
                type="checkbox"
                checked={server.autoConnect}
                onchange={() => settingsStore.updateMCPServer(server.id, { autoConnect: !server.autoConnect })}
              />
              Auto-connect
            </label>
          </div>

          <div class="server-actions">
            <button
              class="btn secondary small"
              onclick={() => handleToggleConnection(server)}
              disabled={isConnecting(server.id)}
            >
              {#if isConnecting(server.id)}
                <Loader2 size={12} class="spin" /> Connecting...
              {:else if server.status === 'connected'}
                <X size={12} /> Disconnect
              {:else}
                <Check size={12} /> Connect
              {/if}
            </button>
            <button
              class="btn secondary small"
              onclick={() => handleRefreshServer(server)}
              title="Refresh capabilities"
              disabled={isConnecting(server.id)}
            >
              <RefreshCw size={12} />
            </button>
            <button
              class="btn secondary small danger"
              onclick={() => handleRemoveServer(server.id)}
              title="Remove server"
              disabled={isConnecting(server.id)}
            >
              <Trash2 size={12} />
            </button>
          </div>
        </div>
      {/each}
    {/if}
  </div>
  
  <!-- Add Server Button -->
  {#if mcpServers.length > 0}
    <button class="btn primary add-btn" onclick={() => showAddModal = true}>
      <Plus size={14} />
      Add MCP Server
    </button>
  {/if}
  
  <!-- Add Server Modal -->
  {#if showAddModal}
    <!-- svelte-ignore a11y_click_events_have_key_events, a11y_no_static_element_interactions -->
    <div class="modal-overlay" onclick={() => showAddModal = false} transition:fade role="button" tabindex="-1" aria-label="Close dialog">
      <!-- svelte-ignore a11y_click_events_have_key_events, a11y_no_static_element_interactions, a11y_no_noninteractive_element_interactions -->
      <div class="modal" onclick={stopPropagation(bubble('click'))} transition:slide role="dialog" aria-modal="true" aria-labelledby="mcp-modal-title">
        <h4 id="mcp-modal-title">Add MCP Server</h4>

        <!-- Quick Add Templates -->
        <div class="templates-section">
          <h5>Quick Add</h5>
          <div class="template-grid">
            {#each DEFAULT_SERVERS as template}
              <button class="template-btn" onclick={() => addFromTemplate(template)}>
                <Terminal size={14} />
                {template.name}
              </button>
            {/each}
          </div>
        </div>

        <div class="divider">
          <span>or add custom</span>
        </div>

        <!-- Server Type Toggle -->
        <div class="form-group">
          <label>Server Type</label>
          <div class="type-toggle">
            <button
              class="type-btn"
              class:active={serverType === 'http'}
              onclick={() => serverType = 'http'}
            >
              <Globe size={14} />
              HTTP URL
            </button>
            <button
              class="type-btn"
              class:active={serverType === 'stdio'}
              onclick={() => serverType = 'stdio'}
            >
              <Terminal size={14} />
              Command (Stdio)
            </button>
          </div>
        </div>

        <div class="form-group">
          <label for="server-name">Server Name</label>
          <input
            id="server-name"
            type="text"
            placeholder="e.g., My Custom Server"
            bind:value={newServer.name}
          />
        </div>

        <div class="form-group">
          <label for="server-description">Description (optional)</label>
          <input
            id="server-description"
            type="text"
            placeholder="What does this server provide?"
            bind:value={newServer.description}
          />
        </div>

        {#if serverType === 'http'}
          <div class="form-group">
            <label for="server-url">Server URL</label>
            <input
              id="server-url"
              type="text"
              placeholder="e.g., http://localhost:3000/mcp"
              bind:value={newServer.url}
            />
          </div>
        {:else}
          <div class="form-group">
            <label for="server-command">Command</label>
            <input
              id="server-command"
              type="text"
              placeholder="e.g., npx, python, node"
              bind:value={newServer.command}
            />
          </div>

          <div class="form-group">
            <label for="server-args">Arguments</label>
            <input
              id="server-args"
              type="text"
              placeholder="e.g., -y @package-name --flag value"
              bind:value={newServer.args}
            />
          </div>
        {/if}

        <div class="form-group">
          <label class="checkbox-label">
            <input type="checkbox" bind:checked={newServer.autoConnect} />
            Auto-connect on startup
          </label>
        </div>

        <div class="modal-actions">
          <button class="btn secondary" onclick={() => showAddModal = false}>Cancel</button>
          <button
            class="btn primary"
            onclick={handleAddServer}
            disabled={!newServer.name || (serverType === 'http' ? !newServer.url : !newServer.command)}
          >
            Add Server
          </button>
        </div>
      </div>
    </div>
  {/if}
  
  <!-- Help Section -->
  <div class="help-section">
    <h4>What is MCP?</h4>
    <p>
      Model Context Protocol (MCP) servers provide additional tools, resources, and prompts to agents.
      Connect servers like file systems, databases, or custom tools to extend agent capabilities.
    </p>
    <a href="https://modelcontextprotocol.io" target="_blank" rel="noopener noreferrer" class="docs-link">
      Learn more about MCP <ExternalLink size={12} />
    </a>
  </div>
</div>

<style>
  .mcp-settings {
    display: flex;
    flex-direction: column;
    gap: 20px;
  }
  
  .header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 16px;
  }
  
  .header-info h3 {
    margin: 0;
    font-size: 18px;
    font-weight: 600;
    color: var(--text-primary);
  }
  
  .description {
    margin: 4px 0 0;
    font-size: 12px;
    color: var(--text-secondary);
  }
  
  .header-stats {
    display: flex;
    gap: 12px;
    flex-shrink: 0;
  }
  
  .stat {
    display: flex;
    align-items: center;
    gap: 4px;
    font-size: 11px;
    color: var(--text-muted);
    padding: 4px 8px;
    background: var(--bg-tertiary);
    border-radius: 4px;
  }

  .stat.success {
    color: var(--accent-success);
    background: rgba(16, 185, 129, 0.1);
  }

  .stat.error {
    color: var(--accent-danger);
    background: rgba(239, 68, 68, 0.1);
  }

  :global(.spin) {
    animation: spin 1s linear infinite;
  }

  @keyframes spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
  }
  
  .server-list {
    display: flex;
    flex-direction: column;
    gap: 12px;
  }
  
  .empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    text-align: center;
    padding: 32px;
    background: var(--bg-tertiary);
    border: 1px dashed var(--border-subtle);
    border-radius: 12px;
    color: var(--text-muted);
  }
  
  .empty-state h4 {
    margin: 12px 0 4px;
    font-size: 14px;
    color: var(--text-primary);
  }
  
  .empty-state p {
    margin: 0 0 16px;
    font-size: 12px;
    max-width: 280px;
  }
  
  .server-card {
    display: flex;
    flex-direction: column;
    gap: 12px;
    padding: 16px;
    background: var(--bg-tertiary);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
    transition: border-color 0.15s;
  }
  
  .server-card.connected {
    border-color: var(--accent-success);
  }

  .server-card.error {
    border-color: var(--accent-danger);
    background: rgba(239, 68, 68, 0.05);
  }
  
  .server-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
  }
  
  .server-info {
    display: flex;
    flex-direction: column;
    gap: 2px;
  }
  
  .server-name {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 13px;
    font-weight: 500;
    color: var(--text-primary);
  }
  
  .server-url {
    font-size: 11px;
    color: var(--text-muted);
    font-family: monospace;
  }
  
  .server-status {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 11px;
    font-weight: 500;
    text-transform: capitalize;
  }

  .status-text {
    text-transform: capitalize;
  }

  .server-error {
    padding: 10px;
    background: rgba(239, 68, 68, 0.1);
    border: 1px solid rgba(239, 68, 68, 0.3);
    border-radius: 6px;
    margin: 8px 0;
  }

  .error-header {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 11px;
    font-weight: 600;
    color: var(--accent-danger);
    margin-bottom: 6px;
  }

  .error-dismiss {
    margin-left: auto;
    background: none;
    border: none;
    color: var(--text-muted);
    cursor: pointer;
    font-size: 12px;
    padding: 0 4px;
  }

  .error-dismiss:hover {
    color: var(--text-primary);
  }

  .error-message {
    font-size: 12px;
    color: var(--text-secondary);
    margin-bottom: 4px;
    font-family: monospace;
  }

  .error-time {
    font-size: 10px;
    color: var(--text-muted);
    margin-bottom: 8px;
  }

  .retry-btn {
    background: var(--accent-primary) !important;
    color: var(--bg-primary) !important;
    font-size: 10px !important;
    padding: 4px 8px !important;
  }

  .status-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: currentColor;
  }
  
  .server-capabilities {
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
  }
  
  .capability-tag {
    font-size: 10px;
    padding: 2px 8px;
    background: var(--bg-primary);
    border-radius: 4px;
    color: var(--text-secondary);
    text-transform: capitalize;
  }
  
  .server-meta {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 10px;
    color: var(--text-muted);
  }
  
  .auto-connect {
    display: flex;
    align-items: center;
    gap: 6px;
    cursor: pointer;
  }
  
  .auto-connect input {
    width: 14px;
    height: 14px;
  }
  
  .server-actions {
    display: flex;
    gap: 8px;
    padding-top: 8px;
    border-top: 1px solid var(--border-subtle);
  }
  
  .btn {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 8px 14px;
    border-radius: 6px;
    font-size: 12px;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.15s;
    border: none;
  }
  
  .btn.primary {
    background: var(--accent-primary);
    color: var(--bg-primary);
  }
  
  .btn.primary:hover:not(:disabled) {
    background: var(--accent-secondary);
  }
  
  .btn.primary:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
  
  .btn.secondary {
    background: var(--bg-primary);
    color: var(--text-secondary);
    border: 1px solid var(--border-subtle);
  }
  
  .btn.secondary:hover {
    background: var(--bg-tertiary);
    color: var(--text-primary);
  }
  
  .btn.secondary.danger:hover {
    color: var(--accent-danger);
    border-color: var(--accent-danger);
  }
  
  .btn.small {
    padding: 6px 10px;
    font-size: 11px;
  }
  
  .add-btn {
    align-self: flex-start;
  }
  
  /* Modal */
  .modal-overlay {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(0, 0, 0, 0.6);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 1000;
  }
  
  .modal {
    background: var(--bg-secondary);
    border-radius: 12px;
    padding: 24px;
    width: 360px;
    max-width: 90%;
    border: 1px solid var(--border-subtle);
  }
  
  .modal h4 {
    margin: 0 0 20px;
    font-size: 16px;
    color: var(--text-primary);
  }
  
  .form-group {
    margin-bottom: 16px;
  }
  
  .form-group label {
    display: block;
    margin-bottom: 6px;
    font-size: 12px;
    font-weight: 500;
    color: var(--text-secondary);
  }
  
  .form-group input[type="text"] {
    width: 100%;
    padding: 10px 12px;
    background: var(--bg-tertiary);
    border: 1px solid var(--border-subtle);
    border-radius: 6px;
    color: var(--text-primary);
    font-size: 13px;
  }
  
  .form-group input:focus {
    outline: none;
    border-color: var(--accent-primary);
  }
  
  .checkbox-label {
    display: flex;
    align-items: center;
    gap: 8px;
    cursor: pointer;
    font-size: 12px;
    color: var(--text-secondary);
  }
  
  .modal-actions {
    display: flex;
    justify-content: flex-end;
    gap: 8px;
    margin-top: 20px;
  }
  
  /* Help Section */
  .help-section {
    padding: 16px;
    background: var(--bg-tertiary);
    border-radius: 8px;
    border: 1px solid var(--border-subtle);
  }
  
  .help-section h4 {
    margin: 0 0 8px;
    font-size: 12px;
    font-weight: 600;
    color: var(--text-primary);
  }
  
  .help-section p {
    margin: 0 0 12px;
    font-size: 12px;
    color: var(--text-secondary);
    line-height: 1.5;
  }
  
  .docs-link {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    font-size: 11px;
    color: var(--accent-primary);
    text-decoration: none;
  }
  
  .docs-link:hover {
    text-decoration: underline;
  }

  /* Templates Section */
  .templates-section {
    margin-bottom: 16px;
  }

  .templates-section h5 {
    margin: 0 0 12px;
    font-size: 12px;
    font-weight: 500;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  .template-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 8px;
  }

  .template-btn {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 8px 12px;
    background: var(--bg-tertiary);
    border: 1px solid var(--border-subtle);
    border-radius: 6px;
    color: var(--text-primary);
    font-size: 12px;
    cursor: pointer;
    transition: all 0.15s ease;
  }

  .template-btn:hover {
    background: var(--bg-secondary);
    border-color: var(--accent-primary);
  }

  .divider {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 16px 0;
    color: var(--text-muted);
    font-size: 11px;
  }

  .divider::before,
  .divider::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border-subtle);
  }

  /* Type Toggle */
  .type-toggle {
    display: flex;
    gap: 8px;
  }

  .type-btn {
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 6px;
    padding: 10px;
    background: var(--bg-tertiary);
    border: 1px solid var(--border-subtle);
    border-radius: 6px;
    color: var(--text-secondary);
    font-size: 12px;
    cursor: pointer;
    transition: all 0.15s ease;
  }

  .type-btn:hover {
    background: var(--bg-secondary);
  }

  .type-btn.active {
    background: var(--accent-primary);
    border-color: var(--accent-primary);
    color: var(--bg-primary);
  }
</style>
