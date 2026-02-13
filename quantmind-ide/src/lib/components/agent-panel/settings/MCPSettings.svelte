<script lang="ts">
  import { fade, slide } from 'svelte/transition';
  import { Plus, Trash2, RefreshCw, Server, Check, X, AlertCircle, ExternalLink } from 'lucide-svelte';
  import { settingsStore } from '../../../stores/settingsStore';
  import type { MCPServer } from '../../../stores/settingsStore';
  
  // State
  let showAddModal = false;
  let newServer = { name: '', url: '', autoConnect: true };
  let editingServer: MCPServer | null = null;
  
  // Reactive state
  $: mcpServers = $settingsStore.mcpServers;
  $: connectedCount = mcpServers.filter(s => s.status === 'connected').length;
  
  // Add new server
  function handleAddServer() {
    if (!newServer.name || !newServer.url) return;
    
    settingsStore.addMCPServer({
      name: newServer.name,
      url: newServer.url,
      status: 'disconnected',
      capabilities: [],
      autoConnect: newServer.autoConnect
    });
    
    newServer = { name: '', url: '', autoConnect: true };
    showAddModal = false;
  }
  
  // Remove server
  function handleRemoveServer(serverId: string) {
    if (confirm('Are you sure you want to remove this server?')) {
      settingsStore.removeMCPServer(serverId);
    }
  }
  
  // Toggle server connection
  function handleToggleConnection(server: MCPServer) {
    const newStatus = server.status === 'connected' ? 'disconnected' : 'connected';
    settingsStore.updateMCPServer(server.id, { 
      status: newStatus,
      lastConnected: newStatus === 'connected' ? new Date() : server.lastConnected
    });
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
      default: return 'var(--text-muted)';
    }
  }
  
  // Format last connected
  function formatLastConnected(date?: Date): string {
    if (!date) return 'Never';
    return new Date(date).toLocaleString();
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
    </div>
  </div>
  
  <!-- Server List -->
  <div class="server-list">
    {#if mcpServers.length === 0}
      <div class="empty-state">
        <Server size={32} />
        <h4>No MCP Servers</h4>
        <p>Add an MCP server to extend agent capabilities with tools, resources, and prompts.</p>
        <button class="btn primary" on:click={() => showAddModal = true}>
          <Plus size={14} />
          Add Server
        </button>
      </div>
    {:else}
      {#each mcpServers as server (server.id)}
        <div class="server-card" class:connected={server.status === 'connected'}>
          <div class="server-header">
            <div class="server-info">
              <div class="server-name">
                <Server size={14} />
                {server.name}
              </div>
              <span class="server-url">{server.url}</span>
            </div>
            <div class="server-status" style="color: {getStatusColor(server.status)}">
              <span class="status-dot"></span>
              {server.status}
            </div>
          </div>
          
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
                on:change={() => settingsStore.updateMCPServer(server.id, { autoConnect: !server.autoConnect })}
              />
              Auto-connect
            </label>
          </div>
          
          <div class="server-actions">
            <button 
              class="btn secondary small"
              on:click={() => handleToggleConnection(server)}
            >
              {#if server.status === 'connected'}
                <X size={12} /> Disconnect
              {:else}
                <Check size={12} /> Connect
              {/if}
            </button>
            <button 
              class="btn secondary small"
              on:click={() => handleRefreshServer(server)}
              title="Refresh capabilities"
            >
              <RefreshCw size={12} />
            </button>
            <button 
              class="btn secondary small danger"
              on:click={() => handleRemoveServer(server.id)}
              title="Remove server"
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
    <button class="btn primary add-btn" on:click={() => showAddModal = true}>
      <Plus size={14} />
      Add MCP Server
    </button>
  {/if}
  
  <!-- Add Server Modal -->
  {#if showAddModal}
    <!-- svelte-ignore a11y-click-events-have-key-events a11y-no-static-element-interactions -->
    <div class="modal-overlay" on:click={() => showAddModal = false} transition:fade role="button" tabindex="-1" aria-label="Close dialog">
      <!-- svelte-ignore a11y-click-events-have-key-events a11y-no-static-element-interactions a11y-no-noninteractive-element-interactions -->
      <div class="modal" on:click|stopPropagation transition:slide role="dialog" aria-modal="true" aria-labelledby="mcp-modal-title">
        <h4 id="mcp-modal-title">Add MCP Server</h4>
        
        <div class="form-group">
          <label for="server-name">Server Name</label>
          <input 
            id="server-name"
            type="text" 
            placeholder="e.g., File System"
            bind:value={newServer.name}
          />
        </div>
        
        <div class="form-group">
          <label for="server-url">Server URL</label>
          <input 
            id="server-url"
            type="text" 
            placeholder="e.g., http://localhost:3000"
            bind:value={newServer.url}
          />
        </div>
        
        <div class="form-group">
          <label class="checkbox-label">
            <input type="checkbox" bind:checked={newServer.autoConnect} />
            Auto-connect on startup
          </label>
        </div>
        
        <div class="modal-actions">
          <button class="btn secondary" on:click={() => showAddModal = false}>Cancel</button>
          <button 
            class="btn primary" 
            on:click={handleAddServer}
            disabled={!newServer.name || !newServer.url}
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
</style>
