<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import {
    Server, Terminal, Plus, Eye, EyeOff, Trash2, RefreshCw,
    AlertCircle
  } from 'lucide-svelte';

  export let mcpServers: Array<{
    id: string;
    name: string;
    command: string;
    args: string[];
    status: 'running' | 'stopped' | 'error';
    type: 'builtin' | 'custom';
    description?: string;
  }> = [];

  export let mcpModalOpen = false;
  export let newMcpServer = {
    name: '',
    command: '',
    args: '',
    description: ''
  };

  const dispatch = createEventDispatcher();

  const DEFAULT_MCP_SERVERS = [
    { name: 'Context7 MCP', command: 'npx', args: ['-y', '@context7/mcp-server'], description: 'MQL5 documentation retrieval' },
    { name: 'Filesystem MCP', command: 'npx', args: ['-y', '@anthropic-ai/mcp-server-filesystem', '--root', './workspace'], description: 'Local filesystem access' },
    { name: 'MetaTrader 5 MCP', command: 'npx', args: ['-y', '@anthropic-ai/mcp-server-mt5'], description: 'MT5 integration' },
    { name: 'Sequential Thinking MCP', command: 'npx', args: ['-y', '@anthropic-ai/mcp-server-sequential-thinking'], description: 'Task decomposition' },
    { name: 'Svelte MCP', command: 'npx', args: ['-y', '@sveltejs/mcp'], description: 'Svelte development tools' },
    { name: 'Chrome DevTools MCP', command: 'npx', args: ['-y', 'chrome-devtools-mcp@latest'], description: 'Browser automation' }
  ];

  function openModal() {
    mcpModalOpen = true;
    dispatch('openModal');
  }

  function closeModal() {
    mcpModalOpen = false;
    dispatch('closeModal');
  }

  function addMcpServer() {
    dispatch('addMcpServer');
  }

  function toggleMcpServer(id: string) {
    dispatch('toggleMcpServer', { id });
  }

  function removeMcpServer(id: string) {
    dispatch('removeMcpServer', { id });
  }

  function applyTemplate(template: typeof DEFAULT_MCP_SERVERS[0]) {
    newMcpServer = {
      name: template.name,
      command: template.command,
      args: template.args.join(' '),
      description: template.description
    };
  }
</script>

<div class="panel">
  <div class="panel-header">
    <h3>MCP Servers</h3>
    <button class="btn primary" on:click={openModal}>
      <Plus size={14} /> Add Server
    </button>
  </div>

  <div class="info-box">
    <Server size={16} />
    <span>MCP servers extend agent capabilities with external tools and data sources.</span>
  </div>

  <div class="servers-list">
    {#each mcpServers as server}
      <div class="server-item">
        <div class="server-icon">
          <Terminal size={20} />
        </div>
        <div class="server-info">
          <div class="server-name">{server.name}</div>
          <div class="server-desc">{server.description || 'Custom MCP server'}</div>
          <div class="server-command">
            <code>{server.command} {server.args.join(' ')}</code>
          </div>
        </div>
        <div class="server-status">
          <span class="status-badge" class:running={server.status === 'running'} class:stopped={server.status === 'stopped'} class:error={server.status === 'error'}>
            {server.status}
          </span>
        </div>
        <div class="server-actions">
          <button
            class="icon-btn"
            on:click={() => toggleMcpServer(server.id)}
            title={server.status === 'running' ? 'Stop' : 'Start'}
          >
            {#if server.status === 'running'}
              <EyeOff size={14} />
            {:else}
              <Eye size={14} />
            {/if}
          </button>
          {#if server.type === 'custom'}
            <button class="icon-btn danger" on:click={() => removeMcpServer(server.id)} title="Remove">
              <Trash2 size={14} />
            </button>
          {/if}
        </div>
      </div>
    {:else}
      <div class="empty-state">
        <Server size={32} />
        <p>No MCP servers configured</p>
        <button class="btn primary" on:click={openModal}>
          Add MCP Server
        </button>
      </div>
    {/each}
  </div>
</div>

<!-- MCP Server Modal -->
{#if mcpModalOpen}
  <div class="modal-overlay" on:click|self={closeModal}>
    <div class="modal">
      <div class="modal-header">
        <h3>Add MCP Server</h3>
        <button on:click={closeModal}><RefreshCw size={20} /></button>
      </div>
      <div class="modal-body">
        <div class="quick-add-section">
          <h4>Quick Add</h4>
          <div class="template-grid">
            {#each DEFAULT_MCP_SERVERS as template}
              <button
                class="template-card"
                on:click={() => applyTemplate(template)}
              >
                <Terminal size={16} />
                <span>{template.name}</span>
              </button>
            {/each}
          </div>
        </div>

        <div class="divider">
          <span>or add custom</span>
        </div>

        <div class="form-group">
          <label>Server Name</label>
          <input type="text" placeholder="My Custom Server" bind:value={newMcpServer.name} />
        </div>
        <div class="form-group">
          <label>Command</label>
          <input type="text" placeholder="npx" bind:value={newMcpServer.command} />
          <small>The executable or command to run</small>
        </div>
        <div class="form-group">
          <label>Arguments (space-separated)</label>
          <input type="text" placeholder="-y @package/server --port 3000" bind:value={newMcpServer.args} />
          <small>Command line arguments, separated by spaces</small>
        </div>
        <div class="form-group">
          <label>Description</label>
          <textarea placeholder="What this server does..." bind:value={newMcpServer.description}></textarea>
        </div>
      </div>
      <div class="modal-footer">
        <button class="btn secondary" on:click={closeModal}>Cancel</button>
        <button class="btn primary" on:click={addMcpServer}>Add Server</button>
      </div>
    </div>
  </div>
{/if}

<style>
  /* Panel Header */
  .panel-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 20px;
  }

  .panel-header h3 {
    margin: 0;
    font-size: 16px;
    font-weight: 600;
    color: var(--text-primary);
  }

  /* Info Box */
  .info-box {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    padding: 12px 16px;
    background: rgba(59, 130, 246, 0.1);
    border: 1px solid rgba(59, 130, 246, 0.2);
    border-radius: 8px;
    margin-bottom: 20px;
    color: #60a5fa;
    font-size: 13px;
    line-height: 1.5;
  }

  .info-box :global(svg) {
    flex-shrink: 0;
    margin-top: 2px;
  }

  /* Servers List */
  .servers-list {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  /* Server Item */
  .server-item {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    padding: 14px 16px;
    background: var(--bg-tertiary);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
    transition: all 0.15s ease;
  }

  .server-item:hover {
    background: var(--bg-surface);
    border-color: var(--accent-primary);
    transform: translateX(2px);
  }

  .server-icon {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 40px;
    height: 40px;
    background: rgba(16, 185, 129, 0.15);
    border-radius: 8px;
    color: #10b981;
    flex-shrink: 0;
  }

  .server-info {
    flex: 1;
    min-width: 0;
  }

  .server-name {
    font-size: 14px;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 4px;
  }

  .server-desc {
    font-size: 13px;
    color: var(--text-secondary);
    margin-bottom: 8px;
  }

  .server-command {
    display: flex;
    align-items: center;
  }

  .server-command code {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: var(--text-muted);
    background: var(--bg-primary);
    padding: 4px 8px;
    border-radius: 4px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 300px;
    display: inline-block;
  }

  /* Status Badge */
  .status-badge {
    display: inline-flex;
    align-items: center;
    padding: 4px 10px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: 500;
    text-transform: capitalize;
  }

  .status-badge.running {
    background: rgba(16, 185, 129, 0.15);
    color: #10b981;
  }

  .status-badge.stopped {
    background: var(--bg-primary);
    color: var(--text-muted);
  }

  .status-badge.error {
    background: rgba(239, 68, 68, 0.15);
    color: #ef4444;
  }

  .server-status {
    flex-shrink: 0;
  }

  .server-actions {
    display: flex;
    gap: 4px;
    flex-shrink: 0;
  }

  /* Icon Button */
  .icon-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 32px;
    height: 32px;
    background: transparent;
    border: none;
    border-radius: 6px;
    color: var(--text-muted);
    cursor: pointer;
    transition: all 0.15s;
  }

  .icon-btn:hover {
    background: var(--bg-primary);
    color: var(--text-primary);
  }

  .icon-btn.danger:hover {
    background: rgba(239, 68, 68, 0.15);
    color: #ef4444;
  }

  /* Empty State */
  .empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 48px 24px;
    text-align: center;
    color: var(--text-muted);
  }

  .empty-state :global(svg) {
    margin-bottom: 16px;
    opacity: 0.5;
  }

  .empty-state p {
    margin: 0 0 16px;
    font-size: 14px;
  }

  /* Modal */
  .modal-overlay {
    position: fixed;
    inset: 0;
    background: rgba(0, 0, 0, 0.7);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 1000;
  }

  .modal {
    background: var(--bg-secondary);
    border-radius: 12px;
    width: 500px;
    max-width: 95vw;
    max-height: 85vh;
    overflow-y: auto;
    box-shadow: 0 25px 50px rgba(0, 0, 0, 0.5);
  }

  .modal-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 16px 20px;
    border-bottom: 1px solid var(--border-subtle);
    position: sticky;
    top: 0;
    background: var(--bg-secondary);
  }

  .modal-header h3 {
    margin: 0;
    font-size: 16px;
    font-weight: 600;
    color: var(--text-primary);
  }

  .modal-header button {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 32px;
    height: 32px;
    background: transparent;
    border: none;
    border-radius: 6px;
    color: var(--text-muted);
    cursor: pointer;
    transition: all 0.15s;
  }

  .modal-header button:hover {
    background: var(--bg-tertiary);
    color: var(--text-primary);
  }

  .modal-body {
    padding: 20px;
    display: flex;
    flex-direction: column;
    gap: 16px;
  }

  /* Quick Add Section */
  .quick-add-section h4 {
    margin: 0 0 12px;
    font-size: 13px;
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

  .template-card {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 10px 12px;
    background: var(--bg-tertiary);
    border: 1px solid var(--border-subtle);
    border-radius: 6px;
    color: var(--text-secondary);
    font-size: 12px;
    cursor: pointer;
    transition: all 0.15s;
    text-align: left;
  }

  .template-card:hover {
    background: var(--bg-surface);
    border-color: var(--accent-primary);
    color: var(--text-primary);
  }

  .template-card :global(svg) {
    flex-shrink: 0;
    color: var(--accent-primary);
  }

  .divider {
    display: flex;
    align-items: center;
    gap: 12px;
    color: var(--text-muted);
    font-size: 12px;
    margin: 8px 0;
  }

  .divider::before,
  .divider::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border-subtle);
  }

  .form-group {
    display: flex;
    flex-direction: column;
    gap: 6px;
  }

  .form-group label {
    font-size: 13px;
    font-weight: 500;
    color: var(--text-secondary);
  }

  .form-group input,
  .form-group select,
  .form-group textarea {
    padding: 10px 12px;
    background: var(--bg-tertiary);
    border: 1px solid var(--border-subtle);
    border-radius: 6px;
    color: var(--text-primary);
    font-size: 14px;
    transition: all 0.15s;
    font-family: inherit;
  }

  .form-group input:focus,
  .form-group select:focus,
  .form-group textarea:focus {
    outline: none;
    border-color: var(--accent-primary);
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.15);
  }

  .form-group input::placeholder,
  .form-group textarea::placeholder {
    color: var(--text-muted);
    opacity: 0.6;
  }

  .form-group textarea {
    min-height: 80px;
    resize: vertical;
  }

  .form-group small {
    font-size: 11px;
    color: var(--text-muted);
  }

  .modal-footer {
    display: flex;
    justify-content: flex-end;
    gap: 8px;
    padding: 16px 20px;
    border-top: 1px solid var(--border-subtle);
    position: sticky;
    bottom: 0;
    background: var(--bg-secondary);
  }

  /* Buttons */
  .btn {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 6px;
    padding: 10px 16px;
    border-radius: 6px;
    font-size: 13px;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.15s;
    border: none;
  }

  .btn.primary {
    background: var(--accent-primary);
    color: white;
  }

  .btn.primary:hover {
    opacity: 0.9;
    transform: translateY(-1px);
  }

  .btn.secondary {
    background: var(--bg-tertiary);
    border: 1px solid var(--border-subtle);
    color: var(--text-secondary);
  }

  .btn.secondary:hover {
    background: var(--bg-surface);
    color: var(--text-primary);
  }
</style>
