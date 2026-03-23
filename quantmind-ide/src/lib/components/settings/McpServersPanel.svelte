<script lang="ts">
  import { onMount } from 'svelte';
  import {
    Server, Terminal, Plus, Play, Square, Trash2, RefreshCw,
    AlertCircle, X, Check
  } from 'lucide-svelte';
  import { apiFetch } from '$lib/api';

  interface McpServer {
    id: string;
    name: string;
    command: string;
    args: string[];
    status: 'running' | 'stopped' | 'error';
    type: 'builtin' | 'custom';
    description?: string;
  }

  const DEFAULT_MCP_SERVERS = [
    { name: 'Context7 MCP', command: 'npx', args: ['-y', '@context7/mcp-server'], description: 'MQL5 documentation retrieval' },
    { name: 'Filesystem MCP', command: 'npx', args: ['-y', '@anthropic-ai/mcp-server-filesystem', '--root', './workspace'], description: 'Local filesystem access' },
    { name: 'MetaTrader 5 MCP', command: 'npx', args: ['-y', '@anthropic-ai/mcp-server-mt5'], description: 'MT5 integration' },
    { name: 'Sequential Thinking MCP', command: 'npx', args: ['-y', '@anthropic-ai/mcp-server-sequential-thinking'], description: 'Task decomposition' },
    { name: 'Svelte MCP', command: 'npx', args: ['-y', '@sveltejs/mcp'], description: 'Svelte development tools' },
    { name: 'Chrome DevTools MCP', command: 'npx', args: ['-y', 'chrome-devtools-mcp@latest'], description: 'Browser automation' }
  ];

  let mcpServers: McpServer[] = $state([]);
  let isLoading = $state(false);
  let error = $state<string | null>(null);
  let success = $state<string | null>(null);
  let showAddModal = $state(false);
  let newServer = $state({ name: '', command: '', args: '', description: '' });

  async function loadServers() {
    isLoading = true;
    error = null;
    try {
      const data = await apiFetch<{ servers: McpServer[] }>('/api/settings/mcp');
      mcpServers = data.servers || [];
    } catch (e) {
      console.warn('MCP settings endpoint unavailable');
    } finally {
      isLoading = false;
    }
  }

  async function toggleServer(id: string) {
    const server = mcpServers.find(s => s.id === id);
    if (!server) return;
    try {
      const action = server.status === 'running' ? 'stop' : 'start';
      await apiFetch(`/api/settings/mcp/${id}/${action}`, { method: 'POST' });
      await loadServers();
    } catch (e) {
      error = `Failed to ${server.status === 'running' ? 'stop' : 'start'} server`;
      console.error(e);
    }
  }

  async function removeServer(id: string) {
    try {
      await apiFetch(`/api/settings/mcp/${id}`, { method: 'DELETE' });
      await loadServers();
    } catch (e) {
      error = 'Failed to remove server';
      console.error(e);
    }
  }

  async function addServer() {
    if (!newServer.name || !newServer.command) return;
    try {
      await apiFetch('/api/settings/mcp', {
        method: 'POST',
        body: JSON.stringify({
          name: newServer.name,
          command: newServer.command,
          args: newServer.args.split(' ').filter(Boolean),
          description: newServer.description,
          type: 'custom'
        })
      });
      showAddModal = false;
      newServer = { name: '', command: '', args: '', description: '' };
      success = 'Server added';
      setTimeout(() => success = null, 3000);
      await loadServers();
    } catch (e) {
      error = 'Failed to add server';
      console.error(e);
    }
  }

  function applyTemplate(template: typeof DEFAULT_MCP_SERVERS[0]) {
    newServer = {
      name: template.name,
      command: template.command,
      args: template.args.join(' '),
      description: template.description
    };
  }

  onMount(() => { loadServers(); });
</script>

<div class="panel">
  <div class="panel-header">
    <h3>MCP Servers</h3>
    <div class="header-actions">
      <button class="icon-btn" onclick={loadServers} title="Refresh" disabled={isLoading}>
        <RefreshCw size={16} class={isLoading ? 'spinning' : ''} />
      </button>
      <button class="icon-btn accent" onclick={() => showAddModal = true} title="Add Server">
        <Plus size={16} />
      </button>
    </div>
  </div>

  {#if error}
    <div class="alert-error"><AlertCircle size={14} /> <span>{error}</span></div>
  {/if}
  {#if success}
    <div class="alert-success"><Check size={14} /> <span>{success}</span></div>
  {/if}

  <div class="info-box">
    <Server size={14} />
    <span>MCP servers extend agent capabilities with external tools and data sources.</span>
  </div>

  <div class="servers-list">
    {#each mcpServers as server}
      <div class="server-item">
        <div class="server-icon">
          <Terminal size={16} />
        </div>
        <div class="server-info">
          <div class="server-name">{server.name}</div>
          <div class="server-desc">{server.description || 'Custom MCP server'}</div>
          <div class="server-command">
            <code>{server.command} {server.args.join(' ')}</code>
          </div>
        </div>
        <div class="server-status">
          <span
            class="status-badge"
            class:running={server.status === 'running'}
            class:stopped={server.status === 'stopped'}
            class:error={server.status === 'error'}
          >
            {server.status}
          </span>
        </div>
        <div class="server-actions">
          <button
            class="icon-btn"
            class:running={server.status === 'running'}
            onclick={() => toggleServer(server.id)}
            title={server.status === 'running' ? 'Stop' : 'Start'}
          >
            {#if server.status === 'running'}
              <Square size={13} />
            {:else}
              <Play size={13} />
            {/if}
          </button>
          {#if server.type === 'custom'}
            <button class="icon-btn danger" onclick={() => removeServer(server.id)} title="Remove">
              <Trash2 size={13} />
            </button>
          {/if}
        </div>
      </div>
    {:else}
      {#if !isLoading}
        <div class="empty-state">
          <Server size={32} />
          <p>No MCP servers configured</p>
          <button class="btn primary" onclick={() => showAddModal = true}>
            <Plus size={13} /> Add Server
          </button>
        </div>
      {/if}
    {/each}
  </div>
</div>

{#if showAddModal}
  <div
    class="modal-backdrop"
    onclick={() => showAddModal = false}
    role="button"
    tabindex="0"
    onkeydown={(e) => e.key === 'Escape' && (showAddModal = false)}
  >
    <div class="modal" onclick={(e) => e.stopPropagation()} role="dialog">
      <div class="modal-header">
        <h4>Add MCP Server</h4>
        <button class="icon-btn" onclick={() => showAddModal = false}><X size={16} /></button>
      </div>
      <div class="modal-body">
        <div class="quick-add-section">
          <div class="subsection-title">Quick Add</div>
          <div class="template-grid">
            {#each DEFAULT_MCP_SERVERS as template}
              <button class="template-card" onclick={() => applyTemplate(template)}>
                <Terminal size={13} />
                <span>{template.name}</span>
              </button>
            {/each}
          </div>
        </div>

        <div class="divider"><span>or add custom</span></div>

        <div class="form-group">
          <label>Server Name</label>
          <input type="text" class="text-input" placeholder="My Custom Server" bind:value={newServer.name} />
        </div>
        <div class="form-group">
          <label>Command</label>
          <input type="text" class="text-input" placeholder="npx" bind:value={newServer.command} />
          <span class="hint">The executable or command to run</span>
        </div>
        <div class="form-group">
          <label>Arguments (space-separated)</label>
          <input type="text" class="text-input" placeholder="-y @package/server --port 3000" bind:value={newServer.args} />
        </div>
        <div class="form-group">
          <label>Description</label>
          <textarea class="text-input textarea" placeholder="What this server does..." bind:value={newServer.description}></textarea>
        </div>
      </div>
      <div class="modal-footer">
        <button class="btn secondary" onclick={() => showAddModal = false}>Cancel</button>
        <button class="btn primary" onclick={addServer} disabled={!newServer.name || !newServer.command}>
          Add Server
        </button>
      </div>
    </div>
  </div>
{/if}

<style>
  .panel-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 16px;
  }

  .panel-header h3 {
    margin: 0;
    font-size: 15px;
    font-weight: 600;
    color: var(--text-primary, #e8eaf0);
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }

  .header-actions { display: flex; gap: 8px; }

  .alert-error {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 10px 12px;
    background: rgba(255, 59, 59, 0.1);
    border: 1px solid rgba(255, 59, 59, 0.25);
    border-radius: 6px;
    color: #ff3b3b;
    font-size: 12px;
    margin-bottom: 12px;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }

  .alert-success {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 10px 12px;
    background: rgba(0, 200, 150, 0.08);
    border: 1px solid rgba(0, 200, 150, 0.2);
    border-radius: 6px;
    color: #00c896;
    font-size: 12px;
    margin-bottom: 12px;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }

  .info-box {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 10px 12px;
    background: rgba(0, 212, 255, 0.06);
    border: 1px solid rgba(0, 212, 255, 0.15);
    border-radius: 6px;
    font-size: 12px;
    color: rgba(0, 212, 255, 0.8);
    margin-bottom: 16px;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }

  .servers-list { display: flex; flex-direction: column; gap: 8px; }

  .server-item {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    padding: 12px 14px;
    background: rgba(8, 13, 20, 0.35);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 8px;
    transition: border-color 0.15s;
  }

  .server-item:hover { border-color: rgba(255, 255, 255, 0.12); }

  .server-icon {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 34px;
    height: 34px;
    background: rgba(0, 200, 150, 0.1);
    border: 1px solid rgba(0, 200, 150, 0.15);
    border-radius: 7px;
    color: #00c896;
    flex-shrink: 0;
  }

  .server-info { flex: 1; min-width: 0; }

  .server-name {
    font-size: 13px;
    font-weight: 500;
    color: #e8eaf0;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    margin-bottom: 2px;
  }

  .server-desc {
    font-size: 11px;
    color: rgba(255, 255, 255, 0.4);
    margin-bottom: 4px;
  }

  .server-command code {
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    font-size: 10px;
    color: rgba(255, 255, 255, 0.3);
    background: rgba(255, 255, 255, 0.04);
    padding: 2px 6px;
    border-radius: 3px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 260px;
    display: inline-block;
  }

  .server-status { flex-shrink: 0; }

  .status-badge {
    display: inline-flex;
    align-items: center;
    padding: 3px 8px;
    border-radius: 10px;
    font-size: 10px;
    font-weight: 600;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }

  .status-badge.running {
    background: rgba(0, 200, 150, 0.15);
    color: #00c896;
    border: 1px solid rgba(0, 200, 150, 0.25);
  }

  .status-badge.stopped {
    background: rgba(255, 255, 255, 0.05);
    color: rgba(255, 255, 255, 0.3);
    border: 1px solid rgba(255, 255, 255, 0.07);
  }

  .status-badge.error {
    background: rgba(255, 59, 59, 0.12);
    color: #ff3b3b;
    border: 1px solid rgba(255, 59, 59, 0.2);
  }

  .server-actions { display: flex; gap: 4px; flex-shrink: 0; }

  .empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 40px;
    color: rgba(255, 255, 255, 0.25);
    gap: 12px;
    text-align: center;
  }

  .empty-state p { margin: 0; font-size: 13px; }

  .icon-btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 28px;
    height: 28px;
    border: none;
    border-radius: 5px;
    background: rgba(255, 255, 255, 0.04);
    color: rgba(255, 255, 255, 0.4);
    cursor: pointer;
    transition: all 0.15s;
  }

  .icon-btn:hover { background: rgba(255, 255, 255, 0.1); color: #e8eaf0; }

  .icon-btn.accent {
    background: rgba(0, 212, 255, 0.12);
    color: #00d4ff;
    border: 1px solid rgba(0, 212, 255, 0.2);
  }

  .icon-btn.accent:hover { background: rgba(0, 212, 255, 0.2); }

  .icon-btn.running { color: #00c896; }
  .icon-btn.running:hover { background: rgba(255, 59, 59, 0.1); color: #ff3b3b; }

  .icon-btn.danger:hover { background: rgba(255, 59, 59, 0.12); color: #ff3b3b; }

  .icon-btn:disabled { opacity: 0.45; cursor: not-allowed; }

  /* Modal */
  .modal-backdrop {
    position: fixed;
    inset: 0;
    background: rgba(0, 0, 0, 0.65);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 1000;
    backdrop-filter: blur(4px);
  }

  .modal {
    background: rgba(8, 13, 20, 0.97);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 10px;
    width: 100%;
    max-width: 500px;
    max-height: 88vh;
    overflow-y: auto;
    box-shadow: 0 24px 48px rgba(0, 0, 0, 0.5);
  }

  .modal-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 14px 18px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.07);
    position: sticky;
    top: 0;
    background: rgba(8, 13, 20, 0.97);
  }

  .modal-header h4 {
    margin: 0;
    font-size: 13px;
    font-weight: 600;
    color: #e8eaf0;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }

  .modal-body {
    padding: 18px;
    display: flex;
    flex-direction: column;
    gap: 14px;
  }

  .modal-footer {
    display: flex;
    justify-content: flex-end;
    gap: 10px;
    padding: 14px 18px;
    border-top: 1px solid rgba(255, 255, 255, 0.07);
    position: sticky;
    bottom: 0;
    background: rgba(8, 13, 20, 0.97);
  }

  .quick-add-section { display: flex; flex-direction: column; gap: 10px; }

  .subsection-title {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: rgba(255, 255, 255, 0.4);
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }

  .template-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 6px;
  }

  .template-card {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 10px;
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 6px;
    color: rgba(255, 255, 255, 0.5);
    font-size: 11px;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    cursor: pointer;
    text-align: left;
    transition: all 0.15s;
  }

  .template-card:hover {
    border-color: rgba(0, 212, 255, 0.3);
    color: #00d4ff;
    background: rgba(0, 212, 255, 0.06);
  }

  .divider {
    display: flex;
    align-items: center;
    gap: 10px;
    color: rgba(255, 255, 255, 0.25);
    font-size: 11px;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }

  .divider::before,
  .divider::after {
    content: '';
    flex: 1;
    height: 1px;
    background: rgba(255, 255, 255, 0.07);
  }

  .form-group { display: flex; flex-direction: column; gap: 5px; }

  .form-group label {
    font-size: 11px;
    font-weight: 500;
    color: rgba(255, 255, 255, 0.45);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }

  .hint {
    font-size: 10px;
    color: rgba(255, 255, 255, 0.3);
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }

  .text-input {
    width: 100%;
    padding: 8px 12px;
    background: rgba(8, 13, 20, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 6px;
    color: #e8eaf0;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    font-size: 12px;
    box-sizing: border-box;
    transition: border-color 0.15s, box-shadow 0.15s;
  }

  .text-input:focus {
    outline: none;
    border-color: rgba(0, 212, 255, 0.5);
    box-shadow: 0 0 0 3px rgba(0, 212, 255, 0.1);
  }

  .textarea { min-height: 72px; resize: vertical; }

  .btn {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 7px 14px;
    border-radius: 6px;
    font-size: 12px;
    font-weight: 600;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    cursor: pointer;
    border: none;
    transition: all 0.15s;
  }

  .btn.primary {
    background: rgba(0, 212, 255, 0.15);
    border: 1px solid rgba(0, 212, 255, 0.4);
    color: #00d4ff;
  }

  .btn.primary:hover { background: rgba(0, 212, 255, 0.25); }
  .btn.primary:disabled { opacity: 0.45; cursor: not-allowed; }

  .btn.secondary {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    color: rgba(255, 255, 255, 0.65);
  }

  .btn.secondary:hover { background: rgba(255, 255, 255, 0.09); color: #fff; }

  .spinning { animation: spin 1s linear infinite; }

  @keyframes spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
  }
</style>
