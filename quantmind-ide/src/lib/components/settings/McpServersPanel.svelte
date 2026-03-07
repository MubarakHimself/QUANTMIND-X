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
