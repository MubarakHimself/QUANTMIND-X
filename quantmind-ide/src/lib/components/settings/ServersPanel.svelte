<script lang="ts">
  import { onMount } from 'svelte';
  import {
    Server, RefreshCw, Trash2, AlertCircle, Plus,
    Save, Check, X, Zap, Shield
  } from 'lucide-svelte';
  import { apiFetch } from '$lib/api';

  interface ServerConfig {
    id: string;
    name: string;
    server_type: string;
    host: string;
    port: number;
    is_active: boolean;
    is_primary: boolean;
    metadata?: Record<string, unknown>;
  }

  const SERVER_TYPES = [
    { id: 'cloudzy', name: 'node_trading (MT5 Trading)', description: 'Live trading server' },
    { id: 'contabo', name: 'node_backend', description: 'Agent/compute server' },
    { id: 'mt5', name: 'MT5 Connection', description: 'MetaTrader 5 gateway' },
  ];

  let servers: ServerConfig[] = $state([]);
  let editingServer: string | null = $state(null);
  let isSaving = $state(false);
  let isLoading = $state(false);
  let isTesting: Record<string, boolean> = $state({});
  let testResults: Record<string, { success: boolean; latency_ms?: number; error?: string }> = $state({});
  let error = $state<string | null>(null);

  let showAddModal = $state(false);
  let newServerForm = $state({
    name: '', server_type: 'contabo', host: '', port: 22,
    username: '', password: '', ssh_key_path: '', api_key: '',
    is_active: true, is_primary: false
  });

  let editForm: Record<string, Partial<ServerConfig>> = $state({});
  let showDeleteConfirm = $state(false);
  let deletingServerId = $state('');

  async function loadServers() {
    isLoading = true;
    error = null;
    try {
      const data = await apiFetch<{ servers: ServerConfig[] }>('/api/servers');
      servers = data.servers || [];
    } catch (e) {
      error = 'Failed to load servers';
      console.error(e);
    } finally {
      isLoading = false;
    }
  }

  function getServerTypeInfo(type: string) {
    return SERVER_TYPES.find(s => s.id === type) || { id: type, name: type, description: '' };
  }

  function startEditing(serverId: string) {
    const server = servers.find(s => s.id === serverId);
    if (server) editForm[serverId] = { ...server };
    editingServer = serverId;
  }

  function cancelEditing(serverId: string) {
    editingServer = null;
    delete editForm[serverId];
    loadServers();
  }

  async function saveServer(serverId: string) {
    isSaving = true;
    const form = editForm[serverId];
    if (!form) { isSaving = false; return; }
    try {
      await apiFetch(`/api/servers/${serverId}`, {
        method: 'PUT',
        body: JSON.stringify({
          name: form.name, server_type: form.server_type,
          host: form.host, port: form.port,
          is_active: form.is_active, is_primary: form.is_primary
        })
      });
      editingServer = null;
      await loadServers();
    } catch (e) {
      error = 'Failed to save server';
      console.error(e);
    } finally {
      isSaving = false;
    }
  }

  async function testServer(serverId: string) {
    isTesting[serverId] = true;
    testResults[serverId] = { success: false };
    try {
      const result = await apiFetch<{ success: boolean; latency_ms?: number; error?: string }>(
        `/api/servers/${serverId}/health`
      );
      testResults[serverId] = result;
    } catch (e) {
      testResults[serverId] = { success: false, error: String(e) };
    } finally {
      isTesting[serverId] = false;
    }
  }

  function confirmDelete(serverId: string) {
    deletingServerId = serverId;
    showDeleteConfirm = true;
  }

  async function deleteServer() {
    if (!deletingServerId) return;
    try {
      await apiFetch(`/api/servers/${deletingServerId}`, { method: 'DELETE' });
      await loadServers();
    } catch (e) {
      error = 'Failed to delete server';
      console.error(e);
    } finally {
      showDeleteConfirm = false;
      deletingServerId = '';
    }
  }

  async function addServer() {
    isSaving = true;
    error = null;
    try {
      await apiFetch('/api/servers', {
        method: 'POST',
        body: JSON.stringify({
          name: newServerForm.name,
          server_type: newServerForm.server_type,
          host: newServerForm.host,
          port: newServerForm.port,
          username: newServerForm.username || undefined,
          password: newServerForm.password || undefined,
          ssh_key_path: newServerForm.ssh_key_path || undefined,
          api_key: newServerForm.api_key || undefined,
          is_active: newServerForm.is_active,
          is_primary: newServerForm.is_primary
        })
      });
      showAddModal = false;
      newServerForm = { name: '', server_type: 'contabo', host: '', port: 22, username: '', password: '', ssh_key_path: '', api_key: '', is_active: true, is_primary: false };
      await loadServers();
    } catch (e) {
      error = 'Failed to add server';
      console.error(e);
    } finally {
      isSaving = false;
    }
  }

  onMount(() => { loadServers(); });
</script>

<div class="panel">
  <div class="panel-header">
    <h3>Server Connections</h3>
    <div class="header-actions">
      <button class="icon-btn" onclick={loadServers} title="Refresh">
        <RefreshCw size={16} class={isLoading ? 'spinning' : ''} />
      </button>
      <button class="icon-btn accent" onclick={() => showAddModal = true} title="Add Server">
        <Plus size={16} />
      </button>
    </div>
  </div>

  {#if error}
    <div class="alert-error">
      <AlertCircle size={14} /> <span>{error}</span>
    </div>
  {/if}

  <div class="info-box">
    <AlertCircle size={14} />
    <span>Configure server connections for infrastructure nodes. Test connectivity after setup.</span>
  </div>

  {#if servers.length === 0 && !isLoading}
    <div class="empty-state">
      <Server size={40} />
      <p>No servers configured</p>
      <button class="btn primary" onclick={() => showAddModal = true}>
        <Plus size={14} /> Add Server
      </button>
    </div>
  {:else}
    <div class="servers-list">
      {#each servers as server}
        {@const typeInfo = getServerTypeInfo(server.server_type)}
        <div class="server-card" class:primary={server.is_primary}>
          <div class="server-header">
            <div class="server-info">
              <Server size={15} />
              <span class="server-name">{server.name}</span>
              {#if server.is_primary}
                <span class="primary-badge"><Shield size={11} /> Primary</span>
              {/if}
            </div>
            <span class="badge" class:active={server.is_active} class:inactive={!server.is_active}>
              {server.is_active ? 'Active' : 'Inactive'}
            </span>
          </div>

          {#if editingServer === server.id && editForm[server.id]}
            <div class="edit-fields">
              <div class="form-row">
                <div class="form-group">
                  <label>Name</label>
                  <input type="text" class="text-input" bind:value={editForm[server.id].name} />
                </div>
                <div class="form-group">
                  <label>Type</label>
                  <select class="text-input" bind:value={editForm[server.id].server_type}>
                    {#each SERVER_TYPES as t}
                      <option value={t.id}>{t.name}</option>
                    {/each}
                  </select>
                </div>
              </div>
              <div class="form-row">
                <div class="form-group">
                  <label>Host</label>
                  <input type="text" class="text-input" bind:value={editForm[server.id].host} />
                </div>
                <div class="form-group">
                  <label>Port</label>
                  <input type="number" class="text-input" bind:value={editForm[server.id].port} />
                </div>
              </div>
              <label class="toggle-label">
                <input type="checkbox" bind:checked={editForm[server.id].is_primary} />
                <span>Set as primary server of this type</span>
              </label>
            </div>
          {:else}
            <div class="server-details">
              <div class="detail-row">
                <span class="detail-label">Type</span>
                <span class="detail-value">{typeInfo.name}</span>
              </div>
              <div class="detail-row">
                <span class="detail-label">Host</span>
                <span class="detail-value mono">{server.host}:{server.port}</span>
              </div>
            </div>
          {/if}

          {#if testResults[server.id]}
            <div class="test-result" class:ok={testResults[server.id].success} class:fail={!testResults[server.id].success}>
              {#if testResults[server.id].success}
                <Check size={13} /> Reachable ({testResults[server.id].latency_ms}ms)
              {:else}
                <AlertCircle size={13} /> {testResults[server.id].error || 'Unreachable'}
              {/if}
            </div>
          {/if}

          <div class="server-actions">
            {#if editingServer === server.id}
              <button class="btn secondary" onclick={() => cancelEditing(server.id)}><X size={13} /> Cancel</button>
              <button class="btn primary" onclick={() => saveServer(server.id)} disabled={isSaving}>
                <Save size={13} /> {isSaving ? 'Saving...' : 'Save'}
              </button>
            {:else}
              <button class="btn secondary" onclick={() => testServer(server.id)} disabled={isTesting[server.id]}>
                <Zap size={13} /> {isTesting[server.id] ? 'Testing...' : 'Ping'}
              </button>
              <button class="btn secondary" onclick={() => startEditing(server.id)}>
                <Server size={13} /> Edit
              </button>
              <button class="btn danger" onclick={() => confirmDelete(server.id)} disabled={server.is_primary}>
                <Trash2 size={13} />
              </button>
            {/if}
          </div>
        </div>
      {/each}
    </div>
  {/if}
</div>

{#if showAddModal}
  <div class="modal-backdrop" onclick={() => showAddModal = false} role="button" tabindex="0"
    onkeydown={(e) => e.key === 'Escape' && (showAddModal = false)}>
    <div class="modal" onclick={(e) => e.stopPropagation()} role="dialog">
      <div class="modal-header">
        <h4>Add Server</h4>
        <button class="icon-btn" onclick={() => showAddModal = false}><X size={16} /></button>
      </div>
      <div class="modal-body">
        <div class="form-group">
          <label>Server Name</label>
          <input type="text" class="text-input" bind:value={newServerForm.name} placeholder="e.g. node_backend Node 1" />
        </div>
        <div class="form-group">
          <label>Server Type</label>
          <select class="text-input" bind:value={newServerForm.server_type}>
            {#each SERVER_TYPES as t}
              <option value={t.id}>{t.name} — {t.description}</option>
            {/each}
          </select>
        </div>
        <div class="form-row">
          <div class="form-group">
            <label>Host</label>
            <input type="text" class="text-input" bind:value={newServerForm.host} placeholder="IP or hostname" />
          </div>
          <div class="form-group">
            <label>Port</label>
            <input type="number" class="text-input" bind:value={newServerForm.port} />
          </div>
        </div>
        <div class="form-group">
          <label>Username <span class="optional">(optional)</span></label>
          <input type="text" class="text-input" bind:value={newServerForm.username} placeholder="SSH username" />
        </div>
        <div class="form-group">
          <label>Password <span class="optional">(optional)</span></label>
          <input type="password" class="text-input" bind:value={newServerForm.password} placeholder="SSH password" />
        </div>
        <div class="form-group">
          <label>SSH Key Path <span class="optional">(optional)</span></label>
          <input type="text" class="text-input" bind:value={newServerForm.ssh_key_path} placeholder="/path/to/key" />
        </div>
        <div class="form-group">
          <label>API Key <span class="optional">(optional)</span></label>
          <input type="password" class="text-input" bind:value={newServerForm.api_key} placeholder="API key" />
        </div>
        <div class="toggle-row">
          <label class="toggle-label">
            <input type="checkbox" bind:checked={newServerForm.is_active} />
            <span>Active</span>
          </label>
          <label class="toggle-label">
            <input type="checkbox" bind:checked={newServerForm.is_primary} />
            <span>Primary of this type</span>
          </label>
        </div>
      </div>
      <div class="modal-footer">
        <button class="btn secondary" onclick={() => showAddModal = false}>Cancel</button>
        <button class="btn primary" onclick={addServer} disabled={isSaving || !newServerForm.name || !newServerForm.host}>
          {isSaving ? 'Adding...' : 'Add Server'}
        </button>
      </div>
    </div>
  </div>
{/if}

{#if showDeleteConfirm}
  <div class="modal-backdrop" onclick={() => showDeleteConfirm = false} role="button" tabindex="0"
    onkeydown={(e) => e.key === 'Escape' && (showDeleteConfirm = false)}>
    <div class="modal" onclick={(e) => e.stopPropagation()} role="dialog">
      <div class="modal-header"><h4>Delete Server</h4></div>
      <div class="modal-body">
        <p class="confirm-text">Remove this server? This cannot be undone.</p>
      </div>
      <div class="modal-footer">
        <button class="btn secondary" onclick={() => showDeleteConfirm = false}>Cancel</button>
        <button class="btn danger" onclick={deleteServer}>Delete</button>
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
    color: var(--text-primary);
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
  }

  .empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 40px;
    color: rgba(255, 255, 255, 0.3);
    gap: 14px;
    text-align: center;
  }

  .empty-state p { margin: 0; font-size: 13px; }

  .servers-list { display: flex; flex-direction: column; gap: 10px; }

  .server-card {
    background: rgba(8, 13, 20, 0.92);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 8px;
    padding: 14px;
    transition: border-color 0.15s;
  }

  .server-card.primary { border-color: rgba(0, 200, 150, 0.25); }

  .server-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 10px;
  }

  .server-info {
    display: flex;
    align-items: center;
    gap: 8px;
    color: var(--text-primary);
  }

  .server-name {
    font-size: 13px;
    font-weight: 500;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }

  .primary-badge {
    display: inline-flex;
    align-items: center;
    gap: 3px;
    padding: 2px 6px;
    border-radius: 8px;
    font-size: 10px;
    font-weight: 500;
    background: rgba(0, 200, 150, 0.15);
    color: #00c896;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }

  .badge {
    display: inline-flex;
    padding: 3px 8px;
    border-radius: 10px;
    font-size: 11px;
    font-weight: 500;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }

  .badge.active {
    background: rgba(0, 200, 150, 0.15);
    color: #00c896;
    border: 1px solid rgba(0, 200, 150, 0.25);
  }

  .badge.inactive {
    background: rgba(255, 255, 255, 0.05);
    color: rgba(255, 255, 255, 0.4);
    border: 1px solid rgba(255, 255, 255, 0.08);
  }

  .server-details { margin-bottom: 10px; }

  .detail-row {
    display: flex;
    gap: 10px;
    font-size: 12px;
    margin-bottom: 3px;
  }

  .detail-label {
    color: rgba(255, 255, 255, 0.35);
    width: 40px;
    flex-shrink: 0;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }

  .detail-value { color: rgba(255, 255, 255, 0.65); }
  .detail-value.mono { font-family: 'JetBrains Mono', 'Fira Code', monospace; }

  .edit-fields { margin-bottom: 10px; }

  .form-row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
    margin-bottom: 10px;
  }

  .form-group { display: flex; flex-direction: column; gap: 5px; }

  .form-group label {
    font-size: 11px;
    font-weight: 500;
    color: rgba(255, 255, 255, 0.4);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }

  .optional { font-weight: 400; opacity: 0.6; text-transform: none; }

  .text-input {
    width: 100%;
    padding: 7px 10px;
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 6px;
    color: var(--text-primary);
    font-size: 12px;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    box-sizing: border-box;
    transition: border-color 0.15s, box-shadow 0.15s;
  }

  .text-input:focus {
    outline: none;
    border-color: rgba(0, 212, 255, 0.5);
    box-shadow: 0 0 0 2px rgba(0, 212, 255, 0.15);
  }

  select.text-input { cursor: pointer; }

  .toggle-label {
    display: flex;
    align-items: center;
    gap: 6px;
    cursor: pointer;
    font-size: 12px;
    color: rgba(255, 255, 255, 0.55);
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }

  .toggle-label input { accent-color: #00d4ff; }

  .toggle-row {
    display: flex;
    gap: 20px;
    margin-top: 8px;
  }

  .test-result {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 7px 10px;
    border-radius: 6px;
    font-size: 12px;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    margin-bottom: 10px;
  }

  .test-result.ok {
    background: rgba(0, 200, 150, 0.08);
    border: 1px solid rgba(0, 200, 150, 0.2);
    color: #00c896;
  }

  .test-result.fail {
    background: rgba(255, 59, 59, 0.08);
    border: 1px solid rgba(255, 59, 59, 0.2);
    color: #ff3b3b;
  }

  .server-actions {
    display: flex;
    justify-content: flex-end;
    gap: 6px;
    padding-top: 10px;
    border-top: 1px solid rgba(255, 255, 255, 0.06);
  }

  .btn {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 5px 10px;
    border: none;
    border-radius: 5px;
    font-size: 12px;
    font-weight: 500;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    cursor: pointer;
    transition: all 0.15s;
  }

  .btn.primary { background: #00d4ff; color: #080d14; }
  .btn.primary:hover { background: #00bce6; }
  .btn.primary:disabled { opacity: 0.45; cursor: not-allowed; }

  .btn.secondary {
    background: rgba(255, 255, 255, 0.06);
    color: rgba(255, 255, 255, 0.7);
    border: 1px solid rgba(255, 255, 255, 0.1);
  }

  .btn.secondary:hover { background: rgba(255, 255, 255, 0.1); color: #fff; }
  .btn.secondary:disabled { opacity: 0.45; cursor: not-allowed; }

  .btn.danger {
    background: rgba(255, 59, 59, 0.12);
    color: #ff3b3b;
    border: 1px solid rgba(255, 59, 59, 0.2);
  }

  .btn.danger:hover { background: rgba(255, 59, 59, 0.22); }
  .btn.danger:disabled { opacity: 0.45; cursor: not-allowed; }

  .icon-btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 30px;
    height: 30px;
    border: none;
    border-radius: 6px;
    background: rgba(255, 255, 255, 0.04);
    color: rgba(255, 255, 255, 0.5);
    cursor: pointer;
    transition: all 0.15s;
  }

  .icon-btn:hover { background: rgba(255, 255, 255, 0.1); color: var(--text-primary); }

  .icon-btn.accent {
    background: rgba(0, 212, 255, 0.12);
    color: #00d4ff;
    border: 1px solid rgba(0, 212, 255, 0.2);
  }

  .icon-btn.accent:hover { background: rgba(0, 212, 255, 0.2); }

  .spinning { animation: spin 1s linear infinite; }

  @keyframes spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
  }

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
    max-height: 90vh;
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
    font-size: 14px;
    font-weight: 600;
    color: var(--text-primary);
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
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

  .confirm-text { margin: 0; font-size: 13px; color: rgba(255, 255, 255, 0.6); }
</style>
