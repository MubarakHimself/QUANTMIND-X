<script lang="ts">
  import { onMount } from 'svelte';
  import { Server, Zap, Save, RefreshCw, Check, AlertCircle, Wifi } from 'lucide-svelte';
  import { apiFetch } from '$lib/api';

  interface ConnectionSettings {
    redisUrl: string;
    zmqEndpoint: string;
    mt5Login: string;
    mt5Password: string;
    mt5Server: string;
  }

  let settings: ConnectionSettings = $state({
    redisUrl: 'redis://localhost:6379',
    zmqEndpoint: 'tcp://localhost:5555',
    mt5Login: '',
    mt5Password: '',
    mt5Server: ''
  });

  let isLoading = $state(false);
  let isSaving = $state(false);
  let isTesting = $state(false);
  let error = $state<string | null>(null);
  let success = $state<string | null>(null);
  let testResult = $state<{ redis: boolean; zmq: boolean } | null>(null);

  async function loadSettings() {
    isLoading = true;
    error = null;
    try {
      const data = await apiFetch<ConnectionSettings>('/api/settings/connection');
      settings = { ...settings, ...data };
    } catch (e) {
      // Graceful degradation — use defaults if endpoint not yet available
      console.warn('Connection settings endpoint unavailable, using defaults');
    } finally {
      isLoading = false;
    }
  }

  async function saveSettings() {
    isSaving = true;
    error = null;
    success = null;
    try {
      await apiFetch('/api/settings/connection', {
        method: 'POST',
        body: JSON.stringify(settings)
      });
      success = 'Connection settings saved';
      setTimeout(() => success = null, 3000);
    } catch (e) {
      error = 'Failed to save connection settings';
      console.error(e);
    } finally {
      isSaving = false;
    }
  }

  async function testConnections() {
    isTesting = true;
    testResult = null;
    error = null;
    try {
      const result = await apiFetch<{ redis: boolean; zmq: boolean }>('/api/settings/connection/test', {
        method: 'POST',
        body: JSON.stringify({ redisUrl: settings.redisUrl, zmqEndpoint: settings.zmqEndpoint })
      });
      testResult = result;
    } catch (e) {
      error = 'Connection test failed';
      console.error(e);
    } finally {
      isTesting = false;
    }
  }

  onMount(() => { loadSettings(); });
</script>

<div class="panel">
  <div class="panel-header">
    <h3>Connection Settings</h3>
    <div class="header-actions">
      <button class="icon-btn" onclick={loadSettings} title="Refresh" disabled={isLoading}>
        <RefreshCw size={16} class={isLoading ? 'spinning' : ''} />
      </button>
    </div>
  </div>

  {#if error}
    <div class="alert-error"><AlertCircle size={14} /> <span>{error}</span></div>
  {/if}
  {#if success}
    <div class="alert-success"><Check size={14} /> <span>{success}</span></div>
  {/if}

  <!-- Infrastructure Connections -->
  <div class="settings-section">
    <div class="section-title">Infrastructure</div>

    <div class="form-group">
      <label>Redis URL</label>
      <input
        type="text"
        class="text-input"
        placeholder="redis://localhost:6379"
        bind:value={settings.redisUrl}
      />
      <span class="hint">Message broker and session store</span>
    </div>

    <div class="form-group">
      <label>ZMQ Endpoint</label>
      <input
        type="text"
        class="text-input"
        placeholder="tcp://localhost:5555"
        bind:value={settings.zmqEndpoint}
      />
      <span class="hint">Internal agent messaging bus</span>
    </div>

    {#if testResult}
      <div class="test-results">
        <div class="test-item" class:ok={testResult.redis} class:fail={!testResult.redis}>
          <Wifi size={12} />
          <span>Redis: {testResult.redis ? 'Connected' : 'Failed'}</span>
        </div>
        <div class="test-item" class:ok={testResult.zmq} class:fail={!testResult.zmq}>
          <Zap size={12} />
          <span>ZMQ: {testResult.zmq ? 'Connected' : 'Failed'}</span>
        </div>
      </div>
    {/if}
  </div>

  <!-- MetaTrader 5 -->
  <div class="settings-section">
    <div class="section-title">MetaTrader 5</div>

    <div class="form-group">
      <label>Login</label>
      <input
        type="text"
        class="text-input"
        placeholder="Account number"
        bind:value={settings.mt5Login}
      />
    </div>
    <div class="form-group">
      <label>Password</label>
      <input
        type="password"
        class="text-input"
        placeholder="••••••••"
        bind:value={settings.mt5Password}
      />
    </div>
    <div class="form-group">
      <label>Server</label>
      <input
        type="text"
        class="text-input"
        placeholder="BrokerName-Server"
        bind:value={settings.mt5Server}
      />
    </div>
  </div>

  <!-- Actions -->
  <div class="action-row">
    <button class="btn secondary" onclick={testConnections} disabled={isTesting}>
      <Zap size={14} />
      {isTesting ? 'Testing...' : 'Test Connections'}
    </button>
    <button class="btn primary" onclick={saveSettings} disabled={isSaving}>
      <Save size={14} />
      {isSaving ? 'Saving...' : 'Save Changes'}
    </button>
  </div>
</div>

<style>
  .panel-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 20px;
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

  .settings-section {
    background: rgba(8, 13, 20, 0.35);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 12px;
  }

  .section-title {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: rgba(255, 255, 255, 0.5);
    margin-bottom: 14px;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }

  .form-group {
    display: flex;
    flex-direction: column;
    gap: 5px;
    margin-bottom: 12px;
  }

  .form-group:last-child { margin-bottom: 0; }

  .form-group label {
    font-size: 11px;
    font-weight: 500;
    color: rgba(255, 255, 255, 0.45);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }

  .hint {
    font-size: 11px;
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

  .text-input::placeholder { color: rgba(255, 255, 255, 0.25); }

  .test-results {
    display: flex;
    gap: 10px;
    margin-top: 12px;
  }

  .test-item {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 5px 10px;
    border-radius: 5px;
    font-size: 11px;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }

  .test-item.ok {
    background: rgba(0, 200, 150, 0.1);
    border: 1px solid rgba(0, 200, 150, 0.2);
    color: #00c896;
  }

  .test-item.fail {
    background: rgba(255, 59, 59, 0.1);
    border: 1px solid rgba(255, 59, 59, 0.2);
    color: #ff3b3b;
  }

  .action-row {
    display: flex;
    justify-content: flex-end;
    gap: 10px;
    margin-top: 4px;
  }

  .btn {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 8px 14px;
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
  .btn.secondary:disabled { opacity: 0.45; cursor: not-allowed; }

  .icon-btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 30px;
    height: 30px;
    border: none;
    border-radius: 6px;
    background: rgba(255, 255, 255, 0.04);
    color: rgba(255, 255, 255, 0.4);
    cursor: pointer;
    transition: all 0.15s;
  }

  .icon-btn:hover { background: rgba(255, 255, 255, 0.1); color: var(--text-primary, #e8eaf0); }
  .icon-btn:disabled { opacity: 0.45; cursor: not-allowed; }

  .spinning { animation: spin 1s linear infinite; }

  @keyframes spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
  }
</style>
