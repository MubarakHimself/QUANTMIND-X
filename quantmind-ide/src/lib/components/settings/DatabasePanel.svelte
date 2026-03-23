<script lang="ts">
  import { onMount } from 'svelte';
  import { Database, Server, HardDrive, Cpu, Save, RefreshCw, Check, AlertCircle } from 'lucide-svelte';
  import { apiFetch } from '$lib/api';

  interface DbSettings {
    connectionType: 'sqlite' | 'postgresql';
    databaseUrl: string;
    sqlitePath: string;
    duckdbPath: string;
    autoBackup: boolean;
    backupInterval: number;
    maxBackups: number;
  }

  interface StorageInfo {
    sqlite_size_mb: number;
    duckdb_size_mb: number;
  }

  let dbSettings: DbSettings = $state({
    connectionType: 'sqlite',
    databaseUrl: '',
    sqlitePath: './data/quantmind.db',
    duckdbPath: './data/analytics.duckdb',
    autoBackup: true,
    backupInterval: 3600,
    maxBackups: 10
  });

  let storageInfo: StorageInfo | null = $state(null);
  let isLoading = $state(false);
  let isSaving = $state(false);
  let error = $state<string | null>(null);
  let success = $state<string | null>(null);

  async function loadSettings() {
    isLoading = true;
    error = null;
    try {
      const data = await apiFetch<{ settings: DbSettings; storage: StorageInfo }>('/api/settings/database');
      dbSettings = { ...dbSettings, ...data.settings };
      storageInfo = data.storage || null;
    } catch (e) {
      console.warn('Database settings endpoint unavailable, using defaults');
    } finally {
      isLoading = false;
    }
  }

  async function saveSettings() {
    isSaving = true;
    error = null;
    success = null;
    try {
      await apiFetch('/api/settings/database', {
        method: 'POST',
        body: JSON.stringify(dbSettings)
      });
      success = 'Database settings saved';
      setTimeout(() => success = null, 3000);
    } catch (e) {
      error = 'Failed to save database settings';
      console.error(e);
    } finally {
      isSaving = false;
    }
  }

  onMount(() => { loadSettings(); });
</script>

<div class="panel">
  <div class="panel-header">
    <h3>Database Configuration</h3>
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

  <!-- Connection Type -->
  <div class="settings-section">
    <div class="section-title">Connection Type</div>
    <div class="type-selector">
      <button
        class="type-btn"
        class:active={dbSettings.connectionType === 'sqlite'}
        onclick={() => dbSettings.connectionType = 'sqlite'}
      >
        <Database size={15} />
        <span>SQLite</span>
      </button>
      <button
        class="type-btn"
        class:active={dbSettings.connectionType === 'postgresql'}
        onclick={() => dbSettings.connectionType = 'postgresql'}
      >
        <Server size={15} />
        <span>PostgreSQL</span>
      </button>
    </div>
  </div>

  <!-- Connection Details -->
  <div class="settings-section">
    <div class="section-title">
      {dbSettings.connectionType === 'postgresql' ? 'PostgreSQL' : 'SQLite'} (Transactional)
    </div>

    {#if dbSettings.connectionType === 'postgresql'}
      <div class="form-group">
        <label>Database URL</label>
        <input
          type="text"
          class="text-input"
          placeholder="postgresql://user:pass@host:5432/dbname"
          bind:value={dbSettings.databaseUrl}
        />
      </div>
    {:else}
      <div class="form-group">
        <label>SQLite Path</label>
        <input
          type="text"
          class="text-input"
          placeholder="./data/quantmind.db"
          bind:value={dbSettings.sqlitePath}
        />
      </div>
    {/if}

    <div class="form-group">
      <label>DuckDB Path (Analytics)</label>
      <input
        type="text"
        class="text-input"
        placeholder="./data/analytics.duckdb"
        bind:value={dbSettings.duckdbPath}
      />
    </div>
  </div>

  <!-- Backup Settings -->
  <div class="settings-section">
    <div class="section-title">Backup</div>

    <div class="setting-row">
      <span>Auto Backup</span>
      <label class="switch">
        <input type="checkbox" bind:checked={dbSettings.autoBackup} />
        <span class="slider"></span>
      </label>
    </div>

    {#if dbSettings.autoBackup}
      <div class="setting-row">
        <span>Interval (seconds)</span>
        <input
          type="number"
          min="300"
          max="86400"
          class="number-input"
          bind:value={dbSettings.backupInterval}
        />
      </div>
      <div class="setting-row">
        <span>Max Backups</span>
        <input
          type="number"
          min="1"
          max="50"
          class="number-input"
          bind:value={dbSettings.maxBackups}
        />
      </div>
    {/if}
  </div>

  <!-- Storage Info -->
  <div class="settings-section">
    <div class="section-title">Storage Info</div>
    <div class="storage-list">
      <div class="storage-item">
        <div class="storage-icon">
          <HardDrive size={15} />
        </div>
        <div class="storage-details">
          <span class="storage-label">SQLite Database</span>
          <span class="storage-size">
            {storageInfo ? storageInfo.sqlite_size_mb.toFixed(1) + ' MB' : '—'}
          </span>
        </div>
      </div>
      <div class="storage-item">
        <div class="storage-icon">
          <Cpu size={15} />
        </div>
        <div class="storage-details">
          <span class="storage-label">DuckDB Analytics</span>
          <span class="storage-size">
            {storageInfo ? storageInfo.duckdb_size_mb.toFixed(1) + ' MB' : '—'}
          </span>
        </div>
      </div>
    </div>
  </div>

  <div class="action-row">
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

  .type-selector {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
  }

  .type-btn {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 8px;
    padding: 14px 12px;
    background: rgba(8, 13, 20, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 7px;
    color: rgba(255, 255, 255, 0.4);
    font-size: 12px;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    cursor: pointer;
    transition: all 0.15s;
  }

  .type-btn:hover {
    border-color: rgba(255, 255, 255, 0.18);
    color: rgba(255, 255, 255, 0.75);
  }

  .type-btn.active {
    background: rgba(0, 212, 255, 0.08);
    border-color: rgba(0, 212, 255, 0.35);
    color: #00d4ff;
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

  .setting-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 0;
    border-bottom: 1px solid rgba(255, 255, 255, 0.04);
    font-size: 12px;
    color: rgba(255, 255, 255, 0.6);
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }

  .setting-row:last-child { border-bottom: none; }

  .number-input {
    width: 90px;
    padding: 6px 10px;
    background: rgba(8, 13, 20, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 5px;
    color: #e8eaf0;
    font-size: 12px;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    text-align: right;
    transition: border-color 0.15s;
    -moz-appearance: textfield;
  }

  .number-input::-webkit-outer-spin-button,
  .number-input::-webkit-inner-spin-button { -webkit-appearance: none; margin: 0; }

  .number-input:focus {
    outline: none;
    border-color: rgba(0, 212, 255, 0.5);
  }

  /* Toggle Switch */
  .switch {
    position: relative;
    display: inline-block;
    width: 40px;
    height: 22px;
  }

  .switch input { opacity: 0; width: 0; height: 0; }

  .slider {
    position: absolute;
    cursor: pointer;
    inset: 0;
    background: rgba(255, 255, 255, 0.1);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 22px;
    transition: 0.2s;
  }

  .slider:before {
    position: absolute;
    content: "";
    height: 14px;
    width: 14px;
    left: 3px;
    bottom: 3px;
    background: rgba(255, 255, 255, 0.5);
    border-radius: 50%;
    transition: 0.2s;
  }

  input:checked + .slider {
    background: rgba(0, 212, 255, 0.25);
    border-color: rgba(0, 212, 255, 0.4);
  }

  input:checked + .slider:before {
    transform: translateX(18px);
    background: #00d4ff;
  }

  /* Storage Info */
  .storage-list { display: flex; flex-direction: column; gap: 8px; }

  .storage-item {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 10px 12px;
    background: rgba(8, 13, 20, 0.5);
    border-radius: 6px;
    border: 1px solid rgba(255, 255, 255, 0.05);
  }

  .storage-icon {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 28px;
    height: 28px;
    background: rgba(0, 212, 255, 0.08);
    border-radius: 5px;
    color: #00d4ff;
    flex-shrink: 0;
  }

  .storage-details {
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex: 1;
  }

  .storage-label {
    font-size: 12px;
    color: rgba(255, 255, 255, 0.6);
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }

  .storage-size {
    font-size: 12px;
    color: rgba(255, 255, 255, 0.35);
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
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

  .icon-btn:hover { background: rgba(255, 255, 255, 0.1); color: #e8eaf0; }
  .icon-btn:disabled { opacity: 0.45; cursor: not-allowed; }

  .spinning { animation: spin 1s linear infinite; }

  @keyframes spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
  }
</style>
