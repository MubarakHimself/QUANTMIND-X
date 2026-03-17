<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import { Database, Server, HardDrive, Cpu } from 'lucide-svelte';

  let { dbSettings = $bindable({
    connectionType: 'sqlite',
    databaseUrl: '',
    sqlitePath: './data/quantmind.db',
    duckdbPath: './data/analytics.duckdb',
    autoBackup: true,
    backupInterval: 3600,
    maxBackups: 10
  }) } = $props();

  const dispatch = createEventDispatcher();

  function updateDbSettings(field: string, value: any) {
    dispatch('updateDbSettings', { field, value });
  }
</script>

<div class="panel">
  <h3>Database Configuration</h3>

  <div class="setting-group">
    <label>Connection Type</label>
    <div class="connection-type-selector">
      <button
        class="connection-type-btn"
        class:active={dbSettings.connectionType === 'sqlite'}
        onclick={() => updateDbSettings('connectionType', 'sqlite')}
      >
        <Database size={16} />
        <span>SQLite</span>
      </button>
      <button
        class="connection-type-btn"
        class:active={dbSettings.connectionType === 'postgresql'}
        onclick={() => updateDbSettings('connectionType', 'postgresql')}
      >
        <Server size={16} />
        <span>PostgreSQL</span>
      </button>
    </div>
  </div>

  {#if dbSettings.connectionType === 'postgresql'}
    <div class="setting-group">
      <label>PostgreSQL Connection</label>
      <div class="setting-row">
        <span>Database URL</span>
        <input
          type="text"
          bind:value={dbSettings.databaseUrl}
          class="text-input"
          placeholder="postgresql://user:pass@host:5432/dbname"
          oninput={(e) => updateDbSettings('databaseUrl', e.currentTarget.value)}
        />
      </div>
    </div>
  {/if}

  {#if dbSettings.connectionType === 'sqlite'}
    <div class="setting-group">
      <label>SQLite (Transactional)</label>
      <div class="setting-row">
        <span>Path</span>
        <input type="text" bind:value={dbSettings.sqlitePath} class="text-input" oninput={(e) => updateDbSettings('sqlitePath', e.currentTarget.value)} />
      </div>
    </div>
  {/if}

  <div class="setting-group">
    <label>DuckDB (Analytics)</label>
    <div class="setting-row">
      <span>Path</span>
      <input type="text" bind:value={dbSettings.duckdbPath} class="text-input" oninput={(e) => updateDbSettings('duckdbPath', e.currentTarget.value)} />
    </div>
  </div>

  <div class="setting-group">
    <label>Backup</label>
    <div class="setting-row">
      <span>Auto Backup</span>
      <label class="switch">
        <input type="checkbox" bind:checked={dbSettings.autoBackup} onchange={(e) => updateDbSettings('autoBackup', e.currentTarget.checked)} />
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
          bind:value={dbSettings.backupInterval}
          class="number-input"
          oninput={(e) => updateDbSettings('backupInterval', parseInt(e.currentTarget.value))}
        />
      </div>
      <div class="setting-row">
        <span>Max Backups</span>
        <input
          type="number"
          min="1"
          max="50"
          bind:value={dbSettings.maxBackups}
          class="number-input"
          oninput={(e) => updateDbSettings('maxBackups', parseInt(e.currentTarget.value))}
        />
      </div>
    {/if}
  </div>

  <div class="setting-group">
    <label>Storage Info</label>
    <div class="storage-info">
      <div class="storage-item">
        <HardDrive size={16} />
        <div class="storage-details">
          <span class="label">SQLite Database</span>
          <span class="size">~2.4 MB</span>
        </div>
      </div>
      <div class="storage-item">
        <Cpu size={16} />
        <div class="storage-details">
          <span class="label">DuckDB Analytics</span>
          <span class="size">~15.8 MB</span>
        </div>
      </div>
    </div>
  </div>
</div>

<style>
  /* Panel Header */
  .panel h3 {
    margin: 0 0 20px;
    font-size: 16px;
    font-weight: 600;
    color: var(--text-primary);
  }

  /* Setting Group */
  .setting-group {
    background: var(--bg-tertiary);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 12px;
  }

  .setting-group:last-child {
    margin-bottom: 0;
  }

  .setting-group > label {
    display: block;
    font-size: 13px;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 12px;
  }

  /* Connection Type Selector */
  .connection-type-selector {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 10px;
  }

  .connection-type-btn {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 8px;
    padding: 16px;
    background: var(--bg-primary);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
    color: var(--text-secondary);
    font-size: 13px;
    cursor: pointer;
    transition: all 0.15s;
  }

  .connection-type-btn:hover {
    border-color: var(--text-muted);
    color: var(--text-primary);
  }

  .connection-type-btn.active {
    background: rgba(99, 102, 241, 0.1);
    border-color: var(--accent-primary);
    color: var(--accent-primary);
  }

  .connection-type-btn :global(svg) {
    color: inherit;
  }

  /* Setting Row */
  .setting-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 0;
  }

  .setting-row:not(:last-child) {
    border-bottom: 1px solid var(--border-subtle);
  }

  .setting-row span {
    font-size: 13px;
    color: var(--text-secondary);
  }

  /* Inputs */
  .text-input,
  .number-input {
    padding: 8px 12px;
    background: var(--bg-primary);
    border: 1px solid var(--border-subtle);
    border-radius: 6px;
    color: var(--text-primary);
    font-size: 13px;
    min-width: 200px;
    transition: all 0.15s;
  }

  .text-input:focus,
  .number-input:focus {
    outline: none;
    border-color: var(--accent-primary);
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.15);
  }

  .text-input::placeholder {
    color: var(--text-muted);
    opacity: 0.6;
  }

  /* Toggle Switch */
  .switch {
    position: relative;
    display: inline-block;
    width: 44px;
    height: 24px;
  }

  .switch input {
    opacity: 0;
    width: 0;
    height: 0;
  }

  .slider {
    position: absolute;
    cursor: pointer;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background-color: var(--bg-primary);
    border: 1px solid var(--border-subtle);
    transition: 0.2s;
    border-radius: 24px;
  }

  .slider:before {
    position: absolute;
    content: "";
    height: 18px;
    width: 18px;
    left: 2px;
    bottom: 2px;
    background-color: var(--text-muted);
    transition: 0.2s;
    border-radius: 50%;
  }

  input:checked + .slider {
    background-color: var(--accent-primary);
    border-color: var(--accent-primary);
  }

  input:checked + .slider:before {
    transform: translateX(20px);
    background-color: white;
  }

  /* Storage Info */
  .storage-info {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .storage-item {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 12px;
    background: var(--bg-primary);
    border-radius: 6px;
  }

  .storage-item :global(svg) {
    color: var(--accent-primary);
  }

  .storage-details {
    display: flex;
    justify-content: space-between;
    flex: 1;
    align-items: center;
  }

  .storage-details .label {
    font-size: 13px;
    color: var(--text-primary);
  }

  .storage-details .size {
    font-size: 12px;
    color: var(--text-muted);
    font-family: monospace;
  }
</style>
