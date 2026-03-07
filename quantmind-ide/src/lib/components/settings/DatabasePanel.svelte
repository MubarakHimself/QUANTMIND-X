<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import { Database, Server, HardDrive, Cpu } from 'lucide-svelte';

  export let dbSettings = {
    connectionType: 'sqlite',
    databaseUrl: '',
    sqlitePath: './data/quantmind.db',
    duckdbPath: './data/analytics.duckdb',
    autoBackup: true,
    backupInterval: 3600,
    maxBackups: 10
  };

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
        on:click={() => updateDbSettings('connectionType', 'sqlite')}
      >
        <Database size={16} />
        <span>SQLite</span>
      </button>
      <button
        class="connection-type-btn"
        class:active={dbSettings.connectionType === 'postgresql'}
        on:click={() => updateDbSettings('connectionType', 'postgresql')}
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
          on:input={(e) => updateDbSettings('databaseUrl', e.currentTarget.value)}
        />
      </div>
    </div>
  {/if}

  {#if dbSettings.connectionType === 'sqlite'}
    <div class="setting-group">
      <label>SQLite (Transactional)</label>
      <div class="setting-row">
        <span>Path</span>
        <input type="text" bind:value={dbSettings.sqlitePath} class="text-input" on:input={(e) => updateDbSettings('sqlitePath', e.currentTarget.value)} />
      </div>
    </div>
  {/if}

  <div class="setting-group">
    <label>DuckDB (Analytics)</label>
    <div class="setting-row">
      <span>Path</span>
      <input type="text" bind:value={dbSettings.duckdbPath} class="text-input" on:input={(e) => updateDbSettings('duckdbPath', e.currentTarget.value)} />
    </div>
  </div>

  <div class="setting-group">
    <label>Backup</label>
    <div class="setting-row">
      <span>Auto Backup</span>
      <label class="switch">
        <input type="checkbox" bind:checked={dbSettings.autoBackup} on:change={(e) => updateDbSettings('autoBackup', e.currentTarget.checked)} />
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
          on:input={(e) => updateDbSettings('backupInterval', parseInt(e.currentTarget.value))}
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
          on:input={(e) => updateDbSettings('maxBackups', parseInt(e.currentTarget.value))}
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
