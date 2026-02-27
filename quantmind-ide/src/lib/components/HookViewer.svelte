<script lang="ts">
  import { onMount } from 'svelte';
  import {
    GitBranch,
    Play,
    ToggleLeft,
    ToggleRight,
    RefreshCw,
    Trash2,
    FileText,
    Clock,
    CheckCircle,
    XCircle,
    Zap,
    X,
    AlertCircle,
    Settings,
    Filter
  } from 'lucide-svelte';
  import { hooksStore, type Hook, type HookLogEntry } from '$lib/stores/hooksStore';
  import * as memoryApi from '$lib/api/memory';

  export let onClose = () => {};

  let showLogs = true;
  let selectedHookName = '';
  let logFilter: 'all' | 'success' | 'failed' = 'all';

  // Subscribe to store
  $: hooks = $hooksStore.hooks;
  $: logs = $hooksStore.logs;
  $: loading = $hooksStore.loading;
  $: error = $hooksStore.error;
  $: selectedHook = hooks.find(h => h.name === selectedHookName) || null;

  // Filter logs by selected hook and status
  $: filteredLogs = logs.filter(log => {
    if (selectedHookName && log.hookName !== selectedHookName) return false;
    if (logFilter !== 'all' && log.status !== logFilter) return false;
    return true;
  });

  onMount(() => {
    loadHooks();
    loadLogs();
  });

  async function loadHooks() {
    hooksStore.setLoading(true);
    try {
      const hooks = await memoryApi.listHooks();
      hooksStore.setHooks(hooks);
    } catch (e) {
      hooksStore.setError(e instanceof Error ? e.message : 'Failed to load hooks');
    } finally {
      hooksStore.setLoading(false);
    }
  }

  async function loadLogs() {
    try {
      const logs = await memoryApi.getHookLogs(100);
      hooksStore.setLogs(logs);
    } catch (e) {
      hooksStore.setError(e instanceof Error ? e.message : 'Failed to load logs');
    }
  }

  async function handleToggleHook(hook: Hook) {
    try {
      const updated = await memoryApi.toggleHook(hook.name, !hook.enabled);
      hooksStore.updateHook(hook.name, { enabled: updated.enabled });
    } catch (e) {
      hooksStore.setError(e instanceof Error ? e.message : 'Failed to toggle hook');
    }
  }

  async function handleExecuteHook(hook: Hook) {
    try {
      await memoryApi.executeHook(hook.name);
      await loadLogs(); // Refresh logs
    } catch (e) {
      hooksStore.setError(e instanceof Error ? e.message : 'Failed to execute hook');
    }
  }

  async function handleClearLogs() {
    try {
      await memoryApi.clearHookLogs();
      hooksStore.clearLogs();
    } catch (e) {
      hooksStore.setError(e instanceof Error ? e.message : 'Failed to clear logs');
    }
  }

  function formatDuration(ms: number): string {
    if (ms < 1) return `${(ms * 1000).toFixed(0)}μs`;
    if (ms < 1000) return `${ms.toFixed(1)}ms`;
    return `${(ms / 1000).toFixed(2)}s`;
  }

  function formatTime(timestamp: string): string {
    return new Date(timestamp).toLocaleTimeString();
  }

  function getTimeAgo(timestamp: string): string {
    const seconds = Math.floor((Date.now() - new Date(timestamp).getTime()) / 1000);
    if (seconds < 60) return `${seconds}s ago`;
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
    return `${Math.floor(seconds / 3600)}h ago`;
  }

  function getHookCategory(hookName: string): string {
    if (hookName.startsWith('pre-')) return 'Pre';
    if (hookName.startsWith('post-')) return 'Post';
    if (hookName.startsWith('worker-')) return 'Worker';
    return 'Other';
  }
</script>

<div class="hooks-panel-overlay" on:click={onClose}>
  <div class="hooks-panel" on:click|stopPropagation>
    <!-- Header -->
    <div class="panel-header">
      <div class="header-left">
        <GitBranch size={20} />
        <div>
          <h2>Hook Viewer</h2>
          <span class="subtitle">{hooks.length} hooks registered</span>
        </div>
      </div>
      <div class="header-actions">
        <button class="icon-btn" on:click={loadHooks} title="Refresh hooks" disabled={loading}>
          <RefreshCw size={16} class={loading ? 'spinning' : ''} />
        </button>
        <button class="icon-btn" on:click={loadLogs} title="Refresh logs">
          <FileText size={16} />
        </button>
        <button class="icon-btn danger" on:click={handleClearLogs} title="Clear logs">
          <Trash2 size={16} />
        </button>
        <button class="icon-btn" on:click={onClose} title="Close">
          <X size={16} />
        </button>
      </div>
    </div>

    <!-- Error Display -->
    {#if error}
    <div class="error-banner">
      <AlertCircle size={16} />
      <span>{error}</span>
      <button on:click={() => hooksStore.setError(null)}><X size={14} /></button>
    </div>
    {/if}

    <div class="panel-content">
      <!-- Hooks List -->
      <div class="hooks-section">
        <div class="section-header">
          <h3>Registered Hooks</h3>
        </div>

        {#if loading}
        <div class="loading-state">
          <RefreshCw size={24} class="spinning" />
          <span>Loading hooks...</span>
        </div>
        {:else if hooks.length === 0}
        <div class="empty-state">
          <GitBranch size={32} />
          <p>No hooks registered</p>
        </div>
        {:else}
        <div class="hooks-list">
          {#each hooks as hook}
          <div
            class="hook-item"
            class:selected={selectedHookName === hook.name}
            on:click={() => selectedHookName = hook.name}
          >
            <div class="hook-header">
              <div class="hook-info">
                <button
                  class="toggle-btn"
                  on:click|stopPropagation={() => handleToggleHook(hook)}
                  title={hook.enabled ? 'Disable' : 'Enable'}
                >
                  {#if hook.enabled}
                  <ToggleRight size={16} />
                  {:else}
                  <ToggleLeft size={16} />
                  {/if}
                </button>
                <span class="hook-name">{hook.name}</span>
                <span class="hook-category">{getHookCategory(hook.name)}</span>
              </div>
              <div class="hook-actions">
                <button
                  class="icon-btn small"
                  on:click|stopPropagation={() => handleExecuteHook(hook)}
                  title="Execute"
                  disabled={!hook.enabled}
                >
                  <Play size={12} />
                </button>
              </div>
            </div>
            <p class="hook-description">{hook.description}</p>
            <div class="hook-stats">
              <span class="stat">
                <Zap size={10} />
                Executed: {hook.executionCount}x
              </span>
              {#if hook.avgExecutionTime}
              <span class="stat">
                <Clock size={10} />
                Avg: {formatDuration(hook.avgExecutionTime)}
              </span>
              {/if}
              {#if hook.lastExecuted}
              <span class="stat">
                Last: {getTimeAgo(hook.lastExecuted)}
              </span>
              {/if}
            </div>
          </div>
          {/each}
        </div>
        {/if}
      </div>

      <!-- Hook Logs -->
      <div class="logs-section">
        <div class="section-header">
          <h3>Execution Logs</h3>
          <div class="log-filters">
            <button
              class="filter-btn"
              class:active={logFilter === 'all'}
              on:click={() => logFilter = 'all'}
            >All</button>
            <button
              class="filter-btn"
              class:active={logFilter === 'success'}
              on:click={() => logFilter = 'success'}
            >Success</button>
            <button
              class="filter-btn"
              class:active={logFilter === 'failed'}
              on:click={() => logFilter = 'failed'}
            >Failed</button>
          </div>
        </div>

        {#if filteredLogs.length === 0}
        <div class="empty-state">
          <FileText size={32} />
          <p>No logs to display</p>
        </div>
        {:else}
        <div class="logs-list">
          {#each filteredLogs as log}
          <div class="log-item" class:success={log.status === 'success'} class:failed={log.status === 'failed'}>
            <div class="log-header">
              <div class="log-info">
                {#if log.status === 'success'}
                <CheckCircle size={14} class="status-icon success" />
                {:else}
                <XCircle size={14} class="status-icon failed" />
                {/if}
                <span class="log-hook-name">{log.hookName}</span>
              </div>
              <div class="log-meta">
                <span class="log-time" title={formatTime(log.timestamp)}>{getTimeAgo(log.timestamp)}</span>
                <span class="log-duration">{formatDuration(log.duration)}</span>
              </div>
            </div>
            {#if log.message}
            <div class="log-message">{log.message}</div>
            {/if}
          </div>
          {/each}
        </div>
        {/if}
      </div>
    </div>
  </div>
</div>

<style>
  .hooks-panel-overlay {
    position: fixed;
    inset: 0;
    background: rgba(0, 0, 0, 0.7);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 1000;
    padding: 20px;
  }

  .hooks-panel {
    background: var(--bg-secondary);
    border-radius: 12px;
    width: 1100px;
    max-width: 100%;
    height: 85vh;
    max-height: 90vh;
    display: flex;
    flex-direction: column;
    box-shadow: 0 25px 50px rgba(0, 0, 0, 0.5);
    border: 1px solid var(--border-subtle);
  }

  .panel-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 16px 20px;
    border-bottom: 1px solid var(--border-subtle);
    background: var(--bg-primary);
    border-radius: 12px 12px 0 0;
  }

  .header-left {
    display: flex;
    align-items: center;
    gap: 12px;
    color: var(--accent-primary);
  }

  .header-left h2 {
    margin: 0;
    font-size: 16px;
    color: var(--text-primary);
  }

  .subtitle {
    font-size: 11px;
    color: var(--text-muted);
  }

  .header-actions {
    display: flex;
    gap: 6px;
  }

  .error-banner {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 20px;
    background: rgba(239, 68, 68, 0.1);
    border-bottom: 1px solid rgba(239, 68, 68, 0.3);
    color: var(--accent-danger);
    font-size: 12px;
  }

  .error-banner button {
    margin-left: auto;
    background: none;
    border: none;
    color: inherit;
    cursor: pointer;
  }

  .panel-content {
    flex: 1;
    display: grid;
    grid-template-columns: 1fr 1fr;
    overflow: hidden;
  }

  .hooks-section,
  .logs-section {
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }

  .hooks-section {
    border-right: 1px solid var(--border-subtle);
  }

  .section-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 16px;
    background: var(--bg-tertiary);
    border-bottom: 1px solid var(--border-subtle);
  }

  .section-header h3 {
    margin: 0;
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    color: var(--text-muted);
  }

  .log-filters {
    display: flex;
    gap: 4px;
  }

  .filter-btn {
    padding: 4px 10px;
    background: transparent;
    border: 1px solid var(--border-subtle);
    border-radius: 4px;
    color: var(--text-muted);
    font-size: 11px;
    cursor: pointer;
    transition: all 0.15s;
  }

  .filter-btn:hover {
    background: var(--bg-input);
  }

  .filter-btn.active {
    background: var(--accent-primary);
    border-color: var(--accent-primary);
    color: var(--bg-primary);
  }

  .loading-state,
  .empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 40px 20px;
    color: var(--text-muted);
    gap: 12px;
  }

  .hooks-list,
  .logs-list {
    flex: 1;
    overflow-y: auto;
    padding: 12px;
  }

  .hook-item {
    padding: 12px;
    background: var(--bg-tertiary);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
    margin-bottom: 8px;
    cursor: pointer;
    transition: all 0.15s;
  }

  .hook-item:hover {
    border-color: var(--border-strong);
  }

  .hook-item.selected {
    border-color: var(--accent-primary);
    background: rgba(99, 102, 241, 0.1);
  }

  .hook-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
  }

  .hook-info {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .toggle-btn {
    display: flex;
    align-items: center;
    background: none;
    border: none;
    color: var(--text-muted);
    cursor: pointer;
    padding: 0;
    transition: color 0.15s;
  }

  .toggle-btn:hover {
    color: var(--accent-primary);
  }

  .hook-name {
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    font-weight: 500;
    color: var(--text-primary);
  }

  .hook-category {
    padding: 2px 6px;
    background: var(--bg-input);
    border-radius: 4px;
    font-size: 10px;
    color: var(--text-muted);
    text-transform: uppercase;
  }

  .hook-actions {
    display: flex;
    gap: 4px;
  }

  .hook-description {
    margin: 0 0 8px 24px;
    font-size: 12px;
    color: var(--text-secondary);
  }

  .hook-stats {
    display: flex;
    gap: 12px;
    padding: 6px 12px;
    background: var(--bg-input);
    border-radius: 6px;
    font-size: 10px;
    color: var(--text-muted);
    margin-left: 24px;
  }

  .hook-stats .stat {
    display: flex;
    align-items: center;
    gap: 4px;
  }

  .log-item {
    padding: 10px 12px;
    background: var(--bg-tertiary);
    border: 1px solid var(--border-subtle);
    border-radius: 6px;
    margin-bottom: 6px;
    font-size: 12px;
  }

  .log-item.success {
    border-left: 3px solid var(--accent-success);
  }

  .log-item.failed {
    border-left: 3px solid var(--accent-danger);
  }

  .log-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 4px;
  }

  .log-info {
    display: flex;
    align-items: center;
    gap: 6px;
  }

  .status-icon.success {
    color: var(--accent-success);
  }

  .status-icon.failed {
    color: var(--accent-danger);
  }

  .log-hook-name {
    font-family: 'JetBrains Mono', monospace;
    font-weight: 500;
    color: var(--text-primary);
  }

  .log-meta {
    display: flex;
    gap: 8px;
    font-size: 10px;
    color: var(--text-muted);
  }

  .log-message {
    margin-top: 6px;
    padding-left: 20px;
    color: var(--text-secondary);
    font-size: 11px;
  }

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
    background: var(--bg-tertiary);
    color: var(--text-primary);
  }

  .icon-btn.small {
    width: 24px;
    height: 24px;
  }

  .icon-btn.danger:hover {
    background: rgba(239, 68, 68, 0.2);
    color: var(--accent-danger);
  }

  .spinning {
    animation: spin 1s linear infinite;
  }

  @keyframes spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
  }
</style>
