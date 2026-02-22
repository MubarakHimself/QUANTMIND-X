<script lang="ts">
  import { createEventDispatcher, onMount, tick, onDestroy } from 'svelte';
  import { Trash2, ChevronDown, AlertCircle, AlertTriangle, Info, CheckCircle, X, Pause, Play } from 'lucide-svelte';

  const dispatch = createEventDispatcher();

  // Log levels
  export type LogLevel = 'debug' | 'info' | 'warn' | 'error' | 'success';

  export interface LogEntry {
    id: string;
    timestamp: Date;
    level: LogLevel;
    message: string;
    source?: string;
  }

  // Props
  export let logs: LogEntry[] = [];
  export let autoScroll = true;
  export let maxLogs = 1000;

  // Filter state
  let activeFilters: Set<LogLevel> = new Set(['debug', 'info', 'warn', 'error', 'success']);
  let filterDropdownOpen = false;
  let searchTerm = '';

  // UI state
  let paused = false;
  let logsContainer: HTMLDivElement;

  // Color mapping for log levels
  const levelColors = {
    debug: '#6b7280',
    info: '#3b82f6',
    warn: '#f59e0b',
    error: '#ef4444',
    success: '#10b981'
  };

  const levelIcons = {
    debug: Info,
    info: Info,
    warn: AlertTriangle,
    error: AlertCircle,
    success: CheckCircle
  };

  // Filtered logs
  $: filteredLogs = logs.filter(log => {
    const matchesLevel = activeFilters.has(log.level);
    const matchesSearch = searchTerm === '' ||
      log.message.toLowerCase().includes(searchTerm.toLowerCase()) ||
      (log.source && log.source.toLowerCase().includes(searchTerm.toLowerCase()));
    return matchesLevel && matchesSearch;
  });

  // Log counts by level
  $: logCounts = logs.reduce((acc, log) => {
    acc[log.level] = (acc[log.level] || 0) + 1;
    return acc;
  }, {} as Record<LogLevel, number>);

  // Auto-scroll to bottom when new logs arrive
  async function scrollToBottom() {
    if (autoScroll && !paused) {
      await tick();
      if (logsContainer) {
        logsContainer.scrollTop = logsContainer.scrollHeight;
      }
    }
  }

  // Watch for new logs and scroll
  $: if (filteredLogs.length > 0) {
    scrollToBottom();
  }

  // Toggle filter for a log level
  function toggleFilter(level: LogLevel) {
    if (activeFilters.has(level)) {
      // Don't allow unchecking the last filter
      if (activeFilters.size > 1) {
        activeFilters.delete(level);
        activeFilters = new Set(activeFilters);
      }
    } else {
      activeFilters.add(level);
      activeFilters = new Set(activeFilters);
    }
  }

  // Clear all logs
  function clearLogs() {
    logs = [];
    dispatch('clear');
  }

  // Toggle pause state
  function togglePause() {
    paused = !paused;
    if (!paused) {
      scrollToBottom();
    }
  }

  // Export logs as text
  function exportLogs() {
    const logText = logs.map(log =>
      `[${log.timestamp.toISOString()}] [${log.level.toUpperCase()}]${log.source ? ` [${log.source}]` : ''} ${log.message}`
    ).join('\n');

    const blob = new Blob([logText], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `logs-${new Date().toISOString()}.log`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    dispatch('export');
  }

  // Format timestamp for display
  function formatTimestamp(date: Date): string {
    return date.toLocaleTimeString('en-US', {
      hour12: false,
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
      fractionalSecondDigits: 3
    });
  }

  // Add a new log entry (utility method)
  export function addLog(level: LogLevel, message: string, source?: string) {
    const newLog: LogEntry = {
      id: `log-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      timestamp: new Date(),
      level,
      message,
      source
    };

    // Keep only the last maxLogs entries
    logs = [...logs.slice(-maxLogs + 1), newLog];
    dispatch('logAdded', newLog);
  }

  // Shortcut methods for adding logs
  export function debug(message: string, source?: string) {
    addLog('debug', message, source);
  }

  export function info(message: string, source?: string) {
    addLog('info', message, source);
  }

  export function warn(message: string, source?: string) {
    addLog('warn', message, source);
  }

  export function error(message: string, source?: string) {
    addLog('error', message, source);
  }

  export function success(message: string, source?: string) {
    addLog('success', message, source);
  }

  // Get level badge color
  function getLevelColor(level: LogLevel): string {
    return levelColors[level];
  }
</script>

<div class="log-viewer">
  <!-- Toolbar -->
  <div class="log-toolbar">
    <div class="toolbar-left">
      <div class="filter-wrapper">
        <button
          class="filter-btn"
          class:active={filterDropdownOpen}
          on:click={() => filterDropdownOpen = !filterDropdownOpen}
          aria-label="Filter logs by level"
        >
          <ChevronDown size={14} class:rotated={filterDropdownOpen} />
          <span>Filter</span>
          <span class="filter-count">{activeFilters.size}</span>
        </button>

        {#if filterDropdownOpen}
          <div class="filter-dropdown">
            {#each ['debug', 'info', 'warn', 'error', 'success'] as level}
              <button
                class:active={activeFilters.has(level)}
                on:click={() => toggleFilter(level as LogLevel)}
              >
                <svelte:component this={levelIcons[level]} size={12} />
                <span>{level.toUpperCase()}</span>
                <span class="count">{logCounts[level] || 0}</span>
              </button>
            {/each}
          </div>
        {/if}
      </div>

      <div class="search-wrapper">
        <input
          type="text"
          placeholder="Search logs..."
          bind:value={searchTerm}
          aria-label="Search logs"
        />
        {#if searchTerm}
          <button class="clear-search" on:click={() => searchTerm = ''} aria-label="Clear search">
            <X size={12} />
          </button>
        {/if}
      </div>
    </div>

    <div class="toolbar-right">
      <button
        class="toolbar-btn"
        on:click={togglePause}
        title={paused ? 'Resume auto-scroll' : 'Pause auto-scroll'}
        aria-label={paused ? 'Resume auto-scroll' : 'Pause auto-scroll'}
      >
        {#if paused}
          <Play size={14} />
        {:else}
          <Pause size={14} />
        {/if}
      </button>

      <button
        class="toolbar-btn"
        on:click={exportLogs}
        title="Export logs"
        aria-label="Export logs"
      >
        <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
          <polyline points="7 10 12 15 17 10"></polyline>
          <line x1="12" y1="15" x2="12" y2="3"></line>
        </svg>
      </button>

      <button
        class="toolbar-btn danger"
        on:click={clearLogs}
        title="Clear logs"
        aria-label="Clear logs"
      >
        <Trash2 size={14} />
      </button>
    </div>
  </div>

  <!-- Log entries container -->
  <div class="log-container" bind:this={logsContainer}>
    {#if filteredLogs.length === 0}
      <div class="empty-state">
        <Info size={32} />
        <p>No logs to display</p>
        {#if logs.length > 0}
          <small>Adjust filters to see more logs</small>
        {:else}
          <small>Logs will appear here as they are generated</small>
        {/if}
      </div>
    {:else}
      <div class="log-entries">
        {#each filteredLogs as log (log.id)}
          <div class="log-entry" class:log-{log.level}>
            <span class="log-timestamp">{formatTimestamp(log.timestamp)}</span>
            <span class="log-level" style="--level-color: {getLevelColor(log.level)}">
              <svelte:component this={levelIcons[log.level]} size={12} />
              {log.level.toUpperCase()}
            </span>
            {#if log.source}
              <span class="log-source">{log.source}</span>
            {/if}
            <span class="log-message">{log.message}</span>
          </div>
        {/each}
      </div>
    {/if}
  </div>

  <!-- Status bar -->
  <div class="log-statusbar">
    <span class="status-left">
      {logs.length} {logs.length === 1 ? 'entry' : 'entries'}
      {#if paused}
        <span class="paused-indicator">Paused</span>
      {/if}
    </span>
    <span class="status-right">
      {filteredLogs.length !== logs.length ? `Showing ${filteredLogs.length} filtered` : ''}
    </span>
  </div>
</div>

<style>
  .log-viewer {
    display: flex;
    flex-direction: column;
    width: 100%;
    height: 100%;
    background: var(--bg-secondary);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
    overflow: hidden;
    font-size: 12px;
  }

  /* Toolbar */
  .log-toolbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 12px;
    background: var(--bg-tertiary);
    border-bottom: 1px solid var(--border-subtle);
    gap: 8px;
  }

  .toolbar-left {
    display: flex;
    align-items: center;
    gap: 8px;
    flex: 1;
    min-width: 0;
  }

  .toolbar-right {
    display: flex;
    align-items: center;
    gap: 4px;
  }

  /* Filter button */
  .filter-wrapper {
    position: relative;
  }

  .filter-btn {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 6px 10px;
    background: var(--bg-primary);
    border: 1px solid var(--border-subtle);
    border-radius: 6px;
    color: var(--text-secondary);
    font-size: 11px;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.15s ease;
  }

  .filter-btn:hover {
    background: var(--bg-secondary);
    border-color: var(--accent-primary);
  }

  .filter-btn.active {
    border-color: var(--accent-primary);
    color: var(--accent-primary);
  }

  .filter-btn :global(.rotated) {
    transform: rotate(180deg);
  }

  .filter-count {
    display: flex;
    align-items: center;
    justify-content: center;
    min-width: 18px;
    height: 18px;
    padding: 0 5px;
    background: var(--accent-primary);
    border-radius: 9px;
    color: var(--bg-primary);
    font-size: 10px;
    font-weight: 600;
  }

  /* Filter dropdown */
  .filter-dropdown {
    position: absolute;
    top: calc(100% + 4px);
    left: 0;
    min-width: 160px;
    background: var(--bg-secondary);
    border: 1px solid var(--border-subtle);
    border-radius: 6px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    z-index: 100;
    overflow: hidden;
  }

  .filter-dropdown button {
    display: flex;
    align-items: center;
    gap: 8px;
    width: 100%;
    padding: 8px 12px;
    background: transparent;
    border: none;
    color: var(--text-secondary);
    font-size: 11px;
    cursor: pointer;
    transition: background 0.15s ease;
  }

  .filter-dropdown button:hover {
    background: var(--bg-tertiary);
  }

  .filter-dropdown button.active {
    background: var(--bg-tertiary);
    color: var(--accent-primary);
  }

  .filter-dropdown .count {
    margin-left: auto;
    background: var(--bg-primary);
    padding: 2px 6px;
    border-radius: 8px;
    font-size: 10px;
  }

  /* Search */
  .search-wrapper {
    position: relative;
    flex: 1;
    max-width: 300px;
  }

  .search-wrapper input {
    width: 100%;
    padding: 6px 30px 6px 10px;
    background: var(--bg-primary);
    border: 1px solid var(--border-subtle);
    border-radius: 6px;
    color: var(--text-primary);
    font-size: 11px;
  }

  .search-wrapper input:focus {
    outline: none;
    border-color: var(--accent-primary);
  }

  .search-wrapper .clear-search {
    position: absolute;
    right: 8px;
    top: 50%;
    transform: translateY(-50%);
    display: flex;
    align-items: center;
    justify-content: center;
    width: 16px;
    height: 16px;
    background: transparent;
    border: none;
    color: var(--text-muted);
    cursor: pointer;
    border-radius: 3px;
  }

  .search-wrapper .clear-search:hover {
    background: var(--bg-tertiary);
    color: var(--text-primary);
  }

  /* Toolbar buttons */
  .toolbar-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 28px;
    height: 28px;
    background: transparent;
    border: 1px solid var(--border-subtle);
    border-radius: 6px;
    color: var(--text-secondary);
    cursor: pointer;
    transition: all 0.15s ease;
  }

  .toolbar-btn:hover {
    background: var(--bg-secondary);
    color: var(--text-primary);
  }

  .toolbar-btn.danger:hover {
    background: #ef4444;
    border-color: #ef4444;
    color: white;
  }

  /* Log container */
  .log-container {
    flex: 1;
    overflow-y: auto;
    background: var(--bg-primary);
  }

  .empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 12px;
    height: 100%;
    color: var(--text-muted);
  }

  .empty-state small {
    font-size: 10px;
  }

  /* Log entries */
  .log-entries {
    display: flex;
    flex-direction: column;
  }

  .log-entry {
    display: flex;
    align-items: flex-start;
    gap: 8px;
    padding: 4px 12px;
    border-bottom: 1px solid var(--border-subtle);
    font-family: 'SF Mono', 'Monaco', 'Inconsolata', 'Roboto Mono', monospace;
    font-size: 11px;
    line-height: 1.5;
    transition: background 0.1s ease;
  }

  .log-entry:hover {
    background: var(--bg-secondary);
  }

  .log-timestamp {
    flex-shrink: 0;
    color: var(--text-muted);
    font-size: 10px;
  }

  .log-level {
    display: flex;
    align-items: center;
    gap: 4px;
    flex-shrink: 0;
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 9px;
    font-weight: 600;
    text-transform: uppercase;
    color: var(--level-color);
    background: color-mix(in srgb, var(--level-color) 15%, transparent);
  }

  .log-source {
    flex-shrink: 0;
    padding: 2px 6px;
    background: var(--bg-tertiary);
    border-radius: 4px;
    color: var(--text-muted);
    font-size: 10px;
  }

  .log-message {
    flex: 1;
    color: var(--text-primary);
    word-break: break-word;
    white-space: pre-wrap;
  }

  /* Level-specific styling */
  .log-entry.log-error {
    background: color-mix(in srgb, #ef4444 5%, transparent);
  }

  .log-entry.log-error:hover {
    background: color-mix(in srgb, #ef4444 10%, var(--bg-secondary));
  }

  .log-entry.log-warn {
    background: color-mix(in srgb, #f59e0b 3%, transparent);
  }

  .log-entry.log-warn:hover {
    background: color-mix(in srgb, #f59e0b 8%, var(--bg-secondary));
  }

  .log-entry.log-success {
    background: color-mix(in srgb, #10b981 3%, transparent);
  }

  .log-entry.log-success:hover {
    background: color-mix(in srgb, #10b981 8%, var(--bg-secondary));
  }

  /* Status bar */
  .log-statusbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 4px 12px;
    background: var(--bg-tertiary);
    border-top: 1px solid var(--border-subtle);
    font-size: 10px;
    color: var(--text-muted);
  }

  .status-left {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .paused-indicator {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 2px 6px;
    background: var(--accent-primary);
    border-radius: 4px;
    color: var(--bg-primary);
    font-weight: 500;
  }

  /* Scrollbar styling */
  .log-container::-webkit-scrollbar {
    width: 8px;
  }

  .log-container::-webkit-scrollbar-track {
    background: var(--bg-secondary);
  }

  .log-container::-webkit-scrollbar-thumb {
    background: var(--border-subtle);
    border-radius: 4px;
  }

  .log-container::-webkit-scrollbar-thumb:hover {
    background: var(--text-muted);
  }
</style>
