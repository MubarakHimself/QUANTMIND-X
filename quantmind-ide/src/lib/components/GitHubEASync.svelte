<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import {
    RefreshCw,
    Github,
    FileCode,
    CheckCircle,
    AlertCircle,
    Clock,
    ChevronRight,
    Loader,
    Plus,
    Filter,
    Search,
    ArrowLeft
  } from 'lucide-svelte';
  import { PUBLIC_API_BASE } from '$env/static/public';
  import { goto } from '$app/navigation';

  // Types
  interface SyncStatus {
    is_running: boolean;
    repo_url: string | null;
    branch: string | null;
    sync_interval_hours: number | null;
    sync_count: number;
    error_count: number;
    last_sync_time: string | null;
    last_commit_hash: string | null;
    next_scheduled_run: string | null;
  }

  interface EA {
    id: number;
    ea_filename: string;
    github_path: string;
    lines_of_code: number;
    strategy_type: string;
    status: string;
    imported_at: string | null;
    last_synced: string | null;
    version: string | null;
    checksum: string;
  }

  interface SyncResult {
    status: string;
    message: string;
    repo_url?: string;
    commit_hash?: string;
    eas_found: number;
    eas_new: number;
    eas_updated: number;
    eas_unchanged: number;
    errors: string[];
    synced_at?: string;
  }

  // State
  let status: SyncStatus | null = null;
  let eas: EA[] = [];
  let selectedEAs: Set<number> = new Set();
  let isSyncing = false;
  let isImporting = false;
  let syncResult: SyncResult | null = null;
  let searchQuery = '';
  let statusFilter = '';
  let isLoading = true;

  const apiBase = PUBLIC_API_BASE || 'http://localhost:8000';

  // Fetch status
  async function fetchStatus() {
    try {
      const response = await fetch(`${apiBase}/api/github/status`);
      if (response.ok) {
        status = await response.json();
      }
    } catch (error) {
      console.error('Failed to fetch status:', error);
    }
  }

  // Fetch EAs
  async function fetchEAs() {
    try {
      const params = new URLSearchParams();
      if (statusFilter) params.append('status', statusFilter);
      params.append('limit', '100');

      const response = await fetch(`${apiBase}/api/github/eas?${params}`);
      if (response.ok) {
        eas = await response.json();
      }
    } catch (error) {
      console.error('Failed to fetch EAs:', error);
    }
  }

  // Trigger sync
  async function triggerSync() {
    isSyncing = true;
    syncResult = null;

    try {
      const response = await fetch(`${apiBase}/api/github/sync`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ force: true })
      });

      if (response.ok) {
        syncResult = await response.json();
        await Promise.all([fetchStatus(), fetchEAs()]);
      }
    } catch (error) {
      console.error('Sync failed:', error);
      syncResult = {
        status: 'error',
        message: 'Failed to connect to server',
        eas_found: 0,
        eas_new: 0,
        eas_updated: 0,
        eas_unchanged: 0,
        errors: [String(error)]
      };
    } finally {
      isSyncing = false;
    }
  }

  // Import selected EAs
  async function importSelected() {
    if (selectedEAs.size === 0) return;

    isImporting = true;

    try {
      const response = await fetch(`${apiBase}/api/github/import`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ea_ids: Array.from(selectedEAs),
          generate_manifest: true
        })
      });

      if (response.ok) {
        const result = await response.json();
        console.log('Import result:', result);
        selectedEAs.clear();
        selectedEAs = selectedEAs; // Trigger reactivity
        await fetchEAs();
      }
    } catch (error) {
      console.error('Import failed:', error);
    } finally {
      isImporting = false;
    }
  }

  // Toggle EA selection
  function toggleEA(eaId: number) {
    if (selectedEAs.has(eaId)) {
      selectedEAs.delete(eaId);
    } else {
      selectedEAs.add(eaId);
    }
    selectedEAs = selectedEAs; // Trigger reactivity
  }

  // Select all EAs
  function selectAll() {
    filteredEAs.forEach(ea => selectedEAs.add(ea.id));
    selectedEAs = selectedEAs;
  }

  // Clear selection
  function clearSelection() {
    selectedEAs.clear();
    selectedEAs = selectedEAs;
  }

  // Filter EAs
  $: filteredEAs = eas.filter(ea => {
    const matchesSearch = !searchQuery || 
      ea.ea_filename.toLowerCase().includes(searchQuery.toLowerCase()) ||
      ea.strategy_type.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesStatus = !statusFilter || ea.status === statusFilter;
    return matchesSearch && matchesStatus;
  });

  // Format date
  function formatDate(dateStr: string | null): string {
    if (!dateStr) return 'Never';
    const date = new Date(dateStr);
    return date.toLocaleString();
  }

  // Get status badge color
  function getStatusColor(status: string): string {
    switch (status) {
      case 'new': return 'badge-new';
      case 'updated': return 'badge-updated';
      case 'imported': return 'badge-imported';
      case 'unchanged': return 'badge-unchanged';
      default: return 'badge-default';
    }
  }

  // Lifecycle
  onMount(async () => {
    await Promise.all([fetchStatus(), fetchEAs()]);
    isLoading = false;
  });
</script>

<div class="github-ea-sync">
  <!-- Header -->
  <div class="panel-header">
    <div class="header-nav">
      <button class="back-button" on:click={() => goto('/?view=ea')}>
        <ArrowLeft size={18} />
        <span>Back to EA Management</span>
      </button>
    </div>
    <div class="header-title">
      <Github size={24} />
      <h2>GitHub EA Sync</h2>
    </div>
    <p class="header-description">
      Synchronize Expert Advisors from GitHub repository
    </p>
  </div>

  <!-- Status Card -->
  <div class="status-card">
    <div class="status-header">
      <h3>Repository Status</h3>
      <button 
        class="sync-button"
        on:click={triggerSync}
        disabled={isSyncing}
      >
        {#if isSyncing}
          <Loader size={16} class="spinning" />
          Syncing...
        {:else}
          <RefreshCw size={16} />
          Sync Now
        {/if}
      </button>
    </div>

    <div class="status-grid">
      <div class="status-item">
        <span class="status-label">Repository</span>
        <span class="status-value">
          {status?.repo_url || 'Not configured'}
        </span>
      </div>
      <div class="status-item">
        <span class="status-label">Branch</span>
        <span class="status-value">{status?.branch || 'main'}</span>
      </div>
      <div class="status-item">
        <span class="status-label">Last Sync</span>
        <span class="status-value">{formatDate(status?.last_sync_time)}</span>
      </div>
      <div class="status-item">
        <span class="status-label">Next Scheduled</span>
        <span class="status-value">{formatDate(status?.next_scheduled_run)}</span>
      </div>
      <div class="status-item">
        <span class="status-label">Commit</span>
        <span class="status-value code">
          {status?.last_commit_hash?.substring(0, 7) || 'N/A'}
        </span>
      </div>
      <div class="status-item">
        <span class="status-label">Sync Interval</span>
        <span class="status-value">Every {status?.sync_interval_hours || 24}h</span>
      </div>
    </div>

    <!-- Sync Result -->
    {#if syncResult}
      <div class="sync-result status-{syncResult.status}">
        {#if syncResult.status === 'success'}
          <CheckCircle size={16} />
          <span>
            Sync complete: {syncResult.eas_new} new, {syncResult.eas_updated} updated
          </span>
        {:else}
          <AlertCircle size={16} />
          <span>{syncResult.message}</span>
        {/if}
      </div>
    {/if}
  </div>

  <!-- EAs List -->
  <div class="eas-section">
    <div class="eas-header">
      <h3>Expert Advisors ({filteredEAs.length})</h3>
      
      <!-- Search and Filter -->
      <div class="eas-controls">
        <div class="search-box">
          <Search size={16} />
          <input 
            type="text" 
            placeholder="Search EAs..." 
            bind:value={searchQuery}
          />
        </div>
        
        <select bind:value={statusFilter}>
          <option value="">All Status</option>
          <option value="new">New</option>
          <option value="updated">Updated</option>
          <option value="imported">Imported</option>
          <option value="unchanged">Unchanged</option>
        </select>

        {#if selectedEAs.size > 0}
          <button 
            class="import-button"
            on:click={importSelected}
            disabled={isImporting}
          >
            {#if isImporting}
              <Loader size={16} class="spinning" />
            {:else}
              <Plus size={16} />
            {/if}
            Import Selected ({selectedEAs.size})
          </button>
          <button class="clear-button" on:click={clearSelection}>
            Clear
          </button>
        {:else}
          <button class="select-all-button" on:click={selectAll}>
            Select All
          </button>
        {/if}
      </div>
    </div>

    {#if isLoading}
      <div class="loading-state">
        <Loader size={24} class="spinning" />
        <span>Loading EAs...</span>
      </div>
    {:else if filteredEAs.length === 0}
      <div class="empty-state">
        <FileCode size={32} />
        <p>No Expert Advisors found</p>
        <p class="hint">Click "Sync Now" to fetch EAs from the repository</p>
      </div>
    {:else}
      <div class="eas-list">
        {#each filteredEAs as ea (ea.id)}
          <div class="ea-item" class:selected={selectedEAs.has(ea.id)}>
            <input 
              type="checkbox" 
              checked={selectedEAs.has(ea.id)}
              on:change={() => toggleEA(ea.id)}
            />
            
            <div class="ea-icon">
              <FileCode size={20} />
            </div>
            
            <div class="ea-info">
              <div class="ea-name">{ea.ea_filename}</div>
              <div class="ea-meta">
                <span class="ea-strategy">{ea.strategy_type}</span>
                <span class="ea-path">{ea.github_path}</span>
              </div>
            </div>
            
            <div class="ea-stats">
              <span class="ea-loc">{ea.lines_of_code} LOC</span>
            </div>
            
            <div class="ea-status">
              <span class="badge {getStatusColor(ea.status)}">{ea.status}</span>
            </div>
            
            <ChevronRight size={16} class="ea-chevron" />
          </div>
        {/each}
      </div>
    {/if}
  </div>
</div>

<style>
  .github-ea-sync {
    display: flex;
    flex-direction: column;
    gap: 1.5rem;
    padding: 1.5rem;
    background: var(--bg-secondary, #1a1a2e);
    border-radius: 12px;
    height: 100%;
    overflow-y: auto;
  }

  .panel-header {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }

  .header-nav {
    display: flex;
    align-items: center;
  }

  .back-button {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    background: var(--bg-tertiary, #16213e);
    border: 1px solid var(--border-color, #2a3f5f);
    border-radius: 6px;
    padding: 0.5rem 0.75rem;
    color: var(--text-secondary, #a0a0a0);
    cursor: pointer;
    font-size: 0.875rem;
    transition: all 0.2s ease;
  }

  .back-button:hover {
    background: var(--bg-hover, #1e2d4d);
    color: var(--text-primary, #ffffff);
    border-color: var(--accent-color, #4a90d9);
  }

  .header-title {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    color: var(--text-primary, #ffffff);
  }

  .header-title h2 {
    margin: 0;
    font-size: 1.25rem;
    font-weight: 600;
  }

  .header-description {
    margin: 0;
    color: var(--text-secondary, #a0a0a0);
    font-size: 0.875rem;
  }

  /* Status Card */
  .status-card {
    background: var(--bg-tertiary, #16213e);
    border-radius: 8px;
    padding: 1.25rem;
  }

  .status-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1rem;
  }

  .status-header h3 {
    margin: 0;
    font-size: 1rem;
    font-weight: 600;
    color: var(--text-primary, #ffffff);
  }

  .sync-button {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem 1rem;
    background: var(--accent-primary, #4f46e5);
    border: none;
    border-radius: 6px;
    color: white;
    font-size: 0.875rem;
    cursor: pointer;
    transition: all 0.2s;
  }

  .sync-button:hover:not(:disabled) {
    background: var(--accent-hover, #4338ca);
  }

  .sync-button:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }

  .status-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
  }

  .status-item {
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
  }

  .status-label {
    font-size: 0.75rem;
    color: var(--text-secondary, #a0a0a0);
  }

  .status-value {
    font-size: 0.875rem;
    color: var(--text-primary, #ffffff);
  }

  .status-value.code {
    font-family: 'Fira Code', monospace;
    background: var(--bg-secondary, #1a1a2e);
    padding: 0.25rem 0.5rem;
    border-radius: 4px;
  }

  .sync-result {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-top: 1rem;
    padding: 0.75rem;
    border-radius: 6px;
    font-size: 0.875rem;
  }

  .sync-result.status-success {
    background: var(--status-success-bg, #1e3f2f);
    color: var(--status-success, #22c55e);
  }

  .sync-result.status-error {
    background: var(--status-error-bg, #3f1e1e);
    color: var(--status-error, #ef4444);
  }

  /* EAs Section */
  .eas-section {
    display: flex;
    flex-direction: column;
    gap: 1rem;
  }

  .eas-header {
    display: flex;
    flex-direction: column;
    gap: 1rem;
  }

  .eas-header h3 {
    margin: 0;
    font-size: 1rem;
    font-weight: 600;
    color: var(--text-primary, #ffffff);
  }

  .eas-controls {
    display: flex;
    gap: 0.75rem;
    flex-wrap: wrap;
    align-items: center;
  }

  .search-box {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem 0.75rem;
    background: var(--bg-secondary, #1a1a2e);
    border: 1px solid var(--border-color, #2a2a4a);
    border-radius: 6px;
    flex: 1;
    min-width: 200px;
  }

  .search-box input {
    background: transparent;
    border: none;
    color: var(--text-primary, #ffffff);
    font-size: 0.875rem;
    outline: none;
    width: 100%;
  }

  .eas-controls select {
    padding: 0.5rem 0.75rem;
    background: var(--bg-secondary, #1a1a2e);
    border: 1px solid var(--border-color, #2a2a4a);
    border-radius: 6px;
    color: var(--text-primary, #ffffff);
    font-size: 0.875rem;
  }

  .import-button {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem 1rem;
    background: var(--status-success, #22c55e);
    border: none;
    border-radius: 6px;
    color: white;
    font-size: 0.875rem;
    cursor: pointer;
  }

  .import-button:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }

  .select-all-button,
  .clear-button {
    padding: 0.5rem 1rem;
    background: var(--bg-secondary, #1a1a2e);
    border: 1px solid var(--border-color, #2a2a4a);
    border-radius: 6px;
    color: var(--text-secondary, #a0a0a0);
    font-size: 0.875rem;
    cursor: pointer;
  }

  .select-all-button:hover,
  .clear-button:hover {
    background: var(--bg-hover, #1f2b47);
    color: var(--text-primary, #ffffff);
  }

  /* EA List */
  .eas-list {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }

  .ea-item {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 1rem;
    background: var(--bg-tertiary, #16213e);
    border-radius: 8px;
    cursor: pointer;
    transition: all 0.2s;
    border: 1px solid transparent;
  }

  .ea-item:hover {
    background: var(--bg-hover, #1f2b47);
  }

  .ea-item.selected {
    border-color: var(--accent-primary, #4f46e5);
    background: rgba(79, 70, 229, 0.1);
  }

  .ea-item input[type="checkbox"] {
    width: 18px;
    height: 18px;
    cursor: pointer;
  }

  .ea-icon {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 40px;
    height: 40px;
    background: var(--bg-secondary, #1a1a2e);
    border-radius: 8px;
    color: var(--accent-primary, #4f46e5);
  }

  .ea-info {
    flex: 1;
    min-width: 0;
  }

  .ea-name {
    font-size: 0.9rem;
    font-weight: 500;
    color: var(--text-primary, #ffffff);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .ea-meta {
    display: flex;
    flex-direction: column;
    gap: 0.125rem;
    font-size: 0.75rem;
  }

  .ea-strategy {
    color: var(--accent-secondary, #60a5fa);
  }

  .ea-path {
    color: var(--text-secondary, #a0a0a0);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .ea-stats {
    display: flex;
    gap: 1rem;
  }

  .ea-loc {
    font-size: 0.75rem;
    color: var(--text-secondary, #a0a0a0);
  }

  .ea-status .badge {
    padding: 0.25rem 0.5rem;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 500;
    text-transform: uppercase;
  }

  .badge-new {
    background: rgba(34, 197, 94, 0.2);
    color: #22c55e;
  }

  .badge-updated {
    background: rgba(251, 191, 36, 0.2);
    color: #fbbf24;
  }

  .badge-imported {
    background: rgba(96, 165, 250, 0.2);
    color: #60a5fa;
  }

  .badge-unchanged {
    background: rgba(156, 163, 175, 0.2);
    color: #9ca3af;
  }

  .ea-chevron {
    color: var(--text-secondary, #a0a0a0);
  }

  /* Loading and Empty States */
  .loading-state,
  .empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 3rem;
    color: var(--text-secondary, #a0a0a0);
    gap: 1rem;
  }

  .empty-state p {
    margin: 0;
  }

  .empty-state .hint {
    font-size: 0.875rem;
    opacity: 0.7;
  }

  .spinning {
    animation: spin 1s linear infinite;
  }

  @keyframes spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
  }

  @media (max-width: 768px) {
    .status-grid {
      grid-template-columns: repeat(2, 1fr);
    }

    .eas-controls {
      flex-direction: column;
    }

    .search-box {
      width: 100%;
    }
  }
</style>