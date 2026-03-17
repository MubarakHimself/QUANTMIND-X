<script lang="ts">
  import { createBubbler, stopPropagation, self } from 'svelte/legacy';

  const bubble = createBubbler();
  import { onMount } from 'svelte';
  import {
    Database,
    Search,
    Filter,
    Plus,
    Trash2,
    Edit3,
    RefreshCw,
    Calendar,
    Tag,
    Brain,
    Clock,
    CheckCircle,
    XCircle,
    ChevronDown,
    ChevronRight,
    X,
    Save,
    AlertCircle
  } from 'lucide-svelte';
  import { memoryStore, type MemoryEntry, type MemoryFilters } from '$lib/stores/memoryStore';
  import * as memoryApi from '$lib/api/memory';

  let { onClose = () => {} } = $props();

  let searchInput = $state('');
  let searchTimeout: ReturnType<typeof setTimeout>;
  let showAddModal = $state(false);
  let showEditModal = $state(false);
  let editingMemory: MemoryEntry | null = $state(null);

  let newMemory = $state({
    key: '',
    content: '',
    namespace: 'default',
    tags: [] as string[],
    agent: ''
  });

  let namespaceOptions = [
    { value: 'all', label: 'All Namespaces' },
    { value: 'default', label: 'Default' },
    { value: 'patterns', label: 'Patterns' },
    { value: 'solutions', label: 'Solutions' },
    { value: 'sessions', label: 'Sessions' },
    { value: 'tasks', label: 'Tasks' }
  ];

  let agents: string[] = ['copilot', 'analyst', 'quantcode'];
  let showFilters = $state(false);
  let minDecay = $state(0);

  // Subscribe to store
  let filteredMemories = $derived($memoryStore.filteredMemories);
  let stats = $derived($memoryStore.stats);
  let loading = $derived($memoryStore.loading);
  let error = $derived($memoryStore.error);
  let filters = $derived($memoryStore.filters);

  onMount(() => {
    loadMemories();
    loadStats();
  });

  async function loadMemories() {
    memoryStore.setLoading(true);
    memoryStore.setError(null);
    try {
      const result = await memoryApi.listMemories('default', 100);
      memoryStore.setMemories(result.memories);
    } catch (e) {
      memoryStore.setError(e instanceof Error ? e.message : 'Failed to load memories');
    } finally {
      memoryStore.setLoading(false);
    }
  }

  async function loadStats() {
    try {
      const stats = await memoryApi.getMemoryStats();
      memoryStore.setStats(stats);
    } catch (e) {
      console.error('Failed to load stats:', e);
    }
  }

  async function handleSearch(query: string) {
    clearTimeout(searchTimeout);
    searchTimeout = setTimeout(async () => {
      if (!query.trim()) {
        loadMemories();
        return;
      }

      memoryStore.setLoading(true);
      try {
        const result = await memoryApi.searchMemories(query, filters.namespace === 'all' ? undefined : filters.namespace);
        memoryStore.setMemories(result.memories);
      } catch (e) {
        memoryStore.setError(e instanceof Error ? e.message : 'Search failed');
      } finally {
        memoryStore.setLoading(false);
      }
    }, 300);
  }

  async function handleAddMemory() {
    if (!newMemory.key || !newMemory.content) return;

    memoryStore.setLoading(true);
    try {
      const memory = await memoryApi.addMemory(
        newMemory.key,
        newMemory.content,
        newMemory.namespace,
        {
          tags: newMemory.tags,
          agent: newMemory.agent || undefined
        }
      );

      // Add to store
      memoryStore.setMemories([...filteredMemories, memory]);
      showAddModal = false;
      newMemory = { key: '', content: '', namespace: 'default', tags: [], agent: '' };
      loadStats();
    } catch (e) {
      memoryStore.setError(e instanceof Error ? e.message : 'Failed to add memory');
    } finally {
      memoryStore.setLoading(false);
    }
  }

  async function handleDeleteMemory(id: string, key: string, namespace: string) {
    if (!confirm(`Delete memory "${key}"?`)) return;

    try {
      await memoryApi.deleteMemory(key, namespace);
      memoryStore.setMemories(filteredMemories.filter(m => m.id !== id));
      loadStats();
    } catch (e) {
      memoryStore.setError(e instanceof Error ? e.message : 'Failed to delete memory');
    }
  }

  async function handleSync() {
    memoryStore.setLoading(true);
    try {
      await memoryApi.syncMemory();
      await loadStats();
      await loadMemories();
    } catch (e) {
      memoryStore.setError(e instanceof Error ? e.message : 'Sync failed');
    } finally {
      memoryStore.setLoading(false);
    }
  }

  function handleFilterChange(key: keyof MemoryFilters, value: any) {
    memoryStore.setFilters({ [key]: value });
  }

  function getDecayColor(decay?: number): string {
    if (!decay) return 'var(--text-muted)';
    if (decay >= 0.8) return 'var(--accent-success)';
    if (decay >= 0.5) return 'var(--accent-warning)';
    return 'var(--text-muted)';
  }

  function formatDate(timestamp: string): string {
    return new Date(timestamp).toLocaleString();
  }

  function getTimeAgo(timestamp: string): string {
    const seconds = Math.floor((Date.now() - new Date(timestamp).getTime()) / 1000);
    if (seconds < 60) return `${seconds}s ago`;
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
    if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
    return `${Math.floor(seconds / 86400)}d ago`;
  }
</script>

<div class="memory-panel-overlay" onclick={onClose}>
  <div class="memory-panel" onclick={stopPropagation(bubble('click'))}>
    <!-- Header -->
    <div class="panel-header">
      <div class="header-left">
        <Database size={20} />
        <div>
          <h2>Memory Management</h2>
          <span class="subtitle">
            {stats ? `${stats.total_count} memories` : 'Loading...'}
          </span>
        </div>
      </div>
      <div class="header-actions">
        <button class="icon-btn" onclick={handleSync} title="Sync Memory" disabled={loading}>
          <RefreshCw size={16} class={loading ? 'spinning' : ''} />
        </button>
        <button class="icon-btn primary" onclick={() => showAddModal = true} title="Add Memory">
          <Plus size={16} />
        </button>
        <button class="icon-btn" onclick={onClose} title="Close">
          <X size={16} />
        </button>
      </div>
    </div>

    <!-- Stats Bar -->
    {#if stats}
    <div class="stats-bar">
      <div class="stat-item">
        <Database size={14} />
        <span>Total: {stats.total_count}</span>
      </div>
      <div class="stat-item">
        <Clock size={14} />
        <span>Last Sync: {getTimeAgo(stats.last_sync)}</span>
      </div>
      <div class="stat-item">
        <Brain size={14} />
        <span>Model: {stats.embedding_model}</span>
      </div>
      <div class="stat-item">
        <Calendar size={14} />
        <span>Range: {stats.oldest_memory ? getTimeAgo(stats.oldest_memory) : 'N/A'} - {stats.newest_memory ? getTimeAgo(stats.newest_memory) : 'N/A'}</span>
      </div>
    </div>
    {/if}

    <!-- Search and Filters -->
    <div class="search-section">
      <div class="search-bar">
        <Search size={16} class="search-icon" />
        <input
          type="text"
          placeholder="Search memories by content, key, or tags..."
          bind:value={searchInput}
          oninput={() => handleFilterChange('searchQuery', searchInput)}
        />
      </div>
      <button
        class="icon-btn"
        class:active={showFilters}
        onclick={() => showFilters = !showFilters}
        title="Filters"
      >
        <Filter size={16} />
      </button>
    </div>

    {#if showFilters}
    <div class="filters-panel">
      <div class="filter-row">
        <label>Namespace</label>
        <select bind:value={filters.namespace} onchange={() => handleFilterChange('namespace', filters.namespace)}>
          {#each namespaceOptions as opt}
            <option value={opt.value}>{opt.label}</option>
          {/each}
        </select>
      </div>
      <div class="filter-row">
        <label>Agent</label>
        <select bind:value={filters.agent} onchange={() => handleFilterChange('agent', filters.agent)}>
          <option value="">All Agents</option>
          {#each agents as agent}
            <option value={agent}>{agent}</option>
          {/each}
        </select>
      </div>
      <div class="filter-row">
        <label>Min Decay: {minDecay.toFixed(2)}</label>
        <input
          type="range"
          min="0"
          max="1"
          step="0.1"
          bind:value={minDecay}
          oninput={() => handleFilterChange('minDecay', minDecay)}
        />
      </div>
    </div>
    {/if}

    <!-- Error Display -->
    {#if error}
    <div class="error-banner">
      <AlertCircle size={16} />
      <span>{error}</span>
      <button onclick={() => memoryStore.setError(null)}><X size={14} /></button>
    </div>
    {/if}

    <!-- Memory List -->
    <div class="memory-list">
      {#if loading}
      <div class="loading-state">
        <RefreshCw size={32} class="spinning" />
        <span>Loading memories...</span>
      </div>
      {:else if filteredMemories.length === 0}
      <div class="empty-state">
        <Database size={48} />
        <p>No memories found</p>
        <button class="btn primary" onclick={() => showAddModal = true}>
          <Plus size={14} /> Add Memory
        </button>
      </div>
      {:else}
      {#each filteredMemories as memory}
      <div class="memory-item">
        <div class="memory-header">
          <div class="memory-key">
            <Tag size={12} />
            <code>{memory.key}</code>
          </div>
          <div class="memory-actions">
            {#if memory.decay !== undefined}
            <div class="decay-indicator" style="color: {getDecayColor(memory.decay)}" title="Decay: {(memory.decay * 100).toFixed(0)}%">
              <div class="decay-bar" style="width: {memory.decay * 100}%"></div>
            </div>
            {/if}
            <button class="icon-btn small" onclick={() => { editingMemory = memory; showEditModal = true; }} title="Edit">
              <Edit3 size={12} />
            </button>
            <button class="icon-btn small danger" onclick={() => handleDeleteMemory(memory.id, memory.key, memory.namespace)} title="Delete">
              <Trash2 size={12} />
            </button>
          </div>
        </div>
        <div class="memory-content">{memory.content}</div>
        <div class="memory-meta">
          <span class="namespace-badge">{memory.namespace}</span>
          {#if memory.agent}
          <span class="agent-badge">{memory.agent}</span>
          {/if}
          {#if memory.tags && memory.tags.length > 0}
          <div class="tags">
            {#each memory.tags as tag}
            <span class="tag">{tag}</span>
            {/each}
          </div>
          {/if}
          <span class="timestamp" title={formatDate(memory.timestamp)}>{getTimeAgo(memory.timestamp)}</span>
        </div>
      </div>
      {/each}
      {/if}
    </div>
  </div>
</div>

<!-- Add Memory Modal -->
{#if showAddModal}
<div class="modal-overlay" onclick={self(() => showAddModal = false)}>
  <div class="modal">
    <div class="modal-header">
      <h3>Add Memory</h3>
      <button onclick={() => showAddModal = false}><X size={18} /></button>
    </div>
    <div class="modal-body">
      <div class="form-group">
        <label>Key *</label>
        <input type="text" bind:value={newMemory.key} placeholder="memory-key-name" />
      </div>
      <div class="form-group">
        <label>Content *</label>
        <textarea bind:value={newMemory.content} rows="4" placeholder="Memory content..."></textarea>
      </div>
      <div class="form-row">
        <div class="form-group">
          <label>Namespace</label>
          <select bind:value={newMemory.namespace}>
            {#each namespaceOptions.slice(1) as opt}
            <option value={opt.value}>{opt.label}</option>
            {/each}
          </select>
        </div>
        <div class="form-group">
          <label>Agent</label>
          <select bind:value={newMemory.agent}>
            <option value="">None</option>
            {#each agents as agent}
            <option value={agent}>{agent}</option>
            {/each}
          </select>
        </div>
      </div>
      <div class="form-group">
        <label>Tags (comma-separated)</label>
        <input type="text" bind:value={newMemory.tags} placeholder="tag1, tag2, tag3" />
      </div>
    </div>
    <div class="modal-footer">
      <button class="btn secondary" onclick={() => showAddModal = false}>Cancel</button>
      <button class="btn primary" onclick={handleAddMemory} disabled={!newMemory.key || !newMemory.content || loading}>
        {#if loading}
        <RefreshCw size={14} class="spinning" />
        {:else}
        <Save size={14} />
        {/if}
        Add Memory
      </button>
    </div>
  </div>
</div>
{/if}

<style>
  .memory-panel-overlay {
    position: fixed;
    inset: 0;
    background: rgba(0, 0, 0, 0.7);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 1000;
    padding: 20px;
  }

  .memory-panel {
    background: var(--bg-secondary);
    border-radius: 12px;
    width: 900px;
    max-width: 100%;
    height: 80vh;
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

  .stats-bar {
    display: flex;
    gap: 16px;
    padding: 10px 20px;
    background: var(--bg-tertiary);
    border-bottom: 1px solid var(--border-subtle);
    flex-wrap: wrap;
  }

  .stat-item {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 11px;
    color: var(--text-secondary);
  }

  .stat-item svg {
    color: var(--accent-primary);
  }

  .search-section {
    display: flex;
    gap: 8px;
    padding: 12px 20px;
    background: var(--bg-secondary);
    border-bottom: 1px solid var(--border-subtle);
  }

  .search-bar {
    flex: 1;
    position: relative;
    display: flex;
    align-items: center;
  }

  .search-icon {
    position: absolute;
    left: 10px;
    color: var(--text-muted);
    pointer-events: none;
  }

  .search-bar input {
    width: 100%;
    padding: 8px 10px 8px 34px;
    background: var(--bg-input);
    border: 1px solid var(--border-subtle);
    border-radius: 6px;
    color: var(--text-primary);
    font-size: 13px;
  }

  .search-bar input:focus {
    outline: none;
    border-color: var(--accent-primary);
  }

  .filters-panel {
    padding: 12px 20px;
    background: var(--bg-primary);
    border-bottom: 1px solid var(--border-subtle);
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
  }

  .filter-row {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .filter-row label {
    font-size: 10px;
    text-transform: uppercase;
    color: var(--text-muted);
    font-weight: 600;
  }

  .filter-row select,
  .filter-row input[type="range"] {
    width: 100%;
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

  .memory-list {
    flex: 1;
    overflow-y: auto;
    padding: 12px;
  }

  .loading-state,
  .empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 60px 20px;
    color: var(--text-muted);
    gap: 16px;
  }

  .memory-item {
    padding: 12px;
    background: var(--bg-tertiary);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
    margin-bottom: 8px;
    transition: border-color 0.15s;
  }

  .memory-item:hover {
    border-color: var(--border-strong);
  }

  .memory-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
  }

  .memory-key {
    display: flex;
    align-items: center;
    gap: 6px;
    color: var(--accent-primary);
  }

  .memory-key code {
    font-size: 12px;
    background: var(--bg-primary);
    padding: 2px 6px;
    border-radius: 4px;
  }

  .memory-actions {
    display: flex;
    align-items: center;
    gap: 4px;
  }

  .decay-indicator {
    position: relative;
    width: 40px;
    height: 4px;
    background: var(--bg-input);
    border-radius: 2px;
    overflow: hidden;
  }

  .decay-bar {
    height: 100%;
    background: currentColor;
    transition: width 0.3s;
  }

  .memory-content {
    color: var(--text-primary);
    font-size: 13px;
    line-height: 1.5;
    margin-bottom: 8px;
  }

  .memory-meta {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
    font-size: 10px;
  }

  .namespace-badge,
  .agent-badge {
    padding: 2px 6px;
    border-radius: 4px;
    font-weight: 500;
    text-transform: uppercase;
  }

  .namespace-badge {
    background: rgba(99, 102, 241, 0.2);
    color: #818cf8;
  }

  .agent-badge {
    background: rgba(16, 185, 129, 0.2);
    color: #10b981;
  }

  .tags {
    display: flex;
    gap: 4px;
  }

  .tag {
    padding: 2px 6px;
    background: var(--bg-input);
    color: var(--text-muted);
    border-radius: 4px;
  }

  .timestamp {
    color: var(--text-muted);
    margin-left: auto;
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

  .icon-btn.primary {
    background: var(--accent-primary);
    color: var(--bg-primary);
  }

  .icon-btn.primary:hover {
    opacity: 0.9;
  }

  .icon-btn.active {
    background: var(--accent-primary);
    color: var(--bg-primary);
  }

  .icon-btn.small {
    width: 24px;
    height: 24px;
  }

  .icon-btn.danger:hover {
    background: rgba(239, 68, 68, 0.2);
    color: var(--accent-danger);
  }

  .btn {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 8px 16px;
    border-radius: 6px;
    font-size: 13px;
    font-weight: 500;
    border: none;
    cursor: pointer;
  }

  .btn.primary {
    background: var(--accent-primary);
    color: var(--bg-primary);
  }

  .btn.secondary {
    background: var(--bg-tertiary);
    color: var(--text-secondary);
  }

  .spinning {
    animation: spin 1s linear infinite;
  }

  @keyframes spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
  }

  /* Modal */
  .modal-overlay {
    position: absolute;
    inset: 0;
    background: rgba(0, 0, 0, 0.7);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 10;
  }

  .modal {
    background: var(--bg-secondary);
    border-radius: 12px;
    width: 480px;
    max-width: 90%;
  }

  .modal-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 16px 20px;
    border-bottom: 1px solid var(--border-subtle);
  }

  .modal-header h3 {
    margin: 0;
    font-size: 16px;
    color: var(--text-primary);
  }

  .modal-body {
    padding: 20px;
  }

  .form-group {
    margin-bottom: 16px;
  }

  .form-group label {
    display: block;
    margin-bottom: 6px;
    font-size: 12px;
    color: var(--text-muted);
  }

  .form-group input,
  .form-group select,
  .form-group textarea {
    width: 100%;
    padding: 8px 12px;
    background: var(--bg-tertiary);
    border: 1px solid var(--border-subtle);
    border-radius: 6px;
    color: var(--text-primary);
    font-size: 13px;
  }

  .form-group textarea {
    min-height: 80px;
    resize: vertical;
  }

  .form-row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
  }

  .modal-footer {
    display: flex;
    justify-content: flex-end;
    gap: 8px;
    padding: 16px 20px;
    border-top: 1px solid var(--border-subtle);
  }
</style>
