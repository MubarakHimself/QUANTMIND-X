<script lang="ts">
  import { onMount } from 'svelte';
  import { fade, slide, fly } from 'svelte/transition';
  import {
    Brain, Search, Filter, Calendar, Tag, FileText, ChevronRight,
    ChevronDown, RefreshCw, Plus, Eye, Edit, Trash2, BookOpen,
    Clock, Hash, FolderOpen, X, AlertCircle, CheckCircle, Zap
  } from 'lucide-svelte';

  // Types
  interface MemoryEntry {
    category: string;
    timestamp: string;
    content: string;
    tags?: string[];
  }

  interface DailyLog {
    date: string;
    entries: LogEntry[];
  }

  interface LogEntry {
    time: string;
    category?: string;
    content: string;
  }

  interface MemoryStats {
    total_entries: number;
    categories: string[];
    daily_log_count: number;
  }

  // Department enum
  const DEPARTMENTS = [
    { id: 'analysis', name: 'Analysis', color: '#3b82f6' },
    { id: 'research', name: 'Research', color: '#8b5cf6' },
    { id: 'risk', name: 'Risk', color: '#ef4444' },
    { id: 'execution', name: 'Execution', color: '#f97316' },
    { id: 'portfolio', name: 'Portfolio', color: '#10b981' },
    { id: 'floor_manager', name: 'Floor Manager', color: '#06b6d4' }
  ];

  // State
  let selectedDepartment = 'analysis';
  let memories: MemoryEntry[] = [];
  let dailyLogs: DailyLog[] = [];
  let stats: MemoryStats | null = null;
  let loading = false;
  let error: string | null = null;

  // View state
  let activeTab = 'memory'; // 'memory' or 'logs'
  let expandedMemory: string | null = null;
  let expandedLog: string | null = null;
  let selectedLog: DailyLog | null = null;
  let searchQuery = '';
  let categoryFilter = 'all';

  // New memory modal
  let addMemoryModalOpen = false;
  let newMemory = {
    category: '',
    content: '',
    tags: [] as string[]
  };

  const API_BASE = 'http://localhost:8000/api';

  // Lifecycle
  onMount(() => {
    loadDepartmentMemory();
  });

  // Load department memory
  async function loadDepartmentMemory() {
    loading = true;
    error = null;

    try {
      const [memRes, logsRes, statsRes] = await Promise.all([
        fetch(`${API_BASE}/memory/${selectedDepartment}`),
        fetch(`${API_BASE}/memory/${selectedDepartment}/logs`),
        fetch(`${API_BASE}/memory/${selectedDepartment}/stats`)
      ]);

      if (!memRes.ok || !logsRes.ok || !statsRes.ok) {
        throw new Error('Failed to load memory data');
      }

      const memData = await memRes.json();
      const logsData = await logsRes.json();
      const statsData = await statsRes.json();

      memories = memData.memories || [];
      dailyLogs = logsData.logs || [];
      stats = statsData;
    } catch (e) {
      error = e instanceof Error ? e.message : 'Failed to load memory';
      console.error('Failed to load department memory:', e);
    } finally {
      loading = false;
    }
  }

  // Add new memory
  async function addMemory() {
    if (!newMemory.category || !newMemory.content) {
      alert('Please fill in category and content');
      return;
    }

    try {
      const res = await fetch(`${API_BASE}/memory/${selectedDepartment}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(newMemory)
      });

      if (!res.ok) throw new Error('Failed to add memory');

      await loadDepartmentMemory();
      addMemoryModalOpen = false;
      resetNewMemory();
    } catch (e) {
      console.error('Failed to add memory:', e);
    }
  }

  function resetNewMemory() {
    newMemory = { category: '', content: '', tags: [] };
  }

  // Search memories
  async function searchMemories() {
    if (!searchQuery.trim()) {
      loadDepartmentMemory();
      return;
    }

    try {
      const res = await fetch(`${API_BASE}/memory/${selectedDepartment}/search?q=${encodeURIComponent(searchQuery)}`);
      if (!res.ok) throw new Error('Search failed');
      const data = await res.json();
      memories = data.results || [];
    } catch (e) {
      console.error('Search failed:', e);
    }
  }

  // View daily log
  function viewDailyLog(log: DailyLog) {
    selectedLog = log;
    expandedLog = log.date;
  }

  // Toggle expanded
  function toggleExpanded(id: string) {
    expandedMemory = expandedMemory === id ? null : id;
  }

  // Get filtered memories
  $: filteredMemories = memories.filter(mem => {
    if (categoryFilter !== 'all' && mem.category.toLowerCase() !== categoryFilter) return false;
    return true;
  });

  // Get categories from memories
  $: categories = [...new Set(memories.map(m => m.category.toLowerCase()))];

  // Get department color
  function getDepartmentColor(deptId: string): string {
    const dept = DEPARTMENTS.find(d => d.id === deptId);
    return dept?.color || '#6b7280';
  }

  // Format timestamp
  function formatTimestamp(isoString: string): string {
    const date = new Date(isoString);
    return date.toLocaleString();
  }

  // Format date
  function formatDate(isoString: string): string {
    const date = new Date(isoString);
    return date.toLocaleDateString();
  }
</script>

<div class="memory-view">
  <!-- Header -->
  <div class="memory-header">
    <div class="header-left">
      <Brain size={24} class="memory-icon" />
      <div>
        <h2>Department Memory</h2>
        <p>Markdown-based memory system for department agents</p>
      </div>
    </div>
    <div class="header-actions">
      <button class="btn" on:click={loadDepartmentMemory}>
        <RefreshCw size={14} />
        <span>Refresh</span>
      </button>
      {#if activeTab === 'memory'}
        <button class="btn primary" on:click={() => addMemoryModalOpen = true}>
          <Plus size={14} />
          <span>Add Memory</span>
        </button>
      {/if}
    </div>
  </div>

  <!-- Error Display -->
  {#if error}
    <div class="error-banner" in:fly={{ y: -20 }}>
      <AlertCircle size={16} />
      <span>{error}</span>
      <button class="dismiss-btn" on:click={() => error = null}>
        <X size={14} />
      </button>
    </div>
  {/if}

  <!-- Department Selector -->
  <div class="department-bar">
    <div class="dept-label">Department:</div>
    <div class="dept-tabs">
      {#each DEPARTMENTS as dept}
        <button
          class="dept-tab"
          class:active={selectedDepartment === dept.id}
          style="--dept-color: {dept.color}"
          on:click={() => { selectedDepartment = dept.id; loadDepartmentMemory(); }}
        >
          {dept.name}
        </button>
      {/each}
    </div>
  </div>

  <!-- Stats Banner -->
  {#if stats}
    <div class="stats-banner">
      <div class="stat-item">
        <Hash size={14} />
        <span class="stat-label">Entries:</span>
        <span class="stat-value">{stats.total_entries}</span>
      </div>
      <div class="stat-item">
        <Tag size={14} />
        <span class="stat-label">Categories:</span>
        <span class="stat-value">{stats.categories.length}</span>
      </div>
      <div class="stat-item">
        <Calendar size={14} />
        <span class="stat-label">Daily Logs:</span>
        <span class="stat-value">{stats.daily_log_count}</span>
      </div>
    </div>
  {/if}

  <!-- Tabs -->
  <div class="tabs-bar">
    <button
      class="tab-btn"
      class:active={activeTab === 'memory'}
      on:click={() => activeTab = 'memory'}
    >
      <BookOpen size={14} />
      <span>Memory</span>
    </button>
    <button
      class="tab-btn"
      class:active={activeTab === 'logs'}
      on:click={() => activeTab = 'logs'}
    >
      <Clock size={14} />
      <span>Daily Logs</span>
    </button>
  </div>

  <!-- Search & Filter -->
  {#if activeTab === 'memory'}
    <div class="filter-bar">
      <div class="search-group">
        <Search size={14} />
        <input
          type="text"
          placeholder="Search memories..."
          bind:value={searchQuery}
          on:keydown={(e) => e.key === 'Enter' && searchMemories()}
        />
        <button class="search-btn" on:click={searchMemories}>Search</button>
      </div>

      <div class="filter-group">
        <Filter size={14} />
        <select bind:value={categoryFilter}>
          <option value="all">All Categories</option>
          {#each categories as cat}
            <option value={cat}>{cat}</option>
          {/each}
        </select>
      </div>
    </div>
  {/if}

  <!-- Content -->
  <div class="memory-content">
    {#if loading}
      <div class="loading-state">
        <RefreshCw size={32} class="spin" />
        <span>Loading memory...</span>
      </div>
    {:else if activeTab === 'memory'}
      <!-- Memory Entries -->
      {#if filteredMemories.length > 0}
        <div class="memory-list">
          {#each filteredMemories as memory, index}
            <div
              class="memory-card"
              class:expanded={expandedMemory === `${memory.timestamp}-${index}`}
              in:fly={{ y: 20 }}
            >
              <div class="memory-header" on:click={() => toggleExpanded(`${memory.timestamp}-${index}`)}>
                <div class="memory-info">
                  <div class="memory-category">
                    <Tag size={12} />
                    <span>{memory.category}</span>
                  </div>
                  <div class="memory-time">
                    <Clock size={12} />
                    <span>{formatTimestamp(memory.timestamp)}</span>
                  </div>
                </div>
                <div class="memory-expand">
                  {#if expandedMemory === `${memory.timestamp}-${index}`}
                    <ChevronUp size={16} />
                  {:else}
                    <ChevronDown size={16} />
                  {/if}
                </div>
              </div>

              {#if expandedMemory === `${memory.timestamp}-${index}`}
                <div class="memory-body">
                  <p class="memory-text">{memory.content}</p>
                  {#if memory.tags && memory.tags.length > 0}
                    <div class="memory-tags">
                      {#each memory.tags as tag}
                        <span class="tag">{tag}</span>
                      {/each}
                    </div>
                  {/if}
                </div>
              {/if}
            </div>
          {/each}
        </div>
      {:else}
        <div class="empty-state">
          <Brain size={48} />
          <p>No memories found</p>
          <button class="btn primary" on:click={() => addMemoryModalOpen = true}>
            <Plus size={14} />
            <span>Add First Memory</span>
          </button>
        </div>
      {/if}
    {:else}
      <!-- Daily Logs -->
      {#if dailyLogs.length > 0}
        <div class="logs-list">
          {#each dailyLogs as log}
            <div
              class="log-card"
              class:expanded={expandedLog === log.date}
              in:slide={{ y: 20 }}
            >
              <div class="log-header" on:click={() => viewDailyLog(log)}>
                <div class="log-info">
                  <div class="log-date">
                    <Calendar size={14} />
                    <span>{formatDate(log.date)}</span>
                  </div>
                  <div class="log-count">
                    <FileText size={12} />
                    <span>{log.entries.length} entries</span>
                  </div>
                </div>
                <div class="log-expand">
                  {#if expandedLog === log.date}
                    <ChevronUp size={16} />
                  {:else}
                    <ChevronDown size={16} />
                  {/if}
                </div>
              </div>

              {#if expandedLog === log.date && selectedLog}
                <div class="log-body">
                  {#each selectedLog.entries as entry}
                    <div class="log-entry">
                      <div class="entry-time">
                        <Clock size={10} />
                        <span>{entry.time}</span>
                      </div>
                      {#if entry.category}
                        <span class="entry-category">{entry.category}</span>
                      {/if}
                      <p class="entry-content">{entry.content}</p>
                    </div>
                  {/each}
                </div>
              {/if}
            </div>
          {/each}
        </div>
      {:else}
        <div class="empty-state">
          <Calendar size={48} />
          <p>No daily logs found</p>
        </div>
      {/if}
    {/if}
  </div>
</div>

<!-- Add Memory Modal -->
{#if addMemoryModalOpen}
  <div
    class="modal-overlay"
    on:click|self={() => addMemoryModalOpen = false}
    role="dialog"
    aria-modal="true"
  >
    <div class="modal">
      <div class="modal-header">
        <div>
          <h3>Add Memory</h3>
          <p class="modal-subtitle">Add to {selectedDepartment} department memory</p>
        </div>
        <button class="icon-btn" on:click={() => addMemoryModalOpen = false}>
          <X size={18} />
        </button>
      </div>

      <div class="modal-content">
        <div class="form-group">
          <label for="category-input">Category</label>
          <input
            id="category-input"
            type="text"
            bind:value={newMemory.category}
            placeholder="e.g., Decision, Observation, Learning"
          />
        </div>

        <div class="form-group">
          <label for="content-textarea">Content</label>
          <textarea
            id="content-textarea"
            bind:value={newMemory.content}
            placeholder="Enter memory content..."
            rows="6"
          ></textarea>
        </div>

        <div class="form-group">
          <label for="tags-input">Tags (comma-separated)</label>
          <input
            id="tags-input"
            type="text"
            placeholder="e.g., strategy, risk, important"
            on:change={(e) => {
              newMemory.tags = e.target.value.split(',').map(s => s.trim()).filter(s => s);
            }}
          />
        </div>
      </div>

      <div class="modal-actions">
        <button class="btn" on:click={() => addMemoryModalOpen = false}>Cancel</button>
        <button class="btn primary" on:click={addMemory}>
          <Plus size={14} />
          <span>Add Memory</span>
        </button>
      </div>
    </div>
  </div>
{/if}

<style>
  .memory-view {
    display: flex;
    flex-direction: column;
    height: 100%;
    background: var(--bg-primary);
    overflow: hidden;
  }

  /* Header */
  .memory-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 20px 24px;
    background: var(--bg-secondary);
    border-bottom: 1px solid var(--border-subtle);
  }

  .header-left {
    display: flex;
    align-items: center;
    gap: 12px;
  }

  .memory-icon {
    color: var(--accent-primary);
  }

  .header-left h2 {
    margin: 0;
    font-size: 18px;
    color: var(--text-primary);
  }

  .header-left p {
    margin: 2px 0 0;
    font-size: 12px;
    color: var(--text-muted);
  }

  .header-actions {
    display: flex;
    gap: 8px;
  }

  .btn {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 8px 14px;
    background: var(--bg-tertiary);
    border: 1px solid var(--border-subtle);
    border-radius: 6px;
    color: var(--text-secondary);
    font-size: 12px;
    cursor: pointer;
    transition: all 0.15s;
  }

  .btn:hover {
    background: var(--bg-surface);
  }

  .btn.primary {
    background: var(--accent-primary);
    border-color: var(--accent-primary);
    color: var(--bg-primary);
  }

  .btn.primary:hover {
    opacity: 0.9;
  }

  /* Error Banner */
  .error-banner {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 12px 24px;
    background: rgba(239, 68, 68, 0.1);
    border-bottom: 1px solid #ef4444;
    color: #ef4444;
    font-size: 12px;
  }

  .dismiss-btn {
    margin-left: auto;
    background: transparent;
    border: none;
    color: inherit;
    cursor: pointer;
    padding: 4px;
  }

  /* Department Bar */
  .department-bar {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 12px 24px;
    background: var(--bg-primary);
    border-bottom: 1px solid var(--border-subtle);
  }

  .dept-label {
    font-size: 12px;
    color: var(--text-muted);
    font-weight: 500;
  }

  .dept-tabs {
    display: flex;
    gap: 4px;
  }

  .dept-tab {
    padding: 6px 12px;
    background: transparent;
    border: none;
    border-radius: 6px;
    color: var(--text-secondary);
    font-size: 12px;
    cursor: pointer;
    transition: all 0.15s;
  }

  .dept-tab:hover {
    background: var(--bg-tertiary);
  }

  .dept-tab.active {
    background: var(--dept-color);
    color: var(--bg-primary);
  }

  /* Stats Banner */
  .stats-banner {
    display: flex;
    gap: 24px;
    padding: 12px 24px;
    background: var(--bg-secondary);
    border-bottom: 1px solid var(--border-subtle);
  }

  .stat-item {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 11px;
    color: var(--text-secondary);
  }

  .stat-label {
    color: var(--text-muted);
  }

  .stat-value {
    color: var(--text-primary);
    font-weight: 500;
  }

  /* Tabs */
  .tabs-bar {
    display: flex;
    gap: 4px;
    padding: 8px 24px 0;
    border-bottom: 1px solid var(--border-subtle);
    background: var(--bg-secondary);
  }

  .tab-btn {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 10px 16px;
    background: transparent;
    border: none;
    border-radius: 8px 8px 0 0;
    color: var(--text-muted);
    cursor: pointer;
    transition: all 0.15s;
    font-size: 13px;
  }

  .tab-btn:hover {
    background: var(--bg-tertiary);
    color: var(--text-primary);
  }

  .tab-btn.active {
    background: var(--bg-primary);
    color: var(--accent-primary);
  }

  /* Filter Bar */
  .filter-bar {
    display: flex;
    gap: 12px;
    padding: 16px 24px;
    background: var(--bg-primary);
    border-bottom: 1px solid var(--border-subtle);
  }

  .search-group,
  .filter-group {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 12px;
    background: var(--bg-secondary);
    border: 1px solid var(--border-subtle);
    border-radius: 6px;
  }

  .search-group input,
  .filter-group select {
    background: transparent;
    border: none;
    color: var(--text-primary);
    font-size: 12px;
    outline: none;
  }

  .search-group input {
    width: 200px;
  }

  .search-btn {
    padding: 4px 8px;
    background: var(--accent-primary);
    border: none;
    border-radius: 4px;
    color: var(--bg-primary);
    font-size: 11px;
    cursor: pointer;
  }

  /* Content */
  .memory-content {
    flex: 1;
    overflow-y: auto;
    padding: 16px 24px;
  }

  /* Memory List */
  .memory-list {
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  .memory-card {
    background: var(--bg-secondary);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
    overflow: hidden;
  }

  .memory-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 14px 16px;
    cursor: pointer;
    transition: background 0.15s;
  }

  .memory-header:hover {
    background: var(--bg-tertiary);
  }

  .memory-info {
    display: flex;
    flex-direction: column;
    gap: 6px;
  }

  .memory-category {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 13px;
    font-weight: 500;
    color: var(--text-primary);
  }

  .memory-category span {
    padding: 2px 8px;
    background: var(--bg-tertiary);
    border-radius: 4px;
    font-size: 11px;
  }

  .memory-time {
    display: flex;
    align-items: center;
    gap: 4px;
    font-size: 11px;
    color: var(--text-muted);
  }

  .memory-body {
    padding: 0 16px 16px;
  }

  .memory-text {
    margin: 0 0 12px;
    font-size: 13px;
    color: var(--text-secondary);
    line-height: 1.6;
    white-space: pre-wrap;
  }

  .memory-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
  }

  .tag {
    padding: 4px 10px;
    background: var(--bg-tertiary);
    border-radius: 6px;
    font-size: 11px;
    color: var(--accent-primary);
  }

  /* Logs List */
  .logs-list {
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  .log-card {
    background: var(--bg-secondary);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
    overflow: hidden;
  }

  .log-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 14px 16px;
    cursor: pointer;
    transition: background 0.15s;
  }

  .log-header:hover {
    background: var(--bg-tertiary);
  }

  .log-info {
    display: flex;
    flex-direction: column;
    gap: 6px;
  }

  .log-date {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 13px;
    font-weight: 500;
    color: var(--text-primary);
  }

  .log-count {
    display: flex;
    align-items: center;
    gap: 4px;
    font-size: 11px;
    color: var(--text-muted);
  }

  .log-body {
    padding: 0 16px 16px;
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  .log-entry {
    padding: 10px;
    background: var(--bg-tertiary);
    border-radius: 6px;
    border-left: 2px solid var(--accent-primary);
  }

  .entry-time {
    display: flex;
    align-items: center;
    gap: 4px;
    font-size: 11px;
    color: var(--text-muted);
    margin-bottom: 6px;
  }

  .entry-category {
    display: inline-block;
    padding: 2px 8px;
    background: var(--bg-surface);
    border-radius: 4px;
    font-size: 10px;
    color: var(--text-secondary);
    margin-bottom: 6px;
  }

  .entry-content {
    margin: 0;
    font-size: 12px;
    color: var(--text-secondary);
    white-space: pre-wrap;
  }

  /* Empty State */
  .empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 60px 24px;
    color: var(--text-muted);
    text-align: center;
    gap: 16px;
  }

  .empty-state p {
    margin: 0;
    font-size: 13px;
  }

  /* Loading State */
  .loading-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 60px 24px;
    color: var(--text-muted);
    gap: 16px;
  }

  @keyframes spin {
    to {
      transform: rotate(360deg);
    }
  }

  .spin {
    animation: spin 1s linear infinite;
  }

  /* Modal */
  .modal-overlay {
    position: fixed;
    inset: 0;
    background: rgba(0, 0, 0, 0.7);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 100;
  }

  .modal {
    background: var(--bg-secondary);
    border-radius: 12px;
    width: 500px;
    max-width: 90%;
    max-height: 85vh;
    overflow: hidden;
    display: flex;
    flex-direction: column;
  }

  .modal-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    padding: 16px 20px;
    border-bottom: 1px solid var(--border-subtle);
  }

  .modal-header h3 {
    margin: 0;
    font-size: 16px;
    color: var(--text-primary);
  }

  .modal-subtitle {
    margin: 4px 0 0;
    font-size: 11px;
    color: var(--text-muted);
  }

  .icon-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 28px;
    height: 28px;
    background: transparent;
    border: none;
    border-radius: 4px;
    color: var(--text-muted);
    cursor: pointer;
  }

  .icon-btn:hover {
    background: var(--bg-tertiary);
    color: var(--text-primary);
  }

  .modal-content {
    flex: 1;
    overflow-y: auto;
    padding: 20px;
  }

  .form-group {
    margin-bottom: 16px;
  }

  .form-group label {
    display: block;
    margin-bottom: 6px;
    font-size: 12px;
    font-weight: 500;
    color: var(--text-primary);
  }

  .form-group input,
  .form-group textarea {
    width: 100%;
    padding: 10px 12px;
    background: var(--bg-tertiary);
    border: 1px solid var(--border-subtle);
    border-radius: 6px;
    color: var(--text-primary);
    font-size: 13px;
    outline: none;
  }

  .form-group input:focus,
  .form-group textarea:focus {
    border-color: var(--accent-primary);
  }

  .modal-actions {
    display: flex;
    justify-content: flex-end;
    gap: 8px;
    padding: 16px 20px;
    border-top: 1px solid var(--border-subtle);
  }
</style>
