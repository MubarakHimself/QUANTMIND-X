<script lang="ts">
  import { self } from 'svelte/legacy';

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
    // OPINION node fields (from graph memory)
    node_type?: string;
    action?: string;
    reasoning?: string;
    confidence?: number;
    alternatives_considered?: string[];
    constraints_applied?: string[];
    agent_role?: string;
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
    { id: 'development', name: 'Development', color: '#3b82f6' },
    { id: 'research', name: 'Research', color: '#8b5cf6' },
    { id: 'risk', name: 'Risk', color: '#ef4444' },
    { id: 'trading', name: 'Trading', color: '#f97316' },
    { id: 'portfolio', name: 'Portfolio', color: '#10b981' },
    { id: 'floor_manager', name: 'Floor Manager', color: '#06b6d4' }
  ];

  // State
  let selectedDepartment = $state('development');
  let memories: MemoryEntry[] = $state([]);
  let dailyLogs: DailyLog[] = $state([]);
  let stats: MemoryStats | null = $state(null);
  let loading = $state(false);
  let error: string | null = $state(null);

  // View state
  let activeTab = $state('memory'); // 'memory' or 'logs'
  let expandedMemory: string | null = $state(null);
  let expandedLog: string | null = $state(null);
  let selectedLog: DailyLog | null = $state(null);
  let searchQuery = $state('');
  let categoryFilter = $state('all');

  // New memory modal
  let addMemoryModalOpen = $state(false);
  let newMemory = $state({
    category: '',
    content: '',
    tags: [] as string[]
  });

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
  let filteredMemories = $derived(memories.filter(mem => {
    if (categoryFilter !== 'all' && mem.category.toLowerCase() !== categoryFilter) return false;
    return true;
  }));

  // Get categories from memories
  let categories = $derived([...new Set(memories.map(m => m.category.toLowerCase()))]);

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
      <button class="btn" onclick={loadDepartmentMemory}>
        <RefreshCw size={14} />
        <span>Refresh</span>
      </button>
      {#if activeTab === 'memory'}
        <button class="btn primary" onclick={() => addMemoryModalOpen = true}>
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
      <button class="dismiss-btn" onclick={() => error = null}>
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
          onclick={() => { selectedDepartment = dept.id; loadDepartmentMemory(); }}
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
      onclick={() => activeTab = 'memory'}
    >
      <BookOpen size={14} />
      <span>Memory</span>
    </button>
    <button
      class="tab-btn"
      class:active={activeTab === 'logs'}
      onclick={() => activeTab = 'logs'}
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
          onkeydown={(e) => e.key === 'Enter' && searchMemories()}
        />
        <button class="search-btn" onclick={searchMemories}>Search</button>
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
              <div class="memory-header" onclick={() => toggleExpanded(`${memory.timestamp}-${index}`)}>
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

                  <!-- OPINION node fields -->
                  {#if memory.node_type === 'OPINION'}
                    <div class="opinion-fields">
                      {#if memory.action}
                        <div class="opinion-row">
                          <span class="opinion-label">Action</span>
                          <span class="opinion-value">{memory.action}</span>
                        </div>
                      {/if}
                      {#if memory.reasoning}
                        <div class="opinion-row">
                          <span class="opinion-label">Reasoning</span>
                          <span class="opinion-value">{memory.reasoning}</span>
                        </div>
                      {/if}
                      {#if memory.confidence != null}
                        <div class="opinion-row">
                          <span class="opinion-label">Confidence</span>
                          <div class="confidence-bar">
                            <div class="confidence-fill" style:width="{Math.round((memory.confidence ?? 0) * 100)}%"></div>
                            <span class="confidence-text">{Math.round((memory.confidence ?? 0) * 100)}%</span>
                          </div>
                        </div>
                      {/if}
                      {#if memory.alternatives_considered && memory.alternatives_considered.length > 0}
                        <div class="opinion-row">
                          <span class="opinion-label">Alternatives</span>
                          <div class="opinion-chips">
                            {#each memory.alternatives_considered as alt}
                              <span class="opinion-chip">{alt}</span>
                            {/each}
                          </div>
                        </div>
                      {/if}
                      {#if memory.constraints_applied && memory.constraints_applied.length > 0}
                        <div class="opinion-row">
                          <span class="opinion-label">Constraints</span>
                          <div class="opinion-chips">
                            {#each memory.constraints_applied as c}
                              <span class="opinion-chip constraint">{c}</span>
                            {/each}
                          </div>
                        </div>
                      {/if}
                      {#if memory.agent_role}
                        <div class="opinion-row">
                          <span class="opinion-label">Agent</span>
                          <span class="opinion-value agent-badge">{memory.agent_role}</span>
                        </div>
                      {/if}
                    </div>
                  {/if}

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
          <button class="btn primary" onclick={() => addMemoryModalOpen = true}>
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
              <div class="log-header" onclick={() => viewDailyLog(log)}>
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
    onclick={self(() => addMemoryModalOpen = false)}
    role="dialog"
    aria-modal="true"
  >
    <div class="modal">
      <div class="modal-header">
        <div>
          <h3>Add Memory</h3>
          <p class="modal-subtitle">Add to {selectedDepartment} department memory</p>
        </div>
        <button class="icon-btn" onclick={() => addMemoryModalOpen = false}>
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
            onchange={(e) => {
              newMemory.tags = e.target.value.split(',').map(s => s.trim()).filter(s => s);
            }}
          />
        </div>
      </div>

      <div class="modal-actions">
        <button class="btn" onclick={() => addMemoryModalOpen = false}>Cancel</button>
        <button class="btn primary" onclick={addMemory}>
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
    background: var(--color-bg-base);
    overflow: hidden;
  }

  /* Header */
  .memory-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 20px 24px;
    background: var(--color-bg-surface);
    border-bottom: 1px solid var(--color-border-subtle);
  }

  .header-left {
    display: flex;
    align-items: center;
    gap: 12px;
  }

  .memory-icon {
    color: var(--color-accent-cyan);
  }

  .header-left h2 {
    margin: 0;
    font-size: 18px;
    color: var(--color-text-primary);
  }

  .header-left p {
    margin: 2px 0 0;
    font-size: 12px;
    color: var(--color-text-muted);
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
    background: var(--color-bg-elevated);
    border: 1px solid var(--color-border-subtle);
    border-radius: 6px;
    color: var(--color-text-secondary);
    font-size: 12px;
    cursor: pointer;
    transition: all 0.15s;
  }

  .btn:hover {
    background: var(--bg-surface);
  }

  .btn.primary {
    background: var(--color-accent-cyan);
    border-color: var(--color-accent-cyan);
    color: var(--color-bg-base);
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
    background: var(--color-bg-base);
    border-bottom: 1px solid var(--color-border-subtle);
  }

  .dept-label {
    font-size: 12px;
    color: var(--color-text-muted);
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
    color: var(--color-text-secondary);
    font-size: 12px;
    cursor: pointer;
    transition: all 0.15s;
  }

  .dept-tab:hover {
    background: var(--color-bg-elevated);
  }

  .dept-tab.active {
    background: var(--dept-color);
    color: var(--color-bg-base);
  }

  /* Stats Banner */
  .stats-banner {
    display: flex;
    gap: 24px;
    padding: 12px 24px;
    background: var(--color-bg-surface);
    border-bottom: 1px solid var(--color-border-subtle);
  }

  .stat-item {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 11px;
    color: var(--color-text-secondary);
  }

  .stat-label {
    color: var(--color-text-muted);
  }

  .stat-value {
    color: var(--color-text-primary);
    font-weight: 500;
  }

  /* Tabs */
  .tabs-bar {
    display: flex;
    gap: 4px;
    padding: 8px 24px 0;
    border-bottom: 1px solid var(--color-border-subtle);
    background: var(--color-bg-surface);
  }

  .tab-btn {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 10px 16px;
    background: transparent;
    border: none;
    border-radius: 8px 8px 0 0;
    color: var(--color-text-muted);
    cursor: pointer;
    transition: all 0.15s;
    font-size: 13px;
  }

  .tab-btn:hover {
    background: var(--color-bg-elevated);
    color: var(--color-text-primary);
  }

  .tab-btn.active {
    background: var(--color-bg-base);
    color: var(--color-accent-cyan);
  }

  /* Filter Bar */
  .filter-bar {
    display: flex;
    gap: 12px;
    padding: 16px 24px;
    background: var(--color-bg-base);
    border-bottom: 1px solid var(--color-border-subtle);
  }

  .search-group,
  .filter-group {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 12px;
    background: var(--color-bg-surface);
    border: 1px solid var(--color-border-subtle);
    border-radius: 6px;
  }

  .search-group input,
  .filter-group select {
    background: transparent;
    border: none;
    color: var(--color-text-primary);
    font-size: 12px;
    outline: none;
  }

  .search-group input {
    width: 200px;
  }

  .search-btn {
    padding: 4px 8px;
    background: var(--color-accent-cyan);
    border: none;
    border-radius: 4px;
    color: var(--color-bg-base);
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
    background: var(--color-bg-surface);
    border: 1px solid var(--color-border-subtle);
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
    background: var(--color-bg-elevated);
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
    color: var(--color-text-primary);
  }

  .memory-category span {
    padding: 2px 8px;
    background: var(--color-bg-elevated);
    border-radius: 4px;
    font-size: 11px;
  }

  .memory-time {
    display: flex;
    align-items: center;
    gap: 4px;
    font-size: 11px;
    color: var(--color-text-muted);
  }

  .memory-body {
    padding: 0 16px 16px;
  }

  .memory-text {
    margin: 0 0 12px;
    font-size: 13px;
    color: var(--color-text-secondary);
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
    background: var(--color-bg-elevated);
    border-radius: 6px;
    font-size: 11px;
    color: var(--color-accent-cyan);
  }

  /* OPINION node fields */
  .opinion-fields {
    margin-top: 10px;
    padding: 10px;
    background: rgba(139, 92, 246, 0.06);
    border: 1px solid rgba(139, 92, 246, 0.15);
    border-radius: 6px;
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .opinion-row {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    font-size: 12px;
  }

  .opinion-label {
    flex-shrink: 0;
    width: 90px;
    font-weight: 600;
    color: rgba(139, 92, 246, 0.8);
    text-transform: uppercase;
    font-size: 10px;
    letter-spacing: 0.04em;
    padding-top: 2px;
  }

  .opinion-value {
    color: var(--color-text-secondary, #94a3b8);
    line-height: 1.4;
  }

  .confidence-bar {
    flex: 1;
    display: flex;
    align-items: center;
    gap: 8px;
    height: 18px;
  }

  .confidence-bar .confidence-fill {
    height: 6px;
    border-radius: 3px;
    background: linear-gradient(90deg, #8b5cf6, #06b6d4);
    transition: width 0.3s ease;
    min-width: 4px;
    max-width: 100%;
  }

  .confidence-bar .confidence-text {
    font-size: 11px;
    color: #8b5cf6;
    font-weight: 600;
    flex-shrink: 0;
  }

  .opinion-chips {
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
  }

  .opinion-chip {
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 11px;
    background: rgba(139, 92, 246, 0.1);
    color: rgba(139, 92, 246, 0.8);
    border: 1px solid rgba(139, 92, 246, 0.15);
  }

  .opinion-chip.constraint {
    background: rgba(239, 68, 68, 0.08);
    color: rgba(239, 68, 68, 0.7);
    border-color: rgba(239, 68, 68, 0.15);
  }

  .agent-badge {
    padding: 2px 8px;
    border-radius: 4px;
    background: rgba(6, 182, 212, 0.1);
    color: #06b6d4;
    font-size: 11px;
    font-weight: 500;
  }

  /* Logs List */
  .logs-list {
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  .log-card {
    background: var(--color-bg-surface);
    border: 1px solid var(--color-border-subtle);
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
    background: var(--color-bg-elevated);
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
    color: var(--color-text-primary);
  }

  .log-count {
    display: flex;
    align-items: center;
    gap: 4px;
    font-size: 11px;
    color: var(--color-text-muted);
  }

  .log-body {
    padding: 0 16px 16px;
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  .log-entry {
    padding: 10px;
    background: var(--color-bg-elevated);
    border-radius: 6px;
    border-left: 2px solid var(--color-accent-cyan);
  }

  .entry-time {
    display: flex;
    align-items: center;
    gap: 4px;
    font-size: 11px;
    color: var(--color-text-muted);
    margin-bottom: 6px;
  }

  .entry-category {
    display: inline-block;
    padding: 2px 8px;
    background: var(--bg-surface);
    border-radius: 4px;
    font-size: 10px;
    color: var(--color-text-secondary);
    margin-bottom: 6px;
  }

  .entry-content {
    margin: 0;
    font-size: 12px;
    color: var(--color-text-secondary);
    white-space: pre-wrap;
  }

  /* Empty State */
  .empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 60px 24px;
    color: var(--color-text-muted);
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
    color: var(--color-text-muted);
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
    background: var(--color-bg-surface);
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
    border-bottom: 1px solid var(--color-border-subtle);
  }

  .modal-header h3 {
    margin: 0;
    font-size: 16px;
    color: var(--color-text-primary);
  }

  .modal-subtitle {
    margin: 4px 0 0;
    font-size: 11px;
    color: var(--color-text-muted);
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
    color: var(--color-text-muted);
    cursor: pointer;
  }

  .icon-btn:hover {
    background: var(--color-bg-elevated);
    color: var(--color-text-primary);
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
    color: var(--color-text-primary);
  }

  .form-group input,
  .form-group textarea {
    width: 100%;
    padding: 10px 12px;
    background: var(--color-bg-elevated);
    border: 1px solid var(--color-border-subtle);
    border-radius: 6px;
    color: var(--color-text-primary);
    font-size: 13px;
    outline: none;
  }

  .form-group input:focus,
  .form-group textarea:focus {
    border-color: var(--color-accent-cyan);
  }

  .modal-actions {
    display: flex;
    justify-content: flex-end;
    gap: 8px;
    padding: 16px 20px;
    border-top: 1px solid var(--color-border-subtle);
  }
</style>
