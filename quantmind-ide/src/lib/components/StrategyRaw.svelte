<script lang="ts">
  import { onMount } from 'svelte';
  import { fade, fly, slide } from 'svelte/transition';
  import {
    FileText, Video, FileType, Search, Filter, RefreshCw, Eye, Download,
    ChevronDown, ChevronUp, ExternalLink, Calendar, Hash, Tag, BookOpen,
    Play, CheckCircle, AlertCircle, Clock, User, FolderOpen, X, Plus
  } from 'lucide-svelte';

  // Types
  interface StrategyRaw {
    id: string;
    name: string;
    source_type: 'video' | 'pdf' | 'text';
    source_url?: string;
    extracted_at: string;
    trd_id?: string;
    status: 'pending' | 'processing' | 'completed' | 'failed';
    raw_content: string;
    metadata: Record<string, any>;
  }

  interface TRDDocument {
    id: string;
    strategy_id: string;
    created_at: string;
    content: string;
  }

  // State
  let strategies: StrategyRaw[] = [];
  let filteredStrategies: StrategyRaw[] = [];
  let selectedStrategy: StrategyRaw | null = null;
  let loading = false;
  let error: string | null = null;

  // View state
  let expandedStrategy: string | null = null;
  let detailModalOpen = false;
  let activeTab = 'strategies'; // 'strategies' or 'trd'

  // Filters
  let searchQuery = '';
  let sourceFilter = 'all';
  let statusFilter = 'all';

  const API_BASE = 'http://localhost:8000/api';

  // Lifecycle
  onMount(() => {
    loadStrategies();
  });

  // Load strategies
  async function loadStrategies() {
    loading = true;
    error = null;

    try {
      const res = await fetch(`${API_BASE}/strategies-raw`);
      if (!res.ok) throw new Error('Failed to load strategies');

      const data = await res.json();
      strategies = data.strategies || [];
      applyFilters();
    } catch (e) {
      error = e instanceof Error ? e.message : 'Failed to load strategies';
      console.error('Failed to load strategies:', e);
    } finally {
      loading = false;
    }
  }

  // Apply filters
  function applyFilters() {
    filteredStrategies = strategies.filter(strategy => {
      // Source filter
      if (sourceFilter !== 'all' && strategy.source_type !== sourceFilter) return false;

      // Status filter
      if (statusFilter !== 'all' && strategy.status !== statusFilter) return false;

      // Search query
      if (searchQuery) {
        const query = searchQuery.toLowerCase();
        return (
          strategy.name.toLowerCase().includes(query) ||
          strategy.raw_content.toLowerCase().includes(query) ||
          (strategy.source_url && strategy.source_url.toLowerCase().includes(query))
        );
      }

      return true;
    });
  }

  // Toggle expanded
  function toggleExpanded(id: string) {
    expandedStrategy = expandedStrategy === id ? null : id;
  }

  // View strategy details
  function viewStrategy(strategy: StrategyRaw) {
    selectedStrategy = strategy;
    detailModalOpen = true;
  }

  // Format timestamp
  function formatTimestamp(isoString: string): string {
    const date = new Date(isoString);
    return date.toLocaleString();
  }

  // Get source icon
  function getSourceIcon(type: string) {
    switch (type) {
      case 'video': return Video;
      case 'pdf': return FileType;
      case 'text': return FileText;
      default: return FileText;
    }
  }

  // Get source color
  function getSourceColor(type: string): string {
    switch (type) {
      case 'video': return '#ef4444';
      case 'pdf': return '#f59e0b';
      case 'text': return '#3b82f6';
      default: return '#6b7280';
    }
  }

  // Get status color
  function getStatusColor(status: string): string {
    switch (status) {
      case 'completed': return '#10b981';
      case 'processing': return '#f59e0b';
      case 'pending': return '#6b7280';
      case 'failed': return '#ef4444';
      default: return '#6b7280';
    }
  }

  // Get status icon
  function getStatusIcon(status: string) {
    switch (status) {
      case 'completed': return CheckCircle;
      case 'failed': return AlertCircle;
      default: return Clock;
    }
  }
</script>

<div class="strategy-raw-view">
  <!-- Header -->
  <div class="strategy-header">
    <div class="header-left">
      <FileText size={24} class="strategy-icon" />
      <div>
        <h2>Strategy Raw Data</h2>
        <p>Raw extracted strategy data from videos, PDFs, and text sources</p>
      </div>
    </div>
    <div class="header-actions">
      <button class="btn" on:click={loadStrategies}>
        <RefreshCw size={14} />
        <span>Refresh</span>
      </button>
      <button class="btn primary">
        <Plus size={14} />
        <span>Extract Strategy</span>
      </button>
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

  <!-- Tabs -->
  <div class="tabs-bar">
    <button
      class="tab-btn"
      class:active={activeTab === 'strategies'}
      on:click={() => activeTab = 'strategies'}
    >
      <FileText size={14} />
      <span>Extracted Strategies</span>
    </button>
    <button
      class="tab-btn"
      class:active={activeTab === 'trd'}
      on:click={() => activeTab = 'trd'}
    >
      <BookOpen size={14} />
      <span>TRD Documents</span>
    </button>
  </div>

  <!-- Filter Bar -->
  <div class="filter-bar">
    <div class="search-group">
      <Search size={14} />
      <input
        type="text"
        placeholder="Search strategies..."
        bind:value={searchQuery}
        on:input={applyFilters}
      />
    </div>

    <div class="filter-group">
      <Filter size={14} />
      <select bind:value={sourceFilter} on:change={applyFilters}>
        <option value="all">All Sources</option>
        <option value="video">Videos</option>
        <option value="pdf">PDFs</option>
        <option value="text">Text</option>
      </select>
    </div>

    <div class="filter-group">
      <Tag size={14} />
      <select bind:value={statusFilter} on:change={applyFilters}>
        <option value="all">All Status</option>
        <option value="completed">Completed</option>
        <option value="processing">Processing</option>
        <option value="pending">Pending</option>
        <option value="failed">Failed</option>
      </select>
    </div>

    <div class="stats-summary">
      <span class="stat-item">{filteredStrategies.length} strategies</span>
    </div>
  </div>

  <!-- Content -->
  <div class="strategy-content">
    {#if loading}
      <div class="loading-state">
        <RefreshCw size={32} class="spin" />
        <span>Loading strategies...</span>
      </div>
    {:else if activeTab === 'strategies'}
      <!-- Strategies List -->
      {#if filteredStrategies.length > 0}
        <div class="strategies-list">
          {#each filteredStrategies as strategy}
            <div
              class="strategy-card"
              class:expanded={expandedStrategy === strategy.id}
              in:fly={{ y: 20 }}
            >
              <div class="strategy-header" on:click={() => toggleExpanded(strategy.id)}>
                <div class="strategy-info">
                  <div class="strategy-name">
                    <svelte:component this={getSourceIcon(strategy.source_type)} size={14} style="color: {getSourceColor(strategy.source_type)}" />
                    <span>{strategy.name}</span>
                  </div>
                  <div class="strategy-meta">
                    <span class="source-type" style="background: {getSourceColor(strategy.source_type)}20; color: {getSourceColor(strategy.source_type)}">
                      {strategy.source_type.toUpperCase()}
                    </span>
                    <span class="status-badge" style="color: {getStatusColor(strategy.status)}">
                      <svelte:component this={getStatusIcon(strategy.status)} size={10} />
                      <span>{strategy.status}</span>
                    </span>
                    <span class="extracted-time">
                      <Clock size={10} />
                      <span>{formatTimestamp(strategy.extracted_at)}</span>
                    </span>
                  </div>
                </div>
                <div class="strategy-actions">
                  <button
                    class="icon-btn"
                    on:click|stopPropagation={() => viewStrategy(strategy)}
                    title="View details"
                  >
                    <Eye size={14} />
                  </button>
                  {#if strategy.source_url}
                    <a
                      href={strategy.source_url}
                      target="_blank"
                      rel="noopener"
                      class="icon-btn"
                      title="Open source"
                    >
                      <ExternalLink size={14} />
                    </a>
                  {/if}
                  <div class="expand-icon">
                    {#if expandedStrategy === strategy.id}
                      <ChevronUp size={16} />
                    {:else}
                      <ChevronDown size={16} />
                    {/if}
                  </div>
                </div>
              </div>

              {#if expandedStrategy === strategy.id}
                <div class="strategy-body">
                  {#if strategy.trd_id}
                    <div class="trd-ref">
                      <Hash size={12} />
                      <span>TRD: {strategy.trd_id}</span>
                    </div>
                  {/if}
                  <div class="raw-content">
                    <pre>{strategy.raw_content.slice(0, 500)}{strategy.raw_content.length > 500 ? '...' : ''}</pre>
                  </div>
                  {#if Object.keys(strategy.metadata).length > 0}
                    <div class="metadata">
                      <h4>Metadata</h4>
                      {#each Object.entries(strategy.metadata) as [key, value]}
                        <div class="meta-item">
                          <span class="meta-key">{key}:</span>
                          <span class="meta-value">{String(value)}</span>
                        </div>
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
          <FileText size={48} />
          <p>No extracted strategies found</p>
          <button class="btn primary" on:click={loadStrategies}>
            <RefreshCw size={14} />
            <span>Refresh</span>
          </button>
        </div>
      {/if}
    {:else}
      <!-- TRD Documents -->
      <div class="trd-content">
        <div class="empty-state">
          <BookOpen size={48} />
          <p>TRD documents will appear here</p>
          <p class="hint">Generate a TRD from an extracted strategy first</p>
        </div>
      </div>
    {/if}
  </div>
</div>

<!-- Detail Modal -->
{#if detailModalOpen && selectedStrategy}
  <div
    class="modal-overlay"
    on:click|self={() => detailModalOpen = false}
    role="dialog"
    aria-modal="true"
  >
    <div class="modal large">
      <div class="modal-header">
        <div>
          <h3>{selectedStrategy.name}</h3>
          <p class="modal-subtitle">
            <span class="source-badge" style="background: {getSourceColor(selectedStrategy.source_type)}20; color: {getSourceColor(selectedStrategy.source_type)}">
              {selectedStrategy.source_type.toUpperCase()}
            </span>
            <span class="status-badge" style="color: {getStatusColor(selectedStrategy.status)}">
              {selectedStrategy.status}
            </span>
          </p>
        </div>
        <button class="icon-btn" on:click={() => detailModalOpen = false}>
          <X size={18} />
        </button>
      </div>

      <div class="modal-content">
        <div class="detail-section">
          <h4>Raw Content</h4>
          <pre class="raw-content-full">{selectedStrategy.raw_content}</pre>
        </div>

        {#if selectedStrategy.source_url}
          <div class="detail-section">
            <h4>Source</h4>
            <a href={selectedStrategy.source_url} target="_blank" rel="noopener" class="source-link">
              <ExternalLink size={12} />
              <span>{selectedStrategy.source_url}</span>
            </a>
          </div>
        {/if}

        {#if Object.keys(selectedStrategy.metadata).length > 0}
          <div class="detail-section">
            <h4>Metadata</h4>
            <div class="metadata-grid">
              {#each Object.entries(selectedStrategy.metadata) as [key, value]}
                <div class="meta-item-full">
                  <span class="meta-key">{key}:</span>
                  <span class="meta-value">{String(value)}</span>
                </div>
              {/each}
            </div>
          </div>
        {/if}

        <div class="detail-actions">
          <button class="btn">
            <Download size={14} />
            <span>Export</span>
          </button>
          {#if selectedStrategy.trd_id}
            <button class="btn">
              <BookOpen size={14} />
              <span>View TRD</span>
            </button>
          {:else}
            <button class="btn primary">
              <Plus size={14} />
              <span>Generate TRD</span>
            </button>
          {/if}
        </div>
      </div>
    </div>
  </div>
{/if}

<style>
  .strategy-raw-view {
    display: flex;
    flex-direction: column;
    height: 100%;
    background: var(--bg-primary);
    overflow: hidden;
  }

  /* Header */
  .strategy-header {
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

  .strategy-icon {
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
    flex-wrap: wrap;
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

  .stats-summary {
    margin-left: auto;
    display: flex;
    gap: 16px;
    font-size: 11px;
    color: var(--text-muted);
  }

  /* Content */
  .strategy-content {
    flex: 1;
    overflow-y: auto;
    padding: 16px 24px;
  }

  /* Strategies List */
  .strategies-list {
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  .strategy-card {
    background: var(--bg-secondary);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
    overflow: hidden;
  }

  .strategy-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 14px 16px;
    cursor: pointer;
    transition: background 0.15s;
  }

  .strategy-header:hover {
    background: var(--bg-tertiary);
  }

  .strategy-info {
    flex: 1;
  }

  .strategy-name {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 14px;
    font-weight: 500;
    color: var(--text-primary);
    margin-bottom: 6px;
  }

  .strategy-meta {
    display: flex;
    gap: 12px;
    font-size: 11px;
  }

  .source-type,
  .status-badge,
  .extracted-time {
    display: flex;
    align-items: center;
    gap: 4px;
  }

  .source-type,
  .status-badge {
    padding: 2px 8px;
    border-radius: 4px;
    font-weight: 500;
  }

  .extracted-time {
    color: var(--text-muted);
  }

  .strategy-actions {
    display: flex;
    align-items: center;
    gap: 4px;
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
    text-decoration: none;
    transition: all 0.15s;
  }

  .icon-btn:hover {
    background: var(--bg-tertiary);
    color: var(--text-primary);
  }

  .expand-icon {
    display: flex;
    align-items: center;
    margin-left: 8px;
    color: var(--text-muted);
  }

  .strategy-body {
    padding: 0 16px 16px;
  }

  .trd-ref {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 8px 12px;
    background: var(--bg-tertiary);
    border-radius: 6px;
    margin-bottom: 12px;
    font-size: 12px;
    color: var(--accent-primary);
  }

  .raw-content {
    background: var(--bg-tertiary);
    border-radius: 6px;
    padding: 12px;
    margin-bottom: 12px;
  }

  .raw-content pre {
    margin: 0;
    font-size: 12px;
    color: var(--text-secondary);
    white-space: pre-wrap;
    word-break: break-word;
  }

  .metadata {
    padding: 12px;
    background: var(--bg-tertiary);
    border-radius: 6px;
  }

  .metadata h4 {
    margin: 0 0 10px;
    font-size: 12px;
    color: var(--text-muted);
  }

  .meta-item {
    display: flex;
    gap: 8px;
    font-size: 11px;
    margin-bottom: 4px;
  }

  .meta-key {
    color: var(--text-muted);
    font-weight: 500;
  }

  .meta-value {
    color: var(--text-primary);
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
  }

  .hint {
    font-size: 11px;
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
    to { transform: rotate(360deg); }
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
    width: 700px;
    max-width: 90%;
    max-height: 85vh;
    overflow: hidden;
    display: flex;
    flex-direction: column;
  }

  .modal.large {
    width: 900px;
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
    display: flex;
    gap: 8px;
    font-size: 11px;
  }

  .modal-content {
    flex: 1;
    overflow-y: auto;
    padding: 20px;
  }

  .detail-section {
    margin-bottom: 20px;
  }

  .detail-section h4 {
    margin: 0 0 10px;
    font-size: 13px;
    color: var(--text-primary);
  }

  .raw-content-full {
    background: var(--bg-tertiary);
    border-radius: 8px;
    padding: 16px;
    margin: 0;
    font-size: 12px;
    color: var(--text-secondary);
    white-space: pre-wrap;
    word-break: break-word;
    max-height: 400px;
    overflow-y: auto;
  }

  .source-link {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 8px 12px;
    background: var(--bg-tertiary);
    border-radius: 6px;
    color: var(--accent-primary);
    text-decoration: none;
    font-size: 12px;
  }

  .metadata-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 8px;
  }

  .meta-item-full {
    display: flex;
    gap: 8px;
    padding: 8px;
    background: var(--bg-tertiary);
    border-radius: 6px;
    font-size: 12px;
  }

  .detail-actions {
    display: flex;
    gap: 8px;
    padding-top: 16px;
    border-top: 1px solid var(--border-subtle);
  }

  .detail-actions .btn {
    flex: 1;
    justify-content: center;
  }
</style>
