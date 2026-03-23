<script lang="ts">
  /**
   * NewsView - Full News View for Research Canvas
   *
   * Displays historical news with filtering capabilities:
   * - Filter by impact tier (HIGH/MEDIUM/LOW)
   * - Filter by symbol
   * - Filter by date range
   * - Pagination or infinite scroll
   * - Send to Copilot functionality
   */
  import { onMount, onDestroy } from 'svelte';
  import GlassTile from '$lib/components/live-trading/GlassTile.svelte';
  import { newsStore, filteredNews, newsLoading, newsError, currentNewsFilter, initNewsStore, cleanupNewsStore, type NewsSeverity } from '$lib/stores/news';
  import type { NewsFeedItem } from '$lib/api/newsApi';
  import {
    Search,
    Filter,
    ChevronDown,
    Clock,
    AlertTriangle,
    Newspaper,
    Loader,
    AlertCircle,
    Send,
    X
  } from 'lucide-svelte';

  // Filter state
  let showFilters = false;
  let searchQuery = '';
  let severityFilter: NewsSeverity | 'all' = 'all';
  let symbolSearch = '';
  let dateFrom = '';
  let dateTo = '';

  // View state
  let selectedItem: NewsFeedItem | null = null;
  let showToast = false;

  onMount(() => {
    initNewsStore();
  });

  onDestroy(() => {
    cleanupNewsStore();
  });

  /**
   * Apply filters
   */
  function applyFilters() {
    newsStore.setFilter({
      severity: severityFilter,
      symbols: symbolSearch ? [symbolSearch.toUpperCase()] : [],
      dateFrom: dateFrom || null,
      dateTo: dateTo || null
    });
    showFilters = false;
  }

  /**
   * Clear all filters
   */
  function clearFilters() {
    severityFilter = 'all';
    symbolSearch = '';
    dateFrom = '';
    dateTo = '';
    newsStore.clearFilter();
    showFilters = false;
  }

  /**
   * Handle search query
   */
  function handleSearch() {
    // Filter items by headline search
    if (searchQuery) {
      newsStore.setFilter({
        severity: severityFilter,
        symbols: symbolSearch ? [symbolSearch.toUpperCase()] : [],
        dateFrom: dateFrom || null,
        dateTo: dateTo || null
      });
    }
  }

  /**
   * View item details
   */
  function viewItem(item: NewsFeedItem) {
    selectedItem = item;
  }

  /**
   * Go back from detail view
   */
  function goBack() {
    selectedItem = null;
  }

  /**
   * Send to Copilot
   */
  function sendToCopilot(item: NewsFeedItem) {
    // TODO: Implement actual copilot integration
    showToast = true;
    setTimeout(() => {
      showToast = false;
    }, 2000);
  }

  /**
   * Format timestamp
   */
  function formatTimestamp(isoString: string): string {
    const date = new Date(isoString);
    return date.toLocaleString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
      timeZone: 'UTC'
    }) + ' UTC';
  }

  /**
   * Get severity badge class
   */
  function getSeverityClass(severity: string | null | undefined): string {
    switch (severity) {
      case 'HIGH': return 'severity-high';
      case 'MEDIUM': return 'severity-medium';
      case 'LOW': return 'severity-low';
      default: return 'severity-unknown';
    }
  }

  /**
   * Filter news by search query
   */
  $: displayItems = searchQuery
    ? $filteredNews.filter(item =>
        item.headline.toLowerCase().includes(searchQuery.toLowerCase())
      )
    : $filteredNews;
</script>

<div class="news-view">
  {#if selectedItem}
    <!-- Detail View -->
    <div class="detail-view">
      <button class="back-btn" on:click={goBack}>
        <ChevronDown size={16} class="rotate-90" />
        Back to News
      </button>

      <GlassTile>
        <div class="detail-content">
          <div class="detail-header">
            <span class="severity-badge {getSeverityClass(selectedItem.severity)}">
              {#if selectedItem.severity === 'HIGH'}
                <AlertTriangle size={12} />
              {/if}
              {selectedItem.severity || 'Unknown'}
            </span>
            <button class="copilot-btn" on:click={() => sendToCopilot(selectedItem)}>
              <Send size={14} />
              Send to Copilot
            </button>
          </div>

          <h2>{selectedItem.headline}</h2>

          {#if selectedItem.summary}
            <p class="summary">{selectedItem.summary}</p>
          {/if}

          <div class="detail-meta">
            {#if selectedItem.source}
              <span class="meta-item">
                <Newspaper size={14} />
                {selectedItem.source}
              </span>
            {/if}
            <span class="meta-item">
              <Clock size={14} />
              {formatTimestamp(selectedItem.published_utc)}
            </span>
          </div>

          {#if selectedItem.related_instruments && selectedItem.related_instruments.length > 0}
            <div class="symbols">
              <span class="symbols-label">Related Symbols:</span>
              {#each selectedItem.related_instruments as symbol}
                <span class="symbol-tag">{symbol}</span>
              {/each}
            </div>
          {/if}

          {#if selectedItem.url}
            <a href={selectedItem.url} target="_blank" rel="noopener" class="source-link">
              Read full article
            </a>
          {/if}
        </div>
      </GlassTile>
    </div>
  {:else}
    <!-- List View -->
    <div class="list-view">
      <!-- Search and Filter Bar -->
      <div class="search-filter-bar">
        <div class="search-input-wrapper">
          <Search size={16} class="search-icon" />
          <input
            type="text"
            bind:value={searchQuery}
            on:input={handleSearch}
            placeholder="Search news..."
            class="search-input"
          />
        </div>

        <button class="filter-btn" on:click={() => showFilters = !showFilters}>
          <Filter size={16} />
          Filters
          <span class="chevron" class:rotated={showFilters}><ChevronDown size={14} /></span>
        </button>
      </div>

      <!-- Filter Panel -->
      {#if showFilters}
        <div class="filter-panel">
          <div class="filter-group">
            <label>Severity</label>
            <div class="severity-options">
              <button
                class="severity-option"
                class:active={severityFilter === 'all'}
                on:click={() => severityFilter = 'all'}
              >All</button>
              <button
                class="severity-option severity-high"
                class:active={severityFilter === 'HIGH'}
                on:click={() => severityFilter = 'HIGH'}
              >HIGH</button>
              <button
                class="severity-option severity-medium"
                class:active={severityFilter === 'MEDIUM'}
                on:click={() => severityFilter = 'MEDIUM'}
              >MEDIUM</button>
              <button
                class="severity-option severity-low"
                class:active={severityFilter === 'LOW'}
                on:click={() => severityFilter = 'LOW'}
              >LOW</button>
            </div>
          </div>

          <div class="filter-group">
            <label>Symbol</label>
            <input
              type="text"
              bind:value={symbolSearch}
              placeholder="e.g., EURUSD"
              class="filter-input"
            />
          </div>

          <div class="filter-group">
            <label>Date Range</label>
            <div class="date-inputs">
              <input
                type="date"
                bind:value={dateFrom}
                class="filter-input"
              />
              <span>to</span>
              <input
                type="date"
                bind:value={dateTo}
                class="filter-input"
              />
            </div>
          </div>

          <div class="filter-actions">
            <button class="clear-btn" on:click={clearFilters}>
              Clear
            </button>
            <button class="apply-btn" on:click={applyFilters}>
              Apply Filters
            </button>
          </div>
        </div>
      {/if}

      <!-- News List -->
      <div class="news-list">
        {#if $newsError}
          <div class="error-state">
            <AlertCircle size={24} />
            <span>Failed to load news</span>
            <button class="retry-btn" on:click={() => newsStore.fetchNews(50)}>
              Retry
            </button>
          </div>
        {:else if $newsLoading && displayItems.length === 0}
          <div class="loading-state">
            <Loader size={24} class="spin" />
            <span>Loading news...</span>
          </div>
        {:else if displayItems.length === 0}
          <div class="empty-state">
            <Newspaper size={32} />
            <span>No news found</span>
            <span class="empty-hint">Try adjusting your filters</span>
          </div>
        {:else}
          {#each displayItems as item (item.item_id)}
            <button class="news-item" on:click={() => viewItem(item)}>
              <div class="item-header">
                <span class="item-headline">{item.headline}</span>
                <span class="severity-badge {getSeverityClass(item.severity)}">
                  {#if item.severity === 'HIGH'}
                    <AlertTriangle size={10} />
                  {/if}
                  {item.severity || 'N/A'}
                </span>
              </div>
              <div class="item-meta">
                <span class="item-source">{item.source || 'Unknown'}</span>
                <span class="item-time">
                  <Clock size={12} />
                  {formatTimestamp(item.published_utc)}
                </span>
              </div>
              {#if item.related_instruments && item.related_instruments.length > 0}
                <div class="item-symbols">
                  {#each item.related_instruments.slice(0, 3) as symbol}
                    <span class="symbol-tag">{symbol}</span>
                  {/each}
                  {#if item.related_instruments.length > 3}
                    <span class="symbol-more">+{item.related_instruments.length - 3}</span>
                  {/if}
                </div>
              {/if}
            </button>
          {/each}
        {/if}
      </div>

      <!-- Results count -->
      {#if displayItems.length > 0}
        <div class="results-count">
          {displayItems.length} news items
        </div>
      {/if}
    </div>
  {/if}

  <!-- Toast -->
  {#if showToast}
    <div class="toast">
      <Send size={14} />
      <span>Sent to Copilot</span>
    </div>
  {/if}
</div>

<style>
  .news-view {
    display: flex;
    flex-direction: column;
    gap: 16px;
    height: 100%;
  }

  /* Search and Filter Bar */
  .search-filter-bar {
    display: flex;
    gap: 12px;
    align-items: center;
  }

  .search-input-wrapper {
    flex: 1;
    display: flex;
    align-items: center;
    gap: 8px;
    background: rgba(8, 13, 20, 0.6);
    border: 1px solid rgba(0, 212, 255, 0.15);
    border-radius: 8px;
    padding: 10px 14px;
  }

  .search-input-wrapper:focus-within {
    border-color: rgba(0, 212, 255, 0.3);
  }

  .search-input-wrapper :global(.search-icon) {
    color: rgba(0, 212, 255, 0.5);
    flex-shrink: 0;
  }

  .search-input {
    flex: 1;
    background: transparent;
    border: none;
    color: #e0e0e0;
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    outline: none;
  }

  .search-input::placeholder {
    color: rgba(224, 224, 224, 0.4);
  }

  .filter-btn {
    display: flex;
    align-items: center;
    gap: 6px;
    background: rgba(0, 212, 255, 0.1);
    border: 1px solid rgba(0, 212, 255, 0.2);
    color: #00d4ff;
    padding: 10px 14px;
    border-radius: 8px;
    cursor: pointer;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    transition: all 0.2s;
  }

  .filter-btn:hover {
    background: rgba(0, 212, 255, 0.2);
  }

  .filter-btn .chevron {
    display: inline-flex;
    transition: transform 0.2s;
  }

  .filter-btn .chevron.rotated {
    transform: rotate(180deg);
  }

  /* Filter Panel */
  .filter-panel {
    background: rgba(8, 13, 20, 0.6);
    border: 1px solid rgba(0, 212, 255, 0.1);
    border-radius: 8px;
    padding: 16px;
    display: flex;
    flex-direction: column;
    gap: 16px;
  }

  .filter-group {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .filter-group label {
    font-size: 11px;
    color: rgba(224, 224, 224, 0.6);
    font-family: 'JetBrains Mono', monospace;
    text-transform: uppercase;
  }

  .severity-options {
    display: flex;
    gap: 8px;
  }

  .severity-option {
    padding: 6px 12px;
    border-radius: 4px;
    font-size: 11px;
    font-family: 'JetBrains Mono', monospace;
    background: rgba(100, 100, 100, 0.2);
    border: 1px solid rgba(100, 100, 100, 0.2);
    color: #e0e0e0;
    cursor: pointer;
    transition: all 0.2s;
  }

  .severity-option:hover {
    border-color: rgba(0, 212, 255, 0.3);
  }

  .severity-option.active {
    background: rgba(0, 212, 255, 0.2);
    border-color: rgba(0, 212, 255, 0.4);
    color: #00d4ff;
  }

  .severity-option.severity-high.active {
    background: rgba(255, 71, 87, 0.2);
    border-color: rgba(255, 71, 87, 0.4);
    color: #ff4757;
  }

  .severity-option.severity-medium.active {
    background: rgba(240, 165, 0, 0.2);
    border-color: rgba(240, 165, 0, 0.4);
    color: #f0a500;
  }

  .severity-option.severity-low.active {
    background: rgba(100, 100, 100, 0.3);
    border-color: rgba(100, 100, 100, 0.5);
    color: rgba(200, 200, 200, 0.9);
  }

  .filter-input {
    background: rgba(0, 0, 0, 0.3);
    border: 1px solid rgba(0, 212, 255, 0.1);
    border-radius: 4px;
    padding: 8px 12px;
    color: #e0e0e0;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    outline: none;
  }

  .filter-input:focus {
    border-color: rgba(0, 212, 255, 0.3);
  }

  .date-inputs {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .date-inputs span {
    color: rgba(224, 224, 224, 0.5);
    font-size: 12px;
  }

  .filter-actions {
    display: flex;
    justify-content: flex-end;
    gap: 8px;
    margin-top: 8px;
  }

  .clear-btn {
    padding: 8px 16px;
    border-radius: 4px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    background: transparent;
    border: 1px solid rgba(100, 100, 100, 0.3);
    color: rgba(224, 224, 224, 0.7);
    cursor: pointer;
    transition: all 0.2s;
  }

  .clear-btn:hover {
    border-color: rgba(0, 212, 255, 0.3);
    color: #e0e0e0;
  }

  .apply-btn {
    padding: 8px 16px;
    border-radius: 4px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    background: rgba(0, 212, 255, 0.15);
    border: 1px solid rgba(0, 212, 255, 0.3);
    color: #00d4ff;
    cursor: pointer;
    transition: all 0.2s;
  }

  .apply-btn:hover {
    background: rgba(0, 212, 255, 0.25);
  }

  /* News List */
  .news-list {
    display: flex;
    flex-direction: column;
    gap: 8px;
    overflow-y: auto;
    flex: 1;
  }

  .news-item {
    display: flex;
    flex-direction: column;
    gap: 6px;
    padding: 12px;
    background: rgba(8, 13, 20, 0.35);
    border: 1px solid rgba(0, 212, 255, 0.08);
    border-radius: 8px;
    cursor: pointer;
    text-align: left;
    transition: all 0.2s;
  }

  .news-item:hover {
    border-color: rgba(0, 212, 255, 0.2);
    transform: translateY(-1px);
  }

  .item-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 12px;
  }

  .item-headline {
    font-size: 13px;
    font-family: 'JetBrains Mono', monospace;
    color: #e0e0e0;
    line-height: 1.4;
    flex: 1;
  }

  .severity-badge {
    display: flex;
    align-items: center;
    gap: 3px;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 9px;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
    text-transform: uppercase;
    flex-shrink: 0;
  }

  .severity-high {
    background: rgba(255, 71, 87, 0.2);
    border: 1px solid rgba(255, 71, 87, 0.4);
    color: #ff4757;
  }

  .severity-medium {
    background: rgba(240, 165, 0, 0.2);
    border: 1px solid rgba(240, 165, 0, 0.4);
    color: #f0a500;
  }

  .severity-low {
    background: rgba(100, 100, 100, 0.2);
    border: 1px solid rgba(100, 100, 100, 0.3);
    color: rgba(180, 180, 180, 0.8);
  }

  .severity-unknown {
    background: rgba(100, 100, 100, 0.1);
    border: 1px solid rgba(100, 100, 100, 0.2);
    color: rgba(180, 180, 180, 0.5);
  }

  .item-meta {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 11px;
    color: rgba(224, 224, 224, 0.5);
    font-family: 'JetBrains Mono', monospace;
  }

  .item-source {
    color: rgba(0, 212, 255, 0.6);
  }

  .item-time {
    display: flex;
    align-items: center;
    gap: 4px;
  }

  .item-symbols {
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
    margin-top: 4px;
  }

  .symbol-tag {
    padding: 2px 6px;
    background: rgba(0, 212, 255, 0.1);
    border: 1px solid rgba(0, 212, 255, 0.2);
    border-radius: 4px;
    font-size: 10px;
    font-family: 'JetBrains Mono', monospace;
    color: #00d4ff;
  }

  .symbol-more {
    padding: 2px 6px;
    font-size: 10px;
    color: rgba(224, 224, 224, 0.5);
    font-family: 'JetBrains Mono', monospace;
  }

  .results-count {
    font-size: 11px;
    color: rgba(224, 224, 224, 0.5);
    font-family: 'JetBrains Mono', monospace;
    text-align: right;
    padding-top: 8px;
    border-top: 1px solid rgba(0, 212, 255, 0.05);
  }

  /* States */
  .loading-state,
  .error-state,
  .empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 12px;
    padding: 48px;
    color: rgba(224, 224, 224, 0.5);
    font-size: 13px;
    font-family: 'JetBrains Mono', monospace;
    text-align: center;
  }

  .error-state {
    color: #ff4757;
  }

  .empty-hint {
    font-size: 11px;
    opacity: 0.7;
  }

  .retry-btn {
    margin-top: 8px;
    padding: 8px 16px;
    background: rgba(255, 71, 87, 0.15);
    border: 1px solid rgba(255, 71, 87, 0.3);
    color: #ff4757;
    border-radius: 4px;
    cursor: pointer;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    transition: all 0.2s;
  }

  .retry-btn:hover {
    background: rgba(255, 71, 87, 0.25);
  }

  /* Detail View */
  .detail-view {
    display: flex;
    flex-direction: column;
    gap: 16px;
  }

  .back-btn {
    display: flex;
    align-items: center;
    gap: 4px;
    background: transparent;
    border: none;
    color: #00d4ff;
    cursor: pointer;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    padding: 0;
    transition: all 0.2s;
  }

  .back-btn:hover {
    color: #00a0cc;
  }

  .back-btn :global(.rotate-90) {
    transform: rotate(90deg);
  }

  .detail-content {
    display: flex;
    flex-direction: column;
    gap: 16px;
  }

  .detail-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .copilot-btn {
    display: flex;
    align-items: center;
    gap: 6px;
    background: rgba(0, 200, 100, 0.15);
    border: 1px solid rgba(0, 200, 100, 0.3);
    color: #00c864;
    padding: 8px 14px;
    border-radius: 6px;
    cursor: pointer;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    transition: all 0.2s;
  }

  .copilot-btn:hover {
    background: rgba(0, 200, 100, 0.25);
  }

  .detail-content h2 {
    font-size: 18px;
    font-family: 'JetBrains Mono', monospace;
    color: #e0e0e0;
    margin: 0;
    line-height: 1.4;
  }

  .summary {
    font-size: 14px;
    color: rgba(224, 224, 224, 0.8);
    font-family: 'JetBrains Mono', monospace;
    line-height: 1.6;
    margin: 0;
  }

  .detail-meta {
    display: flex;
    gap: 16px;
    flex-wrap: wrap;
  }

  .meta-item {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 12px;
    color: rgba(224, 224, 224, 0.6);
    font-family: 'JetBrains Mono', monospace;
  }

  .meta-item:first-child {
    color: rgba(0, 212, 255, 0.7);
  }

  .symbols {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 8px;
  }

  .symbols-label {
    font-size: 11px;
    color: rgba(224, 224, 224, 0.5);
    font-family: 'JetBrains Mono', monospace;
  }

  .source-link {
    color: #00d4ff;
    font-size: 12px;
    font-family: 'JetBrains Mono', monospace;
    text-decoration: none;
  }

  .source-link:hover {
    text-decoration: underline;
  }

  /* Toast */
  .toast {
    position: fixed;
    bottom: 24px;
    right: 24px;
    display: flex;
    align-items: center;
    gap: 8px;
    background: rgba(0, 200, 100, 0.9);
    color: #0a0f1a;
    padding: 12px 20px;
    border-radius: 8px;
    font-size: 13px;
    font-weight: 500;
    font-family: 'JetBrains Mono', monospace;
    z-index: 1000;
    animation: slideIn 0.3s ease;
  }

  @keyframes slideIn {
    from {
      transform: translateY(20px);
      opacity: 0;
    }
    to {
      transform: translateY(0);
      opacity: 1;
    }
  }

  @keyframes spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
  }

  :global(.spin) {
    animation: spin 1s linear infinite;
  }
</style>
