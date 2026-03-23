<script lang="ts">
  /**
   * Research Search Header Component
   * Contains search bar, source filters, and tile row
   */
  import GlassTile from '$lib/components/live-trading/GlassTile.svelte';
  import VideoIngestTile from '$lib/components/research/VideoIngestTile.svelte';
  import {
    SOURCE_FILTERS,
    type SourceFilter,
    type KnowledgeSourceStatus
  } from '$lib/api/knowledgeApi';
  import { Search, FlaskConical, Database, Loader } from 'lucide-svelte';

  export let query = '';
  export let isSearching = false;
  export let activeFilter: SourceFilter = 'all';
  export let filteredResultsLength = 0;
  export let sourceStatuses: KnowledgeSourceStatus[] = [];
  export let onSearch: () => void = () => {};
  export let onKeydown: (e: KeyboardEvent) => void = () => () => {};
  export let onFilterChange: (filter: SourceFilter) => void = () => () => {};

  function getOnlineCount(): number {
    return sourceStatuses.filter(s => s.status === 'online').length;
  }

  function getTotalDocCount(): number {
    return sourceStatuses.reduce((sum, s) => sum + s.document_count, 0);
  }
</script>

<div class="search-header">
  <!-- Search Bar -->
  <div class="search-bar">
    <Search size={18} class="search-icon" />
    <input
      type="text"
      bind:value={query}
      on:keydown={onKeydown}
      placeholder="Search knowledge base..."
      class="search-input"
    />
    <button class="search-btn" on:click={onSearch} disabled={isSearching || !query.trim()}>
      {#if isSearching}
        <Loader size={16} class="spin" />
      {:else}
        Search
      {/if}
    </button>
  </div>

  <!-- Source Filter Chips -->
  <div class="filter-row">
    {#each SOURCE_FILTERS as filter}
      <button
        class="filter-chip"
        class:active={activeFilter === filter.value}
        on:click={() => onFilterChange(filter.value)}
      >
        <span>{filter.label}</span>
        {#if activeFilter === filter.value && filteredResultsLength > 0}
          <span class="filter-count">{filteredResultsLength}</span>
        {/if}
      </button>
    {/each}
  </div>

  <!-- Tile Row -->
  <div class="tile-row">
    <!-- Hypothesis Pipeline Tile (Stub) -->
    <GlassTile clickable={false}>
      <div class="stub-tile">
        <FlaskConical size={24} />
        <span class="stub-label">Hypothesis Pipeline</span>
        <span class="stub-sublabel">Story 7.1</span>
      </div>
    </GlassTile>

    <!-- Video Ingest Tile (Stub) -->
    <VideoIngestTile />

    <!-- Knowledge Base Status Tile -->
    <GlassTile clickable={false}>
      <div class="status-tile">
        <Database size={24} />
        <span class="stub-label">Knowledge Base</span>
        <div class="status-stats">
          <span>{getOnlineCount()}/{sourceStatuses.length} Online</span>
          <span>{getTotalDocCount()} Docs</span>
        </div>
      </div>
    </GlassTile>
  </div>
</div>

<style>
  .search-header {
    display: flex;
    flex-direction: column;
  }

  /* Search Bar */
  .search-bar {
    display: flex;
    align-items: center;
    gap: 12px;
    background: rgba(8, 13, 20, 0.6);
    border: 1px solid rgba(0, 212, 255, 0.15);
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 16px;
    transition: border-color 0.2s;
  }

  .search-bar:focus-within {
    border-color: rgba(0, 212, 255, 0.3);
  }

  .search-bar :global(.search-icon) {
    color: rgba(0, 212, 255, 0.5);
  }

  .search-input {
    flex: 1;
    background: transparent;
    border: none;
    color: #e0e0e0;
    font-family: 'JetBrains Mono', monospace;
    font-size: 14px;
    outline: none;
  }

  .search-input::placeholder {
    color: rgba(224, 224, 224, 0.4);
  }

  .search-btn {
    background: rgba(0, 212, 255, 0.15);
    border: 1px solid rgba(0, 212, 255, 0.3);
    color: #00d4ff;
    padding: 8px 16px;
    border-radius: 6px;
    cursor: pointer;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    transition: all 0.2s;
    display: flex;
    align-items: center;
    gap: 6px;
  }

  .search-btn:hover:not(:disabled) {
    background: rgba(0, 212, 255, 0.25);
  }

  .search-btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .search-btn :global(.spin) {
    animation: spin 1s linear infinite;
  }

  @keyframes spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
  }

  /* Filter Chips */
  .filter-row {
    display: flex;
    gap: 8px;
    margin-bottom: 20px;
    flex-wrap: wrap;
  }

  .filter-chip {
    background: rgba(8, 13, 20, 0.35);
    border: 1px solid rgba(0, 212, 255, 0.08);
    color: #e0e0e0;
    padding: 6px 12px;
    border-radius: 16px;
    cursor: pointer;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    transition: all 0.2s;
    display: flex;
    align-items: center;
    gap: 6px;
  }

  .filter-chip:hover {
    border-color: rgba(0, 212, 255, 0.2);
  }

  .filter-chip.active {
    background: rgba(0, 212, 255, 0.15);
    border-color: rgba(0, 212, 255, 0.3);
    color: #00d4ff;
  }

  .filter-count {
    background: rgba(0, 212, 255, 0.2);
    padding: 2px 6px;
    border-radius: 10px;
    font-size: 10px;
  }

  /* Tile Row */
  .tile-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
    margin-bottom: 24px;
  }

  @media (max-width: 900px) {
    .tile-row {
      grid-template-columns: 1fr;
    }
  }

  .stub-tile,
  .status-tile {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 8px;
    text-align: center;
    color: rgba(224, 224, 224, 0.6);
  }

  .stub-label {
    font-size: 13px;
    font-weight: 500;
  }

  .stub-sublabel {
    font-size: 10px;
    opacity: 0.6;
  }

  .status-stats {
    display: flex;
    flex-direction: column;
    gap: 4px;
    font-size: 11px;
  }
</style>
