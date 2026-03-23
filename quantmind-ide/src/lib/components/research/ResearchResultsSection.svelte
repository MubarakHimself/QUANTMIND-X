<script lang="ts">
  /**
   * Research Results Section Component
   * Contains skeleton loader, error banner, empty state, result cards, and initial state
   */
  import ResearchResultCard from '$lib/components/research/ResearchResultCard.svelte';
  import type { KnowledgeSearchResult } from '$lib/api/knowledgeApi';
  import { BookOpen, Search, AlertCircle } from 'lucide-svelte';

  export let isSearching = false;
  export let searchError: string | null = null;
  export let hasSearched = false;
  export let filteredResults: KnowledgeSearchResult[] = [];
  export let onViewFull: (result: KnowledgeSearchResult) => void = () => {};
</script>

<div class="results-section">
  <!-- Skeleton Loader -->
  {#if isSearching}
    <div class="skeleton-grid">
      {#each [1, 2, 3] as _}
        <div class="skeleton-card">
          <div class="skeleton-line title"></div>
          <div class="skeleton-line badge"></div>
          <div class="skeleton-line excerpt"></div>
          <div class="skeleton-line excerpt short"></div>
        </div>
      {/each}
    </div>
  {:else if searchError}
    <!-- Error Banner -->
    <div class="error-banner">
      <AlertCircle size={16} />
      <span>{searchError}</span>
    </div>
  {:else if hasSearched && filteredResults.length === 0}
    <!-- Empty State -->
    <div class="empty-state">
      <BookOpen size={48} />
      <p>No results found</p>
      <span>Try adjusting your search query or filters</span>
    </div>
  {:else if filteredResults.length > 0}
    <!-- Result Cards -->
    <div class="results-grid">
      {#each filteredResults as result}
        <ResearchResultCard {result} onViewFull={onViewFull} />
      {/each}
    </div>
  {:else}
    <!-- Initial State -->
    <div class="initial-state">
      <Search size={48} />
      <p>Search the knowledge base</p>
      <span>Enter a query above to find articles, books, and logs</span>
    </div>
  {/if}
</div>

<style>
  .results-section {
    min-height: 200px;
  }

  /* Skeleton */
  .skeleton-grid {
    display: grid;
    gap: 16px;
  }

  .skeleton-card {
    background: rgba(8, 13, 20, 0.35);
    border: 1px solid rgba(0, 212, 255, 0.08);
    border-radius: 8px;
    padding: 16px;
  }

  .skeleton-line {
    background: linear-gradient(90deg, rgba(0, 212, 255, 0.1) 25%, rgba(0, 212, 255, 0.2) 50%, rgba(0, 212, 255, 0.1) 75%);
    background-size: 200% 100%;
    animation: shimmer 1.5s infinite;
    border-radius: 4px;
    margin-bottom: 12px;
  }

  .skeleton-line.title {
    height: 20px;
    width: 60%;
  }

  .skeleton-line.badge {
    height: 16px;
    width: 80px;
  }

  .skeleton-line.excerpt {
    height: 14px;
    width: 100%;
  }

  .skeleton-line.excerpt.short {
    width: 70%;
  }

  @keyframes shimmer {
    0% { background-position: 200% 0; }
    100% { background-position: -200% 0; }
  }

  /* Error Banner */
  .error-banner {
    display: flex;
    align-items: center;
    gap: 8px;
    background: rgba(255, 59, 59, 0.15);
    border: 1px solid rgba(255, 59, 59, 0.3);
    color: #ff3b3b;
    padding: 12px 16px;
    border-radius: 8px;
    font-size: 13px;
  }

  /* Empty State */
  .empty-state,
  .initial-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 12px;
    padding: 48px;
    color: rgba(224, 224, 224, 0.5);
    text-align: center;
  }

  .empty-state p,
  .initial-state p {
    font-size: 16px;
    margin: 0;
    color: #e0e0e0;
  }

  .empty-state span,
  .initial-state span {
    font-size: 12px;
  }

  /* Result Cards */
  .results-grid {
    display: grid;
    gap: 16px;
  }
</style>
