<script lang="ts">
  /**
   * Research Result Card Component
   * Displays a single search result with title, badges, excerpt, and view button
   */
  import GlassTile from '$lib/components/live-trading/GlassTile.svelte';
  import { SOURCE_BADGE_COLORS, type KnowledgeSearchResult } from '$lib/api/knowledgeApi';
  import { ChevronRight, ExternalLink } from 'lucide-svelte';

  export let result: KnowledgeSearchResult;
  export let onViewFull: (result: KnowledgeSearchResult) => void = () => {};

  function getSourceBadgeColor(sourceType: string): string {
    return SOURCE_BADGE_COLORS[sourceType] || '#888888';
  }

  function truncateExcerpt(excerpt: string, maxLength: number = 300): string {
    if (excerpt.length <= maxLength) return excerpt;
    return excerpt.slice(0, maxLength) + '...';
  }
</script>

<GlassTile clickable={false}>
  <div class="result-card">
    <h3 class="result-title">{result.title}</h3>
    <div class="result-badges">
      <span
        class="source-badge"
        style="background-color: {getSourceBadgeColor(result.source_type)}"
      >
        {result.source_type}
      </span>
      <span class="relevance-badge">
        {result.relevance_score.toFixed(2)}
      </span>
    </div>
    <p class="result-excerpt">{truncateExcerpt(result.excerpt)}</p>
    <button class="view-full-btn" on:click={() => onViewFull(result)}>
      View Full
      <ChevronRight size={14} />
    </button>
  </div>
</GlassTile>

<style>
  .result-card {
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  .result-title {
    font-size: 16px;
    font-weight: 500;
    margin: 0;
    color: #e0e0e0;
  }

  .result-badges {
    display: flex;
    gap: 8px;
    align-items: center;
  }

  .source-badge {
    padding: 4px 8px;
    border-radius: 4px;
    font-size: 10px;
    text-transform: uppercase;
    color: #0a0f1a;
    font-weight: 600;
  }

  .relevance-badge {
    padding: 4px 8px;
    border-radius: 4px;
    font-size: 11px;
    background: rgba(0, 200, 100, 0.15);
    border: 1px solid rgba(0, 200, 100, 0.3);
    color: #00c864;
  }

  .result-excerpt {
    font-size: 13px;
    color: rgba(224, 224, 224, 0.7);
    line-height: 1.5;
    margin: 0;
  }

  .view-full-btn {
    display: flex;
    align-items: center;
    gap: 4px;
    background: transparent;
    border: 1px solid rgba(0, 212, 255, 0.2);
    color: #00d4ff;
    padding: 6px 12px;
    border-radius: 4px;
    cursor: pointer;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    transition: all 0.2s;
    align-self: flex-start;
  }

  .view-full-btn:hover {
    background: rgba(0, 212, 255, 0.1);
    border-color: rgba(0, 212, 255, 0.4);
  }
</style>
