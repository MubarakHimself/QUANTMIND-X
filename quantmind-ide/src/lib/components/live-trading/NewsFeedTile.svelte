<script lang="ts">
  /**
   * NewsFeedTile - Live Trading News Feed Display
   *
   * Shows the latest 5 news items on the Live Trading canvas.
   * Displays headline, source, timestamp, and severity badge.
   * HIGH-severity items flash amber border for 400ms.
   */
  import { onMount, onDestroy } from 'svelte';
  import GlassTile from './GlassTile.svelte';
  import { newsStore, latestNews, newsLoading, newsError, initNewsStore, cleanupNewsStore } from '$lib/stores/news';
  import type { NewsFeedItem } from '$lib/api/newsApi';
  import { Newspaper, Clock, AlertTriangle, RefreshCw, AlertCircle } from 'lucide-svelte';

  // Flash state for HIGH severity items
  let flashingItems = new Set<string>();

  onMount(() => {
    initNewsStore();
  });

  onDestroy(() => {
    cleanupNewsStore();
  });

  /**
   * Format timestamp to relative time
   */
  function formatRelativeTime(isoString: string): string {
    const date = new Date(isoString);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);

    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    if (diffDays < 7) return `${diffDays}d ago`;
    return date.toLocaleDateString();
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
   * Trigger flash animation for HIGH severity items
   */
  function triggerFlash(itemId: string) {
    flashingItems.add(itemId);
    flashingItems = flashingItems; // Trigger reactivity

    setTimeout(() => {
      flashingItems.delete(itemId);
      flashingItems = flashingItems;
    }, 400);
  }

  // Watch for HIGH severity items and trigger flash
  $: {
    const highItems = $latestNews.filter(item => item.severity === 'HIGH');
    highItems.forEach(item => {
      triggerFlash(item.item_id);
    });
  }

  /**
   * Handle retry
   */
  function handleRetry() {
    newsStore.fetchNews(20);
  }

  /**
   * Get mock exposure count (placeholder)
   */
  function getExposureCount(): number {
    // TODO: Replace with actual strategy exposure calculation
    return Math.floor(Math.random() * 10) + 1;
  }
</script>

<GlassTile>
  <div class="news-feed-tile">
    <div class="tile-header">
      <div class="tile-title">
        <Newspaper size={16} />
        <span>News Feed</span>
      </div>
      <div class="tile-status">
        {#if $newsLoading}
          <RefreshCw size={12} class="spin" />
        {:else if $newsError}
          <AlertCircle size={12} class="error-icon" />
        {/if}
      </div>
    </div>

    <div class="news-list">
      {#if $newsError}
        <div class="error-state">
          <AlertCircle size={20} />
          <span>Failed to load news</span>
          <button class="retry-btn" on:click={handleRetry}>Retry</button>
        </div>
      {:else if $latestNews.length === 0 && !$newsLoading}
        <div class="empty-state">
          <Newspaper size={20} />
          <span>No news available</span>
        </div>
      {:else}
        {#each $latestNews as item (item.item_id)}
          <div
            class="news-item"
            class:flashing={flashingItems.has(item.item_id)}
          >
            <div class="news-header">
              <span class="news-headline" title={item.headline}>
                {item.headline.length > 60 ? item.headline.slice(0, 60) + '...' : item.headline}
              </span>
              <span class="severity-badge {getSeverityClass(item.severity)}">
                {#if item.severity === 'HIGH'}
                  <AlertTriangle size={10} />
                {/if}
                {item.severity || 'N/A'}
              </span>
            </div>
            <div class="news-meta">
              <span class="news-source">
                {item.source || 'Unknown'}
              </span>
              <span class="news-time">
                <Clock size={10} />
                {formatRelativeTime(item.published_utc)}
              </span>
            </div>
            {#if item.severity === 'HIGH' && item.related_instruments && item.related_instruments.length > 0}
              <div class="exposure-info">
                {getExposureCount()} {item.related_instruments[0] || 'strategies'} exposed
              </div>
            {/if}
          </div>
        {/each}
      {/if}
    </div>
  </div>
</GlassTile>

<style>
  .news-feed-tile {
    display: flex;
    flex-direction: column;
    gap: 12px;
    min-height: 200px;
    max-height: 300px;
  }

  .tile-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding-bottom: 8px;
    border-bottom: 1px solid rgba(0, 212, 255, 0.08);
  }

  .tile-title {
    display: flex;
    align-items: center;
    gap: 8px;
    color: #00d4ff;
    font-size: 13px;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
  }

  .tile-status {
    display: flex;
    align-items: center;
    color: rgba(224, 224, 224, 0.5);
  }

  .tile-status :global(.spin) {
    animation: spin 1s linear infinite;
  }

  .tile-status :global(.error-icon) {
    color: #ff4757;
  }

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
    gap: 4px;
    padding: 10px;
    background: rgba(0, 0, 0, 0.2);
    border: 1px solid rgba(0, 212, 255, 0.05);
    border-radius: 6px;
    transition: border-color 0.2s ease;
  }

  .news-item.flashing {
    animation: flash-amber 400ms ease-out;
  }

  @keyframes flash-amber {
    0% {
      border-color: rgba(240, 165, 0, 0.8);
      box-shadow: 0 0 8px rgba(240, 165, 0, 0.4);
    }
    100% {
      border-color: rgba(0, 212, 255, 0.05);
      box-shadow: none;
    }
  }

  .news-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 8px;
  }

  .news-headline {
    font-size: 12px;
    font-family: 'JetBrains Mono', monospace;
    color: #e0e0e0;
    line-height: 1.4;
    flex: 1;
  }

  .severity-badge {
    display: flex;
    align-items: center;
    gap: 3px;
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 9px;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
    text-transform: uppercase;
    white-space: nowrap;
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

  .news-meta {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 10px;
    color: rgba(224, 224, 224, 0.5);
    font-family: 'JetBrains Mono', monospace;
  }

  .news-source {
    color: rgba(0, 212, 255, 0.6);
  }

  .news-time {
    display: flex;
    align-items: center;
    gap: 3px;
  }

  .exposure-info {
    font-size: 10px;
    color: #f0a500;
    font-family: 'JetBrains Mono', monospace;
    padding-top: 4px;
    border-top: 1px solid rgba(240, 165, 0, 0.1);
    margin-top: 4px;
  }

  .empty-state,
  .error-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 8px;
    padding: 24px;
    color: rgba(224, 224, 224, 0.5);
    font-size: 12px;
    font-family: 'JetBrains Mono', monospace;
    text-align: center;
  }

  .error-state {
    color: #ff4757;
  }

  .retry-btn {
    background: rgba(0, 212, 255, 0.15);
    border: 1px solid rgba(0, 212, 255, 0.3);
    color: #00d4ff;
    padding: 6px 12px;
    border-radius: 4px;
    cursor: pointer;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    transition: all 0.2s;
  }

  .retry-btn:hover {
    background: rgba(0, 212, 255, 0.25);
  }

  @keyframes spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
  }
</style>
