<script lang="ts">
  /**
   * Asset List Component
   *
   * Displays list of assets in selected category
   */
  import { FileText, Layout, Code, Sparkles, Workflow, Settings, FolderTree } from 'lucide-svelte';
  import { onDestroy } from 'svelte';
  import GlassTile from '$lib/components/live-trading/GlassTile.svelte';
  import { API_CONFIG } from '$lib/config/api';
  import { sharedAssetsStore, currentAssets, assetsLoading, assetsError } from '$lib/stores/sharedAssets';
  import type { SharedAsset, AssetType } from '$lib/api/sharedAssetsApi';

  // Icon mapping
  const iconMap: Record<AssetType, any> = {
    'docs': FileText,
    'strategy-templates': Layout,
    'indicators': Code,
    'skills': Sparkles,
    'flow-components': Workflow,
    'mcp-configs': Settings,
    'strategies': FolderTree
  };

  interface Props {
    selectedType: AssetType | null;
    onSelectAsset?: (asset: SharedAsset) => void;
    onBack?: () => void;
    onPathChange?: (segments: string[]) => void;
  }

  let { selectedType, onSelectAsset, onBack, onPathChange }: Props = $props();

  // Get current assets from store
  let assets = $derived($currentAssets);
  let loading = $derived($assetsLoading);
  let error = $derived($assetsError);
  type DocGroup = 'books' | 'articles' | 'docs' | 'other';
  let activeDocGroup = $state<DocGroup | null>(null);
  let activeArticleCategory = $state<string | null>(null);
  let activeStrategyFamily = $state<string | null>(null);
  let activeStrategyBucket = $state<string | null>(null);
  let firecrawlSettings = $state<{
    api_key_set: boolean;
    scraper_type: string;
    scraper_available: boolean;
    firecrawl_available: boolean;
  } | null>(null);
  let knowledgeSyncStatus = $state<{
    status: string;
    existing_articles: number;
    progress?: number;
    categories?: Record<string, { count: number; total_size: number }>;
    sync_state?: {
      last_sync?: string | null;
      articles_synced?: number;
      last_error?: string | null;
    };
  } | null>(null);
  let syncPanelLoading = $state(false);
  let syncInFlight = $state(false);
  let syncError = $state<string | null>(null);
  let selectedScraperType = $state('simple');
  let showApiKeyModal = $state(false);
  let firecrawlApiKeyInput = $state('');
  let savingFirecrawlSettings = $state(false);
  let syncBatchSize = $state(10);
  let syncStartIndex = $state(0);
  let syncPollingHandle: ReturnType<typeof setTimeout> | null = null;

  const KNOWLEDGE_API_BASE = `${API_CONFIG.API_BASE}/knowledge`;

  // Get label for selected type
  function getTypeLabel(type: AssetType | null): string {
    if (!type) return 'All Assets';
    const labels: Record<AssetType, string> = {
      'docs': 'Docs',
      'strategy-templates': 'Strategy Templates',
      'indicators': 'Indicators',
      'skills': 'Skills',
      'flow-components': 'Flow Components',
      'mcp-configs': 'MCP Configs',
      'strategies': 'Strategies'
    };
    return labels[type];
  }

  // Format date
  function formatDate(dateStr: string): string {
    try {
      const date = new Date(dateStr);
      return date.toLocaleDateString('en-US', {
        month: 'short',
        day: 'numeric',
        year: 'numeric'
      });
    } catch {
      return dateStr;
    }
  }

  function getDocGroup(asset: SharedAsset): DocGroup {
    if (asset.id.startsWith('knowledge/books/')) return 'books';
    if (asset.id.startsWith('knowledge/articles/') || asset.id.startsWith('scraped/')) return 'articles';
    if (asset.id.startsWith('docs/')) return 'docs';
    return 'other';
  }

  function getDocGroupLabel(group: DocGroup): string {
    switch (group) {
      case 'books': return 'Books';
      case 'articles': return 'Articles';
      case 'docs': return 'Docs';
      default: return 'Other';
    }
  }

  function getArticleCategory(asset: SharedAsset): string {
    if (asset.id.startsWith('scraped/')) {
      const [, remainder] = asset.id.split('scraped/');
      const segment = remainder?.split('/')[0];
      if (segment && segment !== remainder) return segment;
    }
    if (asset.id.startsWith('knowledge/articles/')) {
      return 'knowledge';
    }
    return 'other';
  }

  function formatCategoryLabel(category: string): string {
    return category
      .replace(/[_-]+/g, ' ')
      .replace(/\b\w/g, (char) => char.toUpperCase());
  }

  function formatSyncTimestamp(value: string | null | undefined): string {
    if (!value) return 'Never';
    try {
      return new Date(value).toLocaleString();
    } catch {
      return value;
    }
  }

  function getStrategyFamily(asset: SharedAsset): string {
    const family = asset.details?.strategy_family ?? asset.metadata?.strategy_family;
    if (typeof family === 'string' && family.trim()) {
      return family.trim();
    }
    const parts = String(asset.id || '').replace(/^strategies\//, '').split('/');
    return parts[0] || 'uncategorized';
  }

  function getStrategyBucket(asset: SharedAsset): string {
    const bucket = asset.details?.source_bucket ?? asset.metadata?.source_bucket;
    if (typeof bucket === 'string' && bucket.trim()) {
      return bucket.trim();
    }
    const parts = String(asset.id || '').replace(/^strategies\//, '').split('/');
    return parts[1] || 'unknown';
  }

  function formatStrategyBucketLabel(bucket: string): string {
    return formatCategoryLabel(bucket === 'single-videos' ? 'single videos' : bucket);
  }

  function getListTitle(): string {
    if (selectedType === 'strategies') {
      if (activeStrategyBucket) return formatStrategyBucketLabel(activeStrategyBucket);
      if (activeStrategyFamily) return formatCategoryLabel(activeStrategyFamily);
    }
    if (selectedType === 'docs') {
      if (activeDocGroup === 'articles' && activeArticleCategory) return formatCategoryLabel(activeArticleCategory);
      if (activeDocGroup) return getDocGroupLabel(activeDocGroup);
    }
    return getTypeLabel(selectedType);
  }

  async function fetchFirecrawlSettings() {
    const response = await fetch(`${KNOWLEDGE_API_BASE}/firecrawl/settings`);
    if (!response.ok) {
      throw new Error(`Failed to load Firecrawl settings (${response.status})`);
    }
    firecrawlSettings = await response.json();
    selectedScraperType = firecrawlSettings?.scraper_type || 'simple';
  }

  async function fetchKnowledgeSyncStatus() {
    const response = await fetch(`${KNOWLEDGE_API_BASE}/sync/status`);
    if (!response.ok) {
      throw new Error(`Failed to load article sync status (${response.status})`);
    }
    knowledgeSyncStatus = await response.json();
  }

  async function hydrateKnowledgeSyncPanel() {
    syncPanelLoading = true;
    syncError = null;
    try {
      await Promise.all([fetchFirecrawlSettings(), fetchKnowledgeSyncStatus()]);
    } catch (err) {
      syncError = err instanceof Error ? err.message : 'Failed to load article sync controls';
    } finally {
      syncPanelLoading = false;
    }
  }

  async function saveFirecrawlSettings() {
    savingFirecrawlSettings = true;
    syncError = null;
    try {
      const response = await fetch(`${KNOWLEDGE_API_BASE}/firecrawl/settings`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          api_key: firecrawlApiKeyInput,
          scraper_type: selectedScraperType
        })
      });
      if (!response.ok) {
        throw new Error(`Failed to save Firecrawl settings (${response.status})`);
      }
      firecrawlSettings = await response.json();
      showApiKeyModal = false;
      firecrawlApiKeyInput = '';
    } catch (err) {
      syncError = err instanceof Error ? err.message : 'Failed to save Firecrawl settings';
    } finally {
      savingFirecrawlSettings = false;
    }
  }

  async function pollKnowledgeSyncStatus() {
    if (syncPollingHandle) {
      clearTimeout(syncPollingHandle);
    }

    try {
      await fetchKnowledgeSyncStatus();
      const status = knowledgeSyncStatus?.status || 'idle';
      if (status === 'running' || status === 'syncing') {
        syncPollingHandle = setTimeout(() => {
          void pollKnowledgeSyncStatus();
        }, 2000);
        return;
      }

      syncInFlight = false;
      await sharedAssetsStore.fetchAssetsByType('docs');
    } catch (err) {
      syncError = err instanceof Error ? err.message : 'Failed to poll article sync status';
      syncInFlight = false;
    }
  }

  async function triggerKnowledgeSync() {
    syncInFlight = true;
    syncError = null;
    try {
      const params = new URLSearchParams({
        batch_size: `${syncBatchSize}`,
        start_index: `${syncStartIndex}`,
        sync_mode: 'background',
        scraper_type: selectedScraperType
      });
      const response = await fetch(`${KNOWLEDGE_API_BASE}/sync?${params.toString()}`, {
        method: 'POST'
      });
      if (!response.ok) {
        throw new Error(`Failed to start article sync (${response.status})`);
      }
      await pollKnowledgeSyncStatus();
    } catch (err) {
      syncError = err instanceof Error ? err.message : 'Failed to start article sync';
      syncInFlight = false;
    }
  }

  let docGroupCounts = $derived.by(() => {
    const counts: Record<DocGroup, number> = {
      books: 0,
      articles: 0,
      docs: 0,
      other: 0
    };
    if (selectedType !== 'docs') return counts;
    for (const asset of assets) {
      counts[getDocGroup(asset)] += 1;
    }
    return counts;
  });

  let docGroups = $derived.by(() =>
    (Object.entries(docGroupCounts) as [DocGroup, number][])
      .filter(([, count]) => count > 0)
  );

  let articleCategoryCounts = $derived.by(() => {
    const counts = new Map<string, number>();
    if (selectedType !== 'docs' || activeDocGroup !== 'articles') {
      return counts;
    }
    for (const asset of assets) {
      if (getDocGroup(asset) !== 'articles') continue;
      const category = getArticleCategory(asset);
      counts.set(category, (counts.get(category) ?? 0) + 1);
    }
    return counts;
  });

  let articleCategories = $derived.by(() =>
    Array.from(articleCategoryCounts.entries()).sort((a, b) => a[0].localeCompare(b[0]))
  );

  let strategyFamilyCounts = $derived.by(() => {
    const counts = new Map<string, number>();
    if (selectedType !== 'strategies') {
      return counts;
    }
    for (const asset of assets) {
      const family = getStrategyFamily(asset);
      counts.set(family, (counts.get(family) ?? 0) + 1);
    }
    return counts;
  });

  let strategyFamilies = $derived.by(() =>
    Array.from(strategyFamilyCounts.entries()).sort((a, b) => a[0].localeCompare(b[0]))
  );

  let strategyBucketCounts = $derived.by(() => {
    const counts = new Map<string, number>();
    if (selectedType !== 'strategies' || !activeStrategyFamily) {
      return counts;
    }
    for (const asset of assets) {
      if (getStrategyFamily(asset) !== activeStrategyFamily) continue;
      const bucket = getStrategyBucket(asset);
      counts.set(bucket, (counts.get(bucket) ?? 0) + 1);
    }
    return counts;
  });

  let strategyBuckets = $derived.by(() =>
    Array.from(strategyBucketCounts.entries()).sort((a, b) => a[0].localeCompare(b[0]))
  );

  let visibleAssets = $derived.by(() => {
    if (selectedType === 'strategies') {
      return assets.filter((asset) => {
        if (activeStrategyFamily && getStrategyFamily(asset) !== activeStrategyFamily) return false;
        if (activeStrategyBucket && getStrategyBucket(asset) !== activeStrategyBucket) return false;
        return true;
      });
    }
    if (selectedType !== 'docs' || !activeDocGroup) {
      return assets;
    }
    return assets.filter((asset) => {
      if (getDocGroup(asset) !== activeDocGroup) return false;
      if (activeDocGroup === 'articles' && activeArticleCategory) {
        return getArticleCategory(asset) === activeArticleCategory;
      }
      return true;
    });
  });

  // Handle asset click
  function handleAssetClick(asset: SharedAsset) {
    sharedAssetsStore.selectAsset(asset);
    if (onSelectAsset) {
      onSelectAsset(asset);
    }
  }

  // Handle back click
  function handleBack() {
    if (selectedType === 'strategies' && activeStrategyBucket) {
      activeStrategyBucket = null;
      return;
    }
    if (selectedType === 'strategies' && activeStrategyFamily) {
      activeStrategyFamily = null;
      return;
    }
    if (selectedType === 'docs' && activeDocGroup === 'articles' && activeArticleCategory) {
      activeArticleCategory = null;
      return;
    }
    if (selectedType === 'docs' && activeDocGroup) {
      activeDocGroup = null;
      return;
    }
    sharedAssetsStore.clearSelection();
    if (onBack) {
      onBack();
    }
  }

  function openDocGroup(group: DocGroup) {
    activeDocGroup = group;
  }

  function openArticleCategory(category: string) {
    activeArticleCategory = category;
  }

  function openStrategyFamily(family: string) {
    activeStrategyFamily = family;
  }

  function openStrategyBucket(bucket: string) {
    activeStrategyBucket = bucket;
  }

  function retryLoad() {
    if (selectedType) {
      void sharedAssetsStore.fetchAssetsByType(selectedType);
      return;
    }
    void sharedAssetsStore.fetchAssetCounts();
  }

  $effect(() => {
    if (selectedType !== 'docs') {
      activeDocGroup = null;
      activeArticleCategory = null;
    }
    if (selectedType !== 'strategies') {
      activeStrategyFamily = null;
      activeStrategyBucket = null;
    }
    if (activeDocGroup !== 'articles') {
      activeArticleCategory = null;
    }
    if (!activeStrategyFamily) {
      activeStrategyBucket = null;
    }
  });

  $effect(() => {
    if (selectedType === 'docs' && activeDocGroup === 'articles') {
      void hydrateKnowledgeSyncPanel();
    }
  });

  $effect(() => {
    const segments: string[] = [];
    if (selectedType === 'docs') {
      if (activeDocGroup) segments.push(getDocGroupLabel(activeDocGroup));
      if (activeDocGroup === 'articles' && activeArticleCategory) {
        segments.push(formatCategoryLabel(activeArticleCategory));
      }
    } else if (selectedType === 'strategies') {
      if (activeStrategyFamily) segments.push(formatCategoryLabel(activeStrategyFamily));
      if (activeStrategyBucket) segments.push(formatStrategyBucketLabel(activeStrategyBucket));
    }
    onPathChange?.(segments);
  });

  onDestroy(() => {
    if (syncPollingHandle) {
      clearTimeout(syncPollingHandle);
    }
  });
</script>

<div class="asset-list">
  <!-- Header -->
  <div class="list-header">
    <button class="back-button" onclick={handleBack}>
      {#if selectedType === 'strategies' && activeStrategyBucket}
        ← Back to {formatCategoryLabel(activeStrategyFamily || 'Strategies')}
      {:else if selectedType === 'strategies' && activeStrategyFamily}
        ← Back to Strategies
      {:else if selectedType === 'docs' && activeDocGroup === 'articles' && activeArticleCategory}
        ← Back to Articles
      {:else if selectedType === 'docs' && activeDocGroup}
        ← Back to Docs
      {:else}
        ← Back to Categories
      {/if}
    </button>
    <h2 class="list-title">{getListTitle()}</h2>
    <span class="asset-count">
      {#if selectedType === 'strategies' && (activeStrategyFamily || activeStrategyBucket)}
        {visibleAssets.length} items
      {:else if selectedType === 'docs' && activeDocGroup}
        {visibleAssets.length} items
      {:else}
        {assets.length} items
      {/if}
    </span>
  </div>

  <!-- Loading state -->
  {#if loading}
    <div class="loading-state">
      <div class="loading-spinner"></div>
      <p>Loading assets...</p>
    </div>
  {:else if error}
    <!-- Error state -->
    <div class="error-state">
      <p class="error-message">{error}</p>
      <button class="retry-button" onclick={retryLoad}>
        Retry
      </button>
    </div>
  {:else if assets.length === 0}
    <!-- Empty state -->
    <div class="empty-state">
      <p>No {getTypeLabel(selectedType).toLowerCase()} available</p>
    </div>
  {:else if selectedType === 'docs' && !activeDocGroup}
    <div class="folder-grid">
      {#each docGroups as [group, count]}
        <GlassTile clickable={true}>
          <button class="folder-card" onclick={() => openDocGroup(group)}>
            <div class="asset-icon">
              <FolderTree size={24} />
            </div>
            <div class="asset-info">
              <h3 class="asset-name">{getDocGroupLabel(group)}</h3>
              <p class="asset-description">Open {getDocGroupLabel(group).toLowerCase()} folder</p>
              <span class="asset-updated">{count} items</span>
            </div>
          </button>
        </GlassTile>
      {/each}
    </div>
  {:else if selectedType === 'docs' && activeDocGroup === 'articles' && !activeArticleCategory}
    <section class="knowledge-sync-panel">
      <div class="knowledge-sync-header">
        <div>
          <h3>Article Sync</h3>
          <p>Trigger categorized article scraping for Shared Assets using the configured scraper.</p>
        </div>
        <button class="sync-now-btn" onclick={triggerKnowledgeSync} disabled={syncInFlight || syncPanelLoading}>
          {#if syncInFlight}
            Syncing…
          {:else}
            Sync Now
          {/if}
        </button>
      </div>

      {#if syncError}
        <div class="sync-feedback error">{syncError}</div>
      {/if}

      {#if syncPanelLoading}
        <div class="sync-feedback">Loading sync controls…</div>
      {:else}
        <div class="sync-controls-grid">
          <label class="sync-field">
            <span>Scraper</span>
            <select bind:value={selectedScraperType}>
              <option value="simple">Simple Scraper</option>
              <option value="firecrawl" disabled={!firecrawlSettings?.firecrawl_available}>
                Firecrawl {firecrawlSettings?.firecrawl_available ? '' : '(unavailable)'}
              </option>
            </select>
          </label>

          <label class="sync-field">
            <span>Batch Size</span>
            <input type="number" min="1" max="200" bind:value={syncBatchSize} />
          </label>

          <label class="sync-field">
            <span>Start Index</span>
            <input type="number" min="0" bind:value={syncStartIndex} />
          </label>

          <div class="sync-field api-key-field">
            <span>Firecrawl API</span>
            <button class="ghost-action-btn" onclick={() => showApiKeyModal = true}>
              {firecrawlSettings?.api_key_set ? 'Update API Key' : 'Set API Key'}
            </button>
          </div>
        </div>

        {#if knowledgeSyncStatus}
          <div class="sync-status-grid">
            <div class="sync-stat">
              <span class="sync-stat-label">Status</span>
              <strong>{knowledgeSyncStatus.status}</strong>
            </div>
            <div class="sync-stat">
              <span class="sync-stat-label">Existing Articles</span>
              <strong>{knowledgeSyncStatus.existing_articles}</strong>
            </div>
            <div class="sync-stat">
              <span class="sync-stat-label">Last Sync</span>
              <strong>{formatSyncTimestamp(knowledgeSyncStatus.sync_state?.last_sync)}</strong>
            </div>
            <div class="sync-stat">
              <span class="sync-stat-label">Last Batch</span>
              <strong>{knowledgeSyncStatus.sync_state?.articles_synced ?? 0}</strong>
            </div>
          </div>

          {#if knowledgeSyncStatus.categories && Object.keys(knowledgeSyncStatus.categories).length > 0}
            <div class="category-chip-row">
              {#each Object.entries(knowledgeSyncStatus.categories) as [category, meta]}
                <span class="category-chip">{formatCategoryLabel(category)} · {meta.count}</span>
              {/each}
            </div>
          {/if}
        {/if}
      {/if}
    </section>

    <div class="folder-grid">
      {#each articleCategories as [category, count]}
        <GlassTile clickable={true}>
          <button class="folder-card" onclick={() => openArticleCategory(category)}>
            <div class="asset-icon">
              <FolderTree size={24} />
            </div>
            <div class="asset-info">
              <h3 class="asset-name">{formatCategoryLabel(category)}</h3>
              <p class="asset-description">Open {formatCategoryLabel(category).toLowerCase()} articles</p>
              <span class="asset-updated">{count} items</span>
            </div>
          </button>
        </GlassTile>
      {/each}
    </div>
    {#if showApiKeyModal}
      <div class="modal-overlay" onclick={() => showApiKeyModal = false}>
        <div class="modal-card" onclick={(event) => event.stopPropagation()} role="dialog" aria-modal="true" aria-labelledby="firecrawl-settings-title">
          <div class="modal-card-header">
            <h4 id="firecrawl-settings-title">Firecrawl Settings</h4>
            <button class="ghost-action-btn" onclick={() => showApiKeyModal = false}>Close</button>
          </div>
          <div class="modal-card-body">
            <label class="sync-field">
              <span>API Key</span>
              <input type="password" bind:value={firecrawlApiKeyInput} placeholder="Enter Firecrawl API key" />
            </label>
            <label class="sync-field">
              <span>Default Scraper</span>
              <select bind:value={selectedScraperType}>
                <option value="simple">Simple Scraper</option>
                <option value="firecrawl" disabled={!firecrawlSettings?.firecrawl_available}>
                  Firecrawl {firecrawlSettings?.firecrawl_available ? '' : '(unavailable)'}
                </option>
              </select>
            </label>
          </div>
          <div class="modal-card-footer">
            <button class="ghost-action-btn" onclick={() => showApiKeyModal = false}>Cancel</button>
            <button class="sync-now-btn" onclick={saveFirecrawlSettings} disabled={savingFirecrawlSettings}>
              {savingFirecrawlSettings ? 'Saving…' : 'Save Settings'}
            </button>
          </div>
        </div>
      </div>
    {/if}
  {:else if selectedType === 'strategies' && !activeStrategyFamily}
    <div class="folder-grid">
      {#each strategyFamilies as [family, count]}
        <GlassTile clickable={true}>
          <button class="folder-card" onclick={() => openStrategyFamily(family)}>
            <div class="asset-icon">
              <FolderTree size={24} />
            </div>
            <div class="asset-info">
              <h3 class="asset-name">{formatCategoryLabel(family)}</h3>
              <p class="asset-description">Open {formatCategoryLabel(family).toLowerCase()} strategies</p>
              <span class="asset-updated">{count} items</span>
            </div>
          </button>
        </GlassTile>
      {/each}
    </div>
  {:else if selectedType === 'strategies' && activeStrategyFamily && !activeStrategyBucket}
    <div class="folder-grid">
      {#each strategyBuckets as [bucket, count]}
        <GlassTile clickable={true}>
          <button class="folder-card" onclick={() => openStrategyBucket(bucket)}>
            <div class="asset-icon">
              <FolderTree size={24} />
            </div>
            <div class="asset-info">
              <h3 class="asset-name">{formatStrategyBucketLabel(bucket)}</h3>
              <p class="asset-description">Open {formatStrategyBucketLabel(bucket).toLowerCase()} roots</p>
              <span class="asset-updated">{count} items</span>
            </div>
          </button>
        </GlassTile>
      {/each}
    </div>
  {:else}
    <!-- Asset grid -->
    <div class="asset-grid">
      {#each visibleAssets as asset}
        <GlassTile clickable={true}>
          <button class="asset-card" onclick={() => handleAssetClick(asset)}>
            <div class="asset-icon">
              {#if iconMap[asset.type]}
                <svelte:component this={iconMap[asset.type]} size={24} />
              {:else}
                <FileText size={24} />
              {/if}
            </div>
            <div class="asset-info">
              <h3 class="asset-name">{asset.name}</h3>
              <div class="asset-meta">
                <span class="asset-version">v{asset.metadata.version}</span>
                <span class="asset-usage">{asset.metadata.usage_count} uses</span>
              </div>
              {#if asset.metadata.description}
                <p class="asset-description">{asset.metadata.description}</p>
              {/if}
              <span class="asset-updated">Updated {formatDate(asset.metadata.last_updated)}</span>
            </div>
          </button>
        </GlassTile>
      {/each}
    </div>
  {/if}
</div>

<style>
  .asset-list {
    padding: 16px;
    display: flex;
    flex-direction: column;
    gap: 16px;
  }

  .list-header {
    display: flex;
    align-items: center;
    gap: 16px;
    padding-bottom: 16px;
    border-bottom: 1px solid rgba(0, 212, 255, 0.1);
  }

  .back-button {
    background: transparent;
    border: none;
    color: rgba(0, 212, 255, 0.8);
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    cursor: pointer;
    padding: 8px 12px;
    border-radius: 4px;
    transition: all 0.15s ease;
  }

  .back-button:hover {
    background: rgba(0, 212, 255, 0.1);
    color: rgba(0, 212, 255, 1);
  }

  .list-title {
    font-size: 18px;
    font-weight: 500;
    color: #e0e0e0;
    flex: 1;
    margin: 0;
  }

  .asset-count {
    font-size: 13px;
    color: rgba(255, 255, 255, 0.5);
  }

  .asset-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 16px;
  }

  .folder-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
    gap: 16px;
  }

  .knowledge-sync-panel {
    display: flex;
    flex-direction: column;
    gap: 14px;
    margin-bottom: 16px;
    padding: 16px;
    border: 1px solid rgba(0, 212, 255, 0.16);
    border-radius: 14px;
    background: linear-gradient(180deg, rgba(0, 212, 255, 0.08), rgba(11, 15, 25, 0.86));
  }

  .knowledge-sync-header {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 16px;
  }

  .knowledge-sync-header h3 {
    margin: 0 0 6px;
    font-size: 14px;
    color: var(--color-text-primary);
  }

  .knowledge-sync-header p {
    margin: 0;
    font-size: 12px;
    color: var(--color-text-muted);
    line-height: 1.45;
  }

  .sync-controls-grid,
  .sync-status-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
    gap: 12px;
  }

  .sync-field {
    display: flex;
    flex-direction: column;
    gap: 6px;
    font-size: 12px;
    color: var(--color-text-secondary);
  }

  .sync-field span {
    color: var(--color-text-muted);
  }

  .sync-field select,
  .sync-field input {
    width: 100%;
    padding: 8px 10px;
    border-radius: 8px;
    border: 1px solid var(--color-border-subtle);
    background: rgba(8, 12, 20, 0.88);
    color: var(--color-text-primary);
    font: inherit;
  }

  .api-key-field {
    justify-content: flex-end;
  }

  .sync-stat {
    display: flex;
    flex-direction: column;
    gap: 6px;
    padding: 10px 12px;
    border-radius: 10px;
    background: rgba(5, 10, 18, 0.7);
    border: 1px solid rgba(0, 212, 255, 0.1);
  }

  .sync-stat-label {
    font-size: 11px;
    color: var(--color-text-muted);
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }

  .category-chip-row {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
  }

  .category-chip {
    display: inline-flex;
    align-items: center;
    padding: 4px 10px;
    border-radius: 999px;
    background: rgba(0, 212, 255, 0.12);
    color: var(--color-text-primary);
    font-size: 11px;
    border: 1px solid rgba(0, 212, 255, 0.18);
  }

  .sync-now-btn,
  .ghost-action-btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 6px;
    min-height: 36px;
    padding: 8px 12px;
    border-radius: 8px;
    font: inherit;
    cursor: pointer;
    transition: background 0.15s ease, border-color 0.15s ease, color 0.15s ease;
  }

  .sync-now-btn {
    border: 1px solid rgba(0, 212, 255, 0.26);
    background: rgba(0, 212, 255, 0.16);
    color: var(--color-text-primary);
  }

  .sync-now-btn:disabled {
    opacity: 0.65;
    cursor: not-allowed;
  }

  .ghost-action-btn {
    border: 1px solid var(--color-border-subtle);
    background: rgba(5, 10, 18, 0.5);
    color: var(--color-text-secondary);
  }

  .sync-feedback {
    padding: 10px 12px;
    border-radius: 10px;
    background: rgba(255, 255, 255, 0.04);
    color: var(--color-text-secondary);
    font-size: 12px;
  }

  .sync-feedback.error {
    background: rgba(255, 107, 107, 0.12);
    color: #ffb4b4;
    border: 1px solid rgba(255, 107, 107, 0.22);
  }

  .modal-overlay {
    position: fixed;
    inset: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 24px;
    background: rgba(2, 6, 14, 0.72);
    z-index: 30;
  }

  .modal-card {
    width: min(480px, 92vw);
    border-radius: 16px;
    border: 1px solid rgba(0, 212, 255, 0.18);
    background: rgba(10, 15, 26, 0.98);
    box-shadow: 0 24px 64px rgba(0, 0, 0, 0.45);
  }

  .modal-card-header,
  .modal-card-footer {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    padding: 16px 18px;
    border-bottom: 1px solid rgba(0, 212, 255, 0.1);
  }

  .modal-card-header h4 {
    margin: 0;
    font-size: 14px;
    color: var(--color-text-primary);
  }

  .modal-card-body {
    display: flex;
    flex-direction: column;
    gap: 14px;
    padding: 18px;
  }

  .modal-card-footer {
    border-bottom: none;
    border-top: 1px solid rgba(0, 212, 255, 0.1);
    justify-content: flex-end;
  }

  .folder-card {
    display: flex;
    align-items: flex-start;
    gap: 16px;
    width: 100%;
    background: transparent;
    border: none;
    cursor: pointer;
    color: inherit;
    font-family: inherit;
    text-align: left;
    padding: 0;
  }

  .asset-card {
    display: flex;
    align-items: flex-start;
    gap: 16px;
    width: 100%;
    background: transparent;
    border: none;
    cursor: pointer;
    color: inherit;
    font-family: inherit;
    text-align: left;
    padding: 0;
  }

  .asset-icon {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 48px;
    height: 48px;
    background: rgba(0, 212, 255, 0.1);
    border-radius: 8px;
    color: rgba(0, 212, 255, 0.8);
    flex-shrink: 0;
  }

  .asset-info {
    flex: 1;
    min-width: 0;
  }

  .asset-name {
    font-size: 14px;
    font-weight: 500;
    color: #e0e0e0;
    margin: 0 0 4px 0;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .asset-meta {
    display: flex;
    gap: 12px;
    font-size: 12px;
    color: rgba(255, 255, 255, 0.5);
    margin-bottom: 4px;
  }

  .asset-version {
    color: rgba(0, 212, 255, 0.7);
  }

  .asset-updated {
    font-size: 11px;
    color: rgba(255, 255, 255, 0.4);
  }

  .asset-description {
    font-size: 11px;
    color: rgba(255, 255, 255, 0.62);
    margin: 0 0 6px 0;
    line-height: 1.45;
  }

  /* Loading/Error/Empty states */
  .loading-state,
  .error-state,
  .empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 48px;
    color: rgba(255, 255, 255, 0.5);
  }

  .loading-spinner {
    width: 32px;
    height: 32px;
    border: 2px solid rgba(0, 212, 255, 0.2);
    border-top-color: rgba(0, 212, 255, 0.8);
    border-radius: 50%;
    animation: spin 1s linear infinite;
    margin-bottom: 16px;
  }

  @keyframes spin {
    to { transform: rotate(360deg); }
  }

  .error-message {
    color: rgba(255, 100, 100, 0.8);
    margin-bottom: 16px;
  }

  .retry-button {
    background: rgba(0, 212, 255, 0.1);
    border: 1px solid rgba(0, 212, 255, 0.3);
    color: rgba(0, 212, 255, 0.9);
    padding: 8px 16px;
    border-radius: 4px;
    cursor: pointer;
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    transition: all 0.15s ease;
  }

  .retry-button:hover {
    background: rgba(0, 212, 255, 0.2);
  }

  /* Frosted Terminal glass styling */
  :global(.asset-list .glass-tile) {
    background: rgba(8, 13, 20, 0.35);
    backdrop-filter: blur(16px) saturate(120%);
    -webkit-backdrop-filter: blur(16px) saturate(120%);
    border: 1px solid rgba(0, 212, 255, 0.08);
    border-radius: 8px;
    padding: 16px;
  }

  .asset-card:hover .asset-icon {
    background: rgba(0, 212, 255, 0.2);
    color: rgba(0, 212, 255, 1);
  }

  .asset-card:hover .asset-name {
    color: #ffffff;
  }
</style>
