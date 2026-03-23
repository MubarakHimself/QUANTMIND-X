<script lang="ts">
  /**
   * Asset List Component
   *
   * Displays list of assets in selected category
   */
  import { FileText, Layout, Code, Sparkles, Workflow, Settings } from 'lucide-svelte';
  import GlassTile from '$lib/components/live-trading/GlassTile.svelte';
  import { sharedAssetsStore, currentAssets, assetsLoading, assetsError } from '$lib/stores/sharedAssets';
  import type { SharedAsset, AssetType } from '$lib/api/sharedAssetsApi';

  // Icon mapping
  const iconMap: Record<AssetType, any> = {
    'docs': FileText,
    'strategy-templates': Layout,
    'indicators': Code,
    'skills': Sparkles,
    'flow-components': Workflow,
    'mcp-configs': Settings
  };

  interface Props {
    selectedType: AssetType | null;
    onSelectAsset?: (asset: SharedAsset) => void;
    onBack?: () => void;
  }

  let { selectedType, onSelectAsset, onBack }: Props = $props();

  // Get current assets from store
  let assets = $derived($currentAssets);
  let loading = $derived($assetsLoading);
  let error = $derived($assetsError);

  // Get label for selected type
  function getTypeLabel(type: AssetType | null): string {
    if (!type) return 'All Assets';
    const labels: Record<AssetType, string> = {
      'docs': 'Docs',
      'strategy-templates': 'Strategy Templates',
      'indicators': 'Indicators',
      'skills': 'Skills',
      'flow-components': 'Flow Components',
      'mcp-configs': 'MCP Configs'
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

  // Handle asset click
  function handleAssetClick(asset: SharedAsset) {
    sharedAssetsStore.selectAsset(asset);
    if (onSelectAsset) {
      onSelectAsset(asset);
    }
  }

  // Handle back click
  function handleBack() {
    sharedAssetsStore.clearSelection();
    if (onBack) {
      onBack();
    }
  }
</script>

<div class="asset-list">
  <!-- Header -->
  <div class="list-header">
    <button class="back-button" onclick={handleBack}>
      ← Back to Categories
    </button>
    <h2 class="list-title">{getTypeLabel(selectedType)}</h2>
    <span class="asset-count">{assets.length} items</span>
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
      <button class="retry-button" onclick={() => sharedAssetsStore.fetchAssets()}>
        Retry
      </button>
    </div>
  {:else if assets.length === 0}
    <!-- Empty state -->
    <div class="empty-state">
      <p>No {getTypeLabel(selectedType).toLowerCase()} available</p>
    </div>
  {:else}
    <!-- Asset grid -->
    <div class="asset-grid">
      {#each assets as asset}
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
