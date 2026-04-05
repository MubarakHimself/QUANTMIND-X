<script lang="ts">
  /**
   * Shared Assets Canvas
   *
   * Canvas 7 - Browsable library of docs, templates, indicators, skills,
   * flow components, and MCP configs.
   */
  import { onMount } from 'svelte';
  import AssetTypeGrid from '$lib/components/shared-assets/AssetTypeGrid.svelte';
  import AssetList from '$lib/components/shared-assets/AssetList.svelte';
  import AssetDetail from '$lib/components/shared-assets/AssetDetail.svelte';
  import { canvasContextService } from '$lib/services/canvasContextService';
  import { sharedAssetsStore, selectedAsset, assetCounts } from '$lib/stores/sharedAssets';
  import type { SharedAsset } from '$lib/api/sharedAssetsApi';
  import type { AssetType } from '$lib/api/sharedAssetsApi';

  // View states
  type ViewState = 'grid' | 'list' | 'detail';

  let currentView: ViewState = $state('grid');
  let selectedType: AssetType | null = $state(null);
  let nestedPathSegments = $state<string[]>([]);

  // Get store values
  let sharedAssetsState = $derived($sharedAssetsStore);
  let asset = $derived($selectedAsset);
  let counts = $derived($assetCounts);
  let flattenedAssets = $derived(
    Object.values(sharedAssetsState.assets).flat() as SharedAsset[]
  );

  onMount(async () => {
    // Load canvas context
    try {
      await canvasContextService.loadCanvasContext('shared-assets');
    } catch (e) {
      console.error('Failed to load canvas context:', e);
    }

    // Fetch initial tile counts only; category payloads are lazy-loaded on click.
    await sharedAssetsStore.fetchAssetCounts();
  });

  $effect(() => {
    canvasContextService.setRuntimeState('shared-assets', {
      active_view: currentView,
      selected_type: selectedType,
      selected_asset: asset?.id ?? null,
      counts: {
        ...counts,
        total_assets: flattenedAssets.length,
      },
      attachable_resources: flattenedAssets.slice(0, 250).map((assetItem) => ({
        id: assetItem.id,
        label: assetItem.name,
        canvas: 'shared-assets',
        resource_type: assetItem.type,
        path: assetItem.source_path,
        description: assetItem.metadata?.description,
        metadata: {
          type: assetItem.type,
          version: assetItem.metadata?.version,
          usage_count: assetItem.metadata?.usage_count ?? 0,
          last_updated: assetItem.metadata?.last_updated,
          strategy_family: assetItem.details?.strategy_family,
          source_bucket: assetItem.details?.source_bucket,
        },
      })),
    });
  });

  $effect(() => {
    if (sharedAssetsState.selectedType && sharedAssetsState.selectedType !== selectedType) {
      selectedType = sharedAssetsState.selectedType;
    }
  });

  $effect(() => {
    if (asset && currentView !== 'detail') {
      currentView = 'detail';
      return;
    }
    if (!asset && selectedType && currentView === 'grid') {
      currentView = 'list';
    }
  });

  // Handle type selection from grid
  async function handleSelectType(type: AssetType) {
    selectedType = type;
    currentView = 'list';
    await sharedAssetsStore.fetchAssetsByType(type);
  }

  // Handle asset selection from list
  function handleSelectAsset() {
    currentView = 'detail';
  }

  // Handle back from list to grid
  function handleBackFromList() {
    sharedAssetsStore.clearSelection();
    selectedType = null;
    nestedPathSegments = [];
    currentView = 'grid';
  }

  // Handle back from detail to list
  function handleBackFromDetail() {
    sharedAssetsStore.clearSelectedAsset();
    currentView = 'list';
  }

  // Handle back to return to appropriate view
  function handleBack() {
    if (currentView === 'detail') {
      handleBackFromDetail();
    } else if (currentView === 'list') {
      handleBackFromList();
    }
  }

  // Compute if we should show breadcrumb
  let showBreadcrumb = $derived(
    currentView !== 'grid' && (selectedType || asset)
  );
</script>

<div class="shared-assets-canvas" data-dept="shared">
  <!-- Canvas header -->
  <header class="canvas-header">
    <h1 class="canvas-title">Shared Assets</h1>
    <span class="canvas-subtitle">Browse and manage reusable assets</span>
  </header>

  <!-- Breadcrumb navigation -->
  {#if showBreadcrumb}
    <nav class="breadcrumb-nav" aria-label="Breadcrumb navigation">
      <button class="breadcrumb-item home" onclick={handleBackFromList}>
        Shared Assets
      </button>

      {#if selectedType && currentView !== 'grid'}
        <span class="separator">/</span>
        <button class="breadcrumb-item" onclick={handleBackFromList}>
          {selectedType === 'docs' ? 'Docs' :
           selectedType === 'strategy-templates' ? 'Strategy Templates' :
           selectedType === 'indicators' ? 'Indicators' :
           selectedType === 'skills' ? 'Skills' :
           selectedType === 'flow-components' ? 'Flow Components' :
           selectedType === 'mcp-configs' ? 'MCP Configs' :
           selectedType === 'strategies' ? 'Strategies' : selectedType}
        </button>
      {/if}

      {#each nestedPathSegments as segment}
        <span class="separator">/</span>
        <span class="breadcrumb-item current">{segment}</span>
      {/each}

      {#if asset && currentView === 'detail'}
        <span class="separator">/</span>
        <span class="breadcrumb-item current">{asset.name}</span>
      {/if}
    </nav>
  {/if}

  <!-- Main content area -->
  <main class="canvas-content">
    {#if currentView === 'grid'}
      <!-- Asset type grid view -->
      <AssetTypeGrid onSelectType={handleSelectType} />

    {:else if currentView === 'list'}
      <!-- Asset list view -->
      <AssetList
        {selectedType}
        onSelectAsset={handleSelectAsset}
        onBack={handleBackFromList}
        onPathChange={(segments) => { nestedPathSegments = segments; }}
      />

    {:else if currentView === 'detail'}
      <!-- Asset detail view -->
      <AssetDetail onBack={handleBackFromDetail} />
    {/if}
  </main>
</div>

<style>
  .shared-assets-canvas {
    display: flex;
    flex-direction: column;
    height: 100%;
    width: 100%;
    min-width: 0;
    /* Frosted Terminal Shell-level styling */
    background: rgba(10, 15, 26, 0.9);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    color: #e0e0e0;
    font-family: 'JetBrains Mono', monospace;
  }

  .canvas-header {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    padding: 20px 24px 16px;
    border-bottom: 1px solid rgba(0, 212, 255, 0.1);
    align-items: center;
  }

  .canvas-title {
    font-size: 24px;
    font-weight: 500;
    color: #ffffff;
    margin: 0;
  }

  .canvas-subtitle {
    font-size: 13px;
    color: rgba(255, 255, 255, 0.5);
  }

  /* Breadcrumb navigation */
  .breadcrumb-nav {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 12px 24px;
    background: rgba(8, 13, 20, 0.5);
    border-bottom: 1px solid rgba(0, 212, 255, 0.08);
    font-size: 13px;
  }

  .breadcrumb-item {
    background: transparent;
    border: none;
    color: rgba(0, 212, 255, 0.7);
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    cursor: pointer;
    padding: 4px 8px;
    border-radius: 4px;
    transition: all 0.15s ease;
  }

  .breadcrumb-item:hover {
    background: rgba(0, 212, 255, 0.1);
    color: rgba(0, 212, 255, 1);
  }

  .breadcrumb-item.home {
    color: rgba(255, 255, 255, 0.8);
  }

  .breadcrumb-item.current {
    color: #ffffff;
    font-weight: 500;
    cursor: default;
  }

  .breadcrumb-item.current:hover {
    background: transparent;
  }

  .separator {
    color: rgba(255, 255, 255, 0.3);
  }

  /* Main content area */
  .canvas-content {
    flex: 1;
    overflow-y: auto;
    padding: 0;
  }

  /* Scrollbar styling */
  .canvas-content::-webkit-scrollbar {
    width: 8px;
  }

  .canvas-content::-webkit-scrollbar-track {
    background: rgba(0, 0, 0, 0.2);
  }

  .canvas-content::-webkit-scrollbar-thumb {
    background: rgba(0, 212, 255, 0.2);
    border-radius: 4px;
  }

  .canvas-content::-webkit-scrollbar-thumb:hover {
    background: rgba(0, 212, 255, 0.4);
  }
</style>
