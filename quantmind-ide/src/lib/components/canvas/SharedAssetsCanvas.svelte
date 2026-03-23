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
  import DepartmentKanban from '$lib/components/department-kanban/DepartmentKanban.svelte';
  import DeptKanbanTile from '$lib/components/shared/DeptKanbanTile.svelte';
  import { canvasContextService } from '$lib/services/canvasContextService';
  import { sharedAssetsStore, selectedAsset, assetCounts } from '$lib/stores/sharedAssets';
  import type { AssetType } from '$lib/api/sharedAssetsApi';
  import { ArrowLeft } from 'lucide-svelte';

  // Sub-page routing (Story 12-6 pattern)
  type SharedAssetsSubPage = 'content' | 'dept-kanban';
  let currentSubPage = $state<SharedAssetsSubPage>('content');

  // View states
  type ViewState = 'grid' | 'list' | 'detail';

  let currentView: ViewState = $state('grid');
  let selectedType: AssetType | null = $state(null);

  // Get store values
  let asset = $derived($selectedAsset);
  let counts = $derived($assetCounts);

  onMount(async () => {
    // Load canvas context
    try {
      await canvasContextService.loadCanvasContext('shared-assets');
    } catch (e) {
      console.error('Failed to load canvas context:', e);
    }

    // Fetch initial asset counts
    await sharedAssetsStore.fetchAssets();
  });

  // Handle type selection from grid
  function handleSelectType(type: AssetType) {
    selectedType = type;
    currentView = 'list';
  }

  // Handle asset selection from list
  function handleSelectAsset() {
    currentView = 'detail';
  }

  // Handle back from list to grid
  function handleBackFromList() {
    selectedType = null;
    currentView = 'grid';
  }

  // Handle back from detail to list
  function handleBackFromDetail() {
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
  {#if currentSubPage === 'dept-kanban'}
    <!-- Department Kanban Sub-Page (Story 12-6) -->
    <DepartmentKanban department="shared-assets" onClose={() => currentSubPage = 'content'} />
  {:else}
    <!-- Canvas header -->
    <header class="canvas-header">
      {#if currentSubPage !== 'content'}
        <button class="back-btn" onclick={() => currentSubPage = 'content'} title="Back">
          <ArrowLeft size={14} />
          <span>Back</span>
        </button>
      {/if}
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
             selectedType === 'mcp-configs' ? 'MCP Configs' : selectedType}
          </button>
        {/if}

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

        <!-- Dept Tasks Tile (Story 12-6 — only visible in grid view) -->
        <div class="tile-row">
          <DeptKanbanTile dept="shared-assets" onNavigate={() => currentSubPage = 'dept-kanban'} />
        </div>

      {:else if currentView === 'list'}
        <!-- Asset list view -->
        <AssetList
          {selectedType}
          onSelectAsset={handleSelectAsset}
          onBack={handleBackFromList}
        />

      {:else if currentView === 'detail'}
        <!-- Asset detail view -->
        <AssetDetail onBack={handleBackFromDetail} />
      {/if}
    </main>
  {/if}
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

  /* When DepartmentKanban is a direct child (dept-kanban sub-page), let it expand */
  .shared-assets-canvas > :global(.department-kanban) {
    flex: 1;
    width: 100%;
    min-width: 0;
  }

  .canvas-header {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    padding: 20px 24px 16px;
    border-bottom: 1px solid rgba(0, 212, 255, 0.1);
    align-items: center;
  }

  .back-btn {
    display: flex;
    align-items: center;
    gap: var(--space-1, 4px);
    padding: var(--space-2, 6px) var(--space-3, 10px);
    background: var(--glass-content-bg, rgba(0, 212, 255, 0.08));
    border: 1px solid var(--color-border-subtle, rgba(0, 212, 255, 0.2));
    border-radius: 6px;
    color: var(--color-accent-cyan, var(--dept-accent));
    font-family: var(--font-ambient, 'JetBrains Mono', monospace);
    font-size: var(--text-xs, 12px);
    cursor: pointer;
    transition: background 0.2s ease, border-color 0.2s ease;
    flex-shrink: 0;
  }

  .back-btn:hover {
    background: var(--glass-content-bg-hover, rgba(0, 212, 255, 0.15));
    border-color: var(--color-border-active, rgba(0, 212, 255, 0.4));
  }

  .tile-row {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 16px;
    padding: 16px 24px;
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
