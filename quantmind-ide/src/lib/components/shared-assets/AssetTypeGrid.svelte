<script lang="ts">
  /**
   * Asset Type Grid Component
   *
   * Displays 6 category tiles for shared assets:
   * Docs, Strategy Templates, Indicators, Skills, Flow Components, MCP Configs
   */
  import { FileText, Layout, Code, Sparkles, Workflow, Settings, FolderTree } from 'lucide-svelte';
  import GlassTile from '$lib/components/live-trading/GlassTile.svelte';
  import { sharedAssetsStore, assetCounts } from '$lib/stores/sharedAssets';
  import type { AssetType } from '$lib/api/sharedAssetsApi';

  interface Props {
    onSelectType?: (type: AssetType) => void;
  }

  let { onSelectType }: Props = $props();

  // Asset type definitions with icons and labels
  const assetTypes: { type: AssetType; label: string; icon: any }[] = [
    { type: 'docs', label: 'Docs', icon: FileText },
    { type: 'strategy-templates', label: 'Strategy Templates', icon: Layout },
    { type: 'indicators', label: 'Indicators', icon: Code },
    { type: 'skills', label: 'Skills', icon: Sparkles },
    { type: 'flow-components', label: 'Flow Components', icon: Workflow },
    { type: 'mcp-configs', label: 'MCP Configs', icon: Settings },
    { type: 'strategies', label: 'Strategies', icon: FolderTree }
  ];

  function handleTypeClick(type: AssetType) {
    sharedAssetsStore.setSelectedType(type);
    if (onSelectType) {
      onSelectType(type);
    }
  }

  // Get count for a type
  function getCount(type: AssetType): number {
    return $assetCounts[type] || 0;
  }

  function getCountLabel(type: AssetType): string {
    const count = getCount(type);
    if ($sharedAssetsStore.isLoading && count === 0) {
      return 'Loading...';
    }
    return `${count} items`;
  }
</script>

<div class="asset-type-grid">
  {#each assetTypes as item}
    <GlassTile clickable={true}>
      <button class="asset-type-tile" onclick={() => handleTypeClick(item.type)}>
        <div class="tile-icon">
          <item.icon size={32} />
        </div>
        <div class="tile-label">{item.label}</div>
        <div class="tile-count">{getCountLabel(item.type)}</div>
      </button>
    </GlassTile>
  {/each}
</div>

<style>
  .asset-type-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
    gap: 16px;
    padding: 16px;
  }

  .asset-type-tile {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 12px;
    width: 100%;
    height: 120px;
    background: transparent;
    border: none;
    cursor: pointer;
    color: inherit;
    font-family: inherit;
    transition: transform 0.15s ease, border-color 0.2s ease;
  }

  .tile-icon {
    display: flex;
    align-items: center;
    justify-content: center;
    color: rgba(0, 212, 255, 0.8);
  }

  .tile-label {
    font-size: 14px;
    font-weight: 500;
    color: #e0e0e0;
    text-align: center;
  }

  .tile-count {
    font-size: 12px;
    color: rgba(255, 255, 255, 0.5);
    text-align: center;
  }

  /* Frosted Terminal styling */
  :global(.asset-type-grid .glass-tile) {
    background: rgba(8, 13, 20, 0.35);
    backdrop-filter: blur(16px) saturate(120%);
    -webkit-backdrop-filter: blur(16px) saturate(120%);
    border: 1px solid rgba(0, 212, 255, 0.08);
    border-radius: 8px;
  }

  .asset-type-tile:hover .tile-icon {
    color: rgba(0, 212, 255, 1);
  }

  .asset-type-tile:hover .tile-label {
    color: #ffffff;
  }
</style>
