<script lang="ts">
  import { ChevronRight, Home } from 'lucide-svelte';
  import { activeCanvasStore, CANVASES, type Canvas } from '../stores/canvasStore';

  interface BreadcrumbItem {
    id: string;
    label: string;
    type: 'canvas' | 'subpage';
  }

  interface Props {
    breadcrumbs?: BreadcrumbItem[];
  }

  let { breadcrumbs = [] }: Props = $props();

  // Get current canvas info
  let currentCanvas: Canvas | undefined = $derived(
    CANVASES.find(c => c.id === $activeCanvasStore)
  );

  // Check if we should show breadcrumbs (only when there's a subpage)
  let showBreadcrumbs = $derived(breadcrumbs.length > 0 && breadcrumbs[0]?.type === 'subpage');

  function handleHomeClick() {
    activeCanvasStore.setActiveCanvas('workshop');
  }

  function handleCanvasClick(canvasId: string) {
    activeCanvasStore.setActiveCanvas(canvasId);
  }
</script>

{#if showBreadcrumbs}
  <nav class="breadcrumb-nav" aria-label="Breadcrumb navigation">
    <div class="breadcrumb-list">
      <!-- Canvas home -->
      <button class="breadcrumb-item home" onclick={handleHomeClick}>
        <Home size={14} />
        <span>{currentCanvas?.name || 'Canvas'}</span>
      </button>

      <!-- Separator -->
      <ChevronRight size={14} class="separator" />

      <!-- Sub-pages -->
      {#each breadcrumbs as item}
        {#if item.type === 'subpage'}
          <ChevronRight size={14} class="separator" />
          <span class="breadcrumb-item current">{item.label}</span>
        {/if}
      {/each}
    </div>
  </nav>
{/if}

<style>
  .breadcrumb-nav {
    display: flex;
    align-items: center;
    padding: 8px 16px;
    background: var(--bg-secondary);
    border-bottom: 1px solid var(--border-subtle);
    min-height: 36px;
  }

  .breadcrumb-list {
    display: flex;
    align-items: center;
    gap: 4px;
    font-size: 13px;
  }

  .breadcrumb-item {
    display: flex;
    align-items: center;
    gap: 4px;
    padding: 4px 8px;
    background: transparent;
    border: none;
    color: var(--text-muted);
    cursor: pointer;
    border-radius: 4px;
    transition: all 0.15s ease;
  }

  .breadcrumb-item:hover {
    background: var(--bg-hover);
    color: var(--text-primary);
  }

  .breadcrumb-item.home {
    color: var(--text-secondary);
  }

  .breadcrumb-item.current {
    color: var(--text-primary);
    font-weight: 500;
    cursor: default;
  }

  .breadcrumb-item.current:hover {
    background: transparent;
  }

  .separator {
    color: var(--text-disabled);
    flex-shrink: 0;
  }

  :global(.separator) {
    opacity: 0.5;
  }
</style>
