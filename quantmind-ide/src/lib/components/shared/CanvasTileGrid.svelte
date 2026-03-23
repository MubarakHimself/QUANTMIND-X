<script lang="ts">
  /**
   * CanvasTileGrid — primary layout wrapper for all non-Workshop canvases.
   * Arch-UI-6: All canvases except Workshop must use this wrapper.
   * Story 12-3
   */
  import Breadcrumb from './Breadcrumb.svelte';

  interface Props {
    title: string;
    subtitle?: string;
    dept?: string;          // sets data-dept on root
    showBackButton?: boolean;
    onBack?: () => void;
  }

  let { title, subtitle, dept = '', showBackButton = false, onBack }: Props = $props();
</script>

<div class="canvas-tile-grid" data-dept={dept || undefined}>
  <header class="ctg-header">
    {#if showBackButton && onBack}
      <Breadcrumb {onBack} />
    {/if}
    <h1 class="ctg-title">{title}</h1>
    {#if subtitle}<span class="ctg-subtitle">{subtitle}</span>{/if}
    <slot name="header-actions" />
  </header>
  <div class="ctg-grid">
    <slot />
  </div>
</div>

<style>
  .canvas-tile-grid {
    display: flex;
    flex-direction: column;
    height: 100%;
    overflow: hidden;
  }

  .ctg-header {
    display: flex;
    align-items: center;
    gap: var(--space-3);
    padding: var(--space-4) var(--space-4) var(--space-3);
    border-bottom: 1px solid var(--color-border-subtle);
    flex-shrink: 0;
  }

  .ctg-title {
    font-family: var(--font-heading);
    font-weight: 800;
    font-size: var(--text-xl);
    color: var(--color-text-primary);
    margin: 0;
    line-height: 1.2;
  }

  .ctg-subtitle {
    font-family: var(--font-ambient);
    font-size: var(--text-xs);
    color: var(--color-text-muted);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-left: var(--space-2);
  }

  .ctg-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(var(--tile-min-width), 1fr));
    gap: var(--tile-gap);
    padding: var(--space-4);
    overflow-y: auto;
    align-content: start;
  }
</style>
