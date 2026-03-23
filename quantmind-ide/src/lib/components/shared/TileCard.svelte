<script lang="ts">
  /**
   * TileCard — glass tile with hover state, size variants, and epic badge.
   * AC 12-3-3: Hover border → rgba(255,255,255,0.13)
   * AC 12-3-6: CRM pattern — max 2–4 data points visible on face
   * AC 12-3-7: Financial typography tokens enforced via CSS
   * AC 12-3-8: All new canvas tiles import from shared/TileCard
   * Story 12-3
   */
  import SkeletonLoader from './SkeletonLoader.svelte';

  interface Props {
    title: string;
    size?: 'sm' | 'md' | 'lg' | 'xl';  // xl = full width, lg = span 2
    epicOwner?: string;                  // renders badge e.g. "Epic 4" when set
    isLoading?: boolean;                 // shows SkeletonLoader when true
    navigable?: boolean;                 // shows "→ view detail" hint on hover
    onNavigate?: () => void;
  }

  let { title, size = 'md', epicOwner, isLoading = false, navigable = false, onNavigate }: Props = $props();
</script>

<div
  class="tile-card tile-card--{size}"
  class:tile-card--navigable={navigable}
  onclick={navigable && onNavigate ? onNavigate : undefined}
  role={navigable ? 'button' : undefined}
  tabindex={navigable ? 0 : undefined}
  onkeydown={navigable && onNavigate ? (e) => { if (e.key === 'Enter' || e.key === ' ') onNavigate?.(); } : undefined}
>
  <div class="tile-header">
    <span class="tile-title">{title}</span>
    {#if epicOwner}
      <span class="epic-badge">{epicOwner}</span>
    {/if}
  </div>
  <div class="tile-body">
    {#if isLoading}
      <SkeletonLoader lines={3} />
    {:else}
      <slot />
    {/if}
  </div>
  {#if navigable}
    <div class="tile-hint">→ view detail</div>
  {/if}
</div>

<style>
  .tile-card {
    background: var(--glass-content-bg);
    backdrop-filter: var(--glass-blur);
    -webkit-backdrop-filter: var(--glass-blur);
    border: 1px solid var(--color-border-subtle);
    border-radius: 8px;
    padding: var(--space-4);
    transition: border-color 0.15s ease, background 0.15s ease;
    overflow: hidden;
    min-height: 120px;
  }

  .tile-card:hover {
    border-color: rgba(255, 255, 255, 0.13);
    background: rgba(16, 24, 36, 0.5);
  }

  .tile-card--xl {
    grid-column: 1 / -1;
  }

  .tile-card--lg {
    grid-column: span 2;
  }

  .tile-card--navigable {
    cursor: pointer;
  }

  .tile-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: var(--space-3);
  }

  .tile-title {
    font-family: var(--font-ambient);
    font-size: var(--text-xs);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: var(--color-text-muted);
  }

  .epic-badge {
    font-family: var(--font-ambient);
    font-size: var(--text-xs);
    padding: 2px 6px;
    background: rgba(255, 255, 255, 0.06);
    border: 1px solid var(--color-border-subtle);
    border-radius: 4px;
    color: var(--color-text-muted);
    flex-shrink: 0;
  }

  .tile-body {
    /* CRM pattern: max 2-4 data points visible on tile face (AC 12-3-6) */
    /* --tile-body-max defaults to 140px; ghost-panel theme can override via app.css */
    max-height: var(--tile-body-max, 140px);
    overflow: hidden;
  }

  /* Financial typography: all numeric data in tile body */
  .tile-body :global(.financial-value),
  .tile-body :global(.pnl),
  .tile-body :global(.lot-size),
  .tile-body :global(.risk-score),
  .tile-body :global(.timestamp) {
    font-family: var(--font-data);
  }

  /* Section label headings in tile */
  .tile-body :global(.section-label) {
    font-family: var(--font-ambient);
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  .tile-hint {
    font-family: var(--font-ambient);
    font-size: var(--text-xs);
    color: var(--dept-accent);
    opacity: 0;
    transition: opacity 0.15s ease;
    text-align: right;
    margin-top: var(--space-2);
  }

  .tile-card--navigable:hover .tile-hint {
    opacity: 1;
  }
</style>
