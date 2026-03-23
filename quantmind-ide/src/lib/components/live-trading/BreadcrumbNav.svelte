<script lang="ts">
  /**
   * BreadcrumbNav - Navigation Breadcrumbs
   *
   * Format: Live Trading > [EA Name]
   */
  import { ChevronRight, Home } from 'lucide-svelte';
  import { selectBot } from '$lib/stores/trading';

  export let botName: string = '';

  function goBack() {
    selectBot(null);
  }
</script>

<nav class="breadcrumb">
  <button class="crumb home" on:click={goBack}>
    <Home size={14} />
    <span>Live Trading</span>
  </button>

  <ChevronRight size={12} class="separator" />

  <span class="crumb current">{botName}</span>
</nav>

<style>
  .breadcrumb {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 12px 16px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
  }

  .crumb {
    display: flex;
    align-items: center;
    gap: 6px;
    background: none;
    border: none;
    color: #888;
    cursor: pointer;
    padding: 4px 8px;
    border-radius: 4px;
    transition: color 0.15s ease, background 0.15s ease;
  }

  .crumb:hover {
    color: #00d4ff;
    background: rgba(0, 212, 255, 0.1);
  }

  .crumb.home {
    color: #aaa;
  }

  .crumb.current {
    color: #e0e0e0;
    cursor: default;
  }

  .crumb.current:hover {
    background: transparent;
    color: #e0e0e0;
  }

  .separator {
    color: #444;
    flex-shrink: 0;
  }

  :global(.breadcrumb .separator) {
    color: #444;
  }
</style>
