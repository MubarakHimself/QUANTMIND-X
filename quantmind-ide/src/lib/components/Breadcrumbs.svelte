<script lang="ts">
  import { ChevronRight, Home } from 'lucide-svelte';
  
  export let items: Array<{ label: string; path?: string }> = [];
  export let onNavigate: ((path: string) => void) | undefined = undefined;
  export let showHome = true;
  
  function handleClick(item: { label: string; path?: string }, index: number) {
    if (item.path && onNavigate && index < items.length - 1) {
      onNavigate(item.path);
    }
  }
</script>

<nav class="breadcrumbs" aria-label="Breadcrumb navigation">
  <ol class="breadcrumb-list">
    {#if showHome}
      <li class="breadcrumb-item">
        <button 
          class="breadcrumb-link home" 
          on:click={() => onNavigate?.('/')}
          aria-label="Go to home"
        >
          <Home size={14} />
        </button>
        <span class="separator-icon" aria-hidden="true">
          <ChevronRight size={12} />
        </span>
      </li>
    {/if}
    
    {#each items as item, index}
      <li class="breadcrumb-item">
        {#if index === items.length - 1}
          <span class="breadcrumb-current" aria-current="page">
            {item.label}
          </span>
        {:else}
          <button 
            class="breadcrumb-link" 
            on:click={() => handleClick(item, index)}
          >
            {item.label}
          </button>
          <span class="separator-icon" aria-hidden="true">
            <ChevronRight size={12} />
          </span>
        {/if}
      </li>
    {/each}
  </ol>
</nav>

<style>
  .breadcrumbs {
    display: flex;
    align-items: center;
    padding: 8px 12px;
    background: var(--bg-tertiary);
    border-bottom: 1px solid var(--border-subtle);
    font-size: 12px;
  }
  
  .breadcrumb-list {
    display: flex;
    align-items: center;
    gap: 4px;
    margin: 0;
    padding: 0;
    list-style: none;
  }
  
  .breadcrumb-item {
    display: flex;
    align-items: center;
    gap: 4px;
  }
  
  .breadcrumb-link {
    background: transparent;
    border: none;
    color: var(--text-secondary);
    cursor: pointer;
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 12px;
    transition: all 0.15s ease;
  }
  
  .breadcrumb-link:hover {
    background: var(--bg-secondary);
    color: var(--text-primary);
  }
  
  .breadcrumb-link.home {
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 4px;
  }
  
  .breadcrumb-current {
    color: var(--text-primary);
    font-weight: 500;
    padding: 2px 6px;
  }
  
  .separator-icon {
    color: var(--text-muted);
    flex-shrink: 0;
    display: flex;
    align-items: center;
  }
</style>
