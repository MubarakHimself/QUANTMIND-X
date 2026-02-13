<script lang="ts">
  import { createEventDispatcher, onMount } from 'svelte';
  import { fade, slide } from 'svelte/transition';
  import { Plus, FileText, TrendingUp, Link, Activity, ChevronDown, ChevronUp, X } from 'lucide-svelte';
  import ContextTag from './ContextTag.svelte';
  import type { ChatContext, FileReference, StrategyReference, BrokerReference, BacktestReference } from '../../stores/chatStore';
  
  // Props
  export let context: ChatContext;
  
  const dispatch = createEventDispatcher();
  
  // State
  let showAddMenu = false;
  let showAll = false;
  const maxVisible = 3;
  
  // Count total context items
  $: totalItems = context.files.length + context.strategies.length + context.brokers.length + context.backtests.length;
  
  // Get all context items flattened with type info
  $: allContextItems = [
    ...context.files.map(item => ({ ...item, type: 'file' as const, icon: FileText })),
    ...context.strategies.map(item => ({ ...item, type: 'strategy' as const, icon: TrendingUp })),
    ...context.brokers.map(item => ({ ...item, type: 'broker' as const, icon: Link })),
    ...context.backtests.map(item => ({ ...item, type: 'backtest' as const, icon: Activity }))
  ];
  
  // Visible items (limited unless showAll)
  $: visibleItems = showAll ? allContextItems : allContextItems.slice(0, maxVisible);
  $: hiddenCount = allContextItems.length - maxVisible;
  
  // Menu options for adding context
  const addOptions = [
    { type: 'file', label: 'Attach File', icon: FileText },
    { type: 'strategy', label: 'Add Strategy', icon: TrendingUp },
    { type: 'broker', label: 'Connect Broker', icon: Link },
    { type: 'backtest', label: 'Add Backtest', icon: Activity }
  ];
  
  // Toggle add menu
  function toggleAddMenu() {
    showAddMenu = !showAddMenu;
  }
  
  // Handle add context
  function handleAddContext(type: string) {
    dispatch('addContext', { type });
    showAddMenu = false;
  }
  
  // Handle remove context
  function handleRemoveContext(type: keyof ChatContext, id: string) {
    dispatch('removeContext', { type, id });
  }
  
  // Toggle show all
  function toggleShowAll() {
    showAll = !showAll;
  }

  // Helper to get context type (avoid TypeScript in template)
  function getContextType(type: string): keyof ChatContext {
    return type as keyof ChatContext;
  }
  
  // Close menu on outside click
  function handleClickOutside(e: MouseEvent) {
    const target = e.target as HTMLElement;
    if (!target.closest('.add-menu-container')) {
      showAddMenu = false;
    }
  }
</script>

<svelte:window on:click={handleClickOutside} />

{#if totalItems > 0}
  <div class="context-bar" transition:slide={{ duration: 200 }}>
    <!-- Context Tags -->
    <div class="context-tags">
      {#each visibleItems as item (item.id)}
        <ContextTag
          item={item}
          on:remove={() => handleRemoveContext(getContextType(item.type), item.id)}
        />
      {/each}
      
      <!-- Show more/less button -->
      {#if hiddenCount > 0 && !showAll}
        <button 
          class="show-more-btn" 
          on:click|stopPropagation={toggleShowAll}
          title="Show all context items"
        >
          +{hiddenCount} more
        </button>
      {:else if showAll && allContextItems.length > maxVisible}
        <button 
          class="show-more-btn" 
          on:click|stopPropagation={toggleShowAll}
          title="Show fewer"
        >
          Show less
        </button>
      {/if}
    </div>
    
    <!-- Add Context Button -->
    <div class="add-menu-container">
      <button 
        class="add-btn" 
        on:click|stopPropagation={toggleAddMenu}
        title="Add context"
        aria-label="Add context item"
      >
        <Plus size={14} />
      </button>
      
      {#if showAddMenu}
        <!-- svelte-ignore a11y-click-events-have-key-events a11y-no-static-element-interactions -->
        <div class="add-menu" transition:fade={{ duration: 100 }} on:click|stopPropagation>
          {#each addOptions as option}
            <button 
              class="menu-option" 
              on:click={() => handleAddContext(option.type)}
            >
              <svelte:component this={option.icon} size={14} />
              <span>{option.label}</span>
            </button>
          {/each}
        </div>
      {/if}
    </div>
  </div>
{:else}
  <!-- Empty context bar with add button -->
  <div class="context-bar empty" transition:slide={{ duration: 200 }}>
    <div class="add-menu-container">
      <button 
        class="add-btn empty" 
        on:click|stopPropagation={toggleAddMenu}
        title="Add context"
        aria-label="Add context item"
      >
        <Plus size={14} />
        <span>Add Context</span>
      </button>
      
      {#if showAddMenu}
        <!-- svelte-ignore a11y-click-events-have-key-events a11y-no-static-element-interactions -->
        <div class="add-menu" transition:fade={{ duration: 100 }} on:click|stopPropagation>
          {#each addOptions as option}
            <button 
              class="menu-option" 
              on:click={() => handleAddContext(option.type)}
            >
              <svelte:component this={option.icon} size={14} />
              <span>{option.label}</span>
            </button>
          {/each}
        </div>
      {/if}
    </div>
  </div>
{/if}

<style>
  .context-bar {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 12px;
    background: var(--bg-primary);
    border-bottom: 1px solid var(--border-subtle);
    min-height: 40px;
  }
  
  .context-bar.empty {
    justify-content: center;
    background: transparent;
    border-bottom: none;
    padding: 4px 12px;
  }
  
  .context-tags {
    display: flex;
    align-items: center;
    gap: 6px;
    flex: 1;
    flex-wrap: wrap;
  }
  
  .show-more-btn {
    padding: 4px 8px;
    background: var(--bg-tertiary);
    border: 1px solid var(--border-subtle);
    border-radius: 4px;
    color: var(--text-muted);
    font-size: 10px;
    cursor: pointer;
    transition: all 0.15s;
  }
  
  .show-more-btn:hover {
    background: var(--bg-secondary);
    color: var(--text-primary);
  }
  
  .add-menu-container {
    position: relative;
  }
  
  .add-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 4px;
    padding: 6px;
    background: var(--bg-tertiary);
    border: 1px solid var(--border-subtle);
    border-radius: 6px;
    color: var(--text-muted);
    cursor: pointer;
    transition: all 0.15s;
  }
  
  .add-btn:hover {
    background: var(--bg-secondary);
    color: var(--text-primary);
    border-color: var(--accent-primary);
  }
  
  .add-btn.empty {
    padding: 6px 12px;
    font-size: 11px;
  }
  
  .add-menu {
    position: absolute;
    top: 100%;
    right: 0;
    margin-top: 4px;
    background: var(--bg-secondary);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
    padding: 4px;
    min-width: 160px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    z-index: 100;
  }
  
  .menu-option {
    display: flex;
    align-items: center;
    gap: 8px;
    width: 100%;
    padding: 8px 12px;
    background: transparent;
    border: none;
    border-radius: 4px;
    color: var(--text-primary);
    font-size: 12px;
    cursor: pointer;
    transition: background 0.15s;
    text-align: left;
  }
  
  .menu-option:hover {
    background: var(--bg-tertiary);
  }
  
  .menu-option span {
    flex: 1;
  }
</style>
