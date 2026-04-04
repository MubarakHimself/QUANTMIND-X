<script lang="ts">
  import { createBubbler, stopPropagation } from 'svelte/legacy';

  const bubble = createBubbler();
  import { createEventDispatcher, onMount } from 'svelte';
  import { fade, slide } from 'svelte/transition';
  import { X, FileText, TrendingUp, Link2, Activity, Loader, Search } from 'lucide-svelte';
  import { getStrategies, getBots, getBacktestResults, type StrategyFolder, type Bot } from '../../api';
  import type { FileReference, StrategyReference, BrokerReference, BacktestReference } from '../../stores/chatStore';
  import { contextManager } from '../../services/contextManager';
  
  
  interface Props {
    // Props
    type: 'file' | 'strategy' | 'broker' | 'backtest';
    isOpen?: boolean;
  }

  let { type, isOpen = true }: Props = $props();
  
  const dispatch = createEventDispatcher();
  
  // State
  let searchQuery = $state('');
  let isLoading = $state(false);
  let error: string | null = $state(null);
  
  // Data from API
  let strategies: StrategyFolder[] = $state([]);
  let brokers: Bot[] = $state([]);
  let backtests: { id: string; name: string; status: string }[] = $state([]);
  let files: { path: string; name: string; type: string }[] = [];
  
  // Load data on mount
  onMount(async () => {
    await loadData();
  });
  
  async function loadData() {
    isLoading = true;
    error = null;
    
    try {
      switch (type) {
        case 'strategy':
          strategies = await getStrategies();
          break;
        case 'broker':
          brokers = await getBots();
          break;
        case 'backtest':
          // Load backtests from strategies that have them
          const allStrategies = await getStrategies();
          backtests = allStrategies
            .filter(s => s.has_backtest)
            .map(s => ({
              id: s.id,
              name: `${s.name} Backtest`,
              status: s.status
            }));
          break;
        case 'file':
          // For files, we'll use a file browser approach
          // This would typically call a file listing API
          // For now, we'll show a file path input
          break;
      }
    } catch (e) {
      error = `Failed to load ${type}s: ${e instanceof Error ? e.message : 'Unknown error'}`;
      console.error('Failed to load data:', e);
    } finally {
      isLoading = false;
    }
  }
  
  // Filter items based on search
  let filteredStrategies = $derived(strategies.filter(s => 
    s.name.toLowerCase().includes(searchQuery.toLowerCase())
  ));
  
  let filteredBrokers = $derived(brokers.filter(b => 
    b.name.toLowerCase().includes(searchQuery.toLowerCase())
  ));
  
  let filteredBacktests = $derived(backtests.filter(b => 
    b.name.toLowerCase().includes(searchQuery.toLowerCase())
  ));
  
  // Handle selection
  function selectStrategy(strategy: StrategyFolder) {
    const item: StrategyReference = {
      id: strategy.id,
      name: strategy.name,
      type: strategy.status
    };
    
    const validation = contextManager.validateStrategy(item);
    if (validation.valid) {
      dispatch('select', { type: 'strategies', item });
    } else {
      dispatch('error', { message: validation.errors.join(', ') });
    }
  }
  
  function selectBroker(broker: Bot) {
    const item: BrokerReference = {
      id: broker.id,
      name: broker.name,
      status: broker.state === 'running' ? 'connected' : 'disconnected'
    };
    
    const validation = contextManager.validateBroker(item);
    if (validation.valid) {
      dispatch('select', { type: 'brokers', item });
    } else {
      dispatch('error', { message: validation.errors.join(', ') });
    }
  }
  
  function selectBacktest(backtest: { id: string; name: string; status: string }) {
    const item: BacktestReference = {
      id: backtest.id,
      name: backtest.name,
      status: backtest.status === 'ready' ? 'completed' : 'pending'
    };
    
    const validation = contextManager.validateBacktest(item);
    if (validation.valid) {
      dispatch('select', { type: 'backtests', item });
    } else {
      dispatch('error', { message: validation.errors.join(', ') });
    }
  }
  
  function selectFile() {
    const path = searchQuery.trim();
    if (!path) {
      dispatch('error', { message: 'Please enter a file path' });
      return;
    }
    
    const item: FileReference = contextManager.createFileReference(path);
    const validation = contextManager.validateFile(item);
    
    if (validation.valid) {
      dispatch('select', { type: 'files', item });
    } else {
      dispatch('error', { message: validation.errors.join(', ') });
    }
  }
  
  // Close picker
  function close() {
    dispatch('close');
  }
  
  // Get title based on type
  function getTitle(): string {
    switch (type) {
      case 'file': return 'Select File';
      case 'strategy': return 'Select Strategy';
      case 'broker': return 'Select Broker';
      case 'backtest': return 'Select Backtest';
      default: return 'Select Item';
    }
  }
  
  // Get icon based on type
  function getIcon() {
    switch (type) {
      case 'file': return FileText;
      case 'strategy': return TrendingUp;
      case 'broker': return Link2;
      case 'backtest': return Activity;
      default: return FileText;
    }
  }
</script>

{#if isOpen}
  {@const SvelteComponent = getIcon()}
  <div class="picker-overlay" transition:fade={{ duration: 150 }} onclick={close}>
    <div class="picker-container" transition:slide={{ duration: 200 }} onclick={stopPropagation(bubble('click'))}>
      <!-- Header -->
      <div class="picker-header">
        <div class="header-title">
          <SvelteComponent size={18} />
          <h3>{getTitle()}</h3>
        </div>
        <button class="close-btn" onclick={close} aria-label="Close">
          <X size={18} />
        </button>
      </div>
      
      <!-- Search -->
      <div class="picker-search">
        <Search size={16} />
        {#if type === 'file'}
          <input 
            type="text" 
            bind:value={searchQuery}
            placeholder="Enter file path (e.g., strategies/my_strategy.mq5)"
            onkeydown={(e) => e.key === 'Enter' && selectFile()}
          />
          <button class="add-btn" onclick={selectFile}>Add</button>
        {:else}
          <input 
            type="text" 
            bind:value={searchQuery}
            placeholder="Search..."
          />
        {/if}
      </div>
      
      <!-- Content -->
      <div class="picker-content">
        {#if isLoading}
          <div class="loading-state">
            <Loader size={24} class="spin" />
            <span>Loading {type}s...</span>
          </div>
        {:else if error}
          <div class="error-state">
            <span>{error}</span>
            <button onclick={loadData}>Retry</button>
          </div>
        {:else}
          <!-- File type shows path input only -->
          {#if type !== 'file'}
            <!-- Strategies list -->
            {#if type === 'strategy'}
              {#if filteredStrategies.length === 0}
                <div class="empty-state">
                  <span>No strategies found</span>
                </div>
              {:else}
                <ul class="item-list">
                  {#each filteredStrategies as strategy}
                    <li onclick={() => selectStrategy(strategy)}>
                      <TrendingUp size={16} />
                      <div class="item-info">
                        <span class="item-name">{strategy.name}</span>
                        <span class="item-meta">
                          Status: {strategy.status}
                          {#if strategy.has_video_ingest}• VideoIngest{/if}
                          {#if strategy.has_source_captions}• Captions{/if}
                          {#if strategy.has_source_audio}• Audio{/if}
                          {#if strategy.has_ea}• EA{/if}
                        </span>
                      </div>
                    </li>
                  {/each}
                </ul>
              {/if}
            {/if}
            
            <!-- Brokers list -->
            {#if type === 'broker'}
              {#if filteredBrokers.length === 0}
                <div class="empty-state">
                  <span>No brokers found</span>
                </div>
              {:else}
                <ul class="item-list">
                  {#each filteredBrokers as broker}
                    <li onclick={() => selectBroker(broker)}>
                      <Link2 size={16} />
                      <div class="item-info">
                        <span class="item-name">{broker.name}</span>
                        <span class="item-meta">
                          {broker.symbol} • {broker.state}
                        </span>
                      </div>
                    </li>
                  {/each}
                </ul>
              {/if}
            {/if}
            
            <!-- Backtests list -->
            {#if type === 'backtest'}
              {#if filteredBacktests.length === 0}
                <div class="empty-state">
                  <span>No backtests found</span>
                </div>
              {:else}
                <ul class="item-list">
                  {#each filteredBacktests as backtest}
                    <li onclick={() => selectBacktest(backtest)}>
                      <Activity size={16} />
                      <div class="item-info">
                        <span class="item-name">{backtest.name}</span>
                        <span class="item-meta">
                          Status: {backtest.status}
                        </span>
                      </div>
                    </li>
                  {/each}
                </ul>
              {/if}
            {/if}
          {/if}
        {/if}
      </div>
    </div>
  </div>
{/if}

<style>
  .picker-overlay {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(0, 0, 0, 0.5);
    backdrop-filter: blur(4px);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 1000;
  }
  
  .picker-container {
    background: var(--color-bg-surface);
    border: 1px solid var(--color-border-subtle);
    border-radius: 12px;
    width: 90%;
    max-width: 400px;
    max-height: 500px;
    display: flex;
    flex-direction: column;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
  }
  
  .picker-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 16px;
    border-bottom: 1px solid var(--color-border-subtle);
  }
  
  .header-title {
    display: flex;
    align-items: center;
    gap: 8px;
    color: var(--color-text-primary);
  }
  
  .header-title h3 {
    margin: 0;
    font-size: 14px;
    font-weight: 600;
  }
  
  .close-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    background: none;
    border: none;
    color: var(--color-text-muted);
    cursor: pointer;
    padding: 4px;
    border-radius: 4px;
    transition: all 0.15s;
  }
  
  .close-btn:hover {
    background: var(--color-bg-elevated);
    color: var(--color-text-primary);
  }
  
  .picker-search {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 12px 16px;
    border-bottom: 1px solid var(--color-border-subtle);
    color: var(--color-text-muted);
  }
  
  .picker-search input {
    flex: 1;
    background: transparent;
    border: none;
    color: var(--color-text-primary);
    font-size: 13px;
    outline: none;
  }
  
  .picker-search input::placeholder {
    color: var(--color-text-muted);
  }
  
  .add-btn {
    background: var(--color-accent-cyan);
    border: none;
    border-radius: 4px;
    color: var(--color-bg-base);
    font-size: 12px;
    padding: 4px 12px;
    cursor: pointer;
    transition: all 0.15s;
  }
  
  .add-btn:hover {
    background: var(--color-accent-amber);
  }
  
  .picker-content {
    flex: 1;
    overflow-y: auto;
    padding: 8px;
  }
  
  .loading-state,
  .error-state,
  .empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 8px;
    padding: 32px;
    color: var(--color-text-muted);
    font-size: 13px;
  }
  
  .error-state button {
    background: var(--color-bg-elevated);
    border: 1px solid var(--color-border-subtle);
    border-radius: 4px;
    color: var(--color-text-primary);
    padding: 4px 12px;
    cursor: pointer;
    font-size: 12px;
  }
  
  .item-list {
    list-style: none;
    margin: 0;
    padding: 0;
  }
  
  .item-list li {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 12px;
    border-radius: 8px;
    cursor: pointer;
    transition: background 0.15s;
    color: var(--color-text-secondary);
  }
  
  .item-list li:hover {
    background: var(--color-bg-elevated);
    color: var(--color-text-primary);
  }
  
  .item-info {
    display: flex;
    flex-direction: column;
    gap: 2px;
  }
  
  .item-name {
    font-size: 13px;
    font-weight: 500;
  }
  
  .item-meta {
    font-size: 11px;
    color: var(--color-text-muted);
  }
  
  .spin {
    animation: spin 1s linear infinite;
  }
  
  @keyframes spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
  }
</style>
