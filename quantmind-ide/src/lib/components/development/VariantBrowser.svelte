<script lang="ts">
  /**
   * Variant Browser Component
   *
   * Shows EA variant grid for the Development canvas.
   * Implements Frosted Terminal aesthetic.
   */
  import { onMount } from 'svelte';
  import { variantBrowserStore, type VariantInfo } from '$lib/stores/variant-browser';
  import { Folder, FileCode, GitBranch, TrendingUp, DollarSign, Clock } from 'lucide-svelte';

  let strategies: any[] = [];
  let isLoading = false;
  let error: string | null = null;
  let selectedStrategy: string | null = null;
  let selectedVariant: string | null = null;

  // Subscribe to store
  variantBrowserStore.subscribe(state => {
    strategies = state.strategies;
    isLoading = state.isLoading;
    error = state.error;
    selectedStrategy = state.selectedStrategy;
    selectedVariant = state.selectedVariant;
  });

  onMount(async () => {
    await variantBrowserStore.loadVariants();
  });

  function formatCurrency(value: number): string {
    const sign = value >= 0 ? '+' : '';
    return `${sign}$${value.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
  }

  function formatPercent(value: number): string {
    const sign = value >= 0 ? '+' : '';
    return `${sign}${value.toFixed(1)}%`;
  }

  function getPromotionStatusColor(status: string): string {
    switch (status) {
      case 'live': return '#22c55e';
      case 'sit': return '#f59e0b';
      case 'paper_trading': return '#3b82f6';
      default: return '#6b7280';
    }
  }

  function getPromotionLabel(status: string): string {
    switch (status) {
      case 'live': return 'Live';
      case 'sit': return 'SIT';
      case 'paper_trading': return 'Paper';
      default: return 'Dev';
    }
  }

  function selectVariant(strategyId: string, variantType: string) {
    variantBrowserStore.selectVariant(strategyId, variantType);
  }
</script>

<div class="variant-browser">
  {#if isLoading}
    <div class="loading">
      <div class="spinner"></div>
      <span>Loading variants...</span>
    </div>
  {:else if error}
    <div class="error">
      <span>{error}</span>
      <button onclick={() => variantBrowserStore.loadVariants()}>Retry</button>
    </div>
  {:else}
    <div class="variant-grid">
      <div class="grid-header">
        <div class="strategy-col">Strategy</div>
        <div class="variant-col">Vanilla</div>
        <div class="variant-col">Spiced</div>
        <div class="variant-col">Mode B</div>
        <div class="variant-col">Mode C</div>
      </div>

      {#each strategies as strategy}
        <div class="grid-row">
          <div class="strategy-col">
            <Folder size={14} />
            <span class="strategy-name">{strategy.strategy_name}</span>
            <span class="strategy-id">{strategy.strategy_id}</span>
          </div>

          {#each strategy.variants as variant}
            <button
              class="variant-col variant-card"
              class:active={selectedStrategy === strategy.strategy_id && selectedVariant === variant.variant_type}
              onclick={() => selectVariant(strategy.strategy_id, variant.variant_type)}
            >
              <div class="variant-header">
                <FileCode size={12} />
                <span class="version">{variant.version_tag}</span>
              </div>

              {#if variant.backtest}
                <div class="backtest-summary">
                  <div class="metric">
                    <TrendingUp size={10} />
                    <span class:value={variant.backtest.total_pnl >= 0}>
                      {formatCurrency(variant.backtest.total_pnl)}
                    </span>
                  </div>
                  <div class="metric">
                    <span class="sharpe">Sharpe: {variant.backtest.sharpe_ratio}</span>
                  </div>
                  <div class="metric">
                    <span class="drawdown" class:negative={variant.backtest.max_drawdown > 10}>
                      DD: {variant.backtest.max_drawdown}%
                    </span>
                  </div>
                </div>
              {/if}

              <div class="promotion-status" style="--status-color: {getPromotionStatusColor(variant.promotion_status)}">
                <GitBranch size={10} />
                <span>{getPromotionLabel(variant.promotion_status)}</span>
              </div>
            </button>
          {/each}
        </div>
      {/each}
    </div>

    {#if strategies.length === 0}
      <div class="empty-state">
        <Folder size={32} />
        <p>No strategies found</p>
        <span>Create a strategy to see variants here</span>
      </div>
    {/if}
  {/if}
</div>

<style>
  .variant-browser {
    height: 100%;
    overflow: auto;
    padding: 16px;
    background: rgba(10, 15, 26, 0.95);
  }

  .loading {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 12px;
    color: rgba(255, 255, 255, 0.5);
    font-family: 'JetBrains Mono', monospace;
    font-size: 14px;
    height: 200px;
  }

  .spinner {
    width: 16px;
    height: 16px;
    border: 2px solid rgba(0, 212, 255, 0.2);
    border-top-color: #00d4ff;
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
  }

  @keyframes spin {
    to { transform: rotate(360deg); }
  }

  .error {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 12px;
    color: #ef4444;
    font-family: 'JetBrains Mono', monospace;
    font-size: 14px;
    padding: 24px;
  }

  .error button {
    padding: 8px 16px;
    background: rgba(239, 68, 68, 0.1);
    border: 1px solid rgba(239, 68, 68, 0.3);
    border-radius: 6px;
    color: #ef4444;
    font-family: 'JetBrains Mono', monospace;
    cursor: pointer;
  }

  .variant-grid {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .grid-header {
    display: grid;
    grid-template-columns: 200px repeat(4, 1fr);
    gap: 8px;
    padding: 12px;
    background: rgba(8, 13, 20, 0.6);
    border-radius: 8px;
    border: 1px solid rgba(0, 212, 255, 0.1);
  }

  .grid-header .strategy-col {
    color: rgba(255, 255, 255, 0.6);
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  .grid-header .variant-col {
    color: rgba(255, 255, 255, 0.4);
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    text-align: center;
    text-transform: uppercase;
  }

  .grid-row {
    display: grid;
    grid-template-columns: 200px repeat(4, 1fr);
    gap: 8px;
  }

  .strategy-col {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 12px;
    color: rgba(255, 255, 255, 0.8);
    font-family: 'JetBrains Mono', monospace;
  }

  .strategy-col :global(svg) {
    color: #a855f7;
  }

  .strategy-name {
    font-size: 13px;
    font-weight: 500;
  }

  .strategy-id {
    font-size: 10px;
    color: rgba(255, 255, 255, 0.3);
  }

  .variant-card {
    display: flex;
    flex-direction: column;
    gap: 8px;
    padding: 12px;
    background: rgba(8, 13, 20, 0.6);
    border: 1px solid rgba(0, 212, 255, 0.1);
    border-radius: 8px;
    cursor: pointer;
    transition: all 0.2s ease;
    text-align: left;
  }

  .variant-card:hover {
    background: rgba(0, 212, 255, 0.05);
    border-color: rgba(0, 212, 255, 0.3);
  }

  .variant-card.active {
    background: rgba(0, 212, 255, 0.1);
    border-color: #00d4ff;
  }

  .variant-header {
    display: flex;
    align-items: center;
    gap: 6px;
    color: #00d4ff;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
  }

  .version {
    color: rgba(255, 255, 255, 0.6);
  }

  .backtest-summary {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .metric {
    display: flex;
    align-items: center;
    gap: 4px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: rgba(255, 255, 255, 0.5);
  }

  .metric span.value {
    color: #22c55e;
  }

  .metric span.value.negative {
    color: #ef4444;
  }

  .sharpe {
    color: rgba(255, 255, 255, 0.5);
  }

  .drawdown {
    color: rgba(255, 255, 255, 0.5);
  }

  .drawdown.negative {
    color: #f59e0b;
  }

  .promotion-status {
    display: flex;
    align-items: center;
    gap: 4px;
    padding: 4px 8px;
    background: rgba(var(--status-color), 0.1);
    border-radius: 4px;
    color: var(--status-color);
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    text-transform: uppercase;
    margin-top: auto;
  }

  .promotion-status :global(svg) {
    color: var(--status-color);
  }

  .empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 12px;
    height: 200px;
    color: rgba(255, 255, 255, 0.3);
    font-family: 'JetBrains Mono', monospace;
  }

  .empty-state p {
    margin: 0;
    font-size: 14px;
    color: rgba(255, 255, 255, 0.5);
  }

  .empty-state span {
    font-size: 12px;
  }
</style>