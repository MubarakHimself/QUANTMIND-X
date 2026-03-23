<script lang="ts">
  /**
   * A/B Comparison View Component
   *
   * Side-by-side metrics for two strategy variants.
   * Implements Frosted Terminal aesthetic.
   */
  import { onMount, onDestroy } from 'svelte';
  import { abRaceStore, winningVariant, type ABComparison } from '$lib/stores/ab-race';
  import { Crown, TrendingUp, TrendingDown, DollarSign, Activity, BarChart3 } from 'lucide-svelte';

  // Props
  export let strategyId: string = '';
  export let variantA: string = 'variant_a';
  export let variantB: string = 'variant_b';

  let comparison: ABComparison | null = null;
  let loading = false;
  let error: string | null = null;
  let polling = false;
  let winner: string | null = null;

  // Subscribe to stores
  const unsubscribeRace = abRaceStore.subscribe(state => {
    comparison = state.comparison;
    loading = state.loading;
    error = state.error;
    polling = state.polling;
  });

  const unsubscribeWinner = winningVariant.subscribe(w => {
    winner = w;
  });

  onMount(async () => {
    if (strategyId) {
      await abRaceStore.loadComparison(strategyId, variantA, variantB);
      abRaceStore.startPolling(strategyId, variantA, variantB);
    }
  });

  onDestroy(() => {
    abRaceStore.stopPolling();
    unsubscribeRace();
    unsubscribeWinner();
  });

  function formatCurrency(value: number): string {
    const sign = value >= 0 ? '+' : '';
    return `${sign}$${value.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
  }

  function formatPercent(value: number): string {
    return `${value.toFixed(2)}%`;
  }

  function formatNumber(value: number): string {
    return value.toLocaleString('en-US');
  }

  function getMetricClass(valueA: number, valueB: number): string {
    if (valueA > valueB) return 'variant-a-wins';
    if (valueB > valueA) return 'variant-b-wins';
    return 'tied';
  }
</script>

<div class="ab-comparison">
  <header class="comparison-header">
    <h2>A/B Race Board</h2>
    <div class="header-controls">
      {#if polling}
        <span class="live-indicator">
          <Activity size={12} />
          Live
        </span>
      {/if}
    </div>
  </header>

  {#if loading && !comparison}
    <div class="loading-state">
      <Activity size={32} />
      <span>Loading comparison...</span>
    </div>
  {:else if error}
    <div class="error-state">
      <span>Error: {error}</span>
    </div>
  {:else if comparison}
    <div class="comparison-grid">
      <!-- Variant A Column -->
      <div class="variant-column" class:winner={winner === 'A'}>
        <div class="variant-header">
          <h3>{comparison.variant_a}</h3>
          {#if winner === 'A'}
            <span class="crown-badge">
              <Crown size={14} />
              Winner
            </span>
          {/if}
        </div>

        <div class="metrics-grid">
          <!-- P&L -->
          <div class="metric-card">
            <div class="metric-icon">
              <DollarSign size={16} />
            </div>
            <div class="metric-content">
              <span class="metric-label">P&L</span>
              <span class="metric-value" class:positive={comparison.metrics_a.pnl >= 0} class:negative={comparison.metrics_a.pnl < 0}>
                {formatCurrency(comparison.metrics_a.pnl)}
              </span>
            </div>
          </div>

          <!-- Trade Count -->
          <div class="metric-card">
            <div class="metric-icon">
              <Activity size={16} />
            </div>
            <div class="metric-content">
              <span class="metric-label">Trades</span>
              <span class="metric-value">{formatNumber(comparison.metrics_a.trade_count)}</span>
            </div>
          </div>

          <!-- Drawdown -->
          <div class="metric-card">
            <div class="metric-icon">
              <TrendingDown size={16} />
            </div>
            <div class="metric-content">
              <span class="metric-label">Drawdown</span>
              <span class="metric-value negative">{formatPercent(comparison.metrics_a.drawdown)}</span>
            </div>
          </div>

          <!-- Sharpe -->
          <div class="metric-card">
            <div class="metric-icon">
              <BarChart3 size={16} />
            </div>
            <div class="metric-content">
              <span class="metric-label">Sharpe</span>
              <span class="metric-value">{comparison.metrics_a.sharpe.toFixed(2)}</span>
            </div>
          </div>

          <!-- Win Rate -->
          <div class="metric-card">
            <div class="metric-icon">
              <TrendingUp size={16} />
            </div>
            <div class="metric-content">
              <span class="metric-label">Win Rate</span>
              <span class="metric-value">{formatPercent(comparison.metrics_a.win_rate)}</span>
            </div>
          </div>

          <!-- Profit Factor -->
          <div class="metric-card">
            <div class="metric-icon">
              <BarChart3 size={16} />
            </div>
            <div class="metric-content">
              <span class="metric-label">Profit Factor</span>
              <span class="metric-value">{comparison.metrics_a.profit_factor.toFixed(2)}</span>
            </div>
          </div>
        </div>
      </div>

      <!-- VS Divider -->
      <div class="vs-divider">
        <span>VS</span>
      </div>

      <!-- Variant B Column -->
      <div class="variant-column" class:winner={winner === 'B'}>
        <div class="variant-header">
          <h3>{comparison.variant_b}</h3>
          {#if winner === 'B'}
            <span class="crown-badge">
              <Crown size={14} />
              Winner
            </span>
          {/if}
        </div>

        <div class="metrics-grid">
          <!-- P&L -->
          <div class="metric-card">
            <div class="metric-icon">
              <DollarSign size={16} />
            </div>
            <div class="metric-content">
              <span class="metric-label">P&L</span>
              <span class="metric-value" class:positive={comparison.metrics_b.pnl >= 0} class:negative={comparison.metrics_b.pnl < 0}>
                {formatCurrency(comparison.metrics_b.pnl)}
              </span>
            </div>
          </div>

          <!-- Trade Count -->
          <div class="metric-card">
            <div class="metric-icon">
              <Activity size={16} />
            </div>
            <div class="metric-content">
              <span class="metric-label">Trades</span>
              <span class="metric-value">{formatNumber(comparison.metrics_b.trade_count)}</span>
            </div>
          </div>

          <!-- Drawdown -->
          <div class="metric-card">
            <div class="metric-icon">
              <TrendingDown size={16} />
            </div>
            <div class="metric-content">
              <span class="metric-label">Drawdown</span>
              <span class="metric-value negative">{formatPercent(comparison.metrics_b.drawdown)}</span>
            </div>
          </div>

          <!-- Sharpe -->
          <div class="metric-card">
            <div class="metric-icon">
              <BarChart3 size={16} />
            </div>
            <div class="metric-content">
              <span class="metric-label">Sharpe</span>
              <span class="metric-value">{comparison.metrics_b.sharpe.toFixed(2)}</span>
            </div>
          </div>

          <!-- Win Rate -->
          <div class="metric-card">
            <div class="metric-icon">
              <TrendingUp size={16} />
            </div>
            <div class="metric-content">
              <span class="metric-label">Win Rate</span>
              <span class="metric-value">{formatPercent(comparison.metrics_b.win_rate)}</span>
            </div>
          </div>

          <!-- Profit Factor -->
          <div class="metric-card">
            <div class="metric-icon">
              <BarChart3 size={16} />
            </div>
            <div class="metric-content">
              <span class="metric-label">Profit Factor</span>
              <span class="metric-value">{comparison.metrics_b.profit_factor.toFixed(2)}</span>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Statistical Significance Banner -->
    {#if comparison.statistical_significance}
      <div class="stat-sig-banner" class:significant={comparison.statistical_significance.is_significant}>
        {#if comparison.statistical_significance.is_significant}
          <Crown size={16} />
          <span>
            <strong>Statistically Significant!</strong>
            Variant {comparison.statistical_significance.winner} has edge
            (p={comparison.statistical_significance.p_value}, conf={comparison.statistical_significance.confidence_level}%)
          </span>
        {:else}
          <span>
            Waiting for significance (min 50 trades each, p &lt; 0.05).
            Current: {comparison.statistical_significance.sample_size_a} / {comparison.statistical_significance.sample_size_b} trades
          </span>
        {/if}
      </div>
    {/if}

    <!-- Last Updated -->
    <div class="timestamp">
      Last updated: {new Date(comparison.timestamp).toLocaleTimeString()}
    </div>
  {:else}
    <div class="empty-state">
      <span>Select two variants to compare</span>
    </div>
  {/if}
</div>

<style>
  .ab-comparison {
    display: flex;
    flex-direction: column;
    gap: 16px;
    padding: 16px;
    background: rgba(10, 15, 26, 0.95);
    backdrop-filter: blur(12px);
    border-radius: 8px;
    border: 1px solid rgba(0, 212, 255, 0.1);
  }

  .comparison-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding-bottom: 12px;
    border-bottom: 1px solid rgba(0, 212, 255, 0.1);
  }

  .comparison-header h2 {
    font-family: 'JetBrains Mono', monospace;
    font-size: 16px;
    font-weight: 600;
    color: #a855f7;
    margin: 0;
  }

  .header-controls {
    display: flex;
    align-items: center;
    gap: 12px;
  }

  .live-indicator {
    display: flex;
    align-items: center;
    gap: 4px;
    font-size: 11px;
    color: #22c55e;
    animation: pulse 2s infinite;
  }

  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
  }

  .loading-state, .error-state, .empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 40px;
    color: rgba(255, 255, 255, 0.5);
    gap: 12px;
  }

  .error-state {
    color: #ff3b3b;
  }

  .comparison-grid {
    display: grid;
    grid-template-columns: 1fr auto 1fr;
    gap: 16px;
    align-items: start;
  }

  .variant-column {
    display: flex;
    flex-direction: column;
    gap: 12px;
    padding: 16px;
    background: rgba(8, 13, 20, 0.6);
    border-radius: 8px;
    border: 1px solid rgba(0, 212, 255, 0.1);
    transition: all 0.3s ease;
  }

  .variant-column.winner {
    border-color: #ffaa00;
    box-shadow: 0 0 20px rgba(255, 170, 0, 0.15);
  }

  .variant-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .variant-header h3 {
    font-family: 'JetBrains Mono', monospace;
    font-size: 14px;
    font-weight: 600;
    color: #00d4ff;
    margin: 0;
  }

  .crown-badge {
    display: flex;
    align-items: center;
    gap: 4px;
    padding: 4px 8px;
    background: rgba(255, 170, 0, 0.15);
    border: 1px solid rgba(255, 170, 0, 0.3);
    border-radius: 4px;
    color: #ffaa00;
    font-size: 11px;
    font-weight: 500;
  }

  .vs-divider {
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 20px 8px;
    color: rgba(255, 255, 255, 0.3);
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    font-weight: 600;
  }

  .metrics-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 8px;
  }

  .metric-card {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 10px;
    background: rgba(0, 0, 0, 0.2);
    border-radius: 6px;
  }

  .metric-icon {
    display: flex;
    align-items: center;
    justify-content: center;
    color: rgba(255, 255, 255, 0.4);
  }

  .metric-content {
    display: flex;
    flex-direction: column;
    gap: 2px;
  }

  .metric-label {
    font-size: 10px;
    color: rgba(255, 255, 255, 0.4);
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  .metric-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 14px;
    font-weight: 600;
    color: #fff;
  }

  .metric-value.positive {
    color: #22c55e;
  }

  .metric-value.negative {
    color: #ff3b3b;
  }

  .stat-sig-banner {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 12px 16px;
    background: rgba(0, 0, 0, 0.2);
    border-radius: 6px;
    font-size: 12px;
    color: rgba(255, 255, 255, 0.6);
  }

  .stat-sig-banner.significant {
    background: rgba(255, 170, 0, 0.15);
    border: 1px solid rgba(255, 170, 0, 0.3);
    color: #ffaa00;
  }

  .timestamp {
    font-size: 11px;
    color: rgba(255, 255, 255, 0.3);
    text-align: right;
  }
</style>