<script lang="ts">
  /**
   * PerformancePanel - Portfolio-Level Performance Metrics
   *
   * Story 9.4: Portfolio Canvas — Attribution, Correlation Matrix & Performance
   * AC all: Display portfolio-level metrics
   */
  import { onMount } from 'svelte';
  import { portfolioStore, performance, portfolioLoading, portfolioSummary } from '$lib/stores/portfolio';
  import GlassTile from '$lib/components/live-trading/GlassTile.svelte';
  import { TrendingUp, TrendingDown, Percent, BarChart2, Target, Activity } from 'lucide-svelte';

  onMount(async () => {
    await portfolioStore.fetchPerformance();
  });

  function formatPercent(value: number, decimals = 2): string {
    return `${value >= 0 ? '+' : ''}${value.toFixed(decimals)}%`;
  }

  function formatNumber(value: number, decimals = 2): string {
    return value.toFixed(decimals);
  }
</script>

<div class="performance-panel">
  <header class="panel-header">
    <h2>Performance Metrics</h2>
    <span class="subtitle">Portfolio-level performance indicators</span>
  </header>

  {#if $portfolioLoading}
    <div class="loading">Loading performance data...</div>
  {:else if !$performance}
    <div class="empty">No performance data available</div>
  {:else}
    <div class="metrics-grid">
      <!-- Return Card -->
      <GlassTile>
        <div class="metric-card">
          <div class="metric-icon">
            <TrendingUp size={20} />
          </div>
          <div class="metric-content">
            <span class="metric-label">Total Return</span>
            <span class="metric-value" class:positive={$performance.total_return > 0} class:negative={$performance.total_return < 0}>
              {formatPercent($performance.total_return)}
            </span>
          </div>
        </div>
      </GlassTile>

      <!-- Sharpe Ratio Card -->
      <GlassTile>
        <div class="metric-card">
          <div class="metric-icon">
            <BarChart2 size={20} />
          </div>
          <div class="metric-content">
            <span class="metric-label">Sharpe Ratio</span>
            <span class="metric-value">
              {formatNumber($performance.sharpe_ratio)}
            </span>
            <span class="metric-badge" class:good={$performance.sharpe_ratio >= 1.5} class:warning={$performance.sharpe_ratio >= 1 && $performance.sharpe_ratio < 1.5} class:poor={$performance.sharpe_ratio < 1}>
              {$performance.sharpe_ratio >= 1.5 ? 'Excellent' : $performance.sharpe_ratio >= 1 ? 'Good' : 'Needs Work'}
            </span>
          </div>
        </div>
      </GlassTile>

      <!-- Max Drawdown Card -->
      <GlassTile>
        <div class="metric-card">
          <div class="metric-icon negative">
            <TrendingDown size={20} />
          </div>
          <div class="metric-content">
            <span class="metric-label">Max Drawdown</span>
            <span class="metric-value negative">
              {formatPercent($performance.max_drawdown)}
            </span>
          </div>
        </div>
      </GlassTile>

      <!-- Win Rate Card -->
      <GlassTile>
        <div class="metric-card">
          <div class="metric-icon">
            <Target size={20} />
          </div>
          <div class="metric-content">
            <span class="metric-label">Win Rate</span>
            <span class="metric-value">
              {formatPercent($performance.win_rate, 1)}
            </span>
            <span class="metric-badge" class:good={$performance.win_rate >= 60} class:warning={$performance.win_rate >= 50 && $performance.win_rate < 60} class:poor={$performance.win_rate < 50}>
              {$performance.win_rate >= 60 ? 'Excellent' : $performance.win_rate >= 50 ? 'Average' : 'Low'}
            </span>
          </div>
        </div>
      </GlassTile>

      <!-- Profit Factor Card -->
      <GlassTile>
        <div class="metric-card">
          <div class="metric-icon">
            <Activity size={20} />
          </div>
          <div class="metric-content">
            <span class="metric-label">Profit Factor</span>
            <span class="metric-value">
              {formatNumber($performance.profit_factor)}
            </span>
            <span class="metric-badge" class:good={$performance.profit_factor >= 2} class:warning={$performance.profit_factor >= 1.5 && $performance.profit_factor < 2} class:poor={$performance.profit_factor < 1.5}>
              {$performance.profit_factor >= 2 ? 'Excellent' : $performance.profit_factor >= 1.5 ? 'Good' : 'Needs Work'}
            </span>
          </div>
        </div>
      </GlassTile>

      <!-- Average Trade Card -->
      <GlassTile>
        <div class="metric-card">
          <div class="metric-icon">
            <Percent size={20} />
          </div>
          <div class="metric-content">
            <span class="metric-label">Avg Trade P&L</span>
            <span class="metric-value" class:positive={$performance.avg_trade > 0} class:negative={$performance.avg_trade < 0}>
              ${formatNumber($performance.avg_trade)}
            </span>
          </div>
        </div>
      </GlassTile>
    </div>

    <!-- Trade Statistics -->
    <div class="trade-stats">
      <h3>Trade Statistics</h3>
      <div class="stats-grid">
        <div class="stat-item">
          <span class="stat-label">Total Trades</span>
          <span class="stat-value">{$performance.total_trades}</span>
        </div>
        <div class="stat-item">
          <span class="stat-label">Profitable</span>
          <span class="stat-value positive">{$performance.profitable_trades}</span>
        </div>
        <div class="stat-item">
          <span class="stat-label">Losing</span>
          <span class="stat-value negative">{$performance.losing_trades}</span>
        </div>
        <div class="stat-item">
          <span class="stat-label">Profit Ratio</span>
          <span class="stat-value">
            {($performance.profitable_trades / $performance.total_trades * 100).toFixed(1)}%
          </span>
        </div>
      </div>
    </div>
  {/if}
</div>

<style>
  .performance-panel {
    display: flex;
    flex-direction: column;
    gap: 24px;
    padding: 16px;
  }

  .panel-header {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .panel-header h2 {
    font-family: 'JetBrains Mono', monospace;
    font-size: 16px;
    font-weight: 600;
    color: #f59e0b;
    margin: 0;
  }

  .subtitle {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    color: rgba(255, 255, 255, 0.5);
  }

  .loading, .empty {
    font-family: 'JetBrains Mono', monospace;
    font-size: 14px;
    color: rgba(255, 255, 255, 0.4);
    text-align: center;
    padding: 40px;
  }

  .metrics-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 16px;
  }

  .metric-card {
    display: flex;
    gap: 16px;
    align-items: flex-start;
  }

  .metric-icon {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 40px;
    height: 40px;
    background: rgba(245, 158, 11, 0.1);
    border-radius: 8px;
    color: #f59e0b;
    flex-shrink: 0;
  }

  .metric-icon.negative {
    background: rgba(239, 68, 68, 0.1);
    color: #ef4444;
  }

  .metric-content {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .metric-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    color: rgba(255, 255, 255, 0.5);
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  .metric-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 24px;
    font-weight: 600;
    color: #fff;
  }

  .metric-value.positive {
    color: #10b981;
  }

  .metric-value.negative {
    color: #ef4444;
  }

  .metric-badge {
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    padding: 2px 8px;
    border-radius: 4px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  .metric-badge.good {
    background: rgba(16, 185, 129, 0.15);
    color: #10b981;
  }

  .metric-badge.warning {
    background: rgba(245, 158, 11, 0.15);
    color: #f59e0b;
  }

  .metric-badge.poor {
    background: rgba(239, 68, 68, 0.15);
    color: #ef4444;
  }

  .trade-stats {
    background: rgba(8, 13, 20, 0.4);
    border: 1px solid rgba(245, 158, 11, 0.1);
    border-radius: 8px;
    padding: 16px;
  }

  .trade-stats h3 {
    font-family: 'JetBrains Mono', monospace;
    font-size: 14px;
    font-weight: 600;
    color: rgba(245, 158, 11, 0.8);
    margin: 0 0 16px 0;
  }

  .stats-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
  }

  .stat-item {
    display: flex;
    flex-direction: column;
    gap: 4px;
    text-align: center;
  }

  .stat-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: rgba(255, 255, 255, 0.5);
    text-transform: uppercase;
  }

  .stat-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 18px;
    font-weight: 600;
    color: #fff;
  }

  .stat-value.positive {
    color: #10b981;
  }

  .stat-value.negative {
    color: #ef4444;
  }

  @media (max-width: 768px) {
    .stats-grid {
      grid-template-columns: repeat(2, 1fr);
    }
  }
</style>