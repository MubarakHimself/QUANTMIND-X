<script lang="ts">
  /**
   * PortfolioSummary - Portfolio-wide Summary Display
   *
   * Shows: total equity, daily P&L, total drawdown
   * Uses highlighted GlassTile for emphasis
   */
  import GlassTile from '$lib/components/live-trading/GlassTile.svelte';
  import type { PortfolioSummary as PortfolioSummaryType } from '$lib/stores/portfolio';
  import { TrendingDown, TrendingUp, Wallet, PieChart } from 'lucide-svelte';

  interface Props {
    summary: PortfolioSummaryType | null;
  }

  let { summary }: Props = $props();

  function formatCurrency(value: number): string {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0
    }).format(value);
  }

  function formatPercent(value: number): string {
    return `${value.toFixed(1)}%`;
  }
</script>

<GlassTile>
  <div class="portfolio-summary">
    <div class="header">
      <PieChart size={16} />
      <span class="title">Portfolio Summary</span>
    </div>

    {#if summary}
      <div class="metrics">
        <div class="metric total-equity">
          <div class="metric-header">
            <Wallet size={14} />
            <span class="label">Total Equity</span>
          </div>
          <span class="value">{formatCurrency(summary.totalEquity)}</span>
        </div>

        <div class="metric daily-pnl" class:positive={summary.dailyPnL >= 0} class:negative={summary.dailyPnL < 0}>
          <div class="metric-header">
            {#if summary.dailyPnL >= 0}
              <TrendingUp size={14} />
            {:else}
              <TrendingDown size={14} />
            {/if}
            <span class="label">Daily P&L</span>
          </div>
          <span class="value">{summary.dailyPnL >= 0 ? '+' : ''}{formatCurrency(summary.dailyPnL)}</span>
        </div>

        <div class="metric total-drawdown" class:warning={summary.drawdownPercent > 5}>
          <div class="metric-header">
            <TrendingDown size={14} />
            <span class="label">Total Drawdown</span>
          </div>
          <div class="drawdown-values">
            <span class="value">{formatCurrency(summary.totalDrawdown)}</span>
            <span class="percent">({formatPercent(summary.drawdownPercent)})</span>
          </div>
        </div>
      </div>
    {:else}
      <div class="loading">Loading portfolio data...</div>
    {/if}
  </div>
</GlassTile>

<style>
  .portfolio-summary {
    display: flex;
    flex-direction: column;
    gap: 16px;
    min-width: 240px;
  }

  .header {
    display: flex;
    align-items: center;
    gap: 8px;
    color: #f59e0b;
  }

  .title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
  }

  .metrics {
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  .metric {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .metric-header {
    display: flex;
    align-items: center;
    gap: 6px;
  }

  .metric-header .label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: #666;
    text-transform: uppercase;
  }

  .metric .value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 16px;
    font-weight: 600;
    color: #e0e0e0;
  }

  .metric.total-equity .value {
    color: #00d4ff;
  }

  .metric.daily-pnl.positive .value {
    color: #00c896;
  }

  .metric.daily-pnl.negative .value {
    color: #ff3b3b;
  }

  .metric.total-drawdown.warning .value {
    color: #ff3b3b;
  }

  .metric.total-drawdown.warning .percent {
    color: #ff3b3b;
  }

  .drawdown-values {
    display: flex;
    align-items: baseline;
    gap: 4px;
  }

  .drawdown-values .percent {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    color: #888;
  }

  .loading {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    color: #666;
    text-align: center;
    padding: 20px;
  }
</style>