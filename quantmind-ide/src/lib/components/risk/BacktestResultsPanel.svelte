<script lang="ts">
  /**
   * BacktestResultsPanel - Backtest Results Viewer
   *
   * Displays backtest results with equity curve, drawdown charts,
   * and 6-mode result matrix.
   * AC #4: Shows equity curve and drawdown charts with 6-mode result matrix.
   */

  import { onMount, onDestroy } from 'svelte';
  import { BarChart3, TrendingUp, TrendingDown, Clock, CheckCircle, XCircle, PlayCircle } from 'lucide-svelte';
  import EquityCurveChart from './EquityCurveChart.svelte';
  import DrawdownChart from './DrawdownChart.svelte';
  import {
    backtestStore,
    backtestList,
    selectedBacktest,
    backtestLoading,
    backtestError,
    type BacktestSummary,
    type BacktestDetail
  } from '$lib/stores';

  let list: BacktestSummary[] = $state([]);
  let selected: BacktestDetail | null = $state(null);
  let loading = $state(false);
  let error: string | null = $state(null);

  // Subscribe to store
  const unsubList = backtestList.subscribe(v => list = v);
  const unsubSelected = selectedBacktest.subscribe(v => selected = v);
  const unsubLoading = backtestLoading.subscribe(v => loading = v);
  const unsubError = backtestError.subscribe(v => error = v);

  onMount(() => {
    backtestStore.fetchList();
  });

  onDestroy(() => {
    unsubList();
    unsubSelected();
    unsubLoading();
    unsubError();
  });

  async function selectBacktest(id: string) {
    await backtestStore.fetchDetail(id);
  }

  function getModeLabel(mode: string): string {
    const labels: Record<string, string> = {
      'VANILLA': 'Vanilla',
      'SPICED': 'Spiced',
      'VANILLA_FULL': 'Vanilla Full',
      'SPICED_FULL': 'Spiced Full',
      'MODE_B': 'Mode B',
      'MODE_C': 'Mode C'
    };
    return labels[mode] || mode;
  }

  function getModeColor(mode: string): string {
    const colors: Record<string, string> = {
      'VANILLA': '#00d4ff',
      'SPICED': '#7c3aed',
      'VANILLA_FULL': '#0891b2',
      'SPICED_FULL': '#6d28d9',
      'MODE_B': '#10b981',
      'MODE_C': '#f59e0b'
    };
    return colors[mode] || '#888';
  }

  function formatDate(isoString: string): string {
    return new Date(isoString).toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  }

  // 6-mode matrix data
  const modeMatrix = [
    { mode: 'VANILLA', label: 'Vanilla', description: 'Basic strategy' },
    { mode: 'SPICED', label: 'Spiced', description: 'With filters' },
    { mode: 'VANILLA_FULL', label: 'Vanilla Full', description: 'Walk-forward' },
    { mode: 'SPICED_FULL', label: 'Spiced Full', description: 'Full optimization' },
    { mode: 'MODE_B', label: 'Mode B', description: 'Variant B' },
    { mode: 'MODE_C', label: 'Mode C', description: 'Variant C' }
  ];
</script>

<div class="backtest-panel">
  <div class="panel-header">
    <h3 class="panel-title">
      <BarChart3 size={16} />
      Backtest Results
    </h3>
  </div>

  <div class="panel-content">
    <div class="backtest-list">
      <div class="list-header">
        <span>Available Backtests</span>
        <span class="count">{list.length}</span>
      </div>
      {#if loading && list.length === 0}
        <div class="loading">Loading...</div>
      {:else if list.length === 0}
        <div class="empty">No backtests available</div>
      {:else}
        {#each list as bt}
          <button
            class="backtest-item"
            class:selected={selected?.id === bt.id}
            onclick={() => selectBacktest(bt.id)}
          >
            <div class="bt-header">
              <span class="bt-name">{bt.ea_name}</span>
              <span class="bt-mode" style="color: {getModeColor(bt.mode)}">
                {getModeLabel(bt.mode)}
              </span>
            </div>
            <div class="bt-metrics">
              <span class="pnl" class:positive={bt.net_pnl >= 0} class:negative={bt.net_pnl < 0}>
                {bt.net_pnl >= 0 ? '+' : ''}{bt.net_pnl.toFixed(1)}%
              </span>
              <span class="sharpe">Sharpe: {bt.sharpe.toFixed(2)}</span>
            </div>
            <div class="bt-date">
              <Clock size={10} />
              {formatDate(bt.run_at_utc)}
            </div>
          </button>
        {/each}
      {/if}
    </div>

    {#if selected}
      <div class="backtest-detail">
        <div class="detail-header">
          <h4>{selected.ea_name}</h4>
          <span class="mode-badge" style="background: {getModeColor(selected.mode)}20; color: {getModeColor(selected.mode)}; border-color: {getModeColor(selected.mode)}40">
            {getModeLabel(selected.mode)}
          </span>
        </div>

        <!-- Metrics Summary -->
        <div class="metrics-grid">
          <div class="metric">
            <span class="metric-label">Net P&L</span>
            <span class="metric-value" class:positive={selected.net_pnl >= 0} class:negative={selected.net_pnl < 0}>
              {selected.net_pnl >= 0 ? '+' : ''}{selected.net_pnl.toFixed(1)}%
            </span>
          </div>
          <div class="metric">
            <span class="metric-label">Sharpe</span>
            <span class="metric-value">{selected.sharpe.toFixed(2)}</span>
          </div>
          <div class="metric">
            <span class="metric-label">Max DD</span>
            <span class="metric-value negative">-{selected.max_drawdown.toFixed(1)}%</span>
          </div>
          <div class="metric">
            <span class="metric-label">Win Rate</span>
            <span class="metric-value">{selected.win_rate.toFixed(1)}%</span>
          </div>
        </div>

        <!-- Charts -->
        {#if selected.equity_curve && selected.equity_curve.length > 0}
          <div class="chart-section">
            <h5 class="section-title">
              <TrendingUp size={14} />
              Equity Curve
            </h5>
            <EquityCurveChart data={selected.equity_curve} height={150} />
          </div>
        {/if}

        {#if selected.equity_curve && selected.equity_curve.length > 0}
          <div class="chart-section">
            <h5 class="section-title">
              <TrendingDown size={14} />
              Drawdown
            </h5>
            <DrawdownChart data={selected.equity_curve} height={80} />
          </div>
        {/if}

        <!-- 6-Mode Matrix -->
        <div class="mode-matrix">
          <h5 class="section-title">
            <BarChart3 size={14} />
            6-Mode Results
          </h5>
          <div class="matrix-grid">
            {#each modeMatrix as mode}
              {@const isMatch = selected.mode === mode.mode}
              <div class="matrix-cell" class:current={isMatch}>
                <span class="mode-name" style="color: {getModeColor(mode.mode)}">{mode.label}</span>
                <span class="mode-desc">{mode.description}</span>
                {#if isMatch}
                  <CheckCircle size={12} class="check-icon" />
                {/if}
              </div>
            {/each}
          </div>
        </div>

        <!-- Additional Stats -->
        <div class="extra-stats">
          <div class="stat">
            <span class="stat-label">Total Trades</span>
            <span class="stat-value">{selected.total_trades}</span>
          </div>
          <div class="stat">
            <span class="stat-label">Profit Factor</span>
            <span class="stat-value">{selected.profit_factor.toFixed(2)}</span>
          </div>
          <div class="stat">
            <span class="stat-label">Avg Trade</span>
            <span class="stat-value">{selected.avg_trade_pnl.toFixed(2)}%</span>
          </div>
        </div>
      </div>
    {:else}
      <div class="no-selection">
        <BarChart3 size={32} />
        <span>Select a backtest to view details</span>
      </div>
    {/if}
  </div>
</div>

<style>
  .backtest-panel {
    background: rgba(8, 13, 20, 0.35);
    backdrop-filter: blur(16px) saturate(120%);
    -webkit-backdrop-filter: blur(16px) saturate(120%);
    border: 1px solid rgba(0, 212, 255, 0.08);
    border-radius: 8px;
    padding: 16px;
    display: flex;
    flex-direction: column;
    gap: 12px;
    min-height: 300px;
  }

  .panel-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
  }

  .panel-title {
    font-size: 14px;
    font-weight: 600;
    color: rgba(255, 255, 255, 0.9);
    margin: 0;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    letter-spacing: 0.5px;
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .panel-content {
    display: flex;
    gap: 16px;
    flex: 1;
    overflow: hidden;
  }

  .backtest-list {
    width: 200px;
    display: flex;
    flex-direction: column;
    gap: 8px;
    overflow-y: auto;
  }

  .list-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 11px;
    color: rgba(255, 255, 255, 0.5);
    text-transform: uppercase;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }

  .count {
    background: rgba(0, 212, 255, 0.2);
    padding: 2px 6px;
    border-radius: 3px;
    font-size: 10px;
  }

  .loading, .empty {
    padding: 20px;
    text-align: center;
    color: rgba(255, 255, 255, 0.4);
    font-size: 12px;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }

  .backtest-item {
    display: flex;
    flex-direction: column;
    gap: 4px;
    padding: 10px;
    background: rgba(0, 0, 0, 0.2);
    border: 1px solid transparent;
    border-radius: 6px;
    cursor: pointer;
    text-align: left;
    transition: all 0.2s;
  }

  .backtest-item:hover {
    background: rgba(0, 212, 255, 0.1);
  }

  .backtest-item.selected {
    background: rgba(0, 212, 255, 0.15);
    border-color: rgba(0, 212, 255, 0.3);
  }

  .bt-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .bt-name {
    font-size: 12px;
    font-weight: 600;
    color: rgba(255, 255, 255, 0.9);
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }

  .bt-mode {
    font-size: 9px;
    font-weight: 600;
    text-transform: uppercase;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }

  .bt-metrics {
    display: flex;
    justify-content: space-between;
    font-size: 11px;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }

  .pnl {
    font-weight: 600;
  }

  .pnl.positive { color: #10b981; }
  .pnl.negative { color: #ff3b3b; }

  .sharpe {
    color: rgba(255, 255, 255, 0.5);
  }

  .bt-date {
    display: flex;
    align-items: center;
    gap: 4px;
    font-size: 10px;
    color: rgba(255, 255, 255, 0.4);
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }

  .backtest-detail {
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 12px;
    overflow-y: auto;
  }

  .detail-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .detail-header h4 {
    margin: 0;
    font-size: 14px;
    color: rgba(255, 255, 255, 0.9);
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }

  .mode-badge {
    padding: 3px 8px;
    border-radius: 4px;
    font-size: 10px;
    font-weight: 600;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    border: 1px solid;
  }

  .metrics-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 8px;
  }

  .metric {
    display: flex;
    flex-direction: column;
    gap: 2px;
    padding: 8px;
    background: rgba(0, 0, 0, 0.2);
    border-radius: 4px;
  }

  .metric-label {
    font-size: 9px;
    color: rgba(255, 255, 255, 0.5);
    text-transform: uppercase;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }

  .metric-value {
    font-size: 14px;
    font-weight: 600;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    color: rgba(255, 255, 255, 0.9);
  }

  .metric-value.positive { color: #10b981; }
  .metric-value.negative { color: #ff3b3b; }

  .chart-section {
    display: flex;
    flex-direction: column;
    gap: 6px;
  }

  .section-title {
    font-size: 11px;
    font-weight: 600;
    color: rgba(255, 255, 255, 0.6);
    margin: 0;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    display: flex;
    align-items: center;
    gap: 6px;
  }

  .mode-matrix {
    display: flex;
    flex-direction: column;
    gap: 6px;
  }

  .matrix-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 6px;
  }

  .matrix-cell {
    display: flex;
    flex-direction: column;
    gap: 2px;
    padding: 8px;
    background: rgba(0, 0, 0, 0.2);
    border-radius: 4px;
    position: relative;
    border: 1px solid transparent;
  }

  .matrix-cell.current {
    border-color: rgba(0, 212, 255, 0.3);
    background: rgba(0, 212, 255, 0.1);
  }

  .mode-name {
    font-size: 11px;
    font-weight: 600;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }

  .mode-desc {
    font-size: 9px;
    color: rgba(255, 255, 255, 0.4);
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }

  .matrix-cell :global(.check-icon) {
    position: absolute;
    top: 4px;
    right: 4px;
    color: #10b981;
  }

  .extra-stats {
    display: flex;
    gap: 12px;
    padding-top: 8px;
    border-top: 1px solid rgba(0, 212, 255, 0.1);
  }

  .stat {
    display: flex;
    flex-direction: column;
    gap: 2px;
  }

  .stat-label {
    font-size: 9px;
    color: rgba(255, 255, 255, 0.4);
    text-transform: uppercase;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }

  .stat-value {
    font-size: 12px;
    font-weight: 600;
    color: rgba(255, 255, 255, 0.8);
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }

  .no-selection {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 12px;
    color: rgba(255, 255, 255, 0.3);
    font-size: 12px;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }
</style>
