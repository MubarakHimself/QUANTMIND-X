<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { fade, slide } from 'svelte/transition';
  import {
    WebSocketClient,
    createBacktestClient
  } from '../ws-client';
  import type { WebSocketMessage } from '../ws-client';

  // Props
  export let baseUrl: string = 'http://localhost:8000';

  // Types
  interface BacktestMetrics {
    total_trades: number;
    win_count: number;
    loss_count: number;
    win_rate: number;
    profit_factor: number;
    sharpe_ratio: number;
    max_drawdown: number;
    net_profit: number;
    gross_profit: number;
    gross_loss: number;
    recovery_factor: number;
    average_risk_reward: number;
    consecutive_losses: number;
    expectancy: number;
    avg_win: number;
    avg_loss: number;
  }

  interface Trade {
    id: string;
    symbol: string;
    type: 'buy' | 'sell';
    entry_price: number;
    exit_price: number;
    entry_time: string;
    exit_time: string;
    profit: number;
    profit_pips: number;
  }

  interface BacktestResult {
    backtest_id: string;
    status: 'pending' | 'running' | 'completed' | 'failed';
    metrics: BacktestMetrics | null;
    trades: Trade[];
    equity_curve: number[];
    started_at: string;
    completed_at: string | null;
    config: {
      symbol: string;
      timeframe: string;
      initial_deposit: number;
      start_date: string;
      end_date: string;
    };
  }

  // State
  let result: BacktestResult | null = null;
  let loading = true;
  let error: string | null = null;
  let activeTab = 'overview';

  // WebSocket streaming state
  let wsClient: WebSocketClient | null = null;
  let wsConnected = false;
  let currentBacktestId: string | null = null;
  let isRunning = false;
  let progress = 0;
  let status = '';
  let liveLogs: Array<{ timestamp: string; level: string; message: string }> = [];


  // Fetch backtest result
  async function fetchResult(backtestId: string) {
    try {
      const response = await fetch(`/api/backtest/${backtestId}`);
      if (!response.ok) throw new Error('Failed to fetch backtest result');
      result = await response.json();
    } catch (e) {
      error = 'Failed to load backtest result';
      console.error('Fetch error:', e);
    } finally {
      loading = false;
    }
  }

  // Format currency
  function formatCurrency(value: number): string {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD'
    }).format(value);
  }

  // Format percentage
  function formatPercent(value: number): string {
    return `${(value * 100).toFixed(2)}%`;
  }

  // Format number
  function formatNumber(value: number, decimals: number = 2): string {
    return value.toFixed(decimals);
  }

  // Get metric status
  function getMetricStatus(metric: string, value: number): 'good' | 'warning' | 'bad' {
    const thresholds: Record<string, { good: [number, number]; warning: [number, number] }> = {
      win_rate: { good: [0.5, 1], warning: [0.4, 0.5] },
      sharpe_ratio: { good: [1.5, Infinity], warning: [1.0, 1.5] },
      max_drawdown: { good: [0, 0.15], warning: [0.15, 0.25] },
      profit_factor: { good: [1.5, Infinity], warning: [1.2, 1.5] }
    };

    const threshold = thresholds[metric];
    if (!threshold) return 'good';

    if (value >= threshold.good[0] && value <= threshold.good[1]) return 'good';
    if (value >= threshold.warning[0] && value <= threshold.warning[1]) return 'warning';
    return 'bad';
  }

  // Export results
  function exportResults() {
    if (!result) return;

    const data = JSON.stringify(result, null, 2);
    const blob = new Blob([data], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `backtest_${result.backtest_id}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }

  // WebSocket handlers for real-time streaming
  function handleBacktestStart(message: WebSocketMessage) {
    const payload = message.data as Record<string, unknown> | undefined;
    const data = (payload || message) as Record<string, unknown>;
    if (currentBacktestId && data.backtest_id !== currentBacktestId) return;

    isRunning = true;
    progress = 0;
    status = `Starting ${(data.variant as string) || 'backtest'} for ${(data.symbol as string) || ''}`;
    liveLogs = [];
    error = null;

    addLiveLog('INFO', `Backtest started: ${data.variant} on ${data.symbol} ${data.timeframe}`);
  }

  function handleBacktestProgress(message: WebSocketMessage) {
    const payload = message.data as Record<string, unknown> | undefined;
    const data = (payload || message) as Record<string, unknown>;
    if (currentBacktestId && data.backtest_id !== currentBacktestId) return;

    progress = (data.progress as number) || 0;
    status = (data.status as string) || '';
    addLiveLog('PROGRESS', `[${(data.progress as number)?.toFixed(1) || '0'}%] ${data.status}`);
  }

  function handleBacktestComplete(message: WebSocketMessage) {
    const payload = message.data as Record<string, unknown> | undefined;
    const data = (payload || message) as Record<string, unknown>;
    if (currentBacktestId && data.backtest_id !== currentBacktestId) return;

    isRunning = false;
    progress = 100;
    status = 'Backtest completed';

    // Update result with final data
    result = {
      backtest_id: data.backtest_id as string,
      status: 'completed',
      metrics: {
        total_trades: (data.total_trades as number) || 0,
        win_count: 0,
        loss_count: 0,
        win_rate: ((data.win_rate as number) || 0) / 100,
        profit_factor: 0,
        sharpe_ratio: (data.sharpe_ratio as number) || 0,
        max_drawdown: (data.drawdown as number) || 0,
        net_profit: ((data.final_balance as number) || 0) - 10000,
        gross_profit: 0,
        gross_loss: 0,
        recovery_factor: 0,
        average_risk_reward: 0,
        consecutive_losses: 0,
        expectancy: 0,
        avg_win: 0,
        avg_loss: 0
      },
      trades: [],
      equity_curve: [],
      started_at: new Date().toISOString(),
      completed_at: new Date().toISOString(),
      config: {
        symbol: '',
        timeframe: '',
        initial_deposit: 10000,
        start_date: '',
        end_date: ''
      }
    };

    addLiveLog('COMPLETE', `Backtest complete! Final balance: ${(data.final_balance as number)?.toFixed(2)}, Trades: ${data.total_trades}`);
    loading = false;
  }

  function handleBacktestError(message: WebSocketMessage) {
    const payload = message.data as Record<string, unknown> | undefined;
    const data = (payload || message) as Record<string, unknown>;
    if (currentBacktestId && data.backtest_id !== currentBacktestId) return;

    isRunning = false;
    error = (data.error as string) || 'Unknown error';
    addLiveLog('ERROR', `Backtest error: ${data.error}`);
  }

  function handleLogEntry(message: WebSocketMessage) {
    const payload = message.data as Record<string, unknown> | undefined;
    const data = (payload || message) as Record<string, unknown>;
    if (currentBacktestId && data.backtest_id !== currentBacktestId) return;

    addLiveLog((data.level as string) || 'INFO', (data.message as string) || '');
  }

  function addLiveLog(level: string, message: string) {
    const logEntry = {
      timestamp: new Date().toISOString(),
      level,
      message
    };
    liveLogs = [...liveLogs, logEntry];
  }

  async function connectWebSocket() {
    try {
      wsClient = await createBacktestClient(baseUrl);
      wsConnected = true;

      // Register event handlers
      wsClient.on('backtest_start', handleBacktestStart);
      wsClient.on('backtest_progress', handleBacktestProgress);
      wsClient.on('backtest_complete', handleBacktestComplete);
      wsClient.on('backtest_error', handleBacktestError);
      wsClient.on('log_entry', handleLogEntry);

    } catch (e) {
      console.error('Failed to connect WebSocket:', e);
      wsConnected = false;
      addLiveLog('ERROR', `Failed to connect to WebSocket: ${e}`);
    }
  }

  function disconnectWebSocket() {
    if (wsClient) {
      wsClient.off('backtest_start', handleBacktestStart);
      wsClient.off('backtest_progress', handleBacktestProgress);
      wsClient.off('backtest_complete', handleBacktestComplete);
      wsClient.off('backtest_error', handleBacktestError);
      wsClient.off('log_entry', handleLogEntry);
      wsClient.disconnect();
      wsClient = null;
      wsConnected = false;
    }
  }

  // Lifecycle
  onMount(async () => {
    // Connect WebSocket for real-time streaming
    await connectWebSocket();
    loading = false;
  });

  onDestroy(() => {
    disconnectWebSocket();
  });
</script>

<div class="backtest-dashboard">
  <!-- Header -->
  <div class="dashboard-header">
    <div class="header-left">
      <h3>Backtest Results</h3>
      {#if result}
        <span class="backtest-id">{result.backtest_id}</span>
      {/if}
    </div>
    <div class="header-right">
      <div class="connection-status" class:connected={wsConnected}>
        {wsConnected ? 'Connected' : 'Disconnected'}
      </div>
      <button class="export-btn" on:click={exportResults} disabled={!result}>
        Export
      </button>
    </div>
  </div>

  <!-- Progress Section (for running backtests) -->
  {#if isRunning || progress > 0}
    <div class="progress-section">
      <div class="progress-bar-container">
        <div class="progress-bar" style="width: {progress}%"></div>
      </div>
      <div class="progress-text">
        {progress.toFixed(1)}% - {status}
      </div>
    </div>
  {/if}

  <!-- Live Logs Section -->
  {#if liveLogs.length > 0}
    <div class="live-logs-section">
      <div class="live-logs-header">
        <h4>Live Logs</h4>
        <span class="log-count">{liveLogs.length} entries</span>
      </div>
      <div class="live-logs-container">
        {#each liveLogs as log}
          <div class="log-entry">
            <span class="log-timestamp">{new Date(log.timestamp).toLocaleTimeString()}</span>
            <span class="log-level" class:info={log.level === 'INFO'} class:warning={log.level === 'WARNING' || log.level === 'WARN'} class:error={log.level === 'ERROR'} class:progress-log={log.level === 'PROGRESS'} class:complete={log.level === 'COMPLETE'}>
              {log.level}
            </span>
            <span class="log-message">{log.message}</span>
          </div>
        {/each}
      </div>
    </div>
  {/if}

  <!-- Loading State -->
  {#if loading}
    <div class="loading-state" in:fade>
      <div class="spinner"></div>
      <p>Loading backtest results...</p>
    </div>
  {:else if result}
    <!-- Tabs -->
    <div class="tabs">
      <button 
        class="tab" 
        class:active={activeTab === 'overview'}
        on:click={() => activeTab = 'overview'}
      >
        Overview
      </button>
      <button 
        class="tab" 
        class:active={activeTab === 'trades'}
        on:click={() => activeTab = 'trades'}
      >
        Trades
      </button>
      <button 
        class="tab" 
        class:active={activeTab === 'equity'}
        on:click={() => activeTab = 'equity'}
      >
        Equity Curve
      </button>
    </div>

    <!-- Overview Tab -->
    {#if activeTab === 'overview'}
      <div class="overview-content" in:slide>
        <!-- Key Metrics -->
        <div class="metrics-grid">
          <div class="metric-card">
            <span class="metric-label">Net Profit</span>
            <span class="metric-value profit">
              {formatCurrency(result.metrics?.net_profit || 0)}
            </span>
          </div>
          <div class="metric-card">
            <span class="metric-label">Total Trades</span>
            <span class="metric-value">{result.metrics?.total_trades || 0}</span>
          </div>
          <div class="metric-card">
            <span class="metric-label">Win Rate</span>
            <span 
              class="metric-value"
              class:good={getMetricStatus('win_rate', result.metrics?.win_rate || 0) === 'good'}
              class:warning={getMetricStatus('win_rate', result.metrics?.win_rate || 0) === 'warning'}
              class:bad={getMetricStatus('win_rate', result.metrics?.win_rate || 0) === 'bad'}
            >
              {formatPercent(result.metrics?.win_rate || 0)}
            </span>
          </div>
          <div class="metric-card">
            <span class="metric-label">Sharpe Ratio</span>
            <span 
              class="metric-value"
              class:good={getMetricStatus('sharpe_ratio', result.metrics?.sharpe_ratio || 0) === 'good'}
              class:warning={getMetricStatus('sharpe_ratio', result.metrics?.sharpe_ratio || 0) === 'warning'}
              class:bad={getMetricStatus('sharpe_ratio', result.metrics?.sharpe_ratio || 0) === 'bad'}
            >
              {formatNumber(result.metrics?.sharpe_ratio || 0)}
            </span>
          </div>
          <div class="metric-card">
            <span class="metric-label">Max Drawdown</span>
            <span 
              class="metric-value"
              class:good={getMetricStatus('max_drawdown', result.metrics?.max_drawdown || 0) === 'good'}
              class:warning={getMetricStatus('max_drawdown', result.metrics?.max_drawdown || 0) === 'warning'}
              class:bad={getMetricStatus('max_drawdown', result.metrics?.max_drawdown || 0) === 'bad'}
            >
              {formatPercent(result.metrics?.max_drawdown || 0)}
            </span>
          </div>
          <div class="metric-card">
            <span class="metric-label">Profit Factor</span>
            <span 
              class="metric-value"
              class:good={getMetricStatus('profit_factor', result.metrics?.profit_factor || 0) === 'good'}
              class:warning={getMetricStatus('profit_factor', result.metrics?.profit_factor || 0) === 'warning'}
              class:bad={getMetricStatus('profit_factor', result.metrics?.profit_factor || 0) === 'bad'}
            >
              {formatNumber(result.metrics?.profit_factor || 0)}
            </span>
          </div>
        </div>

        <!-- Detailed Metrics -->
        <div class="detailed-metrics">
          <h4>Detailed Metrics</h4>
          <div class="metrics-table">
            <div class="table-row">
              <span class="table-label">Winning Trades</span>
              <span class="table-value">{result.metrics?.win_count || 0}</span>
            </div>
            <div class="table-row">
              <span class="table-label">Losing Trades</span>
              <span class="table-value">{result.metrics?.loss_count || 0}</span>
            </div>
            <div class="table-row">
              <span class="table-label">Gross Profit</span>
              <span class="table-value profit">{formatCurrency(result.metrics?.gross_profit || 0)}</span>
            </div>
            <div class="table-row">
              <span class="table-label">Gross Loss</span>
              <span class="table-value loss">{formatCurrency(result.metrics?.gross_loss || 0)}</span>
            </div>
            <div class="table-row">
              <span class="table-label">Average Win</span>
              <span class="table-value profit">{formatCurrency(result.metrics?.avg_win || 0)}</span>
            </div>
            <div class="table-row">
              <span class="table-label">Average Loss</span>
              <span class="table-value loss">{formatCurrency(result.metrics?.avg_loss || 0)}</span>
            </div>
            <div class="table-row">
              <span class="table-label">Recovery Factor</span>
              <span class="table-value">{formatNumber(result.metrics?.recovery_factor || 0)}</span>
            </div>
            <div class="table-row">
              <span class="table-label">Avg Risk/Reward</span>
              <span class="table-value">{formatNumber(result.metrics?.average_risk_reward || 0)}</span>
            </div>
            <div class="table-row">
              <span class="table-label">Max Consecutive Losses</span>
              <span class="table-value">{result.metrics?.consecutive_losses || 0}</span>
            </div>
            <div class="table-row">
              <span class="table-label">Expectancy</span>
              <span class="table-value">{formatCurrency(result.metrics?.expectancy || 0)}</span>
            </div>
          </div>
        </div>

        <!-- Configuration -->
        <div class="config-section">
          <h4>Configuration</h4>
          <div class="config-grid">
            <div class="config-item">
              <span class="config-label">Symbol</span>
              <span class="config-value">{result.config.symbol}</span>
            </div>
            <div class="config-item">
              <span class="config-label">Timeframe</span>
              <span class="config-value">{result.config.timeframe}</span>
            </div>
            <div class="config-item">
              <span class="config-label">Initial Deposit</span>
              <span class="config-value">{formatCurrency(result.config.initial_deposit)}</span>
            </div>
            <div class="config-item">
              <span class="config-label">Period</span>
              <span class="config-value">{result.config.start_date} to {result.config.end_date}</span>
            </div>
          </div>
        </div>
      </div>
    {/if}

    <!-- Trades Tab -->
    {#if activeTab === 'trades'}
      <div class="trades-content" in:slide>
        <div class="trades-table">
          <div class="table-header">
            <span>ID</span>
            <span>Symbol</span>
            <span>Type</span>
            <span>Entry</span>
            <span>Exit</span>
            <span>Profit</span>
          </div>
          <div class="table-body">
            {#each result.trades as trade}
              <div class="trade-row">
                <span>{trade.id}</span>
                <span>{trade.symbol}</span>
                <span class:type-buy={trade.type === 'buy'} class:type-sell={trade.type === 'sell'}>
                  {trade.type.toUpperCase()}
                </span>
                <span>{formatNumber(trade.entry_price, 5)}</span>
                <span>{formatNumber(trade.exit_price, 5)}</span>
                <span class:profit={trade.profit > 0} class:loss={trade.profit < 0}>
                  {formatCurrency(trade.profit)}
                </span>
              </div>
            {/each}
          </div>
        </div>
      </div>
    {/if}

    <!-- Equity Curve Tab -->
    {#if activeTab === 'equity'}
      <div class="equity-content" in:slide>
        <div class="chart-placeholder">
          <div class="equity-chart">
            {#each result.equity_curve as value, i}
              <div 
                class="chart-bar"
                style="height: {(value / 15000) * 100}%; background: {value > 10000 ? '#a6e3a1' : '#f38ba8'}"
                title={formatCurrency(value)}
              ></div>
            {/each}
          </div>
          <div class="chart-labels">
            <span>Start: {formatCurrency(result.equity_curve[0])}</span>
            <span>End: {formatCurrency(result.equity_curve[result.equity_curve.length - 1])}</span>
            <span>Peak: {formatCurrency(Math.max(...result.equity_curve))}</span>
            <span>Low: {formatCurrency(Math.min(...result.equity_curve))}</span>
          </div>
        </div>
      </div>
    {/if}
  {/if}
</div>

<style>
  .backtest-dashboard {
    background: var(--bg-secondary, #1e1e2e);
    border-radius: 8px;
    padding: 16px;
    color: var(--text-primary, #cdd6f4);
  }

  .dashboard-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 16px;
    padding-bottom: 12px;
    border-bottom: 1px solid var(--border-color, #313244);
  }

  .header-left {
    display: flex;
    align-items: center;
    gap: 12px;
  }

  .header-right {
    display: flex;
    align-items: center;
    gap: 12px;
  }

  .dashboard-header h3 {
    margin: 0;
    font-size: 1.125rem;
  }

  .backtest-id {
    font-size: 0.75rem;
    color: var(--text-secondary, #a6adc8);
    font-family: monospace;
  }

  .connection-status {
    font-size: 0.75rem;
    padding: 4px 8px;
    border-radius: 4px;
    background: var(--bg-tertiary, #313244);
    color: var(--text-secondary, #a6adc8);
  }

  .connection-status.connected {
    background: #065f46;
    color: #10b981;
  }

  .export-btn {
    background: var(--accent, #89b4fa);
    color: var(--bg-primary, #1e1e2e);
    border: none;
    padding: 6px 12px;
    border-radius: 6px;
    cursor: pointer;
    font-size: 0.875rem;
  }

  .export-btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .progress-section {
    margin-bottom: 16px;
    padding: 12px;
    background: var(--bg-tertiary, #313244);
    border-radius: 8px;
  }

  .progress-bar-container {
    height: 8px;
    background: var(--bg-secondary, #1e1e2e);
    border-radius: 4px;
    overflow: hidden;
    margin-bottom: 8px;
  }

  .progress-bar {
    height: 100%;
    background: linear-gradient(90deg, #10b981, #89b4fa);
    transition: width 0.3s ease;
  }

  .progress-text {
    font-size: 0.875rem;
    color: var(--text-secondary, #a6adc8);
    text-align: center;
  }

  .live-logs-section {
    margin-bottom: 16px;
    background: var(--bg-tertiary, #313244);
    border-radius: 8px;
    overflow: hidden;
  }

  .live-logs-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 12px;
    background: var(--bg-secondary, #1e1e2e);
    border-bottom: 1px solid var(--border-color, #45475a);
  }

  .live-logs-header h4 {
    margin: 0;
    font-size: 0.875rem;
    font-weight: 500;
  }

  .log-count {
    font-size: 0.75rem;
    color: var(--text-secondary, #a6adc8);
  }

  .live-logs-container {
    max-height: 200px;
    overflow-y: auto;
    padding: 8px;
    font-family: Monaco, Menlo, Ubuntu Mono, monospace;
    font-size: 0.75rem;
  }

  .log-entry {
    display: flex;
    gap: 8px;
    padding: 4px 0;
    border-bottom: 1px solid var(--border-color, #45475a);
  }

  .log-entry:last-child {
    border-bottom: none;
  }

  .log-timestamp {
    color: var(--text-secondary, #a6adc8);
    white-space: nowrap;
  }

  .log-level {
    font-weight: 600;
    width: 70px;
    flex-shrink: 0;
  }

  .log-level.info { color: #89b4fa; }
  .log-level.warning { color: #f9e2af; }
  .log-level.error { color: #f38ba8; }
  .log-level.progress-log { color: #10b981; }
  .log-level.complete { color: #a6e3a1; }

  .log-message {
    color: var(--text-primary, #cdd6f4);
    word-break: break-word;
  }

  .loading-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 48px;
    color: var(--text-secondary, #a6adc8);
  }

  .spinner {
    width: 32px;
    height: 32px;
    border: 3px solid var(--border-color, #45475a);
    border-top-color: var(--accent, #89b4fa);
    border-radius: 50%;
    animation: spin 1s linear infinite;
    margin-bottom: 12px;
  }

  @keyframes spin {
    to { transform: rotate(360deg); }
  }

  .tabs {
    display: flex;
    gap: 4px;
    margin-bottom: 16px;
  }

  .tab {
    background: transparent;
    border: none;
    padding: 8px 16px;
    color: var(--text-secondary, #a6adc8);
    cursor: pointer;
    border-radius: 6px;
    font-size: 0.875rem;
  }

  .tab:hover {
    background: var(--bg-tertiary, #313244);
  }

  .tab.active {
    background: var(--accent, #89b4fa);
    color: var(--bg-primary, #1e1e2e);
  }

  .metrics-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
    margin-bottom: 24px;
  }

  .metric-card {
    background: var(--bg-tertiary, #313244);
    border-radius: 8px;
    padding: 16px;
    text-align: center;
  }

  .metric-label {
    display: block;
    font-size: 0.75rem;
    color: var(--text-secondary, #a6adc8);
    margin-bottom: 8px;
  }

  .metric-value {
    font-size: 1.5rem;
    font-weight: 600;
  }

  .metric-value.profit {
    color: #a6e3a1;
  }

  .metric-value.good {
    color: #a6e3a1;
  }

  .metric-value.warning {
    color: #f9e2af;
  }

  .metric-value.bad {
    color: #f38ba8;
  }

  .detailed-metrics, .config-section {
    margin-bottom: 24px;
  }

  .detailed-metrics h4, .config-section h4 {
    margin: 0 0 12px;
    font-size: 0.875rem;
    color: var(--text-secondary, #a6adc8);
  }

  .metrics-table {
    background: var(--bg-tertiary, #313244);
    border-radius: 8px;
    padding: 12px;
  }

  .table-row {
    display: flex;
    justify-content: space-between;
    padding: 8px 0;
    border-bottom: 1px solid var(--border-color, #45475a);
    font-size: 0.875rem;
  }

  .table-row:last-child {
    border-bottom: none;
  }

  .table-label {
    color: var(--text-secondary, #a6adc8);
  }

  .table-value.profit {
    color: #a6e3a1;
  }

  .table-value.loss {
    color: #f38ba8;
  }

  .config-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 12px;
  }

  .config-item {
    background: var(--bg-tertiary, #313244);
    border-radius: 6px;
    padding: 12px;
  }

  .config-label {
    display: block;
    font-size: 0.75rem;
    color: var(--text-secondary, #a6adc8);
    margin-bottom: 4px;
  }

  .config-value {
    font-size: 0.875rem;
  }

  .trades-table {
    background: var(--bg-tertiary, #313244);
    border-radius: 8px;
    overflow: hidden;
  }

  .table-header {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr 1.5fr 1.5fr 1fr;
    gap: 8px;
    padding: 12px;
    background: var(--bg-secondary, #1e1e2e);
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--text-secondary, #a6adc8);
  }

  .table-body {
    max-height: 400px;
    overflow-y: auto;
  }

  .trade-row {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr 1.5fr 1.5fr 1fr;
    gap: 8px;
    padding: 12px;
    font-size: 0.875rem;
    border-bottom: 1px solid var(--border-color, #45475a);
  }

  .type-buy {
    color: #a6e3a1;
  }

  .type-sell {
    color: #f38ba8;
  }

  .chart-placeholder {
    background: var(--bg-tertiary, #313244);
    border-radius: 8px;
    padding: 16px;
  }

  .equity-chart {
    display: flex;
    align-items: flex-end;
    height: 200px;
    gap: 2px;
  }

  .chart-bar {
    flex: 1;
    min-width: 2px;
    border-radius: 1px;
    transition: height 0.2s ease;
  }

  .chart-labels {
    display: flex;
    justify-content: space-between;
    margin-top: 16px;
    font-size: 0.75rem;
    color: var(--text-secondary, #a6adc8);
  }
</style>
