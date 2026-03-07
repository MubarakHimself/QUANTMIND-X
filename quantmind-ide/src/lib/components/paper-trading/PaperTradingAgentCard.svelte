<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import { Play, Square, FileText, Trash2 } from 'lucide-svelte';

  export let agent: {
    agent_id: string;
    container_id: string;
    container_name: string;
    status: string;
    strategy_name: string;
    symbol?: string;
    timeframe?: string;
    mt5_account?: number;
    mt5_server?: string;
    magic_number?: number;
    uptime_seconds?: number;
    created_at: string;
  };

  export let performanceMetrics: {
    agent_id: string;
    total_trades: number;
    winning_trades: number;
    losing_trades: number;
    win_rate: number;
    total_pnl: number;
    average_pnl: number;
    max_drawdown: number;
    profit_factor: number;
    sharpe_ratio: number | null;
    validation_status: string;
    days_validated: number;
    meets_criteria: boolean;
    validation_thresholds?: {
      min_sharpe_ratio: number;
      min_win_rate: number;
      min_validation_days: number;
    };
  } | undefined;

  export let tickData: {
    bid: number;
    ask: number;
    spread: number;
    timestamp: string;
  } | undefined;

  const dispatch = createEventDispatcher();

  function formatUptime(seconds?: number): string {
    if (!seconds) return 'N/A';
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    if (hours > 0) return `${hours}h ${minutes}m`;
    return `${minutes}m`;
  }

  function getStatusColor(status: string): string {
    const colors: Record<string, string> = {
      running: '#10b981',
      stopped: '#6b7280',
      error: '#ef4444',
      pending: '#f59e0b',
    };
    return colors[status] || '#6b7280';
  }

  function getStatusBadgeColor(status: string): string {
    const colors: Record<string, string> = {
      running: 'bg-emerald-500/20 text-emerald-400',
      stopped: 'bg-gray-500/20 text-gray-400',
      error: 'bg-red-500/20 text-red-400',
      pending: 'bg-amber-500/20 text-amber-400',
    };
    return colors[status] || 'bg-gray-500/20 text-gray-400';
  }

  function getValidationBadgeColor(status: string): string {
    const colors: Record<string, string> = {
      validated: 'bg-emerald-500/20 text-emerald-400',
      pending: 'bg-amber-500/20 text-amber-400',
      failed: 'bg-red-500/20 text-red-400',
    };
    return colors[status] || 'bg-gray-500/20 text-gray-400';
  }

  function getSharpeColor(sharpe: number | null): string {
    if (sharpe === null) return '#6b7280';
    if (sharpe >= 2) return '#10b981';
    if (sharpe >= 1) return '#f59e0b';
    return '#ef4444';
  }

  function getWinRateColor(winRate: number): string {
    if (winRate >= 60) return '#10b981';
    if (winRate >= 50) return '#f59e0b';
    return '#ef4444';
  }

  function canPromote(metrics: typeof performanceMetrics): boolean {
    if (!metrics) return false;
    return metrics.meets_criteria && metrics.validation_status === 'validated';
  }
</script>

<div class="agent-card">
  <!-- Agent Header -->
  <div class="agent-header">
    <div class="agent-name">{agent.strategy_name}</div>
    <div class="agent-status" style="--status-color: {getStatusColor(agent.status)}">
      <span class="status-dot"></span>
      <span class="status-text">{agent.status}</span>
    </div>
  </div>

  <!-- Agent Details -->
  <div class="agent-details">
    <div class="detail-row">
      <span class="label">Symbol:</span>
      <span class="value">{agent.symbol || 'N/A'}</span>
    </div>
    <div class="detail-row">
      <span class="label">Timeframe:</span>
      <span class="value">{agent.timeframe || 'N/A'}</span>
    </div>
    <div class="detail-row">
      <span class="label">Uptime:</span>
      <span class="value">{formatUptime(agent.uptime_seconds)}</span>
    </div>
    <div class="detail-row">
      <span class="label">Magic #:</span>
      <span class="value">{agent.magic_number || 'N/A'}</span>
    </div>
  </div>

  <!-- Performance Metrics -->
  {#if performanceMetrics}
    <div class="performance-section">
      <div class="performance-header">
        <span class="perf-title">Performance</span>
        <span class="validation-badge {getValidationBadgeColor(performanceMetrics.validation_status)}">
          {performanceMetrics.validation_status}
        </span>
      </div>
      <div class="metrics-grid">
        <div class="metric-item">
          <span class="metric-label">Win Rate</span>
          <span class="metric-value" style="color: {getWinRateColor(performanceMetrics.win_rate)}">
            {performanceMetrics.win_rate.toFixed(1)}%
          </span>
        </div>
        <div class="metric-item">
          <span class="metric-label">Total P&L</span>
          <span class="metric-value" class:positive={performanceMetrics.total_pnl > 0} class:negative={performanceMetrics.total_pnl < 0}>
            ${performanceMetrics.total_pnl.toFixed(2)}
          </span>
        </div>
        <div class="metric-item">
          <span class="metric-label">Profit Factor</span>
          <span class="metric-value">{performanceMetrics.profit_factor.toFixed(2)}</span>
        </div>
        <div class="metric-item">
          <span class="metric-label">Sharpe</span>
          <span class="metric-value" style="color: {getSharpeColor(performanceMetrics.sharpe_ratio)}">
            {performanceMetrics.sharpe_ratio?.toFixed(2) || 'N/A'}
          </span>
        </div>
      </div>
      <!-- Validation Progress -->
      {#if performanceMetrics.validation_thresholds}
        <div class="validation-progress">
          <div class="progress-label">
            <span>Validation Progress</span>
            <span>{performanceMetrics.days_validated}/{performanceMetrics.validation_thresholds.min_validation_days} days</span>
          </div>
          <div class="progress-bar">
            <div
              class="progress-fill"
              class:complete={performanceMetrics.meets_criteria}
              style="width: {(performanceMetrics.days_validated / performanceMetrics.validation_thresholds.min_validation_days) * 100}%"
            ></div>
          </div>
        </div>
      {/if}
    </div>
  {/if}

  <!-- Tick Data -->
  {#if tickData && agent.symbol}
    <div class="tick-data">
      <div class="tick-row">
        <span class="tick-label">{agent.symbol}</span>
        <span class="tick-value bid">Bid: {tickData.bid.toFixed(5)}</span>
        <span class="tick-value ask">Ask: {tickData.ask.toFixed(5)}</span>
        <span class="tick-value spread">Spread: {tickData.spread.toFixed(1)}</span>
      </div>
    </div>
  {/if}

  <!-- Actions -->
  <div class="agent-actions">
    {#if agent.status === 'running'}
      <button class="action-btn stop" on:click={() => dispatch('stop', agent.agent_id)}>
        <Square size={14} /> Stop
      </button>
    {:else}
      <button class="action-btn start" on:click={() => dispatch('start', agent.agent_id)}>
        <Play size={14} /> Start
      </button>
    {/if}
    <button class="action-btn" on:click={() => dispatch('logs', agent.agent_id)}>
      <FileText size={14} /> Logs
    </button>
    {#if canPromote(performanceMetrics)}
      <button class="action-btn promote" on:click={() => dispatch('promote', agent.agent_id)}>
        <Trash2 size={14} /> Promote
      </button>
    {/if}
  </div>
</div>

<style>
  .agent-card {
    background: #1e293b;
    border-radius: 12px;
    padding: 16px;
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  .agent-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .agent-name {
    font-weight: 600;
    font-size: 16px;
    color: #e2e8f0;
  }

  .agent-status {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 4px 10px;
    border-radius: 16px;
    background: rgba(255, 255, 255, 0.1);
  }

  .status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--status-color);
  }

  .status-text {
    font-size: 12px;
    text-transform: capitalize;
    color: #94a3b8;
  }

  .agent-details {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .detail-row {
    display: flex;
    justify-content: space-between;
    font-size: 13px;
  }

  .label {
    color: #64748b;
  }

  .value {
    color: #e2e8f0;
  }

  .performance-section {
    background: #0f172a;
    border-radius: 8px;
    padding: 12px;
  }

  .performance-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 10px;
  }

  .perf-title {
    font-weight: 600;
    font-size: 14px;
    color: #e2e8f0;
  }

  .validation-badge {
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 11px;
    text-transform: capitalize;
  }

  .metrics-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 10px;
  }

  .metric-item {
    display: flex;
    flex-direction: column;
    gap: 2px;
  }

  .metric-label {
    font-size: 11px;
    color: #64748b;
  }

  .metric-value {
    font-size: 14px;
    font-weight: 600;
    color: #e2e8f0;
  }

  .metric-value.positive {
    color: #10b981;
  }

  .metric-value.negative {
    color: #ef4444;
  }

  .validation-progress {
    margin-top: 10px;
  }

  .progress-label {
    display: flex;
    justify-content: space-between;
    font-size: 11px;
    color: #64748b;
    margin-bottom: 4px;
  }

  .progress-bar {
    height: 6px;
    background: #334155;
    border-radius: 3px;
    overflow: hidden;
  }

  .progress-fill {
    height: 100%;
    background: #3b82f6;
    border-radius: 3px;
    transition: width 0.3s ease;
  }

  .progress-fill.complete {
    background: #10b981;
  }

  .tick-data {
    background: #0f172a;
    border-radius: 8px;
    padding: 10px;
  }

  .tick-row {
    display: flex;
    gap: 12px;
    font-size: 12px;
    align-items: center;
  }

  .tick-label {
    font-weight: 600;
    color: #e2e8f0;
  }

  .tick-value {
    color: #94a3b8;
  }

  .tick-value.bid {
    color: #ef4444;
  }

  .tick-value.ask {
    color: #10b981;
  }

  .tick-value.spread {
    color: #f59e0b;
  }

  .agent-actions {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
  }

  .action-btn {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 6px 12px;
    border-radius: 6px;
    border: none;
    font-size: 12px;
    cursor: pointer;
    transition: all 0.2s;
    background: #334155;
    color: #e2e8f0;
  }

  .action-btn:hover {
    background: #475569;
  }

  .action-btn.start {
    background: #10b981;
    color: white;
  }

  .action-btn.start:hover {
    background: #059669;
  }

  .action-btn.stop {
    background: #ef4444;
    color: white;
  }

  .action-btn.stop:hover {
    background: #dc2626;
  }

  .action-btn.promote {
    background: #8b5cf6;
    color: white;
  }

  .action-btn.promote:hover {
    background: #7c3aed;
  }
</style>
