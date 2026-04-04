<script lang="ts">
  import { onMount } from 'svelte';
  import { Clock, CheckCircle, AlertCircle, RefreshCw, TrendingUp } from 'lucide-svelte';
  import { API_BASE } from '$lib/constants';

  const apiBase = API_BASE || '';

  // Buffer status data
  let poolStatus = $state({
    total_in_buffer: 0,
    total_eligible: 0,
    total_trades: 0,
    avg_lag_remaining: 0,
    trades: [] as Array<{
      trade_id: string;
      bot_id: string;
      close_date: string;
      eligible_date: string;
      outcome: string;
      pnl: number;
      holding_time_minutes: number;
      regime_at_entry: string;
      in_lag_buffer: boolean;
      lag_days_remaining: number;
      time_remaining: string;
      status: string;
    }>
  });

  let loading = $state(true);
  let error = $state<string | null>(null);
  let lastUpdate = $state<string | null>(null);

  async function fetchPoolStatus() {
    try {
      const baseUrl = apiBase || window.location.origin;
      const response = await fetch(`${baseUrl}/api/hmm/training-pool-status`);

      if (!response.ok) {
        throw new Error(`Failed to fetch: ${response.status}`);
      }

      const data = await response.json();
      poolStatus = data;
      lastUpdate = new Date().toLocaleTimeString();
      error = null;
    } catch (e) {
      console.error('Failed to fetch training pool status:', e);
      error = e instanceof Error ? e.message : 'Unknown error';
    } finally {
      loading = false;
    }
  }

  async function processMondayBatch() {
    try {
      const baseUrl = apiBase || window.location.origin;
      const response = await fetch(`${baseUrl}/api/hmm/process-monday-batch`, {
        method: 'POST'
      });

      if (!response.ok) {
        throw new Error(`Failed to process batch: ${response.status}`);
      }

      const result = await response.json();
      console.log('Monday batch processed:', result);
      await fetchPoolStatus();
    } catch (e) {
      console.error('Failed to process Monday batch:', e);
      error = e instanceof Error ? e.message : 'Unknown error';
    }
  }

  function getOutcomeColor(outcome: string): string {
    switch (outcome) {
      case 'WIN': return '#10b981';
      case 'LOSS': return '#ef4444';
      case 'HOLDING': return '#f59e0b';
      default: return '#6b7280';
    }
  }

  function getStatusBadgeClass(inBuffer: boolean): string {
    return inBuffer ? 'badge-buffer' : 'badge-eligible';
  }

  function getStatusIcon(inBuffer: boolean) {
    return inBuffer ? AlertCircle : CheckCircle;
  }

  onMount(() => {
    fetchPoolStatus();
    // Poll every 30 seconds
    const interval = setInterval(fetchPoolStatus, 30000);
    return () => clearInterval(interval);
  });
</script>

<div class="hmm-training-pool-panel">
  <div class="panel-header">
    <Clock size={18} />
    <h3>HMM Training Pool</h3>
    {#if lastUpdate}
      <span class="last-update">Updated: {lastUpdate}</span>
    {/if}
    <button class="refresh-btn" onclick={fetchPoolStatus} title="Refresh">
      <RefreshCw size={14} />
    </button>
  </div>

  {#if error}
    <div class="error-banner">
      <AlertCircle size={16} />
      <span>{error}</span>
    </div>
  {/if}

  {#if loading}
    <div class="loading-state">
      <div class="spinner"></div>
      <span>Loading training pool...</span>
    </div>
  {:else}
    <!-- Summary Stats -->
    <div class="stats-grid">
      <div class="stat-card buffer">
        <div class="stat-value">{poolStatus.total_in_buffer}</div>
        <div class="stat-label">In Buffer</div>
      </div>
      <div class="stat-card eligible">
        <div class="stat-value">{poolStatus.total_eligible}</div>
        <div class="stat-label">Eligible</div>
      </div>
      <div class="stat-card avg-lag">
        <div class="stat-value">{poolStatus.avg_lag_remaining.toFixed(1)}</div>
        <div class="stat-label">Avg Days Left</div>
      </div>
    </div>

    <!-- Trades List -->
    <div class="trades-section">
      <div class="section-header">
        <h4>Trade Records</h4>
        <button class="batch-btn" onclick={processMondayBatch}>
          Process Monday Batch
        </button>
      </div>

      {#if poolStatus.trades.length === 0}
        <div class="empty-state">
          <TrendingUp size={24} />
          <span>No trades in pool</span>
        </div>
      {:else}
        <div class="trades-table">
          <div class="table-header">
            <span class="col-trade-id">Trade ID</span>
            <span class="col-close-date">Close Date</span>
            <span class="col-eligible-date">Eligible Date</span>
            <span class="col-time-remaining">Time Left</span>
            <span class="col-outcome">Outcome</span>
            <span class="col-pnl">PnL</span>
            <span class="col-status">Status</span>
          </div>

          <div class="table-body">
            {#each poolStatus.trades as trade}
              <div class="table-row" class:row-in-buffer={trade.in_lag_buffer}>
                <span class="col-trade-id" title={trade.trade_id}>
                  {trade.trade_id.slice(0, 12)}...
                </span>
                <span class="col-close-date">
                  {new Date(trade.close_date).toLocaleDateString()}
                </span>
                <span class="col-eligible-date">
                  {new Date(trade.eligible_date).toLocaleDateString()}
                </span>
                <span class="col-time-remaining">
                  {trade.time_remaining}
                </span>
                <span class="col-outcome" style="color: {getOutcomeColor(trade.outcome)}">
                  {trade.outcome}
                </span>
                <span class="col-pnl" class:pnl-positive={trade.pnl > 0} class:pnl-negative={trade.pnl < 0}>
                  {trade.pnl >= 0 ? '+' : ''}{trade.pnl.toFixed(2)}
                </span>
                <span class="col-status">
                  <span class="status-badge {getStatusBadgeClass(trade.in_lag_buffer)}">
                    {#if trade.in_lag_buffer}
                      <AlertCircle size={12} />
                    {:else}
                      <CheckCircle size={12} />
                    {/if}
                    {trade.status}
                  </span>
                </span>
              </div>
            {/each}
          </div>
        </div>
      {/if}
    </div>
  {/if}
</div>

<style>
  .hmm-training-pool-panel {
    background: rgba(30, 41, 59, 0.8);
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 16px;
    backdrop-filter: blur(8px);
  }

  .panel-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 12px;
    color: #e2e8f0;
  }

  .panel-header h3 {
    margin: 0;
    font-size: 14px;
    font-weight: 600;
  }

  .last-update {
    margin-left: auto;
    font-size: 10px;
    color: #64748b;
  }

  .refresh-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 4px;
    background: transparent;
    border: 1px solid #475569;
    border-radius: 4px;
    color: #94a3b8;
    cursor: pointer;
    transition: all 0.2s;
  }

  .refresh-btn:hover {
    background: rgba(255, 255, 255, 0.05);
    color: #e2e8f0;
  }

  .error-banner {
    display: flex;
    align-items: center;
    gap: 8px;
    background: rgba(239, 68, 68, 0.1);
    border: 1px solid rgba(239, 68, 68, 0.3);
    border-radius: 6px;
    padding: 8px 12px;
    color: #fca5a5;
    font-size: 12px;
    margin-bottom: 12px;
  }

  .loading-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 8px;
    padding: 32px;
    color: #64748b;
  }

  .spinner {
    width: 24px;
    height: 24px;
    border: 2px solid #334155;
    border-top-color: #3b82f6;
    border-radius: 50%;
    animation: spin 1s linear infinite;
  }

  @keyframes spin {
    to { transform: rotate(360deg); }
  }

  .stats-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
    margin-bottom: 16px;
  }

  .stat-card {
    background: #0f172a;
    border-radius: 6px;
    padding: 12px;
    text-align: center;
    border-left: 3px solid;
  }

  .stat-card.buffer {
    border-color: #f59e0b;
  }

  .stat-card.eligible {
    border-color: #10b981;
  }

  .stat-card.avg-lag {
    border-color: #3b82f6;
  }

  .stat-value {
    font-size: 24px;
    font-weight: 700;
    color: #f1f5f9;
  }

  .stat-label {
    font-size: 10px;
    color: #64748b;
    text-transform: uppercase;
    margin-top: 4px;
  }

  .trades-section {
    background: #0f172a;
    border-radius: 6px;
    padding: 12px;
  }

  .section-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 12px;
  }

  .section-header h4 {
    margin: 0;
    font-size: 12px;
    color: #94a3b8;
    text-transform: uppercase;
  }

  .batch-btn {
    background: rgba(59, 130, 246, 0.2);
    border: 1px solid rgba(59, 130, 246, 0.4);
    border-radius: 4px;
    padding: 4px 8px;
    color: #60a5fa;
    font-size: 11px;
    cursor: pointer;
    transition: all 0.2s;
  }

  .batch-btn:hover {
    background: rgba(59, 130, 246, 0.3);
  }

  .empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 8px;
    padding: 24px;
    color: #475569;
    font-size: 12px;
  }

  .trades-table {
    overflow-x: auto;
  }

  .table-header {
    display: grid;
    grid-template-columns: 100px 90px 90px 70px 60px 70px 80px;
    gap: 8px;
    padding: 8px 0;
    border-bottom: 1px solid #1e293b;
    font-size: 10px;
    color: #64748b;
    text-transform: uppercase;
  }

  .table-body {
    max-height: 200px;
    overflow-y: auto;
  }

  .table-row {
    display: grid;
    grid-template-columns: 100px 90px 90px 70px 60px 70px 80px;
    gap: 8px;
    padding: 8px 0;
    border-bottom: 1px solid #1e293b;
    font-size: 11px;
    color: #cbd5e1;
    align-items: center;
  }

  .table-row:last-child {
    border-bottom: none;
  }

  .table-row.row-in-buffer {
    background: rgba(245, 158, 11, 0.05);
  }

  .col-trade-id {
    font-family: monospace;
    font-size: 10px;
  }

  .col-outcome {
    font-weight: 600;
  }

  .pnl-positive {
    color: #10b981;
  }

  .pnl-negative {
    color: #ef4444;
  }

  .status-badge {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 10px;
    font-weight: 500;
  }

  .badge-buffer {
    background: rgba(245, 158, 11, 0.2);
    color: #fbbf24;
  }

  .badge-eligible {
    background: rgba(16, 185, 129, 0.2);
    color: #34d399;
  }
</style>
