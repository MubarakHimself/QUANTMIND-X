<script lang="ts">
  interface Props {
    rankings: {
    daily: Array<{
      botId: string;
      name: string;
      profit: number;
      trades: number;
      winRate: number;
    }>;
    weekly: Array<{
      botId: string;
      name: string;
      profit: number;
      trades: number;
      winRate: number;
    }>;
  };
  }

  let { rankings }: Props = $props();

  function formatCurrency(value: number) {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2
    }).format(value);
  }
</script>

<div class="rankings-section">
  <div class="rankings-tabs">
    <button class="rank-tab active">Daily</button>
    <button class="rank-tab">Weekly</button>
    <button class="rank-tab">Monthly</button>
  </div>

  <div class="rankings-table">
    <div class="table-header">
      <span>Rank</span>
      <span>Strategy</span>
      <span>Profit</span>
      <span>Trades</span>
      <span>Win Rate</span>
    </div>

    {#each rankings.daily as ranking, index}
      <div class="table-row">
        <span class="rank">#{index + 1}</span>
        <span class="name">{ranking.name}</span>
        <span class="profit" class:positive={ranking.profit > 0}>
          {formatCurrency(ranking.profit)}
        </span>
        <span class="trades">{ranking.trades}</span>
        <span class="winrate">{ranking.winRate.toFixed(1)}%</span>
      </div>
    {/each}
  </div>
</div>

<style>
  .rankings-section {
    padding: 0;
  }

  .rankings-tabs {
    display: flex;
    gap: 4px;
    margin-bottom: 16px;
  }

  .rank-tab {
    padding: 8px 16px;
    background: var(--bg-tertiary);
    border: 1px solid var(--border-subtle);
    border-radius: 6px;
    color: var(--text-secondary);
    font-size: 12px;
    cursor: pointer;
    transition: all 0.15s;
  }

  .rank-tab:hover {
    background: var(--bg-surface);
  }

  .rank-tab.active {
    background: var(--accent-primary);
    border-color: var(--accent-primary);
    color: var(--bg-primary);
  }

  .rankings-table {
    background: var(--bg-secondary);
    border: 1px solid var(--border-subtle);
    border-radius: 10px;
    overflow: hidden;
  }

  .table-header {
    display: grid;
    grid-template-columns: 60px 1fr 100px 80px 80px;
    gap: 12px;
    padding: 12px 16px;
    background: var(--bg-tertiary);
    font-size: 11px;
    font-weight: 500;
    color: var(--text-muted);
    text-transform: uppercase;
  }

  .table-row {
    display: grid;
    grid-template-columns: 60px 1fr 100px 80px 80px;
    gap: 12px;
    padding: 12px 16px;
    border-bottom: 1px solid var(--border-subtle);
    font-size: 13px;
    color: var(--text-primary);
    align-items: center;
  }

  .table-row:last-child {
    border-bottom: none;
  }

  .rank {
    font-weight: 600;
    color: var(--accent-primary);
  }

  .name {
    font-weight: 500;
  }

  .profit {
    font-family: 'JetBrains Mono', monospace;
    font-weight: 500;
  }

  .profit.positive {
    color: #10b981;
  }

  .trades {
    color: var(--text-secondary);
  }

  .winrate {
    color: var(--text-secondary);
  }
</style>
