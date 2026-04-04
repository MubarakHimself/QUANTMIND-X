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
  let activePeriod: 'daily' | 'weekly' = $state('daily');

  function formatCurrency(value: number) {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2
    }).format(value);
  }

  let activeRankings = $derived(rankings[activePeriod] ?? []);
</script>

<div class="rankings-section">
  <div class="rankings-tabs">
    <button class="rank-tab" class:active={activePeriod === 'daily'} onclick={() => activePeriod = 'daily'}>Daily</button>
    <button class="rank-tab" class:active={activePeriod === 'weekly'} onclick={() => activePeriod = 'weekly'}>Weekly</button>
  </div>

  <div class="rankings-table">
    <div class="table-header">
      <span>Rank</span>
      <span>Strategy</span>
      <span>Profit</span>
      <span>Trades</span>
      <span>Win Rate</span>
    </div>

    {#if activeRankings.length > 0}
      {#each activeRankings as ranking, index}
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
    {:else}
      <div class="empty-state">No live {activePeriod} rankings available.</div>
    {/if}
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
    background: var(--color-bg-elevated);
    border: 1px solid var(--color-border-subtle);
    border-radius: 6px;
    color: var(--color-text-secondary);
    font-size: 12px;
    cursor: pointer;
    transition: all 0.15s;
  }

  .rank-tab:hover {
    background: var(--bg-surface);
  }

  .rank-tab.active {
    background: var(--color-accent-cyan);
    border-color: var(--color-accent-cyan);
    color: var(--color-bg-base);
  }

  .empty-state {
    padding: 20px 16px;
    color: var(--color-text-muted);
    font-size: 12px;
  }

  .rankings-table {
    background: var(--color-bg-surface);
    border: 1px solid var(--color-border-subtle);
    border-radius: 10px;
    overflow: hidden;
  }

  .table-header {
    display: grid;
    grid-template-columns: 60px 1fr 100px 80px 80px;
    gap: 12px;
    padding: 12px 16px;
    background: var(--color-bg-elevated);
    font-size: 11px;
    font-weight: 500;
    color: var(--color-text-muted);
    text-transform: uppercase;
  }

  .table-row {
    display: grid;
    grid-template-columns: 60px 1fr 100px 80px 80px;
    gap: 12px;
    padding: 12px 16px;
    border-bottom: 1px solid var(--color-border-subtle);
    font-size: 13px;
    color: var(--color-text-primary);
    align-items: center;
  }

  .table-row:last-child {
    border-bottom: none;
  }

  .rank {
    font-weight: 600;
    color: var(--color-accent-cyan);
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
    color: var(--color-text-secondary);
  }

  .winrate {
    color: var(--color-text-secondary);
  }
</style>
