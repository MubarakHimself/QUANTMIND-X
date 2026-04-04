<script lang="ts">
  import { onMount } from 'svelte';
  import TileCard from '$lib/components/shared/TileCard.svelte';
  import { apiFetch } from '$lib/api';

  interface JournalTrade {
    id: string;
    entryTime: string;
    symbol: string;
    direction: string;
    pnl: number;
    session: string;
    eaName: string;
    status: string;
  }

  let trades = $state<JournalTrade[]>([]);
  let loading = $state(true);
  let error = $state<string | null>(null);

  onMount(async () => {
    try {
      trades = await apiFetch<JournalTrade[]>('/api/journal/trades?limit=25');
    } catch (e) {
      error = e instanceof Error ? e.message : 'Failed to load trading journal';
    } finally {
      loading = false;
    }
  });

  function formatPnL(value: number): string {
    const sign = value >= 0 ? '+' : '';
    return `${sign}${value.toFixed(2)}`;
  }

  function formatTime(value: string): string {
    if (!value) return 'No timestamp';
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return value;
    return date.toLocaleString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  }

  const closedTrades = $derived(trades.filter((trade) => trade.status === 'closed'));
  const totalPnl = $derived(closedTrades.reduce((sum, trade) => sum + (trade.pnl ?? 0), 0));
  const winningTrades = $derived(closedTrades.filter((trade) => (trade.pnl ?? 0) > 0).length);
  const winRate = $derived(closedTrades.length > 0 ? (winningTrades / closedTrades.length) * 100 : 0);
  const latestTrade = $derived(trades[0] ?? null);
</script>

<TileCard title="Trading Journal" size="lg">
  {#if loading}
    <p class="state-copy">Loading journal activity…</p>
  {:else if error}
    <p class="state-copy error">{error}</p>
  {:else if trades.length === 0}
    <div class="empty-state">
      <p class="state-copy">No journal entries recorded yet.</p>
      <p class="support-copy">Paper and live trade logs will appear here as the runtime writes to the journal.</p>
    </div>
  {:else}
    <div class="metrics">
      <div class="metric">
        <span class="section-label">Closed</span>
        <span class="financial-value metric-value">{closedTrades.length}</span>
      </div>
      <div class="metric">
        <span class="section-label">Win Rate</span>
        <span class="financial-value metric-value">{winRate.toFixed(1)}%</span>
      </div>
      <div class="metric">
        <span class="section-label">Net P&amp;L</span>
        <span class="financial-value metric-value" class:negative={totalPnl < 0}>{formatPnL(totalPnl)}</span>
      </div>
      <div class="metric">
        <span class="section-label">Latest</span>
        <span class="metric-value">{latestTrade?.symbol ?? 'Unknown'} · {latestTrade?.session ?? 'n/a'}</span>
      </div>
    </div>

    {#if latestTrade}
      <div class="latest-trade">
        <span class="section-label">Recent Entry</span>
        <div class="latest-row">
          <span class="trade-meta">{latestTrade.eaName || 'Unknown EA'}</span>
          <span class="trade-meta">{formatTime(latestTrade.entryTime)}</span>
        </div>
      </div>
    {/if}
  {/if}
</TileCard>

<style>
  .metrics {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: var(--space-3);
  }

  .metric {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .metric-value {
    font-family: var(--font-data);
    font-size: var(--text-sm);
    color: var(--color-text-primary);
  }

  .metric-value.negative {
    color: var(--color-accent-red);
  }

  .latest-trade {
    margin-top: var(--space-3);
    padding-top: var(--space-3);
    border-top: 1px solid var(--color-border-subtle);
    display: flex;
    flex-direction: column;
    gap: 6px;
  }

  .latest-row {
    display: flex;
    justify-content: space-between;
    gap: var(--space-3);
  }

  .trade-meta {
    font-family: var(--font-data);
    font-size: var(--text-xs);
    color: var(--color-text-secondary);
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .state-copy,
  .support-copy {
    margin: 0;
    font-family: var(--font-ambient);
    font-size: var(--text-xs);
    line-height: 1.5;
    color: var(--color-text-muted);
  }

  .state-copy.error {
    color: var(--color-accent-red);
  }

  .empty-state {
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
  }
</style>
