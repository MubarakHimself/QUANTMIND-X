<script lang="ts">
  /**
   * BacktestResultsTile — shows up to 5 recent backtest runs.
   * AC 12-4-4: ea_name, pass/fail, sharpe, local-timezone date
   * AC 12-4-8: errors handled as empty state
   * Story 12-4
   */
  import { onMount } from 'svelte';
  import TileCard from '$lib/components/shared/TileCard.svelte';
  import { apiFetch } from '$lib/api';
  import { CheckCircle, XCircle } from 'lucide-svelte';

  interface BacktestSummary {
    id: string;
    ea_name: string;
    mode: string;
    run_at_utc: string;
    net_pnl: number;
    sharpe: number;
    max_drawdown: number;
    win_rate: number;
  }

  interface Props {
    navigable?: boolean;
    onNavigate?: () => void;
  }

  let { navigable = false, onNavigate }: Props = $props();

  let results = $state<BacktestSummary[]>([]);
  let empty = $state(false);

  onMount(async () => {
    try {
      const data = await apiFetch<BacktestSummary[]>('/backtests?limit=5');
      results = Array.isArray(data) ? data.slice(0, 5) : [];
      empty = results.length === 0;
    } catch {
      results = [];
      empty = true;
    }
  });

  function isPassing(win_rate: number): boolean {
    return win_rate >= 50;
  }

  function formatDate(utcStr: string): string {
    try {
      return new Date(utcStr).toLocaleDateString(undefined, {
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
      });
    } catch {
      return utcStr;
    }
  }
</script>

<TileCard title="Backtest Results" size="md" {navigable} {onNavigate}>
  {#if empty}
    <p class="empty-state">No backtest results yet</p>
  {:else}
    <ul class="result-list">
      {#each results as r (r.id)}
        <li class="result-row">
          <span
            class="pass-indicator"
            style="color: {isPassing(r.win_rate) ? 'var(--color-accent-cyan)' : 'var(--color-accent-red)'};"
            title={isPassing(r.win_rate) ? 'Pass' : 'Fail'}
          >
            {#if isPassing(r.win_rate)}
              <CheckCircle size={12} />
            {:else}
              <XCircle size={12} />
            {/if}
          </span>
          <span class="ea-name">{r.ea_name}</span>
          <span class="financial-value sharpe" title="Sharpe">{r.sharpe.toFixed(2)}</span>
          <span class="timestamp date">{formatDate(r.run_at_utc)}</span>
        </li>
      {/each}
    </ul>
  {/if}
</TileCard>

<style>
  .empty-state {
    font-family: var(--font-ambient);
    font-size: var(--text-xs);
    color: var(--color-text-muted);
    margin: 0;
    padding: var(--space-2) 0;
  }

  .result-list {
    list-style: none;
    margin: 0;
    padding: 0;
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
  }

  .result-row {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    font-size: var(--text-xs);
  }

  .pass-indicator {
    font-family: var(--font-data);
    font-size: var(--text-xs);
    flex-shrink: 0;
    width: 12px;
    text-align: center;
  }

  .ea-name {
    font-family: var(--font-data);
    color: var(--color-text-primary);
    flex: 1;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    font-size: var(--text-xs);
  }

  .sharpe {
    color: var(--color-text-primary);
    flex-shrink: 0;
    font-size: var(--text-xs);
  }

  .date {
    color: var(--color-text-muted);
    flex-shrink: 0;
    font-size: var(--text-xs);
  }
</style>
