<script lang="ts">
  /**
   * PaperTradingMonitorTile — shows active paper-trading EAs.
   * AC 12-4-2: calm empty state on empty/404
   * AC 12-4-3: ea_name, pair, days_running, pnl_current, win_rate as status dot
   * AC 12-4-8: errors handled as empty state
   * Story 12-4
   */
  import { onMount } from 'svelte';
  import TileCard from '$lib/components/shared/TileCard.svelte';
  import { apiFetch } from '$lib/api';

  interface ActiveAgentItem {
    ea_name: string;
    pair: string;
    days_running: number;
    win_rate: number;
    pnl_current: number;
    status: string;
    started_at: string;
  }

  interface ActiveAgentsResponse {
    items: ActiveAgentItem[];
  }

  interface Props {
    navigable?: boolean;
    onNavigate?: () => void;
  }

  let { navigable = false, onNavigate }: Props = $props();

  let items = $state<ActiveAgentItem[]>([]);
  let empty = $state(true);

  onMount(async () => {
    try {
      const data = await apiFetch<ActiveAgentsResponse>('/paper-trading/active');
      items = data.items ?? [];
      empty = items.length === 0;
    } catch {
      items = [];
      empty = true;
    }
  });

  function statusColor(status: string): string {
    if (status === 'running' || status === 'active') return 'var(--color-accent-cyan)';
    if (status === 'paused') return 'var(--color-accent-amber)';
    return 'var(--color-accent-red)';
  }

  function pnlSign(pnl: number): string {
    return pnl >= 0 ? '+' : '';
  }
</script>

<TileCard title="Paper Trading Monitor" size="lg" {navigable} {onNavigate}>
  {#if empty}
    <p class="empty-state">
      No EAs in paper monitoring phase — Alpha Forge feeds this when EAs reach the paper gate
    </p>
  {:else}
    <ul class="ea-list">
      {#each items.slice(0, 3) as item (item.ea_name)}
        <li class="ea-row">
          <span
            class="status-dot"
            style="background: {statusColor(item.status)};"
            title={item.status}
          ></span>
          <span class="ea-name">{item.ea_name}</span>
          <span class="pair section-label">{item.pair}</span>
          <span class="financial-value days">{item.days_running}d</span>
          <span class="financial-value win-rate">{item.win_rate.toFixed(1)}%</span>
          <span class="financial-value pnl" style="color: {item.pnl_current >= 0 ? 'var(--color-accent-cyan)' : 'var(--color-accent-red)'}">
            {pnlSign(item.pnl_current)}{item.pnl_current.toFixed(2)}
          </span>
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
    line-height: 1.5;
    margin: 0;
    padding: var(--space-2) 0;
  }

  .ea-list {
    list-style: none;
    margin: 0;
    padding: 0;
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
  }

  .ea-row {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    font-size: var(--text-xs);
  }

  .status-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    flex-shrink: 0;
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

  .pair {
    color: var(--color-text-muted);
    flex-shrink: 0;
  }

  .days {
    color: var(--color-text-muted);
    flex-shrink: 0;
    font-size: var(--text-xs);
  }

  .win-rate {
    color: var(--color-text-muted);
    flex-shrink: 0;
    font-size: var(--text-xs);
  }

  .pnl {
    flex-shrink: 0;
    font-size: var(--text-xs);
  }
</style>
