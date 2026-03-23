<script lang="ts">
  /**
   * AttributionPanel - Strategy P&L Attribution Table
   *
   * Story 9.4: Portfolio Canvas — Attribution, Correlation Matrix & Performance
   * AC #1: Shows equity contribution, P&L contribution, drawdown contribution,
   *        % of portfolio, broker account per strategy
   */
  import { onMount } from 'svelte';
  import { portfolioStore, attribution, portfolioLoading } from '$lib/stores/portfolio';
  import { ArrowUpDown } from 'lucide-svelte';

  type SortField = 'strategy_name' | 'equity_contribution' | 'pnl_contribution' | 'drawdown_contribution' | 'portfolio_percent' | 'broker_name';
  type SortDirection = 'asc' | 'desc';

  let sortField = $state<SortField>('portfolio_percent');
  let sortDirection = $state<SortDirection>('desc');

  onMount(async () => {
    await portfolioStore.fetchAttribution();
  });

  function sortBy(field: SortField) {
    if (sortField === field) {
      sortDirection = sortDirection === 'asc' ? 'desc' : 'asc';
    } else {
      sortField = field;
      sortDirection = 'desc';
    }
  }

  let sortedAttribution = $derived([...$attribution].sort((a, b) => {
    const aVal = a[sortField];
    const bVal = b[sortField];
    const multiplier = sortDirection === 'asc' ? 1 : -1;

    if (typeof aVal === 'string' && typeof bVal === 'string') {
      return aVal.localeCompare(bVal) * multiplier;
    }
    return ((aVal as number) - (bVal as number)) * multiplier;
  }));

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

<div class="attribution-panel">
  <header class="panel-header">
    <h2>Strategy Attribution</h2>
    <span class="subtitle">P&L and risk contribution by strategy</span>
  </header>

  {#if $portfolioLoading}
    <div class="loading">Loading attribution data...</div>
  {:else if sortedAttribution.length === 0}
    <div class="empty">No attribution data available</div>
  {:else}
    <div class="table-container">
      <table class="attribution-table">
        <thead>
          <tr>
            <th class="sortable" onclick={() => sortBy('strategy_name')}>
              Strategy
              <ArrowUpDown size={12} />
            </th>
            <th class="sortable" onclick={() => sortBy('equity_contribution')}>
              Equity Contribution
              <ArrowUpDown size={12} />
            </th>
            <th class="sortable" onclick={() => sortBy('pnl_contribution')}>
              P&L Contribution
              <ArrowUpDown size={12} />
            </th>
            <th class="sortable" onclick={() => sortBy('drawdown_contribution')}>
              Drawdown
              <ArrowUpDown size={12} />
            </th>
            <th class="sortable" onclick={() => sortBy('portfolio_percent')}>
              % of Portfolio
              <ArrowUpDown size={12} />
            </th>
            <th class="sortable" onclick={() => sortBy('broker_name')}>
              Broker Account
              <ArrowUpDown size={12} />
            </th>
          </tr>
        </thead>
        <tbody>
          {#each sortedAttribution as item}
            <tr>
              <td class="strategy-name">{item.strategy_name}</td>
              <td class="value">{formatCurrency(item.equity_contribution)}</td>
              <td class="value" class:positive={item.pnl_contribution > 0} class:negative={item.pnl_contribution < 0}>
                {formatCurrency(item.pnl_contribution)}
              </td>
              <td class="value negative">{formatCurrency(item.drawdown_contribution)}</td>
              <td class="value">{formatPercent(item.portfolio_percent)}</td>
              <td class="broker">{item.broker_name}</td>
            </tr>
          {/each}
        </tbody>
      </table>
    </div>
  {/if}
</div>

<style>
  .attribution-panel {
    display: flex;
    flex-direction: column;
    gap: 16px;
    padding: 16px;
  }

  .panel-header {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .panel-header h2 {
    font-family: 'JetBrains Mono', monospace;
    font-size: 16px;
    font-weight: 600;
    color: #f59e0b;
    margin: 0;
  }

  .subtitle {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    color: rgba(255, 255, 255, 0.5);
  }

  .loading, .empty {
    font-family: 'JetBrains Mono', monospace;
    font-size: 14px;
    color: rgba(255, 255, 255, 0.4);
    text-align: center;
    padding: 40px;
  }

  .table-container {
    overflow-x: auto;
  }

  .attribution-table {
    width: 100%;
    border-collapse: collapse;
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
  }

  .attribution-table th {
    text-align: left;
    padding: 12px 16px;
    background: rgba(245, 158, 11, 0.08);
    border-bottom: 1px solid rgba(245, 158, 11, 0.2);
    color: rgba(245, 158, 11, 0.8);
    font-weight: 600;
    white-space: nowrap;
  }

  .attribution-table th.sortable {
    cursor: pointer;
    display: flex;
    align-items: center;
    gap: 6px;
  }

  .attribution-table th.sortable:hover {
    color: #f59e0b;
  }

  .attribution-table td {
    padding: 12px 16px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    color: rgba(255, 255, 255, 0.8);
  }

  .attribution-table tr:hover td {
    background: rgba(245, 158, 11, 0.05);
  }

  .strategy-name {
    font-weight: 500;
    color: #fff;
  }

  .value {
    text-align: right;
    font-variant-numeric: tabular-nums;
  }

  .positive {
    color: #10b981;
  }

  .negative {
    color: #ef4444;
  }

  .broker {
    color: rgba(0, 212, 255, 0.8);
  }
</style>