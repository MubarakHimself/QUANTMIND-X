<script lang="ts">
  /**
   * RoutingMatrix - Strategy-to-Account Routing Configuration
   *
   * Shows matrix of strategies × broker accounts with assignment toggles
   * Includes regime and strategy-type filter dropdowns
   * Story 9.3 - AC #2
   */
  import GlassTile from '$lib/components/live-trading/GlassTile.svelte';
  import {
    portfolioStore,
    accounts,
    routingRules,
    strategies,
    portfolioLoading
  } from '$lib/stores/portfolio';
  import { X, Check, Filter, RefreshCw } from 'lucide-svelte';

  interface Props {
    onClose: () => void;
  }

  let { onClose }: Props = $props();

  // Filter state
  let selectedRegime = $state('all');
  let selectedStrategyType = $state('all');

  const regimeOptions = [
    { value: 'all', label: 'All Regimes' },
    { value: 'LONDON', label: 'London' },
    { value: 'NEW_YORK', label: 'New York' },
    { value: 'ASIAN', label: 'Asian' },
    { value: 'OVERLAP', label: 'Overlap' },
    { value: 'CLOSED', label: 'Closed' }
  ];

  const strategyTypeOptions = [
    { value: 'all', label: 'All Types' },
    { value: 'SCALPER', label: 'Scalper' },
    { value: 'HFT', label: 'HFT' },
    { value: 'STRUCTURAL', label: 'Structural' },
    { value: 'SWING', label: 'Swing' }
  ];

  // Filtered rules based on dropdown selection
  let filteredRules = $derived(
    routingRules.filter(rule => {
      const regimeMatch = selectedRegime === 'all' || rule.regime_filter === selectedRegime;
      const typeMatch = selectedStrategyType === 'all' || rule.strategy_type_filter === selectedStrategyType;
      return regimeMatch && typeMatch;
    })
  );

  // Get unique accounts for matrix columns
  let accountList = $derived(
    accounts.map(acc => ({
      id: acc.account_id,
      name: acc.broker_name,
      type: acc.account_type
    }))
  );

  function formatAccountType(type: string): string {
    const labels: Record<string, string> = {
      'MACHINE_GUN': 'HFT',
      'SNIPER': 'ICT',
      'PROP_FIRM': 'Prop',
      'CRYPTO': 'Crypto',
      'DEMO': 'Demo'
    };
    return labels[type] || type;
  }

  async function handleToggleRule(strategyId: string, currentEnabled: boolean) {
    await portfolioStore.toggleRoutingRule(strategyId, !currentEnabled);
  }

  function getRuleForStrategyAccount(strategyId: string, accountId: string) {
    return routingRules.find(
      rule => rule.strategy_id === strategyId && rule.account_id === accountId
    );
  }
</script>

<div class="routing-matrix-page">
  <header class="page-header">
    <div class="header-title">
      <h1>Routing Matrix</h1>
      <span class="subtitle">Strategy-to-Account Assignment</span>
    </div>
    <button class="close-btn" onclick={onClose} title="Close">
      <X size={18} />
    </button>
  </header>

  <div class="filters">
    <div class="filter-group">
      <Filter size={14} />
      <select bind:value={selectedRegime} class="filter-select">
        {#each regimeOptions as option}
          <option value={option.value}>{option.label}</option>
        {/each}
      </select>
    </div>

    <div class="filter-group">
      <Filter size={14} />
      <select bind:value={selectedStrategyType} class="filter-select">
        {#each strategyTypeOptions as option}
          <option value={option.value}>{option.label}</option>
        {/each}
      </select>
    </div>
  </div>

  <div class="matrix-container">
    {#if $portfolioLoading}
      <div class="loading">
        <RefreshCw size={20} class="spin" />
        <span>Loading routing data...</span>
      </div>
    {:else}
      <div class="matrix-table">
        <!-- Header Row -->
        <div class="matrix-row header">
          <div class="matrix-cell strategy-col">Strategy</div>
          {#each accountList as account}
            <div class="matrix-cell account-col">
              <span class="account-name">{account.name}</span>
              <span class="account-type">{formatAccountType(account.type)}</span>
            </div>
          {/each}
        </div>

        <!-- Strategy Rows -->
        {#each filteredRules as rule}
          <div class="matrix-row">
            <div class="matrix-cell strategy-col">
              <span class="strategy-name">{rule.strategy_name}</span>
              <span class="strategy-filters">
                {#if rule.regime_filter}
                  <span class="filter-tag">{rule.regime_filter}</span>
                {/if}
                {#if rule.strategy_type_filter}
                  <span class="filter-tag">{rule.strategy_type_filter}</span>
                {/if}
              </span>
            </div>

            {#each accountList as account}
              {@const isAssigned = getRuleForStrategyAccount(rule.strategy_id, account.id)?.enabled}
              <div class="matrix-cell toggle-col">
                <button
                  class="toggle-btn"
                  class:active={isAssigned}
                  onclick={() => handleToggleRule(rule.strategy_id, isAssigned || false)}
                  title={isAssigned ? 'Disable routing' : 'Enable routing'}
                >
                  {#if isAssigned}
                    <Check size={14} />
                  {:else}
                    <span class="empty-toggle"></span>
                  {/if}
                </button>
              </div>
            {/each}
          </div>
        {/each}

        {#if filteredRules.length === 0}
          <div class="no-results">
            No routing rules match the current filters.
          </div>
        {/if}
      </div>
    {/if}
  </div>

  <div class="matrix-legend">
    <div class="legend-item">
      <span class="legend-dot active"></span>
      <span>Enabled</span>
    </div>
    <div class="legend-item">
      <span class="legend-dot disabled"></span>
      <span>Disabled</span>
    </div>
  </div>
</div>

<style>
  .routing-matrix-page {
    display: flex;
    flex-direction: column;
    height: 100%;
    background: rgba(10, 15, 26, 0.98);
    backdrop-filter: blur(16px);
  }

  .page-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 16px 24px;
    border-bottom: 1px solid rgba(0, 212, 255, 0.1);
    background: rgba(8, 13, 20, 0.6);
  }

  .header-title {
    display: flex;
    flex-direction: column;
    gap: 2px;
  }

  .header-title h1 {
    font-family: 'JetBrains Mono', monospace;
    font-size: 18px;
    font-weight: 600;
    color: #f59e0b;
    margin: 0;
  }

  .subtitle {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: #666;
  }

  .close-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 32px;
    height: 32px;
    background: transparent;
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 6px;
    color: #888;
    cursor: pointer;
    transition: all 0.15s ease;
  }

  .close-btn:hover {
    background: rgba(255, 59, 59, 0.1);
    border-color: rgba(255, 59, 59, 0.3);
    color: #ff3b3b;
  }

  .filters {
    display: flex;
    gap: 16px;
    padding: 16px 24px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.05);
  }

  .filter-group {
    display: flex;
    align-items: center;
    gap: 8px;
    color: #888;
  }

  .filter-select {
    padding: 6px 12px;
    background: rgba(8, 13, 20, 0.6);
    border: 1px solid rgba(0, 212, 255, 0.15);
    border-radius: 4px;
    color: #e0e0e0;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    cursor: pointer;
  }

  .filter-select:focus {
    outline: none;
    border-color: rgba(0, 212, 255, 0.3);
  }

  .matrix-container {
    flex: 1;
    padding: 24px;
    overflow-y: auto;
  }

  .loading {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 12px;
    padding: 40px;
    color: #888;
    font-family: 'JetBrains Mono', monospace;
    font-size: 14px;
  }

  .loading :global(.spin) {
    animation: spin 1s linear infinite;
  }

  @keyframes spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
  }

  .matrix-table {
    display: flex;
    flex-direction: column;
    gap: 2px;
    background: rgba(255, 255, 255, 0.02);
    border-radius: 8px;
    overflow: hidden;
  }

  .matrix-row {
    display: flex;
    align-items: center;
  }

  .matrix-row.header {
    background: rgba(0, 212, 255, 0.05);
    border-bottom: 1px solid rgba(0, 212, 255, 0.1);
  }

  .matrix-cell {
    padding: 12px 16px;
  }

  .strategy-col {
    flex: 1;
    min-width: 200px;
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .account-col {
    flex: 0 0 140px;
    display: flex;
    flex-direction: column;
    gap: 2px;
    text-align: center;
  }

  .account-name {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    font-weight: 600;
    color: #f59e0b;
  }

  .account-type {
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px;
    color: #666;
    text-transform: uppercase;
  }

  .strategy-name {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    color: #e0e0e0;
  }

  .strategy-filters {
    display: flex;
    gap: 4px;
  }

  .filter-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px;
    color: #666;
    background: rgba(255, 255, 255, 0.05);
    padding: 2px 6px;
    border-radius: 3px;
  }

  .toggle-col {
    flex: 0 0 140px;
    display: flex;
    justify-content: center;
  }

  .toggle-btn {
    width: 24px;
    height: 24px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: transparent;
    border: 1px solid rgba(255, 255, 255, 0.15);
    border-radius: 4px;
    color: #666;
    cursor: pointer;
    transition: all 0.15s ease;
  }

  .toggle-btn:hover {
    background: rgba(255, 255, 255, 0.05);
    border-color: rgba(0, 212, 255, 0.3);
  }

  .toggle-btn.active {
    background: rgba(0, 200, 150, 0.15);
    border-color: rgba(0, 200, 150, 0.4);
    color: #00c896;
  }

  .empty-toggle {
    width: 10px;
    height: 10px;
    border-radius: 2px;
    background: rgba(255, 255, 255, 0.1);
  }

  .no-results {
    padding: 40px;
    text-align: center;
    color: #666;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
  }

  .matrix-legend {
    display: flex;
    gap: 24px;
    justify-content: center;
    padding: 16px;
    border-top: 1px solid rgba(255, 255, 255, 0.05);
  }

  .legend-item {
    display: flex;
    align-items: center;
    gap: 8px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: #666;
  }

  .legend-dot {
    width: 10px;
    height: 10px;
    border-radius: 2px;
  }

  .legend-dot.active {
    background: rgba(0, 200, 150, 0.4);
    border: 1px solid rgba(0, 200, 150, 0.6);
  }

  .legend-dot.disabled {
    background: rgba(255, 255, 255, 0.1);
    border: 1px solid rgba(255, 255, 255, 0.2);
  }
</style>