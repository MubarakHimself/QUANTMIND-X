<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { createEventDispatcher } from 'svelte';
  import { 
    Activity, Cpu, HardDrive, Wifi, Zap, Bot, TrendingUp,
    Database, RefreshCw, Settings, Maximize2, Minimize2, AlertTriangle,
    Info
  } from 'lucide-svelte';

  const dispatch = createEventDispatcher();

  // Tier structure from spec
  interface Tier {
    tier: number;
    minBalance: number;
    maxBalance: number | null;
    maxBots: number;
  }

  interface LimitStatus {
    currentTier: Tier;
    accountBalance: number;
    activeBots: number;
    maxBots: number;
    canAddBot: boolean;
    reason: string;
    safetyBuffer: number;
  }

  const TIERS: Tier[] = [
    { tier: 0, minBalance: 0, maxBalance: 50, maxBots: 0 },
    { tier: 1, minBalance: 50, maxBalance: 100, maxBots: 1 },
    { tier: 2, minBalance: 100, maxBalance: 200, maxBots: 2 },
    { tier: 3, minBalance: 200, maxBalance: 500, maxBots: 3 },
    { tier: 4, minBalance: 500, maxBalance: 1000, maxBots: 5 },
    { tier: 5, minBalance: 1000, maxBalance: 5000, maxBots: 10 },
    { tier: 6, minBalance: 5000, maxBalance: null, maxBots: 20 },
  ];

  const MIN_CAPITAL_PER_BOT = 50;
  const SAFETY_BUFFER_MULTIPLIER = 2;

  let status: LimitStatus | null = null;
  let isLoading = false;
  let error: string | null = null;
  let showDetails = false;
  let refreshInterval: ReturnType<typeof setInterval> | null = null;

  onMount(() => {
    fetchLimitStatus();
    refreshInterval = setInterval(fetchLimitStatus, 30000);
  });

  onDestroy(() => {
    if (refreshInterval) {
      clearInterval(refreshInterval);
    }
  });

  async function fetchLimitStatus() {
    isLoading = true;
    error = null;
    
    try {
      const response = await fetch('/api/bot-limits/status');
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      status = data;
    } catch (e) {
      console.error('Failed to fetch bot limits:', e);
      error = e instanceof Error ? e.message : 'Failed to fetch data';
    } finally {
      isLoading = false;
    }
  }

  function getTierColor(tier: number): string {
    switch (tier) {
      case 0: return 'var(--tier-0, #ef4444)';
      case 1: return 'var(--tier-1, #f97316)';
      case 2: return 'var(--tier-2, #f59e0b)';
      case 3: return 'var(--tier-3, #eab308)';
      case 4: return 'var(--tier-4, #84cc16)';
      case 5: return 'var(--tier-5, #22c55e)';
      case 6: return 'var(--tier-6, #10b981)';
      default: return 'var(--tier-default, #6b7280)';
    }
  }

  function formatBalance(balance: number | null): string {
    if (balance === null) return 'N/A';
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0
    }).format(balance);
  }

  function calculateRequiredCapital(bots: number): number {
    return bots * MIN_CAPITAL_PER_BOT * SAFETY_BUFFER_MULTIPLIER;
  }

  function getProgressPercentage(): number {
    if (!status) return 0;
    return Math.min((status.activeBots / status.maxBots) * 100, 100);
  }

  function getProgressColor(): string {
    const pct = getProgressPercentage();
    if (pct >= 90) return 'var(--accent-danger, #ef4444)';
    if (pct >= 70) return 'var(--accent-warning, #f59e0b)';
    return 'var(--accent-success, #10b981)';
  }
</script>

<div class="limit-indicator">
  <div class="indicator-header">
    <div class="header-left">
      <Bot size={18} />
      <h4>Bot Limit Status</h4>
    </div>
    <div class="header-right">
      <button class="icon-btn" on:click={fetchLimitStatus} title="Refresh">
        <RefreshCw size={14} class:spin={isLoading} />
      </button>
      <button class="icon-btn" on:click={() => showDetails = !showDetails} title="Toggle Details">
        <Info size={14} />
      </button>
    </div>
  </div>

  {#if error}
    <div class="error-banner">
      {error}
    </div>
  {/if}

  {#if status}
    <div class="current-status">
      <div class="tier-badge" style="background-color: {getTierColor(status.currentTier.tier)}">
        Tier {status.currentTier.tier}
      </div>
      
      <div class="balance-info">
        <span class="balance-label">Account Balance:</span>
        <span class="balance-value">{formatBalance(status.accountBalance)}</span>
      </div>
    </div>

    <div class="progress-section">
      <div class="progress-header">
        <span class="progress-label">Active Bots</span>
        <span class="progress-value">
          {status.activeBots} / {status.maxBots}
        </span>
      </div>
      <div class="progress-bar">
        <div 
          class="progress-fill" 
          style="width: {getProgressPercentage()}%; background-color: {getProgressColor()}"
        ></div>
      </div>
    </div>

    <div class="action-status">
      {#if status.canAddBot}
        <div class="status-success">
          <Zap size={16} />
          <span>You can add another bot</span>
        </div>
      {:else}
        <div class="status-error">
          <AlertTriangle size={16} />
          <span>{status.reason}</span>
        </div>
      {/if}
    </div>

    <div class="safety-info">
      <Info size={14} />
      <span>
        Safety buffer: {formatBalance(status.safetyBuffer)} required 
        ({MIN_CAPITAL_PER_BOT} × {status.activeBots + 1} × {SAFETY_BUFFER_MULTIPLIER})
      </span>
    </div>

    {#if showDetails}
      <div class="tier-table">
        <div class="tier-table-header">
          <span>Tier</span>
          <span>Balance Range</span>
          <span>Max Bots</span>
        </div>
        {#each TIERS as tier}
          <div 
            class="tier-row" 
            class:current={status.currentTier.tier === tier.tier}
            style={status.currentTier.tier === tier.tier ? `border-left-color: ${getTierColor(tier.tier)}` : ''}
          >
            <span class="tier-num" style="color: {getTierColor(tier.tier)}">Tier {tier.tier}</span>
            <span class="tier-range">
              {formatBalance(tier.minBalance)} - {tier.maxBalance ? formatBalance(tier.maxBalance) : '∞'}
            </span>
            <span class="tier-bots">{tier.maxBots}</span>
          </div>
        {/each}
      </div>
    {/if}

    <div class="upgrade-recommendation">
      {#if status.currentTier.tier < 6}
        {@const nextTier = TIERS.find(t => t.tier === status!.currentTier.tier + 1)}
        {#if nextTier}
          {@const requiredForNext = calculateRequiredCapital(nextTier.maxBots)}
          {@const needed = requiredForNext - (status.accountBalance || 0)}
          {#if needed > 0}
            <div class="recommendation">
              <TrendingUp size={14} />
              <span>
                Add ${Math.ceil(needed)} more to unlock Tier {nextTier.tier} ({nextTier.maxBots} bots)
              </span>
            </div>
          {/if}
        {/if}
      {/if}
    </div>
  {:else if isLoading}
    <div class="loading">Loading...</div>
  {:else}
    <div class="empty">No limit data available</div>
  {/if}
</div>

<style>
  .limit-indicator {
    background: var(--bg-secondary, #1e293b);
    border-radius: 8px;
    padding: 16px;
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  .indicator-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .header-left {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .header-left h4 {
    margin: 0;
    font-size: 14px;
    font-weight: 600;
  }

  .header-right {
    display: flex;
    gap: 4px;
  }

  .icon-btn {
    background: none;
    border: none;
    color: var(--text-secondary, #94a3b8);
    cursor: pointer;
    padding: 4px;
    border-radius: 4px;
    transition: all 0.2s;
  }

  .icon-btn:hover {
    background: var(--bg-hover, #334155);
    color: var(--text-primary, #f1f5f9);
  }

  :global(.spin) {
    animation: spin 1s linear infinite;
  }

  @keyframes spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
  }

  .error-banner {
    background: var(--accent-danger, #ef4444);
    color: white;
    padding: 8px 12px;
    border-radius: 4px;
    font-size: 12px;
  }

  .current-status {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .tier-badge {
    padding: 4px 12px;
    border-radius: 12px;
    color: white;
    font-size: 12px;
    font-weight: 700;
  }

  .balance-info {
    display: flex;
    flex-direction: column;
    align-items: flex-end;
  }

  .balance-label {
    font-size: 10px;
    color: var(--text-secondary, #94a3b8);
    text-transform: uppercase;
  }

  .balance-value {
    font-size: 16px;
    font-weight: 700;
    color: var(--text-primary, #f1f5f9);
  }

  .progress-section {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .progress-header {
    display: flex;
    justify-content: space-between;
    font-size: 12px;
  }

  .progress-label {
    color: var(--text-secondary, #94a3b8);
  }

  .progress-value {
    font-weight: 600;
    color: var(--text-primary, #f1f5f9);
  }

  .progress-bar {
    height: 8px;
    background: var(--bg-tertiary, #334155);
    border-radius: 4px;
    overflow: hidden;
  }

  .progress-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.3s ease, background-color 0.3s ease;
  }

  .action-status {
    padding: 8px;
    border-radius: 6px;
  }

  .status-success {
    display: flex;
    align-items: center;
    gap: 8px;
    color: var(--accent-success, #10b981);
    font-size: 13px;
  }

  .status-error {
    display: flex;
    align-items: center;
    gap: 8px;
    color: var(--accent-danger, #ef4444);
    font-size: 13px;
  }

  .safety-info {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 11px;
    color: var(--text-secondary, #94a3b8);
    padding: 8px;
    background: var(--bg-tertiary, #334155);
    border-radius: 6px;
  }

  .tier-table {
    display: flex;
    flex-direction: column;
    gap: 2px;
    font-size: 11px;
    background: var(--bg-tertiary, #334155);
    border-radius: 6px;
    overflow: hidden;
  }

  .tier-table-header {
    display: grid;
    grid-template-columns: 1fr 2fr 1fr;
    padding: 8px 12px;
    background: var(--bg-hover, #475569);
    font-weight: 600;
    color: var(--text-primary, #f1f5f9);
  }

  .tier-row {
    display: grid;
    grid-template-columns: 1fr 2fr 1fr;
    padding: 6px 12px;
    border-left: 3px solid transparent;
    color: var(--text-secondary, #94a3b8);
  }

  .tier-row.current {
    background: var(--bg-hover, #475569);
    color: var(--text-primary, #f1f5f9);
  }

  .tier-num {
    font-weight: 600;
  }

  .tier-bots {
    text-align: right;
    font-weight: 600;
  }

  .upgrade-recommendation {
    padding: 8px;
    background: var(--bg-tertiary, #334155);
    border-radius: 6px;
  }

  .recommendation {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 12px;
    color: var(--accent-primary, #3b82f6);
  }

  .loading, .empty {
    text-align: center;
    color: var(--text-secondary, #94a3b8);
    padding: 16px;
  }
</style>
