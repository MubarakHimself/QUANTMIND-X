<script lang="ts">
  /**
   * AccountTile - Individual Broker Account Display
   *
   * Shows: broker name, account type, equity, drawdown, exposure
   * Uses GlassTile container for Frosted Terminal aesthetic
   */
  import GlassTile from '$lib/components/live-trading/GlassTile.svelte';
  import type { BrokerAccount } from '$lib/stores/portfolio';
  import { TrendingDown, TrendingUp, Wallet, Percent, Activity } from 'lucide-svelte';

  interface Props {
    account: BrokerAccount;
  }

  let { account }: Props = $props();

  function formatCurrency(value: number): string {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: account.currency || 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0
    }).format(value);
  }

  function formatPercent(value: number): string {
    return `${value.toFixed(1)}%`;
  }

  function getAccountTypeLabel(type: string): string {
    const labels: Record<string, string> = {
      'MACHINE_GUN': 'HFT/Scalper',
      'SNIPER': 'ICT/Structure',
      'PROP_FIRM': 'Prop Firm',
      'CRYPTO': 'Crypto',
      'DEMO': 'Demo'
    };
    return labels[type] || type;
  }
</script>

<GlassTile clickable>
  <div class="account-tile">
    <div class="header">
      <div class="broker-info">
        <span class="broker-name">{account.broker_name}</span>
        <span class="account-type">{getAccountTypeLabel(account.account_type)}</span>
      </div>
      <div class="connection-status" class:connected={account.connected}>
        <span class="status-dot"></span>
      </div>
    </div>

    <div class="metrics">
      <div class="metric equity">
        <Wallet size={14} />
        <span class="label">Equity</span>
        <span class="value">{formatCurrency(account.equity)}</span>
      </div>

      <div class="metric drawdown" class:warning={account.drawdown && account.drawdown > 5}>
        <Percent size={14} />
        <span class="label">DD</span>
        <span class="value">{formatPercent(account.drawdown || 0)}</span>
      </div>

      <div class="metric exposure">
        <Activity size={14} />
        <span class="label">Exp</span>
        <span class="value">{account.exposure || 0}%</span>
      </div>
    </div>

    <div class="footer">
      <span class="account-number">{account.account_id}</span>
      <span class="server">{account.server}</span>
    </div>
  </div>
</GlassTile>

<style>
  .account-tile {
    display: flex;
    flex-direction: column;
    gap: 12px;
    min-width: 180px;
  }

  .header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
  }

  .broker-info {
    display: flex;
    flex-direction: column;
    gap: 2px;
  }

  .broker-name {
    font-family: 'JetBrains Mono', monospace;
    font-size: 14px;
    font-weight: 600;
    color: #f59e0b;
  }

  .account-type {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: #888;
    text-transform: uppercase;
  }

  .connection-status {
    display: flex;
    align-items: center;
  }

  .status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #666;
  }

  .connection-status.connected .status-dot {
    background: #00c896;
    box-shadow: 0 0 6px rgba(0, 200, 150, 0.5);
  }

  .metrics {
    display: flex;
    gap: 16px;
  }

  .metric {
    display: flex;
    flex-direction: column;
    gap: 2px;
  }

  .metric .label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px;
    color: #666;
    text-transform: uppercase;
  }

  .metric .value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    color: #e0e0e0;
  }

  .metric.equity .value {
    color: #00d4ff;
  }

  .metric.drawdown.warning .value {
    color: #ff3b3b;
  }

  .metric.exposure .value {
    color: #aaa;
  }

  .footer {
    display: flex;
    justify-content: space-between;
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: #555;
  }

  .account-number {
    color: #666;
  }
</style>