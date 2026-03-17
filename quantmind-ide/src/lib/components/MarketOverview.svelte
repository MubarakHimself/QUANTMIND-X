<script lang="ts">
  import { Activity, Currency, DollarSign } from 'lucide-svelte';


  interface Props {
    marketState: {
    regime: {
      quality: number;
      trend: 'bullish' | 'bearish' | 'ranging';
      chaos: number;
      volatility: 'low' | 'medium' | 'high';
    };
    symbols: Array<{
      symbol: string;
      price: number;
      change: number;
      spread: number;
    }>;
  };
    houseMoney: {
    dailyProfit: number;
    threshold: number;
    houseMoneyAmount: number;
    mode: 'conservative' | 'normal' | 'aggressive';
  };
  }

  let { marketState, houseMoney }: Props = $props();

  function formatCurrency(value: number) {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2
    }).format(value);
  }
</script>

<div class="market-overview">
  <div class="overview-card regime">
    <div class="card-header">
      <Activity size={16} />
      <h3>Market Regime</h3>
    </div>
    <div class="regime-display">
      <div class="regime-quality">
        <span class="quality-label">Quality</span>
        <div class="quality-bar">
          <div class="quality-fill" style="width: {marketState.regime.quality * 100}%"></div>
        </div>
        <span class="quality-value">{(marketState.regime.quality * 100).toFixed(1)}%</span>
      </div>
      <div class="regime-details">
        <div class="regime-item">
          <span class="label">Trend</span>
          <span class="value {marketState.regime.trend}">{marketState.regime.trend}</span>
        </div>
        <div class="regime-item">
          <span class="label">Chaos</span>
          <span class="value">{marketState.regime.chaos.toFixed(1)}</span>
        </div>
        <div class="regime-item">
          <span class="label">Volatility</span>
          <span class="value">{marketState.regime.volatility}</span>
        </div>
      </div>
    </div>
  </div>

  <div class="overview-card symbols">
    <div class="card-header">
      <Currency size={16} />
      <h3>Active Symbols</h3>
    </div>
    <div class="symbols-list">
      {#each marketState.symbols as symbol}
        <div class="symbol-item">
          <span class="symbol-name">{symbol.symbol}</span>
          <span class="symbol-price">{symbol.price.toFixed(4)}</span>
          <span
            class="symbol-change"
            class:positive={symbol.change > 0}
            class:negative={symbol.change < 0}
          >
            {symbol.change > 0 ? '+' : ''}{symbol.change.toFixed(2)}%
          </span>
          <span class="symbol-spread">Spread: {symbol.spread}</span>
        </div>
      {/each}
    </div>
  </div>

  <div class="overview-card house-money">
    <div class="card-header">
      <DollarSign size={16} />
      <h3>House Money</h3>
      <span class="mode-badge {houseMoney.mode}">{houseMoney.mode}</span>
    </div>
    <div class="house-money-content">
      <div class="hm-profit">
        <span class="label">Daily Profit</span>
        <span class="value success">{formatCurrency(houseMoney.dailyProfit)}</span>
      </div>
      <div class="hm-house">
        <span class="label">House Money</span>
        <span class="value highlight">{formatCurrency(houseMoney.houseMoneyAmount)}</span>
      </div>
      <div class="hm-threshold">
        <span class="label">Threshold</span>
        <span class="value">{(houseMoney.threshold * 100).toFixed(0)}%</span>
      </div>
    </div>
  </div>
</div>

<style>
  .market-overview {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 16px;
    padding: 20px 24px;
  }

  .overview-card {
    background: var(--bg-secondary);
    border: 1px solid var(--border-subtle);
    border-radius: 10px;
    padding: 16px;
  }

  .card-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 12px;
  }

  .card-header h3 {
    margin: 0;
    font-size: 14px;
    color: var(--text-primary);
  }

  /* Regime Card */
  .regime-quality {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 12px;
  }

  .quality-label {
    font-size: 12px;
    color: var(--text-muted);
  }

  .quality-bar {
    flex: 1;
    height: 8px;
    background: var(--bg-tertiary);
    border-radius: 4px;
    overflow: hidden;
  }

  .quality-fill {
    height: 100%;
    background: linear-gradient(90deg, #ef4444, #f59e0b, #10b981);
    transition: width 0.5s;
  }

  .quality-value {
    font-size: 14px;
    font-weight: 600;
    color: var(--text-primary);
  }

  .regime-details {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 8px;
  }

  .regime-item {
    display: flex;
    flex-direction: column;
    gap: 2px;
  }

  .regime-item .label {
    font-size: 10px;
    color: var(--text-muted);
  }

  .regime-item .value {
    font-size: 12px;
    color: var(--text-primary);
  }

  .regime-item .value.bullish {
    color: #10b981;
  }

  .regime-item .value.bearish {
    color: #ef4444;
  }

  /* Symbols Card */
  .symbols-list {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .symbol-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px;
    background: var(--bg-tertiary);
    border-radius: 6px;
    font-size: 12px;
  }

  .symbol-name {
    font-weight: 600;
    color: var(--text-primary);
  }

  .symbol-price {
    font-family: 'JetBrains Mono', monospace;
    color: var(--text-primary);
  }

  .symbol-change {
    font-size: 11px;
  }

  .symbol-change.positive {
    color: #10b981;
  }

  .symbol-change.negative {
    color: #ef4444;
  }

  .symbol-spread {
    font-size: 10px;
    color: var(--text-muted);
  }

  /* House Money Card */
  .mode-badge {
    margin-left: auto;
    padding: 2px 8px;
    border-radius: 10px;
    font-size: 10px;
    text-transform: uppercase;
  }

  .mode-badge.conservative {
    background: rgba(239, 68, 68, 0.2);
    color: #ef4444;
  }

  .mode-badge.normal {
    background: rgba(245, 158, 11, 0.2);
    color: #f59e0b;
  }

  .mode-badge.aggressive {
    background: rgba(16, 185, 129, 0.2);
    color: #10b981;
  }

  .house-money-content {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .hm-profit,
  .hm-house,
  .hm-threshold {
    display: flex;
    justify-content: space-between;
    font-size: 12px;
  }

  .hm-profit .label,
  .hm-house .label,
  .hm-threshold .label {
    color: var(--text-muted);
  }

  .hm-profit .value,
  .hm-house .value,
  .hm-threshold .value {
    font-weight: 600;
    color: var(--text-primary);
  }

  .hm-profit .value.success {
    color: #10b981;
  }

  .hm-house .value.highlight {
    color: #f59e0b;
  }
</style>
