<script lang="ts">
  import { TrendingUp, TrendingDown, Minus } from 'lucide-svelte';

  interface Props {
    title?: string;
    value?: number | string;
    unit?: string;
    trend?: 'up' | 'down' | 'neutral';
    trendValue?: string;
    threshold?: { warning?: number; critical?: number };
    icon?: any;
    isLoading?: boolean;
  }

  let {
    title = '',
    value = 0,
    unit = '',
    trend = 'neutral',
    trendValue = '',
    threshold = {},
    icon = null,
    isLoading = false
  }: Props = $props();


  function getStatus(): 'normal' | 'warning' | 'critical' {
    if (typeof value !== 'number') return 'normal';

    if (threshold.critical !== undefined && value >= threshold.critical) {
      return 'critical';
    }
    if (threshold.warning !== undefined && value >= threshold.warning) {
      return 'warning';
    }
    return 'normal';
  }

  function formatNumber(num: number): string {
    if (num >= 1000000) {
      return (num / 1000000).toFixed(1) + 'M';
    }
    if (num >= 1000) {
      return (num / 1000).toFixed(1) + 'K';
    }
    if (num < 1 && num > 0) {
      return num.toFixed(2);
    }
    return num.toFixed(1);
  }

  function getTrendIcon() {
    switch (trend) {
      case 'up': return TrendingUp;
      case 'down': return TrendingDown;
      default: return Minus;
    }
  }

  function getThresholdWidth(): number {
    if (typeof value !== 'number' || !threshold.critical) return 0;
    return Math.min((value / threshold.critical) * 100, 100);
  }
  let status = $derived(getStatus());
  let formattedValue = $derived(typeof value === 'number' ? formatNumber(value) : value);
</script>

<div class="metric-card" class:warning={status === 'warning'} class:critical={status === 'critical'}>
  <div class="metric-header">
    {#if icon}
      {@const SvelteComponent = icon}
      <div class="metric-icon">
        <SvelteComponent size={18} />
      </div>
    {/if}
    <span class="metric-title">{title}</span>
  </div>

  <div class="metric-body">
    {#if isLoading}
      <div class="metric-loading">
        <div class="skeleton"></div>
      </div>
    {:else}
      <div class="metric-value">
        <span class="value">{formattedValue}</span>
        {#if unit}
          <span class="unit">{unit}</span>
        {/if}
      </div>

      {#if trendValue}
        {@const SvelteComponent_1 = getTrendIcon()}
        <div class="metric-trend" class:trend-up={trend === 'up'} class:trend-down={trend === 'down'}>
          <SvelteComponent_1 size={14} />
          <span>{trendValue}</span>
        </div>
      {/if}
    {/if}
  </div>

  {#if threshold.warning !== undefined || threshold.critical !== undefined}
    <div class="metric-threshold">
      <div class="threshold-bar">
        <div class="threshold-fill" style="width: {getThresholdWidth()}%"></div>
        <div class="threshold-marker warning" style="left: {threshold.warning ? (threshold.warning / (threshold.critical || 100)) * 100 : 0}%"></div>
        <div class="threshold-marker critical" style="left: {threshold.critical || 100}%"></div>
      </div>
    </div>
  {/if}
</div>

<style>
  .metric-card {
    display: flex;
    flex-direction: column;
    padding: 16px;
    background: var(--color-bg-surface);
    border: 1px solid var(--color-border-subtle);
    border-radius: 8px;
    transition: all 0.2s ease;
  }

  .metric-card:hover {
    border-color: var(--border-default);
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
  }

  .metric-card.warning {
    border-color: var(--color-accent-amber);
    background: rgba(255, 152, 0, 0.05);
  }

  .metric-card.critical {
    border-color: var(--color-accent-red);
    background: rgba(244, 67, 54, 0.05);
    animation: pulse-critical 2s infinite;
  }

  @keyframes pulse-critical {
    0%, 100% { box-shadow: 0 0 0 0 rgba(244, 67, 54, 0.4); }
    50% { box-shadow: 0 0 0 8px rgba(244, 67, 54, 0); }
  }

  .metric-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 12px;
  }

  .metric-icon {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 32px;
    height: 32px;
    background: var(--color-bg-elevated);
    border-radius: 6px;
    color: var(--color-accent-cyan);
  }

  .metric-title {
    font-size: 12px;
    font-weight: 500;
    color: var(--color-text-muted);
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  .metric-body {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: 8px;
  }

  .metric-value {
    display: flex;
    align-items: baseline;
    gap: 4px;
  }

  .value {
    font-size: 28px;
    font-weight: 600;
    color: var(--color-text-primary);
    line-height: 1;
  }

  .unit {
    font-size: 14px;
    color: var(--color-text-muted);
  }

  .metric-trend {
    display: flex;
    align-items: center;
    gap: 4px;
    padding: 4px 8px;
    background: var(--color-bg-elevated);
    border-radius: 12px;
    font-size: 12px;
    color: var(--color-text-muted);
  }

  .metric-trend.trend-up {
    color: var(--color-accent-green);
    background: rgba(76, 175, 80, 0.1);
  }

  .metric-trend.trend-down {
    color: var(--color-accent-red);
    background: rgba(244, 67, 54, 0.1);
  }

  .metric-loading {
    flex: 1;
  }

  .skeleton {
    height: 32px;
    background: linear-gradient(90deg, var(--color-bg-elevated) 25%, var(--color-bg-surface) 50%, var(--color-bg-elevated) 75%);
    background-size: 200% 100%;
    animation: skeleton-loading 1.5s infinite;
    border-radius: 4px;
  }

  @keyframes skeleton-loading {
    0% { background-position: 200% 0; }
    100% { background-position: -200% 0; }
  }

  .metric-threshold {
    margin-top: 12px;
  }

  .threshold-bar {
    height: 4px;
    background: var(--color-bg-elevated);
    border-radius: 2px;
    position: relative;
    overflow: hidden;
  }

  .threshold-fill {
    height: 100%;
    background: var(--color-accent-cyan);
    border-radius: 2px;
    transition: width 0.3s ease;
  }

  .metric-card.warning .threshold-fill {
    background: var(--color-accent-amber);
  }

  .metric-card.critical .threshold-fill {
    background: var(--color-accent-red);
  }

  .threshold-marker {
    position: absolute;
    top: -2px;
    width: 2px;
    height: 8px;
    background: transparent;
    transform: translateX(-50%);
  }

  .threshold-marker.warning {
    background: var(--color-accent-amber);
  }

  .threshold-marker.critical {
    background: var(--color-accent-red);
  }
</style>
