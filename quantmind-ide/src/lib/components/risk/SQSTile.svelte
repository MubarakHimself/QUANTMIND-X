<script lang="ts">
  /**
   * SQSTile - Spread Quality Score Display Tile
   *
   * Displays real-time SQS per active symbol with color coding:
   * - Green (SQS >0.80): Normal trading allowed
   * - Yellow (0.50-0.80): Warning - spreads elevated
   * - Red (SQS <0.50): Critical - hard block active
   *
   * Frosted Terminal glass styling per project conventions.
   *
   * Story: 4-7-spread-quality-score-sqs-system
   */

  import { AlertTriangle, CheckCircle, XCircle, TrendingUp, Clock } from 'lucide-svelte';
  import { sqsStore, sqsData, sqsLoading, getSQSAlertLevel, getSQSColor } from '$lib/stores/sqs';

  export let symbol: string = 'EURUSD';
  export let compact: boolean = false;

  // Alert colors per story requirements
  const alertColors = {
    normal: '#22c55e',    // green
    warning: '#f0a500',  // yellow
    critical: '#ff3b3b' // red
  };

  // Subscribe to store
  let data: any = null;
  let loading = false;

  sqsStore.subscribe(state => {
    data = state.data?.evaluations?.[symbol] ?? null;
    loading = state.loading;
  });

  $: alertLevel = data ? getSQSAlertLevel(data.sqs) : 'normal';
  $: borderColor = alertColors[alertLevel];
  $: iconColor = alertColors[alertLevel];
  $: isAlert = alertLevel !== 'normal';
</script>

<div class="sqstile" style="border-color: {borderColor}">
  <div class="tile-header">
    <div class="symbol-badge">
      <span class="symbol-text">{symbol}</span>
    </div>
    <div class="status-icon" style="color: {iconColor}">
      {#if data?.is_hard_block}
        <XCircle size={16} />
      {:else if data?.allowed}
        <CheckCircle size={16} />
      {:else}
        <AlertTriangle size={16} />
      {/if}
    </div>
  </div>

  {#if loading && !data}
    <div class="loading-state">
      <span class="loading-text">Loading...</span>
    </div>
  {:else if data}
    <div class="tile-content">
      <div class="sqs-value" style="color: {iconColor}">
        <span class="sqs-number">{data.sqs.toFixed(4)}</span>
        <span class="sqs-label">SQS</span>
      </div>

      {#if !compact}
        <div class="metrics-row">
          <div class="metric">
            <span class="metric-label">Spread</span>
            <span class="metric-value">{data.current_spread.toFixed(2)}</span>
          </div>
          <div class="metric">
            <span class="metric-label">History</span>
            <span class="metric-value">{data.historical_avg_spread.toFixed(4)}</span>
          </div>
        </div>

        <div class="threshold-row">
          <div class="threshold-bar">
            <div
              class="threshold-fill"
              style="width: {Math.min(data.sqs * 100, 100)}%; background-color: {iconColor}"
            ></div>
          </div>
          <span class="threshold-text">Threshold: {data.threshold.toFixed(2)}</span>
        </div>
      {/if}

      <div class="status-indicators">
        {#if data.news_override_active}
          <div class="indicator news-indicator" title="News override active">
            <AlertTriangle size={12} />
            <span>News</span>
          </div>
        {/if}

        {#if data.weekend_guard_active}
          <div class="indicator weekend-indicator" title="Weekend guard active">
            <Clock size={12} />
            <span>Weekend</span>
          </div>
        {/if}

        {#if data.warmup_active}
          <div class="indicator warmup-indicator" title="Monday warm-up active">
            <TrendingUp size={12} />
            <span>Warmup</span>
          </div>
        {/if}
      </div>

      {#if data.reason && !compact}
        <div class="reason-text">{data.reason}</div>
      {/if}
    </div>
  {:else}
    <div class="no-data-state">
      <span class="no-data-text">No data</span>
    </div>
  {/if}
</div>

<style>
  .sqstile {
    /* Tier 2 Frosted Terminal Glass */
    background: rgba(8, 13, 20, 0.35);
    backdrop-filter: blur(16px) saturate(120%);
    -webkit-backdrop-filter: blur(16px) saturate(120%);
    border: 1px solid rgba(0, 212, 255, 0.08);
    border-radius: 8px;
    padding: 12px;
    display: flex;
    flex-direction: column;
    gap: 8px;
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
    min-height: 120px;
  }

  .sqstile:hover {
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
  }

  .tile-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
  }

  .symbol-badge {
    background: rgba(0, 212, 255, 0.1);
    border-radius: 4px;
    padding: 2px 8px;
  }

  .symbol-text {
    font-size: 12px;
    font-weight: 600;
    color: rgba(255, 255, 255, 0.9);
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }

  .status-icon {
    display: flex;
    align-items: center;
  }

  .tile-content {
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .sqs-value {
    display: flex;
    align-items: baseline;
    gap: 4px;
  }

  .sqs-number {
    font-size: 24px;
    font-weight: 700;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }

  .sqs-label {
    font-size: 10px;
    color: rgba(255, 255, 255, 0.5);
    text-transform: uppercase;
  }

  .metrics-row {
    display: flex;
    gap: 12px;
  }

  .metric {
    display: flex;
    flex-direction: column;
    gap: 2px;
  }

  .metric-label {
    font-size: 9px;
    color: rgba(255, 255, 255, 0.4);
    text-transform: uppercase;
  }

  .metric-value {
    font-size: 11px;
    color: rgba(255, 255, 255, 0.8);
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }

  .threshold-row {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .threshold-bar {
    height: 4px;
    background: rgba(255, 255, 255, 0.1);
    border-radius: 2px;
    overflow: hidden;
  }

  .threshold-fill {
    height: 100%;
    border-radius: 2px;
    transition: width 0.3s ease;
  }

  .threshold-text {
    font-size: 9px;
    color: rgba(255, 255, 255, 0.4);
  }

  .status-indicators {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
  }

  .indicator {
    display: flex;
    align-items: center;
    gap: 4px;
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 10px;
    background: rgba(255, 255, 255, 0.05);
  }

  .news-indicator {
    color: #f0a500;
  }

  .weekend-indicator {
    color: #00d4ff;
  }

  .warmup-indicator {
    color: #22c55e;
  }

  .reason-text {
    font-size: 10px;
    color: rgba(255, 255, 255, 0.6);
    font-style: italic;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .loading-state,
  .no-data-state {
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: center;
  }

  .loading-text,
  .no-data-text {
    color: rgba(255, 255, 255, 0.4);
    font-size: 12px;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }
</style>
