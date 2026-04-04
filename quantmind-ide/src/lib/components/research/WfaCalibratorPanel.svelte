<script lang="ts">
  /**
   * WfaCalibratorPanel - Walk-Forward Analysis Window Calibrator Display
   *
   * Displays:
   * - Optimal window size (large number display)
   * - Sharpe ratio bar chart for each window candidate
   * - Current vs optimal window comparison
   * - Last calibration timestamp
   * - Regime context
   */

  import { onMount, onDestroy } from 'svelte';
  import { Calendar, TrendingUp, Clock, Brain } from 'lucide-svelte';
  import { wfaStore, type WfaCalibration } from '$lib/stores/wfaStore';

  let calibration = $state<WfaCalibration | null>(null);
  let isLoading = $state(false);
  let error = $state<string | null>(null);
  let windowDeviation = $derived(
    calibration ? calibration.current_window - calibration.optimal_window : null
  );
  let isWindowOptimal = $derived(windowDeviation === 0 && calibration !== null);
  let lastCalibrationFormatted = $derived(
    calibration?.last_calibration ? new Date(calibration.last_calibration).toLocaleString() : null
  );
  let regimeColor = $derived(() => {
    const regime = calibration?.regime || 'UNKNOWN';
    const colors: Record<string, string> = {
      TRENDING: '#10b981',
      RANGING: '#f59e0b',
      BREAKOUT: '#8b5cf6',
      VOLATILE: '#ef4444',
      UNKNOWN: '#6b7280',
    };
    return colors[regime] || colors.UNKNOWN;
  });

  // Subscribe to store
  const unsubscribe = wfaStore.subscribe(state => {
    calibration = state.calibration;
    isLoading = state.isLoading;
    error = state.error;
  });

  onMount(() => {
    wfaStore.refresh();
    wfaStore.startAutoRefresh(60000);
  });

  onDestroy(() => {
    unsubscribe();
    wfaStore.stopAutoRefresh();
  });

  // Find max Sharpe for scaling bars
  let maxSharpe = $derived(
    calibration?.sharpe_ratios?.length
      ? Math.max(...calibration.sharpe_ratios)
      : 1
  );

  // Determine bar color based on Sharpe value
  function getBarColor(sharpe: number, isOptimal: boolean): string {
    if (!isOptimal) return 'rgba(255, 255, 255, 0.2)';
    if (sharpe >= 2.0) return '#10b981';
    if (sharpe >= 1.5) return '#3b82f6';
    if (sharpe >= 1.0) return '#f59e0b';
    return '#ef4444';
  }
</script>

<div class="wfa-panel">
  <!-- Header -->
  <div class="panel-header">
    <div class="header-title">
      <Calendar size={16} />
      <span>WFA Calibrator</span>
    </div>
    <div class="header-badge" style="background: {regimeColor}20; color: {regimeColor}">
      <Brain size={12} />
      <span>{calibration?.regime || 'UNKNOWN'}</span>
    </div>
  </div>

  {#if isLoading && !calibration}
    <div class="loading-state">
      <div class="spinner"></div>
      <span>Loading calibration...</span>
    </div>
  {:else if error && !calibration}
    <div class="error-state">
      <span class="error-text">{error}</span>
      <button class="retry-btn" on:click={() => wfaStore.refresh()}>Retry</button>
    </div>
  {:else if calibration}
    <!-- Optimal Window Display -->
    <div class="optimal-section">
      <div class="optimal-label">Optimal Window</div>
      <div class="optimal-value">
        <span class="big-number">{calibration.optimal_window}</span>
        <span class="unit">days</span>
      </div>
      {#if windowDeviation !== null}
        <div class="deviation" class:optimal={windowDeviation === 0} class:off={windowDeviation !== 0}>
          {#if windowDeviation === 0}
            <span>Current matches optimal</span>
          {:else}
            <span>{windowDeviation > 0 ? '+' : ''}{windowDeviation} days from current</span>
          {/if}
        </div>
      {/if}
    </div>

    <!-- Sharpe Ratio Bars -->
    {#if calibration.window_candidates?.length > 0}
      <div class="sharpe-section">
        <div class="section-label">
          <TrendingUp size={12} />
          <span>Sharpe Ratios by Window</span>
        </div>
        <div class="sharpe-bars">
          {#each calibration.window_candidates as window, i}
            {@const sharpe = calibration.sharpe_ratios[i]}
            {@const isOptimal = window === calibration.optimal_window}
            <div class="sharpe-bar-container" class:optimal={isOptimal}>
              <div class="bar-wrapper">
                <div
                  class="bar-fill"
                  style="height: {(sharpe / maxSharpe) * 100}%; background: {getBarColor(sharpe, isOptimal)}"
                ></div>
              </div>
              <div class="bar-label">{window}d</div>
              <div class="bar-value" style="color: {getBarColor(sharpe, isOptimal)}">
                {sharpe.toFixed(2)}
              </div>
            </div>
          {/each}
        </div>
      </div>
    {/if}

    <!-- Current Window Info -->
    <div class="info-row">
      <div class="info-item">
        <span class="info-label">Current Window</span>
        <span class="info-value">{calibration.current_window} days</span>
      </div>
      <div class="info-item">
        <span class="info-label">Window Type</span>
        <span class="info-value">{calibration.window_type || 'adaptive'}</span>
      </div>
    </div>

    <!-- Last Calibration -->
    {#if lastCalibrationFormatted}
      <div class="timestamp-row">
        <Clock size={12} />
        <span>Last calibrated: {lastCalibrationFormatted}</span>
      </div>
    {/if}

    <!-- Refresh Button -->
    <button class="refresh-btn" on:click={() => wfaStore.refresh()} disabled={isLoading}>
      {isLoading ? 'Refreshing...' : 'Refresh Calibration'}
    </button>
  {:else}
    <div class="empty-state">
      <span>No calibration data available</span>
    </div>
  {/if}
</div>

<style>
  .wfa-panel {
    display: flex;
    flex-direction: column;
    gap: 16px;
    padding: 16px;
    background: rgba(15, 23, 42, 0.8);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 8px;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    color: rgba(255, 255, 255, 0.9);
    min-width: 280px;
  }

  .panel-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding-bottom: 12px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
  }

  .header-title {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 14px;
    font-weight: 600;
    color: rgba(255, 255, 255, 0.9);
  }

  .header-badge {
    display: flex;
    align-items: center;
    gap: 4px;
    padding: 4px 8px;
    border-radius: 4px;
    font-size: 10px;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  .optimal-section {
    text-align: center;
    padding: 16px 0;
  }

  .optimal-label {
    font-size: 11px;
    color: rgba(255, 255, 255, 0.5);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 8px;
  }

  .optimal-value {
    display: flex;
    align-items: baseline;
    justify-content: center;
    gap: 8px;
  }

  .big-number {
    font-size: 48px;
    font-weight: 700;
    color: #00d4ff;
    line-height: 1;
  }

  .unit {
    font-size: 16px;
    color: rgba(255, 255, 255, 0.5);
  }

  .deviation {
    margin-top: 8px;
    font-size: 11px;
    color: rgba(255, 255, 255, 0.5);
  }

  .deviation.optimal {
    color: #10b981;
  }

  .deviation.off {
    color: #f59e0b;
  }

  .sharpe-section {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .section-label {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 11px;
    color: rgba(255, 255, 255, 0.5);
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  .sharpe-bars {
    display: flex;
    justify-content: space-between;
    gap: 8px;
    height: 80px;
    padding: 8px 0;
  }

  .sharpe-bar-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 4px;
    flex: 1;
  }

  .sharpe-bar-container.optimal .bar-fill {
    box-shadow: 0 0 12px rgba(0, 212, 255, 0.5);
  }

  .bar-wrapper {
    width: 100%;
    height: 50px;
    display: flex;
    align-items: flex-end;
    justify-content: center;
  }

  .bar-fill {
    width: 70%;
    max-width: 32px;
    border-radius: 4px 4px 0 0;
    transition: height 0.3s ease, background 0.3s ease;
    min-height: 4px;
  }

  .bar-label {
    font-size: 11px;
    color: rgba(255, 255, 255, 0.6);
    font-weight: 500;
  }

  .bar-value {
    font-size: 10px;
    font-weight: 600;
  }

  .info-row {
    display: flex;
    justify-content: space-between;
    gap: 16px;
    padding: 12px;
    background: rgba(255, 255, 255, 0.05);
    border-radius: 6px;
  }

  .info-item {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .info-label {
    font-size: 10px;
    color: rgba(255, 255, 255, 0.4);
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  .info-value {
    font-size: 13px;
    color: rgba(255, 255, 255, 0.9);
    font-weight: 500;
  }

  .timestamp-row {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 11px;
    color: rgba(255, 255, 255, 0.4);
  }

  .refresh-btn {
    padding: 8px 16px;
    background: rgba(0, 212, 255, 0.15);
    border: 1px solid rgba(0, 212, 255, 0.3);
    border-radius: 6px;
    color: #00d4ff;
    font-size: 12px;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    cursor: pointer;
    transition: background 0.2s ease;
  }

  .refresh-btn:hover:not(:disabled) {
    background: rgba(0, 212, 255, 0.25);
  }

  .refresh-btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .loading-state,
  .error-state,
  .empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 12px;
    padding: 32px 16px;
    color: rgba(255, 255, 255, 0.5);
    font-size: 12px;
  }

  .spinner {
    width: 24px;
    height: 24px;
    border: 2px solid rgba(255, 255, 255, 0.1);
    border-top-color: #00d4ff;
    border-radius: 50%;
    animation: spin 1s linear infinite;
  }

  @keyframes spin {
    to { transform: rotate(360deg); }
  }

  .error-text {
    color: #ef4444;
  }

  .retry-btn {
    padding: 6px 12px;
    background: rgba(239, 68, 68, 0.15);
    border: 1px solid rgba(239, 68, 68, 0.3);
    border-radius: 4px;
    color: #ef4444;
    font-size: 11px;
    cursor: pointer;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }

  .retry-btn:hover {
    background: rgba(239, 68, 68, 0.25);
  }
</style>
