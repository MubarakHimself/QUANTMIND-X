<script lang="ts">
  /**
   * CorrelationSensorPanel - Correlation Regime Visualization
   *
   * Displays:
   * - max_eigenvalue with color coding (green/amber/red)
   * - RMT threshold reference line
   * - Correlation heatmap visualization
   * - Timeframe toggle (M5/H1)
   * - Regime indicator: CORRELATED / UNCORRELATED / NEUTRAL
   *
   * Uses Frosted Terminal glass aesthetic with QUANTMINDX color tokens.
   */

  import { onMount, onDestroy } from 'svelte';
  import { Grid, Activity, AlertTriangle, CheckCircle } from 'lucide-svelte';
  import {
    correlationSensorStore,
    correlationData,
    correlationLoading,
    correlationError,
    correlationRegime,
    eigenvalueStatus,
    eigenvalueColor,
    regimeColor,
    type CorrelationSensorData
  } from '$lib/stores/correlationSensor';

  // Selected timeframe for matrix display
  let selectedTimeframe: 'M5' | 'H1' = 'M5';

  // Heatmap cell size
  const CELL_SIZE = 24;
  const CELL_GAP = 2;

  // Start polling on mount
  onMount(() => {
    correlationSensorStore.startPolling(5000);
  });

  // Stop polling on destroy
  onDestroy(() => {
    correlationSensorStore.stopPolling();
  });

  // Get the active matrix based on selected timeframe
  $: activeMatrix = $correlationData
    ? (selectedTimeframe === 'M5' ? $correlationData.m5_matrix : $correlationData.h1_matrix)
    : null;

  // Get heatmap cell color based on correlation value
  function getCellColor(value: number): string {
    // Correlation values range from -1 to 1
    // Map to color: negative = blue, zero = neutral, positive = warm
    if (value >= 0.9) return 'rgba(255, 59, 59, 0.8)';    // Strong positive - red
    if (value >= 0.7) return 'rgba(255, 107, 107, 0.7)'; // Moderate positive
    if (value >= 0.4) return 'rgba(255, 183, 0, 0.5)';   // Weak positive - amber
    if (value >= 0.2) return 'rgba(255, 255, 255, 0.1)'; // Very weak
    if (value >= -0.2) return 'rgba(255, 255, 255, 0.15)'; // Near zero
    if (value >= -0.4) return 'rgba(0, 212, 255, 0.3)';  // Weak negative
    if (value >= -0.7) return 'rgba(0, 212, 255, 0.5)';  // Moderate negative
    return 'rgba(0, 212, 255, 0.7)';                     // Strong negative - cyan
  }

  // Format eigenvalue for display
  function formatEigenvalue(value: number): string {
    if (value >= 100) return value.toFixed(1);
    if (value >= 10) return value.toFixed(2);
    return value.toFixed(3);
  }

  // Get regime icon
  $: regimeIcon = $correlationRegime === 'CORRELATED'
    ? AlertTriangle
    : $correlationRegime === 'UNCORRELATED'
      ? CheckCircle
      : Activity;
</script>

<div class="correlation-panel">
  <!-- Panel Header -->
  <div class="panel-header">
    <div class="header-title">
      <Grid size={16} />
      <h2>Correlation Sensor</h2>
    </div>
    {#if $correlationData}
      <div class="timestamp">
        {new Date($correlationData.timestamp * 1000).toLocaleTimeString()}
      </div>
    {/if}
  </div>

  {#if $correlationLoading && !$correlationData}
    <div class="loading-state">
      <div class="spinner"></div>
      <span>Loading correlation data...</span>
    </div>
  {:else if $correlationError}
    <div class="error-state">
      <AlertTriangle size={20} />
      <span>{$correlationError}</span>
    </div>
  {:else if $correlationData}
    <div class="panel-content">
      <!-- Eigenvalue Display -->
      <div class="eigenvalue-section">
        <div class="eigenvalue-display">
          <span class="eigenvalue-label">Max Eigenvalue</span>
          <span
            class="eigenvalue-value"
            style="color: {$eigenvalueColor}"
          >
            {formatEigenvalue($correlationData.max_eigenvalue)}
          </span>
        </div>

        <div class="threshold-bar">
          <div class="threshold-label">
            <span>RMT Threshold</span>
            <span class="threshold-value">{$correlationData.rmt_threshold.toFixed(3)}</span>
          </div>
          <div class="bar-track">
            <div
              class="bar-indicator"
              class:critical={$eigenvalueStatus === 'critical'}
              class:warning={$eigenvalueStatus === 'warning'}
              class:normal={$eigenvalueStatus === 'normal'}
              style="width: {Math.min(100, ($correlationData.max_eigenvalue / $correlationData.rmt_threshold) * 50)}%"
            ></div>
            <div class="threshold-line" style="left: 50%"></div>
          </div>
        </div>
      </div>

      <!-- Regime Indicator -->
      <div class="regime-section">
        <div class="regime-badge" style="border-color: {$regimeColor}; color: {$regimeColor}">
          <svelte:component this={regimeIcon} size={14} />
          <span class="regime-text">{$correlationRegime}</span>
        </div>
        <div class="regime-description">
          {#if $correlationRegime === 'CORRELATED'}
            Market synchronized - elevated risk
          {:else if $correlationRegime === 'UNCORRELATED'}
            Market noise-like - normal conditions
          {:else}
            Borderline regime
          {/if}
        </div>
      </div>

      <!-- Timeframe Toggle -->
      <div class="timeframe-toggle">
        <button
          class="tf-btn"
          class:active={selectedTimeframe === 'M5'}
          on:click={() => selectedTimeframe = 'M5'}
        >
          M5
        </button>
        <button
          class="tf-btn"
          class:active={selectedTimeframe === 'H1'}
          on:click={() => selectedTimeframe = 'H1'}
        >
          H1
        </button>
      </div>

      <!-- Correlation Heatmap -->
      {#if activeMatrix}
        <div class="heatmap-section">
          <span class="heatmap-label">Correlation Matrix ({selectedTimeframe})</span>
          <div class="heatmap-container">
            <div class="heatmap-grid" style="--cell-size: {CELL_SIZE}px; --cell-gap: {CELL_GAP}px;">
              {#each activeMatrix as row, i}
                {#each row as cell, j}
                  <div
                    class="heatmap-cell"
                    style="background: {getCellColor(cell)};"
                    title="{$correlationData?.symbols[i] ?? ''} / {$correlationData?.symbols[j] ?? ''}: {cell.toFixed(3)}"
                  ></div>
                {/each}
              {/each}
            </div>

            <!-- Symbol labels -->
            <div class="symbol-labels">
              {#each $correlationData?.symbols ?? [] as symbol}
                <span class="symbol-label">{symbol.slice(0, 3)}</span>
              {/each}
            </div>
          </div>

          <!-- Color scale legend -->
          <div class="color-legend">
            <span class="legend-label">-1</span>
            <div class="legend-gradient"></div>
            <span class="legend-label">+1</span>
          </div>
        </div>
      {/if}

      <!-- Top Eigenvalues -->
      <div class="eigenvalues-section">
        <span class="section-label">Top Eigenvalues</span>
        <div class="eigenvalues-list">
          {#each $correlationData.eigenvalues.slice(0, 5) as ev, idx}
            <div class="eigenvalue-item">
              <span class="ev-index">{idx + 1}</span>
              <span class="ev-value">{formatEigenvalue(ev)}</span>
              {#if idx === 0}
                <span class="ev-badge">max</span>
              {/if}
            </div>
          {/each}
        </div>
      </div>
    </div>
  {/if}
</div>

<style>
  .correlation-panel {
    /* Tier 2 Frosted Terminal Glass */
    background: rgba(8, 13, 20, 0.35);
    backdrop-filter: blur(16px) saturate(120%);
    -webkit-backdrop-filter: blur(16px) saturate(120%);
    border: 1px solid rgba(0, 212, 255, 0.08);
    border-radius: 8px;
    padding: 16px;
    display: flex;
    flex-direction: column;
    gap: 16px;
    min-width: 320px;
    max-width: 400px;
  }

  .panel-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .header-title {
    display: flex;
    align-items: center;
    gap: 8px;
    color: rgba(255, 255, 255, 0.9);
  }

  .header-title h2 {
    font-size: 14px;
    font-weight: 600;
    margin: 0;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    letter-spacing: 0.5px;
  }

  .timestamp {
    font-size: 11px;
    color: rgba(255, 255, 255, 0.4);
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }

  .loading-state,
  .error-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 12px;
    padding: 32px;
    color: rgba(255, 255, 255, 0.5);
    font-size: 12px;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }

  .error-state {
    color: #ff3b3b;
  }

  .spinner {
    width: 24px;
    height: 24px;
    border: 2px solid rgba(0, 212, 255, 0.2);
    border-top-color: #00d4ff;
    border-radius: 50%;
    animation: spin 1s linear infinite;
  }

  @keyframes spin {
    to { transform: rotate(360deg); }
  }

  .panel-content {
    display: flex;
    flex-direction: column;
    gap: 16px;
  }

  /* Eigenvalue Section */
  .eigenvalue-section {
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  .eigenvalue-display {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
  }

  .eigenvalue-label {
    font-size: 11px;
    color: rgba(255, 255, 255, 0.5);
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  .eigenvalue-value {
    font-size: 28px;
    font-weight: 700;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    transition: color 0.3s ease;
  }

  .threshold-bar {
    display: flex;
    flex-direction: column;
    gap: 6px;
  }

  .threshold-label {
    display: flex;
    justify-content: space-between;
    font-size: 10px;
    color: rgba(255, 255, 255, 0.4);
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }

  .threshold-value {
    color: rgba(255, 255, 255, 0.6);
  }

  .bar-track {
    position: relative;
    height: 6px;
    background: rgba(255, 255, 255, 0.1);
    border-radius: 3px;
    overflow: hidden;
  }

  .bar-indicator {
    position: absolute;
    left: 0;
    top: 0;
    height: 100%;
    border-radius: 3px;
    transition: width 0.3s ease, background 0.3s ease;
  }

  .bar-indicator.normal {
    background: linear-gradient(90deg, #00d4ff 0%, rgba(0, 212, 255, 0.5) 100%);
  }

  .bar-indicator.warning {
    background: linear-gradient(90deg, #ffb700 0%, rgba(255, 183, 0, 0.5) 100%);
  }

  .bar-indicator.critical {
    background: linear-gradient(90deg, #ff3b3b 0%, rgba(255, 59, 59, 0.5) 100%);
  }

  .threshold-line {
    position: absolute;
    top: -2px;
    bottom: -2px;
    width: 2px;
    background: rgba(255, 255, 255, 0.6);
    border-radius: 1px;
  }

  /* Regime Section */
  .regime-section {
    display: flex;
    align-items: center;
    gap: 12px;
  }

  .regime-badge {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 6px 12px;
    border: 1px solid;
    border-radius: 4px;
    font-size: 12px;
    font-weight: 600;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    letter-spacing: 0.5px;
    transition: border-color 0.3s ease, color 0.3s ease;
  }

  .regime-text {
    text-transform: uppercase;
  }

  .regime-description {
    font-size: 11px;
    color: rgba(255, 255, 255, 0.5);
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }

  /* Timeframe Toggle */
  .timeframe-toggle {
    display: flex;
    gap: 4px;
    padding: 4px;
    background: rgba(0, 0, 0, 0.2);
    border-radius: 4px;
    width: fit-content;
  }

  .tf-btn {
    padding: 6px 16px;
    border: none;
    border-radius: 3px;
    background: transparent;
    color: rgba(255, 255, 255, 0.5);
    font-size: 11px;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.2s ease;
  }

  .tf-btn:hover {
    color: rgba(255, 255, 255, 0.8);
  }

  .tf-btn.active {
    background: rgba(0, 212, 255, 0.2);
    color: #00d4ff;
  }

  /* Heatmap Section */
  .heatmap-section {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .heatmap-label {
    font-size: 10px;
    color: rgba(255, 255, 255, 0.4);
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  .heatmap-container {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .heatmap-grid {
    display: grid;
    grid-template-columns: repeat(10, var(--cell-size));
    gap: var(--cell-gap);
  }

  .heatmap-cell {
    width: var(--cell-size);
    height: var(--cell-size);
    border-radius: 2px;
    transition: background 0.2s ease;
    cursor: pointer;
  }

  .heatmap-cell:hover {
    transform: scale(1.1);
    z-index: 1;
  }

  .symbol-labels {
    display: flex;
    gap: var(--cell-gap);
    padding-left: 0;
  }

  .symbol-label {
    width: var(--cell-size);
    font-size: 8px;
    color: rgba(255, 255, 255, 0.3);
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    text-align: center;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .color-legend {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-top: 4px;
  }

  .legend-label {
    font-size: 9px;
    color: rgba(255, 255, 255, 0.4);
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }

  .legend-gradient {
    flex: 1;
    height: 4px;
    border-radius: 2px;
    background: linear-gradient(
      90deg,
      rgba(0, 212, 255, 0.7) 0%,
      rgba(255, 255, 255, 0.15) 50%,
      rgba(255, 59, 59, 0.8) 100%
    );
  }

  /* Eigenvalues Section */
  .eigenvalues-section {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .section-label {
    font-size: 10px;
    color: rgba(255, 255, 255, 0.4);
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  .eigenvalues-list {
    display: flex;
    gap: 8px;
  }

  .eigenvalue-item {
    display: flex;
    align-items: center;
    gap: 4px;
    padding: 4px 8px;
    background: rgba(0, 0, 0, 0.2);
    border-radius: 3px;
    font-size: 11px;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }

  .ev-index {
    color: rgba(255, 255, 255, 0.3);
    font-size: 9px;
  }

  .ev-value {
    color: rgba(255, 255, 255, 0.8);
  }

  .ev-badge {
    padding: 2px 4px;
    background: rgba(0, 212, 255, 0.2);
    border-radius: 2px;
    color: #00d4ff;
    font-size: 8px;
    text-transform: uppercase;
  }
</style>
