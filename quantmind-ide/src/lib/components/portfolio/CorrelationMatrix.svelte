<script lang="ts">
  /**
   * CorrelationMatrix - Strategy-to-Strategy Return Correlations Heatmap
   *
   * Story 9.4: Portfolio Canvas — Attribution, Correlation Matrix & Performance
   * AC #2: NxN heatmap of strategy-to-strategy return correlations
   * AC #3: Tooltip on hover showing strategy A, strategy B, coefficient, data period
   */
  import { onMount } from 'svelte';
  import { portfolioStore, correlationMatrix, portfolioLoading } from '$lib/stores/portfolio';

  let hoveredCell = $state<{ strategy_a: string; strategy_b: string; correlation: number; data_period: string } | null>(null);
  let mouseX = $state(0);
  let mouseY = $state(0);

  onMount(async () => {
    await portfolioStore.fetchCorrelation();
  });

  // Get unique strategy names from correlation matrix
  let strategies = $derived([...new Set($correlationMatrix.flatMap(c => [c.strategy_a, c.strategy_b]))].sort());

  // Build 2D matrix for display
  let matrix = $derived(strategies.map(strategyA =>
    strategies.map(strategyB => {
      const cell = $correlationMatrix.find(c =>
        (c.strategy_a === strategyA && c.strategy_b === strategyB) ||
        (c.strategy_a === strategyB && c.strategy_b === strategyA)
      );
      return cell?.correlation ?? (strategyA === strategyB ? 1 : 0);
    })
  ));

  function getCellColor(correlation: number): string {
    // Color scale: blue (negative) -> white (zero) -> red (positive)
    // Red highlight for |correlation| >= 0.7
    if (Math.abs(correlation) >= 0.7) {
      return correlation > 0 ? '#ff3b3b' : '#3b5bff';
    }

    if (correlation > 0) {
      const intensity = Math.min(correlation, 1);
      // Interpolate from white to amber
      return `rgba(245, 158, 11, ${intensity * 0.6})`;
    } else {
      const intensity = Math.min(Math.abs(correlation), 1);
      return `rgba(0, 212, 255, ${intensity * 0.4})`;
    }
  }

  function handleMouseEnter(event: MouseEvent, strategyA: string, strategyB: string) {
    const cell = $correlationMatrix.find(c =>
      (c.strategy_a === strategyA && c.strategy_b === strategyB) ||
      (c.strategy_a === strategyB && c.strategy_b === strategyA)
    );
    if (cell) {
      hoveredCell = cell;
      mouseX = event.clientX;
      mouseY = event.clientY;
    }
  }

  function handleMouseMove(event: MouseEvent) {
    mouseX = event.clientX;
    mouseY = event.clientY;
  }

  function handleMouseLeave() {
    hoveredCell = null;
  }

  function formatCorrelation(value: number): string {
    return value.toFixed(2);
  }
</script>

<div class="correlation-panel">
  <header class="panel-header">
    <h2>Correlation Matrix</h2>
    <span class="subtitle">Strategy-to-strategy return correlations (NxN heatmap)</span>
    <div class="legend">
      <span class="legend-item">
        <span class="legend-color" style="background: rgba(0, 212, 255, 0.4)"></span>
        Negative
      </span>
      <span class="legend-item">
        <span class="legend-color" style="background: transparent"></span>
        Zero
      </span>
      <span class="legend-item">
        <span class="legend-color" style="background: rgba(245, 158, 11, 0.6)"></span>
        Positive
      </span>
      <span class="legend-item">
        <span class="legend-color" style="background: #ff3b3b"></span>
        |r| ≥ 0.7
      </span>
    </div>
  </header>

  {#if $portfolioLoading}
    <div class="loading">Loading correlation data...</div>
  {:else if strategies.length === 0}
    <div class="empty">No correlation data available</div>
  {:else}
    <div class="matrix-container">
      <table class="correlation-table">
        <thead>
          <tr>
            <th class="corner-cell"></th>
            {#each strategies as strategy}
              <th class="strategy-header" title={strategy}>
                <span class="strategy-label">{strategy.substring(0, 12)}{strategy.length > 12 ? '...' : ''}</span>
              </th>
            {/each}
          </tr>
        </thead>
        <tbody>
          {#each strategies as strategyA, i}
            <tr>
              <th class="strategy-header" title={strategyA}>
                <span class="strategy-label">{strategyA.substring(0, 12)}{strategyA.length > 12 ? '...' : ''}</span>
              </th>
              {#each matrix[i] as correlation, j}
                <td
                  class="correlation-cell"
                  style="background: {getCellColor(correlation)}"
                  onmouseenter={(e) => handleMouseEnter(e, strategyA, strategies[j])}
                  onmousemove={handleMouseMove}
                  onmouseleave={handleMouseLeave}
                  role="gridcell"
                >
                  <span class="correlation-value">{formatCorrelation(correlation)}</span>
                </td>
              {/each}
            </tr>
          {/each}
        </tbody>
      </table>
    </div>
  {/if}

  <!-- Tooltip -->
  {#if hoveredCell}
    <div
      class="tooltip"
      style="left: {mouseX + 15}px; top: {mouseY + 15}px"
    >
      <div class="tooltip-row">
        <span class="tooltip-label">Strategy A:</span>
        <span class="tooltip-value">{hoveredCell.strategy_a}</span>
      </div>
      <div class="tooltip-row">
        <span class="tooltip-label">Strategy B:</span>
        <span class="tooltip-value">{hoveredCell.strategy_b}</span>
      </div>
      <div class="tooltip-row">
        <span class="tooltip-label">Correlation:</span>
        <span class="tooltip-value" class:highlight={Math.abs(hoveredCell.correlation) >= 0.7}>
          {formatCorrelation(hoveredCell.correlation)}
        </span>
      </div>
      <div class="tooltip-row">
        <span class="tooltip-label">Data Period:</span>
        <span class="tooltip-value">{hoveredCell.data_period}</span>
      </div>
    </div>
  {/if}
</div>

<style>
  .correlation-panel {
    display: flex;
    flex-direction: column;
    gap: 16px;
    padding: 16px;
    position: relative;
  }

  .panel-header {
    display: flex;
    flex-direction: column;
    gap: 8px;
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

  .legend {
    display: flex;
    gap: 16px;
    margin-top: 8px;
  }

  .legend-item {
    display: flex;
    align-items: center;
    gap: 6px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: rgba(255, 255, 255, 0.6);
  }

  .legend-color {
    width: 16px;
    height: 16px;
    border-radius: 3px;
    border: 1px solid rgba(255, 255, 255, 0.1);
  }

  .loading, .empty {
    font-family: 'JetBrains Mono', monospace;
    font-size: 14px;
    color: rgba(255, 255, 255, 0.4);
    text-align: center;
    padding: 40px;
  }

  .matrix-container {
    overflow-x: auto;
  }

  .correlation-table {
    border-collapse: collapse;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
  }

  .corner-cell, .strategy-header {
    padding: 8px;
    background: rgba(8, 13, 20, 0.6);
    border: 1px solid rgba(245, 158, 11, 0.1);
    color: rgba(245, 158, 11, 0.8);
    font-weight: 500;
    white-space: nowrap;
    text-align: center;
    min-width: 80px;
  }

  .strategy-header {
    writing-mode: vertical-rl;
    transform: rotate(180deg);
    padding: 12px 4px;
    height: 120px;
  }

  .corner-cell {
    width: 80px;
  }

  .strategy-label {
    display: block;
    max-width: 60px;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .correlation-cell {
    padding: 8px;
    text-align: center;
    border: 1px solid rgba(255, 255, 255, 0.05);
    cursor: pointer;
    transition: transform 0.1s ease;
    min-width: 60px;
  }

  .correlation-cell:hover {
    transform: scale(1.05);
    z-index: 1;
    box-shadow: 0 0 8px rgba(245, 158, 11, 0.3);
  }

  .correlation-value {
    color: #fff;
    font-variant-numeric: tabular-nums;
    text-shadow: 0 1px 2px rgba(0, 0, 0, 0.5);
  }

  /* Tooltip */
  .tooltip {
    position: fixed;
    background: rgba(8, 13, 20, 0.95);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(245, 158, 11, 0.3);
    border-radius: 8px;
    padding: 12px;
    z-index: 1000;
    pointer-events: none;
    min-width: 200px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
  }

  .tooltip-row {
    display: flex;
    justify-content: space-between;
    gap: 16px;
    margin-bottom: 6px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
  }

  .tooltip-row:last-child {
    margin-bottom: 0;
  }

  .tooltip-label {
    color: rgba(255, 255, 255, 0.5);
  }

  .tooltip-value {
    color: #fff;
    font-weight: 500;
  }

  .tooltip-value.highlight {
    color: #ff3b3b;
  }
</style>