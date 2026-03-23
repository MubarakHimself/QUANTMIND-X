<script lang="ts">
  /**
   * IsingTile - Ising Model Sensor Visualization
   *
   * Displays:
   * - Magnetization value (-1 to 1)
   * - Correlation matrix (if available)
   * - Alert state
   */

  import PhysicsSensorTile from './PhysicsSensorTile.svelte';
  import type { PhysicsIsingData } from '$lib/stores/risk';

  export let data: PhysicsIsingData | null = null;
  export let isLoading: boolean = false;

  // Get alert message based on state
  $: alertMessage = data ? getAlertMessage(data.alert, data.magnetization) : '';

  function getAlertMessage(alert: string, magnetization: number): string {
    if (alert === 'critical') {
      return `System noise high - magnetization unclear (${magnetization.toFixed(2)})`;
    } else if (alert === 'warning') {
      return `Regime transition in progress (${magnetization.toFixed(2)})`;
    }
    return '';
  }

  // Determine if we have correlation matrix
  $: hasCorrelationMatrix = data?.correlation_matrix && Object.keys(data.correlation_matrix).length > 0;

  // Get bar percentage for visualization
  $: magnetBarPercent = data ? Math.abs(data.magnetization) * 100 : 0;
  $: magnetDirection = data && data.magnetization >= 0 ? 'positive' : 'negative';
</script>

<PhysicsSensorTile
  title="Ising Model"
  alert={data?.alert ?? 'normal'}
  alertMessage={alertMessage}
  {isLoading}
>
  {#if data}
    <div class="ising-data">
      <!-- Magnetization Value -->
      <div class="data-row">
        <span class="data-label">Magnetization</span>
        <span class="data-value" class:positive={magnetDirection === 'positive'} class:negative={magnetDirection === 'negative'}>
          {data.magnetization.toFixed(3)}
        </span>
      </div>

      <!-- Magnetization Bar -->
      <div class="magnet-bar">
        <div class="bar-track">
          <div
            class="bar-fill"
            class:positive={magnetDirection === 'positive'}
            class:negative={magnetDirection === 'negative'}
            style="width: {magnetBarPercent}%"
          ></div>
          <div class="bar-center"></div>
        </div>
        <div class="bar-labels">
          <span>-1</span>
          <span>0</span>
          <span>+1</span>
        </div>
      </div>

      <!-- Correlation Matrix Preview -->
      {#if hasCorrelationMatrix}
        <div class="correlation-preview">
          <span class="data-label">Correlation Matrix</span>
          <div class="matrix-grid">
            {#each Object.entries(data.correlation_matrix).slice(0, 4) as [key, value]}
              <div class="matrix-cell">
                <span class="cell-key">{key.slice(0, 3)}</span>
                <span class="cell-value">{value.toFixed(2)}</span>
              </div>
            {/each}
          </div>
        </div>
      {/if}
    </div>
  {:else}
    <div class="no-data">
      <span>No data available</span>
    </div>
  {/if}
</PhysicsSensorTile>

<style>
  .ising-data {
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  .data-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .data-label {
    font-size: 11px;
    color: rgba(255, 255, 255, 0.5);
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  .data-value {
    font-size: 16px;
    font-weight: 600;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }

  .data-value.positive {
    color: #00d4ff;
  }

  .data-value.negative {
    color: #ff6b6b;
  }

  .magnet-bar {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .bar-track {
    height: 8px;
    background: rgba(255, 255, 255, 0.1);
    border-radius: 4px;
    position: relative;
    overflow: hidden;
  }

  .bar-center {
    position: absolute;
    left: 50%;
    top: 0;
    bottom: 0;
    width: 1px;
    background: rgba(255, 255, 255, 0.3);
    transform: translateX(-50%);
  }

  .bar-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.3s ease;
  }

  .bar-fill.positive {
    background: linear-gradient(90deg, transparent, #00d4ff);
  }

  .bar-fill.negative {
    background: linear-gradient(90deg, #ff6b6b, transparent);
    position: absolute;
    right: 50%;
  }

  .bar-labels {
    display: flex;
    justify-content: space-between;
    font-size: 9px;
    color: rgba(255, 255, 255, 0.3);
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }

  .correlation-preview {
    display: flex;
    flex-direction: column;
    gap: 6px;
  }

  .matrix-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 4px;
  }

  .matrix-cell {
    background: rgba(0, 0, 0, 0.2);
    padding: 4px 6px;
    border-radius: 4px;
    display: flex;
    justify-content: space-between;
    font-size: 10px;
  }

  .cell-key {
    color: rgba(255, 255, 255, 0.5);
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }

  .cell-value {
    color: rgba(255, 255, 255, 0.8);
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }

  .no-data {
    display: flex;
    align-items: center;
    justify-content: center;
    flex: 1;
    color: rgba(255, 255, 255, 0.3);
    font-size: 12px;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }
</style>
