<script lang="ts">
  /**
   * LyapunovTile - Lyapunov Exponent Sensor Visualization
   *
   * Displays:
   * - Exponent value (chaos metric)
   * - Divergence rate
   * - Alert state
   */

  import PhysicsSensorTile from './PhysicsSensorTile.svelte';
  import type { PhysicsLyapunovData } from '$lib/stores/risk';

  export let data: PhysicsLyapunovData | null = null;
  export let isLoading: boolean = false;

  // Get alert message based on state
  $: alertMessage = data ? getAlertMessage(data.alert, data.exponent_value) : '';

  function getAlertMessage(alert: string, exponent: number): string {
    if (alert === 'critical') {
      return `High chaos detected - exponent ${exponent.toFixed(2)}`;
    } else if (alert === 'warning') {
      return `Moderate chaos - exponent ${exponent.toFixed(2)}`;
    }
    return '';
  }

  // Determine chaos level description
  $: chaosLevel = data ? getChaosLevel(data.exponent_value) : 'unknown';

  function getChaosLevel(exponent: number): string {
    if (exponent < 0.2) return 'Stable';
    if (exponent < 0.5) return 'Transitional';
    return 'Chaotic';
  }

  // Get bar color based on exponent
  $: barColor = data ? getBarColor(data.exponent_value) : '#00d4ff';

  function getBarColor(exponent: number): string {
    if (exponent < 0.2) return '#00d4ff'; // Stable - cyan
    if (exponent < 0.5) return '#ffb700'; // Transitional - warning yellow
    return '#ff3b3b'; // Chaotic - critical red
  }

  // Calculate bar percentage (0-1 range mapped to 0-100)
  $: barPercent = data ? Math.min(data.exponent_value * 100, 100) : 0;
</script>

<PhysicsSensorTile
  title="Lyapunov Exponent"
  alert={data?.alert ?? 'normal'}
  alertMessage={alertMessage}
  {isLoading}
>
  {#if data}
    <div class="lyapunov-data">
      <!-- Exponent Value -->
      <div class="data-row">
        <span class="data-label">Exponent</span>
        <span class="data-value" style="color: {barColor}">
          {data.exponent_value.toFixed(3)}
        </span>
      </div>

      <!-- Chaos Level Badge -->
      <div class="chaos-badge" style="background: {barColor}20; color: {barColor}">
        {chaosLevel}
      </div>

      <!-- Exponent Bar (Scalar visualization) -->
      <div class="exponent-bar">
        <div class="bar-track">
          <div class="bar-fill" style="width: {barPercent}%; background: {barColor}"></div>
          <!-- Threshold markers -->
          <div class="threshold-marker" style="left: 20%"></div>
          <div class="threshold-marker warning" style="left: 50%"></div>
        </div>
        <div class="bar-labels">
          <span>0</span>
          <span>0.2</span>
          <span>0.5</span>
          <span>1.0</span>
        </div>
      </div>

      <!-- Divergence Rate (if available) -->
      {#if data.divergence_rate !== null && data.divergence_rate !== undefined}
        <div class="data-row small">
          <span class="data-label">Divergence Rate</span>
          <span class="data-value secondary">
            {data.divergence_rate.toFixed(4)}
          </span>
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
  .lyapunov-data {
    display: flex;
    flex-direction: column;
    gap: 10px;
  }

  .data-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .data-row.small {
    font-size: 11px;
  }

  .data-label {
    font-size: 11px;
    color: rgba(255, 255, 255, 0.5);
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  .data-value {
    font-size: 18px;
    font-weight: 600;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }

  .data-value.secondary {
    font-size: 13px;
    color: rgba(255, 255, 255, 0.7);
  }

  .chaos-badge {
    align-self: flex-start;
    padding: 3px 8px;
    border-radius: 4px;
    font-size: 10px;
    font-weight: 600;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  .exponent-bar {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .bar-track {
    height: 10px;
    background: rgba(255, 255, 255, 0.1);
    border-radius: 5px;
    position: relative;
    overflow: hidden;
  }

  .bar-fill {
    height: 100%;
    border-radius: 5px;
    transition: width 0.3s ease, background 0.3s ease;
  }

  .threshold-marker {
    position: absolute;
    top: 0;
    bottom: 0;
    width: 1px;
    background: rgba(255, 255, 255, 0.2);
  }

  .threshold-marker.warning {
    background: rgba(255, 255, 255, 0.4);
  }

  .bar-labels {
    display: flex;
    justify-content: space-between;
    font-size: 9px;
    color: rgba(255, 255, 255, 0.3);
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
