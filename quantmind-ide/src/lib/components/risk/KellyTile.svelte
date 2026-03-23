<script lang="ts">
  /**
   * KellyTile - Kelly Engine Visualization
   *
   * Displays:
   * - Current Kelly fraction
   * - Physics multiplier
   * - House of money state
   */

  import PhysicsSensorTile from './PhysicsSensorTile.svelte';
  import { Calculator, TrendingUp, TrendingDown } from 'lucide-svelte';
  import type { PhysicsKellyData } from '$lib/stores/risk';

  export let data: PhysicsKellyData | null = null;
  export let isLoading: boolean = false;

  // Get alert message based on state
  $: alertMessage = data ? getAlertMessage(data) : '';

  function getAlertMessage(kelly: PhysicsKellyData): string {
    if (kelly.house_of_money) {
      return 'Favorable market conditions - Kelly boosted';
    }
    if (kelly.multiplier < 0.7) {
      return 'Unfavorable conditions - Kelly reduced';
    }
    return '';
  }

  // Determine if we should show alert based on multiplier
  $: alertState = data ? getAlertState(data) : 'normal';

  function getAlertState(kelly: PhysicsKellyData): 'normal' | 'warning' | 'critical' {
    if (kelly.house_of_money) return 'normal';
    if (kelly.multiplier < 0.5) return 'critical';
    if (kelly.multiplier < 0.8) return 'warning';
    return 'normal';
  }

  // Get house of money color
  $: homColor = data?.house_of_money ? '#00ff88' : 'rgba(255, 255, 255, 0.5)';

  // Format fraction as percentage
  $: fractionPercent = data ? (data.fraction * 100).toFixed(1) : '0';

  // Format multiplier
  $: multiplierDisplay = data ? data.multiplier.toFixed(2) : '0.00';
</script>

<PhysicsSensorTile
  title="Kelly Engine"
  alert={alertState}
  alertMessage={alertMessage}
  {isLoading}
>
  {#if data}
    <div class="kelly-data">
      <!-- Kelly Fraction -->
      <div class="fraction-display">
        <div class="fraction-icon">
          <Calculator size={16} />
        </div>
        <div class="fraction-info">
          <span class="fraction-label">Kelly Fraction</span>
          <span class="fraction-value">{fractionPercent}%</span>
        </div>
      </div>

      <!-- Physics Multiplier -->
      <div class="multiplier-display">
        <span class="data-label">Physics Multiplier</span>
        <div class="multiplier-value">
          {#if data.multiplier >= 1}
            <TrendingUp size={14} color="#00ff88" />
          {:else}
            <TrendingDown size={14} color="#ff6b6b" />
          {/if}
          <span class:positive={data.multiplier >= 1} class:negative={data.multiplier < 1}>
            {multiplierDisplay}x
          </span>
        </div>
      </div>

      <!-- House of Money State -->
      <div class="hom-state" style="border-color: {homColor}">
        <div class="hom-indicator" style="background: {homColor}"></div>
        <span class="hom-text" style="color: {homColor}">
          {data.house_of_money ? 'House of Money' : 'Standard Mode'}
        </span>
      </div>

      <!-- Kelly Setting Info -->
      <div class="setting-info">
        <span class="data-label">Configured</span>
        <span class="setting-value">{(data.kelly_fraction_setting * 100).toFixed(0)}% Kelly</span>
      </div>
    </div>
  {:else}
    <div class="no-data">
      <span>No data available</span>
    </div>
  {/if}
</PhysicsSensorTile>

<style>
  .kelly-data {
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  .data-label {
    font-size: 11px;
    color: rgba(255, 255, 255, 0.5);
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  .fraction-display {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 10px;
    background: rgba(0, 0, 0, 0.2);
    border-radius: 6px;
  }

  .fraction-icon {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 32px;
    height: 32px;
    background: rgba(0, 212, 255, 0.15);
    border-radius: 6px;
    color: #00d4ff;
  }

  .fraction-info {
    display: flex;
    flex-direction: column;
  }

  .fraction-label {
    font-size: 10px;
    color: rgba(255, 255, 255, 0.5);
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    text-transform: uppercase;
  }

  .fraction-value {
    font-size: 20px;
    font-weight: 600;
    color: #00d4ff;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }

  .multiplier-display {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .multiplier-value {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 16px;
    font-weight: 600;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }

  .multiplier-value span.positive {
    color: #00ff88;
  }

  .multiplier-value span.negative {
    color: #ff6b6b;
  }

  .hom-state {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 10px;
    background: rgba(0, 0, 0, 0.15);
    border: 1px solid;
    border-radius: 6px;
  }

  .hom-indicator {
    width: 8px;
    height: 8px;
    border-radius: 50%;
  }

  .hom-text {
    font-size: 12px;
    font-weight: 500;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  .setting-info {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding-top: 4px;
    border-top: 1px solid rgba(255, 255, 255, 0.1);
  }

  .setting-value {
    font-size: 12px;
    color: rgba(255, 255, 255, 0.7);
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
