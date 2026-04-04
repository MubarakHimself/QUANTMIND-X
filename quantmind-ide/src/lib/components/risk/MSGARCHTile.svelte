<script lang="ts">
  /**
   * MSGARCHTile - MS-GARCH Volatility Regime Sensor Visualization
   *
   * Displays:
   * - Current volatility regime state (LOW_VOL / HIGH_VOL)
   * - Conditional volatility forecast (sigma)
   * - Transition probabilities
   * - Alert state
   */

  import PhysicsSensorTile from './PhysicsSensorTile.svelte';
  import { TrendingUp } from 'lucide-svelte';
  import type { PhysicsMSGARCHData } from '$lib/stores/risk';

  export let data: PhysicsMSGARCHData | null = null;
  export let isLoading: boolean = false;

  $: alertMessage = data ? getAlertMessage(data.alert, data.vol_state) : '';

  function getAlertMessage(alert: string, volState: string | null): string {
    if (alert === 'critical') return 'High volatility alert';
    if (alert === 'warning') return `Vol regime: ${volState ?? 'UNKNOWN'}`;
    return '';
  }

  $: volName = data?.vol_state ?? 'Unknown';
  $: volColor = getVolColor(data?.vol_state);

  function getVolColor(state: string | null | undefined): string {
    if (!state) return 'rgba(255, 255, 255, 0.5)';
    const upper = state.toUpperCase();
    if (upper.includes('LOW')) return '#00ff88';
    if (upper.includes('HIGH')) return '#ff6b6b';
    if (upper.includes('MED')) return '#ffb700';
    return 'rgba(255, 255, 255, 0.7)';
  }

  $: sigmaDisplay = data?.sigma_forecast != null
    ? (data.sigma_forecast * 100).toFixed(4) + '%'
    : 'N/A';

  $: regimeDisplay = data?.regime_type ?? 'UNKNOWN';

  $: confDisplay = data?.confidence != null
    ? (data.confidence * 100).toFixed(0) + '%'
    : 'N/A';

  $: transEntries = data?.transition_probs
    ? Object.entries(data.transition_probs)
    : [];

  $: maxProb = transEntries.length > 0
    ? Math.max(...transEntries.map(([_, v]) => v))
    : 1;
</script>

<PhysicsSensorTile
  title="MS-GARCH Vol"
  alert={data?.alert ?? 'normal'}
  alertMessage={alertMessage}
  {isLoading}
>
  {#if data}
    <div class="msgarch-data">
      <!-- Vol State -->
      <div class="state-row">
        <span class="data-label">Vol Regime</span>
        <div class="state-value" style="color: {volColor}">
          <TrendingUp size={14} />
          <span>{volName}</span>
        </div>
      </div>

      <!-- Sigma Forecast -->
      <div class="state-row">
        <span class="data-label">σ Forecast</span>
        <span class="sigma-value">{sigmaDisplay}</span>
      </div>

      <!-- Regime Type -->
      <div class="state-row">
        <span class="data-label">Regime</span>
        <span class="regime-value">{regimeDisplay}</span>
      </div>

      <!-- Confidence -->
      <div class="state-row">
        <span class="data-label">Confidence</span>
        <span class="conf-value">{confDisplay}</span>
      </div>

      <!-- Transition Probabilities -->
      {#if transEntries.length > 0}
        <div class="probabilities">
          <span class="data-label">Regime Probs</span>
          <div class="histogram">
            {#each transEntries as [state, prob]}
              <div class="hist-bar">
                <div class="bar-fill" style="height: {(prob / maxProb) * 100}%; background: {state.includes('LOW') ? 'linear-gradient(180deg, #00ff88 0%, rgba(0,255,136,0.3) 100%)' : 'linear-gradient(180deg, #ff6b6b 0%, rgba(255,107,107,0.3) 100%)'}"></div>
                <span class="bar-label">{state.slice(0,4)}</span>
                <span class="bar-value">{(prob * 100).toFixed(0)}%</span>
              </div>
            {/each}
          </div>
        </div>
      {/if}

      <!-- Model Version -->
      {#if data.model_version}
        <div class="model-version">
          <span class="version-label">Model:</span>
          <span class="version-value">{data.model_version}</span>
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
  .msgarch-data {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .state-row {
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

  .state-value {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 14px;
    font-weight: 600;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }

  .sigma-value {
    font-size: 13px;
    color: rgba(255, 255, 255, 0.8);
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }

  .regime-value {
    font-size: 12px;
    color: rgba(255, 255, 255, 0.7);
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }

  .conf-value {
    font-size: 12px;
    color: rgba(255, 255, 255, 0.7);
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }

  .probabilities {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .histogram {
    display: flex;
    align-items: flex-end;
    gap: 4px;
    height: 40px;
    padding-top: 6px;
  }

  .hist-bar {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    height: 100%;
    position: relative;
  }

  .bar-fill {
    position: absolute;
    bottom: 18px;
    left: 0;
    right: 0;
    border-radius: 2px 2px 0 0;
    transition: height 0.3s ease;
  }

  .bar-label {
    position: absolute;
    bottom: 6px;
    font-size: 8px;
    color: rgba(255, 255, 255, 0.4);
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }

  .bar-value {
    position: absolute;
    bottom: 0;
    font-size: 8px;
    color: rgba(255, 255, 255, 0.6);
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }

  .model-version {
    display: flex;
    gap: 6px;
    align-items: center;
    padding-top: 4px;
    border-top: 1px solid rgba(255,255,255,0.05);
  }

  .version-label {
    font-size: 9px;
    color: rgba(255, 255, 255, 0.3);
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    text-transform: uppercase;
  }

  .version-value {
    font-size: 9px;
    color: rgba(255, 255, 255, 0.4);
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
