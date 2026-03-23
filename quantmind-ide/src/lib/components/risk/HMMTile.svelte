<script lang="ts">
  /**
   * HMMTile - Hidden Markov Model Regime Sensor Visualization
   *
   * Displays:
   * - Current regime state
   * - Transition probabilities (as histogram/distribution)
   * - Shadow mode badge
   * - Alert state
   */

  import PhysicsSensorTile from './PhysicsSensorTile.svelte';
  import { Brain } from 'lucide-svelte';
  import type { PhysicsHMMData } from '$lib/stores/risk';

  export let data: PhysicsHMMData | null = null;
  export let isLoading: boolean = false;

  // Get alert message based on state
  $: alertMessage = data ? getAlertMessage(data.alert, data.is_shadow_mode) : '';

  function getAlertMessage(alert: string, shadowMode: boolean): string {
    if (shadowMode) {
      return 'HMM running in shadow mode - observing only';
    }
    if (alert === 'critical') {
      return 'Regime confidence low';
    }
    return '';
  }

  // Get regime display name
  $: regimeName = data?.current_state ?? 'Unknown';

  // Get regime color
  $: regimeColor = getRegimeColor(data?.current_state);

  function getRegimeColor(state: string | null | undefined): string {
    if (!state) return 'rgba(255, 255, 255, 0.5)';
    const upperState = state.toUpperCase();
    if (upperState.includes('TREND')) return '#00d4ff';
    if (upperState.includes('RANGE')) return '#00ff88';
    if (upperState.includes('BREAKOUT')) return '#ff6b6b';
    if (upperState.includes('CHAOS')) return '#ffb700';
    return 'rgba(255, 255, 255, 0.7)';
  }

  // Get transition probabilities as array for histogram
  $: transitionEntries = data?.transition_probabilities
    ? Object.entries(data.transition_probabilities)
    : [];

  // Find max probability for scaling
  $: maxProb = transitionEntries.length > 0
    ? Math.max(...transitionEntries.map(([_, v]) => v))
    : 1;
</script>

<PhysicsSensorTile
  title="HMM Regime"
  alert={data?.alert ?? 'normal'}
  alertMessage={alertMessage}
  {isLoading}
>
  {#if data}
    <div class="hmm-data">
      <!-- Current State -->
      <div class="state-row">
        <span class="data-label">Current State</span>
        <div class="state-value" style="color: {regimeColor}">
          <Brain size={14} />
          <span>{regimeName}</span>
        </div>
      </div>

      <!-- Shadow Mode Badge -->
      {#if data.is_shadow_mode}
        <div class="shadow-badge">
          <span class="shadow-dot"></span>
          <span>Shadow Mode</span>
        </div>
      {/if}

      <!-- Transition Probabilities (Histogram/Distribution) -->
      {#if transitionEntries.length > 0}
        <div class="probabilities">
          <span class="data-label">State Probabilities</span>
          <div class="histogram">
            {#each transitionEntries as [state, prob]}
              <div class="hist-bar">
                <div class="bar-fill" style="height: {(prob / maxProb) * 100}%"></div>
                <span class="bar-label">{state.slice(0, 4)}</span>
                <span class="bar-value">{(prob * 100).toFixed(0)}%</span>
              </div>
            {/each}
          </div>
        </div>
      {:else}
        <div class="no-probs">
          <span>No transition data</span>
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
  .hmm-data {
    display: flex;
    flex-direction: column;
    gap: 10px;
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

  .shadow-badge {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 4px 8px;
    background: rgba(255, 183, 0, 0.15);
    border: 1px solid rgba(255, 183, 0, 0.3);
    border-radius: 4px;
    font-size: 10px;
    color: #ffb700;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    width: fit-content;
  }

  .shadow-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #ffb700;
    animation: pulse 2s infinite;
  }

  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
  }

  .probabilities {
    display: flex;
    flex-direction: column;
    gap: 6px;
  }

  .histogram {
    display: flex;
    align-items: flex-end;
    gap: 4px;
    height: 50px;
    padding-top: 8px;
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
    bottom: 20px;
    left: 0;
    right: 0;
    background: linear-gradient(180deg, #00d4ff 0%, rgba(0, 212, 255, 0.3) 100%);
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

  .no-probs {
    padding: 8px;
    text-align: center;
    color: rgba(255, 255, 255, 0.3);
    font-size: 11px;
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
