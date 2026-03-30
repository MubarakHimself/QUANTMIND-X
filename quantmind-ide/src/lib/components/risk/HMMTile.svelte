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
  import { apiFetch } from '$lib/api';
  import type { PhysicsHMMData } from '$lib/stores/risk';

  let { data = null, isLoading = false } = $props<{
    data: PhysicsHMMData | null;
    isLoading: boolean;
  }>();

  let togglingShadow = $state(false);

  async function toggleShadowMode() {
    if (togglingShadow) return;
    togglingShadow = true;
    try {
      await apiFetch('/api/hmm/shadow-mode/toggle?enabled=' + String(!data?.is_shadow_mode), {
        method: 'POST'
      });
      // Physics store polls every 5s — no manual refresh needed
    } catch (e) {
      console.error('Failed to toggle shadow mode:', e);
    } finally {
      togglingShadow = false;
    }
  }

  // Get alert message based on state
  let alertMessage = $derived(data ? getAlertMessage(data.alert, data.is_shadow_mode) : '');

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
  let regimeName = $derived(data?.current_state ?? 'Unknown');

  // Get regime color
  let regimeColor = $derived(getRegimeColor(data?.current_state));

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
  let transitionEntries = $derived(
    data?.transition_probabilities
      ? Object.entries(data.transition_probabilities)
      : []
  );

  // Find max probability for scaling
  let maxProb = $derived(
    transitionEntries.length > 0
      ? Math.max(...transitionEntries.map(([_, v]) => v))
      : 1
  );
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

      <!-- Shadow Mode Toggle -->
      <div class="shadow-row">
        <span class="data-label">Shadow Mode</span>
        <button
          class="shadow-toggle"
          class:active={data.is_shadow_mode}
          class:spinning={togglingShadow}
          onclick={toggleShadowMode}
          title={data.is_shadow_mode ? 'Disable shadow mode' : 'Enable shadow mode'}
        >
          <span class="toggle-dot"></span>
          <span>{data.is_shadow_mode ? 'ON' : 'OFF'}</span>
        </button>
      </div>

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

  .shadow-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 6px 0;
    border-bottom: 1px solid rgba(255, 255, 255, 0.04);
  }

  .shadow-toggle {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 10px;
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 20px;
    cursor: pointer;
    font-size: 10px;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    font-weight: 600;
    letter-spacing: 0.5px;
    color: rgba(255, 255, 255, 0.4);
    transition: all 0.2s ease;
    text-transform: uppercase;
  }

  .shadow-toggle:hover {
    border-color: rgba(255, 183, 0, 0.4);
    background: rgba(255, 183, 0, 0.08);
  }

  .shadow-toggle.active {
    background: rgba(255, 183, 0, 0.15);
    border-color: rgba(255, 183, 0, 0.4);
    color: #ffb700;
  }

  .toggle-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: rgba(255, 255, 255, 0.3);
    transition: background 0.2s ease;
  }

  .shadow-toggle.active .toggle-dot {
    background: #ffb700;
    box-shadow: 0 0 6px rgba(255, 183, 0, 0.8);
  }

  .shadow-toggle.spinning {
    opacity: 0.6;
    cursor: not-allowed;
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
