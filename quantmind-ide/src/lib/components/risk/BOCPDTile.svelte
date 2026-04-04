<script lang="ts">
  /**
   * BOCPDTile - Bayesian Online Changepoint Detection Visualization
   *
   * Displays:
   * - Changepoint probability
   * - Is changepoint flagged
   * - Current run length
   * - Regime type
   * - Hazard lambda
   * - Alert state
   */

  import PhysicsSensorTile from './PhysicsSensorTile.svelte';
  import { Radar } from 'lucide-svelte';
  import type { PhysicsBOCPDData } from '$lib/stores/risk';

  export let data: PhysicsBOCPDData | null = null;
  export let isLoading: boolean = false;

  $: alertMessage = data ? getAlertMessage(data.alert, data.is_changepoint) : '';

  function getAlertMessage(alert: string, isChangepoint: boolean): string {
    if (isChangepoint) return 'Changepoint detected — regime shift';
    if (alert === 'critical') return 'High changepoint probability';
    return '';
  }

  $: regimeName = data?.regime_type ?? 'Unknown';
  $: regimeColor = getRegimeColor(data?.regime_type);

  function getRegimeColor(regime: string | null | undefined): string {
    if (!regime) return 'rgba(255, 255, 255, 0.5)';
    const upper = regime.toUpperCase();
    if (upper.includes('STABLE')) return '#00ff88';
    if (upper.includes('VOLATILE')) return '#ff6b6b';
    if (upper.includes('TRANSITION')) return '#ffb700';
    return 'rgba(255, 255, 255, 0.7)';
  }

  $: changepointProbDisplay = data?.changepoint_prob != null
    ? (data.changepoint_prob * 100).toFixed(1) + '%'
    : 'N/A';

  $: runLengthDisplay = data?.current_run_length ?? 'N/A';

  $: hazardDisplay = data?.hazard_lambda != null
    ? data.hazard_lambda.toFixed(3)
    : 'N/A';

  $: confDisplay = data?.confidence != null
    ? (data.confidence * 100).toFixed(0) + '%'
    : 'N/A';

  $: probBarHeight = data?.changepoint_prob != null
    ? Math.min(data.changepoint_prob * 100, 100)
    : 0;
</script>

<PhysicsSensorTile
  title="BOCPD"
  alert={data?.alert ?? 'normal'}
  alertMessage={alertMessage}
  {isLoading}
>
  {#if data}
    <div class="bocpd-data">
      <!-- Regime State -->
      <div class="state-row">
        <span class="data-label">Regime</span>
        <div class="state-value" style="color: {regimeColor}">
          <Radar size={14} />
          <span>{regimeName}</span>
        </div>
      </div>

      <!-- Changepoint Probability Bar -->
      <div class="prob-section">
        <div class="prob-header">
          <span class="data-label">Changepoint Prob</span>
          <span class="prob-value" class:high={data.changepoint_prob > 0.5}>
            {changepointProbDisplay}
          </span>
        </div>
        <div class="prob-bar-track">
          <div
            class="prob-bar-fill"
            class:high={data.changepoint_prob > 0.5}
            style="height: {probBarHeight}%"
          ></div>
        </div>
      </div>

      <!-- Run Length -->
      <div class="state-row">
        <span class="data-label">Run Length</span>
        <span class="run-value">{runLengthDisplay}</span>
      </div>

      <!-- Confidence -->
      <div class="state-row">
        <span class="data-label">Confidence</span>
        <span class="conf-value">{confDisplay}</span>
      </div>

      <!-- Hazard Lambda -->
      <div class="state-row">
        <span class="data-label">Hazard λ</span>
        <span class="hazard-value">{hazardDisplay}</span>
      </div>

      <!-- Changepoint Badge -->
      {#if data.is_changepoint}
        <div class="changepoint-badge">
          <span class="cp-dot"></span>
          <span>Changepoint Detected</span>
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
  .bocpd-data {
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

  .prob-section {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .prob-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .prob-value {
    font-size: 12px;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    color: rgba(255, 255, 255, 0.7);
  }

  .prob-value.high {
    color: #ff6b6b;
    font-weight: 600;
  }

  .prob-bar-track {
    height: 8px;
    background: rgba(255, 255, 255, 0.1);
    border-radius: 4px;
    overflow: hidden;
    position: relative;
  }

  .prob-bar-fill {
    position: absolute;
    bottom: 0;
    left: 0;
    right: 0;
    background: linear-gradient(180deg, #00d4ff 0%, rgba(0, 212, 255, 0.4) 100%);
    border-radius: 4px;
    transition: height 0.3s ease;
  }

  .prob-bar-fill.high {
    background: linear-gradient(180deg, #ff6b6b 0%, rgba(255, 107, 107, 0.4) 100%);
  }

  .run-value {
    font-size: 13px;
    color: rgba(255, 255, 255, 0.8);
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }

  .conf-value {
    font-size: 12px;
    color: rgba(255, 255, 255, 0.7);
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }

  .hazard-value {
    font-size: 12px;
    color: rgba(255, 255, 255, 0.6);
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }

  .changepoint-badge {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 4px 8px;
    background: rgba(255, 107, 107, 0.15);
    border: 1px solid rgba(255, 107, 107, 0.3);
    border-radius: 4px;
    font-size: 10px;
    color: #ff6b6b;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    width: fit-content;
  }

  .cp-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #ff6b6b;
    animation: pulse 1.5s infinite;
  }

  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
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
