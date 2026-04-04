<script lang="ts">
  /**
   * EnsembleTile - Ensemble Voter Visualization
   *
   * Displays:
   * - Ensemble regime type (final voted output)
   * - Confidence score
   * - Transition indicator
   * - Sigma forecast
   * - Ensemble agreement score
   * - Model weights used
   * - Model count
   * - Alert state
   */

  import PhysicsSensorTile from './PhysicsSensorTile.svelte';
  import { Network } from 'lucide-svelte';
  import type { PhysicsEnsembleData } from '$lib/stores/risk';

  export let data: PhysicsEnsembleData | null = null;
  export let isLoading: boolean = false;

  $: alertMessage = data ? getAlertMessage(data.alert, data.is_transition) : '';

  function getAlertMessage(alert: string, isTransition: boolean): string {
    if (isTransition) return 'Regime transition in progress';
    if (alert === 'critical') return 'Low ensemble agreement';
    return '';
  }

  $: regimeName = data?.regime_type ?? 'Unknown';
  $: regimeColor = getRegimeColor(data?.regime_type);

  function getRegimeColor(regime: string | null | undefined): string {
    if (!regime) return 'rgba(255, 255, 255, 0.5)';
    const upper = regime.toUpperCase();
    if (upper.includes('TREND') && upper.includes('BULL')) return '#00ff88';
    if (upper.includes('TREND') && upper.includes('BEAR')) return '#ff6b6b';
    if (upper.includes('RANGE') && upper.includes('STABLE')) return '#00d4ff';
    if (upper.includes('RANGE') && upper.includes('VOLATILE')) return '#ffb700';
    if (upper.includes('BREAKOUT')) return '#ff00ff';
    if (upper.includes('CHAOS')) return '#ff4500';
    return 'rgba(255, 255, 255, 0.7)';
  }

  $: confDisplay = data?.confidence != null
    ? (data.confidence * 100).toFixed(0) + '%'
    : 'N/A';

  $: sigmaDisplay = data?.sigma_forecast != null
    ? (data.sigma_forecast * 100).toFixed(4) + '%'
    : 'N/A';

  $: agreementDisplay = data?.ensemble_agreement != null
    ? (data.ensemble_agreement * 100).toFixed(0) + '%'
    : 'N/A';

  $: weightEntries = data?.weights_used
    ? Object.entries(data.weights_used)
    : [];

  $: sourceEntries = data?.sources
    ? Object.entries(data.sources)
    : [];
</script>

<PhysicsSensorTile
  title="Ensemble"
  alert={data?.alert ?? 'normal'}
  alertMessage={alertMessage}
  {isLoading}
>
  {#if data}
    <div class="ensemble-data">
      <!-- Regime State -->
      <div class="state-row">
        <span class="data-label">Voted Regime</span>
        <div class="state-value" style="color: {regimeColor}">
          <Network size={14} />
          <span>{regimeName}</span>
        </div>
      </div>

      <!-- Confidence -->
      <div class="state-row">
        <span class="data-label">Confidence</span>
        <span class="conf-value">{confDisplay}</span>
      </div>

      <!-- Sigma Forecast -->
      <div class="state-row">
        <span class="data-label">σ Forecast</span>
        <span class="sigma-value">{sigmaDisplay}</span>
      </div>

      <!-- Ensemble Agreement -->
      <div class="state-row">
        <span class="data-label">Agreement</span>
        <span class="agreement-value" class:low={data.ensemble_agreement < 0.7}>
          {agreementDisplay}
        </span>
      </div>

      <!-- Transition Badge -->
      {#if data.is_transition}
        <div class="transition-badge">
          <span class="trans-dot"></span>
          <span>Regime Transition</span>
        </div>
      {/if}

      <!-- Model Weights -->
      {#if weightEntries.length > 0}
        <div class="weights-section">
          <span class="data-label">Model Weights</span>
          <div class="weights-grid">
            {#each weightEntries as [model, weight]}
              <div class="weight-item">
                <span class="weight-model">{model.slice(0, 6)}</span>
                <span class="weight-value">{(weight * 100).toFixed(0)}%</span>
              </div>
            {/each}
          </div>
        </div>
      {/if}

      <!-- Model Count -->
      <div class="state-row">
        <span class="data-label">Models</span>
        <span class="count-value">{data.model_count}</span>
      </div>

      <!-- Sources / Individual Model Outputs -->
      {#if sourceEntries.length > 0}
        <div class="sources-section">
          <span class="data-label">Model Sources</span>
          <div class="sources-list">
            {#each sourceEntries as [model, output]}
              <div class="source-item">
                <span class="source-model">{model.slice(0, 8)}</span>
                <span class="source-regime">{String(output).slice(0, 12)}</span>
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
  .ensemble-data {
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

  .conf-value {
    font-size: 12px;
    color: rgba(255, 255, 255, 0.8);
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }

  .sigma-value {
    font-size: 12px;
    color: rgba(255, 255, 255, 0.7);
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }

  .agreement-value {
    font-size: 12px;
    color: rgba(255, 255, 255, 0.7);
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }

  .agreement-value.low {
    color: #ffb700;
  }

  .transition-badge {
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

  .trans-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #ffb700;
    animation: pulse 1.5s infinite;
  }

  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
  }

  .weights-section {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .weights-grid {
    display: flex;
    gap: 4px;
    flex-wrap: wrap;
  }

  .weight-item {
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 3px 6px;
    background: rgba(255, 255, 255, 0.05);
    border-radius: 3px;
    min-width: 36px;
  }

  .weight-model {
    font-size: 8px;
    color: rgba(255, 255, 255, 0.4);
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    text-transform: uppercase;
  }

  .weight-value {
    font-size: 10px;
    color: rgba(255, 255, 255, 0.7);
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }

  .count-value {
    font-size: 13px;
    color: rgba(255, 255, 255, 0.8);
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }

  .sources-section {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .sources-list {
    display: flex;
    flex-direction: column;
    gap: 2px;
  }

  .source-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 2px 4px;
    background: rgba(255, 255, 255, 0.03);
    border-radius: 2px;
  }

  .source-model {
    font-size: 9px;
    color: rgba(255, 255, 255, 0.4);
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }

  .source-regime {
    font-size: 9px;
    color: rgba(255, 255, 255, 0.6);
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
