<script lang="ts">
  import { onMount } from 'svelte';
  import TileCard from '$lib/components/shared/TileCard.svelte';
  import {
    ensembleData,
    hmmData,
    kellyData,
    physicsError,
    physicsLoading,
    physicsSensorStore,
    physicsUnavailableReason,
  } from '$lib/stores/risk';

  onMount(() => {
    physicsSensorStore.startPolling(5000);
    return () => {
      physicsSensorStore.stopPolling();
    };
  });
</script>

<TileCard title="Risk Physics" size="md">
  {#if $physicsLoading}
    <p class="state-copy">Loading physics state…</p>
  {:else if $physicsUnavailableReason}
    <div class="host-state">
      <p class="state-copy">Physics models are unavailable on this host.</p>
      <p class="support-copy">{$physicsUnavailableReason}</p>
    </div>
  {:else if $physicsError}
    <p class="state-copy error">{$physicsError}</p>
  {:else if $ensembleData || $hmmData || $kellyData}
    <div class="metrics">
      <div class="metric">
        <span class="section-label">Regime</span>
        <span class="metric-value">{$ensembleData?.regime_type ?? $hmmData?.current_state ?? 'Unknown'}</span>
      </div>
      <div class="metric">
        <span class="section-label">Confidence</span>
        <span class="financial-value metric-value">{(($ensembleData?.confidence ?? 0) * 100).toFixed(0)}%</span>
      </div>
      <div class="metric">
        <span class="section-label">Transition</span>
        <span class="metric-value">{$ensembleData?.is_transition ? 'Active' : 'Stable'}</span>
      </div>
      <div class="metric">
        <span class="section-label">Kelly</span>
        <span class="financial-value metric-value">{($kellyData?.multiplier ?? 0).toFixed(2)}x</span>
      </div>
    </div>
  {:else}
    <p class="state-copy">No physics data available yet.</p>
  {/if}
</TileCard>

<style>
  .metrics {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: var(--space-3);
  }

  .metric {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .metric-value {
    font-family: var(--font-data);
    font-size: var(--text-sm);
    color: var(--color-text-primary);
  }

  .host-state {
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
  }

  .state-copy,
  .support-copy {
    margin: 0;
    font-family: var(--font-ambient);
    font-size: var(--text-xs);
    line-height: 1.5;
    color: var(--color-text-muted);
  }

  .state-copy.error {
    color: var(--color-accent-red);
  }
</style>
