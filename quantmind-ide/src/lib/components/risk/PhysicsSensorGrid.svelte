<script lang="ts">
  /**
   * PhysicsSensorGrid - Grid Container for Physics Sensor Tiles
   *
   * Displays all physics sensor tiles in a responsive grid layout.
   * Each tile fetches independently to prevent cascade failures (NFR-R1).
   *
   * All 7 models: Ising, Lyapunov, HMM, MS-GARCH, BOCPD, Ensemble, Kelly
   */

  import { onMount, onDestroy } from 'svelte';
  import IsingTile from './IsingTile.svelte';
  import LyapunovTile from './LyapunovTile.svelte';
  import HMMTile from './HMMTile.svelte';
  import MSGARCHTile from './MSGARCHTile.svelte';
  import BOCPDTile from './BOCPDTile.svelte';
  import EnsembleTile from './EnsembleTile.svelte';
  import KellyTile from './KellyTile.svelte';
  import {
    physicsSensorStore,
    isingData,
    lyapunovData,
    hmmData,
    msgarchData,
    bocpdData,
    ensembleData,
    kellyData,
    physicsLoading,
    physicsError,
    physicsUnavailableReason,
    type PhysicsSensorData
  } from '$lib/stores/risk';

  // Subscribe to store values
  let ising = $state<PhysicsSensorData['ising'] | null>(null);
  let lyapunov = $state<PhysicsSensorData['lyapunov'] | null>(null);
  let hmm = $state<PhysicsSensorData['hmm'] | null>(null);
  let msgarch = $state<PhysicsSensorData['msgarch'] | null>(null);
  let bocpd = $state<PhysicsSensorData['bocpd'] | null>(null);
  let ensemble = $state<PhysicsSensorData['ensemble'] | null>(null);
  let kelly = $state<PhysicsSensorData['kelly'] | null>(null);
  let loading = $state(false);
  let error = $state<string | null>(null);
  let unavailableReason = $state<string | null>(null);

  // Subscribe to stores
  const unsubIsing = isingData.subscribe(v => ising = v);
  const unsubLyapunov = lyapunovData.subscribe(v => lyapunov = v);
  const unsubHMM = hmmData.subscribe(v => hmm = v);
  const unsubMSGARCH = msgarchData.subscribe(v => msgarch = v);
  const unsubBOCPD = bocpdData.subscribe(v => bocpd = v);
  const unsubEnsemble = ensembleData.subscribe(v => ensemble = v);
  const unsubKelly = kellyData.subscribe(v => kelly = v);
  const unsubLoading = physicsLoading.subscribe(v => loading = v);
  const unsubError = physicsError.subscribe(v => error = v);
  const unsubUnavailable = physicsUnavailableReason.subscribe(v => unavailableReason = v);

  onMount(() => {
    // Start polling at 5-second intervals
    physicsSensorStore.startPolling(5000);
  });

  onDestroy(() => {
    // Clean up polling
    physicsSensorStore.stopPolling();

    // Unsubscribe
    unsubIsing();
    unsubLyapunov();
    unsubHMM();
    unsubMSGARCH();
    unsubBOCPD();
    unsubEnsemble();
    unsubKelly();
    unsubLoading();
    unsubError();
    unsubUnavailable();
  });
</script>

<div class="sensor-grid">
  <!-- Ising Model Tile -->
  <div class="tile-wrapper">
    <IsingTile data={ising} isLoading={loading && !ising} />
  </div>

  <!-- Lyapunov Exponent Tile -->
  <div class="tile-wrapper">
    <LyapunovTile data={lyapunov} isLoading={loading && !lyapunov} />
  </div>

  <!-- HMM Regime Tile -->
  <div class="tile-wrapper">
    <HMMTile data={hmm} isLoading={loading && !hmm} />
  </div>

  <!-- MS-GARCH Volatility Tile -->
  <div class="tile-wrapper">
    <MSGARCHTile data={msgarch} isLoading={loading && !msgarch} />
  </div>

  <!-- BOCPD Changepoint Tile -->
  <div class="tile-wrapper">
    <BOCPDTile data={bocpd} isLoading={loading && !bocpd} />
  </div>

  <!-- Ensemble Voter Tile -->
  <div class="tile-wrapper">
    <EnsembleTile data={ensemble} isLoading={loading && !ensemble} />
  </div>

  <!-- Kelly Engine Tile -->
  <div class="tile-wrapper">
    <KellyTile data={kelly} isLoading={loading && !kelly} />
  </div>

  <!-- Error Display -->
  {#if unavailableReason}
    <div class="unavailable-banner">
      <span class="error-icon">ℹ</span>
      <span class="error-text">Physics models are unavailable on this host: {unavailableReason}</span>
      <button class="retry-btn unavailable" onclick={() => physicsSensorStore.fetch()}>
        Retry
      </button>
    </div>
  {:else if error}
    <div class="error-banner">
      <span class="error-icon">⚠</span>
      <span class="error-text">Failed to fetch physics data: {error}</span>
      <button class="retry-btn" onclick={() => physicsSensorStore.fetch()}>
        Retry
      </button>
    </div>
  {/if}
</div>

<style>
  .sensor-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 16px;
    padding: 16px;
  }

  .tile-wrapper {
    min-width: 0; /* Prevent grid overflow */
  }

  .error-banner {
    grid-column: 1 / -1;
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 12px 16px;
    background: rgba(255, 59, 59, 0.15);
    border: 1px solid rgba(255, 59, 59, 0.3);
    border-radius: 6px;
    color: #ff3b3b;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    font-size: 12px;
  }

  .unavailable-banner {
    grid-column: 1 / -1;
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 12px 16px;
    background: color-mix(in srgb, var(--color-accent-amber, #f59e0b) 12%, transparent);
    border: 1px solid color-mix(in srgb, var(--color-accent-amber, #f59e0b) 32%, transparent);
    border-radius: 6px;
    color: var(--color-accent-amber, #f59e0b);
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    font-size: 12px;
  }

  .error-icon {
    font-size: 14px;
  }

  .error-text {
    flex: 1;
  }

  .retry-btn {
    padding: 4px 12px;
    background: rgba(255, 59, 59, 0.2);
    border: 1px solid rgba(255, 59, 59, 0.4);
    border-radius: 4px;
    color: #ff3b3b;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    font-size: 11px;
    cursor: pointer;
    transition: background 0.2s ease;
  }

  .retry-btn:hover {
    background: rgba(255, 59, 59, 0.3);
  }

  .retry-btn.unavailable {
    background: color-mix(in srgb, var(--color-accent-amber, #f59e0b) 16%, transparent);
    border-color: color-mix(in srgb, var(--color-accent-amber, #f59e0b) 32%, transparent);
    color: var(--color-accent-amber, #f59e0b);
  }

  .retry-btn.unavailable:hover {
    background: color-mix(in srgb, var(--color-accent-amber, #f59e0b) 24%, transparent);
  }

  /* Responsive: single column on smaller screens */
  @media (max-width: 768px) {
    .sensor-grid {
      grid-template-columns: 1fr;
    }
  }
</style>
