<script lang="ts">
  /**
   * PhysicsSensorGrid - Grid Container for Physics Sensor Tiles
   *
   * Displays all physics sensor tiles in a responsive grid layout.
   * Each tile fetches independently to prevent cascade failures (NFR-R1).
   */

  import { onMount, onDestroy } from 'svelte';
  import IsingTile from './IsingTile.svelte';
  import LyapunovTile from './LyapunovTile.svelte';
  import HMMTile from './HMMTile.svelte';
  import KellyTile from './KellyTile.svelte';
  import {
    physicsSensorStore,
    isingData,
    lyapunovData,
    hmmData,
    kellyData,
    physicsLoading,
    physicsError,
    type PhysicsSensorData
  } from '$lib/stores/risk';

  // Subscribe to store values
  let ising = $state<PhysicsSensorData['ising'] | null>(null);
  let lyapunov = $state<PhysicsSensorData['lyapunov'] | null>(null);
  let hmm = $state<PhysicsSensorData['hmm'] | null>(null);
  let kelly = $state<PhysicsSensorData['kelly'] | null>(null);
  let loading = $state(false);
  let error = $state<string | null>(null);

  // Subscribe to stores
  const unsubIsing = isingData.subscribe(v => ising = v);
  const unsubLyapunov = lyapunovData.subscribe(v => lyapunov = v);
  const unsubHMM = hmmData.subscribe(v => hmm = v);
  const unsubKelly = kellyData.subscribe(v => kelly = v);
  const unsubLoading = physicsLoading.subscribe(v => loading = v);
  const unsubError = physicsError.subscribe(v => error = v);

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
    unsubKelly();
    unsubLoading();
    unsubError();
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

  <!-- Kelly Engine Tile -->
  <div class="tile-wrapper">
    <KellyTile data={kelly} isLoading={loading && !kelly} />
  </div>

  <!-- Error Display -->
  {#if error}
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

  /* Responsive: single column on smaller screens */
  @media (max-width: 768px) {
    .sensor-grid {
      grid-template-columns: 1fr;
    }
  }
</style>
