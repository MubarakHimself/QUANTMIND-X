<script lang="ts">
  /**
   * PhysicsSensorTile - Base Component for Physics Sensor Tiles
   *
   * Frosted Terminal glass styling with alert state handling.
   * Each sensor tile displays its type, current values, and alert state.
   */

  import { AlertTriangle, CheckCircle, AlertCircle } from 'lucide-svelte';

  export let title: string;
  export let alert: 'normal' | 'warning' | 'critical' = 'normal';
  export let alertMessage: string = '';
  export let isLoading: boolean = false;

  // Alert border color
  const alertColors = {
    normal: 'rgba(0, 212, 255, 0.08)',
    warning: 'rgba(255, 183, 0, 0.4)',
    critical: 'rgba(255, 59, 59, 0.6)'
  };

  const alertIconColors = {
    normal: '#00d4ff',
    warning: '#ffb700',
    critical: '#ff3b3b'
  };

  $: borderColor = alertColors[alert];
  $: iconColor = alertIconColors[alert];
  $: isAlert = alert !== 'normal';
</script>

<div class="sensor-tile" style="border-color: {borderColor}">
  <div class="tile-header">
    <h3 class="tile-title">{title}</h3>
    {#if isAlert}
      <div class="alert-badge" style="color: {iconColor}">
        <AlertTriangle size={14} />
      </div>
    {/if}
    {#if !isAlert}
      <div class="status-badge" style="color: {iconColor}">
        <CheckCircle size={14} />
      </div>
    {/if}
  </div>

  <div class="tile-content">
    {#if isLoading}
      <div class="loading-state">
        <span class="loading-text">Loading...</span>
      </div>
    {:else}
      <slot />
    {/if}
  </div>

  {#if isAlert && alertMessage}
    <div class="alert-message" style="color: {iconColor}">
      <AlertTriangle size={12} />
      <span>{alertMessage}</span>
    </div>
  {/if}
</div>

<style>
  .sensor-tile {
    /* Tier 2 Frosted Terminal Glass */
    background: rgba(8, 13, 20, 0.35);
    backdrop-filter: blur(16px) saturate(120%);
    -webkit-backdrop-filter: blur(16px) saturate(120%);
    border: 1px solid rgba(0, 212, 255, 0.08);
    border-radius: 8px;
    padding: 16px;
    display: flex;
    flex-direction: column;
    gap: 12px;
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
    min-height: 140px;
  }

  .sensor-tile:hover {
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
  }

  .tile-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
  }

  .tile-title {
    font-size: 14px;
    font-weight: 600;
    color: rgba(255, 255, 255, 0.9);
    margin: 0;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    letter-spacing: 0.5px;
  }

  .alert-badge,
  .status-badge {
    display: flex;
    align-items: center;
    justify-content: center;
  }

  .tile-content {
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .loading-state {
    display: flex;
    align-items: center;
    justify-content: center;
    flex: 1;
  }

  .loading-text {
    color: rgba(255, 255, 255, 0.4);
    font-size: 12px;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }

  .alert-message {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 11px;
    padding: 6px 8px;
    background: rgba(0, 0, 0, 0.2);
    border-radius: 4px;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }
</style>
