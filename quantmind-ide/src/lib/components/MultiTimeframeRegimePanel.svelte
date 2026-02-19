<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { createTradingClient } from '$lib/ws-client';
  import type { WebSocketClient } from '$lib/ws-client';
  import { PUBLIC_API_BASE } from '$env/static/public';
  import { Activity, TrendingUp } from 'lucide-svelte';

  // Use configured API base or default to same origin
  const apiBase = PUBLIC_API_BASE || '';

  // TypeScript interfaces for regime data
  interface TimeframeRegime {
    regime: string;
    quality: number;
  }

  interface RegimeUpdateData {
    dominant_regime: string;
    timeframe_regimes: Record<string, TimeframeRegime>;
    consensus_strength: number;
  }

  // Reactive state with defaults
  let dominantRegime: string = 'UNKNOWN';
  let timeframeRegimes: Record<string, TimeframeRegime> = {};
  let consensusStrength: number = 0;
  let wsClient: WebSocketClient | null = null;

  // Helper function to get regime color
  function getRegimeColor(regime: string): string {
    switch (regime) {
      case 'TREND_STABLE':
        return '#10b981'; // green
      case 'RANGE_STABLE':
        return '#3b82f6'; // blue
      case 'BREAKOUT_PRIME':
        return '#f59e0b'; // amber
      case 'HIGH_CHAOS':
        return '#ef4444'; // red
      case 'NEWS_EVENT':
        return '#8b5cf6'; // purple
      default:
        return '#6b7280'; // gray
    }
  }

  // Format regime name for display
  function formatRegimeName(regime: string): string {
    return regime.replace(/_/g, ' ').toLowerCase().replace(/\b\w/g, c => c.toUpperCase());
  }

  onMount(async () => {
    try {
      // Build absolute URL
      const baseUrl = apiBase || window.location.origin;

      // Connect to WebSocket for real-time updates
      wsClient = await createTradingClient(baseUrl);

      // Subscribe to regime update events
      wsClient.on('regime_update', (message) => {
        if (message.data) {
          const data = message.data as RegimeUpdateData;
          dominantRegime = data.dominant_regime || 'UNKNOWN';
          timeframeRegimes = data.timeframe_regimes || {};
          consensusStrength = data.consensus_strength || 0;
        }
      });
    } catch (error) {
      console.error('Failed to connect to WebSocket for regime updates:', error);
    }
  });

  onDestroy(() => {
    if (wsClient) {
      wsClient.disconnect();
    }
  });
</script>

<div class="mtf-regime-panel">
  <div class="panel-header">
    <div class="header-left">
      <Activity size={20} />
      <h3>Multi-Timeframe Regime Analysis</h3>
    </div>
  </div>

  <div class="dominant-regime-card" style="background-color: {getRegimeColor(dominantRegime)}20; border-color: {getRegimeColor(dominantRegime)};">
    <div class="regime-badge" style="background-color: {getRegimeColor(dominantRegime)};">
      {formatRegimeName(dominantRegime)}
    </div>
    <div class="consensus-display">
      <span class="consensus-label">Consensus</span>
      <span class="consensus-value">{consensusStrength.toFixed(0)}%</span>
    </div>
  </div>

  <div class="timeframe-grid">
    {#each Object.entries(timeframeRegimes) as [timeframe, data]}
      <div class="timeframe-card" style="border-color: {getRegimeColor(data.regime)};">
        <div class="timeframe-label">{timeframe}</div>
        <div class="regime-badge-small" style="background-color: {getRegimeColor(data.regime)};">
          {formatRegimeName(data.regime)}
        </div>
        <div class="quality-bar-container">
          <div class="quality-bar" style="width: {data.quality * 100}%; background-color: {getRegimeColor(data.regime)};"></div>
        </div>
        <div class="quality-text">{(data.quality * 100).toFixed(0)}% quality</div>
      </div>
    {/each}
  </div>

  {#if Object.keys(timeframeRegimes).length === 0}
    <div class="no-data">
      <TrendingUp size={24} />
      <span>Waiting for regime data...</span>
    </div>
  {/if}
</div>

<style>
  .mtf-regime-panel {
    background: var(--bg-secondary);
    border: 1px solid var(--border-subtle);
    border-radius: 10px;
    padding: 16px;
    margin-top: 16px;
  }

  .panel-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 16px;
    padding-bottom: 12px;
    border-bottom: 1px solid var(--border-subtle);
  }

  .header-left {
    display: flex;
    align-items: center;
    gap: 8px;
    color: var(--accent-primary);
  }

  .header-left h3 {
    margin: 0;
    font-size: 14px;
    font-weight: 600;
    color: var(--text-primary);
  }

  .dominant-regime-card {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 20px;
    border-radius: 8px;
    border: 2px solid;
    margin-bottom: 16px;
  }

  .regime-badge {
    font-size: 20px;
    font-weight: 700;
    color: white;
    padding: 8px 20px;
    border-radius: 6px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  .consensus-display {
    display: flex;
    flex-direction: column;
    align-items: center;
    margin-top: 12px;
  }

  .consensus-label {
    font-size: 12px;
    color: var(--text-muted);
    text-transform: uppercase;
  }

  .consensus-value {
    font-size: 24px;
    font-weight: 700;
    color: var(--text-primary);
  }

  .timeframe-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
  }

  .timeframe-card {
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 12px;
    border: 2px solid;
    border-radius: 8px;
    background-color: var(--bg-tertiary);
  }

  .timeframe-label {
    font-size: 14px;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 8px;
  }

  .regime-badge-small {
    font-size: 10px;
    font-weight: 600;
    color: white;
    padding: 4px 8px;
    border-radius: 4px;
    text-transform: uppercase;
    margin-bottom: 8px;
    text-align: center;
    max-width: 100%;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .quality-bar-container {
    width: 100%;
    height: 6px;
    background-color: var(--bg-secondary);
    border-radius: 3px;
    overflow: hidden;
    margin-bottom: 6px;
  }

  .quality-bar {
    height: 100%;
    border-radius: 3px;
    transition: width 0.3s ease-in-out;
  }

  .quality-text {
    font-size: 11px;
    color: var(--text-muted);
  }

  .no-data {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 40px;
    color: var(--text-muted);
    gap: 12px;
  }

  @media (max-width: 768px) {
    .timeframe-grid {
      grid-template-columns: repeat(2, 1fr);
    }
  }

  @media (max-width: 480px) {
    .timeframe-grid {
      grid-template-columns: 1fr;
    }
  }
</style>