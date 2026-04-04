<script lang="ts">
/**
 * NodeHealthBadge - Node Status Indicator Component
 *
 * Displays connectivity status for Contabo (agents/compute) and Cloudzy (live trading) nodes.
 * Shows latency and connection status with color coding.
 *
 * Uses Lucide icons: wifi, wifi-off
 */

import { nodeHealthState } from '$lib/stores/node-health';
import { Wifi, WifiOff, Server, Activity } from 'lucide-svelte';

// Node status display
const statusConfig = {
  connected: {
    color: '#00c896',
    bg: 'rgba(0, 200, 150, 0.1)',
    label: 'Connected'
  },
  disconnected: {
    color: '#ff3b3b',
    bg: 'rgba(255, 59, 59, 0.1)',
    label: 'Offline'
  },
  reconnecting: {
    color: '#f59e0b',
    bg: 'rgba(245, 158, 11, 0.1)',
    label: 'Reconnecting'
  }
};

// Get latency display
function formatLatency(ms: number): string {
  if (ms === 0) return '--';
  if (ms < 1000) return `${ms}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}
</script>

<div class="node-health-badge">
  <div class="node contabo" class:degraded={$nodeHealthState.contabo.status !== 'connected'}>
    <div class="node-icon">
      <Server size={10} strokeWidth={2} />
    </div>
    <span class="node-label">node_backend</span>
    <div class="status-indicator" style="background: {statusConfig[$nodeHealthState.contabo.status].color}">
      {#if $nodeHealthState.contabo.status === 'connected'}
        <Wifi size={8} strokeWidth={2} />
      {:else if $nodeHealthState.contabo.status === 'reconnecting'}
        <Activity size={8} strokeWidth={2} class="spinning" />
      {:else}
        <WifiOff size={8} strokeWidth={2} />
      {/if}
    </div>
    <span class="latency" style="color: {statusConfig[$nodeHealthState.contabo.status].color}">
      {formatLatency($nodeHealthState.contabo.latency_ms)}
    </span>
  </div>

  <div class="separator"></div>

  <div class="node cloudzy" class:degraded={$nodeHealthState.cloudzy.status !== 'connected'}>
    <div class="node-icon">
      <Server size={10} strokeWidth={2} />
    </div>
    <span class="node-label">node_trading</span>
    <div class="status-indicator" style="background: {statusConfig[$nodeHealthState.cloudzy.status].color}">
      {#if $nodeHealthState.cloudzy.status === 'connected'}
        <Wifi size={8} strokeWidth={2} />
      {:else}
        <WifiOff size={8} strokeWidth={2} />
      {/if}
    </div>
    <span class="latency" style="color: {statusConfig[$nodeHealthState.cloudzy.status].color}">
      {formatLatency($nodeHealthState.cloudzy.latency_ms)}
    </span>
  </div>
</div>

<style>
  .node-health-badge {
    display: flex;
    align-items: center;
    gap: 8px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
  }

  .node {
    display: flex;
    align-items: center;
    gap: 4px;
    padding: 4px 8px;
    border-radius: 4px;
    background: rgba(0, 0, 0, 0.2);
    border: 1px solid transparent;
    transition: all 0.2s ease;
  }

  .node.degraded {
    background: rgba(245, 158, 11, 0.08);
    border-color: rgba(245, 158, 11, 0.2);
  }

  .node-icon {
    color: #666;
    display: flex;
    align-items: center;
  }

  .node-label {
    color: #888;
    font-weight: 500;
  }

  .status-indicator {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 14px;
    height: 14px;
    border-radius: 50%;
    color: #000;
  }

  .latency {
    font-weight: 500;
    min-width: 40px;
    text-align: right;
  }

  .separator {
    width: 1px;
    height: 16px;
    background: rgba(255, 255, 255, 0.1);
  }

  :global(.spinning) {
    animation: spin 1s linear infinite;
  }

  @keyframes spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
  }
</style>