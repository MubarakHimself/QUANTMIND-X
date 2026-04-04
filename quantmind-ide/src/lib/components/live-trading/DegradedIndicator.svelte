<script lang="ts">
  /**
   * DegradedIndicator - Degraded Mode Badge Component
   *
   * Shows amber "node_backend offline — retrying" label when node_backend is unreachable.
   * Does NOT block node_trading-dependent functionality.
   *
   * Uses pulse animation for attention-grabbing effect.
   */

  import { WifiOff, RefreshCw } from 'lucide-svelte';

  export let message = 'node_backend offline — retrying';
  export let showRetry = true;
</script>

<div class="degraded-indicator">
  <div class="indicator-content">
    <WifiOff size={12} strokeWidth={2} />
    <span class="message">{message}</span>
    {#if showRetry}
      <span class="retry-icon">
        <RefreshCw size={10} />
      </span>
    {/if}
  </div>
</div>

<style>
  .degraded-indicator {
    display: inline-flex;
    align-items: center;
    background: rgba(245, 158, 11, 0.12);
    border: 1px solid rgba(245, 158, 11, 0.35);
    border-radius: 6px;
    padding: 6px 10px;
    margin-bottom: 8px;
    animation: pulse-border 2s ease-in-out infinite;
  }

  .indicator-content {
    display: flex;
    align-items: center;
    gap: 6px;
    color: #f59e0b;
  }

  .message {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.3px;
  }

  .retry-icon {
    display: inline-flex;
    animation: spin 1.5s linear infinite;
  }

  @keyframes pulse-border {
    0%, 100% {
      border-color: rgba(245, 158, 11, 0.35);
      box-shadow: 0 0 4px rgba(245, 158, 11, 0.1);
    }
    50% {
      border-color: rgba(245, 158, 11, 0.6);
      box-shadow: 0 0 8px rgba(245, 158, 11, 0.25);
    }
  }

  @keyframes spin {
    from {
      transform: rotate(0deg);
    }
    to {
      transform: rotate(360deg);
    }
  }
</style>