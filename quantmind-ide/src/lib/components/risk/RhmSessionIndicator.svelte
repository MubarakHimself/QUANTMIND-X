<script lang="ts">
  /**
   * RhmSessionIndicator - Reverse House Money Session Risk State
   *
   * Displays:
   * - Session loss counter (e.g., "2/6 session losses")
   * - Active RHM multiplier (e.g., "0.70x risk")
   * - Visual depleting bar (green -> amber -> red)
   * - HMM mode badge (House Money / Preservation)
   */

  import { onMount } from 'svelte';
  import { AlertTriangle, TrendingUp, Shield } from 'lucide-svelte';

  interface SessionRiskState {
    session_loss_counter: number;
    reverse_hmm_multiplier: number;
    hmm_multiplier: number;
    is_house_money_active: boolean;
    is_preservation_mode: boolean;
    daily_pnl_pct: number;
  }

  let sessionState = $state<SessionRiskState | null>(null);
  let loading = $state(true);

  // Derived values
  let lossRatio = $derived(sessionState ? sessionState.session_loss_counter / 6 : 0);

  let barColor = $derived(
    sessionState
      ? sessionState.session_loss_counter >= 4
        ? '#EF4444'
        : sessionState.session_loss_counter >= 2
          ? '#F59E0B'
          : '#10B981'
      : '#10B981'
  );

  let barWidth = $derived(
    sessionState
      ? sessionState.session_loss_counter >= 6
        ? '16%'
        : sessionState.session_loss_counter >= 4
          ? '33%'
          : sessionState.session_loss_counter >= 2
            ? '66%'
            : '100%'
      : '100%'
  );

  let multiplierDisplay = $derived(
    sessionState?.reverse_hmm_multiplier != null
      ? `${sessionState.reverse_hmm_multiplier.toFixed(2)}x risk`
      : '1.00x risk'
  );

  let hasActiveBadge = $derived(
    sessionState && (sessionState.is_house_money_active || sessionState.is_preservation_mode)
  );

  async function fetchState() {
    try {
      const response = await fetch('/api/trading/session-risk-state');
      if (response.ok) {
        sessionState = await response.json();
      }
    } catch (e) {
      console.error('Failed to fetch session risk state:', e);
    } finally {
      loading = false;
    }
  }

  onMount(() => {
    fetchState();
    const interval = setInterval(fetchState, 5000);
    return () => clearInterval(interval);
  });
</script>

<div class="rhm-indicator bg-white/[0.08] rounded-lg p-3">
  {#if loading}
    <div class="loading-state">
      <span class="text-xs text-white/40 font-mono">Loading...</span>
    </div>
  {:else if sessionState}
    <div class="content-panel bg-white/[0.35] rounded-md p-2">
      <!-- Header row -->
      <div class="flex items-center justify-between mb-2">
        <div class="flex items-center gap-2">
          <Shield size={14} class="text-white/50" />
          <span class="text-xs font-mono text-white/60 uppercase tracking-wide">RHM Session</span>
        </div>
        <span class="text-xs font-mono font-semibold" style="color: {barColor}">
            {multiplierDisplay}
        </span>
      </div>

      <!-- Loss counter with tooltip -->
      <div class="loss-counter mb-2" title="Resets on next win">
        <span class="text-xs text-white/50 font-mono">Session losses</span>
        <div class="flex items-center gap-1 mt-1">
          <span class="text-sm font-mono font-bold text-white">
            {sessionState.session_loss_counter}
          </span>
          <span class="text-xs text-white/40 font-mono">/ 6</span>
          <AlertTriangle size={12} class="text-white/30 ml-1" />
        </div>
      </div>

      <!-- Depleting bar -->
      <div class="bar-container bg-black/30 rounded-full h-2 overflow-hidden mb-2">
        <div
          class="bar-fill h-full rounded-full transition-all duration-300"
          style="width: {barWidth}; background-color: {barColor}"
        ></div>
      </div>

      <!-- HMM mode badge -->
      {#if sessionState.is_house_money_active}
        <div class="badge house-money">
          <TrendingUp size={12} />
          <span>HOUSE MONEY +8% / 1.4x</span>
        </div>
      {:else if sessionState.is_preservation_mode}
        <div class="badge preservation">
          <AlertTriangle size={12} />
          <span>PRESERVATION -10% / 0.5x</span>
        </div>
      {/if}
    </div>
  {:else}
    <div class="no-data">
      <span class="text-xs text-white/30 font-mono">No data</span>
    </div>
  {/if}
</div>

<style>
  .rhm-indicator {
    min-width: 140px;
  }

  .content-panel {
    min-height: 80px;
  }

  .loading-state,
  .no-data {
    display: flex;
    align-items: center;
    justify-content: center;
    min-height: 80px;
  }

  .loss-counter {
    cursor: help;
  }

  .bar-container {
    height: 6px;
  }

  .bar-fill {
    height: 100%;
  }

  .badge {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 4px 8px;
    border-radius: 4px;
    font-size: 10px;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  .badge.house-money {
    background: rgba(234, 179, 8, 0.2);
    color: #eab308;
    border: 1px solid rgba(234, 179, 8, 0.4);
  }

  .badge.preservation {
    background: rgba(239, 68, 68, 0.2);
    color: #ef4444;
    border: 1px solid rgba(239, 68, 68, 0.4);
  }
</style>
