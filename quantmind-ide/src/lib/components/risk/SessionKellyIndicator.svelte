<script lang="ts">
  /**
   * SessionKellyIndicator - Session Kelly Modifier Display
   *
   * Compact display showing current Kelly modifiers per session with:
   * - Kelly fraction per session
   * - Visual bars for Kelly dollar amounts
   * - House money status: NORMAL (green), STRESS (amber), CRITICAL (red)
   * - London-NY overlap special indicator (+4% vs normal +8%)
   * - Color coded per session modifier status
   *
   * Part of Story 4.10: Session-Scoped Kelly Modifiers
   */

  import { onMount, onDestroy } from 'svelte';
  import { sessionKellyStore, type SessionKellyData } from '$lib/stores/sessionKellyStore';
  import { RefreshCw, TrendingUp, TrendingDown, AlertTriangle, Shield, Zap } from 'lucide-svelte';

  // Props
  export let compact: boolean = false;
  export let showHistory: boolean = false;
  export let autoRefresh: boolean = true;

  // Local state
  let isLoading = false;
  let error: string | null = null;
  let currentState: typeof $sessionKellyStore | null = null;
  let lastUpdate: Date | null = null;

  // Subscribe to store
  const unsubscribe = sessionKellyStore.subscribe(state => {
    currentState = state;
    isLoading = state.isLoading;
    error = state.error;
    if (state.current) {
      lastUpdate = new Date(state.lastUpdate);
    }
  });

  // Derived values
  $: sessions = currentState?.current?.sessions || [];
  $: activeSession = sessions.find(s => s.is_active);
  $: londonNyOverlap = sessions.find(s => s.name === 'OVERLAP' && s.is_active);
  $: thresholdText = activeSession
    ? activeSession.is_premium
      ? `+${(activeSession.house_money_threshold * 100).toFixed(0)}% (Premium)`
      : `+${(activeSession.house_money_threshold * 100).toFixed(0)}%`
    : '+8%';
  $: compositeMultiplier = activeSession?.session_kelly_multiplier.toFixed(2) || '1.00';
  $: statusColor = getStatusColor(activeSession?.status);
  $: isHouseMoney = activeSession?.status === 'NORMAL' && (activeSession?.hmm_multiplier || 1) > 1.0;
  $: isPreservation = activeSession?.status === 'CRITICAL' || (activeSession?.hmm_multiplier || 1) < 1.0;

  function getStatusColor(status?: string): string {
    const colors: Record<string, string> = {
      NORMAL: '#00ff88',
      WARNING: '#f59e0b',
      STRESS: '#ef4444',
      CRITICAL: '#dc2626',
    };
    return colors[status || 'NORMAL'] || colors.NORMAL;
  }

  function getMultiplierColor(multiplier: number): string {
    if (multiplier >= 1.5) return '#00ff88';
    if (multiplier >= 1.0) return '#00d4ff';
    if (multiplier >= 0.7) return '#f59e0b';
    return '#ef4444';
  }

  function formatTime(date: Date | null): string {
    if (!date) return '--:--:--';
    return date.toLocaleTimeString('en-US', { hour12: false });
  }

  async function handleRefresh() {
    await sessionKellyStore.loadCurrentKelly();
    if (showHistory) {
      await sessionKellyStore.loadHistory();
    }
  }

  onMount(async () => {
    await handleRefresh();
    if (autoRefresh) {
      sessionKellyStore.startAutoRefresh(30000);
    }
  });

  onDestroy(() => {
    unsubscribe();
    sessionKellyStore.stopAutoRefresh();
  });

  // Session display configuration
  const SESSION_LABELS: Record<string, string> = {
    ASIAN: 'Asian',
    LONDON: 'London',
    NEW_YORK: 'NY',
    OVERLAP: 'L-NY',
  };

  function getSessionLabel(name: string): string {
    return SESSION_LABELS[name] || name;
  }
</script>

<div class="session-kelly-indicator" class:compact>
  <!-- Header -->
  <div class="header">
    <div class="title-row">
      <Shield size={14} />
      <span class="title">Session Kelly</span>
      {#if activeSession?.is_premium}
        <span class="premium-badge">
          <Zap size={10} />
          Premium
        </span>
      {/if}
    </div>
    <button class="refresh-btn" on:click={handleRefresh} disabled={isLoading}>
      <RefreshCw size={12} class={isLoading ? 'spinning' : ''} />
    </button>
  </div>

  {#if error}
    <div class="error-message">
      <AlertTriangle size={12} />
      <span>{error}</span>
    </div>
  {:else if activeSession}
    <!-- Active Session Display -->
    <div class="active-session">
      <div class="session-name">
        {getSessionLabel(activeSession.name)}
        {#if activeSession.is_active}
          <span class="active-dot" style="background: {statusColor}"></span>
        {/if}
      </div>

      <!-- Composite Multiplier -->
      <div class="multiplier-display">
        <span class="multiplier-value" style="color: {getMultiplierColor(activeSession.session_kelly_multiplier)}">
          {compositeMultiplier}x
        </span>
        <span class="multiplier-label">Kelly</span>
      </div>

      <!-- Status Badge -->
      <div class="status-badge" style="background: {statusColor}20; border-color: {statusColor}">
        {#if isPreservation}
          <TrendingDown size={10} />
        {:else}
          <TrendingUp size={10} />
        {/if}
        <span style="color: {statusColor}">{activeSession.status}</span>
      </div>
    </div>

    <!-- Threshold Info -->
    <div class="threshold-info">
      <span class="threshold-label">House Money Threshold</span>
      <span class="threshold-value" class:premium={activeSession.is_premium}>
        {thresholdText}
      </span>
    </div>

    <!-- London-NY Overlap Indicator -->
    {#if londonNyOverlap && londonNyOverlap.is_active}
      <div class="overlap-indicator">
        <Zap size={12} />
        <span>London-NY Overlap Active</span>
        <span class="overlap-threshold">+4% threshold</span>
      </div>
    {/if}

    {#if !compact}
      <!-- All Sessions Grid -->
      <div class="sessions-grid">
        {#each sessions as session}
          <div
            class="session-card"
            class:active={session.is_active}
            class:premium={session.is_premium}
            style="--status-color: {getStatusColor(session.status)}"
          >
            <div class="session-card-header">
              <span class="session-card-name">{getSessionLabel(session.name)}</span>
              {#if session.is_active}
                <span class="active-indicator"></span>
              {/if}
            </div>
            <div class="session-card-multiplier" style="color: {getMultiplierColor(session.session_kelly_multiplier)}">
              {session.session_kelly_multiplier.toFixed(2)}x
            </div>
            <div class="session-card-status">
              {session.status}
            </div>
            {#if session.is_premium}
              <div class="premium-indicator">
                <Zap size={8} />
              </div>
            {/if}
          </div>
        {/each}
      </div>

      <!-- HMM/RHMM Breakdown -->
      <div class="modifier-breakdown">
        <div class="modifier-item">
          <span class="modifier-label">HMM</span>
          <span class="modifier-value" class:boosted={activeSession.hmm_multiplier > 1.0}>
            {activeSession.hmm_multiplier.toFixed(2)}x
          </span>
        </div>
        <div class="modifier-divider"></div>
        <div class="modifier-item">
          <span class="modifier-label">R-HMM</span>
          <span class="modifier-value" class:reduced={activeSession.reverse_hmm_multiplier < 1.0}>
            {activeSession.reverse_hmm_multiplier.toFixed(2)}x
          </span>
        </div>
        <div class="modifier-divider"></div>
        <div class="modifier-item">
          <span class="modifier-label">Losses</span>
          <span class="modifier-value" class:stress={activeSession.session_loss_counter >= 4}>
            {activeSession.session_loss_counter}
          </span>
        </div>
      </div>
    {/if}

    <!-- Last Update -->
    <div class="last-update">
      Updated: {formatTime(lastUpdate)}
    </div>
  {:else}
    <div class="no-data">
      <span>No active session data</span>
    </div>
  {/if}
</div>

<style>
  .session-kelly-indicator {
    display: flex;
    flex-direction: column;
    gap: 10px;
    padding: 12px;
    background: rgba(0, 0, 0, 0.3);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 8px;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    font-size: 11px;
    color: rgba(255, 255, 255, 0.8);
    min-width: 180px;
  }

  .session-kelly-indicator.compact {
    padding: 8px;
    min-width: 140px;
    gap: 6px;
  }

  .header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding-bottom: 8px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
  }

  .title-row {
    display: flex;
    align-items: center;
    gap: 6px;
    color: rgba(255, 255, 255, 0.7);
  }

  .title {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    font-weight: 500;
  }

  .premium-badge {
    display: flex;
    align-items: center;
    gap: 3px;
    padding: 2px 6px;
    background: rgba(139, 92, 246, 0.2);
    border: 1px solid rgba(139, 92, 246, 0.5);
    border-radius: 4px;
    color: #a78bfa;
    font-size: 9px;
    text-transform: uppercase;
  }

  .refresh-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 24px;
    height: 24px;
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 4px;
    color: rgba(255, 255, 255, 0.5);
    cursor: pointer;
    transition: all 0.2s;
  }

  .refresh-btn:hover:not(:disabled) {
    background: rgba(255, 255, 255, 0.1);
    color: rgba(255, 255, 255, 0.8);
  }

  .refresh-btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .refresh-btn :global(.spinning) {
    animation: spin 1s linear infinite;
  }

  @keyframes spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
  }

  .error-message {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 8px;
    background: rgba(239, 68, 68, 0.1);
    border: 1px solid rgba(239, 68, 68, 0.3);
    border-radius: 4px;
    color: #ef4444;
    font-size: 10px;
  }

  .active-session {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 10px;
    background: rgba(0, 0, 0, 0.2);
    border-radius: 6px;
  }

  .session-name {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 14px;
    font-weight: 600;
    color: rgba(255, 255, 255, 0.9);
  }

  .active-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    animation: pulse 2s infinite;
  }

  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
  }

  .multiplier-display {
    display: flex;
    flex-direction: column;
    align-items: flex-end;
  }

  .multiplier-value {
    font-size: 18px;
    font-weight: 700;
  }

  .multiplier-label {
    font-size: 9px;
    color: rgba(255, 255, 255, 0.4);
    text-transform: uppercase;
  }

  .status-badge {
    display: flex;
    align-items: center;
    gap: 4px;
    padding: 4px 8px;
    border: 1px solid;
    border-radius: 4px;
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
  }

  .threshold-info {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 6px 10px;
    background: rgba(0, 0, 0, 0.15);
    border-radius: 4px;
  }

  .threshold-label {
    font-size: 10px;
    color: rgba(255, 255, 255, 0.5);
    text-transform: uppercase;
  }

  .threshold-value {
    font-size: 12px;
    font-weight: 600;
    color: rgba(255, 255, 255, 0.8);
  }

  .threshold-value.premium {
    color: #a78bfa;
  }

  .overlap-indicator {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 6px 10px;
    background: rgba(139, 92, 246, 0.15);
    border: 1px solid rgba(139, 92, 246, 0.3);
    border-radius: 4px;
    color: #a78bfa;
    font-size: 10px;
  }

  .overlap-threshold {
    margin-left: auto;
    color: #c4b5fd;
    font-size: 9px;
  }

  .sessions-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 6px;
  }

  .session-card {
    position: relative;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 4px;
    padding: 8px 4px;
    background: rgba(0, 0, 0, 0.2);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 4px;
    transition: all 0.2s;
  }

  .session-card.active {
    border-color: var(--status-color);
    background: rgba(0, 0, 0, 0.3);
  }

  .session-card.premium {
    border-color: rgba(139, 92, 246, 0.5);
  }

  .session-card-header {
    display: flex;
    align-items: center;
    gap: 4px;
  }

  .session-card-name {
    font-size: 10px;
    font-weight: 600;
    color: rgba(255, 255, 255, 0.7);
    text-transform: uppercase;
  }

  .active-indicator {
    width: 4px;
    height: 4px;
    border-radius: 50%;
    background: var(--status-color);
  }

  .session-card-multiplier {
    font-size: 14px;
    font-weight: 700;
  }

  .session-card-status {
    font-size: 8px;
    color: rgba(255, 255, 255, 0.4);
    text-transform: uppercase;
  }

  .premium-indicator {
    position: absolute;
    top: 2px;
    right: 2px;
    color: #a78bfa;
  }

  .modifier-breakdown {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 12px;
    padding: 8px;
    background: rgba(0, 0, 0, 0.2);
    border-radius: 4px;
  }

  .modifier-item {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 2px;
  }

  .modifier-label {
    font-size: 9px;
    color: rgba(255, 255, 255, 0.4);
    text-transform: uppercase;
  }

  .modifier-value {
    font-size: 12px;
    font-weight: 600;
    color: rgba(255, 255, 255, 0.8);
  }

  .modifier-value.boosted {
    color: #00ff88;
  }

  .modifier-value.reduced {
    color: #f59e0b;
  }

  .modifier-value.stress {
    color: #ef4444;
  }

  .modifier-divider {
    width: 1px;
    height: 24px;
    background: rgba(255, 255, 255, 0.1);
  }

  .last-update {
    font-size: 9px;
    color: rgba(255, 255, 255, 0.3);
    text-align: right;
  }

  .no-data {
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 20px;
    color: rgba(255, 255, 255, 0.3);
    font-size: 11px;
  }
</style>
