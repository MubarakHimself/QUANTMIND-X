<script lang="ts">
  /**
   * ComplianceTile - Compliance Overview Tile
   *
   * Displays BotCircuitBreaker state per account tag, prop firm rules,
   * and Islamic compliance countdown status.
   */

  import { onMount, onDestroy } from 'svelte';
  import { AlertTriangle, CheckCircle, Shield, Clock, XCircle } from 'lucide-svelte';
  import {
    complianceStore,
    complianceData,
    complianceLoading,
    complianceError,
    type ComplianceData
  } from '$lib/stores';

  let data: ComplianceData | null = $state(null);
  let loading = $state(false);
  let error: string | null = $state(null);
  let countdownDisplay = $state('');

  // Subscribe to store
  const unsubData = complianceData.subscribe(v => data = v);
  const unsubLoading = complianceLoading.subscribe(v => loading = v);
  const unsubError = complianceError.subscribe(v => error = v);

  onMount(() => {
    complianceStore.startPolling(5000);
  });

  onDestroy(() => {
    complianceStore.stopPolling();
    unsubData();
    unsubLoading();
    unsubError();
  });

  // Format countdown timer
  $effect(() => {
    if (data?.islamic) {
      const seconds = data.islamic.countdown_seconds;
      if (seconds > 0) {
        const hours = Math.floor(seconds / 3600);
        const mins = Math.floor((seconds % 3600) / 60);
        const secs = seconds % 60;
        countdownDisplay = hours > 0
          ? `${hours}h ${mins}m ${secs}s`
          : `${mins}m ${secs}s`;
      } else {
        countdownDisplay = 'Outside window';
      }
    }
  });

  function getStatusColor(status: string): string {
    switch (status) {
      case 'critical': return '#ff3b3b';
      case 'warning': return '#ffb700';
      default: return '#00d4ff';
    }
  }

  function getCircuitBreakerColor(state: string): string {
    switch (state) {
      case 'triggered': return '#ff3b3b';
      case 'warning': return '#ffb700';
      default: return '#00d4ff';
    }
  }
</script>

<div class="compliance-tile">
  <div class="tile-header">
    <h3 class="tile-title">
      <Shield size={16} />
      Compliance
    </h3>
    {#if data}
      <div class="status-badge" style="color: {getStatusColor(data.overall_status)}">
        {#if data.overall_status === 'compliant'}
          <CheckCircle size={14} />
        {:else if data.overall_status === 'warning'}
          <AlertTriangle size={14} />
        {:else}
          <XCircle size={14} />
        {/if}
      </div>
    {/if}
  </div>

  <div class="tile-content">
    {#if loading}
      <div class="loading-state">
        <span class="loading-text">Loading...</span>
      </div>
    {:else if error}
      <div class="error-state">
        <span class="error-text">{error}</span>
      </div>
    {:else if data}
      <!-- Account Tags Section -->
      <div class="section">
        <h4 class="section-title">Account Circuit Breakers</h4>
        <div class="account-list">
          {#each data.account_tags as account}
            <div class="account-item">
              <span class="account-tag">{account.tag}</span>
              <span class="cb-state" style="color: {getCircuitBreakerColor(account.circuit_breaker_state)}">
                {account.circuit_breaker_state}
              </span>
              <span class="drawdown">
                DD: {account.drawdown_pct.toFixed(1)}%
              </span>
              {#if account.daily_halt_triggered}
                <span class="halt-badge">HALTED</span>
              {/if}
            </div>
          {/each}
        </div>
      </div>

      <!-- Islamic Compliance Section -->
      <div class="section islamic-section">
        <h4 class="section-title">
          <Clock size={14} />
          Islamic Compliance
        </h4>
        <div class="islamic-status">
          {#if data.islamic.is_within_60min_window}
            <div class="countdown" class:critical={data.islamic.is_within_30min_window}>
              <Clock size={16} />
              <span class="countdown-time">{countdownDisplay}</span>
              <span class="countdown-label">
                {#if data.islamic.is_within_30min_window}
                  CRITICAL - Force close imminent
                {:else}
                  Warning - Force close at 21:45 UTC
                {/if}
              </span>
            </div>
            {#if data.islamic.active_positions_count > 0}
              <span class="positions-count">
                {data.islamic.active_positions_count} position(s) will close
              </span>
            {/if}
          {:else}
            <div class="safe-status">
              <CheckCircle size={14} />
              <span>Outside compliance window</span>
            </div>
          {/if}
        </div>
      </div>
    {/if}
  </div>
</div>

<style>
  .compliance-tile {
    background: rgba(8, 13, 20, 0.35);
    backdrop-filter: blur(16px) saturate(120%);
    -webkit-backdrop-filter: blur(16px) saturate(120%);
    border: 1px solid rgba(0, 212, 255, 0.08);
    border-radius: 8px;
    padding: 16px;
    display: flex;
    flex-direction: column;
    gap: 12px;
    min-height: 200px;
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
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .status-badge {
    display: flex;
    align-items: center;
  }

  .tile-content {
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 12px;
    overflow-y: auto;
  }

  .loading-state,
  .error-state {
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

  .error-text {
    color: #ff3b3b;
    font-size: 12px;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }

  .section {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .section-title {
    font-size: 11px;
    font-weight: 600;
    color: rgba(255, 255, 255, 0.6);
    margin: 0;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    display: flex;
    align-items: center;
    gap: 6px;
  }

  .account-list {
    display: flex;
    flex-direction: column;
    gap: 6px;
  }

  .account-item {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 6px 8px;
    background: rgba(0, 0, 0, 0.2);
    border-radius: 4px;
    font-size: 11px;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }

  .account-tag {
    color: rgba(255, 255, 255, 0.8);
    flex: 1;
  }

  .cb-state {
    text-transform: uppercase;
    font-weight: 600;
  }

  .drawdown {
    color: rgba(255, 255, 255, 0.5);
  }

  .halt-badge {
    background: rgba(255, 59, 59, 0.2);
    border: 1px solid rgba(255, 59, 59, 0.4);
    color: #ff3b3b;
    padding: 2px 6px;
    border-radius: 3px;
    font-size: 9px;
    font-weight: 600;
  }

  .islamic-section {
    margin-top: 8px;
    padding-top: 12px;
    border-top: 1px solid rgba(0, 212, 255, 0.1);
  }

  .islamic-status {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .countdown {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px;
    background: rgba(255, 183, 0, 0.1);
    border: 1px solid rgba(255, 183, 0, 0.3);
    border-radius: 4px;
    color: #ffb700;
  }

  .countdown.critical {
    background: rgba(255, 59, 59, 0.1);
    border-color: rgba(255, 59, 59, 0.4);
    color: #ff3b3b;
  }

  .countdown-time {
    font-size: 14px;
    font-weight: 600;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }

  .countdown-label {
    font-size: 10px;
    color: rgba(255, 255, 255, 0.6);
  }

  .positions-count {
    font-size: 10px;
    color: rgba(255, 255, 255, 0.5);
    padding-left: 24px;
  }

  .safe-status {
    display: flex;
    align-items: center;
    gap: 6px;
    color: #00d4ff;
    font-size: 12px;
  }
</style>
