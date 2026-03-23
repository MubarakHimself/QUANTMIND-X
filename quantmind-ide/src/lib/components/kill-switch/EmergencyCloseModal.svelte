<script lang="ts">
  import { ShieldX, X, AlertTriangle, XCircle } from "lucide-svelte";
  import {
    showEmergencyCloseModal,
    killSwitchLoading,
    killSwitchError,
    confirmKillSwitch,
    cancelKillSwitch
  } from "$lib/stores/kill-switch";
  import { onMount } from "svelte";

  // Real data - fetched from API
  let openPositions = $state(0);
  let estimatedExposure = $state(0);
  let loading = $state(true);

  // Fetch position data when modal opens
  $effect(() => {
    if ($showEmergencyCloseModal) {
      fetchPositionData();
    }
  });

  async function fetchPositionData() {
    loading = true;
    try {
      const response = await fetch("/api/v1/trading/bots");
      if (response.ok) {
        const data = await response.json();
        // Calculate total open positions and exposure from bot data
        let totalPositions = 0;
        let totalExposure = 0;

        if (data.bots && Array.isArray(data.bots)) {
          for (const bot of data.bots) {
            totalPositions += bot.open_positions || 0;
            // Estimate exposure - in production, this would come from actual position data
            totalExposure += bot.equity_exposure || 0;
          }
        }

        openPositions = totalPositions;
        estimatedExposure = totalExposure;
      }
    } catch (error) {
      console.error("[EmergencyClose] Failed to fetch position data:", error);
      // Fall back to zeros on error
      openPositions = 0;
      estimatedExposure = 0;
    } finally {
      loading = false;
    }
  }

  let confirmed = $state(false);

  function handleConfirm() {
    if (!confirmed) {
      confirmed = true;
      return;
    }
    // Second confirmation - actually fire
    confirmKillSwitch();
  }

  function handleCancel() {
    cancelKillSwitch();
    confirmed = false;
  }

  function formatCurrency(value: number): string {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD'
    }).format(value);
  }
</script>

<!-- Emergency Close Double-Confirmation Modal -->
{#if $showEmergencyCloseModal}
  <!-- svelte-ignore a11y_click_events_have_key_events a11y_no_static_element_interactions -->
  <div class="modal-overlay" onclick={handleCancel} role="dialog" aria-modal="true" aria-labelledby="emergency-modal-title">
    <!-- svelte-ignore a11y_click_events_have_key_events a11y_no_static_element_interactions -->
    <div class="modal-content emergency" onclick={(e) => e.stopPropagation()}>
      <button class="close-btn" onclick={handleCancel} aria-label="Close">
        <X size={18} />
      </button>

      <div class="modal-header">
        <div class="modal-icon danger">
          <XCircle size={32} />
        </div>
        <h3 id="emergency-modal-title">Emergency Close — Tier 3</h3>
        <p class="modal-subtitle">This action cannot be undone</p>
      </div>

      {#if $killSwitchError}
        <div class="error-banner">
          {$killSwitchError}
        </div>
      {/if}

      <div class="warning-box">
        <div class="warning-icon">
          <AlertTriangle size={20} />
        </div>
        <span>This will close all positions immediately</span>
      </div>

      <div class="exposure-info">
        {#if loading}
          <div class="loading-state">
            <span class="spinner"></span>
            <span>Loading position data...</span>
          </div>
        {:else}
          <div class="info-row">
            <span class="info-label">Open Positions</span>
            <span class="info-value">{openPositions}</span>
          </div>
          <div class="info-row">
            <span class="info-label">Estimated Exposure</span>
            <span class="info-value exposure">{formatCurrency(estimatedExposure)}</span>
          </div>
        {/if}
      </div>

      <div class="confirmation-step">
        {#if !confirmed}
          <p class="step-text">
            Click <strong>Confirm Emergency Close</strong> once to acknowledge the warning.
          </p>
          <p class="step-text secondary">
            A second click will execute the emergency close.
          </p>
        {:else}
          <p class="step-text final">
            Final confirmation — Click again to <strong>EXECUTE EMERGENCY CLOSE</strong>
          </p>
        {/if}
      </div>

      <div class="modal-actions">
        <button class="btn-cancel" onclick={handleCancel} disabled={$killSwitchLoading}>
          Cancel
        </button>
        <button
          class="btn-confirm"
          class:confirmed
          onclick={handleConfirm}
          disabled={$killSwitchLoading}
        >
          {#if $killSwitchLoading}
            <span class="spinner"></span>
            Closing...
          {:else if !confirmed}
            Confirm Emergency Close
          {:else}
            EXECUTE EMERGENCY CLOSE
          {/if}
        </button>
      </div>
    </div>
  </div>
{/if}

<style>
  .modal-overlay {
    position: fixed;
    inset: 0;
    background: rgba(0, 0, 0, 0.8);
    backdrop-filter: blur(8px);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 1100;
  }

  .modal-content {
    background: var(--color-bg-surface);
    border-radius: 16px;
    padding: 24px;
    width: 90%;
    max-width: 480px;
    position: relative;
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.6);
  }

  .modal-content.emergency {
    border: 2px solid var(--color-accent-red);
    box-shadow: 0 0 40px rgba(239, 68, 68, 0.2);
  }

  .close-btn {
    position: absolute;
    top: 16px;
    right: 16px;
    background: transparent;
    border: none;
    color: var(--color-text-muted);
    cursor: pointer;
    padding: 8px;
    border-radius: 8px;
    transition: all 0.15s ease;
    display: flex;
    align-items: center;
    justify-content: center;
  }

  .close-btn:hover {
    background: var(--glass-tier-2);
    color: var(--color-text-primary);
  }

  .modal-header {
    text-align: center;
    margin-bottom: 20px;
  }

  .modal-icon {
    display: flex;
    justify-content: center;
    margin-bottom: 12px;
  }

  .modal-icon.danger {
    color: var(--color-accent-red);
  }

  .modal-header h3 {
    font-family: var(--font-display);
    font-weight: 700;
    font-size: 22px;
    color: var(--color-accent-red);
    margin-bottom: 4px;
  }

  .modal-subtitle {
    font-size: 13px;
    color: var(--color-text-secondary);
  }

  .error-banner {
    background: rgba(239, 68, 68, 0.15);
    border: 1px solid rgba(239, 68, 68, 0.3);
    border-radius: 8px;
    padding: 12px;
    margin-bottom: 16px;
    color: #ef4444;
    font-size: 13px;
    text-align: center;
  }

  .warning-box {
    display: flex;
    align-items: center;
    gap: 12px;
    background: rgba(239, 68, 68, 0.1);
    border: 1px solid rgba(239, 68, 68, 0.3);
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 20px;
  }

  .warning-icon {
    flex-shrink: 0;
    color: var(--color-accent-red);
  }

  .warning-box span {
    font-size: 14px;
    font-weight: 600;
    color: var(--color-accent-red);
  }

  .exposure-info {
    background: var(--color-bg-elevated);
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 20px;
  }

  .info-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 0;
  }

  .info-row:not(:last-child) {
    border-bottom: 1px solid var(--color-border-subtle);
  }

  .info-label {
    font-size: 13px;
    color: var(--color-text-secondary);
  }

  .info-value {
    font-family: var(--font-mono);
    font-size: 14px;
    font-weight: 600;
    color: var(--color-text-primary);
  }

  .info-value.exposure {
    color: var(--color-warning);
  }

  .loading-state {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    padding: 16px;
    color: var(--color-text-secondary);
    font-size: 13px;
  }

  .loading-state .spinner {
    width: 14px;
    height: 14px;
    border: 2px solid transparent;
    border-top-color: var(--color-text-secondary);
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
  }

  .confirmation-step {
    background: var(--bg-surface);
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 24px;
    text-align: center;
  }

  .step-text {
    font-size: 14px;
    color: var(--color-text-primary);
    line-height: 1.5;
  }

  .step-text.secondary {
    font-size: 12px;
    color: var(--color-text-secondary);
    margin-top: 8px;
  }

  .step-text.final {
    color: var(--color-accent-red);
    font-weight: 600;
    animation: pulse-text 1.5s ease-in-out infinite;
  }

  @keyframes pulse-text {
    0%, 100% {
      opacity: 1;
    }
    50% {
      opacity: 0.7;
    }
  }

  .modal-actions {
    display: flex;
    gap: 12px;
    justify-content: center;
  }

  .btn-cancel,
  .btn-confirm {
    padding: 14px 28px;
    border-radius: 8px;
    font-family: var(--font-nav);
    font-size: 14px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.15s ease;
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .btn-cancel {
    background: var(--color-bg-elevated);
    border: 1px solid var(--color-border-subtle);
    color: var(--color-text-secondary);
  }

  .btn-cancel:hover:not(:disabled) {
    background: var(--bg-surface);
    color: var(--color-text-primary);
  }

  .btn-confirm {
    background: var(--color-accent-red);
    border: none;
    color: white;
  }

  .btn-confirm:hover:not(:disabled) {
    background: #ff5555;
    box-shadow: 0 0 20px rgba(255, 85, 85, 0.5);
  }

  .btn-confirm.confirmed {
    background: #dc2626;
    animation: pulse-button 0.8s ease-in-out infinite;
  }

  @keyframes pulse-button {
    0%, 100% {
      box-shadow: 0 0 0 0 rgba(220, 38, 38, 0.5);
    }
    50% {
      box-shadow: 0 0 20px 5px rgba(220, 38, 38, 0.4);
    }
  }

  .btn-cancel:disabled,
  .btn-confirm:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .spinner {
    width: 16px;
    height: 16px;
    border: 2px solid transparent;
    border-top-color: white;
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
  }

  @keyframes spin {
    to {
      transform: rotate(360deg);
    }
  }
</style>
