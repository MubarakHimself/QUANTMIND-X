<script lang="ts">
  /**
   * PositionCloseModal - Confirmation Modal for Single Position Close
   *
   * Displays position details and confirms close action.
   * Follows Frosted Terminal aesthetic (Tier 2 glass).
   */
  import { X, TrendingUp, TrendingDown, AlertTriangle, Loader2 } from 'lucide-svelte';
  import { closePosition, closeLoading, closeError, type PositionInfo } from '$lib/stores/trading';

  interface Props {
    position: PositionInfo;
    onClose: () => void;
    onSuccess?: (result: CloseResult) => void;
  }

  let { position, onClose, onSuccess }: Props = $props();

  let confirmed = $state(false);
  let loading = $derived($closeLoading);
  let error = $derived($closeError);

  function formatPnl(value: number): string {
    return value >= 0 ? `+$${value.toFixed(2)}` : `-$${Math.abs(value).toFixed(2)}`;
  }

  function formatLot(lot: number): string {
    return lot.toFixed(2);
  }

  async function handleConfirm() {
    if (!confirmed) {
      confirmed = true;
      return;
    }

    try {
      const result = await closePosition(position.ticket, position.bot_id);
      if (result && onSuccess) {
        onSuccess(result);
      }
      onClose();
    } catch (e) {
      console.error('[PositionClose] Failed to close position:', e);
    }
  }

  function handleCancel() {
    confirmed = false;
    onClose();
  }

  function handleOverlayClick(e: MouseEvent) {
    if (e.target === e.currentTarget) {
      handleCancel();
    }
  }
</script>

{#if position}
  <!-- svelte-ignore a11y_click_events_have_key_events a11y_no_static_element_interactions -->
  <div class="modal-overlay" onclick={handleOverlayClick} role="dialog" aria-modal="true" aria-labelledby="close-modal-title">
    <!-- svelte-ignore a11y_click_events_have_key_events a11y_no_static_element_interactions -->
    <div class="modal-content" onclick={(e) => e.stopPropagation()}>
      <button class="close-btn" onclick={handleCancel} aria-label="Close">
        <X size={18} />
      </button>

      <div class="modal-header">
        <div class="modal-icon">
          <AlertTriangle size={28} />
        </div>
        <h3 id="close-modal-title">Close Position</h3>
        <p class="modal-subtitle">Review and confirm position closure</p>
      </div>

      {#if error}
        <div class="error-banner">
          {error}
        </div>
      {/if}

      <div class="position-details">
        <div class="detail-row">
          <span class="detail-label">Symbol</span>
          <span class="detail-value symbol">{position.symbol}</span>
        </div>

        <div class="detail-row">
          <span class="detail-label">Direction</span>
          <span class="detail-value direction" class:buy={position.direction === 'buy'} class:sell={position.direction === 'sell'}>
            {#if position.direction === 'buy'}
              <TrendingUp size={14} />
            {:else}
              <TrendingDown size={14} />
            {/if}
            {position.direction.toUpperCase()}
          </span>
        </div>

        <div class="detail-row">
          <span class="detail-label">Lot Size</span>
          <span class="detail-value">{formatLot(position.lot)}</span>
        </div>

        <div class="detail-row">
          <span class="detail-label">Current P&L</span>
          <span class="detail-value pnl" class:positive={position.current_pnl > 0} class:negative={position.current_pnl < 0}>
            {formatPnl(position.current_pnl)}
          </span>
        </div>

        <div class="detail-row">
          <span class="detail-label">Ticket</span>
          <span class="detail-value ticket">#{position.ticket}</span>
        </div>
      </div>

      <div class="confirmation-step">
        {#if !confirmed}
          <p class="step-text">
            Click <strong>Confirm Close</strong> to acknowledge the position closure.
          </p>
        {:else}
          <p class="step-text final">
            Final confirmation — Click again to <strong>CLOSE POSITION</strong>
          </p>
        {/if}
      </div>

      <div class="modal-actions">
        <button class="btn-cancel" onclick={handleCancel} disabled={loading}>
          Cancel
        </button>
        <button
          class="btn-confirm"
          class:confirmed
          onclick={handleConfirm}
          disabled={loading}
        >
          {#if loading}
            <Loader2 size={16} class="spinner" />
            Closing...
          {:else if !confirmed}
            Confirm Close
          {:else}
            CLOSE POSITION
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
    background: var(--bg-secondary, #0d1117);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 24px;
    width: 90%;
    max-width: 420px;
    position: relative;
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.6);
  }

  .close-btn {
    position: absolute;
    top: 16px;
    right: 16px;
    background: transparent;
    border: none;
    color: #888;
    cursor: pointer;
    padding: 8px;
    border-radius: 8px;
    transition: all 0.15s ease;
    display: flex;
    align-items: center;
    justify-content: center;
  }

  .close-btn:hover {
    background: rgba(255, 255, 255, 0.08);
    color: #e0e0e0;
  }

  .modal-header {
    text-align: center;
    margin-bottom: 20px;
  }

  .modal-icon {
    display: flex;
    justify-content: center;
    margin-bottom: 12px;
    color: #f59e0b;
  }

  .modal-header h3 {
    font-family: var(--font-display, 'JetBrains Mono', monospace);
    font-weight: 700;
    font-size: 20px;
    color: #e0e0e0;
    margin-bottom: 4px;
  }

  .modal-subtitle {
    font-size: 13px;
    color: #888;
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

  .position-details {
    background: rgba(8, 13, 20, 0.35);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(0, 212, 255, 0.08);
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 20px;
  }

  .detail-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 10px 0;
  }

  .detail-row:not(:last-child) {
    border-bottom: 1px solid rgba(255, 255, 255, 0.06);
  }

  .detail-label {
    font-size: 13px;
    color: #888;
  }

  .detail-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 14px;
    font-weight: 600;
    color: #e0e0e0;
    display: flex;
    align-items: center;
    gap: 6px;
  }

  .detail-value.symbol {
    color: #00d4ff;
    background: rgba(0, 212, 255, 0.1);
    padding: 2px 8px;
    border-radius: 4px;
  }

  .detail-value.direction.buy {
    color: #00c896;
  }

  .detail-value.direction.sell {
    color: #ff3b3b;
  }

  .detail-value.pnl.positive {
    color: #00c896;
  }

  .detail-value.pnl.negative {
    color: #ff3b3b;
  }

  .detail-value.ticket {
    color: #888;
  }

  .confirmation-step {
    background: rgba(8, 13, 20, 0.35);
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 24px;
    text-align: center;
  }

  .step-text {
    font-size: 14px;
    color: #e0e0e0;
    line-height: 1.5;
  }

  .step-text.final {
    color: #f59e0b;
    font-weight: 600;
  }

  .modal-actions {
    display: flex;
    gap: 12px;
    justify-content: center;
  }

  .btn-cancel,
  .btn-confirm {
    padding: 12px 24px;
    border-radius: 8px;
    font-family: var(--font-nav, 'JetBrains Mono', monospace);
    font-size: 14px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.15s ease;
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .btn-cancel {
    background: rgba(8, 13, 20, 0.35);
    border: 1px solid rgba(255, 255, 255, 0.08);
    color: #888;
  }

  .btn-cancel:hover:not(:disabled) {
    background: rgba(255, 255, 255, 0.08);
    color: #e0e0e0;
  }

  .btn-confirm {
    background: #f59e0b;
    border: none;
    color: #000;
  }

  .btn-confirm:hover:not(:disabled) {
    background: #fbbf24;
    box-shadow: 0 0 20px rgba(245, 158, 11, 0.4);
  }

  .btn-confirm.confirmed {
    background: #d97706;
    animation: pulse-button 0.8s ease-in-out infinite;
  }

  @keyframes pulse-button {
    0%, 100% {
      box-shadow: 0 0 0 0 rgba(217, 119, 6, 0.5);
    }
    50% {
      box-shadow: 0 0 20px 5px rgba(217, 119, 6, 0.4);
    }
  }

  .btn-cancel:disabled,
  .btn-confirm:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  :global(.spinner) {
    animation: spin 0.8s linear infinite;
  }

  @keyframes spin {
    to {
      transform: rotate(360deg);
    }
  }
</style>
