<script lang="ts">
  /**
   * CloseAllModal - Summary Modal for Close All Positions
   *
   * Double-confirmation flow and displays results per position.
   * Follows Frosted Terminal aesthetic (Tier 2 glass).
   */
  import { X, AlertTriangle, Loader2, CheckCircle, AlertCircle, XCircle } from 'lucide-svelte';
  import { closeAllPositions, closeLoading, closeError, type CloseAllResult } from '$lib/stores/trading';

  interface Props {
    positions: PositionInfo[];
    botId?: string;
    onClose: () => void;
    onSuccess?: (result: CloseAllResult) => void;
  }

  interface PositionInfo {
    ticket: number;
    bot_id: string;
    symbol: string;
    direction: 'buy' | 'sell';
    lot: number;
    current_pnl: number;
  }

  let { positions, botId, onClose, onSuccess }: Props = $props();

  let confirmed = $state(false);
  let loading = $derived($closeLoading);
  let error = $derived($closeError);
  let result: CloseAllResult | null = $state(null);

  function formatPnl(value: number): string {
    return value >= 0 ? `+$${value.toFixed(2)}` : `-$${Math.abs(value).toFixed(2)}`;
  }

  async function handleConfirm() {
    if (!confirmed) {
      confirmed = true;
      return;
    }

    try {
      result = await closeAllPositions(botId);
      if (result && onSuccess) {
        onSuccess(result);
      }
      // Don't auto-close on success - show results
    } catch (e) {
      console.error('[CloseAll] Failed to close positions:', e);
    }
  }

  function handleCancel() {
    confirmed = false;
    result = null;
    onClose();
  }

  function handleOverlayClick(e: MouseEvent) {
    if (e.target === e.currentTarget) {
      handleCancel();
    }
  }

  function getStatusIcon(status: string) {
    switch (status) {
      case 'filled':
        return CheckCircle;
      case 'partial':
        return AlertCircle;
      case 'rejected':
        return XCircle;
      default:
        return AlertCircle;
    }
  }

  function getStatusColor(status: string): string {
    switch (status) {
      case 'filled':
        return '#00c896';
      case 'partial':
        return '#f59e0b';
      case 'rejected':
        return '#ff3b3b';
      default:
        return '#888';
    }
  }
</script>

<!-- svelte-ignore a11y_click_events_have_key_events a11y_no_static_element_interactions -->
<div class="modal-overlay" onclick={handleOverlayClick} role="dialog" aria-modal="true" aria-labelledby="close-all-modal-title">
  <!-- svelte-ignore a11y_click_events_have_key_events a11y_no_static_element_interactions -->
  <div class="modal-content" onclick={(e) => e.stopPropagation()}>
    <button class="close-btn" onclick={handleCancel} aria-label="Close">
      <X size={18} />
    </button>

    <div class="modal-header">
      <div class="modal-icon danger">
        <XCircle size={32} />
      </div>
      <h3 id="close-all-modal-title">Close All Positions</h3>
      <p class="modal-subtitle">{botId ? `Close all positions for bot` : 'Close all open positions across all bots'}</p>
    </div>

    {#if error}
      <div class="error-banner">
        {error}
      </div>
    {/if}

    {#if result}
      <!-- Show results -->
      <div class="results-summary">
        <h4>Close Results</h4>
        <div class="results-grid">
          {#each result.results as item}
            {@const StatusIcon = getStatusIcon(item.status)}
            <div class="result-item" class:filled={item.status === 'filled'} class:partial={item.status === 'partial'} class:rejected={item.status === 'rejected'}>
              <div class="result-icon" style="color: {getStatusColor(item.status)}">
                <svelte:component this={StatusIcon} size={20} />
              </div>
              <div class="result-details">
                <span class="result-ticket">#{item.position_ticket}</span>
                <span class="result-status" style="color: {getStatusColor(item.status)}">{item.status}</span>
                {#if item.filled_price}
                  <span class="result-price">@{item.filled_price}</span>
                {/if}
              </div>
            </div>
          {/each}
        </div>

        <div class="summary-stats">
          <div class="stat">
            <span class="stat-label">Filled</span>
            <span class="stat-value filled">{result.results.filter(r => r.status === 'filled').length}</span>
          </div>
          <div class="stat">
            <span class="stat-label">Partial</span>
            <span class="stat-value partial">{result.results.filter(r => r.status === 'partial').length}</span>
          </div>
          <div class="stat">
            <span class="stat-label">Rejected</span>
            <span class="stat-value rejected">{result.results.filter(r => r.status === 'rejected').length}</span>
          </div>
        </div>
      </div>

      <div class="modal-actions">
        <button class="btn-done" onclick={onClose}>
          Done
        </button>
      </div>
    {:else}
      <!-- Show confirmation -->
      <div class="warning-box">
        <div class="warning-icon">
          <AlertTriangle size={20} />
        </div>
        <span>This will attempt to close {positions.length} position{positions.length !== 1 ? 's' : ''}</span>
      </div>

      <div class="positions-preview">
        <h4>Positions to Close</h4>
        <div class="positions-list">
          {#each positions as pos}
            <div class="position-item">
              <span class="pos-symbol">{pos.symbol}</span>
              <span class="pos-pnl" class:positive={pos.current_pnl > 0} class:negative={pos.current_pnl < 0}>
                {formatPnl(pos.current_pnl)}
              </span>
            </div>
          {/each}
        </div>
      </div>

      <div class="confirmation-step">
        {#if !confirmed}
          <p class="step-text">
            Click <strong>Confirm Close All</strong> once to acknowledge.
          </p>
          <p class="step-text secondary">
            A second click will execute the close orders.
          </p>
        {:else}
          <p class="step-text final">
            Final confirmation — Click again to <strong>CLOSE ALL POSITIONS</strong>
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
            Confirm Close All
          {:else}
            CLOSE ALL POSITIONS
          {/if}
        </button>
      </div>
    {/if}
  </div>
</div>

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
    max-width: 480px;
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
  }

  .modal-icon.danger {
    color: #f59e0b;
  }

  .modal-header h3 {
    font-family: var(--font-display, 'JetBrains Mono', monospace);
    font-weight: 700;
    font-size: 22px;
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

  .warning-box {
    display: flex;
    align-items: center;
    gap: 12px;
    background: rgba(245, 158, 11, 0.1);
    border: 1px solid rgba(245, 158, 11, 0.3);
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 20px;
  }

  .warning-icon {
    flex-shrink: 0;
    color: #f59e0b;
  }

  .warning-box span {
    font-size: 14px;
    font-weight: 600;
    color: #f59e0b;
  }

  .positions-preview {
    background: rgba(8, 13, 20, 0.35);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(0, 212, 255, 0.08);
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 20px;
    max-height: 200px;
    overflow-y: auto;
  }

  .positions-preview h4 {
    font-size: 12px;
    color: #888;
    margin-bottom: 12px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  .positions-list {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .position-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px;
    background: rgba(0, 0, 0, 0.2);
    border-radius: 6px;
  }

  .pos-symbol {
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    color: #00d4ff;
  }

  .pos-pnl {
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    font-weight: 600;
  }

  .pos-pnl.positive {
    color: #00c896;
  }

  .pos-pnl.negative {
    color: #ff3b3b;
  }

  .results-summary {
    background: rgba(8, 13, 20, 0.35);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(0, 212, 255, 0.08);
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 20px;
    max-height: 300px;
    overflow-y: auto;
  }

  .results-summary h4 {
    font-size: 14px;
    color: #e0e0e0;
    margin-bottom: 16px;
  }

  .results-grid {
    display: flex;
    flex-direction: column;
    gap: 8px;
    margin-bottom: 16px;
  }

  .result-item {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 10px;
    background: rgba(0, 0, 0, 0.2);
    border-radius: 8px;
  }

  .result-icon {
    flex-shrink: 0;
  }

  .result-details {
    display: flex;
    align-items: center;
    gap: 12px;
    flex-wrap: wrap;
  }

  .result-ticket {
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    color: #e0e0e0;
  }

  .result-status {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
  }

  .result-price {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    color: #888;
  }

  .summary-stats {
    display: flex;
    justify-content: space-around;
    padding-top: 16px;
    border-top: 1px solid rgba(255, 255, 255, 0.08);
  }

  .stat {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 4px;
  }

  .stat-label {
    font-size: 11px;
    color: #888;
    text-transform: uppercase;
  }

  .stat-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 18px;
    font-weight: 700;
  }

  .stat-value.filled {
    color: #00c896;
  }

  .stat-value.partial {
    color: #f59e0b;
  }

  .stat-value.rejected {
    color: #ff3b3b;
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

  .step-text.secondary {
    font-size: 12px;
    color: #888;
    margin-top: 8px;
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
  .btn-confirm,
  .btn-done {
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

  .btn-done {
    background: #00c896;
    border: none;
    color: #000;
    width: 100%;
  }

  .btn-done:hover {
    background: #00e0aa;
    box-shadow: 0 0 20px rgba(0, 200, 150, 0.4);
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
