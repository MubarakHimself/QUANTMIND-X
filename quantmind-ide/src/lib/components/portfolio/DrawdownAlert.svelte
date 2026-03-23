<script lang="ts">
  /**
   * DrawdownAlert - Red Alert Banner for Portfolio Drawdown
   *
   * Shows when portfolio drawdown exceeds 10%
   * Includes dismiss functionality and Copilot notification trigger
   */
  import { AlertTriangle, X, MessageSquare } from 'lucide-svelte';
  import { portfolioStore } from '$lib/stores/portfolio';

  interface DrawdownAlertData {
    active: boolean;
    percent: number;
  }

  interface Props {
    alert: DrawdownAlertData;
  }

  let { alert }: Props = $props();

  function formatPercent(value: number): string {
    return `${value.toFixed(1)}%`;
  }

  function handleDismiss() {
    portfolioStore.dismissDrawdownAlert();
  }

  async function handleNotifyCopilot() {
    // In production, this would trigger a Copilot notification
    // For now, we can trigger via the chat store or emit an event
    console.log(`Copilot notification: Portfolio drawdown alert: ${formatPercent(alert.percent)}`);
  }
</script>

{#if alert.active}
  <div class="drawdown-alert">
    <div class="alert-content">
      <AlertTriangle size={18} />
      <div class="alert-text">
        <span class="alert-title">Portfolio Drawdown Alert</span>
        <span class="alert-value">Current drawdown: {formatPercent(alert.percent)}</span>
      </div>
    </div>

    <div class="alert-actions">
      <button class="action-btn copilot-btn" onclick={handleNotifyCopilot} title="Notify Copilot">
        <MessageSquare size={14} />
      </button>
      <button class="action-btn dismiss-btn" onclick={handleDismiss} title="Dismiss">
        <X size={14} />
      </button>
    </div>
  </div>
{/if}

<style>
  .drawdown-alert {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 16px;
    background: rgba(255, 59, 59, 0.15);
    border: 1px solid rgba(255, 59, 59, 0.3);
    border-radius: 8px;
    animation: slideIn 0.3s ease;
  }

  @keyframes slideIn {
    from {
      opacity: 0;
      transform: translateY(-10px);
    }
    to {
      opacity: 1;
      transform: translateY(0);
    }
  }

  .alert-content {
    display: flex;
    align-items: center;
    gap: 12px;
    color: #ff3b3b;
  }

  .alert-text {
    display: flex;
    flex-direction: column;
    gap: 2px;
  }

  .alert-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    font-weight: 600;
    color: #ff3b3b;
  }

  .alert-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: #ff6b6b;
  }

  .alert-actions {
    display: flex;
    gap: 8px;
  }

  .action-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 6px;
    background: transparent;
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 4px;
    color: #888;
    cursor: pointer;
    transition: all 0.15s ease;
  }

  .action-btn:hover {
    background: rgba(255, 255, 255, 0.08);
    color: #e0e0e0;
  }

  .copilot-btn:hover {
    border-color: rgba(0, 212, 255, 0.3);
    color: #00d4ff;
  }

  .dismiss-btn:hover {
    border-color: rgba(255, 59, 59, 0.3);
    color: #ff3b3b;
  }
</style>