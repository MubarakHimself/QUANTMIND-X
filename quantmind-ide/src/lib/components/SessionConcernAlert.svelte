<script lang="ts">
  import { AlertTriangle, Bot, TrendingDown } from 'lucide-svelte';

  interface Props {
    botId: string;
    strategyName?: string;
    consecutiveNegativeEv?: number;
    threshold?: number;
  }

  let {
    botId,
    strategyName = '',
    consecutiveNegativeEv = 0,
    threshold = 3
  }: Props = $props();

  function formatCount(count: number): string {
    return count === 1 ? '1 session' : `${count} sessions`;
  }
</script>

<div class="session-concern-alert">
  <div class="alert-icon">
    <AlertTriangle size={16} />
  </div>
  <div class="alert-content">
    <div class="alert-header">
      <span class="alert-title">SESSION CONCERN</span>
      <TrendingDown size={12} />
    </div>
    <div class="bot-info">
      <Bot size={14} />
      <span class="bot-id">{botId}</span>
      {#if strategyName}
        <span class="strategy-name">{strategyName}</span>
      {/if}
    </div>
    <div class="concern-details">
      <span class="concern-message">
        {formatCount(consecutiveNegativeEv)} with negative EV
      </span>
      <span class="threshold-note">
        (threshold: {threshold} consecutive sessions)
      </span>
    </div>
  </div>
</div>

<style>
  .session-concern-alert {
    display: flex;
    gap: 10px;
    padding: 10px 12px;
    background: rgba(239, 68, 68, 0.1);
    border: 1px solid rgba(239, 68, 68, 0.3);
    border-radius: 8px;
  }

  .alert-icon {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 32px;
    height: 32px;
    background: rgba(239, 68, 68, 0.2);
    border-radius: 6px;
    color: #ef4444;
    flex-shrink: 0;
  }

  .alert-content {
    flex: 1;
    min-width: 0;
  }

  .alert-header {
    display: flex;
    align-items: center;
    gap: 6px;
    margin-bottom: 4px;
    color: #ef4444;
  }

  .alert-title {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.5px;
  }

  .bot-info {
    display: flex;
    align-items: center;
    gap: 6px;
    margin-bottom: 6px;
    color: #9ca3af;
  }

  .bot-id {
    font-size: 13px;
    font-weight: 600;
    font-family: monospace;
    color: #e4e4e7;
  }

  .strategy-name {
    font-size: 11px;
    color: #6b7280;
    padding: 2px 6px;
    background: rgba(255, 255, 255, 0.05);
    border-radius: 4px;
  }

  .concern-details {
    display: flex;
    flex-direction: column;
    gap: 2px;
  }

  .concern-message {
    font-size: 12px;
    color: #ef4444;
    font-weight: 500;
  }

  .threshold-note {
    font-size: 10px;
    color: #6b7280;
  }
</style>
