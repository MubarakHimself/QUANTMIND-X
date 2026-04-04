<script lang="ts">
  import { Award, TrendingUp } from 'lucide-svelte';

  interface Props {
    botId: string;
    strategyName?: string;
    outperformance?: number;
    regimeExpectation?: number;
    actualPerformance?: number;
  }

  let { botId, strategyName = '', outperformance = 0, regimeExpectation = 0, actualPerformance = 0 }: Props = $props();

  function formatPercent(value: number): string {
    return (value * 100).toFixed(1) + '%';
  }
</script>

<div class="session-specialist-badge">
  <div class="badge-icon">
    <Award size={16} />
  </div>
  <div class="badge-content">
    <div class="badge-header">
      <span class="badge-title">SESSION SPECIALIST</span>
      <TrendingUp size={12} />
    </div>
    <div class="bot-info">
      <span class="bot-id">{botId}</span>
      {#if strategyName}
        <span class="strategy-name">{strategyName}</span>
      {/if}
    </div>
    <div class="performance-details">
      <div class="detail">
        <span class="label">Regime Exp.</span>
        <span class="value">{formatPercent(regimeExpectation)}</span>
      </div>
      <div class="detail">
        <span class="label">Actual</span>
        <span class="value highlight">{formatPercent(actualPerformance)}</span>
      </div>
      <div class="detail">
        <span class="label">Outperformance</span>
        <span class="value positive">+{formatPercent(outperformance)}</span>
      </div>
    </div>
  </div>
</div>

<style>
  .session-specialist-badge {
    display: flex;
    gap: 10px;
    padding: 10px 12px;
    background: rgba(16, 185, 129, 0.1);
    border: 1px solid rgba(16, 185, 129, 0.3);
    border-radius: 8px;
  }

  .badge-icon {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 32px;
    height: 32px;
    background: rgba(16, 185, 129, 0.2);
    border-radius: 6px;
    color: #10b981;
    flex-shrink: 0;
  }

  .badge-content {
    flex: 1;
    min-width: 0;
  }

  .badge-header {
    display: flex;
    align-items: center;
    gap: 6px;
    margin-bottom: 4px;
    color: #10b981;
  }

  .badge-title {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.5px;
  }

  .bot-info {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 6px;
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

  .performance-details {
    display: flex;
    gap: 12px;
  }

  .detail {
    display: flex;
    flex-direction: column;
    gap: 1px;
  }

  .label {
    font-size: 9px;
    color: #6b7280;
    text-transform: uppercase;
  }

  .value {
    font-size: 12px;
    font-weight: 500;
    color: #9ca3af;
  }

  .value.highlight {
    color: #10b981;
  }

  .value.positive {
    color: #10b981;
  }
</style>
