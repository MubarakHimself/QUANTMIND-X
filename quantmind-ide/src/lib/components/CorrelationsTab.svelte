<script lang="ts">
  import { Layers, AlertTriangle } from 'lucide-svelte';


  interface Props {
    correlations: Array<{
    pair: string;
    value: number;
    status: 'ok' | 'warning' | 'danger';
  }>;
    getScoreColor: (score: number) => string;
  }

  let { correlations, getScoreColor }: Props = $props();
</script>

<div class="correlations-section">
  <div class="section-header">
    <h3>Symbol Correlations</h3>
    <span class="info">Active positions with high correlation are limited</span>
  </div>

  <div class="correlations-grid">
    {#if correlations.length > 0}
      {#each correlations as corr}
        <div class="corr-card status-{corr.status}">
          <div class="corr-pair">
            <Layers size={16} />
            <span>{corr.pair}</span>
          </div>
          <div class="corr-value">
            <span class="value" style="color: {getScoreColor(Math.abs(corr.value) * 10)}">
              {corr.value > 0 ? '+' : ''}{corr.value.toFixed(2)}
            </span>
            <span class="label">{corr.status}</span>
          </div>
        </div>
      {/each}
    {:else}
      <div class="empty-state">No live correlation data is available.</div>
    {/if}
  </div>

  <div class="correlation-info">
    <AlertTriangle size={14} />
    <p>High correlation (>=0.7) limits simultaneous positions in correlated pairs</p>
  </div>
</div>

<style>
  .correlations-section {
    display: flex;
    flex-direction: column;
    gap: 16px;
  }

  .section-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .section-header h3 {
    margin: 0;
    font-size: 14px;
    color: var(--color-text-primary);
  }

  .info {
    font-size: 12px;
    color: var(--color-text-muted);
  }

  .correlations-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 12px;
  }

  .corr-card {
    background: var(--color-bg-surface);
    border: 1px solid var(--color-border-subtle);
    border-radius: 10px;
    padding: 16px;
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  .corr-card.status-warning {
    border-color: rgba(245, 158, 11, 0.3);
  }

  .corr-card.status-danger {
    border-color: rgba(239, 68, 68, 0.3);
  }

  .corr-pair {
    display: flex;
    align-items: center;
    gap: 8px;
    color: var(--color-text-primary);
    font-weight: 500;
  }

  .corr-value {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .corr-value .value {
    font-size: 20px;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
  }

  .corr-value .label {
    padding: 2px 8px;
    border-radius: 10px;
    font-size: 10px;
    text-transform: uppercase;
  }

  .status-ok .label {
    background: rgba(16, 185, 129, 0.2);
    color: #10b981;
  }

  .status-warning .label {
    background: rgba(245, 158, 11, 0.2);
    color: #f59e0b;
  }

  .status-danger .label {
    background: rgba(239, 68, 68, 0.2);
    color: #ef4444;
  }

  .correlation-info {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 12px 16px;
    background: rgba(245, 158, 11, 0.1);
    border: 1px solid rgba(245, 158, 11, 0.2);
    border-radius: 8px;
    color: #f59e0b;
    font-size: 12px;
  }

  .correlation-info p {
    margin: 0;
  }

  .empty-state {
    grid-column: 1 / -1;
    padding: 20px 16px;
    border: 1px dashed var(--color-border-subtle);
    border-radius: 10px;
    color: var(--color-text-muted);
    font-size: 12px;
  }
</style>
