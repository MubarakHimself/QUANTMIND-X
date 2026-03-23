<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { AlertTriangle, DollarSign, TrendingDown, Zap } from 'lucide-svelte';
  import { createTradingClient } from '$lib/ws-client';
  import type { WebSocketClient } from '$lib/ws-client';
  import { API_BASE } from '$lib/constants';

  // Use configured API base or default to same origin
  const apiBase = API_BASE || '';

  // Interface matching backend response
  interface FeeBreakdownItem {
    bot_id: string;
    trades: number;
    fees_paid: number;
    fee_pct: number;
  }

  interface FeeData {
    daily_fees: number;
    daily_fee_burn_pct: number;
    kill_switch_active: boolean;
    fee_breakdown: FeeBreakdownItem[];
  }

  // Reactive state with defaults
  let feeData: FeeData = $state({
    daily_fees: 0,
    daily_fee_burn_pct: 0,
    kill_switch_active: false,
    fee_breakdown: []
  });

  let wsClient: WebSocketClient | null = null;
  let isLoading = $state(true);
  let hasError = $state(false);

  onMount(async () => {
    try {
      // Build absolute URL for REST calls
      const baseUrl = apiBase || window.location.origin;

      // Fetch initial data from REST API
      const response = await fetch(`${baseUrl}/api/router/fee-monitor`);

      if (response.ok) {
        feeData = await response.json();
      } else {
        hasError = true;
      }

      // Connect to WebSocket for real-time updates
      wsClient = await createTradingClient(baseUrl);

      // Subscribe to fee update events
      wsClient.on('fee_update', (message) => {
        if (message.data) {
          feeData = message.data as FeeData;
        }
      });

      isLoading = false;
    } catch (error) {
      console.error('Failed to load fee monitoring data:', error);
      hasError = true;
      isLoading = false;
    }
  });

  onDestroy(() => {
    if (wsClient) {
      wsClient.disconnect();
    }
  });

  // Helper functions
  function formatCurrency(value: number): string {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    }).format(value);
  }

  function formatPercentage(value: number): string {
    return value.toFixed(2) + '%';
  }

  // Check if fee burn is critical (> 10%)
  function isCritical(): boolean {
    return feeData.daily_fee_burn_pct > 10;
  }

  // Check if fee burn is warning (> 5%)
  function isWarning(): boolean {
    return feeData.daily_fee_burn_pct > 5;
  }

  // Get gauge color based on fee burn percentage
  function getGaugeColor(): string {
    if (feeData.daily_fee_burn_pct > 10) return '#ef4444'; // red
    if (feeData.daily_fee_burn_pct > 5) return '#f59e0b'; // amber
    return '#10b981'; // green
  }

  // Get gauge width (max 100%)
  function getGaugeWidth(): number {
    return Math.min(feeData.daily_fee_burn_pct, 100);
  }
</script>

<div class="fee-monitor-panel">
  <!-- Header -->
  <div class="panel-header">
    <div class="header-left">
      <DollarSign size={20} />
      <h3>Fee Monitoring</h3>
    </div>
    {#if isLoading}
      <span class="loading-indicator">Loading...</span>
    {/if}
  </div>

  {#if hasError}
    <div class="error-state">
      <AlertTriangle size={24} />
      <span>Failed to load fee data</span>
    </div>
  {:else}
    <div class="panel-content">
      <!-- Fee Burn Gauge -->
      <div class="fee-burn-section">
        <div class="gauge-container">
          <div class="gauge-header">
            <span class="gauge-label">Daily Fee Burn</span>
          </div>
          <div class="gauge-value" class:critical={isCritical()} class:warning={isWarning() && !isCritical()}>
            {formatPercentage(feeData.daily_fee_burn_pct)}
          </div>
          <div class="gauge-bar">
            <div
              class="gauge-fill"
              style="width: {getGaugeWidth()}%; background: {getGaugeColor()};"
            ></div>
          </div>
          <div class="gauge-markers">
            <span>0%</span>
            <span>5%</span>
            <span>10%</span>
            <span>Max</span>
          </div>
        </div>

        <!-- Total Fees -->
        <div class="total-fees">
          <span class="fees-label">Total Fees Today</span>
          <span class="fees-value">{formatCurrency(feeData.daily_fees)}</span>
        </div>
      </div>

      <!-- Kill Switch Status -->
      <div class="kill-switch-status" class:active={feeData.kill_switch_active}>
        {#if feeData.kill_switch_active}
          <Zap size={18} />
          <span class="status-text">KILL SWITCH ACTIVE</span>
        {:else}
          <span class="status-dot"></span>
          <span class="status-text">Normal Operation</span>
        {/if}
      </div>

      <!-- Fee Breakdown Table -->
      <div class="fee-breakdown">
        <h4>Fee Breakdown by Bot</h4>
        {#if feeData.fee_breakdown.length > 0}
          <table class="breakdown-table">
            <thead>
              <tr>
                <th>Bot ID</th>
                <th class="text-center">Trades</th>
                <th class="text-right">Fees Paid</th>
                <th class="text-right">Fee %</th>
              </tr>
            </thead>
            <tbody>
              {#each feeData.fee_breakdown as item (item.bot_id)}
                <tr class:warning-row={item.fee_pct > 5}>
                  <td class="bot-id">{item.bot_id}</td>
                  <td class="text-center">{item.trades}</td>
                  <td class="text-right">{formatCurrency(item.fees_paid)}</td>
                  <td class="text-right" class:warning={item.fee_pct > 5}>
                    {formatPercentage(item.fee_pct)}
                    {#if item.fee_pct > 5}
                      <span class="warning-icon" title="High fee burn">!</span>
                    {/if}
                  </td>
                </tr>
              {/each}
            </tbody>
          </table>
        {:else}
          <div class="no-data">
            <TrendingDown size={24} />
            <span>No bots with fees today</span>
          </div>
        {/if}
      </div>
    </div>
  {/if}
</div>

<style>
  .fee-monitor-panel {
    background: var(--color-bg-surface);
    border: 1px solid var(--color-border-subtle);
    border-radius: 10px;
    padding: 16px;
    margin-top: 16px;
  }

  .panel-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 16px;
    padding-bottom: 12px;
    border-bottom: 1px solid var(--color-border-subtle);
  }

  .header-left {
    display: flex;
    align-items: center;
    gap: 8px;
    color: var(--color-accent-cyan);
  }

  .header-left h3 {
    margin: 0;
    font-size: 14px;
    font-weight: 600;
    color: var(--color-text-primary);
  }

  .loading-indicator {
    font-size: 12px;
    color: var(--color-text-muted);
  }

  .error-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 8px;
    padding: 24px;
    color: var(--color-accent-red);
  }

  .panel-content {
    display: flex;
    flex-direction: column;
    gap: 16px;
  }

  /* Fee Burn Section */
  .fee-burn-section {
    display: flex;
    gap: 24px;
    align-items: flex-start;
  }

  .gauge-container {
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .gauge-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .gauge-label {
    font-size: 11px;
    color: var(--color-text-muted);
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  .gauge-value {
    font-size: 28px;
    font-weight: 700;
    color: #10b981;
    transition: color 0.3s ease;
  }

  .gauge-value.warning {
    color: #f59e0b;
  }

  .gauge-value.critical {
    color: #ef4444;
    animation: pulse-glow 1.5s ease-in-out infinite;
  }

  @keyframes pulse-glow {
    0%, 100% {
      opacity: 1;
    }
    50% {
      opacity: 0.7;
    }
  }

  .gauge-bar {
    height: 8px;
    background: var(--color-bg-elevated);
    border-radius: 4px;
    overflow: hidden;
  }

  .gauge-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.5s ease, background 0.3s ease;
  }

  .gauge-markers {
    display: flex;
    justify-content: space-between;
    font-size: 10px;
    color: var(--color-text-muted);
  }

  .total-fees {
    display: flex;
    flex-direction: column;
    align-items: flex-end;
    gap: 4px;
  }

  .fees-label {
    font-size: 11px;
    color: var(--color-text-muted);
  }

  .fees-value {
    font-size: 20px;
    font-weight: 600;
    color: var(--color-text-primary);
  }

  /* Kill Switch Status */
  .kill-switch-status {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 10px 14px;
    border-radius: 6px;
    background: rgba(16, 185, 129, 0.1);
    border: 1px solid rgba(16, 185, 129, 0.3);
    transition: all 0.3s ease;
  }

  .kill-switch-status.active {
    background: rgba(239, 68, 68, 0.15);
    border-color: rgba(239, 68, 68, 0.5);
  }

  .status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #10b981;
  }

  .kill-switch-status.active .status-dot {
    display: none;
  }

  .kill-switch-status.active :global(svg) {
    color: #ef4444;
  }

  .status-text {
    font-size: 13px;
    font-weight: 500;
    color: #10b981;
  }

  .kill-switch-status.active .status-text {
    color: #ef4444;
  }

  /* Fee Breakdown Table */
  .fee-breakdown {
    margin-top: 8px;
  }

  .fee-breakdown h4 {
    margin: 0 0 12px;
    font-size: 12px;
    font-weight: 600;
    color: var(--color-text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  .breakdown-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
  }

  .breakdown-table th {
    padding: 8px 10px;
    text-align: left;
    font-weight: 600;
    color: var(--color-text-muted);
    border-bottom: 1px solid var(--color-border-subtle);
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.3px;
  }

  .breakdown-table td {
    padding: 10px;
    border-bottom: 1px solid var(--color-border-subtle);
    color: var(--color-text-primary);
  }

  .breakdown-table tr:last-child td {
    border-bottom: none;
  }

  .breakdown-table tr:hover {
    background: var(--color-bg-elevated);
  }

  .breakdown-table tr.warning-row {
    background: rgba(245, 158, 11, 0.05);
  }

  .breakdown-table tr.warning-row:hover {
    background: rgba(245, 158, 11, 0.1);
  }

  .bot-id {
    font-weight: 500;
    font-family: 'Monaco', 'Courier New', monospace;
    font-size: 11px;
  }

  .text-center {
    text-align: center;
  }

  .text-right {
    text-align: right;
    font-family: 'Monaco', 'Courier New', monospace;
  }

  .warning {
    color: #f59e0b;
    font-weight: 500;
  }

  .warning-icon {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 14px;
    height: 14px;
    border-radius: 50%;
    background: #f59e0b;
    color: #000;
    font-size: 10px;
    font-weight: 700;
    margin-left: 4px;
    font-family: sans-serif;
  }

  .no-data {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 8px;
    padding: 24px;
    color: var(--color-text-muted);
    font-size: 12px;
  }

  /* Responsive */
  @media (max-width: 640px) {
    .fee-burn-section {
      flex-direction: column;
      gap: 16px;
    }

    .total-fees {
      align-items: flex-start;
    }

    .gauge-value {
      font-size: 24px;
    }
  }
</style>
