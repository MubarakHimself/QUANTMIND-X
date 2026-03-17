<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { createEventDispatcher } from 'svelte';
  import {
    Activity, Cpu, HardDrive, Wifi, Zap, Bot, TrendingUp,
    Database, RefreshCw, Settings, Maximize2, Minimize2
  } from 'lucide-svelte';

  import MetricCard from './MetricCard.svelte';
  import TickStreamChart from './charts/TickStreamChart.svelte';
  import ResourceUsageChart from './charts/ResourceUsageChart.svelte';
  import AlertsPanel from './AlertsPanel.svelte';

  import {
    metricsWebSocket,
    systemMetrics,
    tradingMetrics,
    databaseMetrics,
    tickStreamMetrics,
    connectionState,
    lastUpdate,
    isHealthy
  } from '../services/metricsWebSocket';

  const dispatch = createEventDispatcher();

  // Time range for charts
  let chartTimeRange: '1m' | '5m' | '15m' | '1h' = $state('5m');

  // Auto-refresh state
  let isFullscreen = $state(false);
  let refreshInterval: ReturnType<typeof setInterval> | null = null;


  function getConnectionColor(state: string): string {
    switch (state) {
      case 'connected': return 'var(--accent-success)';
      case 'connecting':
      case 'reconnecting': return 'var(--accent-warning)';
      default: return 'var(--accent-danger)';
    }
  }

  function formatTime(date: Date): string {
    return new Date(date).toLocaleTimeString('en-US', {
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
      hour12: false
    });
  }

  function toggleFullscreen() {
    isFullscreen = !isFullscreen;
    dispatch('fullscreen', { isFullscreen });
  }

  function handleAcknowledge(event: CustomEvent) {
    const { alert } = event.detail;
    metricsWebSocket.acknowledgeAlert(alert.id);
  }

  function handleClearAll() {
    metricsWebSocket.clearAlerts();
  }

  function handleClear(event: CustomEvent) {
    const { alert } = event.detail;
    metricsWebSocket.removeAlert(alert.id);
  }

  onMount(() => {
    // Connect to WebSocket
    metricsWebSocket.connect();

    // Request notification permission
    metricsWebSocket.requestNotificationPermission();
  });

  onDestroy(() => {
    // Keep connection alive, don't disconnect on component destroy
    // as other components might use it
  });
  // Connection status display
  let connectionStatus = $derived($connectionState);
  let connectionColor = $derived(getConnectionColor($connectionState));
  let formattedLastUpdate = $derived(formatTime($lastUpdate));
</script>

<div class="monitoring-dashboard" class:fullscreen={isFullscreen}>
  <!-- Header -->
  <div class="dashboard-header">
    <div class="header-left">
      <Activity size={20} />
      <h2>System Monitoring</h2>
      <div class="connection-status" style="background: {connectionColor}">
        <span>{connectionStatus}</span>
      </div>
    </div>
    <div class="header-right">
      <span class="last-update">Last update: {formattedLastUpdate}</span>
      <div class="time-range-selector">
        {#each ['1m', '5m', '15m', '1h'] as range (range)}
          <button
            class:active={chartTimeRange === range}
            onclick={() => chartTimeRange = range}
          >
            {range}
          </button>
        {/each}
      </div>
      <button class="icon-btn" onclick={toggleFullscreen} title="Toggle Fullscreen">
        {#if isFullscreen}
          <Minimize2 size={18} />
        {:else}
          <Maximize2 size={18} />
        {/if}
      </button>
    </div>
  </div>

  <!-- Main Content -->
  <div class="dashboard-content">
    <!-- Left Column - Metrics -->
    <div class="metrics-column">
      <!-- System Metrics -->
      <div class="metrics-section">
        <h3 class="section-title">
          <Cpu size={16} />
          System Resources
        </h3>
        <div class="metrics-grid">
          <MetricCard
            title="CPU Usage"
            value={$systemMetrics.cpu_usage}
            unit="%"
            icon={Cpu}
            trend={$systemMetrics.cpu_usage > 80 ? 'up' : 'neutral'}
            threshold={{ warning: 70, critical: 90 }}
          />
          <MetricCard
            title="Memory"
            value={$systemMetrics.memory_usage}
            unit="%"
            icon={HardDrive}
            trend={$systemMetrics.memory_usage > 80 ? 'up' : 'neutral'}
            threshold={{ warning: 70, critical: 90 }}
          />
          <MetricCard
            title="Network In"
            value={$systemMetrics.network_in}
            unit="MB"
            icon={Wifi}
          />
          <MetricCard
            title="Chaos Score"
            value={$systemMetrics.chaos_score}
            trend={$systemMetrics.chaos_score > 0.5 ? 'up' : 'down'}
            threshold={{ warning: 0.5, critical: 0.8 }}
          />
        </div>
      </div>

      <!-- Trading Metrics -->
      <div class="metrics-section">
        <h3 class="section-title">
          <TrendingUp size={16} />
          Trading
        </h3>
        <div class="metrics-grid">
          <MetricCard
            title="Tick Latency"
            value={$tradingMetrics.tick_latency_ms}
            unit="ms"
            icon={Zap}
            trend={$tradingMetrics.tick_latency_ms > 50 ? 'up' : 'neutral'}
            threshold={{ warning: 50, critical: 100 }}
          />
          <MetricCard
            title="Active Bots"
            value={$tradingMetrics.active_bots}
            icon={Bot}
          />
          <MetricCard
            title="Positions"
            value={$tradingMetrics.active_positions}
            icon={TrendingUp}
          />
          <MetricCard
            title="Daily P&L"
            value={$tradingMetrics.daily_pnl}
            trend={$tradingMetrics.daily_pnl >= 0 ? 'up' : 'down'}
          />
        </div>
      </div>

      <!-- Database Metrics -->
      <div class="metrics-section">
        <h3 class="section-title">
          <Database size={16} />
          Database
        </h3>
        <div class="metrics-grid small">
          <MetricCard
            title="Query Latency"
            value={$databaseMetrics.query_latency_ms}
            unit="ms"
            icon={Database}
            threshold={{ warning: 50, critical: 100 }}
          />
          <MetricCard
            title="Connections"
            value={$databaseMetrics.active_connections}
            icon={Database}
          />
        </div>
      </div>
    </div>

    <!-- Middle Column - Charts -->
    <div class="charts-column">
      <div class="chart-section">
        <h3 class="section-title">
          <Activity size={16} />
          Tick Stream Rate
        </h3>
        <div class="chart-container">
          <TickStreamChart height={200} timeRange={chartTimeRange} />
        </div>
      </div>

      <div class="chart-section">
        <h3 class="section-title">
          <Cpu size={16} />
          Resource Usage
        </h3>
        <div class="chart-container">
          <ResourceUsageChart height={200} timeRange={chartTimeRange} />
        </div>
      </div>
    </div>

    <!-- Right Column - Alerts -->
    <div class="alerts-column">
      <AlertsPanel
        maxHeight={600}
        showFilters={true}
        showAcknowledge={true}
        on:acknowledge={handleAcknowledge}
        on:clear={handleClear}
        on:clearAll={handleClearAll}
      />
    </div>
  </div>

  <!-- Health Status Bar -->
  <div class="health-bar" class:healthy={$isHealthy} class:unhealthy={!$isHealthy}>
    <div class="health-indicator">
      {#if $isHealthy}
        <span class="status-dot healthy"></span>
        <span>System Healthy</span>
      {:else}
        <span class="status-dot unhealthy"></span>
        <span>System Issues Detected</span>
      {/if}
    </div>
    <div class="health-metrics">
      <span>CPU: {$systemMetrics.cpu_usage}%</span>
      <span>Mem: {$systemMetrics.memory_usage}%</span>
      <span>Latency: {$tradingMetrics.tick_latency_ms}ms</span>
    </div>
  </div>
</div>

<style>
  .monitoring-dashboard {
    display: flex;
    flex-direction: column;
    height: 100%;
    background: var(--bg-primary);
    overflow: hidden;
  }

  .monitoring-dashboard.fullscreen {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    z-index: 9999;
    border-radius: 0;
  }

  .dashboard-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 20px;
    background: var(--bg-secondary);
    border-bottom: 1px solid var(--border-subtle);
  }

  .header-left {
    display: flex;
    align-items: center;
    gap: 12px;
  }

  .header-left h2 {
    margin: 0;
    font-size: 16px;
    font-weight: 600;
    color: var(--text-primary);
  }

  .connection-status {
    padding: 4px 10px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: 500;
    color: #000;
    text-transform: capitalize;
  }

  .header-right {
    display: flex;
    align-items: center;
    gap: 16px;
  }

  .last-update {
    font-size: 12px;
    color: var(--text-muted);
  }

  .time-range-selector {
    display: flex;
    gap: 4px;
    padding: 4px;
    background: var(--bg-tertiary);
    border-radius: 6px;
  }

  .time-range-selector button {
    padding: 4px 10px;
    background: transparent;
    border: none;
    border-radius: 4px;
    font-size: 12px;
    color: var(--text-muted);
    cursor: pointer;
    transition: all 0.2s ease;
  }

  .time-range-selector button:hover {
    color: var(--text-primary);
  }

  .time-range-selector button.active {
    background: var(--accent-primary);
    color: #000;
  }

  .icon-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 32px;
    height: 32px;
    background: transparent;
    border: 1px solid var(--border-subtle);
    border-radius: 6px;
    color: var(--text-secondary);
    cursor: pointer;
    transition: all 0.2s ease;
  }

  .icon-btn:hover {
    background: var(--bg-tertiary);
    color: var(--text-primary);
  }

  .dashboard-content {
    display: grid;
    grid-template-columns: 1fr 1.5fr 1fr;
    gap: 16px;
    flex: 1;
    padding: 16px;
    overflow: auto;
  }

  .metrics-column,
  .charts-column,
  .alerts-column {
    display: flex;
    flex-direction: column;
    gap: 16px;
  }

  .metrics-section,
  .chart-section {
    background: var(--bg-secondary);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
    padding: 16px;
  }

  .section-title {
    display: flex;
    align-items: center;
    gap: 8px;
    margin: 0 0 16px 0;
    font-size: 14px;
    font-weight: 600;
    color: var(--text-primary);
  }

  .metrics-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 12px;
  }

  .metrics-grid.small {
    grid-template-columns: repeat(2, 1fr);
  }

  .chart-container {
    position: relative;
    min-height: 200px;
  }

  .health-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 20px;
    background: var(--bg-secondary);
    border-top: 1px solid var(--border-subtle);
  }

  .health-bar.healthy {
    border-top-color: var(--accent-success);
  }

  .health-bar.unhealthy {
    border-top-color: var(--accent-danger);
    background: rgba(244, 67, 54, 0.05);
  }

  .health-indicator {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 13px;
    font-weight: 500;
  }

  .status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    animation: pulse 2s infinite;
  }

  .status-dot.healthy {
    background: var(--accent-success);
  }

  .status-dot.unhealthy {
    background: var(--accent-danger);
  }

  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
  }

  .health-metrics {
    display: flex;
    gap: 24px;
    font-size: 12px;
    color: var(--text-muted);
  }

  /* Responsive adjustments */
  @media (max-width: 1200px) {
    .dashboard-content {
      grid-template-columns: 1fr 1fr;
    }

    .alerts-column {
      grid-column: span 2;
    }
  }

  @media (max-width: 768px) {
    .dashboard-content {
      grid-template-columns: 1fr;
    }

    .alerts-column {
      grid-column: span 1;
    }
  }
</style>
