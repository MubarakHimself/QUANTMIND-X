<script lang="ts">
  import { Activity, Server, RefreshCw, AlertTriangle, CheckCircle, XCircle, Clock } from 'lucide-svelte';
  import { fireServerHealthAlert, clearServerHealthAlert, type ServerHealthAlert } from '$lib/stores/serverHealthAlerts';

  interface NodeMetrics {
    cpu: number;
    memory: number;
    disk: number;
    latency_ms: number;
    uptime_seconds: number;
    last_heartbeat: string;
    status: string;
  }

  interface ServerHealthResponse {
    contabo: NodeMetrics;
    cloudzy: NodeMetrics;
    timestamp: string;
  }

  let healthData: ServerHealthResponse | null = $state(null);
  let isLoading = $state(true);
  let error = $state<string | null>(null);
  let lastUpdated = $state<string | null>(null);

  // Threshold constants
  const CPU_THRESHOLD = 85;
  const MEMORY_THRESHOLD = 90;
  const DISK_THRESHOLD = 90;
  const LATENCY_THRESHOLD = 500;

  // Polling interval
  let pollInterval: ReturnType<typeof setInterval> | null = null;
  const POLL_INTERVAL = 10000; // 10 seconds

  // Track which node+metric combos are currently breached to avoid alert spam on every poll
  const currentlyBreached = new Set<string>();

  /**
   * Check each metric against its threshold. Fire a serverHealthAlertEvent only
   * when a metric transitions into the critical state (not on every poll while breached).
   * AC3: "Contabo: disk usage at 91%. Action recommended."
   */
  function checkThresholdBreaches(data: ServerHealthResponse) {
    const nodes: Array<{ name: string; metrics: NodeMetrics }> = [
      { name: 'Contabo', metrics: data.contabo },
      { name: 'Cloudzy', metrics: data.cloudzy }
    ];

    for (const { name, metrics } of nodes) {
      if (metrics.status === 'disconnected' || metrics.status === 'unknown') continue;

      const checks: Array<{ key: string; metric: string; value: number; threshold: number; unit: string }> = [
        { key: `${name}-cpu`, metric: 'CPU', value: metrics.cpu, threshold: CPU_THRESHOLD, unit: '%' },
        { key: `${name}-memory`, metric: 'Memory', value: metrics.memory, threshold: MEMORY_THRESHOLD, unit: '%' },
        { key: `${name}-disk`, metric: 'Disk', value: metrics.disk, threshold: DISK_THRESHOLD, unit: '%' },
        { key: `${name}-latency`, metric: 'Latency', value: metrics.latency_ms, threshold: LATENCY_THRESHOLD, unit: 'ms' }
      ];

      for (const check of checks) {
        const isCritical = check.value >= check.threshold;
        if (isCritical && !currentlyBreached.has(check.key)) {
          // Metric just crossed the threshold — fire the alert
          currentlyBreached.add(check.key);
          clearServerHealthAlert(); // clear first so store fires even if same node+metric fires twice
          const displayValue = check.unit === 'ms'
            ? check.value.toFixed(0)
            : check.value.toFixed(0);
          const alert: ServerHealthAlert = {
            id: `${check.key}-${Date.now()}`,
            node: name,
            metric: check.metric,
            value: check.value,
            unit: check.unit,
            message: `${name}: ${check.metric.toLowerCase()} usage at ${displayValue}${check.unit}. Action recommended.`,
            timestamp: new Date()
          };
          fireServerHealthAlert(alert);
        } else if (!isCritical) {
          // Metric recovered — clear so it can alert again if it breaches later
          currentlyBreached.delete(check.key);
        }
      }
    }
  }

  async function loadHealthData() {
    error = null;
    try {
      const response = await fetch('/api/server/health/metrics');
      if (response.ok) {
        healthData = await response.json();
        lastUpdated = new Date().toLocaleTimeString();
        checkThresholdBreaches(healthData!);
      } else {
        error = 'Failed to load server health data';
      }
    } catch (e) {
      error = 'Failed to load server health data';
      console.error(e);
    } finally {
      isLoading = false;
    }
  }

  function getStatusColor(status: string): string {
    switch (status) {
      case 'healthy': return '#22c55e';
      case 'warning': return '#f59e0b';
      case 'critical': return '#ff3b3b';
      case 'disconnected': return '#9ca3af';
      default: return '#9ca3af';
    }
  }

  function getMetricStatus(value: number, threshold: number): string {
    if (value >= threshold) return 'critical';
    if (value >= threshold * 0.8) return 'warning';
    return 'healthy';
  }

  function formatUptime(seconds: number): string {
    if (seconds === 0) return 'N/A';
    const days = Math.floor(seconds / 86400);
    const hours = Math.floor((seconds % 86400) / 3600);
    const mins = Math.floor((seconds % 3600) / 60);

    if (days > 0) return `${days}d ${hours}h`;
    if (hours > 0) return `${hours}h ${mins}m`;
    return `${mins}m`;
  }

  function formatHeartbeat(timestamp: string): string {
    if (!timestamp) return 'N/A';
    try {
      const date = new Date(timestamp);
      return date.toLocaleTimeString();
    } catch {
      return 'N/A';
    }
  }

  function startPolling() {
    if (pollInterval) return;
    loadHealthData();
    pollInterval = setInterval(loadHealthData, POLL_INTERVAL);
  }

  function stopPolling() {
    if (pollInterval) {
      clearInterval(pollInterval);
      pollInterval = null;
    }
  }

  // Load on mount and start polling
  import { onMount, onDestroy } from 'svelte';
  onMount(() => {
    startPolling();
  });

  onDestroy(() => {
    stopPolling();
  });
</script>

<div class="panel">
  <div class="panel-header">
    <h3>Server Health</h3>
    <div class="header-actions">
      <span class="last-updated">Updated: {lastUpdated || '-'}</span>
      <button class="icon-btn" onclick={loadHealthData} title="Refresh">
        <RefreshCw size={16} class={isLoading ? 'spinning' : ''} />
      </button>
    </div>
  </div>

  {#if error}
    <div class="alert error">
      <AlertTriangle size={16} />
      <span>{error}</span>
    </div>
  {/if}

  {#if isLoading && !healthData}
    <div class="loading">
      <RefreshCw size={24} class="spinning" />
      <span>Loading server health...</span>
    </div>
  {:else if healthData}
    <div class="nodes-grid">
      <!-- Contabo Node -->
      <div class="node-card">
        <div class="node-header">
          <div class="node-title">
            <Server size={20} />
            <h4>Contabo</h4>
          </div>
          <div class="node-status" style="color: {getStatusColor(healthData.contabo.status)}">
            {#if healthData.contabo.status === 'healthy'}
              <CheckCircle size={16} />
            {:else if healthData.contabo.status === 'critical'}
              <XCircle size={16} />
            {:else}
              <AlertTriangle size={16} />
            {/if}
            <span>{healthData.contabo.status}</span>
          </div>
        </div>

        <div class="metrics-list">
          <div class="metric" class:critical={getMetricStatus(healthData.contabo.cpu, CPU_THRESHOLD) === 'critical'}>
            <span class="metric-label">CPU</span>
            <div class="metric-bar">
              <div class="metric-fill" style="width: {Math.min(healthData.contabo.cpu, 100)}%; background: {getMetricStatus(healthData.contabo.cpu, CPU_THRESHOLD) === 'critical' ? '#ff3b3b' : getMetricStatus(healthData.contabo.cpu, CPU_THRESHOLD) === 'warning' ? '#f59e0b' : '#22c55e'}"></div>
            </div>
            <span class="metric-value">{healthData.contabo.cpu.toFixed(1)}%</span>
          </div>

          <div class="metric" class:critical={getMetricStatus(healthData.contabo.memory, MEMORY_THRESHOLD) === 'critical'}>
            <span class="metric-label">Memory</span>
            <div class="metric-bar">
              <div class="metric-fill" style="width: {Math.min(healthData.contabo.memory, 100)}%; background: {getMetricStatus(healthData.contabo.memory, MEMORY_THRESHOLD) === 'critical' ? '#ff3b3b' : getMetricStatus(healthData.contabo.memory, MEMORY_THRESHOLD) === 'warning' ? '#f59e0b' : '#22c55e'}"></div>
            </div>
            <span class="metric-value">{healthData.contabo.memory.toFixed(1)}%</span>
          </div>

          <div class="metric" class:critical={getMetricStatus(healthData.contabo.disk, DISK_THRESHOLD) === 'critical'}>
            <span class="metric-label">Disk</span>
            <div class="metric-bar">
              <div class="metric-fill" style="width: {Math.min(healthData.contabo.disk, 100)}%; background: {getMetricStatus(healthData.contabo.disk, DISK_THRESHOLD) === 'critical' ? '#ff3b3b' : getMetricStatus(healthData.contabo.disk, DISK_THRESHOLD) === 'warning' ? '#f59e0b' : '#22c55e'}"></div>
            </div>
            <span class="metric-value">{healthData.contabo.disk.toFixed(1)}%</span>
          </div>

          <div class="metric" class:critical={healthData.contabo.latency_ms > LATENCY_THRESHOLD}>
            <span class="metric-label">Latency</span>
            <span class="metric-value">{healthData.contabo.latency_ms.toFixed(1)}ms</span>
          </div>

          <div class="metric">
            <span class="metric-label">Uptime</span>
            <span class="metric-value">{formatUptime(healthData.contabo.uptime_seconds)}</span>
          </div>

          <div class="metric">
            <span class="metric-label">Last Heartbeat</span>
            <span class="metric-value">{formatHeartbeat(healthData.contabo.last_heartbeat)}</span>
          </div>
        </div>
      </div>

      <!-- Cloudzy Node -->
      <div class="node-card">
        <div class="node-header">
          <div class="node-title">
            <Activity size={20} />
            <h4>Cloudzy</h4>
          </div>
          <div class="node-status" style="color: {getStatusColor(healthData.cloudzy.status)}">
            {#if healthData.cloudzy.status === 'healthy'}
              <CheckCircle size={16} />
            {:else if healthData.cloudzy.status === 'critical'}
              <XCircle size={16} />
            {:else if healthData.cloudzy.status === 'disconnected'}
              <XCircle size={16} />
            {:else}
              <AlertTriangle size={16} />
            {/if}
            <span>{healthData.cloudzy.status}</span>
          </div>
        </div>

        <div class="metrics-list">
          <div class="metric" class:critical={getMetricStatus(healthData.cloudzy.cpu, CPU_THRESHOLD) === 'critical'}>
            <span class="metric-label">CPU</span>
            <div class="metric-bar">
              <div class="metric-fill" style="width: {Math.min(healthData.cloudzy.cpu, 100)}%; background: {getMetricStatus(healthData.cloudzy.cpu, CPU_THRESHOLD) === 'critical' ? '#ff3b3b' : getMetricStatus(healthData.cloudzy.cpu, CPU_THRESHOLD) === 'warning' ? '#f59e0b' : '#22c55e'}"></div>
            </div>
            <span class="metric-value">{healthData.cloudzy.cpu.toFixed(1)}%</span>
          </div>

          <div class="metric" class:critical={getMetricStatus(healthData.cloudzy.memory, MEMORY_THRESHOLD) === 'critical'}>
            <span class="metric-label">Memory</span>
            <div class="metric-bar">
              <div class="metric-fill" style="width: {Math.min(healthData.cloudzy.memory, 100)}%; background: {getMetricStatus(healthData.cloudzy.memory, MEMORY_THRESHOLD) === 'critical' ? '#ff3b3b' : getMetricStatus(healthData.cloudzy.memory, MEMORY_THRESHOLD) === 'warning' ? '#f59e0b' : '#22c55e'}"></div>
            </div>
            <span class="metric-value">{healthData.cloudzy.memory.toFixed(1)}%</span>
          </div>

          <div class="metric" class:critical={getMetricStatus(healthData.cloudzy.disk, DISK_THRESHOLD) === 'critical'}>
            <span class="metric-label">Disk</span>
            <div class="metric-bar">
              <div class="metric-fill" style="width: {Math.min(healthData.cloudzy.disk, 100)}%; background: {getMetricStatus(healthData.cloudzy.disk, DISK_THRESHOLD) === 'critical' ? '#ff3b3b' : getMetricStatus(healthData.cloudzy.disk, DISK_THRESHOLD) === 'warning' ? '#f59e0b' : '#22c55e'}"></div>
            </div>
            <span class="metric-value">{healthData.cloudzy.disk.toFixed(1)}%</span>
          </div>

          <div class="metric" class:critical={healthData.cloudzy.latency_ms > LATENCY_THRESHOLD}>
            <span class="metric-label">Latency</span>
            <span class="metric-value">{healthData.cloudzy.latency_ms.toFixed(1)}ms</span>
          </div>

          <div class="metric">
            <span class="metric-label">Uptime</span>
            <span class="metric-value">{formatUptime(healthData.cloudzy.uptime_seconds)}</span>
          </div>

          <div class="metric">
            <span class="metric-label">Last Heartbeat</span>
            <span class="metric-value">{formatHeartbeat(healthData.cloudzy.last_heartbeat)}</span>
          </div>
        </div>
      </div>
    </div>
  {/if}
</div>

<style>
  .panel {
    padding: 0;
  }

  .panel-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 16px;
  }

  .panel-header h3 {
    margin: 0;
    font-size: 15px;
    font-weight: 600;
    color: var(--text-primary, #e8eaf0);
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }

  .header-actions {
    display: flex;
    align-items: center;
    gap: 12px;
  }

  .last-updated {
    font-size: 11px;
    color: rgba(255, 255, 255, 0.3);
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }

  .icon-btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 30px;
    height: 30px;
    border: none;
    border-radius: 6px;
    background: rgba(255, 255, 255, 0.04);
    color: rgba(255, 255, 255, 0.4);
    cursor: pointer;
    transition: all 0.15s;
  }

  .icon-btn:hover {
    background: rgba(255, 255, 255, 0.1);
    color: #e8eaf0;
  }

  .spinning {
    animation: spin 1s linear infinite;
  }

  @keyframes spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
  }

  .alert {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 10px 12px;
    border-radius: 6px;
    font-size: 12px;
    margin-bottom: 12px;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }

  .alert.error {
    background: rgba(255, 59, 59, 0.1);
    border: 1px solid rgba(255, 59, 59, 0.25);
    color: #ff3b3b;
  }

  .loading {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 40px;
    color: rgba(255, 255, 255, 0.3);
    gap: 12px;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    font-size: 12px;
  }

  .nodes-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
  }

  @media (max-width: 768px) {
    .nodes-grid {
      grid-template-columns: 1fr;
    }
  }

  .node-card {
    background: rgba(8, 13, 20, 0.35);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 8px;
    padding: 14px;
  }

  .node-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 14px;
    padding-bottom: 10px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.05);
  }

  .node-title {
    display: flex;
    align-items: center;
    gap: 8px;
    color: rgba(255, 255, 255, 0.75);
  }

  .node-title h4 {
    margin: 0;
    font-size: 13px;
    font-weight: 600;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }

  .node-status {
    display: flex;
    align-items: center;
    gap: 4px;
    font-size: 11px;
    font-weight: 600;
    text-transform: capitalize;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }

  .metrics-list {
    display: flex;
    flex-direction: column;
    gap: 10px;
  }

  .metric {
    display: grid;
    grid-template-columns: 72px 1fr 56px;
    align-items: center;
    gap: 8px;
  }

  .metric-label {
    font-size: 11px;
    color: rgba(255, 255, 255, 0.35);
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }

  .metric-value {
    font-size: 12px;
    font-weight: 500;
    color: rgba(255, 255, 255, 0.7);
    text-align: right;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }

  .metric.critical .metric-value {
    color: #ff3b3b;
  }

  .metric-bar {
    height: 4px;
    background: rgba(255, 255, 255, 0.07);
    border-radius: 2px;
    overflow: hidden;
  }

  .metric-fill {
    height: 100%;
    border-radius: 2px;
    transition: width 0.3s ease;
  }
</style>
