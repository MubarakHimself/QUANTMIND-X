<!-- @migration-task Error while migrating Svelte code: This type of directive is not valid on components
https://svelte.dev/e/component_invalid_directive -->
<!-- @migration-task Error while migrating Svelte code: This type of directive is not valid on components
https://svelte.dev/e/component_invalid_directive -->
<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { createEventDispatcher } from 'svelte';
  import { 
    Activity, Radar, Zap, AlertTriangle, Clock, RefreshCw, 
    Settings, Maximize2, Minimize2, TrendingUp, Bot, CheckCircle,
    XCircle, Bell
  } from 'lucide-svelte';

  const dispatch = createEventDispatcher();

  // Scanner alert types
  type ScanType = 'SESSION_BREAKOUT' | 'VOLATILITY_SPIKE' | 'NEWS_EVENT' | 'ICT_SETUP';
  type AlertPriority = 'low' | 'medium' | 'high' | 'critical';

  interface ScannerAlert {
    id: number;
    type: ScanType;
    symbol: string;
    session: string;
    setup: string;
    confidence: number;
    recommended_bots: string[];
    metadata: Record<string, any>;
    timestamp: string;
    priority: AlertPriority;
    status: 'active' | 'expired' | 'triggered';
  }

  interface ScannerStatus {
    isRunning: boolean;
    lastScan: string;
    nextScan: string;
    activeAlerts: number;
  }

  let alerts: ScannerAlert[] = [];
  let status: ScannerStatus = {
    isRunning: false,
    lastScan: 'Never',
    nextScan: 'N/A',
    activeAlerts: 0
  };
  let isLoading = false;
  let error: string | null = null;
  let isFullscreen = false;
  let ws: WebSocket | null = null;
  let refreshInterval: ReturnType<typeof setInterval> | null = null;

  // Filter state
  let filterType: ScanType | 'all' = 'all';
  let filterSymbol: string | 'all' = 'all';

  $: filteredAlerts = alerts.filter(alert => {
    if (filterType !== 'all' && alert.type !== filterType) return false;
    if (filterSymbol !== 'all' && alert.symbol !== filterSymbol) return false;
    return true;
  });

  $: symbols = [...new Set(alerts.map(a => a.symbol))];

  onMount(() => {
    fetchScannerData();
    connectWebSocket();
    refreshInterval = setInterval(fetchScannerData, 30000);
  });

  onDestroy(() => {
    if (ws) ws.close();
    if (refreshInterval) clearInterval(refreshInterval);
  });

  async function fetchScannerData() {
    isLoading = true;
    error = null;
    
    try {
      const [alertsRes, statusRes] = await Promise.all([
        fetch('/api/market-scanner/alerts'),
        fetch('/api/market-scanner/status')
      ]);
      
      if (!alertsRes.ok || !statusRes.ok) {
        throw new Error('Failed to fetch scanner data');
      }
      
      alerts = await alertsRes.json();
      status = await statusRes.json();
    } catch (e) {
      console.error('Failed to fetch scanner data:', e);
      error = e instanceof Error ? e.message : 'Failed to fetch data';
    } finally {
      isLoading = false;
    }
  }

  function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;
    
    ws = new WebSocket(wsUrl);
    
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === 'market_opportunity') {
          alerts = [data.alert, ...alerts].slice(0, 100);
          status.activeAlerts = alerts.filter(a => a.status === 'active').length;
        }
      } catch (e) {
        console.error('WebSocket message parse error:', e);
      }
    };
  }

  function getTypeColor(type: ScanType): string {
    switch (type) {
      case 'SESSION_BREAKOUT': return 'var(--scan-breakout, #3b82f6)';
      case 'VOLATILITY_SPIKE': return 'var(--scan-volatility, #f59e0b)';
      case 'NEWS_EVENT': return 'var(--scan-news, #ef4444)';
      case 'ICT_SETUP': return 'var(--scan-ict, #10b981)';
      default: return 'var(--scan-default, #6b7280)';
    }
  }

  function getTypeIcon(type: ScanType) {
    switch (type) {
      case 'SESSION_BREAKOUT': return TrendingUp;
      case 'VOLATILITY_SPIKE': return AlertTriangle;
      case 'NEWS_EVENT': return Bell;
      case 'ICT_SETUP': return Activity;
      default: return Radar;
    }
  }

  function getPriorityColor(priority: AlertPriority): string {
    switch (priority) {
      case 'critical': return 'var(--priority-critical, #dc2626)';
      case 'high': return 'var(--priority-high, #f97316)';
      case 'medium': return 'var(--priority-medium, #eab308)';
      case 'low': return 'var(--priority-low, #6b7280)';
      default: return 'var(--priority-default, #9ca3af)';
    }
  }

  function getConfidenceColor(confidence: number): string {
    if (confidence >= 0.8) return 'var(--confidence-high, #10b981)';
    if (confidence >= 0.5) return 'var(--confidence-medium, #f59e0b)';
    return 'var(--confidence-low, #ef4444)';
  }

  function getConfidenceLabel(confidence: number): string {
    if (confidence >= 0.8) return 'High';
    if (confidence >= 0.5) return 'Medium';
    return 'Low';
  }

  function formatTimestamp(timestamp: string): string {
    const date = new Date(timestamp);
    return date.toLocaleString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit'
    });
  }

  function toggleFullscreen() {
    isFullscreen = !isFullscreen;
    dispatch('fullscreen', { isFullscreen });
  }

  async function activateBot(alert: ScannerAlert) {
    try {
      const response = await fetch('/api/market-scanner/activate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ alert_id: alert.id, bot_ids: alert.recommended_bots })
      });
      
      if (response.ok) {
        alert.status = 'triggered';
        alerts = alerts;
      }
    } catch (e) {
      console.error('Failed to activate bot:', e);
    }
  }
</script>

<div class="scanner-panel" class:fullscreen={isFullscreen}>
  <div class="panel-header">
    <div class="header-left">
      <Radar size={20} />
      <h3>Market Scanner</h3>
      <div class="status-indicator" class:active={status.isRunning}>
        <span class="pulse"></span>
        <span>{status.isRunning ? 'Active' : 'Inactive'}</span>
      </div>
    </div>
    <div class="header-right">
      <span class="scan-info">
        <Clock size={14} />
        Last: {status.lastScan}
      </span>
      <button class="icon-btn" on:click={fetchScannerData} title="Refresh">
        <RefreshCw size={16} class:spin={isLoading} />
      </button>
      <button class="icon-btn" on:click={toggleFullscreen} title={isFullscreen ? 'Minimize' : 'Maximize'}>
        {#if isFullscreen}
          <Minimize2 size={16} />
        {:else}
          <Maximize2 size={16} />
        {/if}
      </button>
    </div>
  </div>

  <div class="status-bar">
    <div class="status-item">
      <AlertTriangle size={14} />
      <span>Active: {status.activeAlerts}</span>
    </div>
    <div class="status-item">
      <Clock size={14} />
      <span>Next: {status.nextScan}</span>
    </div>
  </div>

  {#if error}
    <div class="error-banner">
      {error}
    </div>
  {/if}

  <div class="filters">
    <select bind:value={filterType} class="filter-select">
      <option value="all">All Types</option>
      <option value="SESSION_BREAKOUT">Session Breakout</option>
      <option value="VOLATILITY_SPIKE">Volatility Spike</option>
      <option value="NEWS_EVENT">News Event</option>
      <option value="ICT_SETUP">ICT Setup</option>
    </select>
    
    <select bind:value={filterSymbol} class="filter-select">
      <option value="all">All Symbols</option>
      {#each symbols as symbol}
        <option value={symbol}>{symbol}</option>
      {/each}
    </select>
  </div>

  <div class="alerts-list">
    {#if isLoading && alerts.length === 0}
      <div class="loading">Scanning market...</div>
    {:else if filteredAlerts.length === 0}
      <div class="empty">
        <Radar size={48} />
        <span>No opportunities detected</span>
      </div>
    {:else}
      {#each filteredAlerts as alert (alert.id)}
        <div 
          class="alert-card" 
          class:expired={alert.status === 'expired'}
          class:triggered={alert.status === 'triggered'}
        >
          <div class="alert-header">
            <div class="alert-type" style="background-color: {getTypeColor(alert.type)}">
              <svelte:component this={getTypeIcon(alert.type)} size={14} />
              <span>{alert.type.replace('_', ' ')}</span>
            </div>
            <div class="alert-priority" style="background-color: {getPriorityColor(alert.priority)}">
              {alert.priority}
            </div>
          </div>
          
          <div class="alert-body">
            <div class="alert-symbol">{alert.symbol}</div>
            <div class="alert-setup">{alert.setup}</div>
            <div class="alert-session">{alert.session}</div>
          </div>
          
          <div class="alert-metrics">
            <div class="confidence">
              <span class="confidence-label">Confidence</span>
              <div class="confidence-bar">
                <div 
                  class="confidence-fill" 
                  style="width: {alert.confidence * 100}%; background-color: {getConfidenceColor(alert.confidence)}"
                ></div>
              </div>
              <span class="confidence-value" style="color: {getConfidenceColor(alert.confidence)}">
                {Math.round(alert.confidence * 100)}% ({getConfidenceLabel(alert.confidence)})
              </span>
            </div>
          </div>
          
          {#if alert.recommended_bots.length > 0}
            <div class="recommended-bots">
              <Bot size={14} />
              <span class="bots-label">Recommended:</span>
              {#each alert.recommended_bots as bot}
                <span class="bot-tag">{bot}</span>
              {/each}
            </div>
          {/if}
          
          <div class="alert-footer">
            <span class="timestamp">{formatTimestamp(alert.timestamp)}</span>
            <div class="alert-actions">
              {#if alert.status === 'active'}
                <button class="action-btn activate" on:click={() => activateBot(alert)}>
                  <Zap size={14} />
                  Activate
                </button>
              {:else if alert.status === 'triggered'}
                <span class="status-triggered">
                  <CheckCircle size={14} />
                  Triggered
                </span>
              {:else}
                <span class="status-expired">
                  <XCircle size={14} />
                  Expired
                </span>
              {/if}
            </div>
          </div>
        </div>
      {/each}
    {/if}
  </div>
</div>

<style>
  .scanner-panel {
    background: var(--bg-secondary, #1e293b);
    border-radius: 8px;
    padding: 16px;
    display: flex;
    flex-direction: column;
    gap: 12px;
    height: 100%;
    overflow: hidden;
  }

  .scanner-panel.fullscreen {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    z-index: 1000;
    border-radius: 0;
  }

  .panel-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .header-left {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .header-left h3 {
    margin: 0;
    font-size: 16px;
    font-weight: 600;
  }

  .status-indicator {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 12px;
    color: var(--text-secondary, #94a3b8);
  }

  .status-indicator.active {
    color: var(--accent-success, #10b981);
  }

  .pulse {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--text-secondary, #94a3b8);
  }

  .status-indicator.active .pulse {
    background: var(--accent-success, #10b981);
    animation: pulse 2s infinite;
  }

  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
  }

  .header-right {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .scan-info {
    display: flex;
    align-items: center;
    gap: 4px;
    font-size: 12px;
    color: var(--text-secondary, #94a3b8);
  }

  .icon-btn {
    background: none;
    border: none;
    color: var(--text-secondary, #94a3b8);
    cursor: pointer;
    padding: 4px;
    border-radius: 4px;
    transition: all 0.2s;
  }

  .icon-btn:hover {
    background: var(--bg-hover, #334155);
    color: var(--text-primary, #f1f5f9);
  }

  :global(.spin) {
    animation: spin 1s linear infinite;
  }

  @keyframes spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
  }

  .status-bar {
    display: flex;
    gap: 16px;
    padding: 8px 12px;
    background: var(--bg-tertiary, #334155);
    border-radius: 6px;
  }

  .status-item {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 12px;
    color: var(--text-secondary, #94a3b8);
  }

  .error-banner {
    background: var(--accent-danger, #ef4444);
    color: white;
    padding: 8px 12px;
    border-radius: 4px;
    font-size: 12px;
  }

  .filters {
    display: flex;
    gap: 8px;
  }

  .filter-select {
    background: var(--bg-tertiary, #334155);
    color: var(--text-primary, #f1f5f9);
    border: 1px solid var(--border-color, #475569);
    border-radius: 4px;
    padding: 6px 10px;
    font-size: 12px;
    cursor: pointer;
  }

  .alerts-list {
    flex: 1;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .loading, .empty {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 12px;
    color: var(--text-secondary, #94a3b8);
    padding: 48px;
  }

  .alert-card {
    background: var(--bg-tertiary, #334155);
    border-radius: 8px;
    padding: 12px;
    display: flex;
    flex-direction: column;
    gap: 8px;
    border-left: 3px solid var(--accent-primary, #3b82f6);
  }

  .alert-card.expired {
    opacity: 0.6;
    border-left-color: var(--text-muted, #64748b);
  }

  .alert-card.triggered {
    border-left-color: var(--accent-success, #10b981);
  }

  .alert-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .alert-type {
    display: flex;
    align-items: center;
    gap: 4px;
    padding: 2px 8px;
    border-radius: 4px;
    color: white;
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
  }

  .alert-priority {
    padding: 2px 6px;
    border-radius: 4px;
    color: white;
    font-size: 9px;
    font-weight: 600;
    text-transform: uppercase;
  }

  .alert-body {
    display: flex;
    gap: 12px;
    align-items: baseline;
  }

  .alert-symbol {
    font-size: 18px;
    font-weight: 700;
    color: var(--text-primary, #f1f5f9);
  }

  .alert-setup {
    font-size: 13px;
    color: var(--text-primary, #f1f5f9);
  }

  .alert-session {
    font-size: 11px;
    color: var(--text-secondary, #94a3b8);
  }

  .alert-metrics {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .confidence {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .confidence-label {
    font-size: 11px;
    color: var(--text-secondary, #94a3b8);
    width: 70px;
  }

  .confidence-bar {
    flex: 1;
    height: 6px;
    background: var(--bg-hover, #475569);
    border-radius: 3px;
    overflow: hidden;
  }

  .confidence-fill {
    height: 100%;
    border-radius: 3px;
    transition: width 0.3s ease;
  }

  .confidence-value {
    font-size: 11px;
    font-weight: 600;
    width: 90px;
    text-align: right;
  }

  .recommended-bots {
    display: flex;
    align-items: center;
    gap: 6px;
    flex-wrap: wrap;
  }

  .bots-label {
    font-size: 11px;
    color: var(--text-secondary, #94a3b8);
  }

  .bot-tag {
    font-size: 10px;
    padding: 2px 6px;
    background: var(--bg-hover, #475569);
    border-radius: 4px;
    color: var(--accent-primary, #3b82f6);
  }

  .alert-footer {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding-top: 8px;
    border-top: 1px solid var(--border-color, #475569);
  }

  .timestamp {
    font-size: 10px;
    color: var(--text-muted, #64748b);
  }

  .alert-actions {
    display: flex;
    gap: 8px;
  }

  .action-btn {
    display: flex;
    align-items: center;
    gap: 4px;
    padding: 4px 8px;
    border-radius: 4px;
    font-size: 11px;
    cursor: pointer;
    border: none;
    transition: all 0.2s;
  }

  .action-btn.activate {
    background: var(--accent-primary, #3b82f6);
    color: white;
  }

  .action-btn.activate:hover {
    background: var(--accent-primary-hover, #2563eb);
  }

  .status-triggered {
    display: flex;
    align-items: center;
    gap: 4px;
    color: var(--accent-success, #10b981);
    font-size: 11px;
  }

  .status-expired {
    display: flex;
    align-items: center;
    gap: 4px;
    color: var(--text-muted, #64748b);
    font-size: 11px;
  }
</style>
