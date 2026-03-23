<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { Activity, TrendingUp, AlertCircle, RefreshCw, Play, Cloud, Database, Zap } from 'lucide-svelte';
  import { hmmStore, type HMMStatus, type HMMRegime } from '../../stores/hmmStore';
  import { API_BASE } from '$lib/constants';

  const apiBase = API_BASE || '';

  
  interface Props {
    // Current symbol and timeframe
    symbol?: string;
    timeframe?: string;
  }

  let { symbol = 'EURUSD', timeframe = 'H1' }: Props = $props();

  // Local state
  let status: HMMStatus | null = $state(null);
  let currentRegime: HMMRegime | null = $state(null);
  let isLoading = $state(false);
  let error: string | null = $state(null);
  let isSyncing = $state(false);
  let isTraining = $state(false);

  // Subscribe to store
  const unsubscribe = hmmStore.subscribe(state => {
    status = state.status;
    currentRegime = state.currentRegime;
    isLoading = state.isLoading;
    error = state.error;
  });

  onMount(() => {
    // Load initial data
    hmmStore.loadStatus();
    hmmStore.loadCurrentRegime(symbol, timeframe);
    hmmStore.startAutoRefresh(30000); // Refresh every 30 seconds
  });

  onDestroy(() => {
    unsubscribe();
    hmmStore.stopAutoRefresh();
  });

  // Actions
  async function handleRefresh() {
    await hmmStore.loadStatus();
    await hmmStore.loadCurrentRegime(symbol, timeframe);
  }

  async function handleSync() {
    isSyncing = true;
    try {
      await hmmStore.sync();
    } finally {
      isSyncing = false;
    }
  }

  async function handleTrain() {
    isTraining = true;
    try {
      await hmmStore.train(symbol);
    } finally {
      isTraining = false;
    }
  }

  // Helper functions
  function getRegimeLabel(regime: string): string {
    const labels: Record<string, string> = {
      'TRENDING_LOW_VOL': 'Trending (Low Vol)',
      'TRENDING_HIGH_VOL': 'Trending (High Vol)',
      'RANGING_LOW_VOL': 'Ranging (Low Vol)',
      'RANGING_HIGH_VOL': 'Ranging (High Vol)',
      'BREAKOUT': 'Breakout',
      'UNKNOWN': 'Unknown'
    };
    return labels[regime] || regime;
  }

  function getRegimeColor(regime: string): string {
    const colors: Record<string, string> = {
      'TRENDING_LOW_VOL': '#10b981',  // green
      'TRENDING_HIGH_VOL': '#3b82f6', // blue
      'RANGING_LOW_VOL': '#f59e0b',   // amber
      'RANGING_HIGH_VOL': '#ef4444',  // red
      'BREAKOUT': '#8b5cf6',          // purple
      'UNKNOWN': '#6b7280'            // gray
    };
    return colors[regime] || colors['UNKNOWN'];
  }

  function getModeLabel(mode: string): string {
    const labels: Record<string, string> = {
      'ising_only': 'Ising Only',
      'hmm_shadow': 'HMM Shadow Mode',
      'hmm_hybrid_20': 'Hybrid (20% HMM)',
      'hmm_hybrid_50': 'Hybrid (50% HMM)',
      'hmm_hybrid_80': 'Hybrid (80% HMM)',
      'hmm_only': 'HMM Only'
    };
    return labels[mode] || mode;
  }

  function formatTimestamp(timestamp: string | null): string {
    if (!timestamp) return 'Never';
    const date = new Date(timestamp);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60000);

    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    const diffHours = Math.floor(diffMins / 60);
    if (diffHours < 24) return `${diffHours}h ago`;
    return date.toLocaleDateString();
  }
</script>

<div class="hmm-dashboard">
  <!-- Header -->
  <div class="hmm-header">
    <div class="hmm-title">
      <Activity size={20} />
      <h3>Hidden Macro Model</h3>
    </div>
    <div class="hmm-actions">
      <button
        class="action-btn"
        class:loading={isLoading}
        onclick={handleRefresh}
        disabled={isLoading}
        title="Refresh status"
      >
        <RefreshCw size={16} class={isLoading ? 'spinning' : ''} />
      </button>
      <button
        class="action-btn sync-btn"
        class:loading={isSyncing}
        onclick={handleSync}
        disabled={isSyncing}
        title="Sync from Contabo"
      >
        <Cloud size={16} class={isSyncing ? 'spinning' : ''} />
        <span>Sync</span>
      </button>
    </div>
  </div>

  <!-- Error display -->
  {#if error}
    <div class="error-banner">
      <AlertCircle size={16} />
      <span>{error}</span>
    </div>
  {/if}

  <!-- Status Cards -->
  <div class="status-cards">
    <!-- Model Status Card -->
    <div class="status-card model-card">
      <div class="card-header">
        <Database size={18} />
        <span class="card-title">Model Status</span>
      </div>
      <div class="card-content">
        <div class="status-row">
          <span class="status-label">Loaded</span>
          <span class="status-value {status?.model_loaded ? 'success' : 'error'}">
            {status?.model_loaded ? 'Yes' : 'No'}
          </span>
        </div>
        <div class="status-row">
          <span class="status-label">Version</span>
          <span class="status-value">{status?.model_version || 'N/A'}</span>
        </div>
        <div class="status-row">
          <span class="status-label">Last Sync</span>
          <span class="status-value">{formatTimestamp(status?.last_sync || null)}</span>
        </div>
        <div class="status-row">
          <span class="status-label">Sync Status</span>
          <span class="status-value">
            {status?.sync_status || 'Unknown'}
          </span>
        </div>
      </div>
    </div>

    <!-- Current Regime Card -->
    <div class="status-card regime-card">
      <div class="card-header">
        <TrendingUp size={18} />
        <span class="card-title">Current Regime</span>
      </div>
      <div class="card-content">
        {#if currentRegime}
          <div class="regime-display" style="border-color: {getRegimeColor(currentRegime.regime)}">
            <span class="regime-name" style="color: {getRegimeColor(currentRegime.regime)}">
              {getRegimeLabel(currentRegime.regime)}
            </span>
            <div class="regime-details">
              <span class="regime-confidence">
                {Math.round(currentRegime.confidence * 100)}% confidence
              </span>
              <span class="regime-state">State: {currentRegime.state}</span>
            </div>
          </div>
        {:else}
          <div class="regime-display" style="border-color: #6b7280">
            <span class="regime-name" style="color: #6b7280">
              No regime data
            </span>
          </div>
        {/if}
        <div class="regime-symbol">
          {symbol} / {timeframe}
        </div>
      </div>
    </div>

    <!-- Deployment Mode Card -->
    <div class="status-card mode-card">
      <div class="card-header">
        <Zap size={18} />
        <span class="card-title">Deployment Mode</span>
      </div>
      <div class="card-content">
        <div class="mode-display">
          <span class="mode-name">{getModeLabel(status?.deployment_mode || 'ising_only')}</span>
          <span class="mode-weight">
            HMM Weight: {Math.round((status?.hmm_weight || 0) * 100)}%
          </span>
        </div>
        <div class="mode-indicators">
          <div class="mode-indicator" class:active={status?.shadow_mode_active}>
            <span class="indicator-dot"></span>
            <span>Shadow Mode</span>
          </div>
          <div class="mode-indicator" class:active={!status?.version_mismatch && status?.model_loaded}>
            <span class="indicator-dot"></span>
            <span>In Sync</span>
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- Training Section -->
  <div class="training-section">
    <div class="training-header">
      <h4>Model Training</h4>
      <button
        class="train-btn"
        class:loading={isTraining}
        onclick={handleTrain}
        disabled={isTraining || !status?.model_loaded}
      >
        <Play size={14} />
        <span>{isTraining ? 'Training...' : 'Start Training'}</span>
      </button>
    </div>
    <p class="training-hint">
      Train the HMM model on the latest market data from Contabo.
    </p>
  </div>

  <!-- Agreement Metrics (if available) -->
  {#if status?.agreement_metrics && Object.keys(status.agreement_metrics).length > 0}
    <div class="agreement-section">
      <h4>Ising vs HMM Agreement</h4>
      <div class="agreement-metrics">
        {#each Object.entries(status.agreement_metrics) as [key, value]}
          <div class="metric-item">
            <span class="metric-key">{key}</span>
            <span class="metric-value">{typeof value === 'number' ? value.toFixed(2) : value}</span>
          </div>
        {/each}
      </div>
    </div>
  {/if}
</div>

<style>
  .hmm-dashboard {
    display: flex;
    flex-direction: column;
    gap: 16px;
    padding: 16px;
    background: var(--color-bg-surface);
    border: 1px solid var(--color-border-subtle);
    border-radius: 12px;
  }

  .hmm-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .hmm-title {
    display: flex;
    align-items: center;
    gap: 8px;
    color: var(--color-text-primary);
  }

  .hmm-title h3 {
    margin: 0;
    font-size: 16px;
    font-weight: 600;
  }

  .hmm-actions {
    display: flex;
    gap: 8px;
  }

  .action-btn {
    display: flex;
    align-items: center;
    gap: 4px;
    padding: 8px 12px;
    background: var(--color-bg-elevated);
    border: 1px solid var(--color-border-subtle);
    border-radius: 6px;
    color: var(--color-text-secondary);
    cursor: pointer;
    font-size: 12px;
    transition: all 0.15s ease;
  }

  .action-btn:hover:not(:disabled) {
    background: var(--bg-hover);
    border-color: var(--color-accent-cyan);
    color: var(--color-accent-cyan);
  }

  .action-btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .sync-btn {
    color: var(--color-accent-cyan);
  }

  .sync-btn:hover:not(:disabled) {
    background: rgba(59, 130, 246, 0.1);
  }

  .error-banner {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 12px;
    background: rgba(239, 68, 68, 0.1);
    border: 1px solid rgba(239, 68, 68, 0.3);
    border-radius: 8px;
    color: #ef4444;
    font-size: 13px;
  }

  .status-cards {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
  }

  .status-card {
    padding: 16px;
    background: var(--color-bg-base);
    border: 1px solid var(--color-border-subtle);
    border-radius: 10px;
  }

  .card-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 12px;
    color: var(--color-text-secondary);
    font-size: 13px;
    font-weight: 500;
  }

  .card-title {
    font-size: 13px;
  }

  .card-content {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .status-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .status-label {
    font-size: 12px;
    color: var(--color-text-muted);
  }

  .status-value {
    font-size: 12px;
    font-weight: 500;
    color: var(--color-text-secondary);
  }

  .status-value.success {
    color: #10b981;
  }

  .status-value.error {
    color: #ef4444;
  }

  .regime-display {
    padding: 16px;
    background: var(--color-bg-surface);
    border: 2px solid;
    border-radius: 8px;
    text-align: center;
  }

  .regime-name {
    display: block;
    font-size: 16px;
    font-weight: 600;
    margin-bottom: 8px;
  }

  .regime-details {
    display: flex;
    justify-content: center;
    gap: 12px;
    font-size: 11px;
    color: var(--color-text-muted);
  }

  .regime-symbol {
    text-align: center;
    margin-top: 12px;
    font-size: 12px;
    color: var(--color-text-muted);
  }

  .mode-display {
    display: flex;
    flex-direction: column;
    gap: 4px;
    text-align: center;
    margin-bottom: 12px;
  }

  .mode-name {
    font-size: 14px;
    font-weight: 600;
    color: var(--color-text-primary);
  }

  .mode-weight {
    font-size: 11px;
    color: var(--color-text-muted);
  }

  .mode-indicators {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .mode-indicator {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 12px;
    color: var(--color-text-muted);
  }

  .mode-indicator.active {
    color: #10b981;
  }

  .indicator-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--color-border-subtle);
  }

  .mode-indicator.active .indicator-dot {
    background: #10b981;
  }

  .training-section {
    padding: 16px;
    background: var(--color-bg-base);
    border: 1px solid var(--color-border-subtle);
    border-radius: 10px;
  }

  .training-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
  }

  .training-header h4 {
    margin: 0;
    font-size: 14px;
    color: var(--color-text-primary);
  }

  .train-btn {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 8px 16px;
    background: var(--color-accent-cyan);
    border: none;
    border-radius: 6px;
    color: var(--color-bg-base);
    font-size: 12px;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.15s ease;
  }

  .train-btn:hover:not(:disabled) {
    opacity: 0.9;
  }

  .train-btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .training-hint {
    margin: 0;
    font-size: 11px;
    color: var(--color-text-muted);
  }

  .agreement-section {
    padding: 16px;
    background: var(--color-bg-base);
    border: 1px solid var(--color-border-subtle);
    border-radius: 10px;
  }

  .agreement-section h4 {
    margin: 0 0 12px;
    font-size: 14px;
    color: var(--color-text-primary);
  }

  .agreement-metrics {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 8px;
  }

  .metric-item {
    display: flex;
    justify-content: space-between;
    padding: 8px 12px;
    background: var(--color-bg-surface);
    border-radius: 6px;
  }

  .metric-key {
    font-size: 11px;
    color: var(--color-text-muted);
  }

  .metric-value {
    font-size: 12px;
    font-weight: 500;
    color: var(--color-text-secondary);
  }

  .spinning {
    animation: spin 1s linear infinite;
  }

  @keyframes spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
  }

  .loading {
    opacity: 0.7;
  }

  @media (max-width: 768px) {
    .status-cards {
      grid-template-columns: 1fr;
    }
  }
</style>
