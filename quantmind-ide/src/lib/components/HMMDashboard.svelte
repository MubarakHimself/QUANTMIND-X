<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { 
    Activity, TrendingUp, AlertTriangle, CheckCircle, XCircle, 
    RefreshCw, Download, Settings, Database, Clock, BarChart3,
    Layers, Zap, ArrowRight
  } from 'lucide-svelte';
  import HMMRegimePanel from '$lib/components/HMMRegimePanel.svelte';
  import HMMControlPanel from '$lib/components/HMMControlPanel.svelte';
  import { createTradingClient } from '$lib/ws-client';
  import type { WebSocketClient } from '$lib/ws-client';
  import { API_BASE } from '$lib/constants';

  const apiBase = API_BASE || '';

  // HMM Status
  let hmmStatus = $state({
    model_loaded: false,
    model_version: '',
    deployment_mode: 'ising_only',
    hmm_weight: 0,
    shadow_mode_active: false,
    contabo_version: '',
    cloudzy_version: '',
    version_mismatch: false,
    agreement_metrics: {
      total_predictions: 0,
      agreement_count: 0,
      agreement_pct: 0
    }
  });

  // Model hierarchy
  let modelHierarchy = $state({
    universal: { trained: false, version: '', accuracy: 0 },
    per_symbol: {} as Record<string, { trained: boolean; version: string; accuracy: number }>,
    per_symbol_timeframe: {} as Record<string, Record<string, { trained: boolean; version: string; accuracy: number }>>
  });

  // Training status
  let trainingStatus = $state({
    status: 'idle',
    progress: 0,
    message: '',
    next_training: ''
  });

  // Available models
  let availableModels: Array<{
    version: string;
    model_type: string;
    symbol?: string;
    timeframe?: string;
    training_date?: string;
    training_samples?: number;
    log_likelihood?: number;
  }> = $state([]);

  let wsClient: WebSocketClient | null = null;

  onMount(async () => {
    try {
      const baseUrl = apiBase || window.location.origin;

      // Fetch initial data
      await Promise.all([
        fetchStatus(baseUrl),
        fetchAvailableModels(baseUrl)
      ]);

      // Connect to WebSocket
      wsClient = await createTradingClient(baseUrl);

      // Subscribe to training status
      wsClient.on('hmm_training_status', (message) => {
        if (message.data) {
          trainingStatus = message.data;
        }
      });

    } catch (error) {
      console.error('Failed to initialize HMM dashboard:', error);
    }
  });

  onDestroy(() => {
    if (wsClient) {
      wsClient.disconnect();
    }
  });

  async function fetchStatus(baseUrl: string) {
    try {
      const res = await fetch(`${baseUrl}/api/hmm/status`);
      if (res.ok) {
        hmmStatus = await res.json();
      }
    } catch (error) {
      console.error('Failed to fetch HMM status:', error);
    }
  }

  async function fetchAvailableModels(baseUrl: string) {
    try {
      const res = await fetch(`${baseUrl}/api/hmm/models`);
      if (res.ok) {
        availableModels = await res.json();
        
        // Build hierarchy from models
        buildModelHierarchy();
      }
    } catch (error) {
      console.error('Failed to fetch available models:', error);
    }
  }

  function buildModelHierarchy() {
    // Reset hierarchy
    modelHierarchy = {
      universal: { trained: false, version: '', accuracy: 0 },
      per_symbol: {},
      per_symbol_timeframe: {}
    };

    for (const model of availableModels) {
      if (model.model_type === 'universal') {
        modelHierarchy.universal = {
          trained: true,
          version: model.version,
          accuracy: model.log_likelihood || 0
        };
      } else if (model.model_type === 'per_symbol') {
        modelHierarchy.per_symbol[model.symbol || ''] = {
          trained: true,
          version: model.version,
          accuracy: model.log_likelihood || 0
        };
      } else if (model.model_type === 'per_symbol_timeframe') {
        const symbol = model.symbol || '';
        const tf = model.timeframe || '';
        
        if (!modelHierarchy.per_symbol_timeframe[symbol]) {
          modelHierarchy.per_symbol_timeframe[symbol] = {};
        }
        
        modelHierarchy.per_symbol_timeframe[symbol][tf] = {
          trained: true,
          version: model.version,
          accuracy: model.log_likelihood || 0
        };
      }
    }
  }

  function getModeColor(mode: string): string {
    const colors: Record<string, string> = {
      'ising_only': '#3b82f6',
      'hmm_shadow': '#f59e0b',
      'hmm_hybrid_20': '#8b5cf6',
      'hmm_hybrid_50': '#a855f7',
      'hmm_hybrid_80': '#d946ef',
      'hmm_only': '#10b981'
    };
    return colors[mode] || '#6b7280';
  }

  function getModeLabel(mode: string): string {
    const labels: Record<string, string> = {
      'ising_only': 'Ising Only',
      'hmm_shadow': 'HMM Shadow',
      'hmm_hybrid_20': 'Hybrid 20%',
      'hmm_hybrid_50': 'Hybrid 50%',
      'hmm_hybrid_80': 'Hybrid 80%',
      'hmm_only': 'HMM Only'
    };
    return labels[mode] || mode;
  }

  const symbols = ['EURUSD', 'GBPUSD', 'XAUUSD'];
  const timeframes = ['M5', 'H1', 'H4'];
</script>

<div class="hmm-dashboard">
  <div class="dashboard-header">
    <div class="header-left">
      <Activity size={24} />
      <h1>HMM Regime Detection Dashboard</h1>
    </div>
    <div class="header-right">
      <div class="mode-badge" style="background: {getModeColor(hmmStatus.deployment_mode)}20; border-color: {getModeColor(hmmStatus.deployment_mode)}">
        <span style="color: {getModeColor(hmmStatus.deployment_mode)}">{getModeLabel(hmmStatus.deployment_mode)}</span>
      </div>
    </div>
  </div>

  <!-- Stats Grid -->
  <div class="stats-grid">
    <div class="stat-card">
      <div class="stat-icon">
        <Activity size={20} />
      </div>
      <div class="stat-content">
        <span class="stat-value">{hmmStatus.model_loaded ? 'Active' : 'Inactive'}</span>
        <span class="stat-label">Model Status</span>
      </div>
    </div>

    <div class="stat-card">
      <div class="stat-icon">
        <TrendingUp size={20} />
      </div>
      <div class="stat-content">
        <span class="stat-value">{hmmStatus.agreement_metrics.agreement_pct.toFixed(1)}%</span>
        <span class="stat-label">Ising Agreement</span>
      </div>
    </div>

    <div class="stat-card">
      <div class="stat-icon">
        <Database size={20} />
      </div>
      <div class="stat-content">
        <span class="stat-value">{hmmStatus.model_version || 'None'}</span>
        <span class="stat-label">Model Version</span>
      </div>
    </div>

    <div class="stat-card">
      <div class="stat-icon">
        <Clock size={20} />
      </div>
      <div class="stat-content">
        <span class="stat-value">{hmmStatus.agreement_metrics.total_predictions}</span>
        <span class="stat-label">Predictions Logged</span>
      </div>
    </div>
  </div>

  <!-- Main Content Grid -->
  <div class="content-grid">
    <!-- Left Column: Regime Comparison -->
    <div class="left-column">
      <HMMRegimePanel />
      <HMMControlPanel />
    </div>

    <!-- Right Column: Model Hierarchy & Training -->
    <div class="right-column">
      <!-- Model Hierarchy Section -->
      <div class="section-card">
        <div class="section-header">
          <Layers size={18} />
          <h3>Model Hierarchy</h3>
        </div>

        <div class="hierarchy-tree">
          <!-- Universal Model -->
          <div class="hierarchy-level">
            <div class="level-header">
              <span class="level-name">Universal Model</span>
              {#if modelHierarchy.universal.trained}
                <CheckCircle size={14} class="trained" />
              {:else}
                <XCircle size={14} class="not-trained" />
              {/if}
            </div>
            {#if modelHierarchy.universal.trained}
              <div class="level-details">
                <span>v{modelHierarchy.universal.version}</span>
                <span>LL: {modelHierarchy.universal.accuracy.toFixed(2)}</span>
              </div>
            {/if}
          </div>

          <!-- Per-Symbol Models -->
          <div class="hierarchy-level">
            <div class="level-header">
              <span class="level-name">Per-Symbol Models</span>
              <ArrowRight size={14} />
            </div>
            <div class="level-children">
              {#each symbols as symbol}
                <div class="child-item">
                  <span>{symbol}</span>
                  {#if modelHierarchy.per_symbol[symbol]?.trained}
                    <CheckCircle size={12} class="trained" />
                  {:else}
                    <XCircle size={12} class="not-trained" />
                  {/if}
                </div>
              {/each}
            </div>
          </div>

          <!-- Per-Symbol-Timeframe Models -->
          <div class="hierarchy-level">
            <div class="level-header">
              <span class="level-name">Per-Symbol-Timeframe</span>
              <ArrowRight size={14} />
            </div>
            <div class="level-children nested">
              {#each symbols as symbol}
                <div class="nested-group">
                  <span class="group-label">{symbol}</span>
                  <div class="group-items">
                    {#each timeframes as tf}
                      <div class="child-item small">
                        <span>{tf}</span>
                        {#if modelHierarchy.per_symbol_timeframe[symbol]?.[tf]?.trained}
                          <CheckCircle size={10} class="trained" />
                        {:else}
                          <XCircle size={10} class="not-trained" />
                        {/if}
                      </div>
                    {/each}
                  </div>
                </div>
              {/each}
            </div>
          </div>
        </div>
      </div>

      <!-- Model Cards Grid -->
      <div class="section-card">
        <div class="section-header">
          <BarChart3 size={18} />
          <h3>Model Performance</h3>
        </div>

        <div class="model-cards-grid">
          {#each availableModels.filter(m => m.model_type !== 'universal').slice(0, 6) as model}
            <div class="model-card">
              <div class="model-header">
                <span class="model-symbol">{model.symbol || 'Universal'}</span>
                <span class="model-tf">{model.timeframe || 'All'}</span>
              </div>
              <div class="model-metrics">
                <div class="metric">
                  <span class="metric-label">Samples</span>
                  <span class="metric-value">{model.training_samples?.toLocaleString() || 'N/A'}</span>
                </div>
                <div class="metric">
                  <span class="metric-label">Log-Likelihood</span>
                  <span class="metric-value">{model.log_likelihood?.toFixed(2) || 'N/A'}</span>
                </div>
              </div>
              <div class="model-footer">
                <span class="model-version">v{model.version}</span>
              </div>
            </div>
          {/each}
        </div>
      </div>

      <!-- Training Schedule -->
      <div class="section-card">
        <div class="section-header">
          <Zap size={18} />
          <h3>Training Schedule</h3>
        </div>

        <div class="training-status">
          <div class="status-row">
            <span class="status-label">Status:</span>
            <span class="status-value">{trainingStatus.status}</span>
          </div>
          <div class="status-row">
            <span class="status-label">Next Training:</span>
            <span class="status-value">{trainingStatus.next_training || 'Saturday 02:00 UTC'}</span>
          </div>
          {#if trainingStatus.status === 'in_progress'}
            <div class="training-progress">
              <div class="progress-bar">
                <div class="progress-fill" style="width: {trainingStatus.progress}%"></div>
              </div>
              <span class="progress-message">{trainingStatus.message}</span>
            </div>
          {/if}
        </div>
      </div>
    </div>
  </div>
</div>

<style>
  .hmm-dashboard {
    padding: 24px;
    max-width: 1400px;
    margin: 0 auto;
  }

  .dashboard-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 24px;
  }

  .header-left {
    display: flex;
    align-items: center;
    gap: 12px;
    color: #e2e8f0;
  }

  .header-left h1 {
    margin: 0;
    font-size: 20px;
    font-weight: 600;
  }

  .mode-badge {
    padding: 6px 12px;
    border-radius: 6px;
    border: 1px solid;
    font-size: 12px;
    font-weight: 500;
  }

  .stats-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin-bottom: 24px;
  }

  .stat-card {
    background: #1e293b;
    border-radius: 8px;
    padding: 16px;
    display: flex;
    align-items: center;
    gap: 12px;
  }

  .stat-icon {
    width: 40px;
    height: 40px;
    border-radius: 8px;
    background: rgba(59, 130, 246, 0.1);
    display: flex;
    align-items: center;
    justify-content: center;
    color: #3b82f6;
  }

  .stat-content {
    display: flex;
    flex-direction: column;
  }

  .stat-value {
    font-size: 18px;
    font-weight: 600;
    color: #e2e8f0;
  }

  .stat-label {
    font-size: 11px;
    color: #64748b;
  }

  .content-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 24px;
  }

  .left-column, .right-column {
    display: flex;
    flex-direction: column;
    gap: 16px;
  }

  .section-card {
    background: #1e293b;
    border-radius: 8px;
    padding: 16px;
  }

  .section-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 16px;
    color: #e2e8f0;
  }

  .section-header h3 {
    margin: 0;
    font-size: 14px;
    font-weight: 600;
  }

  .hierarchy-tree {
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  .hierarchy-level {
    background: #0f172a;
    border-radius: 6px;
    padding: 12px;
  }

  .level-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    color: #e2e8f0;
    font-size: 12px;
    font-weight: 500;
  }

  .level-details {
    display: flex;
    gap: 16px;
    margin-top: 8px;
    font-size: 10px;
    color: #64748b;
  }

  .level-children {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 8px;
  }

  .level-children.nested {
    flex-direction: column;
  }

  .child-item {
    display: flex;
    align-items: center;
    gap: 6px;
    background: #1e293b;
    padding: 6px 10px;
    border-radius: 4px;
    font-size: 11px;
    color: #94a3b8;
  }

  .child-item.small {
    padding: 4px 8px;
    font-size: 10px;
  }

  .nested-group {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .group-label {
    font-size: 10px;
    color: #64748b;
    font-weight: 500;
  }

  .group-items {
    display: flex;
    gap: 4px;
  }

  .trained {
    color: #10b981;
  }

  .not-trained {
    color: #64748b;
  }

  .model-cards-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 12px;
  }

  .model-card {
    background: #0f172a;
    border-radius: 6px;
    padding: 12px;
  }

  .model-header {
    display: flex;
    justify-content: space-between;
    margin-bottom: 8px;
  }

  .model-symbol {
    font-size: 12px;
    font-weight: 600;
    color: #e2e8f0;
  }

  .model-tf {
    font-size: 10px;
    color: #64748b;
    background: #1e293b;
    padding: 2px 6px;
    border-radius: 3px;
  }

  .model-metrics {
    display: flex;
    flex-direction: column;
    gap: 4px;
    margin-bottom: 8px;
  }

  .metric {
    display: flex;
    justify-content: space-between;
    font-size: 10px;
  }

  .metric-label {
    color: #64748b;
  }

  .metric-value {
    color: #94a3b8;
    font-weight: 500;
  }

  .model-footer {
    border-top: 1px solid #1e293b;
    padding-top: 8px;
    font-size: 10px;
    color: #10b981;
  }

  .training-status {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .status-row {
    display: flex;
    justify-content: space-between;
    font-size: 12px;
  }

  .status-label {
    color: #64748b;
  }

  .status-value {
    color: #e2e8f0;
    font-weight: 500;
  }

  .training-progress {
    margin-top: 8px;
  }

  .progress-bar {
    height: 6px;
    background: #0f172a;
    border-radius: 3px;
    overflow: hidden;
    margin-bottom: 4px;
  }

  .progress-fill {
    height: 100%;
    background: #3b82f6;
    transition: width 0.3s ease;
  }

  .progress-message {
    font-size: 10px;
    color: #64748b;
  }

  @media (max-width: 1024px) {
    .stats-grid {
      grid-template-columns: repeat(2, 1fr);
    }

    .content-grid {
      grid-template-columns: 1fr;
    }
  }

  @media (max-width: 640px) {
    .stats-grid {
      grid-template-columns: 1fr;
    }

    .model-cards-grid {
      grid-template-columns: 1fr;
    }
  }
</style>