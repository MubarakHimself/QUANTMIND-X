<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { Activity, TrendingUp, AlertTriangle, CheckCircle, XCircle } from 'lucide-svelte';
  import { createTradingClient } from '$lib/ws-client';
  import type { WebSocketClient } from '$lib/ws-client';
  import { API_BASE } from '$lib/constants';

  const apiBase = API_BASE || '';

  // HMM Status data
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

  // Latest predictions
  let isingRegime = $state('UNKNOWN');
  let hmmRegime = $state('UNKNOWN');
  let hmmConfidence = $state(0);
  let agreement = $state(false);

  let wsClient: WebSocketClient | null = null;

  onMount(async () => {
    try {
      const baseUrl = apiBase || window.location.origin;

      // Fetch initial HMM status
      const statusRes = await fetch(`${baseUrl}/api/hmm/status`);
      if (statusRes.ok) {
        hmmStatus = await statusRes.json();
      }

      // Connect to WebSocket for real-time updates
      wsClient = await createTradingClient(baseUrl);

      // Subscribe to HMM predictions
      wsClient.on('hmm_prediction', (message) => {
        if (message.data) {
          isingRegime = message.data.ising_regime || 'UNKNOWN';
          hmmRegime = message.data.hmm_regime || 'UNKNOWN';
          hmmConfidence = message.data.hmm_confidence || 0;
          agreement = message.data.agreement || false;
        }
      });

      // Subscribe to mode changes
      wsClient.on('hmm_mode_change', (message) => {
        if (message.data) {
          hmmStatus.deployment_mode = message.data.new_mode;
          hmmStatus.hmm_weight = message.data.hmm_weight;
        }
      });

    } catch (error) {
      console.error('Failed to connect to HMM API:', error);
    }
  });

  onDestroy(() => {
    if (wsClient) {
      wsClient.disconnect();
    }
  });

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

  function getRegimeColor(regime: string): string {
    if (regime.includes('TRENDING')) return '#10b981';
    if (regime.includes('RANGING')) return '#f59e0b';
    if (regime.includes('CHAOS') || regime.includes('HIGH_VOL')) return '#ef4444';
    return '#6b7280';
  }
</script>

<div class="hmm-regime-panel">
  <div class="panel-header">
    <Activity size={18} />
    <h3>HMM Regime Comparison</h3>
    {#if hmmStatus.shadow_mode_active}
      <span class="shadow-badge">Shadow Mode</span>
    {/if}
  </div>

  <div class="mode-indicator" style="border-color: {getModeColor(hmmStatus.deployment_mode)}">
    <span class="mode-label">Current Mode:</span>
    <span class="mode-value" style="color: {getModeColor(hmmStatus.deployment_mode)}">
      {getModeLabel(hmmStatus.deployment_mode)}
    </span>
    {#if hmmStatus.hmm_weight > 0}
      <span class="weight-badge">HMM Weight: {(hmmStatus.hmm_weight * 100).toFixed(0)}%</span>
    {/if}
  </div>

  <div class="comparison-grid">
    <!-- Ising Model Card -->
    <div class="model-card ising">
      <div class="card-header">
        <TrendingUp size={16} />
        <span>Ising Model</span>
      </div>
      <div class="regime-display" style="color: {getRegimeColor(isingRegime)}">
        {isingRegime}
      </div>
      <div class="card-footer">
        <span class="source-label">Primary System</span>
      </div>
    </div>

    <!-- HMM Model Card -->
    <div class="model-card hmm">
      <div class="card-header">
        <Activity size={16} />
        <span>HMM Model</span>
        {#if !hmmStatus.model_loaded}
          <span class="not-loaded">Not Loaded</span>
        {/if}
      </div>
      <div class="regime-display" style="color: {getRegimeColor(hmmRegime)}">
        {hmmRegime}
      </div>
      <div class="confidence-bar">
        <div class="confidence-fill" style="width: {hmmConfidence * 100}%"></div>
        <span class="confidence-label">Confidence: {(hmmConfidence * 100).toFixed(1)}%</span>
      </div>
      <div class="card-footer">
        {#if hmmStatus.model_version}
          <span class="version-label">v{hmmStatus.model_version}</span>
        {/if}
      </div>
    </div>
  </div>

  <!-- Agreement Section -->
  <div class="agreement-section">
    <div class="agreement-header">
      {#if agreement}
        <CheckCircle size={16} class="agreement-icon agree" />
      {:else}
        <XCircle size={16} class="agreement-icon disagree" />
      {/if}
      <span>Model Agreement</span>
    </div>
    
    <div class="agreement-stats">
      <div class="agreement-bar">
        <div 
          class="agreement-fill" 
          style="width: {hmmStatus.agreement_metrics.agreement_pct}%"
        ></div>
      </div>
      <span class="agreement-pct">
        {hmmStatus.agreement_metrics.agreement_pct.toFixed(1)}%
      </span>
    </div>
    
    <div class="agreement-details">
      <span>{hmmStatus.agreement_metrics.agreement_count} / {hmmStatus.agreement_metrics.total_predictions} predictions agree</span>
    </div>
  </div>

  <!-- Version Mismatch Warning -->
  {#if hmmStatus.version_mismatch}
    <div class="version-warning">
      <AlertTriangle size={16} />
      <span>Version mismatch detected! Contabo: {hmmStatus.contabo_version || 'Unknown'}, Cloudzy: {hmmStatus.cloudzy_version || 'Unknown'}</span>
    </div>
  {/if}
</div>

<style>
  .hmm-regime-panel {
    background: #1e293b;
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 16px;
  }

  .panel-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 12px;
    color: #e2e8f0;
  }

  .panel-header h3 {
    margin: 0;
    font-size: 14px;
    font-weight: 600;
  }

  .shadow-badge {
    background: #f59e0b;
    color: #000;
    font-size: 10px;
    padding: 2px 6px;
    border-radius: 4px;
    font-weight: 600;
    animation: pulse 2s infinite;
  }

  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.7; }
  }

  .mode-indicator {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 12px;
    border-radius: 6px;
    border-left: 3px solid;
    background: rgba(255, 255, 255, 0.05);
    margin-bottom: 12px;
  }

  .mode-label {
    color: #94a3b8;
    font-size: 12px;
  }

  .mode-value {
    font-weight: 600;
    font-size: 13px;
  }

  .weight-badge {
    background: rgba(139, 92, 246, 0.2);
    color: #a78bfa;
    font-size: 10px;
    padding: 2px 6px;
    border-radius: 4px;
    margin-left: auto;
  }

  .comparison-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
    margin-bottom: 12px;
  }

  .model-card {
    background: #0f172a;
    border-radius: 6px;
    padding: 12px;
  }

  .model-card.ising {
    border-left: 3px solid #3b82f6;
  }

  .model-card.hmm {
    border-left: 3px solid #10b981;
  }

  .card-header {
    display: flex;
    align-items: center;
    gap: 6px;
    color: #94a3b8;
    font-size: 12px;
    margin-bottom: 8px;
  }

  .not-loaded {
    background: #ef4444;
    color: #fff;
    font-size: 9px;
    padding: 1px 4px;
    border-radius: 3px;
    margin-left: auto;
  }

  .regime-display {
    font-size: 14px;
    font-weight: 600;
    margin-bottom: 8px;
  }

  .confidence-bar {
    position: relative;
    height: 20px;
    background: #1e293b;
    border-radius: 4px;
    overflow: hidden;
  }

  .confidence-fill {
    position: absolute;
    top: 0;
    left: 0;
    height: 100%;
    background: linear-gradient(90deg, #10b981, #34d399);
    transition: width 0.3s ease;
  }

  .confidence-label {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    font-size: 10px;
    color: #fff;
    font-weight: 500;
  }

  .card-footer {
    display: flex;
    justify-content: space-between;
    margin-top: 8px;
    font-size: 10px;
    color: #64748b;
  }

  .source-label {
    color: #3b82f6;
  }

  .version-label {
    color: #10b981;
  }

  .agreement-section {
    background: #0f172a;
    border-radius: 6px;
    padding: 12px;
    margin-bottom: 12px;
  }

  .agreement-header {
    display: flex;
    align-items: center;
    gap: 6px;
    color: #94a3b8;
    font-size: 12px;
    margin-bottom: 8px;
  }

  .agreement-icon.agree {
    color: #10b981;
  }

  .agreement-icon.disagree {
    color: #ef4444;
  }

  .agreement-stats {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .agreement-bar {
    flex: 1;
    height: 8px;
    background: #1e293b;
    border-radius: 4px;
    overflow: hidden;
  }

  .agreement-fill {
    height: 100%;
    background: linear-gradient(90deg, #10b981, #34d399);
    transition: width 0.3s ease;
  }

  .agreement-pct {
    font-size: 12px;
    font-weight: 600;
    color: #10b981;
    min-width: 45px;
    text-align: right;
  }

  .agreement-details {
    margin-top: 4px;
    font-size: 10px;
    color: #64748b;
  }

  .version-warning {
    display: flex;
    align-items: center;
    gap: 8px;
    background: rgba(239, 68, 68, 0.1);
    border: 1px solid rgba(239, 68, 68, 0.3);
    border-radius: 6px;
    padding: 8px 12px;
    color: #fca5a5;
    font-size: 11px;
  }
</style>