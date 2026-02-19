<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { 
    RefreshCw, Download, Upload, AlertTriangle, CheckCircle, 
    Settings, Play, Pause, RotateCcw, Info, ChevronDown, ChevronUp
  } from 'lucide-svelte';
  import { createTradingClient } from '$lib/ws-client';
  import type { WebSocketClient } from '$lib/ws-client';
  import { PUBLIC_API_BASE } from '$env/static/public';

  const apiBase = PUBLIC_API_BASE || '';

  // HMM Status
  let hmmStatus = {
    model_loaded: false,
    model_version: '',
    deployment_mode: 'ising_only',
    hmm_weight: 0,
    shadow_mode_active: false,
    contabo_version: '',
    cloudzy_version: '',
    version_mismatch: false,
    last_sync: '',
    sync_status: ''
  };

  // Sync progress
  let syncProgress = {
    status: 'idle',
    progress: 0,
    message: '',
    error: ''
  };

  // Pending approvals
  let pendingApprovals: Array<{
    token: string;
    target_mode: string;
    requester: string;
    requested_at: string;
  }> = [];

  // Shadow logs
  let shadowLogs: Array<{
    timestamp: string;
    symbol: string;
    timeframe: string;
    ising_regime: string;
    hmm_regime: string;
    agreement: boolean;
  }> = [];

  // UI state
  let isSyncing = false;
  let showLogs = false;
  let selectedMode = 'ising_only';
  let approvalToken = '';
  let showApprovalModal = false;
  let pendingMode = '';

  let wsClient: WebSocketClient | null = null;

  onMount(async () => {
    try {
      const baseUrl = apiBase || window.location.origin;

      // Fetch initial status
      await fetchStatus(baseUrl);
      await fetchShadowLogs(baseUrl);

      // Connect to WebSocket
      wsClient = await createTradingClient(baseUrl);

      // Subscribe to sync progress
      wsClient.on('hmm_sync_progress', (message) => {
        if (message.data) {
          syncProgress = message.data;
          isSyncing = message.data.status === 'in_progress';
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
      console.error('Failed to initialize HMM control panel:', error);
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
        selectedMode = hmmStatus.deployment_mode;
      }
    } catch (error) {
      console.error('Failed to fetch HMM status:', error);
    }
  }

  async function fetchShadowLogs(baseUrl: string) {
    try {
      const res = await fetch(`${baseUrl}/api/hmm/shadow-mode/logs?limit=20`);
      if (res.ok) {
        const data = await res.json();
        shadowLogs = data.logs || [];
      }
    } catch (error) {
      console.error('Failed to fetch shadow logs:', error);
    }
  }

  async function syncModel() {
    try {
      isSyncing = true;
      const baseUrl = apiBase || window.location.origin;
      
      const res = await fetch(`${baseUrl}/api/hmm/sync`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ verify_checksum: true })
      });

      if (!res.ok) {
        const error = await res.json();
        throw new Error(error.detail || 'Sync failed');
      }
    } catch (error) {
      console.error('Sync failed:', error);
      isSyncing = false;
    }
  }

  async function toggleShadowMode(enabled: boolean) {
    try {
      const baseUrl = apiBase || window.location.origin;
      
      const res = await fetch(`${baseUrl}/api/hmm/shadow-mode/toggle?enabled=${enabled}`, {
        method: 'POST'
      });

      if (res.ok) {
        await fetchStatus(baseUrl);
      }
    } catch (error) {
      console.error('Failed to toggle shadow mode:', error);
    }
  }

  async function requestModeChange(mode: string) {
    const restrictedModes = ['hmm_hybrid_50', 'hmm_hybrid_80', 'hmm_only'];
    
    if (restrictedModes.includes(mode)) {
      // Need approval token
      pendingMode = mode;
      showApprovalModal = true;
      
      // Request approval token
      try {
        const baseUrl = apiBase || window.location.origin;
        const res = await fetch(`${baseUrl}/api/hmm/approval-token`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            target_mode: mode,
            requester: 'dashboard'
          })
        });

        if (res.ok) {
          const data = await res.json();
          approvalToken = data.token;
        }
      } catch (error) {
        console.error('Failed to request approval:', error);
      }
    } else {
      // Direct mode change
      await changeMode(mode);
    }
  }

  async function changeMode(mode: string, token?: string) {
    try {
      const baseUrl = apiBase || window.location.origin;
      
      const body: any = { mode };
      if (token) {
        body.approval_token = token;
      }

      const res = await fetch(`${baseUrl}/api/hmm/mode`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
      });

      if (res.ok) {
        await fetchStatus(baseUrl);
        showApprovalModal = false;
      } else {
        const error = await res.json();
        console.error('Mode change failed:', error.detail);
      }
    } catch (error) {
      console.error('Failed to change mode:', error);
    }
  }

  async function rollback() {
    try {
      const baseUrl = apiBase || window.location.origin;
      
      const res = await fetch(`${baseUrl}/api/hmm/rollback`, {
        method: 'POST'
      });

      if (res.ok) {
        await fetchStatus(baseUrl);
      }
    } catch (error) {
      console.error('Rollback failed:', error);
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
      'hmm_hybrid_20': 'Hybrid 20% HMM',
      'hmm_hybrid_50': 'Hybrid 50% HMM',
      'hmm_hybrid_80': 'Hybrid 80% HMM',
      'hmm_only': 'HMM Only'
    };
    return labels[mode] || mode;
  }
</script>

<div class="hmm-control-panel">
  <div class="panel-header">
    <Settings size={18} />
    <h3>HMM Control Panel</h3>
  </div>

  <!-- Version Control Section -->
  <div class="section">
    <h4>Version Control</h4>
    <div class="version-grid">
      <div class="version-card">
        <span class="server-label">Contabo (Training)</span>
        <span class="version-value">{hmmStatus.contabo_version || 'No model'}</span>
      </div>
      <div class="version-card">
        <span class="server-label">Cloudzy (Trading)</span>
        <span class="version-value">{hmmStatus.cloudzy_version || 'No model'}</span>
      </div>
    </div>

    {#if hmmStatus.version_mismatch}
      <div class="mismatch-warning">
        <AlertTriangle size={14} />
        <span>Version mismatch detected</span>
      </div>
    {/if}

    <button 
      class="sync-btn" 
      class:syncing={isSyncing}
      on:click={syncModel}
      disabled={isSyncing}
    >
      <Download size={14} />
      {#if isSyncing}
        Syncing... {syncProgress.progress.toFixed(0)}%
      {:else}
        Sync Model
      {/if}
    </button>

    {#if isSyncing}
      <div class="sync-progress">
        <div class="progress-bar">
          <div class="progress-fill" style="width: {syncProgress.progress}%"></div>
        </div>
        <span class="progress-message">{syncProgress.message}</span>
      </div>
    {/if}
  </div>

  <!-- Shadow Mode Section -->
  <div class="section">
    <h4>Shadow Mode</h4>
    <p class="section-description">
      Run HMM in parallel with Ising for validation without affecting trading decisions.
    </p>
    
    <div class="toggle-container">
      <label class="toggle">
        <input 
          type="checkbox" 
          checked={hmmStatus.shadow_mode_active}
          on:change={(e) => toggleShadowMode(e.currentTarget.checked)}
        />
        <span class="toggle-slider"></span>
      </label>
      <span class="toggle-label">
        {hmmStatus.shadow_mode_active ? 'Shadow Mode Active' : 'Shadow Mode Disabled'}
      </span>
    </div>
  </div>

  <!-- Production Mode Section -->
  <div class="section">
    <h4>Production Mode</h4>
    <div class="mode-options">
      {#each ['ising_only', 'hmm_shadow', 'hmm_hybrid_20', 'hmm_hybrid_50', 'hmm_hybrid_80', 'hmm_only'] as mode}
        <button 
          class="mode-btn"
          class:active={hmmStatus.deployment_mode === mode}
          class:restricted={['hmm_hybrid_50', 'hmm_hybrid_80', 'hmm_only'].includes(mode)}
          style="border-color: {getModeColor(mode)}; color: {hmmStatus.deployment_mode === mode ? getModeColor(mode) : '#94a3b8'}"
          on:click={() => requestModeChange(mode)}
        >
          {getModeLabel(mode)}
          {#if ['hmm_hybrid_50', 'hmm_hybrid_80', 'hmm_only'].includes(mode)}
            <span class="approval-badge">Approval Required</span>
          {/if}
        </button>
      {/each}
    </div>

    <button class="rollback-btn" on:click={rollback}>
      <RotateCcw size={14} />
      Rollback to Previous Mode
    </button>
  </div>

  <!-- HMM Logs Section -->
  <div class="section">
    <div class="section-header" on:click={() => showLogs = !showLogs}>
      <h4>Recent Predictions</h4>
      {#if showLogs}
        <ChevronUp size={16} />
      {:else}
        <ChevronDown size={16} />
      {/if}
    </div>

    {#if showLogs}
      <div class="logs-container">
        {#each shadowLogs as log}
          <div class="log-entry" class:agree={log.agreement} class:disagree={!log.agreement}>
            <span class="log-time">
              {new Date(log.timestamp).toLocaleTimeString()}
            </span>
            <span class="log-symbol">{log.symbol}</span>
            <span class="log-regime ising">{log.ising_regime}</span>
            <span class="log-arrow">vs</span>
            <span class="log-regime hmm">{log.hmm_regime}</span>
            {#if log.agreement}
              <CheckCircle size={12} class="agree-icon" />
            {:else}
              <AlertTriangle size={12} class="disagree-icon" />
            {/if}
          </div>
        {:else}
          <div class="no-logs">No shadow logs available</div>
        {/each}
      </div>
    {/if}
  </div>
</div>

<!-- Approval Modal -->
{#if showApprovalModal}
  <div class="modal-overlay" on:click={() => showApprovalModal = false}>
    <div class="modal" on:click|stopPropagation>
      <h3>Approval Required</h3>
      <p>Transitioning to <strong>{getModeLabel(pendingMode)}</strong> requires approval.</p>
      
      <div class="approval-info">
        <Info size={14} />
        <span>Approval token has been generated. Click "Approve" to proceed.</span>
      </div>

      <div class="token-display">
        <code>{approvalToken.slice(0, 16)}...</code>
      </div>

      <div class="modal-actions">
        <button class="cancel-btn" on:click={() => showApprovalModal = false}>
          Cancel
        </button>
        <button 
          class="approve-btn" 
          on:click={() => changeMode(pendingMode, approvalToken)}
        >
          Approve Transition
        </button>
      </div>
    </div>
  </div>
{/if}

<style>
  .hmm-control-panel {
    background: #1e293b;
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 16px;
  }

  .panel-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 16px;
    color: #e2e8f0;
  }

  .panel-header h3 {
    margin: 0;
    font-size: 14px;
    font-weight: 600;
  }

  .section {
    margin-bottom: 16px;
    padding-bottom: 16px;
    border-bottom: 1px solid #334155;
  }

  .section:last-child {
    margin-bottom: 0;
    padding-bottom: 0;
    border-bottom: none;
  }

  .section h4 {
    margin: 0 0 8px 0;
    font-size: 12px;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  .section-description {
    font-size: 11px;
    color: #64748b;
    margin: 0 0 12px 0;
  }

  .section-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    cursor: pointer;
  }

  .version-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
    margin-bottom: 12px;
  }

  .version-card {
    background: #0f172a;
    border-radius: 6px;
    padding: 10px;
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .server-label {
    font-size: 10px;
    color: #64748b;
  }

  .version-value {
    font-size: 12px;
    color: #e2e8f0;
    font-weight: 500;
  }

  .mismatch-warning {
    display: flex;
    align-items: center;
    gap: 6px;
    background: rgba(239, 68, 68, 0.1);
    border-radius: 4px;
    padding: 6px 10px;
    color: #fca5a5;
    font-size: 11px;
    margin-bottom: 12px;
  }

  .sync-btn {
    display: flex;
    align-items: center;
    gap: 6px;
    background: #3b82f6;
    color: #fff;
    border: none;
    border-radius: 6px;
    padding: 8px 16px;
    font-size: 12px;
    cursor: pointer;
    transition: background 0.2s;
  }

  .sync-btn:hover:not(:disabled) {
    background: #2563eb;
  }

  .sync-btn:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }

  .sync-btn.syncing {
    background: #6b7280;
  }

  .sync-progress {
    margin-top: 8px;
  }

  .progress-bar {
    height: 4px;
    background: #334155;
    border-radius: 2px;
    overflow: hidden;
  }

  .progress-fill {
    height: 100%;
    background: #3b82f6;
    transition: width 0.3s ease;
  }

  .progress-message {
    font-size: 10px;
    color: #94a3b8;
    margin-top: 4px;
  }

  .toggle-container {
    display: flex;
    align-items: center;
    gap: 12px;
  }

  .toggle {
    position: relative;
    width: 44px;
    height: 24px;
  }

  .toggle input {
    opacity: 0;
    width: 0;
    height: 0;
  }

  .toggle-slider {
    position: absolute;
    cursor: pointer;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background-color: #334155;
    transition: 0.3s;
    border-radius: 24px;
  }

  .toggle-slider:before {
    position: absolute;
    content: "";
    height: 18px;
    width: 18px;
    left: 3px;
    bottom: 3px;
    background-color: #e2e8f0;
    transition: 0.3s;
    border-radius: 50%;
  }

  .toggle input:checked + .toggle-slider {
    background-color: #f59e0b;
  }

  .toggle input:checked + .toggle-slider:before {
    transform: translateX(20px);
  }

  .toggle-label {
    font-size: 12px;
    color: #94a3b8;
  }

  .mode-options {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 8px;
    margin-bottom: 12px;
  }

  .mode-btn {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 4px;
    background: #0f172a;
    border: 2px solid;
    border-radius: 6px;
    padding: 10px;
    font-size: 11px;
    cursor: pointer;
    transition: all 0.2s;
  }

  .mode-btn:hover {
    background: #1e293b;
  }

  .mode-btn.active {
    background: rgba(255, 255, 255, 0.05);
  }

  .mode-btn.restricted {
    position: relative;
  }

  .approval-badge {
    font-size: 8px;
    background: rgba(239, 68, 68, 0.2);
    color: #fca5a5;
    padding: 2px 4px;
    border-radius: 3px;
  }

  .rollback-btn {
    display: flex;
    align-items: center;
    gap: 6px;
    background: transparent;
    border: 1px solid #ef4444;
    color: #fca5a5;
    border-radius: 6px;
    padding: 6px 12px;
    font-size: 11px;
    cursor: pointer;
    transition: all 0.2s;
  }

  .rollback-btn:hover {
    background: rgba(239, 68, 68, 0.1);
  }

  .logs-container {
    max-height: 200px;
    overflow-y: auto;
    margin-top: 8px;
  }

  .log-entry {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 6px 8px;
    border-radius: 4px;
    font-size: 10px;
    margin-bottom: 4px;
  }

  .log-entry.agree {
    background: rgba(16, 185, 129, 0.1);
  }

  .log-entry.disagree {
    background: rgba(239, 68, 68, 0.1);
  }

  .log-time {
    color: #64748b;
    min-width: 60px;
  }

  .log-symbol {
    color: #e2e8f0;
    font-weight: 500;
    min-width: 40px;
  }

  .log-regime {
    font-size: 9px;
    padding: 2px 4px;
    border-radius: 3px;
  }

  .log-regime.ising {
    background: rgba(59, 130, 246, 0.2);
    color: #93c5fd;
  }

  .log-regime.hmm {
    background: rgba(16, 185, 129, 0.2);
    color: #6ee7b7;
  }

  .log-arrow {
    color: #64748b;
  }

  .agree-icon {
    color: #10b981;
  }

  .disagree-icon {
    color: #ef4444;
  }

  .no-logs {
    text-align: center;
    color: #64748b;
    font-size: 11px;
    padding: 20px;
  }

  /* Modal styles */
  .modal-overlay {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(0, 0, 0, 0.7);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 1000;
  }

  .modal {
    background: #1e293b;
    border-radius: 12px;
    padding: 24px;
    max-width: 400px;
    width: 90%;
  }

  .modal h3 {
    margin: 0 0 12px 0;
    color: #e2e8f0;
  }

  .modal p {
    margin: 0 0 16px 0;
    color: #94a3b8;
    font-size: 13px;
  }

  .approval-info {
    display: flex;
    align-items: flex-start;
    gap: 8px;
    background: rgba(59, 130, 246, 0.1);
    border-radius: 6px;
    padding: 10px;
    margin-bottom: 16px;
    font-size: 11px;
    color: #93c5fd;
  }

  .token-display {
    background: #0f172a;
    border-radius: 6px;
    padding: 10px;
    margin-bottom: 16px;
    text-align: center;
  }

  .token-display code {
    font-family: monospace;
    color: #10b981;
    font-size: 12px;
  }

  .modal-actions {
    display: flex;
    gap: 12px;
    justify-content: flex-end;
  }

  .cancel-btn {
    background: transparent;
    border: 1px solid #64748b;
    color: #94a3b8;
    border-radius: 6px;
    padding: 8px 16px;
    font-size: 12px;
    cursor: pointer;
  }

  .approve-btn {
    background: #10b981;
    border: none;
    color: #fff;
    border-radius: 6px;
    padding: 8px 16px;
    font-size: 12px;
    cursor: pointer;
  }

  .approve-btn:hover {
    background: #059669;
  }
</style>