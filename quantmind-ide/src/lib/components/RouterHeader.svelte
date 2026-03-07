<script lang="ts">
  import {
    Server,
    RefreshCw,
    Play,
    MonitorPlay,
    Brain
  } from 'lucide-svelte';

  export let routerState: {
    active: boolean;
    mode: 'auction' | 'priority' | 'round-robin';
    auctionInterval: number;
  };

  export let hmmTraining: {
    isTraining: boolean;
    progress: number;
    message: string;
  };

  export let autoRefresh: boolean;
  export let toggleRouter: () => Promise<void>;
  export let loadRouterState: () => Promise<void>;
  export let runAuction: () => Promise<void>;
  export let startHMMTraining: () => Promise<void>;

  function handleAutoRefreshToggle() {
    autoRefresh = !autoRefresh;
  }
</script>

<div class="router-header">
  <div class="header-left">
    <Server size={24} class="router-icon" />
    <div>
      <h2>Strategy Router</h2>
      <p>Auction-based trade signal selection and routing</p>
    </div>
  </div>
  <div class="header-actions">
    <div class="router-status" class:active={routerState.active}>
      <div class="status-indicator"></div>
      <span>{routerState.active ? 'Active' : 'Paused'}</span>
    </div>
    <button
      class="btn"
      class:active={autoRefresh}
      on:click={handleAutoRefreshToggle}
    >
      <MonitorPlay size={14} />
      <span>Auto</span>
    </button>
    <button class="btn" on:click={loadRouterState}>
      <RefreshCw size={14} />
      <span>Refresh</span>
    </button>
    <button class="btn primary" on:click={runAuction}>
      <Play size={14} />
      <span>Run Auction</span>
    </button>
    <button
      class="btn hmm-train-btn"
      class:training={hmmTraining.isTraining}
      on:click={startHMMTraining}
      disabled={hmmTraining.isTraining}
      title="Train HMM Model"
    >
      <Brain size={14} />
      <span>{hmmTraining.isTraining ? 'Training...' : 'Train HMM'}</span>
    </button>
  </div>
</div>

<style>
  .router-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 20px 24px;
    background: var(--bg-secondary);
    border-bottom: 1px solid var(--border-subtle);
  }

  .header-left {
    display: flex;
    align-items: center;
    gap: 12px;
  }

  :global(.router-icon) {
    color: var(--accent-primary);
  }

  .header-left h2 {
    margin: 0;
    font-size: 18px;
    color: var(--text-primary);
  }

  .header-left p {
    margin: 2px 0 0;
    font-size: 12px;
    color: var(--text-muted);
  }

  .header-actions {
    display: flex;
    align-items: center;
    gap: 12px;
  }

  .router-status {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 6px 12px;
    background: var(--bg-tertiary);
    border-radius: 20px;
    font-size: 12px;
    color: var(--text-muted);
  }

  .router-status.active {
    background: rgba(16, 185, 129, 0.2);
    color: #10b981;
  }

  .status-indicator {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: currentColor;
    animation: pulse 2s infinite;
  }

  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
  }

  .btn {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 8px 14px;
    background: var(--bg-tertiary);
    border: 1px solid var(--border-subtle);
    border-radius: 6px;
    color: var(--text-secondary);
    font-size: 12px;
    cursor: pointer;
    transition: all 0.15s;
  }

  .btn:hover {
    background: var(--bg-surface);
  }

  .btn.active {
    background: var(--accent-primary);
    border-color: var(--accent-primary);
    color: var(--bg-primary);
  }

  .btn.primary {
    background: var(--accent-primary);
    border-color: var(--accent-primary);
    color: var(--bg-primary);
  }

  .btn.hmm-train-btn {
    background: #7c3aed;
    border-color: #7c3aed;
    color: white;
  }

  .btn.hmm-train-btn:hover:not(:disabled) {
    background: #6d28d9;
    border-color: #6d28d9;
  }

  .btn.hmm-train-btn.training {
    background: #4c1d95;
    border-color: #4c1d95;
    opacity: 0.8;
    cursor: not-allowed;
  }
</style>
