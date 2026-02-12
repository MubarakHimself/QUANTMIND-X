<script lang="ts">
  import { onMount } from 'svelte';
  import { createEventDispatcher } from 'svelte';
  import { AlertTriangle, CheckCircle, Power, Activity, Bot, TrendingUp, Wifi, WifiOff } from 'lucide-svelte';
  
  const dispatch = createEventDispatcher();
  
  const API_BASE = 'http://localhost:8000/api';
  
  let connected = false;
  let regime = 'Unknown';
  let kelly = 0.0;
  let activeBots = 0;
  let pnlToday = 0.0;
  let killSwitchActive = false;
  let loading = true;
  
  onMount(() => {
    fetchStatus();
    // Poll every 5 seconds
    const interval = setInterval(fetchStatus, 5000);
    return () => clearInterval(interval);
  });
  
  async function fetchStatus() {
    try {
      const response = await fetch(`${API_BASE}/trading/status`);
      if (response.ok) {
        const data = await response.json();
        connected = data.connected ?? false;
        regime = data.regime ?? 'Unknown';
        kelly = data.kelly ?? 0.0;
        activeBots = data.active_bots ?? 0;
        pnlToday = data.pnl_today ?? 0.0;
      } else {
        connected = false;
      }
    } catch (e) {
      connected = false;
      // Use demo values when backend not available
      regime = 'Trending';
      kelly = 0.85;
      activeBots = 3;
      pnlToday = 12.50;
    } finally {
      loading = false;
    }
  }
  
  async function toggleKillSwitch() {
    if (killSwitchActive) {
      // Already active, just toggle display
      killSwitchActive = false;
      return;
    }
    
    killSwitchActive = true;
    
    try {
      const response = await fetch(`${API_BASE}/trading/kill`, { method: 'POST' });
      if (response.ok) {
        dispatch('killSwitch', { triggered: true });
      }
    } catch (e) {
      console.error('Kill switch failed:', e);
    }
  }
  
  function openLiveTrading() {
    dispatch('openLiveTrading');
  }
  
  function formatPnL(value: number): string {
    const sign = value >= 0 ? '+' : '';
    return `${sign}$${Math.abs(value).toFixed(2)}`;
  }
</script>

<footer class="status-bar">
  <div class="left-section">
    <button class="status-item connection" class:connected on:click={fetchStatus}>
      {#if connected}
        <Wifi size={12} />
        <span>Connected</span>
      {:else}
        <WifiOff size={12} />
        <span>Offline</span>
      {/if}
    </button>
    
    <div class="status-item">
      <Activity size={12} />
      <span>Regime: {regime}</span>
    </div>
    
    <div class="status-item">
      <TrendingUp size={12} />
      <span>Kelly: {kelly.toFixed(2)}</span>
    </div>
  </div>
  
  <div class="right-section">
    <button class="status-item bots" on:click={openLiveTrading}>
      <Bot size={12} />
      <span>Active: {activeBots}</span>
    </button>
    
    <div class="status-item pnl" class:positive={pnlToday >= 0} class:negative={pnlToday < 0}>
      <span>P&L: {formatPnL(pnlToday)}</span>
    </div>
    
    <div class="status-item nprd">
      <span>NPRD: Gemini CLI</span>
    </div>
    
    <button
      class="kill-switch"
      class:active={killSwitchActive}
      on:click={toggleKillSwitch}
      title="Emergency Kill Switch"
    >
      <Power size={12} />
      <span>Kill</span>
    </button>
  </div>
</footer>

<style>
  .status-bar {
    grid-column: 1 / -1;
    grid-row: 2;
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0 12px;
    background: var(--accent-primary);
    color: var(--bg-primary);
    font-size: 11px;
    height: var(--status-height);
  }
  
  .left-section, .right-section {
    display: flex;
    align-items: center;
    gap: 16px;
  }
  
  .status-item {
    display: flex;
    align-items: center;
    gap: 4px;
    background: none;
    border: none;
    color: inherit;
    font-size: inherit;
    cursor: default;
    padding: 0;
  }
  
  button.status-item {
    cursor: pointer;
    opacity: 0.9;
  }
  
  button.status-item:hover {
    opacity: 1;
  }
  
  .connection {
    padding: 2px 6px;
    border-radius: 4px;
    background: rgba(0, 0, 0, 0.2);
  }
  
  .connection.connected {
    background: rgba(0, 0, 0, 0.2);
  }
  
  .connection:not(.connected) {
    background: rgba(255, 100, 100, 0.3);
  }
  
  .bots {
    padding: 2px 6px;
    border-radius: 4px;
    background: rgba(0, 0, 0, 0.15);
  }
  
  .bots:hover {
    background: rgba(0, 0, 0, 0.25);
  }
  
  .pnl {
    font-weight: 600;
  }
  
  .pnl.positive {
    color: #90EE90;
  }
  
  .pnl.negative {
    color: #FFB6C1;
  }
  
  .kill-switch {
    display: flex;
    align-items: center;
    gap: 4px;
    padding: 2px 8px;
    background: rgba(0, 0, 0, 0.2);
    border: 1px solid rgba(255, 255, 255, 0.3);
    border-radius: 4px;
    color: inherit;
    font-size: 10px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.15s ease;
  }
  
  .kill-switch:hover {
    background: rgba(255, 50, 50, 0.4);
    border-color: rgba(255, 50, 50, 0.6);
  }
  
  .kill-switch.active {
    background: #e74c3c;
    border-color: #c0392b;
    animation: pulse 1s infinite;
  }
  
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.7; }
  }
</style>
