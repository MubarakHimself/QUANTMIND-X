<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { createEventDispatcher } from 'svelte';
  import { PlayCircle, ShieldAlert, Newspaper, FileText, Activity, Bot, Wallet, TrendingUp, ExternalLink } from 'lucide-svelte';
  import KillSwitchView from './KillSwitchView.svelte';
  import NewsView from './NewsView.svelte';
  import TradeJournalView from './TradeJournalView.svelte';
  import FeeMonitorPanel from './FeeMonitorPanel.svelte';
  import MultiTimeframeRegimePanel from './MultiTimeframeRegimePanel.svelte';
  import TradingViewChart from './TradingViewChart.svelte';
  import { navigationStore } from '../stores/navigationStore';
  import { createTradingClient } from '$lib/ws-client';
  import type { WebSocketClient } from '$lib/ws-client';
  import { PUBLIC_API_BASE } from '$env/static/public';

  const dispatch = createEventDispatcher();

  // Use configured API base or default to same origin
  const apiBase = PUBLIC_API_BASE || '';

  export let activeView = 'dashboard';

  const subTabs = [
    { id: 'dashboard', icon: Activity, label: 'Dashboard' },
    { id: 'bots', icon: PlayCircle, label: 'Active Bots' },
    { id: 'chart', icon: TrendingUp, label: 'Live Chart' },
    { id: 'kill-switch', icon: ShieldAlert, label: 'Kill Switch' },
    { id: 'news', icon: Newspaper, label: 'News & Kill Zones' },
    { id: 'journal', icon: FileText, label: 'Trade Journal' }
  ];
  
  // Chart settings
  let chartSymbol = 'EURUSD';
  let chartTimeframe = 'H1';

  // Real-time data from API
  let systemStatus = {
    active_bots: 0,
    pnl_today: 0,
    regime: 'UNKNOWN',
    kelly: 0
  };
  
  let bots: Array<{id: string, name: string, state: string, symbol: string}> = [];
  let brokerAccounts: Array<{broker_id: string, broker_name: string, balance: number, equity: number, connected: boolean}> = [];
  let wsClient: WebSocketClient | null = null;
  let isConnected = false;

  onMount(async () => {
    try {
      // Build absolute URL for REST calls (use apiBase or default to same origin)
      const baseUrl = apiBase || window.location.origin;

      // Fetch initial data from REST API (using existing backend routes)
      const [statusRes, botsRes, accountsRes] = await Promise.all([
        fetch(`${baseUrl}/api/router/system-status`),
        fetch(`${baseUrl}/api/router/active-bots`),
        fetch(`${baseUrl}/api/trading/broker-accounts`)
      ]);

      if (statusRes.ok) {
        systemStatus = await statusRes.json();
      }

      if (botsRes.ok) {
        bots = await botsRes.json();
      }

      if (accountsRes.ok) {
        brokerAccounts = await accountsRes.json();
      }

      // Connect to WebSocket for real-time updates
      // Pass the base URL so WebSocket helper derives ws:// vs wss:// correctly
      wsClient = await createTradingClient(baseUrl);
      isConnected = true;

      // Subscribe to system status updates
      wsClient.on('system_status', (message) => {
        if (message.data) {
          systemStatus = message.data as typeof systemStatus;
        }
      });

      // Subscribe to bot updates
      wsClient.on('bot_update', (message) => {
        if (message.data?.bots) {
          bots = message.data.bots;
        }
      });

    } catch (error) {
      console.error('Failed to connect to trading API:', error);
    }
  });

  onDestroy(() => {
    if (wsClient) {
      wsClient.disconnect();
    }
  });

  function selectSubTab(tabId: string) {
    activeView = tabId;
    dispatch('subPageChange', { subPage: tabId });
  }

  function getStatusColor(status: string) {
    const colors: Record<string, string> = {
      primal: '#10b981',
      ready: '#3b82f6',
      pending: '#f59e0b',
      paused: '#6b7280',
      processing: '#8b5cf6'
    };
    return colors[status] || '#6b7280';
  }

  function formatCurrency(value: number): string {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD'
    }).format(value);
  }
</script>

<div class="live-trading-container">
  <div class="sub-tabs">
    {#each subTabs as tab}
      <button
        class="sub-tab"
        class:active={activeView === tab.id}
        on:click={() => selectSubTab(tab.id)}
      >
        <svelte:component this={tab.icon} size={16} />
        <span>{tab.label}</span>
      </button>
    {/each}
  </div>

  <div class="sub-content">
    {#if activeView === 'dashboard'}
      <!-- Live Trading Dashboard content -->
      <div class="live-dashboard">
        <div class="stats-row">
          <div class="stat-card">
            <Bot size={24} />
            <div class="stat-info">
              <span class="stat-value">{systemStatus.active_bots}</span>
              <span class="stat-label">Active Bots</span>
            </div>
          </div>
          <div class="stat-card">
            <Wallet size={24} />
            <div class="stat-info">
              <span class="stat-value positive">{formatCurrency(systemStatus.pnl_today)}</span>
              <span class="stat-label">Today's P&L</span>
            </div>
          </div>
          <div class="stat-card">
            <Activity size={24} />
            <div class="stat-info">
              <span class="stat-value">{systemStatus.regime}</span>
              <span class="stat-label">Market Regime</span>
            </div>
          </div>
          <div class="stat-card">
            <TrendingUp size={24} />
            <div class="stat-info">
              <span class="stat-value">{systemStatus.kelly.toFixed(2)}</span>
              <span class="stat-label">Kelly Factor</span>
            </div>
          </div>
        </div>

        <!-- Fee Monitoring Panel -->
        <FeeMonitorPanel />
        
        <!-- Multi-Timeframe Regime Panel -->
        <MultiTimeframeRegimePanel />

        <div class="sections-row">
          <div class="section-card clickable" on:click={() => selectSubTab('bots')}>
            <PlayCircle size={20} />
            <span>View All Bots</span>
          </div>
          <div class="section-card danger" on:click={() => selectSubTab('kill-switch')}>
            <ShieldAlert size={20} />
            <span>Kill Switch</span>
          </div>
        </div>

        <div class="broker-section">
          <h3>Broker Accounts</h3>
          <div class="account-cards">
            {#if brokerAccounts.length > 0}
              {#each brokerAccounts as account}
                <div class="account-card">
                  <span class="broker-name">{account.broker_name}</span>
                  <div class="balance-row">
                    <span>Balance: {formatCurrency(account.balance)}</span>
                    <span>Equity: {formatCurrency(account.equity)}</span>
                  </div>
                  <button class="login-btn" class:connected={account.connected}>
                    <ExternalLink size={12} />
                    {account.connected ? 'Connected' : 'Connect'}
                  </button>
                </div>
              {/each}
            {:else}
              <div class="account-card">
                <span class="broker-name">No accounts configured</span>
                <div class="balance-row">
                  <span>Add a broker account to start trading</span>
                </div>
                <button class="login-btn">
                  <ExternalLink size={12} />
                  Add Account
                </button>
              </div>
            {/if}
          </div>
        </div>

              </div>

    {:else if activeView === 'bots'}
      <!-- Active Bots content -->
      <div class="bots-view">
        <h2>Active Bots</h2>
        <div class="bot-cards">
          {#each bots as bot}
            <div class="bot-detail-card">
              <div class="bot-status-indicator" style="background: {getStatusColor(bot.state)}"></div>
              <div class="bot-main">
                <h4>{bot.name}</h4>
                <p>{bot.symbol}</p>
              </div>
              <span class="state-badge" style="background: {getStatusColor(bot.state)}20; color: {getStatusColor(bot.state)}">
                {bot.state.charAt(0).toUpperCase() + bot.state.slice(1)}
              </span>
            </div>
          {/each}
        </div>
      </div>

    {:else if activeView === 'kill-switch'}
      <!-- Kill Switch View -->
      <KillSwitchView />

    {:else if activeView === 'news'}
      <!-- News & Kill Zones View -->
      <NewsView />

    {:else if activeView === 'chart'}
      <!-- Live Chart View -->
      <div class="chart-view">
        <div class="chart-controls-bar">
          <div class="symbol-selector">
            <label>Symbol:</label>
            <select bind:value={chartSymbol}>
              <option value="EURUSD">EURUSD</option>
              <option value="GBPUSD">GBPUSD</option>
              <option value="USDJPY">USDJPY</option>
              <option value="XAUUSD">XAUUSD</option>
              <option value="NAS100">NAS100</option>
            </select>
          </div>
          <div class="timeframe-selector">
            <label>Timeframe:</label>
            <select bind:value={chartTimeframe}>
              <option value="M1">M1</option>
              <option value="M5">M5</option>
              <option value="M15">M15</option>
              <option value="H1">H1</option>
              <option value="H4">H4</option>
              <option value="D1">D1</option>
            </select>
          </div>
        </div>
        <div class="chart-wrapper">
          <TradingViewChart 
            symbol={chartSymbol} 
            timeframe={chartTimeframe}
            wsEnabled={true}
            showVolume={true}
            showTrades={true}
            showRegimes={true}
          />
        </div>
      </div>

    {:else if activeView === 'journal'}
      <!-- Trade Journal View -->
      <TradeJournalView />
    {/if}
  </div>
</div>

<style>
  .live-trading-container {
    display: flex;
    flex-direction: column;
    height: 100%;
    background: var(--bg-primary);
    overflow: hidden;
  }

  .sub-tabs {
    display: flex;
    gap: 4px;
    padding: 8px;
    border-bottom: 1px solid var(--border-subtle);
    background: var(--bg-secondary);
  }

  .sub-tab {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 10px 16px;
    background: transparent;
    border: none;
    border-radius: 8px;
    color: var(--text-muted);
    cursor: pointer;
    transition: all 0.15s ease;
    font-size: 13px;
  }

  .sub-tab:hover {
    background: var(--bg-tertiary);
    color: var(--text-primary);
  }

  .sub-tab.active {
    background: var(--accent-primary);
    color: var(--bg-primary);
  }

  .sub-content {
    flex: 1;
    padding: 16px;
    overflow-y: auto;
  }

  /* Dashboard Styles */
  .live-dashboard {
    display: flex;
    flex-direction: column;
    gap: 20px;
  }

  .stats-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
  }

  .stat-card {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 16px;
    background: var(--bg-secondary);
    border: 1px solid var(--border-subtle);
    border-radius: 10px;
  }

  .stat-card :global(svg) {
    color: var(--accent-primary);
  }

  .stat-info {
    display: flex;
    flex-direction: column;
  }

  .stat-value {
    font-size: 18px;
    font-weight: 600;
    color: var(--text-primary);
  }

  .stat-value.positive {
    color: #10b981;
  }

  .stat-label {
    font-size: 11px;
    color: var(--text-muted);
  }

  .sections-row {
    display: flex;
    gap: 12px;
  }

  .section-card {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 12px 16px;
    background: var(--bg-secondary);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
    color: var(--text-secondary);
    cursor: pointer;
    flex: 1;
    font-size: 13px;
  }

  .section-card:hover {
    border-color: var(--accent-primary);
    color: var(--accent-primary);
  }

  .section-card.danger {
    border-color: #ef4444;
    color: #ef4444;
  }

  .section-card.danger:hover {
    background: rgba(239, 68, 68, 0.1);
  }

  .broker-section h3 {
    font-size: 14px;
    margin: 0 0 12px;
    color: var(--text-primary);
  }

  .account-cards {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
  }

  .account-card {
    padding: 16px;
    background: var(--bg-secondary);
    border: 1px solid var(--border-subtle);
    border-radius: 10px;
  }

  .broker-name {
    display: block;
    font-size: 14px;
    font-weight: 500;
    color: var(--text-primary);
    margin-bottom: 8px;
  }

  .balance-row {
    display: flex;
    justify-content: space-between;
    font-size: 12px;
    color: var(--text-muted);
    margin-bottom: 12px;
  }

  .login-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 4px;
    padding: 6px 10px;
    background: var(--accent-primary);
    border: none;
    border-radius: 4px;
    color: var(--bg-primary);
    font-size: 11px;
    cursor: pointer;
  }

  /* Bots View Styles */
  .bots-view h2 {
    margin: 0 0 20px;
    color: var(--text-primary);
  }

  .bot-cards {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 12px;
  }

  .bot-detail-card {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 16px;
    background: var(--bg-secondary);
    border: 1px solid var(--border-subtle);
    border-radius: 10px;
  }

  .bot-status-indicator {
    width: 8px;
    height: 8px;
    border-radius: 50%;
  }

  .bot-main {
    flex: 1;
  }

  .bot-main h4 {
    margin: 0;
    font-size: 13px;
    color: var(--text-primary);
  }

  .bot-main p {
    margin: 4px 0 0;
    font-size: 11px;
    color: var(--text-muted);
  }

  .state-badge {
    padding: 4px 10px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 500;
    text-transform: uppercase;
  }

  /* Terminal Section */
  .terminal-section {
    margin-top: 20px;
    background: var(--bg-secondary);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
    overflow: hidden;
    transition: all 0.3s ease;
  }

  .terminal-section.collapsed {
    max-height: 48px;
  }

  .terminal-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 16px;
    background: var(--bg-tertiary);
    cursor: pointer;
    user-select: none;
    border-bottom: 1px solid var(--border-subtle);
  }

  .terminal-header:hover {
    background: rgba(0, 255, 0, 0.05);
  }

  .terminal-title {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 13px;
    font-weight: 600;
    color: var(--text-primary);
  }

  .terminal-icon {
    color: var(--accent-primary);
    font-family: 'Monaco', 'Courier New', monospace;
  }

  .terminal-toggle {
    background: transparent;
    border: none;
    color: var(--text-muted);
    cursor: pointer;
    padding: 4px;
    border-radius: 4px;
    transition: all 0.15s;
  }

  .terminal-toggle:hover {
    background: var(--bg-tertiary);
    color: var(--text-primary);
  }

  .terminal-content {
    padding: 16px;
    max-height: 300px;
    overflow-y: auto;
  }

  .terminal-output {
    font-family: 'Monaco', 'Courier New', monospace;
    font-size: 12px;
    line-height: 1.4;
    color: var(--text-primary);
    margin-bottom: 12px;
  }

  .terminal-line {
    margin-bottom: 2px;
    word-wrap: break-word;
  }

  .terminal-line.info {
    color: var(--text-secondary);
  }

  .terminal-line.trade {
    color: var(--accent-success);
    font-weight: 600;
  }

  .terminal-line.error {
    color: var(--accent-danger);
  }

  .terminal-input {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px;
    background: var(--bg-primary);
    border: 1px solid var(--border-subtle);
    border-radius: 4px;
  }

  .prompt {
    color: var(--accent-primary);
    font-family: 'Monaco', 'Courier New', monospace;
    font-weight: 600;
  }

  .terminal-input-field {
    flex: 1;
    background: transparent;
    border: none;
    outline: none;
    color: var(--text-primary);
    font-family: 'Monaco', 'Courier New', monospace;
    font-size: 12px;
  }

  .terminal-input-field::placeholder {
    color: var(--text-muted);
  }

  .chart-view {
    display: flex;
    flex-direction: column;
    height: 100%;
    padding: 16px;
  }

  .chart-controls-bar {
    display: flex;
    gap: 16px;
    margin-bottom: 16px;
    padding: 12px;
    background-color: #111827;
    border-radius: 8px;
  }

  .symbol-selector,
  .timeframe-selector {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .symbol-selector label,
  .timeframe-selector label {
    color: #9ca3af;
    font-size: 14px;
  }

  .symbol-selector select,
  .timeframe-selector select {
    padding: 6px 12px;
    background-color: #1f2937;
    color: #f3f4f6;
    border: 1px solid #374151;
    border-radius: 6px;
    font-size: 14px;
    cursor: pointer;
  }

  .symbol-selector select:hover,
  .timeframe-selector select:hover {
    border-color: #4b5563;
  }

  .chart-wrapper {
    flex: 1;
    min-height: 400px;
    border-radius: 8px;
    overflow: hidden;
  }
</style>
