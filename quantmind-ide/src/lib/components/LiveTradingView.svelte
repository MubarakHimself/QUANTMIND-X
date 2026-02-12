<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import { PlayCircle, ShieldAlert, Newspaper, FileText, Activity, Bot, Wallet, TrendingUp, ExternalLink } from 'lucide-svelte';
  import KillSwitchView from './KillSwitchView.svelte';
  import NewsView from './NewsView.svelte';
  import TradeJournalView from './TradeJournalView.svelte';
  import { navigationStore } from '../stores/navigationStore';

  const dispatch = createEventDispatcher();

  export let activeView = 'dashboard';

  const subTabs = [
    { id: 'dashboard', icon: Activity, label: 'Dashboard' },
    { id: 'bots', icon: PlayCircle, label: 'Active Bots' },
    { id: 'kill-switch', icon: ShieldAlert, label: 'Kill Switch' },
    { id: 'news', icon: Newspaper, label: 'News & Kill Zones' },
    { id: 'journal', icon: FileText, label: 'Trade Journal' }
  ];

  // System status mock data
  let systemStatus = {
    active_bots: 3,
    pnl_today: 1250.50,
    regime: 'Trending',
    kelly: 0.85
  };

  // Mock bots data
  let bots = [
    { id: 'ict-eu', name: 'ICT_Scalper @EURUSD', state: 'primal', symbol: 'EURUSD' },
    { id: 'ict-gb', name: 'ICT_Scalper @GBPUSD', state: 'primal', symbol: 'GBPUSD' },
    { id: 'smc-ej', name: 'SMC_Reversal @USDJPY', state: 'ready', symbol: 'USDJPY' }
  ];

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
            <div class="account-card">
              <span class="broker-name">RoboForex Prime</span>
              <div class="balance-row">
                <span>Balance: $5,000</span>
                <span>Equity: $5,250.50</span>
              </div>
              <button class="login-btn">
                <ExternalLink size={12} />
                Connect
              </button>
            </div>
            <div class="account-card">
              <span class="broker-name">Exness Raw</span>
              <div class="balance-row">
                <span>Balance: $2,500</span>
                <span>Equity: $2,500</span>
              </div>
              <button class="login-btn">
                <ExternalLink size={12} />
                Connect
              </button>
            </div>
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
</style>
