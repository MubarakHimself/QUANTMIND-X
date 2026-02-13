<script lang="ts">
  import { createEventDispatcher, onMount } from 'svelte';
  import { theme, currentTheme } from '../stores/themeStore';
  import {
    TrendingUp, Activity, Bot, BookOpen, Edit3, Settings,
    Zap, Target, BarChart3, MonitorPlay, Database, ChevronRight,
    Play, ArrowRight, Star, Clock, AlertTriangle
  } from 'lucide-svelte';

  const dispatch = createEventDispatcher();

  let marketData = {
    eurUsd: { price: 1.0876, change: 0.0023, changePercent: 0.21 },
    gbpUsd: { price: 1.2743, change: -0.0018, changePercent: -0.14 },
    usdJpy: { price: 148.92, change: 0.47, changePercent: 0.32 },
    btcUsd: { price: 43256.78, change: 1256.34, changePercent: 2.98 }
  };

  let activeBots = 3;
  let totalPnL = 2456.89;
  let todayTrades = 47;
  let winRate = 68.5;

  // Animated background based on theme
  $: themeColors = $theme.colors;
  $: themeEffects = $theme.effects;

  function navigateToView(view: string) {
    dispatch('viewChange', { view });
  }

  function openLiveTrading() {
    navigateToView('live');
  }

  function openEditor() {
    navigateToView('editor');
  }

  function openKnowledge() {
    navigateToView('knowledge');
  }

  // Auto-update market data
  onMount(() => {
    const interval = setInterval(() => {
      // Simulate market data updates
      marketData.eurUsd.price += (Math.random() - 0.5) * 0.001;
      marketData.gbpUsd.price += (Math.random() - 0.5) * 0.001;
      marketData.usdJpy.price += (Math.random() - 0.5) * 0.5;
      marketData.btcUsd.price += (Math.random() - 0.5) * 100;
    }, 2000);

    return () => clearInterval(interval);
  });
</script>

<div class="welcome-page" 
     class:scanlines={themeEffects.scanlines}
     class:glow={themeEffects.glow}
     data-theme={$theme.name}>
  
  <!-- Animated Background -->
  {#if $theme.wallpaper}
    <div class="background" style="background: {$theme.wallpaper.value}"></div>
  {/if}
  
  <div class="content">
    <!-- Header -->
    <header class="header">
      <div class="logo">
        <div class="logo-icon">
          <TrendingUp size={32} />
        </div>
        <div class="logo-text">
          <h1>QuantMindX</h1>
          <span class="tagline">AI-Powered Trading System</span>
        </div>
      </div>
      
      <div class="header-actions">
        <button class="action-btn primary" on:click={openLiveTrading}>
          <MonitorPlay size={16} />
          Live Trading
        </button>
        <button class="action-btn" on:click={openEditor}>
          <Edit3 size={16} />
          Editor
        </button>
        <button class="action-btn" on:click={() => navigateToView('settings')}>
          <Settings size={16} />
          Settings
        </button>
      </div>
    </header>

    <!-- Main Dashboard -->
    <main class="main">
      <!-- Live Market Overview -->
      <section class="market-overview">
        <div class="section-header">
          <h2>Live Market Overview</h2>
          <div class="live-indicator">
            <span class="pulse-dot"></span>
            LIVE
          </div>
        </div>
        
        <div class="market-grid">
          {#each Object.entries(marketData) as [symbol, data]}
            <div class="market-card">
              <div class="symbol">{symbol.replace('Usd', '/USD').toUpperCase()}</div>
              <div class="price">{data.price.toFixed(4)}</div>
              <div class="change" class:positive={data.change > 0} class:negative={data.change < 0}>
                {data.change > 0 ? '+' : ''}{data.change.toFixed(4)} ({data.changePercent > 0 ? '+' : ''}{data.changePercent.toFixed(2)}%)
              </div>
            </div>
          {/each}
        </div>
      </section>

      <!-- Quick Stats -->
      <section class="quick-stats">
        <div class="stats-grid">
          <div class="stat-card">
            <div class="stat-icon">
              <Bot size={24} />
            </div>
            <div class="stat-content">
              <div class="stat-value">{activeBots}</div>
              <div class="stat-label">Active Bots</div>
            </div>
          </div>
          
          <div class="stat-card">
            <div class="stat-icon">
              <TrendingUp size={24} />
            </div>
            <div class="stat-content">
              <div class="stat-value" class:positive={totalPnL > 0}>${totalPnL.toFixed(2)}</div>
              <div class="stat-label">Today's P&L</div>
            </div>
          </div>
          
          <div class="stat-card">
            <div class="stat-icon">
              <BarChart3 size={24} />
            </div>
            <div class="stat-content">
              <div class="stat-value">{todayTrades}</div>
              <div class="stat-label">Today's Trades</div>
            </div>
          </div>
          
          <div class="stat-card">
            <div class="stat-icon">
              <Target size={24} />
            </div>
            <div class="stat-content">
              <div class="stat-value">{winRate.toFixed(1)}%</div>
              <div class="stat-label">Win Rate</div>
            </div>
          </div>
        </div>
      </section>

      <!-- Quick Actions -->
      <section class="quick-actions">
        <h2>Quick Actions</h2>
        <div class="actions-grid">
          <button class="action-card" on:click={openLiveTrading}>
            <div class="action-icon">
              <MonitorPlay size={32} />
            </div>
            <div class="action-content">
              <h3>Live Trading</h3>
              <p>Monitor and control active trading bots</p>
            </div>
            <ChevronRight size={20} class="action-arrow" />
          </button>

          <button class="action-card" on:click={openEditor}>
            <div class="action-icon">
              <Edit3 size={32} />
            </div>
            <div class="action-content">
              <h3>Editor Workspace</h3>
              <p>Edit strategies, indicators, and code</p>
            </div>
            <ChevronRight size={20} class="action-arrow" />
          </button>

          <button class="action-card" on:click={openKnowledge}>
            <div class="action-icon">
              <BookOpen size={32} />
            </div>
            <div class="action-content">
              <h3>Knowledge Hub</h3>
              <p>Access 1,800+ trading articles and resources</p>
            </div>
            <ChevronRight size={20} class="action-arrow" />
          </button>

          <button class="action-card" on:click={() => navigateToView('backtest')}>
            <div class="action-icon">
              <Activity size={32} />
            </div>
            <div class="action-content">
              <h3>Backtesting</h3>
              <p>Test strategies with historical data</p>
            </div>
            <ChevronRight size={20} class="action-arrow" />
          </button>
        </div>
      </section>

      <!-- Recent Activity -->
      <section class="recent-activity">
        <h2>Recent Activity</h2>
        <div class="activity-list">
          <div class="activity-item">
            <div class="activity-icon success">
              <Play size={16} />
            </div>
            <div class="activity-content">
              <span class="activity-text">ICT Scalper v2 started trading EUR/USD</span>
              <span class="activity-time">2 minutes ago</span>
            </div>
          </div>
          
          <div class="activity-item">
            <div class="activity-icon warning">
              <AlertTriangle size={16} />
            </div>
            <div class="activity-content">
              <span class="activity-text">SMC Reversal hit maximum drawdown limit</span>
              <span class="activity-time">15 minutes ago</span>
            </div>
          </div>
          
          <div class="activity-item">
            <div class="activity-icon info">
              <Star size={16} />
            </div>
            <div class="activity-content">
              <span class="activity-text">New backtest completed for Breakthrough EA</span>
              <span class="activity-time">1 hour ago</span>
            </div>
          </div>
        </div>
      </section>
    </main>
  </div>
</div>

<style>
  .welcome-page {
    display: flex;
    flex-direction: column;
    height: 100vh;
    position: relative;
    overflow: hidden;
    background: var(--bg-primary);
    color: var(--text-primary);
  }

  .background {
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    z-index: 0;
  }

  .content {
    position: relative;
    z-index: 1;
    display: flex;
    flex-direction: column;
    height: 100%;
  }

  /* Theme effects */
  .welcome-page.scanlines::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: linear-gradient(
      transparent 50%,
      rgba(0, 255, 0, 0.03) 50%
    );
    background-size: 100% 4px;
    pointer-events: none;
    z-index: 1;
  }

  .welcome-page.glow {
    box-shadow: inset 0 0 50px rgba(0, 255, 0, 0.1);
  }

  /* Header */
  .header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 20px 40px;
    background: rgba(0, 0, 0, 0.3);
    backdrop-filter: blur(10px);
    border-bottom: 1px solid var(--border-subtle);
  }

  .logo {
    display: flex;
    align-items: center;
    gap: 16px;
  }

  .logo-icon {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 56px;
    height: 56px;
    background: var(--accent-primary);
    border-radius: 12px;
    color: white;
    animation: pulse 2s infinite;
  }

  .logo-text h1 {
    margin: 0;
    font-size: 28px;
    font-weight: 700;
    color: var(--text-primary);
  }

  .tagline {
    font-size: 14px;
    color: var(--text-muted);
  }

  .header-actions {
    display: flex;
    gap: 12px;
  }

  .action-btn {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 12px 20px;
    background: transparent;
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
    color: var(--text-secondary);
    cursor: pointer;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  }

  .action-btn:hover {
    background: var(--bg-tertiary);
    color: var(--text-primary);
    transform: translateY(-2px);
  }

  .action-btn.primary {
    background: var(--accent-primary);
    border-color: var(--accent-primary);
    color: white;
  }

  .action-btn.primary:hover {
    background: var(--accent-secondary);
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(0, 255, 0, 0.3);
  }

  /* Main Content */
  .main {
    flex: 1;
    padding: 40px;
    overflow-y: auto;
    display: grid;
    grid-template-columns: 1fr;
    gap: 32px;
    max-width: 1400px;
    margin: 0 auto;
  }

  /* Market Overview */
  .market-overview {
    background: rgba(0, 0, 0, 0.3);
    backdrop-filter: blur(10px);
    border: 1px solid var(--border-subtle);
    border-radius: 16px;
    padding: 24px;
  }

  .section-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 20px;
  }

  .section-header h2 {
    margin: 0;
    font-size: 20px;
    font-weight: 600;
    color: var(--text-primary);
  }

  .live-indicator {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 6px 12px;
    background: rgba(255, 0, 0, 0.1);
    border: 1px solid rgba(255, 0, 0, 0.3);
    border-radius: 20px;
    color: #ff0000;
    font-size: 12px;
    font-weight: 600;
  }

  .pulse-dot {
    width: 8px;
    height: 8px;
    background: #ff0000;
    border-radius: 50%;
    animation: pulse 1s infinite;
  }

  .market-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 16px;
  }

  .market-card {
    background: var(--bg-secondary);
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    padding: 16px;
    text-align: center;
    transition: all 0.3s;
  }

  .market-card:hover {
    transform: translateY(-2px);
    border-color: var(--accent-primary);
  }

  .symbol {
    font-size: 14px;
    color: var(--text-muted);
    margin-bottom: 8px;
  }

  .price {
    font-size: 24px;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 8px;
  }

  .change {
    font-size: 14px;
    font-weight: 600;
  }

  .change.positive {
    color: var(--accent-success);
  }

  .change.negative {
    color: var(--accent-danger);
  }

  /* Quick Stats */
  .quick-stats {
    background: rgba(0, 0, 0, 0.3);
    backdrop-filter: blur(10px);
    border: 1px solid var(--border-subtle);
    border-radius: 16px;
    padding: 24px;
  }

  .stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 20px;
  }

  .stat-card {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 20px;
    background: var(--bg-secondary);
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    transition: all 0.3s;
  }

  .stat-card:hover {
    transform: translateY(-2px);
    border-color: var(--accent-primary);
  }

  .stat-icon {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 48px;
    height: 48px;
    background: var(--accent-primary);
    border-radius: 12px;
    color: white;
  }

  .stat-value {
    font-size: 24px;
    font-weight: 700;
    color: var(--text-primary);
  }

  .stat-value.positive {
    color: var(--accent-success);
  }

  .stat-label {
    font-size: 14px;
    color: var(--text-muted);
  }

  /* Quick Actions */
  .quick-actions h2 {
    margin: 0 0 20px 0;
    font-size: 20px;
    font-weight: 600;
    color: var(--text-primary);
  }

  .actions-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 20px;
  }

  .action-card {
    display: flex;
    align-items: center;
    gap: 20px;
    padding: 24px;
    background: var(--bg-secondary);
    border: 1px solid var(--border-subtle);
    border-radius: 16px;
    cursor: pointer;
    transition: all 0.3s;
    text-align: left;
  }

  .action-card:hover {
    transform: translateY(-2px);
    border-color: var(--accent-primary);
    box-shadow: 0 8px 25px rgba(0, 255, 0, 0.2);
  }

  .action-icon {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 64px;
    height: 64px;
    background: var(--accent-primary);
    border-radius: 16px;
    color: white;
  }

  .action-content h3 {
    margin: 0 0 8px 0;
    font-size: 18px;
    font-weight: 600;
    color: var(--text-primary);
  }

  .action-content p {
    margin: 0;
    font-size: 14px;
    color: var(--text-muted);
  }

  .action-arrow {
    margin-left: auto;
    color: var(--text-muted);
  }

  /* Recent Activity */
  .recent-activity h2 {
    margin: 0 0 20px 0;
    font-size: 20px;
    font-weight: 600;
    color: var(--text-primary);
  }

  .activity-list {
    display: flex;
    flex-direction: column;
    gap: 16px;
  }

  .activity-item {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 16px;
    background: var(--bg-secondary);
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
  }

  .activity-icon {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 32px;
    height: 32px;
    border-radius: 8px;
    color: white;
  }

  .activity-icon.success {
    background: var(--accent-success);
  }

  .activity-icon.warning {
    background: var(--accent-warning);
  }

  .activity-icon.info {
    background: var(--accent-primary);
  }

  .activity-content {
    flex: 1;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .activity-text {
    font-size: 14px;
    color: var(--text-primary);
  }

  .activity-time {
    font-size: 12px;
    color: var(--text-muted);
  }

  /* Animations */
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
  }

  /* Theme-specific overrides */
  .welcome-page[data-theme="trading-terminal"] .logo-icon {
    background: #00ff00;
    color: #000000;
    text-shadow: 0 0 10px #00ff00;
  }

  .welcome-page[data-theme="trading-terminal"] .live-indicator {
    color: #00ff00;
    border-color: #00ff00;
    background: rgba(0, 255, 0, 0.1);
  }

  .welcome-page[data-theme="trading-terminal"] .pulse-dot {
    background: #00ff00;
  }

  .welcome-page[data-theme="matrix"] .logo-icon {
    background: #00ff00;
    color: #000000;
    font-family: 'Courier New', monospace;
  }

  .welcome-page[data-theme="cyberpunk"] .logo-icon {
    background: linear-gradient(45deg, #00ffff, #ff00ff);
  }
</style>
