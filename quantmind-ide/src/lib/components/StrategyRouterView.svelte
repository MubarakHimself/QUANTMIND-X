<script lang="ts">
  import { createEventDispatcher, onMount, onDestroy } from 'svelte';
  import {
    Server, Zap, Activity, TrendingUp, AlertTriangle, Shield,
    Clock, DollarSign, BarChart3, Layers, Eye, EyeOff,
    RefreshCw, Play, Pause, SkipForward, FastForward,
    ChevronRight, ChevronDown, X, Check, AlertCircle,
    ArrowUpDown, Target, Award, Trophy, Gauge, MonitorPlay,
    Globe, Currency, Package, Settings as SettingsIcon, Scale,
    Calculator, TrendingUp as TrendingUpIcon
  } from 'lucide-svelte';

  const dispatch = createEventDispatcher();

  // Router State
  let routerState = {
    active: true,
    mode: 'auction' as 'auction' | 'priority' | 'round-robin',
    auctionInterval: 5000, // ms
    lastAuction: null as Date | null,
    queuedSignals: [] as Array<any>,
    activeAuctions: [] as Array<any>
  };

  // Market State
  let marketState = {
    regime: {
      quality: 0.82,
      trend: 'bullish' as 'bullish' | 'bearish' | 'ranging',
      chaos: 18.5,
      volatility: 'medium' as 'low' | 'medium' | 'high'
    },
    symbols: [
      { symbol: 'EURUSD', price: 1.0876, change: +0.12, spread: 1.2 },
      { symbol: 'GBPUSD', price: 1.2654, change: -0.08, spread: 1.5 },
      { symbol: 'USDJPY', price: 149.85, change: +0.25, spread: 0.8 }
    ]
  };

  // Bots/Strategies
  let bots = [
    {
      id: 'ict-eur',
      name: 'ICT Scalper',
      symbol: 'EURUSD',
      status: 'ready' as 'idle' | 'ready' | 'paused' | 'quarantined',
      signalStrength: 0.85,
      conditions: ['fvg', 'order_block', 'london_session'],
      score: 8.5,
      lastSignal: new Date(Date.now() - 120000)
    },
    {
      id: 'smc-gbp',
      name: 'SMC Reversal',
      symbol: 'GBPUSD',
      status: 'ready',
      signalStrength: 0.72,
      conditions: ['choch', 'bos', 'session_filter'],
      score: 7.2,
      lastSignal: new Date(Date.now() - 300000)
    },
    {
      id: 'breakthrough-eur',
      name: 'Breakthrough Hunter',
      symbol: 'EURUSD',
      status: 'idle',
      signalStrength: 0.0,
      conditions: ['breakout', 'volume_confirm'],
      score: 0,
      lastSignal: null
    }
  ];

  // Auction Queue
  let auctionQueue = [
    {
      id: 'auction-1',
      timestamp: new Date(),
      participants: ['ict-eur', 'smc-gbp'],
      winner: 'ict-eur',
      winningScore: 8.5,
      status: 'completed'
    }
  ];

  // Rankings
  let rankings = {
    daily: [
      { botId: 'ict-eur', name: 'ICT Scalper', profit: 245.80, trades: 12, winRate: 75 },
      { botId: 'smc-gbp', name: 'SMC Reversal', profit: 128.50, trades: 8, winRate: 62.5 },
      { botId: 'breakthrough-eur', name: 'Breakthrough', profit: 0, trades: 0, winRate: 0 }
    ],
    weekly: [
      { botId: 'ict-eur', name: 'ICT Scalper', profit: 1245.60, trades: 58, winRate: 70.7 },
      { botId: 'smc-gbp', name: 'SMC Reversal', profit: 678.30, trades: 42, winRate: 64.3 },
      { botId: 'breakthrough-eur', name: 'Breakthrough', profit: 234.20, trades: 15, winRate: 66.7 }
    ]
  };

  // Correlations
  let correlations = [
    { pair: 'EURUSD/GBPUSD', value: 0.72, status: 'warning' as 'ok' | 'warning' | 'danger' },
    { pair: 'EURUSD/USDJPY', value: -0.45, status: 'ok' },
    { pair: 'GBPUSD/USDJPY', value: -0.38, status: 'ok' }
  ];

  // House Money State
  let houseMoney = {
    dailyProfit: 374.30,
    threshold: 0.5,
    houseMoneyAmount: 187.15,
    mode: 'aggressive' as 'conservative' | 'normal' | 'aggressive'
  };

  // Kelly Criterion State
  interface KellyData {
    kellyFraction: number;
    halfKelly: number;
    winRate: number;
    avgWin: number;
    avgLoss: number;
    expectedValue: number;
    suggestedFraction: number;
  }

  let kellyData: Record<string, KellyData> = {
    'ict-eur': { kellyFraction: 0.125, halfKelly: 0.0625, winRate: 0.72, avgWin: 45.20, avgLoss: 22.80, expectedValue: 16.22, suggestedFraction: 0.08 },
    'smc-gbp': { kellyFraction: 0.085, halfKelly: 0.0425, winRate: 0.65, avgWin: 38.50, avgLoss: 25.40, expectedValue: 9.84, suggestedFraction: 0.06 },
    'breakthrough-eur': { kellyFraction: 0.042, halfKelly: 0.021, winRate: 0.58, avgWin: 52.30, avgLoss: 41.20, expectedValue: 3.86, suggestedFraction: 0.03 }
  };

  let kellyHistory: Array<{date: string, botId: string, fraction: number, result: number}> = [
    { date: '2024-01-10', botId: 'ict-eur', fraction: 0.125, result: 245.80 },
    { date: '2024-01-09', botId: 'smc-gbp', fraction: 0.085, result: 128.50 },
    { date: '2024-01-08', botId: 'ict-eur', fraction: 0.125, result: 312.40 }
  ];

  // MT5 Connection State
  let mt5Connected = false;
  let mt5Testing = false;
  let mt5Error = '';
  let mt5SymbolMappingPlaceholder = '{"EURUSDm": "EURUSD", "GBPUSDm": "GBPUSD", "USDJPYm": "USDJPY"}';
  let mt5Config = {
    server: '',
    port: 443,
    login: '',
    password: '',
    symbolMapping: mt5SymbolMappingPlaceholder
  };

  async function testMt5Connection() {
    mt5Testing = true;
    mt5Error = '';
    
    try {
      const res = await fetch('http://localhost:8000/api/mt5/test', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(mt5Config)
      });
      
      if (res.ok) {
        mt5Connected = true;
      } else {
        mt5Error = 'Connection failed. Please check your credentials.';
        mt5Connected = false;
      }
    } catch (e) {
      mt5Error = 'Connection failed. Is the MT5 bridge running?';
      mt5Connected = false;
    } finally {
      mt5Testing = false;
    }
  }

  async function saveMt5Config() {
    try {
      await fetch('http://localhost:8000/api/mt5/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(mt5Config)
      });
    } catch (e) {
      console.error('Failed to save MT5 config:', e);
    }
  }

  // Kelly Rankings
  $: kellyRankings = Object.entries(kellyData)
    .map(([botId, data]) => ({
      botId,
      name: bots.find(b => b.id === botId)?.name || botId,
      kellyFraction: data.kellyFraction,
      halfKelly: data.halfKelly,
      winRate: data.winRate,
      expectedValue: data.expectedValue,
      suggestedFraction: data.suggestedFraction,
      kellyScore: (data.kellyFraction * data.expectedValue * 100).toFixed(2)
    }))
    .sort((a, b) => parseFloat(b.kellyScore) - parseFloat(a.kellyScore));

  // View state
  let activeTab: 'auction' | 'rankings' | 'kelly' | 'correlations' | 'settings' = 'auction';
  let autoRefresh = true;
  let refreshInterval: number | null = null;

  onMount(() => {
    loadRouterState();
    if (autoRefresh) {
      refreshInterval = window.setInterval(() => {
        loadRouterState();
      }, 5000);
    }
  });

  onDestroy(() => {
    if (refreshInterval) {
      clearInterval(refreshInterval);
    }
  });

  async function loadRouterState() {
    try {
      const res = await fetch('http://localhost:8000/api/router/state');
      if (res.ok) {
        const data = await res.json();
        routerState = { ...routerState, ...data };
      }
    } catch (e) {
      console.error('Failed to load router state:', e);
    }
  }

  async function toggleRouter() {
    routerState.active = !routerState.active;

    try {
      await fetch('http://localhost:8000/api/router/toggle', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ active: routerState.active })
      });
    } catch (e) {
      console.error('Failed to toggle router:', e);
    }
  }

  async function runAuction() {
    try {
      const res = await fetch('http://localhost:8000/api/router/auction', {
        method: 'POST'
      });
      if (res.ok) {
        const result = await res.json();
        auctionQueue = [result, ...auctionQueue].slice(0, 10);
      }
    } catch (e) {
      console.error('Failed to run auction:', e);
    }
  }

  function getScoreColor(score: number) {
    if (score >= 8) return '#10b981';
    if (score >= 6) return '#f59e0b';
    return '#6b7280';
  }

  function getStatusColor(status: string) {
    const colors: Record<string, string> = {
      ready: '#10b981',
      idle: '#6b7280',
      paused: '#f59e0b',
      quarantined: '#ef4444'
    };
    return colors[status] || '#6b7280';
  }

  function getCorrelationStatus(value: number) {
    if (Math.abs(value) >= 0.7) return 'danger';
    if (Math.abs(value) >= 0.5) return 'warning';
    return 'ok';
  }

  function formatCurrency(value: number) {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2
    }).format(value);
  }

  function timeAgo(date: Date | null) {
    if (!date) return 'Never';
    const seconds = Math.floor((Date.now() - date.getTime()) / 1000);
    if (seconds < 60) return `${seconds}s ago`;
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
    return `${Math.floor(seconds / 3600)}h ago`;
  }
</script>

<div class="router-view">
  <!-- Header -->
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
      <button class="btn" on:click={() => autoRefresh = !autoRefresh} class:active={autoRefresh}>
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
    </div>
  </div>

  <!-- Market Overview -->
  <div class="market-overview">
    <div class="overview-card regime">
      <div class="card-header">
        <Activity size={16} />
        <h3>Market Regime</h3>
      </div>
      <div class="regime-display">
        <div class="regime-quality">
          <span class="quality-label">Quality</span>
          <div class="quality-bar">
            <div class="quality-fill" style="width: {marketState.regime.quality * 100}%"></div>
          </div>
          <span class="quality-value">{(marketState.regime.quality * 100).toFixed(1)}%</span>
        </div>
        <div class="regime-details">
          <div class="regime-item">
            <span class="label">Trend</span>
            <span class="value {marketState.regime.trend}">{marketState.regime.trend}</span>
          </div>
          <div class="regime-item">
            <span class="label">Chaos</span>
            <span class="value">{marketState.regime.chaos.toFixed(1)}</span>
          </div>
          <div class="regime-item">
            <span class="label">Volatility</span>
            <span class="value">{marketState.regime.volatility}</span>
          </div>
        </div>
      </div>
    </div>

    <div class="overview-card symbols">
      <div class="card-header">
        <Currency size={16} />
        <h3>Active Symbols</h3>
      </div>
      <div class="symbols-list">
        {#each marketState.symbols as symbol}
          <div class="symbol-item">
            <span class="symbol-name">{symbol.symbol}</span>
            <span class="symbol-price">{symbol.price.toFixed(4)}</span>
            <span class="symbol-change" class:positive={symbol.change > 0} class:negative={symbol.change < 0}>
              {symbol.change > 0 ? '+' : ''}{symbol.change.toFixed(2)}%
            </span>
            <span class="symbol-spread">Spread: {symbol.spread}</span>
          </div>
        {/each}
      </div>
    </div>

    <div class="overview-card house-money">
      <div class="card-header">
        <DollarSign size={16} />
        <h3>House Money</h3>
        <span class="mode-badge {houseMoney.mode}">{houseMoney.mode}</span>
      </div>
      <div class="house-money-content">
        <div class="hm-profit">
          <span class="label">Daily Profit</span>
          <span class="value success">{formatCurrency(houseMoney.dailyProfit)}</span>
        </div>
        <div class="hm-house">
          <span class="label">House Money</span>
          <span class="value highlight">{formatCurrency(houseMoney.houseMoneyAmount)}</span>
        </div>
        <div class="hm-threshold">
          <span class="label">Threshold</span>
          <span class="value">{(houseMoney.threshold * 100).toFixed(0)}%</span>
        </div>
      </div>
    </div>
  </div>

  <!-- Tabs -->
  <div class="router-tabs">
    <button class="tab" class:active={activeTab === 'auction'} on:click={() => activeTab = 'auction'}>
      <Scale size={14} />
      <span>Auction Queue</span>
    </button>
    <button class="tab" class:active={activeTab === 'rankings'} on:click={() => activeTab = 'rankings'}>
      <Trophy size={14} />
      <span>Rankings</span>
    </button>
    <button class="tab" class:active={activeTab === 'kelly'} on:click={() => activeTab = 'kelly'}>
      <Calculator size={14} />
      <span>Kelly Criterion</span>
    </button>
    <button class="tab" class:active={activeTab === 'correlations'} on:click={() => activeTab = 'correlations'}>
      <Layers size={14} />
      <span>Correlations</span>
    </button>
    <button class="tab" class:active={activeTab === 'settings'} on:click={() => activeTab = 'settings'}>
      <SettingsIcon size={14} />
      <span>Settings</span>
    </button>
  </div>

  <!-- Tab Content -->
  <div class="router-content">
    {#if activeTab === 'auction'}
      <!-- Auction Queue -->
      <div class="auction-section">
        <div class="section-header">
          <h3>Live Auctions</h3>
          <span class="count">{auctionQueue.length} auctions</span>
        </div>

        <div class="auction-list">
          {#each auctionQueue as auction}
            <div class="auction-card">
              <div class="auction-header">
                <div class="auction-time">
                  <Clock size={12} />
                  <span>{timeAgo(auction.timestamp)}</span>
                </div>
                <span class="auction-status {auction.status}">{auction.status}</span>
              </div>

              <div class="auction-participants">
                <span class="participants-label">Participants</span>
                <div class="participants-list">
                  {#each auction.participants as participant}
                    <div class="participant-badge" class:winner={participant === auction.winner}>
                      <span>{participant}</span>
                      {#if participant === auction.winner}
                        <Award size={12} />
                      {/if}
                    </div>
                  {/each}
                </div>
              </div>

              <div class="auction-result">
                <span class="result-label">Winner</span>
                <div class="winner-display">
                  <span class="winner-name">{auction.winner}</span>
                  <span class="winner-score" style="color: {getScoreColor(auction.winningScore)}">
                    Score: {auction.winningScore.toFixed(1)}
                  </span>
                </div>
              </div>
            </div>
          {/each}

          {#if auctionQueue.length === 0}
            <div class="empty-state">
              <Server size={32} />
              <p>No auctions yet. Click "Run Auction" to start.</p>
            </div>
          {/if}
        </div>
      </div>

      <!-- Bot Signals -->
      <div class="signals-section">
        <div class="section-header">
          <h3>Bot Signals</h3>
          <span class="count">{bots.filter(b => b.status === 'ready').length} ready</span>
        </div>

        <div class="bots-grid">
          {#each bots as bot}
            <div class="bot-card" class:status-ready={bot.status === 'ready'} class:status-idle={bot.status === 'idle'}>
              <div class="bot-status" style="background: {getStatusColor(bot.status)}"></div>
              <div class="bot-header">
                <span class="bot-name">{bot.name}</span>
                <span class="bot-symbol">{bot.symbol}</span>
              </div>

              {#if bot.status === 'ready'}
                <div class="bot-signal">
                  <span class="signal-strength" style="color: {getScoreColor(bot.score)}">
                    {bot.score.toFixed(1)}
                  </span>
                  <span class="signal-label">Signal Strength</span>
                </div>

                <div class="bot-conditions">
                  {#each bot.conditions as condition}
                    <span class="condition-tag">{condition}</span>
                  {/each}
                </div>

                <div class="bot-footer">
                  <span class="last-signal">{timeAgo(bot.lastSignal)}</span>
                  <span class="bot-status-text">{bot.status}</span>
                </div>
              {:else}
                <div class="bot-idle">
                  <span class="idle-text">{bot.status}</span>
                </div>
              {/if}
            </div>
          {/each}
        </div>
      </div>

    {:else if activeTab === 'rankings'}
      <!-- Rankings -->
      <div class="rankings-section">
        <div class="rankings-tabs">
          <button class="rank-tab active">Daily</button>
          <button class="rank-tab">Weekly</button>
          <button class="rank-tab">Monthly</button>
        </div>

        <div class="rankings-table">
          <div class="table-header">
            <span>Rank</span>
            <span>Strategy</span>
            <span>Profit</span>
            <span>Trades</span>
            <span>Win Rate</span>
          </div>

          {#each rankings.daily as ranking, index}
            <div class="table-row">
              <span class="rank">#{index + 1}</span>
              <span class="name">{ranking.name}</span>
              <span class="profit" class:positive={ranking.profit > 0}>
                {formatCurrency(ranking.profit)}
              </span>
              <span class="trades">{ranking.trades}</span>
              <span class="winrate">{ranking.winRate.toFixed(1)}%</span>
            </div>
          {/each}
        </div>
      </div>

    {:else if activeTab === 'kelly'}
      <!-- Kelly Criterion -->
      <div class="kelly-section">
        <div class="kelly-header">
          <div class="kelly-info">
            <Calculator size={20} />
            <div>
              <h3>Kelly Criterion Analysis</h3>
              <p>Optimal position sizing based on win rate and risk/reward</p>
            </div>
          </div>
          <div class="kelly-summary">
            <div class="summary-item">
              <span class="label">Avg Kelly</span>
              <span class="value">{(Object.values(kellyData).reduce((a, b) => a + b.kellyFraction, 0) / Object.values(kellyData).length * 100).toFixed(1)}%</span>
            </div>
            <div class="summary-item">
              <span class="label">Avg Half-Kelly</span>
              <span class="value">{(Object.values(kellyData).reduce((a, b) => a + b.halfKelly, 0) / Object.values(kellyData).length * 100).toFixed(1)}%</span>
            </div>
            <div class="summary-item">
              <span class="label">Best Kelly</span>
              <span class="value success">{Math.max(...Object.values(kellyData).map(k => k.kellyFraction * 100)).toFixed(1)}%</span>
            </div>
          </div>
        </div>

        <!-- Kelly Rankings -->
        <div class="kelly-rankings">
          <h4>Bot Rankings by Kelly Score</h4>
          <div class="kelly-table">
            <div class="table-header kelly-header-row">
              <span>Rank</span>
              <span>Bot</span>
              <span>Full Kelly</span>
              <span>Half Kelly</span>
              <span>Win Rate</span>
              <span>EV/Trade</span>
              <span>Kelly Score</span>
            </div>

            {#each kellyRankings as ranking, index}
              <div class="table-row kelly-row">
                <span class="rank kelly-rank">#{index + 1}</span>
                <span class="name">{ranking.name}</span>
                <span class="kelly-value full-kelly" title="Full Kelly - aggressive">
                  <div class="kelly-bar" style="width: {ranking.kellyFraction * 100 * 4}%"></div>
                  {(ranking.kellyFraction * 100).toFixed(1)}%
                </span>
                <span class="kelly-value half-kelly" title="Half Kelly - conservative">
                  <div class="kelly-bar half" style="width: {ranking.halfKelly * 100 * 4}%"></div>
                  {(ranking.halfKelly * 100).toFixed(1)}%
                </span>
                <span class="winrate">{ranking.winRate * 100}%</span>
                <span class="expected-value" class:positive={ranking.expectedValue > 0}>
                  ${ranking.expectedValue.toFixed(2)}
                </span>
                <span class="kelly-score" class:top={index === 0}>
                  {ranking.kellyScore}
                </span>
              </div>
            {/each}
          </div>
        </div>

        <!-- Kelly Details Grid -->
        <div class="kelly-details-grid">
          {#each Object.entries(kellyData) as [botId, data]}
            <div class="kelly-card">
              <div class="kelly-card-header">
                <span class="bot-name">{bots.find(b => b.id === botId)?.name || botId}</span>
                <span class="status-badge" class:optimal={data.kellyFraction < 0.15} class:caution={data.kellyFraction >= 0.15 && data.kellyFraction < 0.25} class:warning={data.kellyFraction >= 0.25}>
                  {data.kellyFraction < 0.15 ? 'Optimal' : data.kellyFraction < 0.25 ? 'Moderate' : 'High Risk'}
                </span>
              </div>
              <div class="kelly-card-body">
                <div class="metric-row">
                  <span class="metric-label">Win Rate</span>
                  <span class="metric-value">{(data.winRate * 100).toFixed(0)}%</span>
                </div>
                <div class="metric-row">
                  <span class="metric-label">Avg Win</span>
                  <span class="metric-value success">${data.avgWin.toFixed(2)}</span>
                </div>
                <div class="metric-row">
                  <span class="metric-label">Avg Loss</span>
                  <span class="metric-value danger">${data.avgLoss.toFixed(2)}</span>
                </div>
                <div class="kelly-visual">
                  <div class="kelly-gauge">
                    <div class="gauge-track">
                      <div class="gauge-fill" style="width: {data.kellyFraction * 100}%"></div>
                      <div class="gauge-half" style="left: {data.halfKelly * 100}%"></div>
                    </div>
                    <div class="gauge-labels">
                      <span>0%</span>
                      <span class="half-mark">Half: {(data.halfKelly * 100).toFixed(1)}%</span>
                      <span>{(data.kellyFraction * 100).toFixed(1)}%</span>
                    </div>
                  </div>
                </div>
                <div class="suggested-fraction">
                  <span class="label">Suggested Fraction:</span>
                  <span class="value">{(data.suggestedFraction * 100).toFixed(0)}% of Kelly</span>
                </div>
              </div>
            </div>
          {/each}
        </div>

        <!-- Kelly History -->
        <div class="kelly-history">
          <h4>Recent Kelly Adjustments</h4>
          <div class="history-list">
            {#each kellyHistory.slice(0, 5) as entry}
              <div class="history-item">
                <span class="history-date">{entry.date}</span>
                <span class="history-bot">{bots.find(b => b.id === entry.botId)?.name || entry.botId}</span>
                <span class="history-fraction">Kelly: {(entry.fraction * 100).toFixed(1)}%</span>
                <span class="history-result" class:positive={entry.result > 0} class:negative={entry.result < 0}>
                  {entry.result > 0 ? '+' : ''}{formatCurrency(entry.result)}
                </span>
              </div>
            {/each}
          </div>
        </div>
      </div>

    {:else if activeTab === 'correlations'}
      <!-- Correlations -->
      <div class="correlations-section">
        <div class="section-header">
          <h3>Symbol Correlations</h3>
          <span class="info">Active positions with high correlation are limited</span>
        </div>

        <div class="correlations-grid">
          {#each correlations as corr}
            <div class="corr-card status-{corr.status}">
              <div class="corr-pair">
                <Layers size={16} />
                <span>{corr.pair}</span>
              </div>
              <div class="corr-value">
                <span class="value" style="color: {getScoreColor(Math.abs(corr.value) * 10)}">
                  {corr.value > 0 ? '+' : ''}{corr.value.toFixed(2)}
                </span>
                <span class="label">{corr.status}</span>
              </div>
            </div>
          {/each}
        </div>

        <div class="correlation-info">
          <AlertTriangle size={14} />
          <p>High correlation (≥0.7) limits simultaneous positions in correlated pairs</p>
        </div>
      </div>

    {:else if activeTab === 'settings'}
      <!-- Settings -->
      <div class="settings-section">
        <div class="setting-group">
          <h3>Router Mode</h3>
          <div class="setting-options">
            <label class="radio-option">
              <input type="radio" bind:group={routerState.mode} value="auction" />
              <span>Auction (Recommended)</span>
              <small>Best signal wins based on scoring</small>
            </label>
            <label class="radio-option">
              <input type="radio" bind:group={routerState.mode} value="priority" />
              <span>Priority</span>
              <small>Higher priority bots go first</small>
            </label>
            <label class="radio-option">
              <input type="radio" bind:group={routerState.mode} value="round-robin" />
              <span>Round Robin</span>
              <small>Equal opportunity for all bots</small>
            </label>
          </div>
        </div>

        <div class="setting-group">
          <h3>Auction Settings</h3>
          <div class="setting-row">
            <span>Interval (ms)</span>
            <input type="number" bind:value={routerState.auctionInterval} min="1000" max="60000" step="1000" />
          </div>
        </div>

        <div class="setting-group">
          <h3>Risk Limits</h3>
          <div class="setting-row">
            <span>Max Correlated Positions</span>
            <input type="number" value="2" min="1" max="5" />
          </div>
          <div class="setting-row">
            <span>Correlation Threshold</span>
            <input type="number" value="0.7" min="0" max="1" step="0.1" />
          </div>
        </div>

        <!-- MT5 Connection -->
        <div class="setting-group mt5-connection">
          <div class="setting-header">
            <Globe size={16} />
            <h3>MT5 Connection</h3>
            <span class="connection-status" class:connected={mt5Connected}>
              <span class="status-dot"></span>
              {mt5Connected ? 'Connected' : 'Disconnected'}
            </span>
          </div>
          
          <div class="mt5-form">
            <div class="form-row">
              <label>Server</label>
              <input type="text" bind:value={mt5Config.server} placeholder="e.g., ICMarkets-Live" />
            </div>
            <div class="form-row">
              <label>Port</label>
              <input type="number" bind:value={mt5Config.port} placeholder="443" />
            </div>
            <div class="form-row">
              <label>Login</label>
              <input type="text" bind:value={mt5Config.login} placeholder="Account ID" />
            </div>
            <div class="form-row">
              <label>Password</label>
              <input type="password" bind:value={mt5Config.password} placeholder="***" />
            </div>
            <div class="form-row">
              <label>Symbol Mapping</label>
              <textarea 
                bind:value={mt5Config.symbolMapping}
                placeholder={mt5SymbolMappingPlaceholder}
                rows="3"
              ></textarea>
            </div>
          </div>
          
          <div class="mt5-actions">
            <button class="btn" on:click={testMt5Connection} disabled={mt5Testing}>
              {mt5Testing ? 'Testing...' : 'Test Connection'}
            </button>
            <button class="btn primary" on:click={saveMt5Config}>
              Save Configuration
            </button>
          </div>
          
          {#if mt5Error}
            <div class="mt5-error">
              <AlertCircle size={14} />
              <span>{mt5Error}</span>
            </div>
          {/if}
        </div>
      </div>
    {/if}
  </div>
</div>

<style>
  .router-view {
    display: flex;
    flex-direction: column;
    height: 100%;
    background: var(--bg-primary);
    overflow: hidden;
  }

  /* Header */
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

  .router-icon {
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

  /* Market Overview */
  .market-overview {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 16px;
    padding: 20px 24px;
  }

  .overview-card {
    background: var(--bg-secondary);
    border: 1px solid var(--border-subtle);
    border-radius: 10px;
    padding: 16px;
  }

  .card-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 12px;
  }

  .card-header h3 {
    margin: 0;
    font-size: 14px;
    color: var(--text-primary);
  }

  /* Regime Card */
  .regime-quality {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 12px;
  }

  .quality-label {
    font-size: 12px;
    color: var(--text-muted);
  }

  .quality-bar {
    flex: 1;
    height: 8px;
    background: var(--bg-tertiary);
    border-radius: 4px;
    overflow: hidden;
  }

  .quality-fill {
    height: 100%;
    background: linear-gradient(90deg, #ef4444, #f59e0b, #10b981);
    transition: width 0.5s;
  }

  .quality-value {
    font-size: 14px;
    font-weight: 600;
    color: var(--text-primary);
  }

  .regime-details {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 8px;
  }

  .regime-item {
    display: flex;
    flex-direction: column;
    gap: 2px;
  }

  .regime-item .label {
    font-size: 10px;
    color: var(--text-muted);
  }

  .regime-item .value {
    font-size: 12px;
    color: var(--text-primary);
  }

  .regime-item .value.bullish {
    color: #10b981;
  }

  .regime-item .value.bearish {
    color: #ef4444;
  }

  /* Symbols Card */
  .symbols-list {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .symbol-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px;
    background: var(--bg-tertiary);
    border-radius: 6px;
    font-size: 12px;
  }

  .symbol-name {
    font-weight: 600;
    color: var(--text-primary);
  }

  .symbol-price {
    font-family: 'JetBrains Mono', monospace;
    color: var(--text-primary);
  }

  .symbol-change {
    font-size: 11px;
  }

  .symbol-change.positive {
    color: #10b981;
  }

  .symbol-change.negative {
    color: #ef4444;
  }

  .symbol-spread {
    font-size: 10px;
    color: var(--text-muted);
  }

  /* House Money Card */
  .mode-badge {
    margin-left: auto;
    padding: 2px 8px;
    border-radius: 10px;
    font-size: 10px;
    text-transform: uppercase;
  }

  .mode-badge.conservative {
    background: rgba(239, 68, 68, 0.2);
    color: #ef4444;
  }

  .mode-badge.normal {
    background: rgba(245, 158, 11, 0.2);
    color: #f59e0b;
  }

  .mode-badge.aggressive {
    background: rgba(16, 185, 129, 0.2);
    color: #10b981;
  }

  .house-money-content {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .hm-profit,
  .hm-house,
  .hm-threshold {
    display: flex;
    justify-content: space-between;
    font-size: 12px;
  }

  .hm-profit .label,
  .hm-house .label,
  .hm-threshold .label {
    color: var(--text-muted);
  }

  .hm-profit .value,
  .hm-house .value,
  .hm-threshold .value {
    font-weight: 600;
    color: var(--text-primary);
  }

  .hm-profit .value.success {
    color: #10b981;
  }

  .hm-house .value.highlight {
    color: #f59e0b;
  }

  /* Tabs */
  .router-tabs {
    display: flex;
    gap: 4px;
    padding: 12px 24px;
    background: var(--bg-secondary);
    border-bottom: 1px solid var(--border-subtle);
  }

  .router-tabs .tab {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 8px 14px;
    background: transparent;
    border: none;
    border-radius: 6px;
    color: var(--text-secondary);
    font-size: 12px;
    cursor: pointer;
  }

  .router-tabs .tab:hover {
    background: var(--bg-tertiary);
    color: var(--text-primary);
  }

  .router-tabs .tab.active {
    background: var(--accent-primary);
    color: var(--bg-primary);
  }

  /* Content */
  .router-content {
    flex: 1;
    padding: 24px;
    overflow-y: auto;
  }

  /* Auction Section */
  .auction-section {
    margin-bottom: 24px;
  }

  .section-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 16px;
  }

  .section-header h3 {
    margin: 0;
    font-size: 14px;
    color: var(--text-primary);
  }

  .count {
    padding: 2px 8px;
    background: var(--bg-tertiary);
    border-radius: 10px;
    font-size: 11px;
    color: var(--text-muted);
  }

  .auction-list {
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  .auction-card {
    background: var(--bg-secondary);
    border: 1px solid var(--border-subtle);
    border-radius: 10px;
    padding: 16px;
  }

  .auction-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
  }

  .auction-time {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 11px;
    color: var(--text-muted);
  }

  .auction-status {
    padding: 4px 10px;
    border-radius: 10px;
    font-size: 10px;
    font-weight: 500;
    text-transform: uppercase;
  }

  .auction-status.completed {
    background: rgba(16, 185, 129, 0.2);
    color: #10b981;
  }

  .auction-participants {
    margin-bottom: 12px;
  }

  .participants-label {
    display: block;
    font-size: 11px;
    color: var(--text-muted);
    margin-bottom: 6px;
  }

  .participants-list {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
  }

  .participant-badge {
    display: flex;
    align-items: center;
    gap: 4px;
    padding: 6px 10px;
    background: var(--bg-tertiary);
    border-radius: 6px;
    font-size: 11px;
    color: var(--text-secondary);
  }

  .participant-badge.winner {
    background: var(--accent-primary);
    color: var(--bg-primary);
  }

  .auction-result {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding-top: 12px;
    border-top: 1px solid var(--border-subtle);
  }

  .result-label {
    font-size: 11px;
    color: var(--text-muted);
  }

  .winner-display {
    display: flex;
    align-items: center;
    gap: 12px;
  }

  .winner-name {
    font-size: 13px;
    font-weight: 500;
    color: var(--text-primary);
  }

  .winner-score {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
  }

  /* Empty State */
  .empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 40px;
    color: var(--text-muted);
    text-align: center;
  }

  .empty-state p {
    margin-top: 12px;
    font-size: 13px;
  }

  /* Signals Section */
  .signals-section {
    margin-bottom: 24px;
  }

  .bots-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 12px;
  }

  .bot-card {
    background: var(--bg-secondary);
    border: 1px solid var(--border-subtle);
    border-radius: 10px;
    padding: 16px;
    position: relative;
    transition: all 0.15s;
  }

  .bot-card.status-ready {
    border-color: #10b981;
  }

  .bot-status {
    position: absolute;
    top: 16px;
    right: 16px;
    width: 8px;
    height: 8px;
    border-radius: 50%;
  }

  .bot-header {
    display: flex;
    justify-content: space-between;
    margin-bottom: 12px;
  }

  .bot-name {
    font-size: 13px;
    font-weight: 500;
    color: var(--text-primary);
  }

  .bot-symbol {
    font-size: 11px;
    color: var(--text-muted);
  }

  .bot-signal {
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 12px;
    margin-bottom: 12px;
    background: var(--bg-tertiary);
    border-radius: 8px;
  }

  .signal-strength {
    font-size: 24px;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
  }

  .signal-label {
    font-size: 10px;
    color: var(--text-muted);
    margin-top: 2px;
  }

  .bot-conditions {
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
    margin-bottom: 12px;
  }

  .condition-tag {
    padding: 3px 8px;
    background: var(--bg-tertiary);
    border-radius: 4px;
    font-size: 10px;
    color: var(--text-secondary);
  }

  .bot-footer {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 11px;
  }

  .last-signal {
    color: var(--text-muted);
  }

  .bot-status-text {
    color: #10b981;
  }

  .bot-idle {
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 20px;
    color: var(--text-muted);
  }

  /* Rankings */
  .rankings-tabs {
    display: flex;
    gap: 4px;
    margin-bottom: 16px;
  }

  .rank-tab {
    padding: 8px 16px;
    background: var(--bg-tertiary);
    border: none;
    border-radius: 6px;
    color: var(--text-secondary);
    font-size: 12px;
    cursor: pointer;
  }

  .rank-tab.active {
    background: var(--accent-primary);
    color: var(--bg-primary);
  }

  .rankings-table {
    background: var(--bg-secondary);
    border-radius: 10px;
    overflow: hidden;
  }

  .table-header {
    display: grid;
    grid-template-columns: 50px 1fr 120px 80px 80px;
    gap: 16px;
    padding: 12px 16px;
    background: var(--bg-tertiary);
    font-size: 11px;
    font-weight: 500;
    color: var(--text-muted);
  }

  .table-row {
    display: grid;
    grid-template-columns: 50px 1fr 120px 80px 80px;
    gap: 16px;
    padding: 12px 16px;
    border-bottom: 1px solid var(--border-subtle);
    font-size: 12px;
  }

  .table-row:last-child {
    border-bottom: none;
  }

  .rank {
    font-weight: 600;
    color: var(--text-muted);
  }

  .profit {
    font-family: 'JetBrains Mono', monospace;
    color: var(--text-primary);
  }

  .profit.positive {
    color: #10b981;
  }

  /* Correlations */
  .correlations-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 12px;
    margin-bottom: 16px;
  }

  .corr-card {
    padding: 16px;
    background: var(--bg-secondary);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
  }

  .corr-card.status-danger {
    border-color: #ef4444;
    background: rgba(239, 68, 68, 0.1);
  }

  .corr-card.status-warning {
    border-color: #f59e0b;
  }

  .corr-pair {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 8px;
    font-size: 13px;
    color: var(--text-primary);
  }

  .corr-value {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .corr-value .value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 14px;
    font-weight: 600;
  }

  .corr-value .label {
    font-size: 10px;
    text-transform: uppercase;
    color: var(--text-muted);
  }

  .correlation-info {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 12px;
    background: var(--bg-tertiary);
    border-radius: 8px;
    font-size: 11px;
    color: var(--text-muted);
  }

  /* Settings */
  .setting-group {
    background: var(--bg-secondary);
    border: 1px solid var(--border-subtle);
    border-radius: 10px;
    padding: 16px;
    margin-bottom: 16px;
  }

  .setting-group h3 {
    margin: 0 0 12px;
    font-size: 13px;
    color: var(--text-primary);
  }

  .setting-options {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .radio-option {
    display: flex;
    flex-direction: column;
    gap: 4px;
    padding: 12px;
    background: var(--bg-tertiary);
    border-radius: 8px;
    cursor: pointer;
  }

  .radio-option span {
    font-size: 13px;
    color: var(--text-primary);
  }

  .radio-option small {
    font-size: 11px;
    color: var(--text-muted);
  }

  .setting-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 0;
    font-size: 13px;
  }

  .setting-row span:first-child {
    color: var(--text-muted);
  }

  .setting-row input[type="number"] {
    width: 100px;
    padding: 6px 10px;
    background: var(--bg-tertiary);
    border: 1px solid var(--border-subtle);
    border-radius: 6px;
    color: var(--text-primary);
    font-size: 13px;
  }

  /* Kelly Criterion */
  .kelly-section {
    display: flex;
    flex-direction: column;
    gap: 24px;
  }

  .kelly-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    padding: 20px;
    background: var(--bg-secondary);
    border: 1px solid var(--border-subtle);
    border-radius: 10px;
  }

  .kelly-info {
    display: flex;
    align-items: flex-start;
    gap: 12px;
  }

  .kelly-info > :global(svg) {
    color: var(--accent-primary);
    margin-top: 2px;
  }

  .kelly-info h3 {
    margin: 0;
    font-size: 16px;
    color: var(--text-primary);
  }

  .kelly-info p {
    margin: 4px 0 0;
    font-size: 12px;
    color: var(--text-muted);
  }

  .kelly-summary {
    display: flex;
    gap: 24px;
  }

  .summary-item {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 4px;
  }

  .summary-item .label {
    font-size: 10px;
    text-transform: uppercase;
    color: var(--text-muted);
  }

  .summary-item .value {
    font-size: 18px;
    font-weight: 600;
    color: var(--text-primary);
  }

  .summary-item .value.success {
    color: #10b981;
  }

  /* Kelly Rankings */
  .kelly-rankings {
    background: var(--bg-secondary);
    border: 1px solid var(--border-subtle);
    border-radius: 10px;
    overflow: hidden;
  }

  .kelly-rankings h4 {
    margin: 0;
    padding: 16px;
    font-size: 14px;
    color: var(--text-primary);
    border-bottom: 1px solid var(--border-subtle);
  }

  .kelly-table {
    display: flex;
    flex-direction: column;
  }

  .kelly-header-row {
    display: grid;
    grid-template-columns: 50px 1fr 100px 100px 80px 80px 80px;
    gap: 16px;
    padding: 12px 16px;
    background: var(--bg-tertiary);
    font-size: 11px;
    text-transform: uppercase;
    color: var(--text-muted);
  }

  .kelly-row {
    display: grid;
    grid-template-columns: 50px 1fr 100px 100px 80px 80px 80px;
    gap: 16px;
    padding: 14px 16px;
    border-bottom: 1px solid var(--border-subtle);
    font-size: 12px;
    align-items: center;
  }

  .kelly-row:last-child {
    border-bottom: none;
  }

  .kelly-rank {
    font-weight: 600;
    color: var(--accent-primary);
  }

  .kelly-value {
    position: relative;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
  }

  .kelly-bar {
    position: absolute;
    top: 50%;
    left: 0;
    height: 6px;
    background: var(--accent-primary);
    border-radius: 3px;
    transform: translateY(-50%);
    opacity: 0.3;
  }

  .kelly-bar.half {
    background: #10b981;
  }

  .expected-value {
    font-family: 'JetBrains Mono', monospace;
    color: var(--text-primary);
  }

  .expected-value.positive {
    color: #10b981;
  }

  .kelly-score {
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
    color: var(--text-muted);
  }

  .kelly-score.top {
    color: #f59e0b;
    font-size: 14px;
  }

  /* Kelly Details Grid */
  .kelly-details-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 16px;
  }

  .kelly-card {
    background: var(--bg-secondary);
    border: 1px solid var(--border-subtle);
    border-radius: 10px;
    overflow: hidden;
  }

  .kelly-card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 16px;
    background: var(--bg-tertiary);
    border-bottom: 1px solid var(--border-subtle);
  }

  .kelly-card-header .bot-name {
    font-size: 13px;
    font-weight: 500;
    color: var(--text-primary);
  }

  .status-badge {
    padding: 4px 8px;
    border-radius: 10px;
    font-size: 10px;
    font-weight: 500;
    text-transform: uppercase;
  }

  .status-badge.optimal {
    background: rgba(16, 185, 129, 0.2);
    color: #10b981;
  }

  .status-badge.caution {
    background: rgba(245, 158, 11, 0.2);
    color: #f59e0b;
  }

  .status-badge.warning {
    background: rgba(239, 68, 68, 0.2);
    color: #ef4444;
  }

  .kelly-card-body {
    padding: 16px;
  }

  .metric-row {
    display: flex;
    justify-content: space-between;
    padding: 6px 0;
    font-size: 12px;
  }

  .metric-label {
    color: var(--text-muted);
  }

  .metric-value {
    font-family: 'JetBrains Mono', monospace;
    color: var(--text-primary);
  }

  .metric-value.success {
    color: #10b981;
  }

  .metric-value.danger {
    color: #ef4444;
  }

  .kelly-visual {
    margin: 16px 0;
  }

  .kelly-gauge {
    display: flex;
    flex-direction: column;
    gap: 6px;
  }

  .gauge-track {
    position: relative;
    height: 12px;
    background: var(--bg-tertiary);
    border-radius: 6px;
    overflow: visible;
  }

  .gauge-fill {
    height: 100%;
    background: linear-gradient(90deg, #f59e0b, #ef4444);
    border-radius: 6px;
  }

  .gauge-half {
    position: absolute;
    top: -4px;
    width: 2px;
    height: 20px;
    background: #10b981;
    transform: translateX(-50%);
  }

  .gauge-labels {
    display: flex;
    justify-content: space-between;
    font-size: 10px;
    color: var(--text-muted);
  }

  .half-mark {
    color: #10b981;
    font-weight: 500;
  }

  .suggested-fraction {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 10px;
    background: var(--bg-tertiary);
    border-radius: 6px;
    font-size: 12px;
  }

  .suggested-fraction .label {
    color: var(--text-muted);
  }

  .suggested-fraction .value {
    font-weight: 600;
    color: var(--accent-primary);
  }

  /* Kelly History */
  .kelly-history {
    background: var(--bg-secondary);
    border: 1px solid var(--border-subtle);
    border-radius: 10px;
  }

  .kelly-history h4 {
    margin: 0;
    padding: 16px;
    font-size: 14px;
    color: var(--text-primary);
    border-bottom: 1px solid var(--border-subtle);
  }

  .history-list {
    display: flex;
    flex-direction: column;
  }

  .history-item {
    display: grid;
    grid-template-columns: 80px 1fr 100px 100px;
    gap: 16px;
    padding: 12px 16px;
    border-bottom: 1px solid var(--border-subtle);
    font-size: 12px;
  }

  .history-item:last-child {
    border-bottom: none;
  }

  .history-date {
    color: var(--text-muted);
  }

  .history-bot {
    color: var(--text-primary);
  }

  .history-fraction {
    font-family: 'JetBrains Mono', monospace;
    color: var(--accent-primary);
  }

  .history-result {
    font-family: 'JetBrains Mono', monospace;
    font-weight: 500;
  }

  .history-result.positive {
    color: #10b981;
  }

  .history-result.negative {
    color: #ef4444;
  }

  /* MT5 Connection */
  .mt5-connection {
    border-color: var(--accent-primary);
  }

  .setting-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 16px;
  }

  .setting-header h3 {
    margin: 0;
    flex: 1;
  }

  .connection-status {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 4px 10px;
    border-radius: 12px;
    font-size: 11px;
    background: rgba(239, 68, 68, 0.2);
    color: #ef4444;
  }

  .connection-status.connected {
    background: rgba(16, 185, 129, 0.2);
    color: #10b981;
  }

  .status-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: currentColor;
  }

  .mt5-form {
    display: flex;
    flex-direction: column;
    gap: 12px;
    margin-bottom: 16px;
  }

  .form-row {
    display: flex;
    flex-direction: column;
    gap: 6px;
  }

  .form-row label {
    font-size: 12px;
    color: var(--text-muted);
  }

  .form-row input,
  .form-row textarea {
    padding: 8px 12px;
    background: var(--bg-tertiary);
    border: 1px solid var(--border-subtle);
    border-radius: 6px;
    color: var(--text-primary);
    font-size: 13px;
  }

  .form-row input:focus,
  .form-row textarea:focus {
    outline: none;
    border-color: var(--accent-primary);
  }

  .form-row textarea {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    resize: vertical;
  }

  .mt5-actions {
    display: flex;
    gap: 12px;
    margin-top: 16px;
  }

  .mt5-error {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 10px 12px;
    margin-top: 12px;
    background: rgba(239, 68, 68, 0.1);
    border: 1px solid rgba(239, 68, 68, 0.3);
    border-radius: 6px;
    font-size: 12px;
    color: #ef4444;
  }
</style>
