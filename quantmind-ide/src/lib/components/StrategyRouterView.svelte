<script lang="ts">
  import { createEventDispatcher, onMount, onDestroy } from 'svelte';
  import {
    Server, Zap, Activity, TrendingUp, AlertTriangle, Shield,
    Clock, DollarSign, BarChart3, Layers, Eye, EyeOff,
    RefreshCw, Play, Pause, SkipForward, FastForward,
    ChevronRight, ChevronDown, X, Check, AlertCircle,
    ArrowUpDown, Target, Award, Trophy, Gauge, MonitorPlay,
    Globe, Currency, Package, Settings as SettingsIcon, Scale,
    Calculator, TrendingUp as TrendingUpIcon, Brain
  } from 'lucide-svelte';

  // Import new components
  import RouterHeader from './RouterHeader.svelte';
  import MarketOverview from './MarketOverview.svelte';
  import AuctionQueue from './AuctionQueue.svelte';
  import RankingsTab from './RankingsTab.svelte';
  import KellyCriterionTab from './KellyCriterionTab.svelte';
  import CorrelationsTab from './CorrelationsTab.svelte';
  import SettingsTab from './SettingsTab.svelte';
  import HmmTrainingStatus from './HmmTrainingStatus.svelte';

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

  // HMM Training State
  let hmmTraining = {
    isTraining: false,
    jobId: '',
    progress: 0,
    message: '',
    modelType: 'universal' as 'universal' | 'per_symbol' | 'per_symbol_timeframe',
    lastJob: null as { jobId: string; status: string; message: string } | null
  };

  // HMM Training config
  let hmmConfig = {
    modelType: 'universal',
    symbol: '',
    timeframe: 'H1',
    nStates: 4,
    forceRetrain: false
  };

  async function startHMMTraining() {
    hmmTraining.isTraining = true;
    hmmTraining.progress = 0;
    hmmTraining.message = 'Starting training...';
    hmmTraining.jobId = '';

    try {
      const res = await fetch('http://localhost:8000/api/hmm/train', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model_type: hmmConfig.modelType,
          symbol: hmmConfig.symbol || null,
          timeframe: hmmConfig.timeframe || null,
          n_states: hmmConfig.nStates,
          force_retrain: hmmConfig.forceRetrain
        })
      });

      if (res.ok) {
        const data = await res.json();
        hmmTraining.jobId = data.job_id;
        hmmTraining.message = data.message;

        // Poll for status
        pollTrainingStatus(data.job_id);
      } else {
        const error = await res.json();
        hmmTraining.message = `Error: ${error.detail || 'Failed to start training'}`;
        hmmTraining.isTraining = false;
      }
    } catch (e) {
      hmmTraining.message = `Error: ${e instanceof Error ? e.message : 'Unknown error'}`;
      hmmTraining.isTraining = false;
    }
  }

  async function pollTrainingStatus(jobId: string) {
    const pollInterval = setInterval(async () => {
      try {
        const res = await fetch(`http://localhost:8000/api/hmm/train/${jobId}/status`);
        if (res.ok) {
          const status = await res.json();
          hmmTraining.progress = status.progress;
          hmmTraining.message = status.message;

          if (status.status === 'completed') {
            hmmTraining.isTraining = false;
            hmmTraining.lastJob = { jobId, status: 'completed', message: 'Training completed successfully' };
            clearInterval(pollInterval);
          } else if (status.status === 'failed') {
            hmmTraining.isTraining = false;
            hmmTraining.lastJob = { jobId, status: 'failed', message: status.message };
            clearInterval(pollInterval);
          }
        }
      } catch (e) {
        console.error('Failed to poll training status:', e);
      }
    }, 2000);
  }

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
  <RouterHeader
    {routerState}
    {hmmTraining}
    {autoRefresh}
    {toggleRouter}
    {loadRouterState}
    {runAuction}
    {startHMMTraining}
  />

  <!-- HMM Training Status -->
  <HmmTrainingStatus {hmmTraining} />

  <!-- Market Overview -->
  <MarketOverview {marketState} {houseMoney} />

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
      <AuctionQueue
        {auctionQueue}
        {bots}
        {getScoreColor}
        {getStatusColor}
        {timeAgo}
      />

    {:else if activeTab === 'rankings'}
      <RankingsTab {rankings} />

    {:else if activeTab === 'kelly'}
      <KellyCriterionTab
        {kellyData}
        {bots}
        {kellyRankings}
        {kellyHistory}
      />

    {:else if activeTab === 'correlations'}
      <CorrelationsTab
        {correlations}
        {getScoreColor}
      />

    {:else if activeTab === 'settings'}
      <SettingsTab
        {routerState}
        {mt5Connected}
        {mt5Testing}
        {mt5Error}
        {mt5Config}
        {testMt5Connection}
        {saveMt5Config}
      />
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

  .hmm-training-status {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 0 24px 16px;
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .hmm-training-status.training {
    border-color: #7c3aed;
    background: #1e1b2e;
  }

  .training-info {
    display: flex;
    align-items: center;
    gap: 8px;
    color: #e2e8f0;
  }

  .training-label {
    font-weight: 500;
    color: #94a3b8;
  }

  .training-message {
    flex: 1;
  }

  .training-progress-text {
    font-weight: 600;
    color: #a78bfa;
  }

  .training-bar {
    height: 4px;
    background: #334155;
    border-radius: 2px;
    overflow: hidden;
  }

  .training-bar-fill {
    height: 100%;
    background: linear-gradient(90deg, #7c3aed, #a78bfa);
    transition: width 0.3s ease;
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
