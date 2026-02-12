<script lang="ts">
  import { createEventDispatcher, onMount } from 'svelte';
  import {
    BarChart3, TrendingUp, TrendingDown, Activity, Target, Award,
    Download, RefreshCw, Play, Pause, SkipForward, CheckCircle, XCircle,
    AlertTriangle, ChevronRight, ChevronDown, ChevronUp, Layers, Filter,
    Calendar, Clock, DollarSign, Eye, EyeOff, X, Settings as SettingsIcon, FileText,
    Globe, Zap, Shield, Gauge, PieChart, LineChart, FastForward,
    Search, ChevronLeft, ChevronRight as ChevronRightIcon, TrendingUp as TrendingUpIcon
  } from 'lucide-svelte';

  const dispatch = createEventDispatcher();

  // TypeScript interfaces for backtest results
  interface BacktestResult {
    test_type: 'historical' | 'walk_forward' | 'monte_carlo';
    mode: 'MODE_A' | 'MODE_B' | 'MODE_C';
    version: 'vanilla' | 'spiced';
    sharpe_ratio: number;
    max_drawdown: number;
    total_trades: number;
    win_rate: number;
    profit_factor: number;
    status: 'PASS' | 'FAIL';
    confidence?: number;
    num_runs?: number;
    num_windows?: number;
    trades: Array<{
      entry_time: string;
      exit_time: string;
      symbol: string;
      pnl: number;
      why: string;
    }>;
    equity_curve?: Array<{time: string, value: number}>;
    metadata?: {
      startDate: string;
      endDate: string;
      initialCapital: number;
      finalCapital: number;
      totalReturn: number;
    };
  }

  // Backtest result types
  export let results: BacktestResult[] = [];
  export let loading = false;
  export let selectedMode: 'MODE_A' | 'MODE_B' | 'MODE_C' = 'MODE_C';
  export let selectedVersion: 'vanilla' | 'spiced' = 'spiced';

  // View state
  let activeTab: 'historical' | 'walk_forward' | 'monte_carlo' = 'historical';
  let expandedTrades: Set<string> = new Set();
  let showModeComparison = false;
  let showCharts = true;
  let selectedTrade: any = null;
  let detailModalOpen = false;

  // Filter state
  let filters = {
    symbol: 'all',
    status: 'all',
    minProfit: null as number | null,
    maxProfit: null as number | null,
    strategy: 'all' as string
  };

  // Available strategies for single strategy testing
  let availableStrategies = [
    { id: 'all', name: 'All Strategies' },
    { id: 'ict-scalper', name: 'ICT Scalper' },
    { id: 'smc-reversal', name: 'SMC Reversal' },
    { id: 'breakthrough-hunter', name: 'Breakthrough Hunter' },
    { id: 'momentum-trend', name: 'Momentum Trend' }
  ];

  // Forward Test State
  let forwardTestRunning = false;
  let forwardTestProgress = 0;
  let forwardTestResults: any = null;

  // Monte Carlo Settings
  let monteCarloSettings = {
    numRuns: 1000,
    confidenceLevel: 0.95,
    bootstrap: true
  };

  // Aggregated statistics
  let aggregateStats = {
    totalTrades: 0,
    totalProfit: 0,
    avgSharpe: 0,
    maxDrawdown: 0,
    passRate: 0
  };

  onMount(() => {
    if (results.length === 0) {
      loadMockResults();
    }
    calculateAggregateStats();
  });

  async function loadBacktestResults() {
    loading = true;
    try {
      const res = await fetch('http://localhost:8000/api/backtesting/results');
      if (res.ok) {
        results = await res.json();
        calculateAggregateStats();
      }
    } catch (e) {
      console.error('Failed to load backtest results:', e);
    } finally {
      loading = false;
    }
  }

  function calculateAggregateStats() {
    const filtered = results.filter(r =>
      r.mode === selectedMode && r.version === selectedVersion
    );

    if (filtered.length === 0) {
      aggregateStats = {
        totalTrades: 0,
        totalProfit: 0,
        avgSharpe: 0,
        maxDrawdown: 0,
        passRate: 0
      };
      return;
    }

    aggregateStats = {
      totalTrades: filtered.reduce((sum, r) => sum + r.total_trades, 0),
      totalProfit: filtered.reduce((sum, r) => sum + (r.profit_factor * 100), 0),
      avgSharpe: filtered.reduce((sum, r) => sum + r.sharpe_ratio, 0) / filtered.length,
      maxDrawdown: Math.max(...filtered.map(r => r.max_drawdown)),
      passRate: (filtered.filter(r => r.status === 'PASS').length / filtered.length) * 100
    };
  }

  function getTestTypeResults(testType: 'historical' | 'walk_forward' | 'monte_carlo'): BacktestResult[] {
    return results.filter(r =>
      r.test_type === testType &&
      r.mode === selectedMode &&
      r.version === selectedVersion
    );
  }

  function toggleTradeExpanded(tradeId: string) {
    if (expandedTrades.has(tradeId)) {
      expandedTrades.delete(tradeId);
    } else {
      expandedTrades.add(tradeId);
    }
    expandedTrades = expandedTrades;
  }

  function selectTrade(trade: any) {
    selectedTrade = trade;
    detailModalOpen = true;
  }

  function getTestTypeIcon(type: string) {
    switch (type) {
      case 'historical': return LineChart;
      case 'walk_forward': return Scatter;
      case 'monte_carlo': return PieChart;
      default: return BarChart3;
    }
  }

  function getTestTypeLabel(type: string): string {
    switch (type) {
      case 'historical': return 'Historical Backtest';
      case 'walk_forward': return 'Walk-Forward Testing';
      case 'monte_carlo': return 'Monte Carlo Simulation';
      default: return type;
    }
  }

  function getStatusColor(status: string): string {
    return status === 'PASS' ? '#10b981' : '#ef4444';
  }

  function getStatusBg(status: string): string {
    return status === 'PASS' ? 'rgba(16, 185, 129, 0.2)' : 'rgba(239, 68, 68, 0.2)';
  }

  function getModeLabel(mode: string): string {
    switch (mode) {
      case 'MODE_A': return 'EA Only';
      case 'MODE_B': return 'EA + Kelly';
      case 'MODE_C': return 'Full System';
      default: return mode;
    }
  }

  function formatCurrency(value: number): string {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2
    }).format(value);
  }

  function formatPercent(value: number): string {
    return `${(value * 100).toFixed(2)}%`;
  }

  function formatDateTime(dateStr: string): string {
    const date = new Date(dateStr);
    return date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], {
      hour: '2-digit',
      minute: '2-digit'
    });
  }

  function exportResults(format: 'json' | 'csv' | 'pine') {
    const currentResults = getTestTypeResults(activeTab);

    if (format === 'json') {
      const blob = new Blob([JSON.stringify(currentResults, null, 2)], {
        type: 'application/json'
      });
      downloadBlob(blob, `backtest-${activeTab}-${Date.now()}.json`);
    } else if (format === 'csv') {
      const csv = convertToCSV(currentResults);
      const blob = new Blob([csv], { type: 'text/csv' });
      downloadBlob(blob, `backtest-${activeTab}-${Date.now()}.csv`);
    } else if (format === 'pine') {
      const pine = convertToPineScript(currentResults[0]);
      const blob = new Blob([pine], { type: 'text/plain' });
      downloadBlob(blob, `backtest-${activeTab}-${Date.now()}.pine`);
    }
  }

  function downloadBlob(blob: Blob, filename: string) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  }

  function convertToCSV(results: BacktestResult[]): string {
    const headers = ['Test Type', 'Mode', 'Version', 'Sharpe Ratio', 'Max Drawdown',
      'Total Trades', 'Win Rate', 'Profit Factor', 'Status'];
    const rows = results.map(r => [
      r.test_type,
      r.mode,
      r.version,
      r.sharpe_ratio.toFixed(2),
      r.max_drawdown.toFixed(2),
      r.total_trades,
      r.win_rate.toFixed(2),
      r.profit_factor.toFixed(2),
      r.status
    ]);
    return [headers, ...rows].map(row => row.join(',')).join('\n');
  }

  function convertToPineScript(result: BacktestResult): string {
    return `//@version=5
strategy("QuantMind Backtest ${result.test_type}", overlay=true)

// Parameters
useKelly = ${result.mode === 'MODE_B' || result.mode === 'MODE_C'}
useRegimeFilter = ${result.mode === 'MODE_C'}

// Entry conditions (placeholder - implement based on actual strategy)
longCondition = ta.crossover(ta.sma(close, 14), ta.sma(close, 28))
shortCondition = ta.crossunder(ta.sma(close, 14), ta.sma(close, 28))

// Position sizing with Kelly
kellyFraction = ${result.profit_factor > 0 ? (1 / result.profit_factor).toFixed(3) : '0.02'}
positionSize = strategy.position_size * (useKelly ? kellyFraction : 1)

// Entry
if (longCondition and strategy.position_size == 0)
    strategy.entry("Long", strategy.long, qty=positionSize)
if (shortCondition and strategy.position_size == 0)
    strategy.entry("Short", strategy.short, qty=positionSize)

// Exit
strategy.exit("Exit", "Long", stop=strategy.position_avg_price * 0.98, limit=strategy.position_avg_price * 1.02)
strategy.exit("Exit", "Short", stop=strategy.position_avg_price * 1.02, limit=strategy.position_avg_price * 0.98)

// Results: ${result.status}
// Sharpe: ${result.sharpe_ratio.toFixed(2)}
// Max DD: ${formatPercent(result.max_drawdown)}
`;
  }

  function runNewBacktest() {
    dispatch('runBacktest', { type: activeTab, mode: selectedMode, version: selectedVersion, strategy: filters.strategy });
  }

  // Single Strategy Testing
  function runSingleStrategyTest(strategyId: string) {
    if (strategyId === 'all') {
      runNewBacktest();
      return;
    }
    dispatch('runBacktest', { 
      type: activeTab, 
      mode: selectedMode, 
      version: selectedVersion, 
      strategy: strategyId 
    });
  }

  // Forward Test
  async function runForwardTest() {
    forwardTestRunning = true;
    forwardTestProgress = 0;
    
    // Simulate forward test progress
    const interval = setInterval(() => {
      forwardTestProgress += 5;
      if (forwardTestProgress >= 100) {
        clearInterval(interval);
        forwardTestRunning = false;
        forwardTestResults = {
          sharpeRatio: 1.92,
          maxDrawdown: 0.12,
          winRate: 0.62,
          profitFactor: 1.85,
          totalTrades: 45
        };
      }
    }, 200);
  }

  // Monte Carlo Simulation
  async function runMonteCarlo() {
    loading = true;
    try {
      const res = await fetch('http://localhost:8000/api/backtesting/monte-carlo', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          strategyId: filters.strategy,
          ...monteCarloSettings
        })
      });
      if (res.ok) {
        const data = await res.json();
        results = [...results, data];
        calculateAggregateStats();
        activeTab = 'monte_carlo';
      }
    } catch (e) {
      console.error('Monte Carlo simulation failed:', e);
      // Load mock data on error
      loadMockResults();
    } finally {
      loading = false;
    }
  }

  // Results Comparison
  function compareResults() {
    showModeComparison = true;
  }

  // Mock data for development
  function loadMockResults() {
    results = [
      {
        test_type: 'historical',
        mode: 'MODE_A',
        version: 'vanilla',
        sharpe_ratio: 1.85,
        max_drawdown: 0.15,
        total_trades: 156,
        win_rate: 0.58,
        profit_factor: 1.75,
        status: 'PASS',
        trades: [
          {
            entry_time: new Date(Date.now() - 86400000 * 5).toISOString(),
            exit_time: new Date(Date.now() - 86400000 * 5 + 3600000).toISOString(),
            symbol: 'EURUSD',
            pnl: 125.50,
            why: 'FVG setup at key support level with ATR confirmation'
          },
          {
            entry_time: new Date(Date.now() - 86400000 * 4).toISOString(),
            exit_time: new Date(Date.now() - 86400000 * 4 + 7200000).toISOString(),
            symbol: 'GBPUSD',
            pnl: -45.20,
            why: 'Order block failure - volatility spiked during news'
          },
          {
            entry_time: new Date(Date.now() - 86400000 * 3).toISOString(),
            exit_time: new Date(Date.now() - 86400000 * 3 + 5400000).toISOString(),
            symbol: 'USDJPY',
            pnl: 89.30,
            why: 'Premium discount zone with RSI divergence confirmation'
          }
        ],
        metadata: {
          startDate: '2024-01-01',
          endDate: '2024-12-31',
          initialCapital: 10000,
          finalCapital: 12750,
          totalReturn: 0.275
        }
      },
      {
        test_type: 'historical',
        mode: 'MODE_B',
        version: 'vanilla',
        sharpe_ratio: 2.15,
        max_drawdown: 0.12,
        total_trades: 142,
        win_rate: 0.61,
        profit_factor: 1.95,
        status: 'PASS',
        trades: [],
        metadata: {
          startDate: '2024-01-01',
          endDate: '2024-12-31',
          initialCapital: 10000,
          finalCapital: 13420,
          totalReturn: 0.342
        }
      },
      {
        test_type: 'historical',
        mode: 'MODE_C',
        version: 'spiced',
        sharpe_ratio: 2.68,
        max_drawdown: 0.09,
        total_trades: 98,
        win_rate: 0.67,
        profit_factor: 2.35,
        status: 'PASS',
        trades: [
          {
            entry_time: new Date(Date.now() - 86400000 * 2).toISOString(),
            exit_time: new Date(Date.now() - 86400000 * 2 + 4800000).toISOString(),
            symbol: 'EURUSD',
            pnl: 178.90,
            why: 'Perfect regime alignment (quality 0.85) + Kelly optimal sizing + House Money aggressive mode'
          }
        ],
        metadata: {
          startDate: '2024-01-01',
          endDate: '2024-12-31',
          initialCapital: 10000,
          finalCapital: 15280,
          totalReturn: 0.528
        }
      },
      {
        test_type: 'walk_forward',
        mode: 'MODE_C',
        version: 'spiced',
        sharpe_ratio: 2.42,
        max_drawdown: 0.11,
        total_trades: 245,
        win_rate: 0.64,
        profit_factor: 2.15,
        status: 'PASS',
        num_windows: 6,
        trades: [],
        metadata: {
          startDate: '2024-01-01',
          endDate: '2024-12-31',
          initialCapital: 10000,
          finalCapital: 14650,
          totalReturn: 0.465
        }
      },
      {
        test_type: 'monte_carlo',
        mode: 'MODE_C',
        version: 'spiced',
        sharpe_ratio: 2.55,
        max_drawdown: 0.10,
        total_trades: 50000,
        win_rate: 0.65,
        profit_factor: 2.25,
        status: 'PASS',
        confidence: 0.95,
        num_runs: 1000,
        trades: [],
        metadata: {
          startDate: '2024-01-01',
          endDate: '2024-12-31',
          initialCapital: 10000,
          finalCapital: 14950,
          totalReturn: 0.495
        }
      }
    ];
  }

  $: currentResults = getTestTypeResults(activeTab);
  $: filteredTrades = currentResults.length > 0 ? currentResults[0].trades.filter(t => {
    if (filters.symbol !== 'all' && t.symbol !== filters.symbol) return false;
    if (filters.status === 'winning' && t.pnl <= 0) return false;
    if (filters.status === 'losing' && t.pnl >= 0) return false;
    if (filters.minProfit && t.pnl < filters.minProfit) return false;
    if (filters.maxProfit && t.pnl > filters.maxProfit) return false;
    return true;
  }) : [];
</script>

<div class="backtest-results-view">
  <!-- Header -->
  <div class="view-header">
    <div class="header-left">
      <BarChart3 size={24} class="header-icon" />
      <div>
        <h2>Backtest Results</h2>
        <p>Historical, Walk-Forward, and Monte Carlo analysis</p>
      </div>
    </div>
    <div class="header-actions">
      <!-- Strategy Selector -->
      <div class="strategy-selector">
        <label><Search size={12} /> Strategy:</label>
        <select bind:value={filters.strategy} on:change={() => calculateAggregateStats()}>
          {#each availableStrategies as strategy}
            <option value={strategy.id}>{strategy.name}</option>
          {/each}
        </select>
      </div>
      
      <div class="mode-selector">
        <label>Mode:</label>
        <select bind:value={selectedMode} on:change={calculateAggregateStats}>
          <option value="MODE_A">EA Only</option>
          <option value="MODE_B">EA + Kelly</option>
          <option value="MODE_C">Full System</option>
        </select>
      </div>
      <div class="version-selector">
        <label>Version:</label>
        <select bind:value={selectedVersion} on:change={calculateAggregateStats}>
          <option value="vanilla">Vanilla</option>
          <option value="spiced">Spiced</option>
        </select>
      </div>
      <button class="btn" on:click={() => showModeComparison = !showModeComparison} class:active={showModeComparison}>
        <Layers size={14} />
        <span>Compare</span>
      </button>
      <button class="btn" on:click={loadBacktestResults}>
        <RefreshCw size={14} />
        <span>Refresh</span>
      </button>
      <button class="btn primary" on:click={runNewBacktest}>
        <Play size={14} />
        <span>Run Backtest</span>
      </button>
      
      <!-- Forward Test & Monte Carlo -->
      <div class="test-type-actions">
        <button 
          class="btn" 
          on:click={runForwardTest}
          disabled={forwardTestRunning}
          title="Run Forward Test"
        >
          <FastForward size={14} />
          <span>Forward</span>
        </button>
        <button 
          class="btn" 
          on:click={runMonteCarlo}
          disabled={loading}
          title="Run Monte Carlo Simulation"
        >
          <PieChart size={14} />
          <span>MC</span>
        </button>
      </div>
    </div>
  </div>

  <!-- Forward Test Progress -->
  {#if forwardTestRunning}
    <div class="forward-test-progress">
      <div class="progress-header">
        <FastForward size={16} />
        <span>Running Forward Test...</span>
      </div>
      <div class="progress-bar">
        <div class="progress-fill" style="width: {forwardTestProgress}%"></div>
      </div>
      <span class="progress-text">{forwardTestProgress}%</span>
    </div>
  {/if}

  <!-- Forward Test Results -->
  {#if forwardTestResults}
    <div class="forward-test-results">
      <div class="results-header">
        <TrendingUpIcon size={16} />
        <span>Forward Test Results</span>
        <button class="close-btn" on:click={() => forwardTestResults = null}><X size={14} /></button>
      </div>
      <div class="results-grid">
        <div class="result-item">
          <span class="label">Sharpe</span>
          <span class="value">{forwardTestResults.sharpeRatio.toFixed(2)}</span>
        </div>
        <div class="result-item">
          <span class="label">Max DD</span>
          <span class="value danger">{(forwardTestResults.maxDrawdown * 100).toFixed(1)}%</span>
        </div>
        <div class="result-item">
          <span class="label">Win Rate</span>
          <span class="value">{(forwardTestResults.winRate * 100).toFixed(0)}%</span>
        </div>
        <div class="result-item">
          <span class="label">PF</span>
          <span class="value success">{forwardTestResults.profitFactor.toFixed(2)}</span>
        </div>
        <div class="result-item">
          <span class="label">Trades</span>
          <span class="value">{forwardTestResults.totalTrades}</span>
        </div>
      </div>
    </div>
  {/if}

  <!-- Aggregate Statistics -->
  <div class="aggregate-stats">
    <div class="stat-card">
      <div class="stat-icon"><Activity size={18} /></div>
      <div class="stat-info">
        <span class="stat-value">{aggregateStats.totalTrades}</span>
        <span class="stat-label">Total Trades</span>
      </div>
    </div>
    <div class="stat-card">
      <div class="stat-icon"><TrendingUp size={18} /></div>
      <div class="stat-info">
        <span class="stat-value">{aggregateStats.avgSharpe.toFixed(2)}</span>
        <span class="stat-label">Avg Sharpe</span>
      </div>
    </div>
    <div class="stat-card">
      <div class="stat-icon danger"><AlertTriangle size={18} /></div>
      <div class="stat-info">
        <span class="stat-value">{formatPercent(aggregateStats.maxDrawdown)}</span>
        <span class="stat-label">Max Drawdown</span>
      </div>
    </div>
    <div class="stat-card">
      <div class="stat-icon"><CheckCircle size={18} /></div>
      <div class="stat-info">
        <span class="stat-value">{aggregateStats.passRate.toFixed(1)}%</span>
        <span class="stat-label">Pass Rate</span>
      </div>
    </div>
  </div>

  <!-- Test Type Tabs -->
  <div class="test-tabs">
    <button
      class="tab"
      class:active={activeTab === 'historical'}
      on:click={() => activeTab = 'historical'}
    >
      <LineChart size={14} />
      <span>Historical Backtest</span>
      {#if getTestTypeResults('historical').length > 0}
        <span class="count">{getTestTypeResults('historical').length}</span>
      {/if}
    </button>
    <button
      class="tab"
      class:active={activeTab === 'walk_forward'}
      on:click={() => activeTab = 'walk_forward'}
    >
      <Target size={14} />
      <span>Walk-Forward</span>
      {#if getTestTypeResults('walk_forward').length > 0}
        <span class="count">{getTestTypeResults('walk_forward').length}</span>
      {/if}
    </button>
    <button
      class="tab"
      class:active={activeTab === 'monte_carlo'}
      on:click={() => activeTab = 'monte_carlo'}
    >
      <PieChart size={14} />
      <span>Monte Carlo</span>
      {#if getTestTypeResults('monte_carlo').length > 0}
        <span class="count">{getTestTypeResults('monte_carlo').length}</span>
      {/if}
    </button>
    <div class="tab-actions">
      <button class="icon-btn" on:click={() => showCharts = !showCharts} title="Toggle charts">
        {#if showCharts}
          <Eye size={14} />
        {:else}
          <EyeOff size={14} />
        {/if}
      </button>
    </div>
  </div>

  <!-- Content Area -->
  <div class="content-area">
    {#if loading}
      <div class="loading-state">
        <div class="spinner"></div>
        <p>Running backtest...</p>
      </div>
    {:else if currentResults.length === 0}
      <div class="empty-state">
        <svelte:component this={getTestTypeIcon(activeTab)} size={48} />
        <h3>No {getTestTypeLabel(activeTab)} Results</h3>
        <p>Run a backtest to see results here</p>
        <button class="btn primary" on:click={runNewBacktest}>
          <Play size={14} />
          Run {getTestTypeLabel(activeTab)}
        </button>
      </div>
    {:else}
      <!-- Mode Comparison View -->
      {#if showModeComparison}
        <div class="mode-comparison">
          <h3>Mode Comparison</h3>
          <div class="comparison-grid">
            {#each ['MODE_A', 'MODE_B', 'MODE_C'] as mode}
              {@const modeResult = results.find(r => r.test_type === activeTab && r.mode === mode && r.version === selectedVersion)}
              <div class="comparison-card" class:best={mode === 'MODE_C'}>
                <div class="comparison-header">
                  <h4>{getModeLabel(mode)}</h4>
                  {#if modeResult}
                    <span class="status-badge" style="background: {getStatusBg(modeResult.status)}; color: {getStatusColor(modeResult.status)}">
                      {modeResult.status}
                    </span>
                  {/if}
                </div>
                {#if modeResult}
                  <div class="comparison-metrics">
                    <div class="metric">
                      <span class="label">Sharpe Ratio</span>
                      <span class="value">{modeResult.sharpe_ratio.toFixed(2)}</span>
                    </div>
                    <div class="metric">
                      <span class="label">Max Drawdown</span>
                      <span class="value">{formatPercent(modeResult.max_drawdown)}</span>
                    </div>
                    <div class="metric">
                      <span class="label">Win Rate</span>
                      <span class="value">{formatPercent(modeResult.win_rate)}</span>
                    </div>
                    <div class="metric">
                      <span class="label">Profit Factor</span>
                      <span class="value">{modeResult.profit_factor.toFixed(2)}</span>
                    </div>
                    <div class="metric">
                      <span class="label">Total Trades</span>
                      <span class="value">{modeResult.total_trades}</span>
                    </div>
                  </div>
                {:else}
                  <p class="no-data">No data for this mode</p>
                {/if}
              </div>
            {/each}
          </div>
        </div>
      {/if}

      <!-- Main Results Display -->
      {#each currentResults as result}
        <div class="result-card">
          <!-- Result Header -->
          <div class="result-header">
            <div class="header-info">
              <div class="test-type">
                <svelte:component this={getTestTypeIcon(result.test_type)} size={20} />
                <span>{getTestTypeLabel(result.test_type)}</span>
              </div>
              <div class="mode-version">
                <span class="mode-badge">{getModeLabel(result.mode)}</span>
                <span class="version-badge">{result.version}</span>
              </div>
            </div>
            <div class="header-status">
              <span class="status-badge" style="background: {getStatusBg(result.status)}; color: {getStatusColor(result.status)}">
                {#if result.status === 'PASS'}
                  <CheckCircle size={14} />
                {:else}
                  <XCircle size={14} />
                {/if}
                {result.status}
              </span>
            </div>
          </div>

          <!-- Metrics Grid -->
          <div class="metrics-grid">
            <div class="metric-card">
              <div class="metric-icon"><TrendingUp size={16} /></div>
              <div class="metric-content">
                <span class="metric-value">{result.sharpe_ratio.toFixed(2)}</span>
                <span class="metric-label">Sharpe Ratio</span>
              </div>
            </div>
            <div class="metric-card">
              <div class="metric-icon danger"><AlertTriangle size={16} /></div>
              <div class="metric-content">
                <span class="metric-value">{formatPercent(result.max_drawdown)}</span>
                <span class="metric-label">Max Drawdown</span>
              </div>
            </div>
            <div class="metric-card">
              <div class="metric-icon"><Target size={16} /></div>
              <div class="metric-content">
                <span class="metric-value">{formatPercent(result.win_rate)}</span>
                <span class="metric-label">Win Rate</span>
              </div>
            </div>
            <div class="metric-card">
              <div class="metric-icon"><DollarSign size={16} /></div>
              <div class="metric-content">
                <span class="metric-value">{result.profit_factor.toFixed(2)}</span>
                <span class="metric-label">Profit Factor</span>
              </div>
            </div>
            <div class="metric-card">
              <div class="metric-icon"><Activity size={16} /></div>
              <div class="metric-content">
                <span class="metric-value">{result.total_trades}</span>
                <span class="metric-label">Total Trades</span>
              </div>
            </div>
            {#if result.confidence}
              <div class="metric-card">
                <div class="metric-icon"><Shield size={16} /></div>
                <div class="metric-content">
                  <span class="metric-value">{formatPercent(result.confidence)}</span>
                  <span class="metric-label">Confidence</span>
                </div>
              </div>
            {/if}
            {#if result.num_runs}
              <div class="metric-card">
                <div class="metric-icon"><Zap size={16} /></div>
                <div class="metric-content">
                  <span class="metric-value">{result.num_runs}</span>
                  <span class="metric-label">Runs</span>
                </div>
              </div>
            {/if}
            {#if result.num_windows}
              <div class="metric-card">
                <div class="metric-icon"><Layers size={16} /></div>
                <div class="metric-content">
                  <span class="metric-value">{result.num_windows}</span>
                  <span class="metric-label">Windows</span>
                </div>
              </div>
            {/if}
          </div>

          <!-- Charts Placeholder -->
          {#if showCharts}
            <div class="charts-section">
              <h4>Performance Charts</h4>
              <div class="charts-grid">
                <div class="chart-placeholder">
                  <div class="placeholder-header">
                    <LineChart size={16} />
                    <span>Equity Curve</span>
                  </div>
                  <div class="placeholder-content">
                    <p>Equity curve visualization will appear here</p>
                    <small>Shows capital growth over time</small>
                  </div>
                </div>
                {#if result.test_type === 'monte_carlo'}
                  <div class="chart-placeholder">
                    <div class="placeholder-header">
                      <PieChart size={16} />
                      <span>Distribution Histogram</span>
                    </div>
                    <div class="placeholder-content">
                      <p>Monte Carlo distribution will appear here</p>
                      <small>Shows probability distribution of returns</small>
                    </div>
                  </div>
                {:else if result.test_type === 'walk_forward'}
                  <div class="chart-placeholder">
                    <div class="placeholder-header">
                      <Target size={16} />
                      <span>Window Comparison</span>
                    </div>
                    <div class="placeholder-content">
                      <p>Walk-forward window analysis will appear here</p>
                      <small>Shows performance across test windows</small>
                    </div>
                  </div>
                {/if}
                <div class="chart-placeholder">
                  <div class="placeholder-header">
                    <BarChart3 size={16} />
                    <span>Trade Analysis</span>
                  </div>
                  <div class="placeholder-content">
                    <p>Trade entry/exit markers will appear here</p>
                    <small>Shows individual trade performance</small>
                  </div>
                </div>
              </div>
            </div>
          {/if}

          <!-- Metadata -->
          {#if result.metadata}
            <div class="metadata-section">
              <div class="metadata-item">
                <span class="label">Period:</span>
                <span class="value">{result.metadata.startDate} to {result.metadata.endDate}</span>
              </div>
              <div class="metadata-item">
                <span class="label">Initial Capital:</span>
                <span class="value">{formatCurrency(result.metadata.initialCapital)}</span>
              </div>
              <div class="metadata-item">
                <span class="label">Final Capital:</span>
                <span class="value">{formatCurrency(result.metadata.finalCapital)}</span>
              </div>
              <div class="metadata-item">
                <span class="label">Total Return:</span>
                <span class="value success">{formatPercent(result.metadata.totalReturn)}</span>
              </div>
            </div>
          {/if}

          <!-- Export Actions -->
          <div class="export-actions">
            <span class="export-label">Export:</span>
            <button class="export-btn" on:click={() => exportResults('json')}>
              <FileText size={12} />
              JSON
            </button>
            <button class="export-btn" on:click={() => exportResults('csv')}>
              <FileText size={12} />
              CSV
            </button>
            <button class="export-btn" on:click={() => exportResults('pine')}>
              <Globe size={12} />
              Pine Script
            </button>
          </div>
        </div>
      {/each}

      <!-- Trade List -->
      {#if filteredTrades.length > 0}
        <div class="trades-section">
          <div class="trades-header">
            <h3>Trade List</h3>
            <div class="trade-filters">
              <select bind:value={filters.symbol}>
                <option value="all">All Symbols</option>
                <option value="EURUSD">EURUSD</option>
                <option value="GBPUSD">GBPUSD</option>
                <option value="USDJPY">USDJPY</option>
              </select>
              <select bind:value={filters.status}>
                <option value="all">All Trades</option>
                <option value="winning">Winning</option>
                <option value="losing">Losing</option>
              </select>
            </div>
          </div>
          <div class="trades-list">
            {#each filteredTrades as trade}
              {@const tradeId = `${trade.entry_time}-${trade.symbol}`}
              <div class="trade-item" class:expanded={expandedTrades.has(tradeId)}>
                <div class="trade-summary" on:click={() => toggleTradeExpanded(tradeId)}>
                  <div class="trade-info">
                    <Clock size={14} class="trade-icon" />
                    <div class="trade-time">{formatDateTime(trade.entry_time)}</div>
                    <span class="trade-symbol">{trade.symbol}</span>
                  </div>
                  <div class="trade-pnl" class:positive={trade.pnl > 0} class:negative={trade.pnl < 0}>
                    {formatCurrency(trade.pnl)}
                  </div>
                  <button class="expand-btn">
                    {#if expandedTrades.has(tradeId)}
                      <ChevronUp size={14} />
                    {:else}
                      <ChevronDown size={14} />
                    {/if}
                  </button>
                </div>
                {#if expandedTrades.has(tradeId)}
                  <div class="trade-details">
                    <div class="detail-row">
                      <span class="detail-label">Entry Time:</span>
                      <span class="detail-value">{formatDateTime(trade.entry_time)}</span>
                    </div>
                    <div class="detail-row">
                      <span class="detail-label">Exit Time:</span>
                      <span class="detail-value">{formatDateTime(trade.exit_time)}</span>
                    </div>
                    <div class="detail-row">
                      <span class="detail-label">Symbol:</span>
                      <span class="detail-value">{trade.symbol}</span>
                    </div>
                    <div class="detail-row">
                      <span class="detail-label">P&L:</span>
                      <span class="detail-value" style="color: {trade.pnl > 0 ? '#10b981' : '#ef4444'}">
                        {formatCurrency(trade.pnl)}
                      </span>
                    </div>
                    <div class="detail-row full-width">
                      <span class="detail-label">Why?</span>
                      <span class="detail-value reason">{trade.why}</span>
                    </div>
                    <button class="view-details-btn" on:click={() => selectTrade(trade)}>
                      <Eye size={12} />
                      View Full Details
                    </button>
                  </div>
                {/if}
              </div>
            {/each}
          </div>
        </div>
      {/if}
    {/if}
  </div>

  <!-- Trade Detail Modal -->
  {#if detailModalOpen && selectedTrade}
    <div class="modal-overlay" on:click|self={() => detailModalOpen = false}>
      <div class="modal">
        <div class="modal-header">
          <h3>Trade Details</h3>
          <button class="icon-btn" on:click={() => detailModalOpen = false}>
            <X size={18} />
          </button>
        </div>
        <div class="modal-content">
          <div class="trade-detail-section">
            <h4>Trade Information</h4>
            <div class="detail-grid">
              <div class="detail-item">
                <span class="label">Symbol</span>
                <span class="value">{selectedTrade.symbol}</span>
              </div>
              <div class="detail-item">
                <span class="label">Entry Time</span>
                <span class="value">{formatDateTime(selectedTrade.entry_time)}</span>
              </div>
              <div class="detail-item">
                <span class="label">Exit Time</span>
                <span class="value">{formatDateTime(selectedTrade.exit_time)}</span>
              </div>
              <div class="detail-item">
                <span class="label">Profit/Loss</span>
                <span class="value" style="color: {selectedTrade.pnl > 0 ? '#10b981' : '#ef4444'}">
                  {formatCurrency(selectedTrade.pnl)}
                </span>
              </div>
            </div>
          </div>
          <div class="trade-detail-section">
            <h4>Why This Trade?</h4>
            <p class="trade-reason">{selectedTrade.why}</p>
          </div>
        </div>
      </div>
    </div>
  {/if}
</div>

<style>
  .backtest-results-view {
    display: flex;
    flex-direction: column;
    height: 100%;
    background: var(--bg-primary);
    overflow: hidden;
  }

  /* Header */
  .view-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 20px 24px;
    background: var(--bg-secondary);
    border-bottom: 1px solid var(--border-subtle);
    flex-wrap: wrap;
    gap: 12px;
  }

  .header-left {
    display: flex;
    align-items: center;
    gap: 12px;
  }

  .header-icon {
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
    flex-wrap: wrap;
  }

  .mode-selector,
  .version-selector {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 12px;
    color: var(--text-secondary);
  }

  .mode-selector select,
  .version-selector select {
    padding: 4px 8px;
    background: var(--bg-tertiary);
    border: 1px solid var(--border-subtle);
    border-radius: 4px;
    color: var(--text-primary);
    font-size: 12px;
  }

  /* Strategy Selector */
  .strategy-selector {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 12px;
    color: var(--text-secondary);
  }

  .strategy-selector label {
    display: flex;
    align-items: center;
    gap: 4px;
  }

  .strategy-selector select {
    padding: 4px 8px;
    background: var(--bg-tertiary);
    border: 1px solid var(--border-subtle);
    border-radius: 4px;
    color: var(--text-primary);
    font-size: 12px;
    min-width: 150px;
  }

  /* Test Type Actions */
  .test-type-actions {
    display: flex;
    gap: 8px;
  }

  .test-type-actions .btn {
    padding: 6px 10px;
  }

  /* Forward Test Progress */
  .forward-test-progress {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 12px 24px;
    background: var(--accent-primary);
    color: var(--bg-primary);
  }

  .progress-header {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 13px;
    font-weight: 500;
  }

  .progress-bar {
    flex: 1;
    height: 8px;
    background: rgba(0, 0, 0, 0.2);
    border-radius: 4px;
    overflow: hidden;
  }

  .progress-fill {
    height: 100%;
    background: var(--bg-primary);
    transition: width 0.3s;
  }

  .progress-text {
    font-size: 12px;
    font-weight: 600;
    min-width: 40px;
  }

  /* Forward Test Results */
  .forward-test-results {
    margin: 16px 24px;
    background: var(--bg-secondary);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
    overflow: hidden;
  }

  .results-header {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 12px 16px;
    background: var(--bg-tertiary);
    border-bottom: 1px solid var(--border-subtle);
    font-size: 13px;
    font-weight: 500;
    color: var(--text-primary);
  }

  .results-header .close-btn {
    margin-left: auto;
    background: transparent;
    border: none;
    color: var(--text-muted);
    cursor: pointer;
    padding: 4px;
  }

  .results-header .close-btn:hover {
    color: var(--text-primary);
  }

  .results-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 1px;
    background: var(--border-subtle);
  }

  .result-item {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 4px;
    padding: 16px;
    background: var(--bg-secondary);
  }

  .result-item .label {
    font-size: 10px;
    text-transform: uppercase;
    color: var(--text-muted);
  }

  .result-item .value {
    font-size: 16px;
    font-weight: 600;
    color: var(--text-primary);
  }

  .result-item .value.success {
    color: #10b981;
  }

  .result-item .value.danger {
    color: #ef4444;
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

  .icon-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 28px;
    height: 28px;
    background: transparent;
    border: none;
    border-radius: 4px;
    color: var(--text-muted);
    cursor: pointer;
  }

  .icon-btn:hover {
    background: var(--bg-tertiary);
    color: var(--text-primary);
  }

  /* Aggregate Stats */
  .aggregate-stats {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 12px;
    padding: 16px 24px;
    background: var(--bg-secondary);
    border-bottom: 1px solid var(--border-subtle);
  }

  .stat-card {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 12px;
    background: var(--bg-tertiary);
    border-radius: 8px;
  }

  .stat-icon {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 36px;
    height: 36px;
    background: var(--bg-surface);
    border-radius: 8px;
    color: var(--text-muted);
  }

  .stat-icon.danger {
    background: rgba(239, 68, 68, 0.2);
    color: #ef4444;
  }

  .stat-info {
    display: flex;
    flex-direction: column;
  }

  .stat-value {
    font-size: 16px;
    font-weight: 600;
    color: var(--text-primary);
  }

  .stat-label {
    font-size: 11px;
    color: var(--text-muted);
  }

  /* Test Tabs */
  .test-tabs {
    display: flex;
    gap: 4px;
    padding: 12px 24px;
    background: var(--bg-secondary);
    border-bottom: 1px solid var(--border-subtle);
  }

  .tab {
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
    transition: all 0.15s;
  }

  .tab:hover {
    background: var(--bg-tertiary);
    color: var(--text-primary);
  }

  .tab.active {
    background: var(--accent-primary);
    color: var(--bg-primary);
  }

  .tab .count {
    padding: 2px 6px;
    background: rgba(0, 0, 0, 0.2);
    border-radius: 10px;
    font-size: 10px;
  }

  .tab-actions {
    margin-left: auto;
  }

  /* Content Area */
  .content-area {
    flex: 1;
    overflow-y: auto;
    padding: 24px;
  }

  /* Loading State */
  .loading-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 60px 24px;
    color: var(--text-muted);
  }

  .spinner {
    width: 40px;
    height: 40px;
    border: 3px solid var(--bg-tertiary);
    border-top-color: var(--accent-primary);
    border-radius: 50%;
    animation: spin 1s linear infinite;
  }

  @keyframes spin {
    to { transform: rotate(360deg); }
  }

  .loading-state p {
    margin-top: 16px;
    font-size: 13px;
  }

  /* Empty State */
  .empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 60px 24px;
    color: var(--text-muted);
    text-align: center;
  }

  .empty-state h3 {
    margin: 16px 0 8px;
    font-size: 16px;
    color: var(--text-primary);
  }

  .empty-state p {
    margin-bottom: 20px;
    font-size: 13px;
  }

  /* Mode Comparison */
  .mode-comparison {
    background: var(--bg-secondary);
    border-radius: 10px;
    padding: 20px;
    margin-bottom: 24px;
  }

  .mode-comparison h3 {
    margin: 0 0 16px;
    font-size: 14px;
    color: var(--text-primary);
  }

  .comparison-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 12px;
  }

  .comparison-card {
    background: var(--bg-tertiary);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
    padding: 16px;
    transition: all 0.15s;
  }

  .comparison-card.best {
    border-color: var(--accent-primary);
    background: rgba(14, 165, 233, 0.1);
  }

  .comparison-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
  }

  .comparison-header h4 {
    margin: 0;
    font-size: 13px;
    color: var(--text-primary);
  }

  .status-badge {
    display: flex;
    align-items: center;
    gap: 4px;
    padding: 4px 8px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 500;
  }

  .comparison-metrics {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .comparison-metrics .metric {
    display: flex;
    justify-content: space-between;
    font-size: 11px;
  }

  .comparison-metrics .label {
    color: var(--text-muted);
  }

  .comparison-metrics .value {
    color: var(--text-primary);
    font-weight: 500;
  }

  .no-data {
    color: var(--text-muted);
    font-size: 12px;
    text-align: center;
    padding: 20px;
  }

  /* Result Cards */
  .result-card {
    background: var(--bg-secondary);
    border: 1px solid var(--border-subtle);
    border-radius: 10px;
    padding: 20px;
    margin-bottom: 16px;
  }

  .result-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 16px;
  }

  .header-info {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .test-type {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 14px;
    font-weight: 500;
    color: var(--text-primary);
  }

  .mode-version {
    display: flex;
    gap: 8px;
  }

  .mode-badge,
  .version-badge {
    padding: 4px 8px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 500;
  }

  .mode-badge {
    background: var(--accent-primary);
    color: var(--bg-primary);
  }

  .version-badge {
    background: var(--bg-tertiary);
    color: var(--text-secondary);
  }

  /* Metrics Grid */
  .metrics-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
    gap: 12px;
    margin-bottom: 16px;
  }

  .metric-card {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 12px;
    background: var(--bg-tertiary);
    border-radius: 8px;
  }

  .metric-icon {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 32px;
    height: 32px;
    background: var(--bg-surface);
    border-radius: 6px;
    color: var(--text-muted);
  }

  .metric-icon.danger {
    background: rgba(239, 68, 68, 0.2);
    color: #ef4444;
  }

  .metric-content {
    display: flex;
    flex-direction: column;
  }

  .metric-value {
    font-size: 14px;
    font-weight: 600;
    color: var(--text-primary);
  }

  .metric-label {
    font-size: 10px;
    color: var(--text-muted);
  }

  /* Charts Section */
  .charts-section {
    margin-bottom: 16px;
  }

  .charts-section h4 {
    margin: 0 0 12px;
    font-size: 13px;
    color: var(--text-primary);
  }

  .charts-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 12px;
  }

  .chart-placeholder {
    background: var(--bg-tertiary);
    border: 1px dashed var(--border-subtle);
    border-radius: 8px;
    padding: 16px;
  }

  .placeholder-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 8px;
    font-size: 12px;
    color: var(--text-secondary);
  }

  .placeholder-content {
    text-align: center;
    padding: 20px;
    color: var(--text-muted);
  }

  .placeholder-content p {
    margin: 0 0 4px;
    font-size: 13px;
  }

  .placeholder-content small {
    font-size: 11px;
  }

  /* Metadata Section */
  .metadata-section {
    display: flex;
    flex-wrap: wrap;
    gap: 16px;
    padding: 12px;
    background: var(--bg-tertiary);
    border-radius: 8px;
    margin-bottom: 16px;
  }

  .metadata-item {
    display: flex;
    gap: 8px;
    font-size: 12px;
  }

  .metadata-item .label {
    color: var(--text-muted);
  }

  .metadata-item .value {
    color: var(--text-primary);
    font-weight: 500;
  }

  .metadata-item .value.success {
    color: #10b981;
  }

  /* Export Actions */
  .export-actions {
    display: flex;
    align-items: center;
    gap: 8px;
    padding-top: 12px;
    border-top: 1px solid var(--border-subtle);
  }

  .export-label {
    font-size: 11px;
    color: var(--text-muted);
  }

  .export-btn {
    display: flex;
    align-items: center;
    gap: 4px;
    padding: 6px 10px;
    background: var(--bg-tertiary);
    border: 1px solid var(--border-subtle);
    border-radius: 4px;
    color: var(--text-secondary);
    font-size: 11px;
    cursor: pointer;
    transition: all 0.15s;
  }

  .export-btn:hover {
    background: var(--bg-surface);
    color: var(--text-primary);
  }

  /* Trades Section */
  .trades-section {
    background: var(--bg-secondary);
    border-radius: 10px;
    padding: 20px;
  }

  .trades-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 16px;
  }

  .trades-header h3 {
    margin: 0;
    font-size: 14px;
    color: var(--text-primary);
  }

  .trade-filters {
    display: flex;
    gap: 8px;
  }

  .trade-filters select {
    padding: 4px 8px;
    background: var(--bg-tertiary);
    border: 1px solid var(--border-subtle);
    border-radius: 4px;
    color: var(--text-primary);
    font-size: 11px;
  }

  .trades-list {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .trade-item {
    background: var(--bg-tertiary);
    border-radius: 8px;
    overflow: hidden;
    transition: all 0.15s;
  }

  .trade-item.expanded {
    background: var(--bg-surface);
  }

  .trade-summary {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 12px 16px;
    cursor: pointer;
  }

  .trade-summary:hover {
    background: var(--bg-surface);
  }

  .trade-info {
    display: flex;
    align-items: center;
    gap: 12px;
    flex: 1;
  }

  .trade-icon {
    color: var(--text-muted);
  }

  .trade-time {
    font-size: 12px;
    color: var(--text-primary);
    font-family: 'JetBrains Mono', monospace;
  }

  .trade-symbol {
    padding: 4px 8px;
    background: var(--bg-secondary);
    border-radius: 4px;
    font-size: 11px;
    font-weight: 500;
    color: var(--text-primary);
  }

  .trade-pnl {
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    font-weight: 600;
    padding: 6px 12px;
    border-radius: 4px;
    margin-right: 12px;
  }

  .trade-pnl.positive {
    color: #10b981;
    background: rgba(16, 185, 129, 0.1);
  }

  .trade-pnl.negative {
    color: #ef4444;
    background: rgba(239, 68, 68, 0.1);
  }

  .expand-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 24px;
    height: 24px;
    background: transparent;
    border: none;
    color: var(--text-muted);
    cursor: pointer;
  }

  .trade-details {
    padding: 16px;
    border-top: 1px solid var(--border-subtle);
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .detail-row {
    display: flex;
    justify-content: space-between;
    font-size: 12px;
  }

  .detail-row.full-width {
    flex-direction: column;
    gap: 4px;
  }

  .detail-label {
    color: var(--text-muted);
  }

  .detail-value {
    color: var(--text-primary);
  }

  .detail-value.reason {
    background: var(--bg-secondary);
    padding: 8px;
    border-radius: 4px;
    font-size: 13px;
  }

  .view-details-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 6px;
    padding: 8px 12px;
    background: var(--accent-primary);
    border: none;
    border-radius: 6px;
    color: var(--bg-primary);
    font-size: 12px;
    cursor: pointer;
    align-self: flex-start;
    margin-top: 8px;
  }

  /* Modal */
  .modal-overlay {
    position: fixed;
    inset: 0;
    background: rgba(0, 0, 0, 0.7);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 100;
  }

  .modal {
    background: var(--bg-secondary);
    border-radius: 12px;
    width: 600px;
    max-width: 90%;
    max-height: 80vh;
    overflow: hidden;
    display: flex;
    flex-direction: column;
  }

  .modal-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 16px 20px;
    border-bottom: 1px solid var(--border-subtle);
  }

  .modal-header h3 {
    margin: 0;
    font-size: 16px;
    color: var(--text-primary);
  }

  .modal-content {
    flex: 1;
    overflow-y: auto;
    padding: 20px;
  }

  .trade-detail-section {
    margin-bottom: 20px;
  }

  .trade-detail-section h4 {
    margin: 0 0 12px;
    font-size: 13px;
    color: var(--text-primary);
  }

  .detail-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 12px;
  }

  .detail-item {
    display: flex;
    justify-content: space-between;
    padding: 8px 12px;
    background: var(--bg-tertiary);
    border-radius: 6px;
    font-size: 12px;
  }

  .detail-item .label {
    color: var(--text-muted);
  }

  .detail-item .value {
    color: var(--text-primary);
    font-weight: 500;
  }

  .trade-reason {
    padding: 12px;
    background: var(--bg-tertiary);
    border-radius: 8px;
    font-size: 13px;
    color: var(--text-primary);
    line-height: 1.5;
  }
</style>
