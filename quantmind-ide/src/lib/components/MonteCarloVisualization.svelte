<script lang="ts">
  import { onMount, onDestroy, createEventDispatcher } from 'svelte';
  import {
    TrendingUp, Grid, BarChart2, Download, Play, Pause,
    Settings, RefreshCw, ChevronDown, Share2, Save
  } from 'lucide-svelte';

  import FanChart from './charts/FanChart.svelte';
  import MonteCarloHeatmap from './charts/MonteCarloHeatmap.svelte';
  import ProbabilityDistribution from './charts/ProbabilityDistribution.svelte';

  export let runId: string = '';
  export let initialCapital: number = 10000;
  export let numSimulations: number = 10000;
  export let tradingDays: number = 252;

  const dispatch = createEventDispatcher();

  // Active tab
  let activeTab: 'fan' | 'heatmap' | 'distribution' = 'fan';

  // Simulation state
  let isRunning = false;
  let progress = 0;
  let completedSimulations = 0;

  // Results data
  let fanChartData = {
    percentiles: { p10: [], p25: [], p50: [], p75: [], p90: [] } as { p10: number[]; p25: number[]; p50: number[]; p75: number[]; p90: number[] },
    days: [] as number[],
    initialValue: initialCapital
  };

  let heatmapData = {
    runs: [] as number[][],
    days: [] as number[],
    minValue: 0,
    maxValue: 0
  };

  let distributionData = {
    values: [] as number[],
    bins: [] as number[],
    frequencies: [] as number[],
    statistics: {
      mean: 0,
      median: 0,
      stdDev: 0,
      percentile5: 0,
      percentile95: 0,
      riskOfRuin: 0
    }
  };

  // Summary statistics
  let summaryStats = {
    expectedValue: 0,
    worstCase: 0,
    bestCase: 0,
    riskOfRuin: 0,
    sharpeRatio: 0,
    maxDrawdown: 0,
    winProbability: 0
  };

  // WebSocket connection
  let ws: WebSocket | null = null;

  function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.hostname;
    const port = import.meta.env.VITE_API_PORT || '8000';

    ws = new WebSocket(`${protocol}//${host}:${port}/api/monte-carlo/ws`);

    ws.onopen = () => {
      console.log('Monte Carlo WebSocket connected');
      if (runId) {
        // Request existing results
        ws?.send(JSON.stringify({ type: 'get_results', run_id: runId }));
      }
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        handleWebSocketMessage(data);
      } catch (error) {
        console.error('Failed to parse message:', error);
      }
    };

    ws.onclose = () => {
      console.log('Monte Carlo WebSocket disconnected');
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };
  }

  function handleWebSocketMessage(data: any) {
    switch (data.type) {
      case 'progress':
        progress = data.progress;
        completedSimulations = data.completed;
        break;

      case 'results':
        updateResults(data.payload);
        isRunning = false;
        progress = 100;
        break;

      case 'partial_results':
        updatePartialResults(data.payload);
        break;

      case 'error':
        console.error('Simulation error:', data.message);
        isRunning = false;
        break;
    }
  }

  function updateResults(payload: any) {
    // Fan chart data
    if (payload.fan_chart) {
      fanChartData = {
        percentiles: payload.fan_chart.percentiles,
        days: payload.fan_chart.days,
        initialValue: payload.fan_chart.initial_value || initialCapital
      };
    }

    // Heatmap data
    if (payload.heatmap) {
      heatmapData = {
        runs: payload.heatmap.runs,
        days: payload.heatmap.days,
        minValue: payload.heatmap.min_value,
        maxValue: payload.heatmap.max_value
      };
    }

    // Distribution data
    if (payload.distribution) {
      distributionData = {
        values: payload.distribution.values,
        bins: payload.distribution.bins,
        frequencies: payload.distribution.frequencies,
        statistics: payload.distribution.statistics
      };
    }

    // Summary statistics
    if (payload.statistics) {
      summaryStats = {
        expectedValue: payload.statistics.expected_value || 0,
        worstCase: payload.statistics.worst_case || 0,
        bestCase: payload.statistics.best_case || 0,
        riskOfRuin: payload.statistics.risk_of_ruin || 0,
        sharpeRatio: payload.statistics.sharpe_ratio || 0,
        maxDrawdown: payload.statistics.max_drawdown || 0,
        winProbability: payload.statistics.win_probability || 0
      };
    }

    dispatch('results', { results: payload });
  }

  function updatePartialResults(payload: any) {
    // Update with partial results for real-time visualization
    if (payload.fan_chart) {
      fanChartData = { ...fanChartData, ...payload.fan_chart };
    }
  }

  function startSimulation() {
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      connectWebSocket();
      setTimeout(startSimulation, 500);
      return;
    }

    isRunning = true;
    progress = 0;
    completedSimulations = 0;

    ws.send(JSON.stringify({
      type: 'start_simulation',
      config: {
        initial_capital: initialCapital,
        num_simulations: numSimulations,
        trading_days: tradingDays,
        run_id: runId
      }
    }));

    dispatch('start');
  }

  function pauseSimulation() {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'pause' }));
      isRunning = false;
      dispatch('pause');
    }
  }

  function resumeSimulation() {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'resume' }));
      isRunning = true;
      dispatch('resume');
    }
  }

  function exportPNG() {
    const canvas = document.querySelector('.chart-container canvas') as HTMLCanvasElement;
    if (canvas) {
      const link = document.createElement('a');
      link.download = `monte-carlo-${activeTab}-${Date.now()}.png`;
      link.href = canvas.toDataURL('image/png');
      link.click();
    }
    dispatch('export', { format: 'png' });
  }

  function exportSVG() {
    // For D3-based heatmap
    const svg = document.querySelector('.chart-container svg');
    if (svg) {
      const svgData = new XMLSerializer().serializeToString(svg);
      const blob = new Blob([svgData], { type: 'image/svg+xml' });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.download = `monte-carlo-heatmap-${Date.now()}.svg`;
      link.href = url;
      link.click();
      URL.revokeObjectURL(url);
    }
    dispatch('export', { format: 'svg' });
  }

  function exportCSV() {
    // Export simulation data as CSV
    const headers = ['Run', ...fanChartData.days.map(d => `Day_${d}`)];
    const rows = heatmapData.runs.map((run, i) => [i + 1, ...run]);

    const csv = [
      headers.join(','),
      ...rows.map(row => row.join(','))
    ].join('\n');

    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.download = `monte-carlo-data-${Date.now()}.csv`;
    link.href = url;
    link.click();
    URL.revokeObjectURL(url);

    dispatch('export', { format: 'csv' });
  }

  onMount(() => {
    if (runId) {
      connectWebSocket();
    }
  });

  onDestroy(() => {
    if (ws) {
      ws.close();
    }
  });
</script>

<div class="monte-carlo-visualization">
  <!-- Header -->
  <div class="visualization-header">
    <div class="header-left">
      <TrendingUp size={20} />
      <h3>Monte Carlo Simulation</h3>
      {#if isRunning}
        <span class="progress-badge">
          {completedSimulations.toLocaleString()} / {numSimulations.toLocaleString()} ({progress.toFixed(0)}%)
        </span>
      {/if}
    </div>
    <div class="header-actions">
      {#if !isRunning && progress < 100}
        <button class="action-btn primary" on:click={startSimulation}>
          <Play size={14} />
          <span>Run Simulation</span>
        </button>
      {:else if isRunning}
        <button class="action-btn" on:click={pauseSimulation}>
          <Pause size={14} />
          <span>Pause</span>
        </button>
      {:else}
        <button class="action-btn" on:click={resumeSimulation}>
          <Play size={14} />
          <span>Resume</span>
        </button>
      {/if}

      <div class="dropdown">
        <button class="action-btn">
          <Download size={14} />
          <ChevronDown size={12} />
        </button>
        <div class="dropdown-menu">
          <button on:click={exportPNG}>PNG Image</button>
          <button on:click={exportSVG}>SVG Vector</button>
          <button on:click={exportCSV}>CSV Data</button>
        </div>
      </div>
    </div>
  </div>

  <!-- Tabs -->
  <div class="tab-bar">
    <button class:active={activeTab === 'fan'} on:click={() => activeTab = 'fan'}>
      <TrendingUp size={16} />
      <span>Fan Chart</span>
    </button>
    <button class:active={activeTab === 'heatmap'} on:click={() => activeTab = 'heatmap'}>
      <Grid size={16} />
      <span>Heatmap</span>
    </button>
    <button class:active={activeTab === 'distribution'} on:click={() => activeTab = 'distribution'}>
      <BarChart2 size={16} />
      <span>Distribution</span>
    </button>
  </div>

  <!-- Content -->
  <div class="visualization-content">
    <!-- Summary Statistics -->
    <div class="summary-panel">
      <h4>Summary Statistics</h4>
      <div class="stats-grid">
        <div class="stat-card">
          <span class="stat-label">Expected Value</span>
          <span class="stat-value positive">${summaryStats.expectedValue.toLocaleString()}</span>
        </div>
        <div class="stat-card">
          <span class="stat-label">Worst Case (5%)</span>
          <span class="stat-value negative">${summaryStats.worstCase.toLocaleString()}</span>
        </div>
        <div class="stat-card">
          <span class="stat-label">Best Case (95%)</span>
          <span class="stat-value positive">${summaryStats.bestCase.toLocaleString()}</span>
        </div>
        <div class="stat-card">
          <span class="stat-label">Risk of Ruin</span>
          <span class="stat-value warning">{(summaryStats.riskOfRuin * 100).toFixed(1)}%</span>
        </div>
        <div class="stat-card">
          <span class="stat-label">Sharpe Ratio</span>
          <span class="stat-value">{summaryStats.sharpeRatio.toFixed(2)}</span>
        </div>
        <div class="stat-card">
          <span class="stat-label">Max Drawdown</span>
          <span class="stat-value negative">{(summaryStats.maxDrawdown * 100).toFixed(1)}%</span>
        </div>
        <div class="stat-card">
          <span class="stat-label">Win Probability</span>
          <span class="stat-value positive">{(summaryStats.winProbability * 100).toFixed(1)}%</span>
        </div>
      </div>
    </div>

    <!-- Chart Container -->
    <div class="chart-container">
      {#if activeTab === 'fan'}
        <FanChart data={fanChartData} height={400} showLegend={true} />
      {:else if activeTab === 'heatmap'}
        <MonteCarloHeatmap data={heatmapData} height={450} showAxis={true} />
      {:else}
        <ProbabilityDistribution data={distributionData} height={400} showStatistics={true} />
      {/if}
    </div>
  </div>

  <!-- Progress Bar (when running) -->
  {#if isRunning}
    <div class="progress-bar">
      <div class="progress-fill" style="width: {progress}%"></div>
    </div>
  {/if}
</div>

<style>
  .monte-carlo-visualization {
    display: flex;
    flex-direction: column;
    height: 100%;
    background: var(--bg-primary);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
    overflow: hidden;
  }

  .visualization-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 16px;
    background: var(--bg-secondary);
    border-bottom: 1px solid var(--border-subtle);
  }

  .header-left {
    display: flex;
    align-items: center;
    gap: 10px;
  }

  .header-left h3 {
    margin: 0;
    font-size: 16px;
    font-weight: 600;
    color: var(--text-primary);
  }

  .progress-badge {
    padding: 4px 10px;
    background: var(--accent-primary);
    color: #000;
    border-radius: 12px;
    font-size: 12px;
    font-weight: 500;
  }

  .header-actions {
    display: flex;
    gap: 8px;
  }

  .action-btn {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 8px 12px;
    background: var(--bg-tertiary);
    border: 1px solid var(--border-subtle);
    border-radius: 6px;
    color: var(--text-secondary);
    font-size: 13px;
    cursor: pointer;
    transition: all 0.2s ease;
  }

  .action-btn:hover {
    background: var(--bg-secondary);
    color: var(--text-primary);
  }

  .action-btn.primary {
    background: var(--accent-primary);
    border-color: var(--accent-primary);
    color: #000;
  }

  .action-btn.primary:hover {
    background: var(--accent-secondary);
  }

  .dropdown {
    position: relative;
  }

  .dropdown-menu {
    display: none;
    position: absolute;
    top: 100%;
    right: 0;
    margin-top: 4px;
    background: var(--bg-secondary);
    border: 1px solid var(--border-subtle);
    border-radius: 6px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    z-index: 10;
    min-width: 120px;
  }

  .dropdown:hover .dropdown-menu {
    display: block;
  }

  .dropdown-menu button {
    display: block;
    width: 100%;
    padding: 8px 12px;
    background: transparent;
    border: none;
    color: var(--text-primary);
    font-size: 13px;
    text-align: left;
    cursor: pointer;
  }

  .dropdown-menu button:hover {
    background: var(--bg-tertiary);
  }

  .tab-bar {
    display: flex;
    border-bottom: 1px solid var(--border-subtle);
    background: var(--bg-secondary);
  }

  .tab-bar button {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 10px 20px;
    background: transparent;
    border: none;
    border-bottom: 2px solid transparent;
    color: var(--text-muted);
    font-size: 13px;
    cursor: pointer;
    transition: all 0.2s ease;
  }

  .tab-bar button:hover {
    color: var(--text-primary);
    background: var(--bg-tertiary);
  }

  .tab-bar button.active {
    color: var(--accent-primary);
    border-bottom-color: var(--accent-primary);
  }

  .visualization-content {
    display: grid;
    grid-template-columns: 280px 1fr;
    flex: 1;
    overflow: hidden;
  }

  .summary-panel {
    padding: 16px;
    background: var(--bg-secondary);
    border-right: 1px solid var(--border-subtle);
    overflow-y: auto;
  }

  .summary-panel h4 {
    margin: 0 0 12px 0;
    font-size: 14px;
    font-weight: 600;
    color: var(--text-primary);
  }

  .stats-grid {
    display: flex;
    flex-direction: column;
    gap: 10px;
  }

  .stat-card {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 10px 12px;
    background: var(--bg-tertiary);
    border-radius: 6px;
  }

  .stat-label {
    font-size: 12px;
    color: var(--text-muted);
  }

  .stat-value {
    font-size: 14px;
    font-weight: 600;
    color: var(--text-primary);
  }

  .stat-value.positive {
    color: var(--accent-success);
  }

  .stat-value.negative {
    color: var(--accent-danger);
  }

  .stat-value.warning {
    color: var(--accent-warning);
  }

  .chart-container {
    padding: 16px;
    overflow: auto;
  }

  .progress-bar {
    height: 3px;
    background: var(--bg-tertiary);
  }

  .progress-fill {
    height: 100%;
    background: var(--accent-primary);
    transition: width 0.3s ease;
  }
</style>
