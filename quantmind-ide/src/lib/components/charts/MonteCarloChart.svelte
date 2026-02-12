<script lang="ts">
  /**
   * Monte Carlo Visualization Component
   * 
   * Displays histogram of Monte Carlo simulation results with confidence intervals.
   * Uses native SVG for visualization - no external dependencies.
   */
  
  import { onMount } from 'svelte';
  import { TrendingUp, TrendingDown, Activity, Target } from 'lucide-svelte';
  
  export let data: {
    returns: number[];
    confidence: number;
    worstCase: number;
    bestCase: number;
    median: number;
    mean: number;
  } = {
    returns: [],
    confidence: 0,
    worstCase: 0,
    bestCase: 0,
    median: 0,
    mean: 0
  };
  
  // Chart dimensions
  const width = 600;
  const height = 300;
  const margin = { top: 20, right: 20, bottom: 40, left: 50 };
  const chartWidth = width - margin.left - margin.right;
  const chartHeight = height - margin.top - margin.bottom;
  
  // Histogram bins
  let bins: { x: number; y: number; width: number; height: number; count: number; label: string }[] = [];
  let maxCount = 0;
  let minValue = 0;
  let maxValue = 0;
  
  $: if (data.returns.length > 0) {
    calculateHistogram();
  }
  
  function calculateHistogram() {
    const numBins = 30;
    const values = data.returns;
    
    minValue = Math.min(...values);
    maxValue = Math.max(...values);
    const range = maxValue - minValue;
    const binWidth = range / numBins;
    
    // Create bins
    const binCounts = new Array(numBins).fill(0);
    values.forEach(v => {
      const binIndex = Math.min(Math.floor((v - minValue) / binWidth), numBins - 1);
      binCounts[binIndex]++;
    });
    
    maxCount = Math.max(...binCounts);
    
    // Convert to chart coordinates
    bins = binCounts.map((count, i) => ({
      x: margin.left + (i / numBins) * chartWidth,
      y: margin.top + chartHeight - (count / maxCount) * chartHeight,
      width: chartWidth / numBins - 1,
      height: (count / maxCount) * chartHeight,
      count,
      label: ((minValue + i * binWidth)).toFixed(1)
    }));
  }
  
  // Get x position for a value
  function getX(value: number): number {
    return margin.left + ((value - minValue) / (maxValue - minValue)) * chartWidth;
  }
  
  // Format percentage
  function formatPct(value: number): string {
    const sign = value >= 0 ? '+' : '';
    return `${sign}${value.toFixed(2)}%`;
  }
</script>

<div class="monte-carlo-chart">
  <div class="chart-header">
    <h3><Activity size={18} /> Monte Carlo Distribution</h3>
    <span class="simulations">{data.returns.length.toLocaleString()} simulations</span>
  </div>
  
  <svg {width} {height}>
    <!-- Histogram bars -->
    {#each bins as bin}
      <rect
        x={bin.x}
        y={bin.y}
        width={bin.width}
        height={bin.height}
        class="bar"
        class:positive={parseFloat(bin.label) >= 0}
        class:negative={parseFloat(bin.label) < 0}
      />
    {/each}
    
    <!-- Zero line -->
    {#if minValue < 0 && maxValue > 0}
      <line
        x1={getX(0)}
        y1={margin.top}
        x2={getX(0)}
        y2={margin.top + chartHeight}
        class="zero-line"
      />
      <text x={getX(0)} y={margin.top + chartHeight + 15} class="axis-label">0%</text>
    {/if}
    
    <!-- Confidence line (95%) -->
    {#if data.confidence}
      <line
        x1={getX(data.confidence)}
        y1={margin.top}
        x2={getX(data.confidence)}
        y2={margin.top + chartHeight}
        class="confidence-line"
      />
      <text x={getX(data.confidence)} y={margin.top - 5} class="confidence-label">
        95% Confidence
      </text>
    {/if}
    
    <!-- Mean line -->
    {#if data.mean}
      <line
        x1={getX(data.mean)}
        y1={margin.top}
        x2={getX(data.mean)}
        y2={margin.top + chartHeight}
        class="mean-line"
      />
    {/if}
    
    <!-- X Axis -->
    <line
      x1={margin.left}
      y1={margin.top + chartHeight}
      x2={margin.left + chartWidth}
      y2={margin.top + chartHeight}
      class="axis"
    />
    
    <!-- X Axis labels -->
    <text x={margin.left} y={margin.top + chartHeight + 25} class="axis-label">
      {formatPct(minValue)}
    </text>
    <text x={margin.left + chartWidth} y={margin.top + chartHeight + 25} class="axis-label" text-anchor="end">
      {formatPct(maxValue)}
    </text>
    
    <!-- Y Axis -->
    <line
      x1={margin.left}
      y1={margin.top}
      x2={margin.left}
      y2={margin.top + chartHeight}
      class="axis"
    />
  </svg>
  
  <div class="stats-grid">
    <div class="stat positive">
      <TrendingUp size={16} />
      <span class="label">Best Case</span>
      <span class="value">{formatPct(data.bestCase)}</span>
    </div>
    <div class="stat neutral">
      <Target size={16} />
      <span class="label">Mean Return</span>
      <span class="value">{formatPct(data.mean)}</span>
    </div>
    <div class="stat highlight">
      <Activity size={16} />
      <span class="label">95% Confidence</span>
      <span class="value">{formatPct(data.confidence)}</span>
    </div>
    <div class="stat negative">
      <TrendingDown size={16} />
      <span class="label">Worst Case</span>
      <span class="value">{formatPct(data.worstCase)}</span>
    </div>
  </div>
</div>

<style>
  .monte-carlo-chart {
    background: var(--bg-secondary);
    border-radius: 12px;
    padding: 20px;
    border: 1px solid var(--border-subtle);
  }
  
  .chart-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 16px;
  }
  
  .chart-header h3 {
    display: flex;
    align-items: center;
    gap: 8px;
    margin: 0;
    font-size: 14px;
    color: var(--text-primary);
  }
  
  .simulations {
    font-size: 11px;
    color: var(--text-muted);
    background: var(--bg-tertiary);
    padding: 4px 8px;
    border-radius: 4px;
  }
  
  svg {
    display: block;
    width: 100%;
    height: auto;
    max-width: 100%;
  }
  
  .bar {
    transition: opacity 0.15s ease;
  }
  
  .bar.positive {
    fill: rgba(16, 185, 129, 0.6);
  }
  
  .bar.negative {
    fill: rgba(239, 68, 68, 0.6);
  }
  
  .bar:hover {
    opacity: 0.8;
  }
  
  .axis {
    stroke: var(--border-subtle);
    stroke-width: 1;
  }
  
  .axis-label {
    font-size: 10px;
    fill: var(--text-muted);
  }
  
  .zero-line {
    stroke: var(--text-muted);
    stroke-width: 1;
    stroke-dasharray: 4 4;
  }
  
  .confidence-line {
    stroke: var(--accent-primary);
    stroke-width: 2;
    stroke-dasharray: 6 3;
  }
  
  .confidence-label {
    font-size: 10px;
    fill: var(--accent-primary);
    font-weight: 600;
  }
  
  .mean-line {
    stroke: #fbbf24;
    stroke-width: 2;
  }
  
  .stats-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-top: 16px;
  }
  
  .stat {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 4px;
    padding: 12px;
    background: var(--bg-tertiary);
    border-radius: 8px;
  }
  
  .stat .label {
    font-size: 10px;
    color: var(--text-muted);
  }
  
  .stat .value {
    font-size: 14px;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
  }
  
  .stat.positive {
    color: #10b981;
  }
  
  .stat.negative {
    color: #ef4444;
  }
  
  .stat.neutral {
    color: #fbbf24;
  }
  
  .stat.highlight {
    color: var(--accent-primary);
    border: 1px solid var(--accent-primary);
    background: rgba(99, 102, 241, 0.1);
  }
  
  @media (max-width: 600px) {
    .stats-grid {
      grid-template-columns: repeat(2, 1fr);
    }
  }
</style>
