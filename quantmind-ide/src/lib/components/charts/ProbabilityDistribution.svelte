<script lang="ts">
  import { run } from 'svelte/legacy';

  import { onMount, onDestroy } from 'svelte';
  import Chart from 'chart.js/auto';
  import annotationPlugin from 'chartjs-plugin-annotation';
  import { theme } from '../../stores/themeStore';

  // Register the annotation plugin with Chart.js
  Chart.register(annotationPlugin);


  interface Props {
    data?: {
    values: number[];
    bins: number[];
    frequencies: number[];
    statistics: {
      mean: number;
      median: number;
      stdDev: number;
      percentile5: number;
      percentile95: number;
      riskOfRuin: number;
    };
  };
    height?: number;
    showStatistics?: boolean;
  }

  let { data = {
    values: [],
    bins: [],
    frequencies: [],
    statistics: {
      mean: 0,
      median: 0,
      stdDev: 0,
      percentile5: 0,
      percentile95: 0,
      riskOfRuin: 0
    }
  }, height = 300, showStatistics = true }: Props = $props();

  let canvas: HTMLCanvasElement = $state();
  let chart: Chart | null = $state(null);

  function getChartColors() {
    const isDark = $theme.name.includes('dark') || $theme.name.includes('matrix') || $theme.name.includes('trading');
    return {
      background: isDark ? '#1e1e1e' : '#ffffff',
      grid: isDark ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)',
      text: isDark ? '#999999' : '#666666',
      primary: isDark ? '#00ff00' : '#4caf50',
      primaryBg: isDark ? 'rgba(0, 255, 0, 0.2)' : 'rgba(76, 175, 80, 0.2)',
      negative: '#f44336',
      positive: '#4caf50',
      warning: '#ff9800',
      info: '#2196f3'
    };
  }

  function initChart() {
    if (!canvas || !data.bins.length) return;

    const colors = getChartColors();

    // Create annotations for key percentiles
    const annotations: any = {};

    if (data.statistics.mean) {
      annotations.mean = {
        type: 'line',
        xMin: data.statistics.mean,
        xMax: data.statistics.mean,
        borderColor: colors.primary,
        borderWidth: 2,
        borderDash: [5, 5],
        label: {
          display: true,
          content: `Mean: $${data.statistics.mean.toLocaleString()}`,
          position: 'start'
        }
      };
    }

    if (data.statistics.percentile5) {
      annotations.p5 = {
        type: 'line',
        xMin: data.statistics.percentile5,
        xMax: data.statistics.percentile5,
        borderColor: colors.negative,
        borderWidth: 2,
        label: {
          display: true,
          content: `5th: $${data.statistics.percentile5.toLocaleString()}`,
          position: 'start'
        }
      };
    }

    if (data.statistics.percentile95) {
      annotations.p95 = {
        type: 'line',
        xMin: data.statistics.percentile95,
        xMax: data.statistics.percentile95,
        borderColor: colors.positive,
        borderWidth: 2,
        label: {
          display: true,
          content: `95th: $${data.statistics.percentile95.toLocaleString()}`,
          position: 'end'
        }
      };
    }

    chart = new Chart(canvas, {
      type: 'bar',
      data: {
        labels: data.bins.map(b => `$${b.toLocaleString()}`),
        datasets: [{
          label: 'Frequency',
          data: data.frequencies,
          backgroundColor: colors.primaryBg,
          borderColor: colors.primary,
          borderWidth: 1,
          borderRadius: 2
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            display: false
          },
          tooltip: {
            backgroundColor: 'rgba(0, 0, 0, 0.8)',
            titleColor: '#fff',
            bodyColor: '#fff',
            callbacks: {
              title: (items: any[]) => `Value Range: ${items[0]?.label}`,
              label: (item: any) => `Frequency: ${item.formattedValue} simulations`
            }
          },
          annotation: {
            annotations
          }
        },
        scales: {
          x: {
            display: true,
            title: {
              display: true,
              text: 'Portfolio Value',
              color: colors.text
            },
            grid: {
              color: colors.grid,
              drawBorder: false
            },
            ticks: {
              color: colors.text,
              maxTicksLimit: 10,
              maxRotation: 45
            }
          },
          y: {
            display: true,
            title: {
              display: true,
              text: 'Frequency',
              color: colors.text
            },
            grid: {
              color: colors.grid,
              drawBorder: false
            },
            ticks: {
              color: colors.text
            }
          }
        }
      }
    });
  }

  function updateChart() {
    if (!chart) return;

    const colors = getChartColors();

    chart.data.labels = data.bins.map(b => `$${b.toLocaleString()}`);
    chart.data.datasets[0].data = data.frequencies;
    chart.data.datasets[0].backgroundColor = colors.primaryBg;
    chart.data.datasets[0].borderColor = colors.primary;

    if (chart.options.scales?.x) {
      chart.options.scales.x.grid!.color = colors.grid;
      chart.options.scales.x.ticks!.color = colors.text;
      if (chart.options.scales.x.title) {
        chart.options.scales.x.title.color = colors.text;
      }
    }
    if (chart.options.scales?.y) {
      chart.options.scales.y.grid!.color = colors.grid;
      chart.options.scales.y.ticks!.color = colors.text;
      if (chart.options.scales.y.title) {
        chart.options.scales.y.title.color = colors.text;
      }
    }

    chart.update();
  }

  onMount(() => {
    initChart();
  });

  onDestroy(() => {
    if (chart) {
      chart.destroy();
      chart = null;
    }
  });

  run(() => {
    if (chart && data) {
      updateChart();
    }
  });

  run(() => {
    if (chart && $theme) {
      updateChart();
    }
  });
</script>

<div class="probability-distribution" style="height: {height}px;">
  {#if !data.bins.length}
    <div class="no-data">
      <span>No distribution data available</span>
    </div>
  {:else}
    <canvas bind:this={canvas}></canvas>

    {#if showStatistics && data.statistics}
      <div class="statistics-overlay">
        <div class="stat-item">
          <span class="stat-label">Mean</span>
          <span class="stat-value">${data.statistics.mean.toLocaleString()}</span>
        </div>
        <div class="stat-item">
          <span class="stat-label">Median</span>
          <span class="stat-value">${data.statistics.median.toLocaleString()}</span>
        </div>
        <div class="stat-item">
          <span class="stat-label">Std Dev</span>
          <span class="stat-value">${data.statistics.stdDev.toLocaleString()}</span>
        </div>
        <div class="stat-item warning">
          <span class="stat-label">Risk of Ruin</span>
          <span class="stat-value">{(data.statistics.riskOfRuin * 100).toFixed(1)}%</span>
        </div>
      </div>
    {/if}
  {/if}
</div>

<style>
  .probability-distribution {
    position: relative;
    width: 100%;
  }

  .no-data {
    display: flex;
    align-items: center;
    justify-content: center;
    height: 100%;
    color: var(--color-text-muted);
    font-size: 14px;
    background: var(--color-bg-surface);
    border-radius: 8px;
  }

  .statistics-overlay {
    position: absolute;
    top: 10px;
    right: 10px;
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    max-width: 200px;
  }

  .stat-item {
    display: flex;
    flex-direction: column;
    padding: 6px 10px;
    background: rgba(0, 0, 0, 0.6);
    border-radius: 4px;
    font-size: 11px;
  }

  .stat-label {
    color: var(--color-text-muted);
    font-size: 10px;
    text-transform: uppercase;
  }

  .stat-value {
    color: var(--color-text-primary);
    font-weight: 600;
  }

  .stat-item.warning .stat-value {
    color: var(--color-accent-amber);
  }
</style>
