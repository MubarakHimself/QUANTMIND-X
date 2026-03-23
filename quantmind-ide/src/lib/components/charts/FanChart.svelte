<script lang="ts">
  import { run } from 'svelte/legacy';

  import { onMount, onDestroy } from 'svelte';
  import Chart from 'chart.js/auto';
  import { theme } from '../../stores/themeStore';


  interface Props {
    data?: {
    percentiles: {
      p10: number[];
      p25: number[];
      p50: number[];
      p75: number[];
      p90: number[];
    };
    days: number[];
    initialValue: number;
  };
    height?: number;
    showLegend?: boolean;
  }

  let { data = {
    percentiles: { p10: [], p25: [], p50: [], p75: [], p90: [] },
    days: [],
    initialValue: 10000
  }, height = 300, showLegend = true }: Props = $props();

  let canvas: HTMLCanvasElement = $state();
  let chart: Chart | null = $state(null);

  function getChartColors() {
    const isDark = $theme.name.includes('dark') || $theme.name.includes('matrix') || $theme.name.includes('trading');
    return {
      grid: isDark ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)',
      text: isDark ? '#999999' : '#666666',
      p90: 'rgba(76, 175, 80, 0.15)',
      p75: 'rgba(76, 175, 80, 0.25)',
      p50: 'rgba(76, 175, 80, 0.4)',
      p25: 'rgba(255, 152, 0, 0.25)',
      p10: 'rgba(244, 67, 54, 0.15)',
      median: isDark ? '#00ff00' : '#4caf50',
      medianBg: 'rgba(76, 175, 80, 0.3)'
    };
  }

  function initChart() {
    if (!canvas || !data.days.length) return;

    const colors = getChartColors();
    const labels = data.days.map(d => `Day ${d}`);

    chart = new Chart(canvas, {
      type: 'line',
      data: {
        labels,
        datasets: [
          // 10th-90th percentile band (widest)
          {
            label: '10th-90th Percentile',
            data: data.percentiles.p90.map((v, i) => ({
              x: labels[i],
              y: v
            })),
            borderColor: 'transparent',
            backgroundColor: colors.p90,
            fill: '+1',
            pointRadius: 0,
            tension: 0.4
          },
          // 90th percentile (for filling)
          {
            label: '90th Percentile',
            data: data.percentiles.p10.map((v, i) => ({
              x: labels[i],
              y: v
            })),
            borderColor: 'transparent',
            backgroundColor: colors.p10,
            fill: false,
            pointRadius: 0,
            tension: 0.4
          },
          // 25th-75th percentile band
          {
            label: '25th-75th Percentile',
            data: data.percentiles.p75.map((v, i) => ({
              x: labels[i],
              y: v
            })),
            borderColor: 'transparent',
            backgroundColor: colors.p75,
            fill: '+1',
            pointRadius: 0,
            tension: 0.4
          },
          // 75th percentile (for filling)
          {
            label: '75th Percentile',
            data: data.percentiles.p25.map((v, i) => ({
              x: labels[i],
              y: v
            })),
            borderColor: 'transparent',
            backgroundColor: colors.p25,
            fill: false,
            pointRadius: 0,
            tension: 0.4
          },
          // Median line
          {
            label: 'Median (50th)',
            data: data.percentiles.p50,
            borderColor: colors.median,
            backgroundColor: colors.medianBg,
            borderWidth: 3,
            fill: false,
            pointRadius: 2,
            pointHoverRadius: 5,
            tension: 0.4,
            pointBackgroundColor: colors.median
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: {
          intersect: false,
          mode: 'index'
        },
        plugins: {
          legend: {
            display: showLegend,
            position: 'top',
            labels: {
              color: colors.text,
              usePointStyle: true,
              filter: (item) => item.text === 'Median (50th)' || item.text === '25th-75th Percentile' || item.text === '10th-90th Percentile'
            }
          },
          tooltip: {
            backgroundColor: 'rgba(0, 0, 0, 0.8)',
            titleColor: '#fff',
            bodyColor: '#fff',
            callbacks: {
              label: (context) => {
                const idx = context.dataIndex;
                if (context.dataset.label === 'Median (50th)') {
                  return `Median: $${data.percentiles.p50[idx]?.toLocaleString() || 0}`;
                }
                return null;
              },
              afterBody: (contexts) => {
                if (contexts.length === 0) return [];
                const idx = contexts[0].dataIndex;
                return [
                  `90th: $${data.percentiles.p90[idx]?.toLocaleString() || 0}`,
                  `75th: $${data.percentiles.p75[idx]?.toLocaleString() || 0}`,
                  `50th: $${data.percentiles.p50[idx]?.toLocaleString() || 0}`,
                  `25th: $${data.percentiles.p25[idx]?.toLocaleString() || 0}`,
                  `10th: $${data.percentiles.p10[idx]?.toLocaleString() || 0}`
                ];
              }
            }
          }
        },
        scales: {
          x: {
            display: true,
            title: {
              display: true,
              text: 'Trading Days',
              color: colors.text
            },
            grid: {
              color: colors.grid,
              drawBorder: false
            },
            ticks: {
              color: colors.text,
              maxTicksLimit: 10
            }
          },
          y: {
            display: true,
            title: {
              display: true,
              text: 'Portfolio Value ($)',
              color: colors.text
            },
            grid: {
              color: colors.grid,
              drawBorder: false
            },
            ticks: {
              color: colors.text,
              callback: (value) => `$${Number(value).toLocaleString()}`
            }
          }
        }
      }
    });
  }

  function updateChart() {
    if (!chart) return;

    const colors = getChartColors();

    // Update data
    const labels = data.days.map(d => `Day ${d}`);
    chart.data.labels = labels;

    // Update datasets
    if (chart.data.datasets[0]) {
      chart.data.datasets[0].data = data.percentiles.p90.map((v, i) => ({ x: labels[i], y: v }));
      chart.data.datasets[0].backgroundColor = colors.p90;
    }
    if (chart.data.datasets[1]) {
      chart.data.datasets[1].data = data.percentiles.p10.map((v, i) => ({ x: labels[i], y: v }));
      chart.data.datasets[1].backgroundColor = colors.p10;
    }
    if (chart.data.datasets[2]) {
      chart.data.datasets[2].data = data.percentiles.p75.map((v, i) => ({ x: labels[i], y: v }));
      chart.data.datasets[2].backgroundColor = colors.p75;
    }
    if (chart.data.datasets[3]) {
      chart.data.datasets[3].data = data.percentiles.p25.map((v, i) => ({ x: labels[i], y: v }));
      chart.data.datasets[3].backgroundColor = colors.p25;
    }
    if (chart.data.datasets[4]) {
      chart.data.datasets[4].data = data.percentiles.p50;
      chart.data.datasets[4].borderColor = colors.median;
      chart.data.datasets[4].backgroundColor = colors.medianBg;
      chart.data.datasets[4].pointBackgroundColor = colors.median;
    }

    // Update colors
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
    if (chart.options.plugins?.legend?.labels) {
      chart.options.plugins.legend.labels.color = colors.text;
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

<div class="fan-chart" style="height: {height}px;">
  {#if !data.days.length}
    <div class="no-data">
      <span>No simulation data available</span>
    </div>
  {:else}
    <canvas bind:this={canvas}></canvas>
  {/if}
</div>

<style>
  .fan-chart {
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
</style>
