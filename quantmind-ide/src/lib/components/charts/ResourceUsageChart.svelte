<script lang="ts">
  import { run } from 'svelte/legacy';

  import { onMount, onDestroy } from 'svelte';
  import Chart from 'chart.js/auto';
  import { cpuHistory, memoryHistory } from '../../services/metricsWebSocket';
  import { theme } from '../../stores/themeStore';

  interface Props {
    height?: number;
    timeRange?: '1m' | '5m' | '15m' | '1h';
    showCpu?: boolean;
    showMemory?: boolean;
  }

  let {
    height = 200,
    timeRange = '5m',
    showCpu = true,
    showMemory = true
  }: Props = $props();

  let canvas: HTMLCanvasElement = $state();
  let chart: Chart | null = $state(null);

  function getTimeRangeSeconds(): number {
    switch (timeRange) {
      case '1m': return 60;
      case '5m': return 300;
      case '15m': return 900;
      case '1h': return 3600;
      default: return 300;
    }
  }

  function getChartColors() {
    const isDark = $theme.name.includes('dark') || $theme.name.includes('matrix') || $theme.name.includes('trading');
    return {
      cpu: isDark ? '#00ff00' : '#4caf50',
      cpuBg: isDark ? 'rgba(0, 255, 0, 0.1)' : 'rgba(76, 175, 80, 0.1)',
      memory: isDark ? '#ff9800' : '#ff9800',
      memoryBg: isDark ? 'rgba(255, 152, 0, 0.1)' : 'rgba(255, 152, 0, 0.1)',
      grid: isDark ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)',
      text: isDark ? '#999999' : '#666666'
    };
  }

  function initChart() {
    if (!canvas) return;

    const colors = getChartColors();
    const timeRangeSeconds = getTimeRangeSeconds();
    const now = new Date();
    const cutoff = new Date(now.getTime() - timeRangeSeconds * 1000);

    const cpuData = $cpuHistory.filter(d => d.time >= cutoff);
    const memData = $memoryHistory.filter(d => d.time >= cutoff);

    const datasets: Chart.ChartData['datasets'] = [];

    if (showCpu) {
      datasets.push({
        label: 'CPU',
        data: cpuData.map(d => d.value),
        borderColor: colors.cpu,
        backgroundColor: colors.cpuBg,
        fill: true,
        tension: 0.4,
        pointRadius: 0,
        borderWidth: 2,
        yAxisID: 'y'
      });
    }

    if (showMemory) {
      datasets.push({
        label: 'Memory',
        data: memData.map(d => d.value),
        borderColor: colors.memory,
        backgroundColor: colors.memoryBg,
        fill: true,
        tension: 0.4,
        pointRadius: 0,
        borderWidth: 2,
        yAxisID: 'y'
      });
    }

    chart = new Chart(canvas, {
      type: 'line',
      data: {
        labels: cpuData.map(d => formatTime(d.time)),
        datasets
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: {
          duration: 0
        },
        interaction: {
          intersect: false,
          mode: 'index'
        },
        plugins: {
          legend: {
            display: true,
            position: 'top',
            labels: {
              color: colors.text,
              usePointStyle: true,
              pointStyle: 'circle',
              padding: 15
            }
          },
          tooltip: {
            backgroundColor: 'rgba(0, 0, 0, 0.8)',
            titleColor: '#fff',
            bodyColor: '#fff',
            callbacks: {
              label: (item) => `${item.dataset.label}: ${item.formattedValue}%`
            }
          }
        },
        scales: {
          x: {
            display: true,
            grid: {
              color: colors.grid,
              drawBorder: false
            },
            ticks: {
              color: colors.text,
              maxRotation: 0,
              autoSkip: true,
              maxTicksLimit: 6
            }
          },
          y: {
            display: true,
            position: 'left',
            min: 0,
            max: 100,
            grid: {
              color: colors.grid,
              drawBorder: false
            },
            ticks: {
              color: colors.text,
              callback: (value) => `${value}%`
            }
          }
        }
      }
    });
  }

  function updateChart() {
    if (!chart) return;

    const colors = getChartColors();
    const timeRangeSeconds = getTimeRangeSeconds();
    const now = new Date();
    const cutoff = new Date(now.getTime() - timeRangeSeconds * 1000);

    const cpuData = $cpuHistory.filter(d => d.time >= cutoff);
    const memData = $memoryHistory.filter(d => d.time >= cutoff);

    chart.data.labels = cpuData.map(d => formatTime(d.time));

    let datasetIndex = 0;
    if (showCpu && chart.data.datasets[datasetIndex]) {
      chart.data.datasets[datasetIndex].data = cpuData.map(d => d.value);
      chart.data.datasets[datasetIndex].borderColor = colors.cpu;
      chart.data.datasets[datasetIndex].backgroundColor = colors.cpuBg;
      datasetIndex++;
    }

    if (showMemory && chart.data.datasets[datasetIndex]) {
      chart.data.datasets[datasetIndex].data = memData.map(d => d.value);
      chart.data.datasets[datasetIndex].borderColor = colors.memory;
      chart.data.datasets[datasetIndex].backgroundColor = colors.memoryBg;
    }

    // Update colors
    if (chart.options.scales?.x) {
      chart.options.scales.x.grid!.color = colors.grid;
      chart.options.scales.x.ticks!.color = colors.text;
    }
    if (chart.options.scales?.y) {
      chart.options.scales.y.grid!.color = colors.grid;
      chart.options.scales.y.ticks!.color = colors.text;
    }
    if (chart.options.plugins?.legend?.labels) {
      chart.options.plugins.legend.labels.color = colors.text;
    }

    chart.update('none');
  }

  function formatTime(date: Date): string {
    return date.toLocaleTimeString('en-US', {
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
      hour12: false
    });
  }

  let unsubscribeCpu: (() => void) | null = null;
  let unsubscribeMem: (() => void) | null = null;

  onMount(() => {
    initChart();
    unsubscribeCpu = cpuHistory.subscribe(() => updateChart());
    unsubscribeMem = memoryHistory.subscribe(() => updateChart());
  });

  onDestroy(() => {
    if (unsubscribeCpu) unsubscribeCpu();
    if (unsubscribeMem) unsubscribeMem();
    if (chart) {
      chart.destroy();
      chart = null;
    }
  });

  run(() => {
    if (chart && $theme) updateChart();
  });
  run(() => {
    if (chart && timeRange) updateChart();
  });
</script>

<div class="resource-usage-chart" style="height: {height}px;">
  <canvas bind:this={canvas}></canvas>
</div>

<style>
  .resource-usage-chart {
    position: relative;
    width: 100%;
  }
</style>
