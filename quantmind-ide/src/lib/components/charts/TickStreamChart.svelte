<script lang="ts">
  import { onMount, onDestroy, tick } from 'svelte';
  import Chart from 'chart.js/auto';
  import { tickRateHistory } from '../../services/metricsWebSocket';
  import { theme } from '../../stores/themeStore';

  export let height = 200;
  export let timeRange: '1m' | '5m' | '15m' | '1h' = '5m';

  let canvas: HTMLCanvasElement;
  let chart: Chart | null = null;

  // Get time range in seconds
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
      primary: isDark ? '#00ff00' : '#4caf50',
      secondary: isDark ? 'rgba(0, 255, 0, 0.1)' : 'rgba(76, 175, 80, 0.1)',
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

    // Filter data to time range
    const data = $tickRateHistory.filter(d => d.time >= cutoff);

    chart = new Chart(canvas, {
      type: 'line',
      data: {
        labels: data.map(d => formatTime(d.time)),
        datasets: [{
          label: 'Ticks/sec',
          data: data.map(d => d.value),
          borderColor: colors.primary,
          backgroundColor: colors.secondary,
          fill: true,
          tension: 0.4,
          pointRadius: 0,
          pointHoverRadius: 4,
          borderWidth: 2
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: {
          duration: 0 // Disable animation for real-time updates
        },
        interaction: {
          intersect: false,
          mode: 'index'
        },
        plugins: {
          legend: {
            display: false
          },
          tooltip: {
            backgroundColor: 'rgba(0, 0, 0, 0.8)',
            titleColor: '#fff',
            bodyColor: '#fff',
            borderColor: colors.primary,
            borderWidth: 1,
            displayColors: false,
            callbacks: {
              title: (items) => items[0]?.label || '',
              label: (item) => `${item.formattedValue} ticks/sec`
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
            beginAtZero: true,
            grid: {
              color: colors.grid,
              drawBorder: false
            },
            ticks: {
              color: colors.text,
              callback: (value) => `${value}`
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

    // Filter data to time range
    const data = $tickRateHistory.filter(d => d.time >= cutoff);

    chart.data.labels = data.map(d => formatTime(d.time));
    chart.data.datasets[0].data = data.map(d => d.value);
    chart.data.datasets[0].borderColor = colors.primary;
    chart.data.datasets[0].backgroundColor = colors.secondary;

    chart.options.scales!.x!.grid!.color = colors.grid;
    chart.options.scales!.x!.ticks!.color = colors.text;
    chart.options.scales!.y!.grid!.color = colors.grid;
    chart.options.scales!.y!.ticks!.color = colors.text;

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

  // Subscribe to data changes
  let unsubscribe: (() => void) | null = null;

  onMount(() => {
    initChart();
    unsubscribe = tickRateHistory.subscribe(() => {
      updateChart();
    });
  });

  onDestroy(() => {
    if (unsubscribe) unsubscribe();
    if (chart) {
      chart.destroy();
      chart = null;
    }
  });

  // Watch for theme changes
  $: if (chart && $theme) {
    updateChart();
  }

  // Watch for time range changes
  $: if (chart && timeRange) {
    updateChart();
  }
</script>

<div class="tick-stream-chart" style="height: {height}px;">
  <canvas bind:this={canvas}></canvas>
</div>

<style>
  .tick-stream-chart {
    position: relative;
    width: 100%;
  }
</style>
