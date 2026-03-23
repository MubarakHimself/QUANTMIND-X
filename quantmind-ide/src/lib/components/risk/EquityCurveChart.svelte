<script lang="ts">
  /**
   * EquityCurveChart - Equity Curve Visualization
   *
   * Renders equity curve data points as a line chart.
   */

  import { onMount } from 'svelte';

  interface DataPoint {
    timestamp: string;
    equity: number;
  }

  let { data = [], height = 150 }: { data?: DataPoint[]; height?: number } = $props();

  let canvas: HTMLCanvasElement;
  let ctx: CanvasRenderingContext2D | null = null;

  onMount(() => {
    ctx = canvas.getContext('2d');
    drawChart();
  });

  $effect(() => {
    if (ctx && data) {
      drawChart();
    }
  });

  function drawChart() {
    if (!ctx || !canvas || data.length === 0) return;

    const width = canvas.width;
    const chartHeight = height;
    const padding = { top: 10, right: 10, bottom: 20, left: 50 };
    const chartWidth = width - padding.left - padding.right;

    // Clear canvas
    ctx.clearRect(0, 0, width, height);

    // Find min/max values
    const equities = data.map(d => d.equity);
    const minEquity = Math.min(...equities);
    const maxEquity = Math.max(...equities);
    const range = maxEquity - minEquity || 1;

    // Draw grid lines
    ctx.strokeStyle = 'rgba(0, 212, 255, 0.1)';
    ctx.lineWidth = 1;

    // Horizontal grid lines
    for (let i = 0; i <= 4; i++) {
      const y = padding.top + (chartHeight - padding.top - padding.bottom) * (i / 4);
      ctx.beginPath();
      ctx.moveTo(padding.left, y);
      ctx.lineTo(width - padding.right, y);
      ctx.stroke();

      // Y-axis labels
      const value = maxEquity - (range * i / 4);
      ctx.fillStyle = 'rgba(255, 255, 255, 0.4)';
      ctx.font = '10px JetBrains Mono, monospace';
      ctx.textAlign = 'right';
      ctx.fillText(value.toFixed(0), padding.left - 5, y + 3);
    }

    // Draw the equity curve
    ctx.strokeStyle = '#00d4ff';
    ctx.lineWidth = 2;
    ctx.beginPath();

    data.forEach((point, i) => {
      const x = padding.left + (chartWidth / (data.length - 1 || 1)) * i;
      const y = padding.top + (chartHeight - padding.top - padding.bottom) * (1 - (point.equity - minEquity) / range);

      if (i === 0) {
        ctx?.moveTo(x, y);
      } else {
        ctx?.lineTo(x, y);
      }
    });

    ctx.stroke();

    // Fill area under curve
    ctx.lineTo(padding.left + chartWidth, padding.top + chartHeight - padding.bottom);
    ctx.lineTo(padding.left, padding.top + chartHeight - padding.bottom);
    ctx.closePath();

    const gradient = ctx.createLinearGradient(0, padding.top, 0, height - padding.bottom);
    gradient.addColorStop(0, 'rgba(0, 212, 255, 0.3)');
    gradient.addColorStop(1, 'rgba(0, 212, 255, 0)');
    ctx.fillStyle = gradient;
    ctx.fill();

    // Draw start and end points
    if (data.length > 0) {
      const firstPoint = data[0];
      const lastPoint = data[data.length - 1];

      const startX = padding.left;
      const startY = padding.top + (chartHeight - padding.top - padding.bottom) * (1 - (firstPoint.equity - minEquity) / range);

      const endX = padding.left + chartWidth;
      const endY = padding.top + (chartHeight - padding.top - padding.bottom) * (1 - (lastPoint.equity - minEquity) / range);

      // Start point
      ctx.fillStyle = '#00d4ff';
      ctx.beginPath();
      ctx.arc(startX, startY, 3, 0, Math.PI * 2);
      ctx.fill();

      // End point
      ctx.fillStyle = lastPoint.equity >= firstPoint.equity ? '#10b981' : '#ff3b3b';
      ctx.beginPath();
      ctx.arc(endX, endY, 3, 0, Math.PI * 2);
      ctx.fill();
    }
  }
</script>

<div class="equity-chart" style="height: {height}px">
  <canvas bind:this={canvas} width="400" height={height}></canvas>
</div>

<style>
  .equity-chart {
    width: 100%;
    position: relative;
  }

  canvas {
    width: 100%;
    height: 100%;
  }
</style>
