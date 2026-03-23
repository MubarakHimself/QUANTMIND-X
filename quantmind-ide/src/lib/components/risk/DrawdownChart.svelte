<script lang="ts">
  /**
   * DrawdownChart - Drawdown Visualization
   *
   * Renders drawdown data as a filled area chart showing equity vs peak.
   */

  import { onMount } from 'svelte';

  interface DataPoint {
    timestamp: string;
    equity: number;
  }

  let { data = [], height = 100 }: { data?: DataPoint[]; height?: number } = $props();

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

    // Calculate peak equity and drawdown at each point
    let peak = data[0].equity;
    const drawdowns: number[] = [];

    data.forEach(point => {
      if (point.equity > peak) {
        peak = point.equity;
      }
      const drawdown = ((peak - point.equity) / peak) * 100;
      drawdowns.push(drawdown);
    });

    const maxDrawdown = Math.max(...drawdowns, 1);

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
      const value = maxDrawdown * (1 - i / 4);
      ctx.fillStyle = 'rgba(255, 255, 255, 0.4)';
      ctx.font = '10px JetBrains Mono, monospace';
      ctx.textAlign = 'right';
      ctx.fillText(`-${value.toFixed(1)}%`, padding.left - 5, y + 3);
    }

    // Draw the drawdown area
    ctx.beginPath();

    // Start from baseline
    const baseline = padding.top + chartHeight - padding.bottom;

    drawdowns.forEach((dd, i) => {
      const x = padding.left + (chartWidth / (drawdowns.length - 1 || 1)) * i;
      const y = baseline - (chartHeight - padding.top - padding.bottom) * (dd / maxDrawdown);

      if (i === 0) {
        ctx?.moveTo(x, baseline);
        ctx?.lineTo(x, y);
      } else {
        ctx?.lineTo(x, y);
      }
    });

    // Close path back to baseline
    ctx?.lineTo(padding.left + chartWidth, baseline);
    ctx?.closePath();

    // Fill with gradient
    const gradient = ctx.createLinearGradient(0, padding.top, 0, baseline);
    gradient.addColorStop(0, 'rgba(255, 59, 59, 0.4)');
    gradient.addColorStop(1, 'rgba(255, 59, 59, 0.1)');
    ctx.fillStyle = gradient;
    ctx.fill();

    // Draw drawdown line
    ctx.strokeStyle = '#ff3b3b';
    ctx.lineWidth = 2;
    ctx.beginPath();

    drawdowns.forEach((dd, i) => {
      const x = padding.left + (chartWidth / (drawdowns.length - 1 || 1)) * i;
      const y = baseline - (chartHeight - padding.top - padding.bottom) * (dd / maxDrawdown);

      if (i === 0) {
        ctx?.moveTo(x, y);
      } else {
        ctx?.lineTo(x, y);
      }
    });

    ctx.stroke();

    // Mark max drawdown point
    const maxDdIndex = drawdowns.indexOf(maxDrawdown);
    if (maxDdIndex >= 0) {
      const x = padding.left + (chartWidth / (drawdowns.length - 1 || 1)) * maxDdIndex;
      const y = baseline - (chartHeight - padding.top - padding.bottom) * (maxDrawdown / maxDrawdown);

      ctx.fillStyle = '#ff3b3b';
      ctx.beginPath();
      ctx.arc(x, y, 4, 0, Math.PI * 2);
      ctx.fill();
    }
  }
</script>

<div class="drawdown-chart" style="height: {height}px">
  <canvas bind:this={canvas} width="400" height={height}></canvas>
</div>

<style>
  .drawdown-chart {
    width: 100%;
    position: relative;
  }

  canvas {
    width: 100%;
    height: 100%;
  }
</style>
