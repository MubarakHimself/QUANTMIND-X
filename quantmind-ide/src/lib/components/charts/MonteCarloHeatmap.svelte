<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import * as d3 from 'd3';
  import { theme } from '../../stores/themeStore';

  export let data: {
    runs: number[][];
    days: number[];
    minValue: number;
    maxValue: number;
  } = {
    runs: [],
    days: [],
    minValue: 0,
    maxValue: 0
  };

  export let height = 400;
  export let showAxis = true;

  let container: HTMLDivElement;
  let svg: d3.Selection<SVGSVGElement, unknown, null, undefined> | null = null;

  $: colors = getColors();

  function getColors() {
    const isDark = $theme.name.includes('dark') || $theme.name.includes('matrix') || $theme.name.includes('trading');
    return {
      background: isDark ? '#1e1e1e' : '#ffffff',
      text: isDark ? '#cccccc' : '#333333',
      grid: isDark ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)',
      positive: '#4caf50',
      negative: '#f44336',
      neutral: isDark ? '#333333' : '#e0e0e0'
    };
  }

  function initHeatmap() {
    if (!container || !data.runs.length) return;

    // Clear existing SVG
    d3.select(container).selectAll('*').remove();

    const margin = { top: 20, right: 80, bottom: 50, left: 60 };
    const width = container.clientWidth - margin.left - margin.right;
    const heightInner = height - margin.top - margin.bottom;

    svg = d3.select(container)
      .append('svg')
      .attr('width', width + margin.left + margin.right)
      .attr('height', height + margin.top + margin.bottom)
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // Scales
    const xScale = d3.scaleBand()
      .domain(data.days.map(d => d.toString()))
      .range([0, width]);

    const yScale = d3.scaleBand()
      .domain(d3.range(data.runs.length).map(i => i.toString()))
      .range([0, heightInner]);

    // Color scale
    const colorScale = d3.scaleSequential<d3.RGBColor>()
      .interpolator(d3.interpolateRdYlGn)
      .domain([data.minValue, data.maxValue]);

    // Create tooltip
    const tooltip = d3.select(container)
      .append('div')
      .attr('class', 'heatmap-tooltip')
      .style('opacity', 0)
      .style('position', 'absolute')
      .style('background', 'rgba(0, 0, 0, 0.8)')
      .style('color', '#fff')
      .style('padding', '8px 12px')
      .style('border-radius', '4px')
      .style('font-size', '12px')
      .style('pointer-events', 'none');

    // Draw cells
    const cells = svg.selectAll('.cell')
      .data(data.runs.flatMap((run, runIndex) =>
        run.map((value, dayIndex) => ({ runIndex, dayIndex, value }))
      ))
      .enter()
      .append('rect')
      .attr('class', 'cell')
      .attr('x', d => xScale(data.days[d.dayIndex].toString()) || 0)
      .attr('y', d => yScale(d.runIndex.toString()) || 0)
      .attr('width', xScale.bandwidth())
      .attr('height', yScale.bandwidth())
      .attr('fill', d => colorScale(d.value))
      .attr('stroke', 'transparent')
      .attr('stroke-width', 0.5);

    // Tooltip interactions
    cells.on('mouseover', (event: MouseEvent, d) => {
      d3.select(event.target as Element)
        .attr('stroke', '#fff')
        .attr('stroke-width', 2);

      tooltip.transition()
        .duration(200)
        .style('opacity', 0.9);

      tooltip.html(`
        <div>Run: ${d.runIndex + 1}</div>
        <div>Day: ${data.days[d.dayIndex]}</div>
        <div>P&L: $${d.value.toLocaleString()}</div>
      `)
        .style('left', `${event.offsetX + 10}px`)
        .style('top', `${event.offsetY - 28}px`);
    })
    .on('mouseout', (event: MouseEvent) => {
      d3.select(event.target as Element)
        .attr('stroke', 'transparent')
        .attr('stroke-width', 0.5);

      tooltip.transition()
        .duration(500)
        .style('opacity', 0);
    });

    if (showAxis) {
      // X axis
      svg.append('g')
        .attr('transform', `translate(0,${heightInner})`)
        .call(d3.axisBottom(xScale).tickValues(xScale.domain().filter((_, i) => i % Math.ceil(data.days.length / 10) === 0)))
        .selectAll('text')
        .attr('fill', colors.text)
        .attr('font-size', '11px');

      svg.append('text')
        .attr('x', width / 2)
        .attr('y', heightInner + 40)
        .attr('fill', colors.text)
        .attr('text-anchor', 'middle')
        .attr('font-size', '12px')
        .text('Trading Days');

      // Y axis (show subset of runs)
      svg.append('g')
        .call(d3.axisLeft(yScale).tickValues(yScale.domain().filter((_, i) => i % Math.ceil(data.runs.length / 10) === 0)))
        .selectAll('text')
        .attr('fill', colors.text)
        .attr('font-size', '11px');

      svg.append('text')
        .attr('transform', 'rotate(-90)')
        .attr('x', -heightInner / 2)
        .attr('y', -45)
        .attr('fill', colors.text)
        .attr('text-anchor', 'middle')
        .attr('font-size', '12px')
        .text('Simulation Run');

      // Color legend
      const legendWidth = 20;
      const legendHeight = heightInner;

      const legendScale = d3.scaleLinear()
        .domain([data.minValue, data.maxValue])
        .range([legendHeight, 0]);

      const legendAxis = d3.axisRight(legendScale)
        .ticks(5)
        .tickFormat(d => `$${d.valueOf().toLocaleString()}`);

      const legend = svg.append('g')
        .attr('transform', `translate(${width + 10}, 0)`);

      // Legend gradient
      const defs = svg.append('defs');
      const gradient = defs.append('linearGradient')
        .attr('id', 'legend-gradient')
        .attr('x1', '0%')
        .attr('y1', '100%')
        .attr('x2', '0%')
        .attr('y2', '0%');

      const numStops = 10;
      for (let i = 0; i <= numStops; i++) {
        const value = data.minValue + (data.maxValue - data.minValue) * (i / numStops);
        gradient.append('stop')
          .attr('offset', `${(i / numStops) * 100}%`)
          .attr('stop-color', colorScale(value));
      }

      legend.append('rect')
        .attr('width', legendWidth)
        .attr('height', legendHeight)
        .style('fill', 'url(#legend-gradient)');

      legend.append('g')
        .attr('transform', `translate(${legendWidth}, 0)`)
        .call(legendAxis)
        .selectAll('text')
        .attr('fill', colors.text)
        .attr('font-size', '10px');

      legend.selectAll('.domain, .tick line')
        .attr('stroke', colors.grid);
    }
  }

  function updateHeatmap() {
    if (svg && data.runs.length) {
      initHeatmap();
    }
  }

  onMount(() => {
    initHeatmap();
    window.addEventListener('resize', updateHeatmap);
  });

  onDestroy(() => {
    window.removeEventListener('resize', updateHeatmap);
    if (container) {
      d3.select(container).selectAll('*').remove();
    }
  });

  $: if (data) {
    updateHeatmap();
  }

  $: if ($theme) {
    updateHeatmap();
  }
</script>

<div class="heatmap-container" bind:this={container} style="height: {height}px;">
  {#if !data.runs.length}
    <div class="no-data">
      <span>No simulation data available</span>
    </div>
  {/if}
</div>

<style>
  .heatmap-container {
    position: relative;
    width: 100%;
    overflow: visible;
  }

  .no-data {
    display: flex;
    align-items: center;
    justify-content: center;
    height: 100%;
    color: var(--text-muted);
    font-size: 14px;
    background: var(--bg-secondary);
    border-radius: 8px;
  }

  :global(.heatmap-tooltip) {
    z-index: 1000 !important;
  }
</style>
