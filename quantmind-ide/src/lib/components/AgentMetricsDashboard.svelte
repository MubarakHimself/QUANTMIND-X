<!-- @migration-task Error while migrating Svelte code: This type of directive is not valid on components
https://svelte.dev/e/component_invalid_directive -->
<!-- @migration-task Error while migrating Svelte code: This type of directive is not valid on components
https://svelte.dev/e/component_invalid_directive -->
<script lang="ts">
  import { onMount } from 'svelte';
  import {
    Activity, Bot, DollarSign, Clock, CheckCircle, XCircle,
    TrendingUp, RefreshCw, BarChart3, PieChart
  } from 'lucide-svelte';
  import Chart from 'chart.js/auto';
  import MetricCard from './MetricCard.svelte';

  // Types
  interface AgentTokenUsage { agent_id: string; agent_name: string; input_tokens: number; output_tokens: number; total_tokens: number; cost: number; }
  interface AgentTaskStats { agent_id: string; agent_name: string; total_tasks: number; successful_tasks: number; failed_tasks: number; success_rate: number; avg_latency_ms: number; }
  interface AgentCostBreakdown { agent_id: string; agent_name: string; input_cost: number; output_cost: number; total_cost: number; cost_percentage: number; }
  interface MetricsSummary { total_agents: number; total_tokens: number; total_cost: number; overall_success_rate: number; avg_latency_ms: number; period_start: string; period_end: string; }
  interface AgentMetricsResponse { summary: MetricsSummary; token_usage: AgentTokenUsage[]; task_stats: AgentTaskStats[]; latency_metrics: any[]; cost_breakdown: AgentCostBreakdown[]; timestamp: string; }

  // State
  let isLoading = true, error: string | null = null, metrics: AgentMetricsResponse | null = null, periodHours = 24;
  let tokenChartCanvas: HTMLCanvasElement, costChartCanvas: HTMLCanvasElement, successChartCanvas: HTMLCanvasElement;
  let tokenChart: Chart | null = null, costChart: Chart | null = null, successChart: Chart | null = null;

  const colors = { primary: '#6366f1', success: '#22c55e', warning: '#f59e0b', danger: '#ef4444', info: '#3b82f6', purple: '#a855f7', pink: '#ec4899', cyan: '#06b6d4', grid: 'rgba(255, 255, 255, 0.1)', text: '#9ca3af' };
  const agentColors = [colors.primary, colors.success, colors.warning, colors.danger, colors.info, colors.purple, colors.pink, colors.cyan];

  async function fetchMetrics() {
    isLoading = true; error = null;
    try {
      const response = await fetch(`/api/agent-metrics?period_hours=${periodHours}`);
      if (!response.ok) throw new Error(`HTTP error: ${response.status}`);
      metrics = await response.json();
      setTimeout(initCharts, 0);
    } catch (e) { error = e instanceof Error ? e.message : 'Failed to fetch metrics'; }
    finally { isLoading = false; }
  }

  function getChartOptions(title: string) {
    return { responsive: true, maintainAspectRatio: false, plugins: { legend: { position: 'top' as const, labels: { color: colors.text } }, title: { display: false, text: title } }, scales: { x: { grid: { color: colors.grid }, ticks: { color: colors.text } }, y: { grid: { color: colors.grid }, ticks: { color: colors.text } } } };
  }

  function initCharts() {
    if (!metrics) return;
    [tokenChart, costChart, successChart].forEach(c => c?.destroy());
    const ctx1 = tokenChartCanvas?.getContext('2d'), ctx2 = costChartCanvas?.getContext('2d'), ctx3 = successChartCanvas?.getContext('2d');
    if (ctx1) tokenChart = new Chart(ctx1, { type: 'bar', data: { labels: metrics.token_usage.map(t => t.agent_name), datasets: [{ label: 'Input Tokens', data: metrics.token_usage.map(t => t.input_tokens), backgroundColor: colors.primary, borderRadius: 4 }, { label: 'Output Tokens', data: metrics.token_usage.map(t => t.output_tokens), backgroundColor: colors.info, borderRadius: 4 }] }, options: getChartOptions('Token Usage') });
    if (ctx2) costChart = new Chart(ctx2, { type: 'doughnut', data: { labels: metrics.cost_breakdown.map(c => c.agent_name), datasets: [{ data: metrics.cost_breakdown.map(c => c.total_cost), backgroundColor: agentColors.slice(0, metrics.cost_breakdown.length), borderWidth: 0 }] }, options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { position: 'right', labels: { color: colors.text, padding: 10 } } } } });
    if (ctx3) successChart = new Chart(ctx3, { type: 'bar', data: { labels: metrics.task_stats.map(t => t.agent_name), datasets: [{ label: 'Success Rate %', data: metrics.task_stats.map(t => t.success_rate), backgroundColor: metrics.task_stats.map(t => t.success_rate >= 90 ? colors.success : t.success_rate >= 70 ? colors.warning : colors.danger), borderRadius: 4 }] }, options: getChartOptions('Task Success Rate') });
  }

  const formatNumber = (num: number): string => num >= 1000000 ? (num / 1000000).toFixed(1) + 'M' : num >= 1000 ? (num / 1000).toFixed(1) + 'K' : num.toFixed(0);
  const formatCost = (cost: number): string => '$' + cost.toFixed(4);

  onMount(() => { fetchMetrics(); return () => { [tokenChart, costChart, successChart].forEach(c => c?.destroy()); }; });
  $: if (periodHours && !isLoading) fetchMetrics();
</script>

<div class="agent-metrics-dashboard">
  <div class="dashboard-header">
    <div class="header-left"><Activity size={20} /><h2>Agent Metrics Dashboard</h2></div>
    <div class="header-right">
      <select bind:value={periodHours} class="period-select">
        <option value={1}>Last 1 hour</option><option value={6}>Last 6 hours</option><option value={24}>Last 24 hours</option>
        <option value={72}>Last 3 days</option><option value={168}>Last 7 days</option>
      </select>
      <button class="refresh-btn" on:click={fetchMetrics} disabled={isLoading}><RefreshCw size={16} class:spinning={isLoading} /></button>
    </div>
  </div>

  {#if isLoading}
    <div class="loading-state"><div class="spinner"></div><p>Loading agent metrics...</p></div>
  {:else if error}
    <div class="error-state"><XCircle size={48} /><p>{error}</p><button on:click={fetchMetrics}>Retry</button></div>
  {:else if metrics}
    <div class="summary-grid">
      <MetricCard title="Total Agents" value={metrics.summary.total_agents} icon={Bot} trend="neutral" />
      <MetricCard title="Total Tokens" value={formatNumber(metrics.summary.total_tokens)} unit="tokens" icon={Activity} trend="neutral" />
      <MetricCard title="Total Cost" value={formatCost(metrics.summary.total_cost)} icon={DollarSign} trend="neutral" />
      <MetricCard title="Success Rate" value={metrics.summary.overall_success_rate.toFixed(1)} unit="%" icon={CheckCircle} trend={metrics.summary.overall_success_rate >= 90 ? 'up' : 'neutral'} threshold={{ warning: 70, critical: 50 }} />
      <MetricCard title="Avg Latency" value={metrics.summary.avg_latency_ms.toFixed(0)} unit="ms" icon={Clock} trend={metrics.summary.avg_latency_ms > 500 ? 'up' : 'neutral'} threshold={{ warning: 500, critical: 1000 }} />
    </div>

    <div class="charts-row">
      <div class="chart-container"><h3><BarChart3 size={16} /> Token Usage</h3><div class="chart-wrapper"><canvas bind:this={tokenChartCanvas}></canvas></div></div>
      <div class="chart-container"><h3><PieChart size={16} /> Cost Distribution</h3><div class="chart-wrapper"><canvas bind:this={costChartCanvas}></canvas></div></div>
      <div class="chart-container"><h3><TrendingUp size={16} /> Success Rate</h3><div class="chart-wrapper"><canvas bind:this={successChartCanvas}></canvas></div></div>
    </div>

    <div class="tables-row">
      <div class="table-container">
        <h3>Token Usage Details</h3>
        <table><thead><tr><th>Agent</th><th>Input</th><th>Output</th><th>Total</th><th>Cost</th></tr></thead>
          <tbody>{#each metrics.token_usage as token}<tr><td>{token.agent_name}</td><td>{formatNumber(token.input_tokens)}</td><td>{formatNumber(token.output_tokens)}</td><td>{formatNumber(token.total_tokens)}</td><td class="cost-cell">{formatCost(token.cost)}</td></tr>{/each}</tbody>
        </table>
      </div>
      <div class="table-container">
        <h3>Task Performance</h3>
        <table><thead><tr><th>Agent</th><th>Total</th><th>Success</th><th>Failed</th><th>Rate</th><th>Latency</th></tr></thead>
          <tbody>{#each metrics.task_stats as task}<tr><td>{task.agent_name}</td><td>{task.total_tasks}</td><td class="success-cell">{task.successful_tasks}</td><td class="danger-cell">{task.failed_tasks}</td><td class:success-rate-high={task.success_rate >= 90} class:success-rate-med={task.success_rate >= 70 && task.success_rate < 90} class:success-rate-low={task.success_rate < 70}>{task.success_rate.toFixed(1)}%</td><td>{task.avg_latency_ms.toFixed(0)}ms</td></tr>{/each}</tbody>
        </table>
      </div>
    </div>
  {/if}
</div>

<style>
  .agent-metrics-dashboard { display: flex; flex-direction: column; gap: 20px; padding: 20px; background: var(--bg-primary); height: 100%; overflow-y: auto; }
  .dashboard-header { display: flex; justify-content: space-between; align-items: center; padding-bottom: 16px; border-bottom: 1px solid var(--border-subtle); }
  .header-left { display: flex; align-items: center; gap: 12px; }
  .header-left h2 { font-size: 20px; font-weight: 600; color: var(--text-primary); margin: 0; }
  .header-right { display: flex; align-items: center; gap: 12px; }
  .period-select { padding: 8px 12px; background: var(--bg-secondary); border: 1px solid var(--border-subtle); border-radius: 6px; color: var(--text-primary); font-size: 14px; cursor: pointer; }
  .refresh-btn { display: flex; align-items: center; justify-content: center; width: 36px; height: 36px; background: var(--bg-secondary); border: 1px solid var(--border-subtle); border-radius: 6px; color: var(--text-primary); cursor: pointer; transition: all 0.2s; }
  .refresh-btn:hover:not(:disabled) { background: var(--bg-tertiary); border-color: var(--accent-primary); }
  .refresh-btn:disabled { opacity: 0.5; cursor: not-allowed; }
  .spinning { animation: spin 1s linear infinite; }
  @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
  .loading-state, .error-state { display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 16px; padding: 60px; color: var(--text-muted); }
  .spinner { width: 40px; height: 40px; border: 3px solid var(--border-subtle); border-top-color: var(--accent-primary); border-radius: 50%; animation: spin 1s linear infinite; }
  .error-state { color: var(--accent-danger); }
  .error-state button { padding: 8px 16px; background: var(--accent-primary); color: white; border: none; border-radius: 6px; cursor: pointer; }
  .summary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 16px; }
  .charts-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 16px; }
  .chart-container { background: var(--bg-secondary); border: 1px solid var(--border-subtle); border-radius: 8px; padding: 16px; }
  .chart-container h3 { display: flex; align-items: center; gap: 8px; font-size: 14px; font-weight: 500; color: var(--text-muted); margin: 0 0 12px 0; text-transform: uppercase; letter-spacing: 0.5px; }
  .chart-wrapper { height: 220px; position: relative; }
  .tables-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 16px; }
  .table-container { background: var(--bg-secondary); border: 1px solid var(--border-subtle); border-radius: 8px; padding: 16px; overflow-x: auto; }
  .table-container h3 { font-size: 14px; font-weight: 500; color: var(--text-muted); margin: 0 0 12px 0; text-transform: uppercase; letter-spacing: 0.5px; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th { text-align: left; padding: 8px 12px; color: var(--text-muted); font-weight: 500; border-bottom: 1px solid var(--border-subtle); }
  td { padding: 10px 12px; color: var(--text-primary); border-bottom: 1px solid var(--border-subtle); }
  tr:hover { background: var(--bg-tertiary); }
  .cost-cell { color: var(--accent-success); font-weight: 500; }
  .success-cell { color: var(--accent-success); }
  .danger-cell { color: var(--accent-danger); }
  .success-rate-high { color: var(--accent-success); font-weight: 500; }
  .success-rate-med { color: var(--accent-warning); font-weight: 500; }
  .success-rate-low { color: var(--accent-danger); font-weight: 500; }
</style>
