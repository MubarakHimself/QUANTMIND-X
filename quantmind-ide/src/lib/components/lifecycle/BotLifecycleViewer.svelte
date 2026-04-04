<!--
  BotLifecycleViewer.svelte

  Component for viewing a bot's complete lifecycle stage progression.
  Accessible from bot detail view.

  Features:
  - Stage progress indicator (5 steps: Born → Backtest → Paper → Live → Review)
  - Current stage highlighted
  - Stage report viewer with structured Q&A data
  - Metrics per stage (win rate, drawdown, PnL, etc.)
  - Stage transition timeline
  - Decline/Recovery loop status if applicable
-->
<script lang="ts">
  import { onMount } from 'svelte';
  import {
    Activity, CheckCircle, XCircle, AlertTriangle, Clock,
    TrendingUp, TrendingDown, DollarSign, BarChart3,
    ChevronRight, RefreshCw, Bot, Zap
  } from 'lucide-svelte';

  import type {
    BotLifecycle,
    StageReport,
    LifecycleStage,
    QAAnswer
  } from '$lib/stores/lifecycleStore';

  import {
    lifecycleStore,
    selectedBotLifecycle,
    selectedStageReport,
    stageProgress,
    isInRecovery,
    failedQuestions,
    passedQuestions,
    lifecycleLoading,
    lifecycleError,
    formatTimestamp,
    formatDurationDays,
    getStageColor,
    getRecoveryStatusColor,
    formatPnL,
    formatPercent
  } from '$lib/stores/lifecycleStore';

  // Props
  export let botId: string;
  export let compact = false;

  // Local state
  let showQAanswers = false;

  // Reactive
  $: lifecycle = $selectedBotLifecycle;
  $: progress = $stageProgress;
  $: currentReport = $selectedStageReport;
  $: failed = $failedQuestions;
  $: passed = $passedQuestions;
  $: loading = $lifecycleLoading;
  $: error = $lifecycleError;
  $: inRecovery = $isInRecovery;

  // Stages for progress indicator
  const stages: LifecycleStage[] = ['Born', 'Backtest', 'Paper', 'Live', 'Review'];

  onMount(() => {
    if (botId) {
      lifecycleStore.selectBot(botId);
    }
  });

  function selectStage(stage: LifecycleStage) {
    lifecycleStore.selectStage(stage);
    showQAanswers = true;
  }

  function handleRefresh() {
    if (botId) {
      lifecycleStore.fetchBotLifecycle(botId);
    }
  }

  function getMetricIcon(metric: string) {
    switch (metric) {
      case 'win_rate':
        return TrendingUp;
      case 'drawdown':
        return TrendingDown;
      case 'pnl':
        return DollarSign;
      default:
        return BarChart3;
    }
  }

  function getMetricLabel(metric: string): string {
    switch (metric) {
      case 'win_rate':
        return 'Win Rate';
      case 'drawdown':
        return 'Max Drawdown';
      case 'pnl':
        return 'P&L';
      case 'sharpe_ratio':
        return 'Sharpe Ratio';
      case 'profit_factor':
        return 'Profit Factor';
      case 'total_trades':
        return 'Total Trades';
      case 'consecutive_losses':
        return 'Max Consecutive Losses';
      case 'avg_win':
        return 'Avg Win';
      case 'avg_loss':
        return 'Avg Loss';
      case 'recovery_factor':
        return 'Recovery Factor';
      case 'max_drawdown_duration':
        return 'Max DD Duration';
      default:
        return metric;
    }
  }

  function formatMetricValue(metric: string, value: unknown): string {
    if (value === undefined || value === null) return 'N/A';

    switch (metric) {
      case 'win_rate':
      case 'drawdown':
        return formatPercent(value as number);
      case 'pnl':
      case 'avg_win':
      case 'avg_loss':
        return formatPnL(value as number);
      case 'sharpe_ratio':
      case 'profit_factor':
      case 'recovery_factor':
        return (value as number).toFixed(2);
      case 'total_trades':
      case 'consecutive_losses':
      case 'max_drawdown_duration':
        return value.toString();
      default:
        return String(value);
    }
  }
</script>

<div class="lifecycle-viewer" class:compact>
  {#if error}
    <div class="error-banner">
      <AlertTriangle size={16} />
      <span>{error}</span>
      <button class="retry-btn" on:click={handleRefresh}>
        <RefreshCw size={14} />
        Retry
      </button>
    </div>
  {/if}

  {#if loading && !lifecycle}
    <div class="loading-state">
      <RefreshCw size={24} class="spin" />
      <span>Loading lifecycle data...</span>
    </div>
  {:else if lifecycle}
    <div class="viewer-content">
      <!-- Header -->
      <div class="viewer-header">
        <div class="header-left">
          <Bot size={20} />
          <div class="header-info">
            <h3>{lifecycle.bot_id}</h3>
            <span class="header-meta">
              Created: {formatTimestamp(lifecycle.created_at)}
            </span>
          </div>
        </div>
        <div class="header-right">
          {#if inRecovery}
            <div class="recovery-badge" style="background-color: {getRecoveryStatusColor(lifecycle.current_report.decline_recovery_status)}">
              <AlertTriangle size={14} />
              <span>Recovery Mode</span>
            </div>
          {/if}
          <button class="icon-btn" on:click={handleRefresh} title="Refresh">
            <RefreshCw size={16} class={loading ? 'spin' : ''} />
          </button>
        </div>
      </div>

      <!-- Stage Progress Indicator -->
      {#if progress}
        <div class="stage-progress">
          {#each progress.stages as stage, index}
            <button
              class="stage-step"
              class:completed={progress.isCompleted(index)}
              class:current={progress.isCurrent(index)}
              class:pending={progress.isPending(index)}
              on:click={() => selectStage(stage)}
              style="--stage-color: {getStageColor(stage)}"
            >
              <div class="step-indicator">
                {#if progress.isCompleted(index)}
                  <CheckCircle size={16} />
                {:else if progress.isCurrent(index)}
                  <Activity size={16} />
                {:else}
                  <span class="step-number">{index + 1}</span>
                {/if}
              </div>
              <span class="step-label">{stage}</span>
            </button>

            {#if index < progress.stages.length - 1}
              <div class="step-connector" class:active={index < progress.currentIndex}></div>
            {/if}
          {/each}
        </div>
      {/if}

      <!-- Stage Report Viewer -->
      {#if currentReport}
        <div class="stage-report">
          <div class="report-header">
            <div class="report-title">
              <h4>{currentReport.stage} Stage Report</h4>
              <span class="report-dates">
                {formatTimestamp(currentReport.entered_at)}
                {#if currentReport.exited_at}
                  → {formatTimestamp(currentReport.exited_at)}
                {:else}
                  (Current)
                {/if}
              </span>
            </div>

            {#if currentReport.notes}
              <div class="report-notes">
                <span class="notes-label">Notes:</span>
                <span class="notes-text">{currentReport.notes}</span>
              </div>
            {/if}
          </div>

          <!-- Metrics Grid -->
          <div class="metrics-grid">
            {#each Object.entries(currentReport.metrics) as [metric, value]}
              {#if value !== undefined && value !== null}
                <div class="metric-card">
                  <div class="metric-icon">
                    <svelte:component this={getMetricIcon(metric)} size={16} />
                  </div>
                  <div class="metric-content">
                    <span class="metric-value">
                      {formatMetricValue(metric, value)}
                    </span>
                    <span class="metric-label">{getMetricLabel(metric)}</span>
                  </div>
                </div>
              {/if}
            {/each}
          </div>

          <!-- Q1-Q20 Answers -->
          <div class="qa-section">
            <button
              class="qa-toggle"
              on:click={() => showQAanswers = !showQAanswers}
            >
              <span>Q1-Q20 Answers</span>
              <span class="qa-summary">
                <span class="passed-count">{passed.length} passed</span>
                {#if failed.length > 0}
                  <span class="failed-count">{failed.length} failed</span>
                {/if}
              </span>
              <ChevronRight size={16} class={showQAanswers ? 'rotated' : ''} />
            </button>

            {#if showQAanswers}
              <div class="qa-list">
                {#each currentReport.q1_q20_answers as qa}
                  <div class="qa-item" class:failed={qa.passed === false}>
                    <div class="qa-status">
                      {#if qa.passed === true}
                        <CheckCircle size={14} class="status-pass" />
                      {:else if qa.passed === false}
                        <XCircle size={14} class="status-fail" />
                      {:else}
                        <span class="status-neutral">—</span>
                      {/if}
                    </div>
                    <div class="qa-content">
                      <span class="qa-question">
                        <span class="qa-id">{qa.question_id}:</span>
                        {qa.question}
                      </span>
                      <span class="qa-answer">
                        {typeof qa.answer === 'boolean' ? (qa.answer ? 'Yes' : 'No') : String(qa.answer)}
                      </span>
                    </div>
                  </div>
                {/each}
              </div>
            {/if}
          </div>
        </div>
      {:else if !compact}
        <div class="no-report">
          <span>Select a stage to view its report</span>
        </div>
      {/if}

      <!-- Stage Transition Timeline -->
      {#if !compact && lifecycle.stage_history.length > 1}
        <div class="timeline-section">
          <h4>Stage History</h4>
          <div class="timeline">
            {#each lifecycle.stage_history as stage, index}
              <div class="timeline-item" style="--stage-color: {getStageColor(stage.stage)}">
                <div class="timeline-marker">
                  {#if index === 0}
                    <Bot size={12} />
                  {:else if index === lifecycle.stage_history.length - 1 && lifecycle.current_stage !== stage.stage}
                    <CheckCircle size={12} />
                  {:else}
                    <CheckCircle size={12} />
                  {/if}
                </div>
                <div class="timeline-content">
                  <div class="timeline-header">
                    <span class="timeline-stage">{stage.stage}</span>
                    <span class="timeline-date">{formatTimestamp(stage.entered_at)}</span>
                  </div>
                  {#if stage.metrics.total_trades}
                    <div class="timeline-metrics">
                      <span>{stage.metrics.total_trades} trades</span>
                      {#if stage.metrics.win_rate}
                        <span>· {stage.metrics.win_rate}% WR</span>
                      {/if}
                      {#if stage.metrics.pnl}
                        <span>· {formatPnL(stage.metrics.pnl)}</span>
                      {/if}
                    </div>
                  {/if}
                </div>
              </div>
            {/each}

            {#if lifecycle.current_report.stage !== lifecycle.stage_history[lifecycle.stage_history.length - 1]?.stage}
              <div class="timeline-item current" style="--stage-color: {getStageColor(lifecycle.current_stage)}">
                <div class="timeline-marker">
                  <Activity size={12} />
                </div>
                <div class="timeline-content">
                  <div class="timeline-header">
                    <span class="timeline-stage">{lifecycle.current_stage}</span>
                    <span class="timeline-date">Current</span>
                  </div>
                </div>
              </div>
            {/if}
          </div>
        </div>
      {/if}
    </div>
  {:else if !loading}
    <div class="empty-state">
      <Bot size={32} />
      <span>No lifecycle data available for this bot</span>
    </div>
  {/if}
</div>

<style>
  .lifecycle-viewer {
    background: var(--bg-secondary, #1e293b);
    border-radius: 8px;
    padding: 16px;
    display: flex;
    flex-direction: column;
    gap: 16px;
    height: 100%;
    overflow: hidden;
  }

  .lifecycle-viewer.compact {
    padding: 12px;
    gap: 12px;
  }

  /* Error Banner */
  .error-banner {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 12px;
    background: var(--accent-danger, #ef4444);
    color: white;
    border-radius: 4px;
    font-size: 13px;
  }

  .retry-btn {
    display: flex;
    align-items: center;
    gap: 4px;
    margin-left: auto;
    padding: 4px 8px;
    background: rgba(255, 255, 255, 0.2);
    border: none;
    border-radius: 4px;
    color: white;
    cursor: pointer;
    font-size: 12px;
  }

  .retry-btn:hover {
    background: rgba(255, 255, 255, 0.3);
  }

  /* Loading State */
  .loading-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 12px;
    padding: 48px;
    color: var(--text-secondary, #94a3b8);
  }

  :global(.spin) {
    animation: spin 1s linear infinite;
  }

  @keyframes spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
  }

  /* Empty State */
  .empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 12px;
    padding: 48px;
    color: var(--text-secondary, #94a3b8);
  }

  /* Viewer Content */
  .viewer-content {
    display: flex;
    flex-direction: column;
    gap: 16px;
    overflow-y: auto;
  }

  /* Header */
  .viewer-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .header-left {
    display: flex;
    align-items: center;
    gap: 12px;
  }

  .header-info h3 {
    margin: 0;
    font-size: 16px;
    font-weight: 600;
  }

  .header-meta {
    font-size: 12px;
    color: var(--text-secondary, #94a3b8);
  }

  .header-right {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .recovery-badge {
    display: flex;
    align-items: center;
    gap: 4px;
    padding: 4px 8px;
    border-radius: 4px;
    color: white;
    font-size: 12px;
    font-weight: 500;
  }

  .icon-btn {
    background: none;
    border: none;
    color: var(--text-secondary, #94a3b8);
    cursor: pointer;
    padding: 4px;
    border-radius: 4px;
    transition: all 0.2s;
  }

  .icon-btn:hover {
    background: var(--bg-hover, #334155);
    color: var(--text-primary, #f1f5f9);
  }

  /* Stage Progress */
  .stage-progress {
    display: flex;
    align-items: center;
    gap: 4px;
    padding: 16px 0;
    overflow-x: auto;
  }

  .stage-step {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 8px;
    padding: 8px 16px;
    background: var(--bg-tertiary, #334155);
    border: 2px solid transparent;
    border-radius: 8px;
    cursor: pointer;
    transition: all 0.2s;
    min-width: 80px;
  }

  .stage-step:hover {
    background: var(--bg-hover, #3d4a5c);
  }

  .stage-step.completed {
    border-color: var(--stage-color);
    background: color-mix(in srgb, var(--stage-color) 15%, var(--bg-tertiary));
  }

  .stage-step.completed .step-indicator {
    color: var(--stage-color);
  }

  .stage-step.current {
    border-color: var(--stage-color);
    background: color-mix(in srgb, var(--stage-color) 25%, var(--bg-tertiary));
  }

  .stage-step.current .step-indicator {
    color: var(--stage-color);
  }

  .stage-step.pending {
    opacity: 0.6;
  }

  .step-indicator {
    width: 24px;
    height: 24px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: var(--bg-secondary, #1e293b);
    border-radius: 50%;
    color: var(--text-secondary, #94a3b8);
  }

  .step-number {
    font-size: 11px;
    font-weight: 600;
  }

  .step-label {
    font-size: 11px;
    font-weight: 500;
    color: var(--text-primary, #f1f5f9);
  }

  .step-connector {
    flex: 1;
    height: 2px;
    min-width: 20px;
    background: var(--bg-tertiary, #334155);
    border-radius: 1px;
  }

  .step-connector.active {
    background: var(--accent-primary, #3b82f6);
  }

  /* Stage Report */
  .stage-report {
    background: var(--bg-tertiary, #334155);
    border-radius: 8px;
    padding: 16px;
    display: flex;
    flex-direction: column;
    gap: 16px;
  }

  .report-header {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .report-title h4 {
    margin: 0;
    font-size: 14px;
    font-weight: 600;
  }

  .report-dates {
    font-size: 12px;
    color: var(--text-secondary, #94a3b8);
  }

  .report-notes {
    display: flex;
    gap: 8px;
    font-size: 12px;
    padding: 8px 12px;
    background: var(--bg-secondary, #1e293b);
    border-radius: 4px;
  }

  .notes-label {
    font-weight: 500;
    color: var(--text-secondary, #94a3b8);
  }

  .notes-text {
    color: var(--text-primary, #f1f5f9);
  }

  /* Metrics Grid */
  .metrics-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
    gap: 8px;
  }

  .metric-card {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 10px 12px;
    background: var(--bg-secondary, #1e293b);
    border-radius: 6px;
  }

  .metric-icon {
    color: var(--text-secondary, #94a3b8);
  }

  .metric-content {
    display: flex;
    flex-direction: column;
  }

  .metric-value {
    font-size: 14px;
    font-weight: 600;
    color: var(--text-primary, #f1f5f9);
  }

  .metric-label {
    font-size: 10px;
    color: var(--text-secondary, #94a3b8);
    text-transform: uppercase;
  }

  /* Q&A Section */
  .qa-section {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .qa-toggle {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 10px 12px;
    background: var(--bg-secondary, #1e293b);
    border: none;
    border-radius: 6px;
    color: var(--text-primary, #f1f5f9);
    cursor: pointer;
    font-size: 13px;
    font-weight: 500;
    transition: all 0.2s;
  }

  .qa-toggle:hover {
    background: var(--bg-hover, #3d4a5c);
  }

  .qa-summary {
    display: flex;
    gap: 8px;
    font-size: 11px;
    font-weight: 400;
  }

  .passed-count {
    color: var(--accent-success, #10b981);
  }

  .failed-count {
    color: var(--accent-danger, #ef4444);
  }

  :global(.rotated) {
    transform: rotate(90deg);
  }

  .qa-list {
    display: flex;
    flex-direction: column;
    gap: 4px;
    max-height: 300px;
    overflow-y: auto;
    padding: 4px;
  }

  .qa-item {
    display: flex;
    gap: 8px;
    padding: 8px 10px;
    background: var(--bg-secondary, #1e293b);
    border-radius: 4px;
    border-left: 2px solid transparent;
  }

  .qa-item.failed {
    border-left-color: var(--accent-danger, #ef4444);
    background: color-mix(in srgb, var(--accent-danger, #ef4444) 10%, var(--bg-secondary));
  }

  .qa-status {
    flex-shrink: 0;
    width: 16px;
    display: flex;
    align-items: center;
    justify-content: center;
  }

  :global(.status-pass) {
    color: var(--accent-success, #10b981);
  }

  :global(.status-fail) {
    color: var(--accent-danger, #ef4444);
  }

  .status-neutral {
    color: var(--text-secondary, #94a3b8);
  }

  .qa-content {
    display: flex;
    flex-direction: column;
    gap: 2px;
    flex: 1;
    min-width: 0;
  }

  .qa-question {
    font-size: 12px;
    color: var(--text-primary, #f1f5f9);
  }

  .qa-id {
    font-weight: 600;
    color: var(--text-secondary, #94a3b8);
  }

  .qa-answer {
    font-size: 11px;
    color: var(--text-secondary, #94a3b8);
  }

  /* No Report */
  .no-report {
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 32px;
    color: var(--text-secondary, #94a3b8);
    font-size: 13px;
  }

  /* Timeline Section */
  .timeline-section {
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  .timeline-section h4 {
    margin: 0;
    font-size: 13px;
    font-weight: 600;
    color: var(--text-primary, #f1f5f9);
  }

  .timeline {
    display: flex;
    flex-direction: column;
    gap: 8px;
    padding-left: 8px;
    border-left: 2px solid var(--bg-tertiary, #334155);
  }

  .timeline-item {
    display: flex;
    gap: 12px;
    padding: 8px;
    margin-left: -10px;
  }

  .timeline-marker {
    width: 20px;
    height: 20px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: var(--bg-tertiary, #334155);
    border: 2px solid var(--stage-color);
    border-radius: 50%;
    color: var(--stage-color);
    flex-shrink: 0;
  }

  .timeline-item.current .timeline-marker {
    background: var(--stage-color);
    color: white;
  }

  .timeline-content {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .timeline-header {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .timeline-stage {
    font-size: 12px;
    font-weight: 600;
    color: var(--text-primary, #f1f5f9);
  }

  .timeline-date {
    font-size: 11px;
    color: var(--text-secondary, #94a3b8);
  }

  .timeline-metrics {
    display: flex;
    gap: 4px;
    font-size: 11px;
    color: var(--text-secondary, #94a3b8);
  }

  /* Compact Mode */
  .compact .stage-progress {
    padding: 8px 0;
  }

  .compact .stage-step {
    padding: 6px 12px;
    min-width: 60px;
  }

  .compact .step-label {
    font-size: 10px;
  }

  .compact .stage-report {
    padding: 12px;
  }

  .compact .metrics-grid {
    grid-template-columns: repeat(auto-fill, minmax(100px, 1fr));
  }
</style>
