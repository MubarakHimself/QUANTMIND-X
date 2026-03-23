<script lang="ts">
  /**
   * Pipeline Board Component
   *
   * Alpha Forge Pipeline Status Board - displays strategy runs through 9-stage pipeline.
   * Implements Frosted Terminal aesthetic with Lucide icons.
   */
  import { onMount, onDestroy } from 'svelte';
  import { alphaForgeStore, type PipelineRun, type StageStatus, type ApprovalStatus, type PipelineStage } from '$lib/stores/alpha-forge';
  import { Play, CheckCircle, XCircle, Clock, AlertCircle, RefreshCw, Kanban, Code, Cpu, Bot } from 'lucide-svelte';

  // Local component state
  let runs: PipelineRun[] = [];
  let loading = false;
  let error: string | null = null;
  let lastUpdated: string | null = null;
  let showDetails: string | null = null;

  // Subscribe to store
  const unsubscribe = alphaForgeStore.subscribe((state) => {
    runs = state.runs;
    loading = state.loading;
    error = state.error;
    lastUpdated = state.lastUpdated;
  });

  onMount(() => {
    // Start polling (5s interval)
    alphaForgeStore.startPolling(5000);
  });

  onDestroy(() => {
    alphaForgeStore.stopPolling();
    unsubscribe();
  });

  function formatTime(isoString: string | null): string {
    if (!isoString) return '--';
    try {
      const date = new Date(isoString);
      return date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
    } catch {
      return '--';
    }
  }

  function getStageDisplayName(stage: PipelineStage): string {
    const names: Record<PipelineStage, string> = {
      VIDEO_INGEST: 'Video',
      RESEARCH: 'Research',
      TRD: 'TRD',
      DEVELOPMENT: 'Dev',
      COMPILE: 'Compile',
      BACKTEST: 'Backtest',
      VALIDATION: 'Valid',
      EA_LIFECYCLE: 'EA',
      APPROVAL: 'Approval',
    };
    return names[stage] || stage;
  }

  function getStatusIcon(status: StageStatus) {
    switch (status) {
      case 'running':
        return Play;
      case 'passed':
        return CheckCircle;
      case 'failed':
        return XCircle;
      case 'waiting':
      default:
        return Clock;
    }
  }

  function getStatusColor(status: StageStatus): string {
    switch (status) {
      case 'running':
        return '#00d4ff'; // Cyan
      case 'passed':
        return '#00d4ff'; // Cyan
      case 'failed':
        return '#ff3b3b'; // Red
      case 'waiting':
      default:
        return '#6b7280'; // Gray
    }
  }

  function isApprovalPending(approvalStatus: ApprovalStatus): boolean {
    return approvalStatus === 'pending_review';
  }

  function toggleDetails(strategyId: string) {
    showDetails = showDetails === strategyId ? null : strategyId;
  }
</script>

<div class="pipeline-board">
  <header class="board-header">
    <div class="header-left">
      <Kanban size={16} />
      <h2>Pipeline Status</h2>
      <span class="active-badge">{runs.filter(r => r.stage_status === 'running').length} active</span>
    </div>
    <div class="header-right">
      {#if lastUpdated}
        <span class="last-updated">Updated {formatTime(lastUpdated)}</span>
      {/if}
      <button class="refresh-btn" onclick={() => alphaForgeStore.fetchPipelineStatus()} title="Refresh">
        <RefreshCw size={14} class={loading ? 'spinning' : ''} />
      </button>
    </div>
  </header>

  {#if error}
    <div class="error-banner">
      <AlertCircle size={14} />
      <span>{error}</span>
      <button onclick={() => alphaForgeStore.clearError()}>Dismiss</button>
    </div>
  {/if}

  <div class="board-content">
    {#if loading && runs.length === 0}
      <div class="loading-state">
        <RefreshCw size={24} class="spinning" />
        <span>Loading pipeline status...</span>
      </div>
    {:else if runs.length === 0}
      <div class="empty-state">
        <Bot size={32} />
        <p>No pipeline runs found</p>
        <span>Start a new strategy to see pipeline status</span>
      </div>
    {:else}
      <div class="pipeline-rows">
        {#each runs as run (run.strategy_id)}
          <div
            class="pipeline-row"
            class:expanded={showDetails === run.strategy_id}
            class:has-pending-approval={isApprovalPending(run.approval_status)}
          >
            <!-- Row Header -->
            <button class="row-header" onclick={() => toggleDetails(run.strategy_id)}>
              <div class="strategy-info">
                <span class="strategy-name">{run.strategy_name}</span>
                <span class="strategy-id">{run.strategy_id}</span>
              </div>

              <div class="stage-info">
                <span class="current-stage">{getStageDisplayName(run.current_stage)}</span>
                <div class="stage-status" style="color: {getStatusColor(run.stage_status)}">
                  <svelte:component this={getStatusIcon(run.stage_status)} size={14} />
                  <span>{run.stage_status}</span>
                </div>
              </div>

              {#if isApprovalPending(run.approval_status)}
                <div class="approval-badge">
                  <AlertCircle size={12} />
                  <span>Awaiting Approval</span>
                </div>
              {/if}
            </button>

            <!-- Expandable Stage Details -->
            {#if showDetails === run.strategy_id}
              <div class="stage-details">
                <div class="stages-timeline">
                  {#each run.stages as stageInfo, idx}
                    <div
                      class="stage-node"
                      class:passed={stageInfo.status === 'passed'}
                      class:running={stageInfo.status === 'running'}
                      class:failed={stageInfo.status === 'failed'}
                      class:waiting={stageInfo.status === 'waiting'}
                    >
                      <div class="stage-icon" style="color: {getStatusColor(stageInfo.status)}">
                        <svelte:component this={getStatusIcon(stageInfo.status)} size={12} />
                      </div>
                      <span class="stage-label">{getStageDisplayName(stageInfo.stage)}</span>
                      {#if stageInfo.status === 'running'}
                        <div class="running-indicator"></div>
                      {/if}
                    </div>
                    {#if idx < run.stages.length - 1}
                      <div class="stage-connector" class:completed={stageInfo.status === 'passed'}></div>
                    {/if}
                  {/each}
                </div>
                <div class="stage-timestamps">
                  <span>Started: {formatTime(run.started_at)}</span>
                  <span>Updated: {formatTime(run.updated_at)}</span>
                </div>
              </div>
            {/if}
          </div>
        {/each}
      </div>
    {/if}
  </div>
</div>

<style>
  .pipeline-board {
    display: flex;
    flex-direction: column;
    height: 100%;
    background: rgba(10, 15, 26, 0.95);
    backdrop-filter: blur(12px);
    border-radius: 8px;
    overflow: hidden;
  }

  .board-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 16px;
    border-bottom: 1px solid rgba(0, 212, 255, 0.1);
    background: rgba(8, 13, 20, 0.6);
  }

  .header-left {
    display: flex;
    align-items: center;
    gap: 8px;
    color: #a855f7;
  }

  .header-left h2 {
    font-family: 'JetBrains Mono', monospace;
    font-size: 14px;
    font-weight: 600;
    margin: 0;
    color: #e2e8f0;
  }

  .active-badge {
    padding: 2px 8px;
    background: rgba(0, 212, 255, 0.1);
    border: 1px solid rgba(0, 212, 255, 0.2);
    border-radius: 12px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: #00d4ff;
  }

  .header-right {
    display: flex;
    align-items: center;
    gap: 12px;
  }

  .last-updated {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: rgba(255, 255, 255, 0.4);
  }

  .refresh-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 28px;
    height: 28px;
    background: rgba(168, 85, 247, 0.08);
    border: 1px solid rgba(168, 85, 247, 0.2);
    border-radius: 6px;
    color: #a855f7;
    cursor: pointer;
    transition: all 0.2s ease;
  }

  .refresh-btn:hover {
    background: rgba(168, 85, 247, 0.15);
  }

  .refresh-btn :global(.spinning) {
    animation: spin 1s linear infinite;
  }

  @keyframes spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
  }

  .error-banner {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 16px;
    background: rgba(255, 59, 59, 0.1);
    border-bottom: 1px solid rgba(255, 59, 59, 0.2);
    color: #ff3b3b;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
  }

  .error-banner button {
    margin-left: auto;
    padding: 4px 8px;
    background: transparent;
    border: 1px solid rgba(255, 59, 59, 0.3);
    border-radius: 4px;
    color: #ff3b3b;
    font-size: 11px;
    cursor: pointer;
  }

  .board-content {
    flex: 1;
    overflow-y: auto;
    padding: 16px;
  }

  .loading-state,
  .empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 200px;
    color: rgba(255, 255, 255, 0.4);
    font-family: 'JetBrains Mono', monospace;
    gap: 12px;
  }

  .empty-state p {
    margin: 0;
    font-size: 14px;
    color: rgba(255, 255, 255, 0.6);
  }

  .empty-state span {
    font-size: 12px;
  }

  .pipeline-rows {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .pipeline-row {
    background: rgba(15, 23, 42, 0.6);
    border: 1px solid rgba(0, 212, 255, 0.1);
    border-radius: 8px;
    overflow: hidden;
    transition: all 0.2s ease;
  }

  .pipeline-row:hover {
    border-color: rgba(0, 212, 255, 0.2);
  }

  .pipeline-row.has-pending-approval {
    border-color: rgba(255, 170, 0, 0.3);
  }

  .row-header {
    display: flex;
    align-items: center;
    width: 100%;
    padding: 12px 16px;
    background: transparent;
    border: none;
    cursor: pointer;
    text-align: left;
    gap: 16px;
  }

  .strategy-info {
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 2px;
  }

  .strategy-name {
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    font-weight: 500;
    color: #e2e8f0;
  }

  .strategy-id {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: rgba(255, 255, 255, 0.4);
  }

  .stage-info {
    display: flex;
    align-items: center;
    gap: 12px;
  }

  .current-stage {
    padding: 4px 10px;
    background: rgba(168, 85, 247, 0.1);
    border: 1px solid rgba(168, 85, 247, 0.2);
    border-radius: 4px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: #a855f7;
  }

  .stage-status {
    display: flex;
    align-items: center;
    gap: 4px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    text-transform: capitalize;
  }

  .approval-badge {
    display: flex;
    align-items: center;
    gap: 4px;
    padding: 4px 8px;
    background: rgba(255, 170, 0, 0.1);
    border: 1px solid rgba(255, 170, 0, 0.3);
    border-radius: 4px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: #ffaa00;
  }

  .stage-details {
    padding: 16px;
    border-top: 1px solid rgba(0, 212, 255, 0.1);
    background: rgba(8, 13, 20, 0.4);
  }

  .stages-timeline {
    display: flex;
    align-items: center;
    gap: 4px;
    margin-bottom: 12px;
  }

  .stage-node {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 4px;
    position: relative;
  }

  .stage-icon {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 24px;
    height: 24px;
    background: rgba(15, 23, 42, 0.8);
    border-radius: 50%;
    border: 2px solid currentColor;
  }

  .stage-node.running .stage-icon {
    animation: pulse 1.5s ease-in-out infinite;
  }

  @keyframes pulse {
    0%, 100% { box-shadow: 0 0 0 0 currentColor; }
    50% { box-shadow: 0 0 8px 2px currentColor; }
  }

  .running-indicator {
    position: absolute;
    top: 0;
    width: 6px;
    height: 6px;
    background: #00d4ff;
    border-radius: 50%;
    animation: blink 1s ease-in-out infinite;
  }

  @keyframes blink {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
  }

  .stage-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px;
    color: rgba(255, 255, 255, 0.5);
    white-space: nowrap;
  }

  .stage-node.passed .stage-label,
  .stage-node.running .stage-label {
    color: rgba(255, 255, 255, 0.8);
  }

  .stage-connector {
    flex: 1;
    height: 2px;
    min-width: 20px;
    background: rgba(107, 114, 128, 0.3);
    margin: 0 2px;
  }

  .stage-connector.completed {
    background: rgba(0, 212, 255, 0.5);
  }

  .stage-timestamps {
    display: flex;
    justify-content: space-between;
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: rgba(255, 255, 255, 0.4);
  }
</style>