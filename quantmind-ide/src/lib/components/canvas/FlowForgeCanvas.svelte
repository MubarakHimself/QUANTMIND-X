<script lang="ts">
  /**
   * FlowForge Canvas
   *
   * Canvas 9: Prefect workflow Kanban with node graph viewer.
   * Story 11.5: FlowForge Canvas — Prefect Kanban & Node Graph
   * Story 11.6: Builder tab — workflow template library + editor
   * Story 11.7: Workflows tab redesigned — Workflow Launcher + Job Tracker
   *
   * Features:
   * - 6-column Kanban: PENDING, RUNNING, PENDING_REVIEW, DONE, CANCELLED, EXPIRED_REVIEW
   * - Per-card workflow kill switch
   * - Node graph viewer with zoom/pan
   * - Builder tab: template library + code editor
   * - Workflows tab: AlphaForge launcher + active job tracker + pipeline stage indicator
   */

  import { onMount } from 'svelte';
  import {
    flowForgeStore,
    KANBAN_COLUMNS,
    type PrefectWorkflow,
    type WorkflowState,
    type WorkflowsByState,
  } from '$lib/stores/flowforge';
  import { canvasContextService } from '$lib/services/canvasContextService';
  import PrefectKanbanCard from '$lib/components/flowforge/PrefectKanbanCard.svelte';
  import FlowForgeNodeGraph from '$lib/components/flowforge/FlowForgeNodeGraph.svelte';
  import WorkflowKillSwitchModal from '$lib/components/flowforge/WorkflowKillSwitchModal.svelte';
  import { RefreshCw, GitBranch, Search, ArrowLeft, Layers, Code2, LayoutGrid, Play, X, CheckCircle, AlertCircle, Loader2, Clock } from 'lucide-svelte';
  import DepartmentKanban from '$lib/components/department-kanban/DepartmentKanban.svelte';
  import DeptKanbanTile from '$lib/components/shared/DeptKanbanTile.svelte';
  import { submitVideoJob, getJobStatus, getAuthStatus } from '$lib/api/videoIngestApi';
  import { activeCanvasStore } from '$lib/stores/canvasStore';
  import { API_CONFIG } from '$lib/config/api';

  // =============================================================================
  // Prefect store subscriptions (PRESERVED — do not remove)
  // =============================================================================
  let showNodeGraph = $derived($flowForgeStore.showNodeGraph);
  let selectedWorkflowForNodeGraph = $derived($flowForgeStore.selectedWorkflowForNodeGraph);
  let loading = $derived($flowForgeStore.loading);
  let workflowBoard = $state<WorkflowsByState>({
    PENDING: [],
    RUNNING: [],
    PENDING_REVIEW: [],
    DONE: [],
    CANCELLED: [],
    EXPIRED_REVIEW: [],
  });
  let workflowBoardTotal = $derived(
    KANBAN_COLUMNS.reduce((count, column) => count + (workflowBoard[column.id]?.length ?? 0), 0)
  );
  let workflowBoardLoading = $state(false);
  let workflowBoardError = $state<string | null>(null);

  // Sub-page routing — two tabs (Floor Manager uses shell AgentPanel on right sidebar)
  type FlowForgeSubPage = 'prefect' | 'dept-kanban';
  let activeTab = $state<FlowForgeSubPage>('prefect');

  // Modal state
  let showKillSwitchModal = $state(false);
  let workflowToCancel = $state<PrefectWorkflow | null>(null);

  // Refresh interval
  let refreshInterval: ReturnType<typeof setInterval> | null = null;

  // =============================================================================
  // Workflow Launcher + Job Tracker — state (Svelte 5 runes)
  // =============================================================================
  interface VideoJob {
    id: string;
    url: string;
    title?: string;
    status: 'PENDING' | 'DOWNLOADING' | 'PROCESSING' | 'ANALYZING' | 'COMPLETED' | 'FAILED';
    progress?: number;
    error?: string;
    submittedAt: Date;
    alphaForgeStage?: 'video' | 'research' | 'development' | 'compile' | 'backtest' | 'done';
    source?: 'manual' | 'scheduled';
    hypotheses_count?: number;
    dispatched_count?: number;
  }

  let launchUrl = $state('');
  let launching = $state(false);
  let launchError = $state<string | null>(null);
  let jobs = $state<VideoJob[]>([]);
  let authStatus = $state<'checking' | 'ready' | 'unconfigured' | 'unreachable'>('checking');

  // Poll interval handle
  let jobPollInterval: ReturnType<typeof setInterval> | null = null;

  // =============================================================================
  // Autonomous Scheduler — state (Svelte 5 runes)
  // =============================================================================
  interface SchedulerStatus {
    enabled: boolean;
    schedule_time: string;
    next_run_iso: string | null;
    last_run: { status: string; completed_at: string; hypotheses_count: number; dispatched_count: number } | null;
    status: 'idle' | 'running' | 'failed';
  }

  let schedulerStatus = $state<SchedulerStatus | null>(null);
  let schedulerLoading = $state(false);
  let schedulerTriggering = $state(false);
  let schedulerPollInterval: ReturnType<typeof setInterval> | null = null;

  function getFlowForgeApiUrl(path: string): string {
    return `${API_CONFIG.API_URL}${path}`;
  }

  // Derived: whether any batch has jobs in COMPLETED state
  let anyBatchComplete = $derived(jobs.some(j => j.status === 'COMPLETED'));

  // Derived: whether URL looks like a playlist
  let isPlaylistUrl = $derived(launchUrl.includes('playlist') || launchUrl.includes('list='));

  $effect(() => {
    const workflowResources = KANBAN_COLUMNS.flatMap((column) =>
      (workflowBoard[column.id] ?? []).map((workflow) => ({
        id: workflow.id,
        label: workflow.name,
        canvas: 'flowforge',
        resource_type: 'workflow-card',
        metadata: {
          state: workflow.state,
          column: column.id,
          department: workflow.department,
          started_at: workflow.started_at,
        },
      })),
    ).slice(0, 100);

    const jobResources = jobs.slice(0, 50).map((job) => ({
      id: job.id,
      label: job.title || job.id,
      canvas: 'flowforge',
      resource_type: 'video-job',
      metadata: {
        status: job.status,
        progress: job.progress,
        alphaForgeStage: job.alphaForgeStage,
        source: job.source,
      },
    }));

    canvasContextService.setRuntimeState('flowforge', {
      active_tab: activeTab,
      workflow_board_total: workflowBoardTotal,
      workflow_board_loading: workflowBoardLoading,
      workflow_board_error: workflowBoardError,
      scheduler_status: schedulerStatus,
      attachable_resources: [...workflowResources, ...jobResources],
    });
  });

  // =============================================================================
  // onMount — canvas init + Prefect auto-refresh + auth check + job poll
  // =============================================================================
  onMount(() => {
    canvasContextService.loadCanvasContext('flowforge')
      .then(() => Promise.allSettled([flowForgeStore.fetchWorkflows(), loadWorkflowBoard()]))
      .catch((e) => console.error('Failed to load FlowForge canvas:', e));

    // Auto-refresh Prefect workflows every 30 seconds
    refreshInterval = setInterval(() => {
      flowForgeStore.fetchWorkflows();
      loadWorkflowBoard();
    }, 30000);

    // Check video ingest auth on mount
    checkVideoIngestAuth();

    // Poll active (non-terminal) jobs every 5 seconds
    jobPollInterval = setInterval(() => {
      pollActiveJobs();
    }, 5000);

    loadSchedulerStatus();
    loadScheduledRuns();
    schedulerPollInterval = setInterval(() => {
      loadSchedulerStatus();
      loadScheduledRuns();
    }, 30000);

    return function cleanup() {
      if (refreshInterval) clearInterval(refreshInterval);
      if (jobPollInterval) clearInterval(jobPollInterval);
      if (schedulerPollInterval) clearInterval(schedulerPollInterval);
    };
  });

  // =============================================================================
  // Video ingest auth check
  // =============================================================================
  async function checkVideoIngestAuth() {
    authStatus = 'checking';
    try {
      const status = await getAuthStatus();
      authStatus = (status.openrouter || status.gemini || status.qwen) ? 'ready' : 'unconfigured';
    } catch {
      authStatus = 'unreachable';
    }
  }

  // =============================================================================
  // Autonomous Scheduler — functions
  // =============================================================================
  async function loadSchedulerStatus() {
    try {
      const res = await fetch(getFlowForgeApiUrl('/api/autonomous-scheduler/status'));
      if (res.ok) schedulerStatus = await res.json();
    } catch { /* scheduler offline — graceful */ }
  }

  async function loadScheduledRuns() {
    try {
      const res = await fetch(getFlowForgeApiUrl('/api/autonomous-scheduler/runs'));
      if (!res.ok) return;
      const runs = await res.json();
      // Convert to VideoJob format, only add ones not already in jobs list
      const existingIds = new Set(jobs.map(j => j.id));
      const newJobs: VideoJob[] = runs
        .filter((r: any) => !existingIds.has(r.job_id))
        .map((r: any) => ({
          id: r.job_id,
          url: '',
          title: r.title,
          status: r.status as VideoJob['status'],
          alphaForgeStage: r.alphaForgeStage as VideoJob['alphaForgeStage'],
          submittedAt: new Date(r.submittedAt),
          source: 'scheduled' as const,
          hypotheses_count: r.hypotheses_count,
          dispatched_count: r.dispatched_count,
        }));
      if (newJobs.length > 0) jobs = [...newJobs, ...jobs];
    } catch { /* scheduler offline */ }
  }

  async function handleSchedulerToggle() {
    if (!schedulerStatus) return;
    schedulerLoading = true;
    try {
      await fetch(getFlowForgeApiUrl('/api/autonomous-scheduler/config'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ enabled: !schedulerStatus.enabled }),
      });
      await loadSchedulerStatus();
    } catch {} finally { schedulerLoading = false; }
  }

  async function handleSchedulerTrigger() {
    schedulerTriggering = true;
    try {
      const res = await fetch(getFlowForgeApiUrl('/api/autonomous-scheduler/trigger'), { method: 'POST' });
      if (res.ok) {
        await loadScheduledRuns();
        await loadSchedulerStatus();
      }
    } catch {} finally { schedulerTriggering = false; }
  }

  function formatNextRun(iso: string | null): string {
    if (!iso) return 'Not scheduled';
    const d = new Date(iso);
    return d.toLocaleString('en-US', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit', timeZoneName: 'short' });
  }

  // =============================================================================
  // Launch workflow
  // =============================================================================
  async function handleLaunch() {
    if (!launchUrl.trim() || launching || authStatus !== 'ready') return;
    launchError = null;
    launching = true;
    try {
      const isPlaylist = isPlaylistUrl;
      const response = await submitVideoJob(launchUrl.trim(), undefined, isPlaylist);
      const submittedAt = new Date();
      const jobIds = response.job_ids?.length ? response.job_ids : [response.job_id];
      const newJobs: VideoJob[] = jobIds.map((jobId, index) => ({
        id: jobId,
        url: launchUrl.trim(),
        title: jobIds.length > 1 ? `Playlist item ${index + 1}` : undefined,
        status: 'PENDING',
        progress: 0,
        submittedAt: new Date(submittedAt.getTime() + index),
        alphaForgeStage: 'video',
      }));
      jobs = [...newJobs, ...jobs];
      launchUrl = '';
    } catch (err: unknown) {
      launchError = err instanceof Error ? err.message : 'Failed to submit job. Check backend connection.';
    } finally {
      launching = false;
    }
  }

  function handleLaunchKeydown(e: KeyboardEvent) {
    if (e.key === 'Enter') handleLaunch();
  }

  // =============================================================================
  // Job polling
  // =============================================================================
  async function pollActiveJobs() {
    const activeJobs = jobs.filter(j => j.status !== 'COMPLETED' && j.status !== 'FAILED');
    if (activeJobs.length === 0) return;

    const updates = await Promise.allSettled(
      activeJobs.map(j => getJobStatus(j.id))
    );

    jobs = jobs.map((job, _idx) => {
      const activeIdx = activeJobs.findIndex(aj => aj.id === job.id);
      if (activeIdx === -1) return job;
      const result = updates[activeIdx];
      if (result.status !== 'fulfilled') return job;
      const statusData = result.value;
      const rawStatus = (statusData.current_stage || statusData.status || 'PENDING').toUpperCase();
      const mappedStatus = mapBackendStatus(rawStatus);
      const updated: VideoJob = {
        ...job,
        status: mappedStatus,
        progress: statusData.progress ?? job.progress,
        error: statusData.error,
      };
      if (mappedStatus === 'COMPLETED') {
        updated.alphaForgeStage = 'research';
      }
      return updated;
    });
  }

  function mapBackendStatus(raw: string): VideoJob['status'] {
    if (raw === 'COMPLETED' || raw === 'COMPLETE' || raw === 'DONE') return 'COMPLETED';
    if (raw === 'FAILED' || raw === 'ERROR') return 'FAILED';
    if (raw === 'DOWNLOADING' || raw === 'DOWNLOAD') return 'DOWNLOADING';
    if (raw === 'PROCESSING' || raw === 'PROCESS') return 'PROCESSING';
    if (raw === 'ANALYZING' || raw === 'ANALYSE' || raw === 'ANALYZE') return 'ANALYZING';
    return 'PENDING';
  }

  // =============================================================================
  // Cancel a job
  // =============================================================================
  function handleCancelJob(jobId: string) {
    // Mark as FAILED locally; TODO: wire to backend cancel endpoint
    jobs = jobs.map(j => j.id === jobId ? { ...j, status: 'FAILED', error: 'Cancelled by user' } : j);
  }

  // =============================================================================
  // Navigate to canvas based on AlphaForge stage
  // =============================================================================
  function handleJobClick(job: VideoJob) {
    if (job.alphaForgeStage === 'research') {
      activeCanvasStore.setActiveCanvas('research');
    } else if (job.alphaForgeStage === 'development' || job.alphaForgeStage === 'compile' || job.alphaForgeStage === 'backtest') {
      activeCanvasStore.setActiveCanvas('development');
    }
  }

  // =============================================================================
  // Prefect handlers (PRESERVED)
  // =============================================================================
  async function handleRefresh() {
    await Promise.allSettled([flowForgeStore.fetchWorkflows(), loadWorkflowBoard()]);
  }

  function handleWorkflowClick(workflow: PrefectWorkflow) {
    flowForgeStore.openNodeGraph(workflow);
  }

  function handleKillSwitch(workflow: PrefectWorkflow) {
    workflowToCancel = workflow;
    showKillSwitchModal = true;
  }

  async function confirmCancellation() {
    if (!workflowToCancel) return;
    const success = await flowForgeStore.cancelWorkflow(workflowToCancel.id);
    if (success) {
      showKillSwitchModal = false;
      workflowToCancel = null;
    }
  }

  function closeKillSwitchModal() {
    showKillSwitchModal = false;
    workflowToCancel = null;
  }

  function closeNodeGraph() {
    flowForgeStore.closeNodeGraph();
  }

  function getColumnIcon(state: WorkflowState): string {
    const icons: Record<WorkflowState, string> = {
      PENDING: 'clock',
      RUNNING: 'play',
      PENDING_REVIEW: 'eye',
      DONE: 'check',
      CANCELLED: 'x',
      EXPIRED_REVIEW: 'alert',
    };
    return icons[state];
  }

  async function loadWorkflowBoard() {
    workflowBoardLoading = true;
    workflowBoardError = null;
    try {
      const response = await fetch(getFlowForgeApiUrl('/api/prefect/workflows'));
      if (!response.ok) throw new Error('Failed to fetch workflows');
      const data = await response.json();
      workflowBoard = {
        PENDING: data.by_state?.PENDING ?? [],
        RUNNING: data.by_state?.RUNNING ?? [],
        PENDING_REVIEW: data.by_state?.PENDING_REVIEW ?? [],
        DONE: data.by_state?.DONE ?? [],
        CANCELLED: data.by_state?.CANCELLED ?? [],
        EXPIRED_REVIEW: data.by_state?.EXPIRED_REVIEW ?? [],
      };
    } catch (err: unknown) {
      workflowBoardError = err instanceof Error ? err.message : 'Failed to fetch workflows';
    } finally {
      workflowBoardLoading = false;
    }
  }

  // =============================================================================
  // UI helpers — job display
  // =============================================================================
  function formatTimestamp(date: Date): string {
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  }

  function truncateUrl(url: string, max = 48): string {
    return url.length > max ? url.slice(0, max) + '…' : url;
  }

  function jobStatusLabel(status: VideoJob['status']): string {
    const labels: Record<VideoJob['status'], string> = {
      PENDING: 'Pending',
      DOWNLOADING: 'Downloading',
      PROCESSING: 'Processing',
      ANALYZING: 'Analyzing',
      COMPLETED: 'Done',
      FAILED: 'Failed',
    };
    return labels[status];
  }

  const PIPELINE_STAGES: Array<{ key: VideoJob['alphaForgeStage']; label: string }> = [
    { key: 'video', label: 'VIDEO' },
    { key: 'research', label: 'RESEARCH' },
    { key: 'development', label: 'DEVELOPMENT' },
    { key: 'compile', label: 'COMPILE' },
    { key: 'backtest', label: 'BACKTEST' },
  ];

  function getStageState(job: VideoJob, stageKey: VideoJob['alphaForgeStage']): 'done' | 'active' | 'pending' {
    const order: Array<VideoJob['alphaForgeStage']> = ['video', 'research', 'development', 'compile', 'backtest', 'done'];
    const currentIdx = order.indexOf(job.alphaForgeStage ?? 'video');
    const stageIdx = order.indexOf(stageKey);
    if (stageIdx < currentIdx) return 'done';
    if (stageIdx === currentIdx) {
      return job.status === 'COMPLETED' && stageKey === 'video' ? 'done' : 'active';
    }
    return 'pending';
  }

</script>

<div class="flowforge-canvas" data-dept="flowforge">
  {#if activeTab === 'dept-kanban'}
    <!-- Department Kanban Sub-Page (Story 12-6) — distinct from Prefect Kanban (AC 12-6-8) -->
    <div class="dept-kanban-header">
      <button class="back-btn" onclick={() => activeTab = 'prefect'} title="Back to FlowForge">
        <ArrowLeft size={14} />
        <span>Back</span>
      </button>
    </div>
    <DepartmentKanban department="flowforge" onClose={() => activeTab = 'prefect'} />
  {:else}
    <!-- Prefect / Workflows / Builder views share the header -->
    <div class="canvas-header">
      <div class="header-left">
        <GitBranch size={22} />
        <div class="header-title">
          <h2>FlowForge</h2>
          <span class="subtitle">Global Workflow Orchestration &amp; Floor Manager</span>
        </div>
      </div>
      <div class="header-right">
        <!-- AC 12-6-8: DeptKanbanTile in header-right — separate from Prefect Kanban -->
        <DeptKanbanTile dept="flowforge" onNavigate={() => activeTab = 'dept-kanban'} />
        {#if activeTab === 'prefect'}
          <button class="refresh-btn" onclick={handleRefresh} disabled={loading}>
            <span class="icon-wrapper" class:spinning={loading}>
              <RefreshCw size={16} />
            </span>
            <span>Refresh</span>
          </button>
        {/if}
      </div>
    </div>

    <!-- Tab Nav Strip -->
    <div class="tab-nav">
      <button
        class="tab-btn"
        class:active={activeTab === 'prefect'}
        onclick={() => activeTab = 'prefect'}
      >
        <LayoutGrid size={14} />
        <span>Workflows</span>
      </button>
      <button
        class="tab-btn"
        class:active={activeTab === 'dept-kanban'}
        onclick={() => activeTab = 'dept-kanban'}
      >
        <Layers size={14} />
        <span>Dept Tasks</span>
      </button>
    </div>

    {#if activeTab === 'prefect'}
      <!-- ================================================================
           Workflows Tab — Workflow Launcher + Job Tracker
           ================================================================ -->

      <!-- Auth warning banner -->
      {#if authStatus === 'unconfigured'}
        <div class="auth-warning-banner">
          <AlertCircle size={14} />
          <span>Video ingest provider not configured — check Settings → Providers</span>
        </div>
      {:else if authStatus === 'unreachable'}
        <div class="auth-warning-banner">
          <AlertCircle size={14} />
          <span>Video ingest backend unavailable — check API routing and server health</span>
        </div>
      {/if}

      <div class="workflows-tab-body">
        <!-- ── Section 1: AlphaForge Launcher ────────────────────────────── -->
        <section class="launcher-section">
          <div class="launcher-header">
            <Play size={14} class="launcher-icon" />
            <span class="launcher-label">AlphaForge — Workflow 1</span>
            {#if authStatus === 'checking'}
              <span class="auth-badge auth-checking">
                <Loader2 size={11} />
                <span>Checking auth…</span>
              </span>
            {:else if authStatus === 'ready'}
              <span class="auth-badge auth-ready">
                <CheckCircle size={11} />
                <span>Provider ready</span>
              </span>
            {:else if authStatus === 'unconfigured'}
              <span class="auth-badge auth-error">
                <AlertCircle size={11} />
                <span>No provider configured</span>
              </span>
            {:else if authStatus === 'unreachable'}
              <span class="auth-badge auth-error">
                <AlertCircle size={11} />
                <span>Backend unavailable</span>
              </span>
            {/if}
          </div>

          <div class="launcher-input-row">
            <input
              class="url-input"
              type="text"
              placeholder="YouTube video or playlist URL…"
              bind:value={launchUrl}
              onkeydown={handleLaunchKeydown}
              disabled={launching || authStatus !== 'ready'}
            />
            <button
              class="launch-btn"
              onclick={handleLaunch}
              disabled={!launchUrl.trim() || launching || authStatus !== 'ready'}
            >
              {#if launching}
                <span class="btn-spinner"><Loader2 size={14} /></span>
                <span>Submitting…</span>
              {:else}
                <Play size={14} />
                <span>Launch</span>
              {/if}
            </button>
          </div>

          {#if isPlaylistUrl && launchUrl.trim()}
            <p class="playlist-note">
              Playlist detected — each video will be tracked separately.
            </p>
          {/if}

          {#if launchError}
            <p class="launch-error">
              <AlertCircle size={12} />
              {launchError}
            </p>
          {/if}
        </section>

        <!-- ── Autonomous Scheduler Control ──────────────────────────────── -->
        <section class="scheduler-section">
          <div class="scheduler-header">
            <Clock size={14} class="scheduler-icon" />
            <span class="scheduler-label">Autonomous Scheduler</span>
            {#if schedulerStatus}
              <button
                class="scheduler-toggle"
                class:enabled={schedulerStatus.enabled}
                onclick={handleSchedulerToggle}
                disabled={schedulerLoading}
                title={schedulerStatus.enabled ? 'Disable scheduler' : 'Enable scheduler'}
              >
                <span class="toggle-pill"></span>
              </button>
              <button
                class="run-now-btn"
                onclick={handleSchedulerTrigger}
                disabled={schedulerTriggering || schedulerStatus.status === 'running'}
              >
                {#if schedulerTriggering}
                  <Loader2 size={12} class="spin-icon" />
                  Running…
                {:else}
                  <Play size={12} />
                  Run Now
                {/if}
              </button>
            {/if}
          </div>
          {#if schedulerStatus}
            <div class="scheduler-meta">
              <span class="meta-item">
                <span class="meta-key">Schedule</span>
                <span class="meta-val">{schedulerStatus.schedule_time} UTC daily</span>
              </span>
              <span class="meta-item">
                <span class="meta-key">Next run</span>
                <span class="meta-val">{formatNextRun(schedulerStatus.next_run_iso)}</span>
              </span>
              {#if schedulerStatus.last_run}
                <span class="meta-item">
                  <span class="meta-key">Last run</span>
                  <span class="meta-val">{schedulerStatus.last_run.hypotheses_count} hypotheses, {schedulerStatus.last_run.dispatched_count} dispatched</span>
                </span>
              {/if}
              {#if schedulerStatus.status === 'running'}
                <span class="scheduler-running-badge">
                  <span class="pulse-dot"></span>
                  Running…
                </span>
              {/if}
            </div>
          {:else}
            <p class="scheduler-offline">Scheduler offline — backend not connected</p>
          {/if}
        </section>

        <!-- ── Section 2: Active Jobs ─────────────────────────────────────── -->
        <section class="jobs-section">
          <div class="jobs-header">
            <span class="jobs-title">Active Jobs</span>
            <span class="jobs-count">{jobs.length}</span>
          </div>

          {#if jobs.length === 0}
            <div class="jobs-empty">
              <span>No jobs yet — submit a YouTube URL to begin.</span>
            </div>
          {:else}
            <div class="jobs-list">
              {#each jobs as job (job.id)}
                <!-- svelte-ignore a11y_click_events_have_key_events -->
                <!-- svelte-ignore a11y_no_static_element_interactions -->
                <div
                  class="job-card"
                  data-status={job.status}
                  onclick={() => handleJobClick(job)}
                  title={job.alphaForgeStage === 'research' ? 'Click to open Research canvas' : job.alphaForgeStage === 'development' ? 'Click to open Development canvas' : undefined}
                >
                  <!-- Card top row -->
                  <div class="job-top-row">
                    <span class="status-pill" data-status={job.status}>
                      {#if job.status === 'COMPLETED'}
                        <CheckCircle size={10} />
                      {:else if job.status === 'FAILED'}
                        <AlertCircle size={10} />
                      {:else if job.status === 'DOWNLOADING' || job.status === 'PROCESSING' || job.status === 'ANALYZING'}
                        <span class="pulse-dot"></span>
                      {/if}
                      {jobStatusLabel(job.status)}
                    </span>

                    {#if job.source === 'scheduled'}
                      <span class="job-source-label scheduled">
                        <Clock size={11} />
                        {job.title ?? 'Scheduled Research'}
                      </span>
                    {:else}
                      <span class="job-url">{truncateUrl(job.url)}</span>
                    {/if}

                    <span class="job-timestamp">{formatTimestamp(job.submittedAt)}</span>

                    {#if job.status === 'PENDING' || job.status === 'DOWNLOADING' || job.status === 'PROCESSING' || job.status === 'ANALYZING'}
                      <button
                        class="cancel-btn"
                        onclick={(e) => { e.stopPropagation(); handleCancelJob(job.id); }}
                        title="Cancel job"
                      >
                        <X size={12} />
                      </button>
                    {/if}
                  </div>

                  <!-- Progress bar (shown for non-terminal, non-pending states) -->
                  {#if job.status === 'DOWNLOADING' || job.status === 'PROCESSING' || job.status === 'ANALYZING'}
                    <div class="job-progress-row">
                      <div class="progress-track">
                        <div
                          class="progress-fill"
                          style="width: {job.progress ?? 0}%"
                          data-status={job.status}
                        ></div>
                      </div>
                      <span class="progress-label">{job.status} ({job.progress ?? 0}%)</span>
                    </div>
                  {/if}

                  <!-- Job ID -->
                  <div class="job-meta-row">
                    <span class="job-id">Job ID: {job.id.slice(0, 12)}…</span>
                  </div>

                  <!-- Error message for FAILED -->
                  {#if job.status === 'FAILED' && job.error}
                    <div class="job-error-msg">
                      <AlertCircle size={11} />
                      <span>{job.error}</span>
                    </div>
                  {/if}

                  <!-- Pipeline stage indicator — AlphaForge journey after video ingest -->
                  <!-- TODO: connect post-ingest stages to backend pipeline status API -->
                  <div class="pipeline-stages">
                    {#each PIPELINE_STAGES as stage, i}
                      {@const stageState = getStageState(job, stage.key)}
                      <span class="stage-item" data-state={stageState}>
                        {stage.label}
                        {#if stageState === 'done'}
                          <span class="stage-check">✓</span>
                        {:else if stageState === 'active'}
                          <span class="stage-spin">⟳</span>
                        {:else}
                          <span class="stage-dash">—</span>
                        {/if}
                      </span>
                      {#if i < PIPELINE_STAGES.length - 1}
                        <span class="stage-arrow">→</span>
                      {/if}
                    {/each}
                  </div>
                </div>
              {/each}
            </div>
          {/if}
        </section>

        <!-- ── Section 3: WF2 Banner (shown only when any batch complete) ── -->
        {#if anyBatchComplete}
          <div class="wf2-banner">
            <RefreshCw size={13} />
            <span>Improvement Loop (WF2) will trigger automatically when batch completes</span>
          </div>
        {/if}

        <section class="kanban-section">
          <div class="kanban-section-header">
            <span class="kanban-section-title">Workflow Board</span>
            <span class="kanban-section-subtitle">Real workflow runs from the backend persistence layer</span>
          </div>

          {#if workflowBoardLoading}
            <div class="kanban-status loading">
              <Loader2 size={14} />
              <span>Loading workflows…</span>
            </div>
          {:else if workflowBoardError}
            <div class="kanban-status error">
              <AlertCircle size={14} />
              <span>{workflowBoardError}</span>
            </div>
          {:else if workflowBoardTotal === 0}
            <div class="kanban-status empty">
              <span>No persisted workflows yet.</span>
            </div>
          {:else}
            <div class="kanban-grid">
              {#each KANBAN_COLUMNS as column}
                <section class="kanban-column">
                  <div class="kanban-column-header">
                    <span class="kanban-column-title">{column.label}</span>
                    <span class="kanban-column-count">{workflowBoard[column.id]?.length ?? 0}</span>
                  </div>

                  <div class="kanban-column-body">
                    {#if workflowBoard[column.id]?.length}
                      {#each workflowBoard[column.id] as workflow (workflow.id)}
                        <PrefectKanbanCard
                          workflow={workflow}
                          onClick={handleWorkflowClick}
                          onKillSwitch={column.id === 'RUNNING' ? handleKillSwitch : undefined}
                        />
                      {/each}
                    {:else}
                      <div class="kanban-column-empty">No workflows</div>
                    {/if}
                  </div>
                </section>
              {/each}
            </div>
          {/if}
        </section>
      </div>

      <!-- Node Graph Modal — stays on Workflows tab -->
      {#if showNodeGraph && selectedWorkflowForNodeGraph}
        <FlowForgeNodeGraph
          workflow={selectedWorkflowForNodeGraph}
          onClose={closeNodeGraph}
        />
      {/if}

      <!-- Kill Switch Confirmation Modal -->
      {#if showKillSwitchModal && workflowToCancel}
        <WorkflowKillSwitchModal
          workflow={workflowToCancel}
          onConfirm={confirmCancellation}
          onCancel={closeKillSwitchModal}
        />
      {/if}
    {/if}
  {/if}
</div>

<style>
  .flowforge-canvas {
    display: flex;
    flex-direction: column;
    height: 100%;
    width: 100%;
    min-width: 0;
    background: linear-gradient(180deg, rgba(20, 22, 28, 0.95) 0%, rgba(15, 17, 22, 0.98) 100%);
  }

  /* When DepartmentKanban is a direct child (dept-kanban sub-page), let it expand */
  .flowforge-canvas > :global(.department-kanban) {
    flex: 1;
    width: 100%;
    min-width: 0;
  }

  /* Dept Kanban sub-page header — back button (AC 12-6-4) */
  .dept-kanban-header {
    display: flex;
    align-items: center;
    padding: var(--space-3, 12px) var(--space-4, 16px);
    background: var(--glass-shell-bg, rgba(30, 32, 40, 0.6));
    border-bottom: 1px solid var(--color-border-subtle, rgba(255, 255, 255, 0.06));
    flex-shrink: 0;
  }

  .back-btn {
    display: flex;
    align-items: center;
    gap: var(--space-2, 8px);
    padding: var(--space-2, 6px) var(--space-3, 12px);
    background: var(--glass-content-bg, rgba(255, 255, 255, 0.04));
    border: 1px solid var(--color-border-subtle, rgba(255, 255, 255, 0.08));
    border-radius: 6px;
    color: var(--dept-accent, #06b6d4);
    font-family: var(--font-ambient, 'JetBrains Mono', monospace);
    font-size: var(--text-xs, 12px);
    cursor: pointer;
    transition: background 0.15s ease, border-color 0.15s ease;
  }

  .back-btn:hover {
    background: var(--glass-content-bg-hover, rgba(255, 255, 255, 0.08));
    border-color: var(--color-border-active, rgba(255, 255, 255, 0.15));
  }

  /* Header */
  .canvas-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 16px 20px;
    background: rgba(30, 32, 40, 0.6);
    border-bottom: 1px solid rgba(255, 255, 255, 0.06);
    flex-shrink: 0;
  }

  .header-left {
    display: flex;
    align-items: center;
    gap: 12px;
    color: #06b6d4;
  }

  .header-title h2 {
    margin: 0;
    font-size: 18px;
    font-weight: 600;
    color: #f1f5f9;
  }

  .header-title .subtitle {
    font-size: 12px;
    color: #64748b;
  }

  .header-right {
    display: flex;
    gap: 12px;
    align-items: center;
  }

  .refresh-btn {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 14px;
    background: rgba(6, 182, 212, 0.15);
    border: 1px solid rgba(6, 182, 212, 0.3);
    border-radius: 6px;
    color: #06b6d4;
    font-size: 13px;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.2s;
  }

  .refresh-btn:hover:not(:disabled) {
    background: rgba(6, 182, 212, 0.25);
    border-color: rgba(6, 182, 212, 0.5);
  }

  .refresh-btn:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }

  .icon-wrapper.spinning {
    animation: spin 1s linear infinite;
  }

  @keyframes spin {
    from { transform: rotate(0deg); }
    to   { transform: rotate(360deg); }
  }

  /* Tab Nav Strip */
  .tab-nav {
    display: flex;
    align-items: center;
    gap: 4px;
    padding: 8px 20px;
    background: rgba(20, 22, 28, 0.7);
    border-bottom: 1px solid rgba(255, 255, 255, 0.06);
    flex-shrink: 0;
    backdrop-filter: blur(8px);
  }

  .tab-btn {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 6px 14px;
    background: transparent;
    border: 1px solid transparent;
    border-radius: 6px;
    color: #64748b;
    font-family: var(--font-ambient, 'JetBrains Mono', monospace);
    font-size: 12px;
    font-weight: 500;
    cursor: pointer;
    transition: background 0.15s ease, color 0.15s ease, border-color 0.15s ease;
  }

  .tab-btn:hover {
    background: rgba(255, 255, 255, 0.05);
    color: #94a3b8;
    border-color: rgba(255, 255, 255, 0.08);
  }

  .tab-btn.active {
    background: rgba(6, 182, 212, 0.12);
    border-color: rgba(6, 182, 212, 0.3);
    color: #06b6d4;
  }

  /* =========================================================================
     Auth warning banner
     ========================================================================= */
  .auth-warning-banner {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 9px 20px;
    background: rgba(240, 165, 0, 0.12);
    border-bottom: 1px solid rgba(240, 165, 0, 0.25);
    color: #f0a500;
    font-size: 12px;
    font-family: var(--font-ambient, 'JetBrains Mono', monospace);
    flex-shrink: 0;
  }

  /* =========================================================================
     Workflows tab body — scrollable container
     ========================================================================= */
  .workflows-tab-body {
    flex: 1;
    overflow-y: auto;
    padding: 20px;
    display: flex;
    flex-direction: column;
    gap: 20px;
  }

  /* =========================================================================
     Section 1: AlphaForge Launcher
     ========================================================================= */
  .launcher-section {
    background: rgba(8, 13, 20, 0.92);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(0, 212, 255, 0.18);
    border-radius: 10px;
    padding: 16px 18px;
    display: flex;
    flex-direction: column;
    gap: 10px;
  }

  .launcher-header {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .launcher-header :global(svg) {
    color: #00d4ff;
    flex-shrink: 0;
  }

  .launcher-label {
    font-family: var(--font-ambient, 'JetBrains Mono', monospace);
    font-size: 13px;
    font-weight: 600;
    color: #00d4ff;
    letter-spacing: 0.4px;
  }

  .auth-badge {
    display: flex;
    align-items: center;
    gap: 5px;
    padding: 3px 8px;
    border-radius: 20px;
    font-size: 11px;
    font-family: var(--font-ambient, 'JetBrains Mono', monospace);
    margin-left: auto;
  }

  .auth-badge.auth-checking {
    background: rgba(100, 116, 139, 0.15);
    color: #64748b;
    border: 1px solid rgba(100, 116, 139, 0.2);
  }

  .auth-badge.auth-ready {
    background: rgba(0, 200, 150, 0.12);
    color: #00c896;
    border: 1px solid rgba(0, 200, 150, 0.2);
  }

  .auth-badge.auth-error {
    background: rgba(240, 165, 0, 0.12);
    color: #f0a500;
    border: 1px solid rgba(240, 165, 0, 0.2);
  }

  .launcher-input-row {
    display: flex;
    gap: 10px;
  }

  .url-input {
    flex: 1;
    padding: 9px 13px;
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 7px;
    color: #e2e8f0;
    font-family: var(--font-ambient, 'JetBrains Mono', monospace);
    font-size: 12px;
    outline: none;
    transition: border-color 0.15s ease;
  }

  .url-input::placeholder {
    color: #475569;
  }

  .url-input:focus {
    border-color: rgba(0, 212, 255, 0.4);
    background: rgba(0, 212, 255, 0.04);
  }

  .url-input:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .launch-btn {
    display: flex;
    align-items: center;
    gap: 7px;
    padding: 9px 18px;
    background: rgba(0, 212, 255, 0.18);
    border: 1px solid rgba(0, 212, 255, 0.4);
    border-radius: 7px;
    color: #00d4ff;
    font-family: var(--font-ambient, 'JetBrains Mono', monospace);
    font-size: 12px;
    font-weight: 600;
    cursor: pointer;
    transition: background 0.15s ease, border-color 0.15s ease;
    white-space: nowrap;
    flex-shrink: 0;
  }

  .launch-btn:hover:not(:disabled) {
    background: rgba(0, 212, 255, 0.28);
    border-color: rgba(0, 212, 255, 0.6);
  }

  .launch-btn:disabled {
    opacity: 0.45;
    cursor: not-allowed;
  }

  .btn-spinner {
    display: flex;
    animation: spin 1s linear infinite;
  }

  .playlist-note {
    margin: 0;
    font-size: 11px;
    font-family: var(--font-ambient, 'JetBrains Mono', monospace);
    color: #94a3b8;
    padding: 6px 10px;
    background: rgba(255, 255, 255, 0.04);
    border-left: 2px solid rgba(0, 212, 255, 0.4);
    border-radius: 0 4px 4px 0;
  }

  .launch-error {
    display: flex;
    align-items: center;
    gap: 6px;
    margin: 0;
    font-size: 12px;
    font-family: var(--font-ambient, 'JetBrains Mono', monospace);
    color: #ff3b3b;
  }

  /* =========================================================================
     Section 2: Active Jobs
     ========================================================================= */
  .jobs-section {
    display: flex;
    flex-direction: column;
    gap: 10px;
  }

  .jobs-header {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .jobs-title {
    font-family: var(--font-ambient, 'JetBrains Mono', monospace);
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: #64748b;
  }

  .jobs-count {
    font-size: 11px;
    color: #64748b;
    background: rgba(255, 255, 255, 0.06);
    padding: 1px 7px;
    border-radius: 10px;
    font-family: var(--font-ambient, 'JetBrains Mono', monospace);
  }

  .jobs-empty {
    padding: 24px 16px;
    text-align: center;
    font-size: 12px;
    font-family: var(--font-ambient, 'JetBrains Mono', monospace);
    color: #475569;
    background: rgba(8, 13, 20, 0.5);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 8px;
  }

  .jobs-list {
    display: flex;
    flex-direction: column;
    gap: 10px;
  }

  /* Job Card */
  .job-card {
    background: rgba(8, 13, 20, 0.92);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 9px;
    padding: 12px 14px;
    display: flex;
    flex-direction: column;
    gap: 8px;
    cursor: default;
    transition: border-color 0.15s ease, background 0.15s ease;
  }

  .job-card[data-status="COMPLETED"] {
    border-color: rgba(0, 200, 150, 0.2);
  }

  .job-card[data-status="FAILED"] {
    border-color: rgba(255, 59, 59, 0.2);
  }

  .job-card[data-status="DOWNLOADING"],
  .job-card[data-status="PROCESSING"],
  .job-card[data-status="ANALYZING"] {
    border-color: rgba(0, 212, 255, 0.15);
  }

  /* Clickable cards in research/development stage */
  :global(.job-card[data-status="COMPLETED"]) {
    cursor: pointer;
  }

  .job-card:hover {
    background: rgba(8, 13, 20, 0.97);
  }

  /* Top row */
  .job-top-row {
    display: flex;
    align-items: center;
    gap: 8px;
    min-width: 0;
  }

  /* Status pill */
  .status-pill {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 3px 9px;
    border-radius: 20px;
    font-size: 10px;
    font-family: var(--font-ambient, 'JetBrains Mono', monospace);
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.4px;
    flex-shrink: 0;
  }

  .status-pill[data-status="PENDING"] {
    background: rgba(100, 116, 139, 0.15);
    color: #94a3b8;
    border: 1px solid rgba(100, 116, 139, 0.2);
  }

  .status-pill[data-status="DOWNLOADING"] {
    background: rgba(240, 165, 0, 0.12);
    color: #f0a500;
    border: 1px solid rgba(240, 165, 0, 0.25);
  }

  .status-pill[data-status="PROCESSING"] {
    background: rgba(0, 212, 255, 0.1);
    color: #00d4ff;
    border: 1px solid rgba(0, 212, 255, 0.22);
  }

  .status-pill[data-status="ANALYZING"] {
    background: rgba(0, 212, 255, 0.1);
    color: #00d4ff;
    border: 1px solid rgba(0, 212, 255, 0.22);
  }

  .status-pill[data-status="COMPLETED"] {
    background: rgba(0, 200, 150, 0.12);
    color: #00c896;
    border: 1px solid rgba(0, 200, 150, 0.22);
  }

  .status-pill[data-status="FAILED"] {
    background: rgba(255, 59, 59, 0.1);
    color: #ff3b3b;
    border: 1px solid rgba(255, 59, 59, 0.22);
  }

  /* Pulse dot for in-progress statuses */
  .pulse-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: currentColor;
    animation: pulse-dot 1.4s ease-in-out infinite;
    flex-shrink: 0;
  }

  @keyframes pulse-dot {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: 0.4; transform: scale(0.7); }
  }

  .job-url {
    flex: 1;
    font-size: 12px;
    font-family: var(--font-ambient, 'JetBrains Mono', monospace);
    color: #cbd5e1;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    min-width: 0;
  }

  .job-timestamp {
    font-size: 10px;
    color: #475569;
    font-family: var(--font-ambient, 'JetBrains Mono', monospace);
    flex-shrink: 0;
  }

  .cancel-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 22px;
    height: 22px;
    padding: 0;
    background: rgba(255, 59, 59, 0.08);
    border: 1px solid rgba(255, 59, 59, 0.2);
    border-radius: 5px;
    color: #ff3b3b;
    cursor: pointer;
    flex-shrink: 0;
    transition: background 0.15s ease;
  }

  .cancel-btn:hover {
    background: rgba(255, 59, 59, 0.18);
  }

  /* Progress row */
  .job-progress-row {
    display: flex;
    align-items: center;
    gap: 10px;
  }

  .progress-track {
    flex: 1;
    height: 4px;
    background: rgba(255, 255, 255, 0.07);
    border-radius: 2px;
    overflow: hidden;
  }

  .progress-fill {
    height: 100%;
    border-radius: 2px;
    transition: width 0.4s ease;
  }

  .progress-fill[data-status="DOWNLOADING"] {
    background: linear-gradient(90deg, #f0a500, #f5c842);
  }

  .progress-fill[data-status="PROCESSING"],
  .progress-fill[data-status="ANALYZING"] {
    background: linear-gradient(90deg, #00d4ff, #00b4e0);
  }

  .progress-label {
    font-size: 10px;
    font-family: var(--font-ambient, 'JetBrains Mono', monospace);
    color: #64748b;
    white-space: nowrap;
    flex-shrink: 0;
    text-transform: uppercase;
  }

  /* Meta row */
  .job-meta-row {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .job-id {
    font-size: 10px;
    font-family: var(--font-ambient, 'JetBrains Mono', monospace);
    color: #475569;
  }

  /* Error message */
  .job-error-msg {
    display: flex;
    align-items: flex-start;
    gap: 6px;
    font-size: 11px;
    font-family: var(--font-ambient, 'JetBrains Mono', monospace);
    color: #ff3b3b;
    line-height: 1.4;
  }

  .job-error-msg :global(svg) {
    flex-shrink: 0;
    margin-top: 1px;
  }

  /* Pipeline stage indicator */
  .pipeline-stages {
    display: flex;
    align-items: center;
    gap: 4px;
    flex-wrap: wrap;
    padding-top: 4px;
    border-top: 1px solid rgba(255, 255, 255, 0.05);
  }

  .stage-item {
    display: inline-flex;
    align-items: center;
    gap: 3px;
    font-size: 10px;
    font-family: var(--font-ambient, 'JetBrains Mono', monospace);
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.3px;
    padding: 2px 6px;
    border-radius: 4px;
  }

  .stage-item[data-state="done"] {
    color: #00c896;
    background: rgba(0, 200, 150, 0.08);
  }

  .stage-item[data-state="active"] {
    color: #00d4ff;
    background: rgba(0, 212, 255, 0.08);
  }

  .stage-item[data-state="pending"] {
    color: #475569;
    background: rgba(255, 255, 255, 0.03);
  }

  .stage-check {
    color: #00c896;
    font-weight: 700;
  }

  .stage-spin {
    color: #00d4ff;
    display: inline-block;
    animation: spin 1.5s linear infinite;
  }

  .stage-dash {
    color: #475569;
  }

  .stage-arrow {
    font-size: 10px;
    color: #334155;
    flex-shrink: 0;
  }

  /* =========================================================================
     Section 3: WF2 Banner
     ========================================================================= */
  .wf2-banner {
    display: flex;
    align-items: center;
    gap: 9px;
    padding: 11px 16px;
    background: rgba(8, 13, 20, 0.7);
    backdrop-filter: blur(8px);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 8px;
    font-size: 12px;
    font-family: var(--font-ambient, 'JetBrains Mono', monospace);
    color: #64748b;
  }

  .wf2-banner :global(svg) {
    color: #475569;
    flex-shrink: 0;
  }

  .kanban-section {
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  .kanban-section-header {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .kanban-section-title {
    font-family: var(--font-ambient, 'JetBrains Mono', monospace);
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: #64748b;
  }

  .kanban-section-subtitle {
    font-size: 12px;
    color: #475569;
  }

  .kanban-status {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 16px;
    border-radius: 8px;
    font-size: 12px;
    font-family: var(--font-ambient, 'JetBrains Mono', monospace);
  }

  .kanban-status.loading {
    background: rgba(8, 13, 20, 0.7);
    border: 1px solid rgba(0, 212, 255, 0.12);
    color: #94a3b8;
  }

  .kanban-status.error {
    background: rgba(255, 59, 59, 0.08);
    border: 1px solid rgba(255, 59, 59, 0.18);
    color: #ff8a8a;
  }

  .kanban-status.empty {
    background: rgba(8, 13, 20, 0.5);
    border: 1px solid rgba(255, 255, 255, 0.05);
    color: #64748b;
  }

  .kanban-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 14px;
  }

  .kanban-column {
    min-width: 0;
    display: flex;
    flex-direction: column;
    gap: 10px;
    background: rgba(8, 13, 20, 0.62);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 10px;
    padding: 12px;
  }

  .kanban-column-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
  }

  .kanban-column-title {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: #cbd5e1;
  }

  .kanban-column-count {
    font-size: 10px;
    font-family: var(--font-ambient, 'JetBrains Mono', monospace);
    color: #94a3b8;
    background: rgba(255, 255, 255, 0.06);
    border-radius: 999px;
    padding: 2px 8px;
  }

  .kanban-column-body {
    display: flex;
    flex-direction: column;
    gap: 10px;
    min-height: 92px;
  }

  .kanban-column-empty {
    border: 1px dashed rgba(255, 255, 255, 0.08);
    border-radius: 8px;
    padding: 14px 10px;
    text-align: center;
    color: #475569;
    font-size: 11px;
    font-family: var(--font-ambient, 'JetBrains Mono', monospace);
  }

  @media (max-width: 1200px) {
    .kanban-grid {
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }
  }

  @media (max-width: 760px) {
    .kanban-grid {
      grid-template-columns: 1fr;
    }
  }

  /* Autonomous Scheduler Section */
  .scheduler-section {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(0, 212, 255, 0.12);
    border-radius: 10px;
    padding: 14px 16px;
    display: flex;
    flex-direction: column;
    gap: 10px;
  }

  .scheduler-header {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  :global(.scheduler-icon) {
    color: #00d4ff;
    flex-shrink: 0;
  }

  .scheduler-label {
    font-size: 12px;
    font-family: 'JetBrains Mono', monospace;
    color: rgba(255,255,255,0.75);
    font-weight: 600;
    flex: 1;
  }

  .scheduler-toggle {
    width: 36px;
    height: 20px;
    border-radius: 10px;
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.15);
    cursor: pointer;
    position: relative;
    transition: background 0.2s;
    flex-shrink: 0;
  }

  .scheduler-toggle.enabled {
    background: rgba(0,212,255,0.3);
    border-color: rgba(0,212,255,0.5);
  }

  .toggle-pill {
    position: absolute;
    top: 2px;
    left: 2px;
    width: 14px;
    height: 14px;
    border-radius: 50%;
    background: rgba(255,255,255,0.5);
    transition: transform 0.2s, background 0.2s;
  }

  .scheduler-toggle.enabled .toggle-pill {
    transform: translateX(16px);
    background: #00d4ff;
  }

  .run-now-btn {
    display: flex;
    align-items: center;
    gap: 5px;
    padding: 4px 10px;
    background: transparent;
    border: 1px solid rgba(0,212,255,0.3);
    border-radius: 5px;
    color: #00d4ff;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    cursor: pointer;
    transition: background 0.15s;
    flex-shrink: 0;
  }

  .run-now-btn:hover:not(:disabled) {
    background: rgba(0,212,255,0.1);
  }

  .run-now-btn:disabled {
    opacity: 0.45;
    cursor: not-allowed;
  }

  .scheduler-meta {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
  }

  .meta-item {
    display: flex;
    gap: 5px;
    font-size: 11px;
    font-family: 'JetBrains Mono', monospace;
  }

  .meta-key {
    color: rgba(255,255,255,0.35);
  }

  .meta-val {
    color: rgba(255,255,255,0.7);
  }

  .scheduler-running-badge {
    display: flex;
    align-items: center;
    gap: 5px;
    font-size: 11px;
    font-family: 'JetBrains Mono', monospace;
    color: #00d4ff;
  }

  .scheduler-offline {
    font-size: 11px;
    font-family: 'JetBrains Mono', monospace;
    color: rgba(255,255,255,0.3);
    margin: 0;
  }

  .job-source-label {
    display: flex;
    align-items: center;
    gap: 4px;
    font-size: 11px;
    font-family: 'JetBrains Mono', monospace;
    color: rgba(255,255,255,0.45);
  }

  .job-source-label.scheduled {
    color: #00d4ff;
    opacity: 0.8;
  }

  :global(.spin-icon) {
    animation: spin 1s linear infinite;
  }
</style>
