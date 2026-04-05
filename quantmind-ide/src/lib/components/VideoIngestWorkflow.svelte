<script lang="ts">
  import { onDestroy, onMount } from 'svelte';
  import { get } from 'svelte/store';
  import { Loader2, Play, CheckCircle, XCircle, AlertCircle, Youtube, ListVideo, KeyRound } from 'lucide-svelte';
  import { activeCanvasStore } from '$lib/stores/canvasStore';
  import { sharedAssetsStore } from '$lib/stores/sharedAssets';
  import {
    getAuthStatus,
    listVideoJobs,
    submitVideoJob,
    getPreferredVideoProvider,
    type VideoIngestProviderSummary,
    type VideoIngestJobStatus
  } from '$lib/api/videoIngestApi';

  interface VideoIngestJob {
    id: string;
    status: string;
    youtube_url: string;
    created_at: string;
    progress?: number;
    updated_at?: string;
    current_stage?: string;
    error?: string | null;
    workflow_id?: string;
    workflow_status?: string;
    waiting_reason?: string | null;
    blocking_error?: string | null;
    blocking_error_detail?: string | null;
    strategy_id?: string;
    strategy_folder?: string;
    strategy_asset_id?: string;
    strategy_status?: string;
    strategy_family?: string;
    source_bucket?: string;
    has_video_ingest?: boolean;
  }

  interface AuthStatus {
    openrouter?: boolean;
    gemini: boolean;
    qwen: boolean;
  }

  let youtubeUrl = $state('');
  let isPlaylist = $state(false);
  let loading = $state(false);
  let error = $state('');
  let jobs: VideoIngestJob[] = $state([]);
  let authStatus: AuthStatus = $state({ openrouter: false, gemini: false, qwen: false });
  let checkingAuth = $state(true);
  let providerSummary: VideoIngestProviderSummary | null = $state(null);
  let refreshHandle: ReturnType<typeof setInterval> | null = null;

  onMount(async () => {
    await Promise.all([checkAuth(), loadProviderSummary(), loadJobs()]);
    refreshHandle = setInterval(() => {
      void refreshJobs();
    }, 8000);
  });

  onDestroy(() => {
    if (refreshHandle) {
      clearInterval(refreshHandle);
      refreshHandle = null;
    }
  });

  async function checkAuth() {
    checkingAuth = true;
    try {
      authStatus = await getAuthStatus();
    } catch {
      authStatus = { openrouter: false, gemini: false, qwen: false };
    } finally {
      checkingAuth = false;
    }
  }

  async function loadProviderSummary() {
    try {
      providerSummary = await getPreferredVideoProvider();
    } catch {
      providerSummary = null;
    }
  }

  function mapJob(job: VideoIngestJobStatus): VideoIngestJob {
    return {
      id: job.job_id,
      status: job.status,
      youtube_url: job.video_url || '',
      created_at: job.created_at || new Date().toISOString(),
      updated_at: job.updated_at || job.created_at,
      progress: job.progress,
      current_stage: job.current_stage,
      error: job.error,
      workflow_id: job.workflow_id,
      workflow_status: job.workflow_status,
      waiting_reason: job.waiting_reason,
      blocking_error: job.blocking_error,
      blocking_error_detail: job.blocking_error_detail,
      strategy_id: job.strategy_id,
      strategy_folder: job.strategy_folder,
      strategy_asset_id: job.strategy_asset_id,
      strategy_status: job.strategy_status,
      strategy_family: job.strategy_family,
      source_bucket: job.source_bucket,
      has_video_ingest: job.has_video_ingest,
    };
  }

  async function loadJobs() {
    try {
      const recentJobs = await listVideoJobs(25);
      jobs = recentJobs.map(mapJob);
    } catch (e: unknown) {
      error = e instanceof Error ? e.message : 'Failed to load recent jobs';
    }
  }

  async function submitJob() {
    if (!youtubeUrl) return;

    loading = true;
    error = '';

    try {
      await submitVideoJob(
        youtubeUrl,
        extractVideoId(youtubeUrl) || 'video_ingest',
        isPlaylist
      );
      youtubeUrl = '';
      await loadJobs();
    } catch (e: unknown) {
      error = e instanceof Error ? e.message : 'Failed to submit job';
    } finally {
      loading = false;
    }
  }

  async function refreshJobs() {
    await loadJobs();
  }

  async function openStrategyAsset(job: VideoIngestJob) {
    if (!job.strategy_asset_id) return;

    error = '';
    try {
      await sharedAssetsStore.fetchAssetsByType('strategies');
      const state = get(sharedAssetsStore);
      const strategyAsset = state.assets.strategies.find((asset) =>
        asset.id === job.strategy_asset_id
        || asset.name === job.strategy_folder
        || asset.details?.relative_root === job.strategy_asset_id?.replace(/^strategies\//, '')
      );

      if (!strategyAsset) {
        throw new Error(`Strategy asset not found for ${job.strategy_asset_id}`);
      }

      sharedAssetsStore.setSelectedType('strategies');
      await sharedAssetsStore.selectAsset(strategyAsset);
      activeCanvasStore.setActiveCanvas('shared-assets');
    } catch (e: unknown) {
      error = e instanceof Error ? e.message : 'Failed to open strategy asset';
    }
  }

  function extractVideoId(url: string): string | null {
    // YouTube URL patterns
    const patterns = [
      /(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([^&\s?]+)/,
      /youtube\.com\/playlist\?list=([^&\s?]+)/
    ];
    for (const pattern of patterns) {
      const match = url.match(pattern);
      if (match) return match[1];
    }
    return null;
  }

  function getStatusIcon(status: string) {
    switch (status.toUpperCase()) {
      case 'COMPLETED': return CheckCircle;
      case 'FAILED': return XCircle;
      case 'PROCESSING':
      case 'DOWNLOADING':
      case 'ANALYZING':
        return Loader2;
      case 'PENDING': return ListVideo;
      default: return AlertCircle;
    }
  }

  function getStatusColor(status: string): string {
    switch (status.toUpperCase()) {
      case 'COMPLETED': return 'var(--color-accent-green)';
      case 'FAILED': return 'var(--color-accent-red)';
      case 'PROCESSING':
      case 'DOWNLOADING':
      case 'ANALYZING':
        return 'var(--color-accent-amber)';
      case 'PENDING': return 'var(--color-accent-cyan)';
      default: return 'var(--color-text-muted)';
    }
  }

  function getStatusText(status: string): string {
    switch (status.toUpperCase()) {
      case 'PENDING': return 'In Queue';
      case 'DOWNLOADING': return 'Downloading';
      case 'PROCESSING': return 'Processing';
      case 'ANALYZING': return 'Analyzing';
      case 'COMPLETED': return 'Completed';
      case 'FAILED': return 'Failed';
      default: return status;
    }
  }

  function formatStage(job: VideoIngestJob): string {
    if (job.current_stage) return job.current_stage.replace(/_/g, ' ');
    if (job.waiting_reason) return job.waiting_reason.replace(/_/g, ' ');
    return getStatusText(job.status);
  }

  function formatBucket(job: VideoIngestJob): string | null {
    if (!job.strategy_family || !job.source_bucket) return null;
    return `${job.strategy_family}/${job.source_bucket}`;
  }

  function summarizeError(message: string | null | undefined): string {
    const raw = (message || '').trim();
    if (!raw) return '';
    const firstLine = raw.split('\n').map((line) => line.trim()).find(Boolean) || raw;
    return firstLine.length > 220 ? `${firstLine.slice(0, 217)}...` : firstLine;
  }
</script>

<div class="video-ingest-workflow">
  <div class="header">
    <h3>Video Ingest</h3>
    <p>Process YouTube videos and playlists into structured trading data</p>
  </div>

  <!-- Authentication Status -->
  <div class="auth-section">
    <div class="auth-header">
      <KeyRound size={14} />
      <span>AI Providers</span>
    </div>
    <div class="auth-status">
      <div class="auth-item" class:authenticated={authStatus.openrouter}>
        <span class="auth-dot"></span>
        <span>OpenRouter</span>
        {#if checkingAuth}
          <Loader2 size={12} class="spin" />
        {:else if authStatus.openrouter}
          <CheckCircle size={12} />
        {:else}
          <XCircle size={12} />
        {/if}
      </div>
      <div class="auth-item" class:authenticated={authStatus.gemini}>
        <span class="auth-dot"></span>
        <span>Gemini CLI</span>
        {#if checkingAuth}
          <Loader2 size={12} class="spin" />
        {:else if authStatus.gemini}
          <CheckCircle size={12} />
        {:else}
          <XCircle size={12} />
        {/if}
      </div>
      <div class="auth-item" class:authenticated={authStatus.qwen}>
        <span class="auth-dot"></span>
        <span>Qwen CLI</span>
        {#if checkingAuth}
          <Loader2 size={12} class="spin" />
        {:else if authStatus.qwen}
          <CheckCircle size={12} />
        {:else}
          <XCircle size={12} />
        {/if}
      </div>
    </div>
    {#if providerSummary?.selected_model}
      <div class="provider-summary">
        <span class="provider-model">{providerSummary.selected_model.name || providerSummary.primary_model}</span>
        {#if providerSummary.selected_model.pricing}
          <span class="provider-pricing">{providerSummary.selected_model.pricing}</span>
        {/if}
        {#if providerSummary.selected_model.capabilities}
          <span class="provider-capabilities">{providerSummary.selected_model.capabilities}</span>
        {/if}
      </div>
    {/if}
    <p class="auth-hint">Select the OpenRouter video model and compare pricing in Settings → Providers.</p>
  </div>

  <!-- Submit Form -->
  <div class="submit-form">
    <div class="input-group">
      <input
        type="text"
        bind:value={youtubeUrl}
        placeholder="Enter YouTube URL or playlist..."
        disabled={loading}
        onkeydown={(e) => e.key === 'Enter' && submitJob()}
      />
      <button
        class="playlist-toggle"
        class:active={isPlaylist}
        onclick={() => isPlaylist = !isPlaylist}
        title="Toggle playlist mode"
      >
        <ListVideo size={16} />
      </button>
    </div>
    <button onclick={submitJob} disabled={loading || !youtubeUrl}>
      {#if loading}
        <Loader2 size={14} class="spin" />
        Processing...
      {:else}
        <Play size={14} />
        {isPlaylist ? 'Process Playlist' : 'Process Video'}
      {/if}
    </button>
  </div>

  {#if error}
    <div class="error-message">
      {error}
    </div>
  {/if}

  <!-- Jobs List -->
  <div class="jobs-section">
    <div class="jobs-header">
      <h4>Recent Jobs</h4>
      <button class="refresh-btn" onclick={refreshJobs} title="Refresh">
        <Loader2 size={14} />
      </button>
    </div>

    {#if jobs.length === 0}
      <div class="empty-state">
        <Youtube size={32} />
        <p>No jobs yet. Submit a YouTube URL to extract trading data.</p>
        <p class="hint">Supports single videos and playlists</p>
      </div>
    {:else}
      <div class="jobs-list">
        {#each jobs as job}
          {@const SvelteComponent = getStatusIcon(job.status)}
          <div class="job-card">
            <div class="job-main">
              <div class="job-status" style="color: {getStatusColor(job.status)}">
                <SvelteComponent size={16} />
                <span>{getStatusText(job.status)}</span>
              </div>
              {#if job.progress !== undefined && job.progress > 0}
                <div class="progress-bar">
                  <div class="progress-fill" style="width: {job.progress}%"></div>
                </div>
              {/if}
            </div>
            <div class="job-url">{job.youtube_url}</div>
            <div class="job-meta">
              <span class="job-time">
                Queued {new Date(job.created_at).toLocaleString()}
              </span>
              {#if job.updated_at}
                <span class="job-time">
                  Updated {new Date(job.updated_at).toLocaleString()}
                </span>
              {/if}
              {#if job.strategy_folder}
                <span class="job-folder">📁 {job.strategy_folder}</span>
              {/if}
              {#if formatBucket(job)}
                <span class="job-bucket">{formatBucket(job)}</span>
              {/if}
            </div>
            <div class="job-details">
              <span class="job-pill stage">{formatStage(job)}</span>
              {#if job.workflow_status}
                <span class="job-pill workflow">{job.workflow_status}</span>
              {/if}
              {#if job.strategy_status}
                <span class:quarantined={job.strategy_status === 'quarantined'} class="job-pill strategy">
                  {job.strategy_status}
                </span>
              {/if}
              {#if job.progress !== undefined}
                <span class="job-pill progress">{job.progress}%</span>
              {/if}
            </div>
            {#if job.blocking_error || job.error}
              <div class="job-error" title={job.blocking_error_detail || job.blocking_error || job.error || undefined}>
                {summarizeError(job.blocking_error || job.error)}
              </div>
            {:else if job.has_video_ingest}
              <div class="job-result">
                <span>Video ingest artifacts are available in Shared Assets.</span>
              </div>
            {/if}
            {#if job.strategy_asset_id}
              <div class="job-actions">
                <button class="job-link" onclick={() => openStrategyAsset(job)}>
                  Open Strategy
                </button>
              </div>
            {/if}
          </div>
        {/each}
      </div>
    {/if}
  </div>
</div>

<style>
  .video-ingest-workflow {
    display: flex;
    flex-direction: column;
    gap: 16px;
  }

  .header h3 {
    margin: 0;
    font-size: 18px;
    color: var(--color-text-primary);
  }

  .header p {
    margin: 4px 0 0;
    font-size: 12px;
    color: var(--color-text-secondary);
  }

  /* Auth Section */
  .auth-section {
    background: var(--color-bg-elevated);
    border: 1px solid var(--color-border-subtle);
    border-radius: 6px;
    padding: 12px;
  }

  .auth-header {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 12px;
    font-weight: 500;
    color: var(--color-text-secondary);
    margin-bottom: 8px;
  }

  .auth-status {
    display: flex;
    gap: 16px;
    flex-wrap: wrap;
  }

  .auth-item {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 11px;
    color: var(--color-text-muted);
  }

  .auth-item.authenticated {
    color: var(--color-accent-green);
  }

  .auth-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--color-text-muted);
  }

  .auth-item.authenticated .auth-dot {
    background: var(--color-accent-green);
  }

  .auth-hint {
    margin: 8px 0 0;
    font-size: 11px;
    color: var(--color-text-muted);
  }

  .provider-summary {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 10px;
    font-size: 11px;
    color: var(--color-text-secondary);
  }

  .provider-model {
    color: var(--color-text-primary);
    font-weight: 600;
  }

  .provider-pricing,
  .provider-capabilities {
    padding: 2px 8px;
    border-radius: 999px;
    background: color-mix(in srgb, var(--color-bg-elevated) 75%, transparent);
    border: 1px solid var(--color-border-subtle);
  }

  /* Submit Form */
  .submit-form {
    display: flex;
    gap: 8px;
  }

  .input-group {
    flex: 1;
    display: flex;
    gap: 0;
  }

  .input-group input {
    flex: 1;
    padding: 10px 14px;
    background: var(--color-bg-elevated);
    border: 1px solid var(--color-border-subtle);
    border-right: none;
    border-radius: 6px 0 0 6px;
    color: var(--color-text-primary);
    font-size: 13px;
  }

  .input-group input:focus {
    outline: none;
    border-color: var(--color-accent-cyan);
  }

  .playlist-toggle {
    padding: 10px 12px;
    background: var(--color-bg-elevated);
    border: 1px solid var(--color-border-subtle);
    border-left: none;
    border-radius: 0 6px 6px 0;
    color: var(--color-text-muted);
    cursor: pointer;
    transition: all 0.2s;
  }

  .playlist-toggle:hover {
    background: var(--color-bg-surface);
  }

  .playlist-toggle.active {
    background: var(--color-accent-cyan);
    color: var(--color-bg-base);
  }

  .submit-form > button {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 10px 16px;
    background: var(--color-accent-cyan);
    color: var(--color-bg-base);
    border: none;
    border-radius: 6px;
    font-size: 13px;
    font-weight: 500;
    cursor: pointer;
  }

  .submit-form button:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .error-message {
    padding: 10px;
    background: rgba(239, 68, 68, 0.1);
    border: 1px solid var(--color-accent-red);
    border-radius: 6px;
    color: var(--color-accent-red);
    font-size: 12px;
  }

  /* Jobs Section */
  .jobs-section {
    flex: 1;
  }

  .jobs-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 12px;
  }

  .jobs-header h4 {
    margin: 0;
    font-size: 14px;
    color: var(--color-text-primary);
  }

  .refresh-btn {
    padding: 4px;
    background: transparent;
    border: none;
    color: var(--color-text-muted);
    cursor: pointer;
    border-radius: 4px;
  }

  .refresh-btn:hover {
    background: var(--color-bg-elevated);
    color: var(--color-text-primary);
  }

  .empty-state {
    padding: 32px;
    text-align: center;
    background: var(--color-bg-elevated);
    border-radius: 8px;
    color: var(--color-text-muted);
  }

  .empty-state p {
    margin: 8px 0 0;
    font-size: 12px;
  }

  .empty-state .hint {
    font-size: 11px;
    opacity: 0.7;
  }

  .jobs-list {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .job-card {
    padding: 12px;
    background: var(--color-bg-elevated);
    border: 1px solid var(--color-border-subtle);
    border-radius: 6px;
  }

  .job-main {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 8px;
  }

  .job-status {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 12px;
    font-weight: 500;
    text-transform: capitalize;
  }

  .progress-bar {
    width: 80px;
    height: 4px;
    background: var(--color-bg-surface);
    border-radius: 2px;
    overflow: hidden;
  }

  .progress-fill {
    height: 100%;
    background: var(--color-accent-cyan);
    transition: width 0.3s;
  }

  .job-url {
    font-size: 11px;
    color: var(--color-text-secondary);
    font-family: monospace;
    word-break: break-all;
    margin-bottom: 4px;
  }

  .job-meta {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    font-size: 10px;
    color: var(--color-text-muted);
  }

  .job-folder {
    color: var(--color-accent-cyan);
  }

  .job-bucket {
    color: var(--color-text-secondary);
  }

  .job-details {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 8px;
  }

  .job-pill {
    padding: 3px 8px;
    border-radius: 999px;
    border: 1px solid var(--color-border-subtle);
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--color-text-secondary);
    background: color-mix(in srgb, var(--color-bg-elevated) 80%, transparent);
  }

  .job-pill.stage {
    color: var(--color-accent-cyan);
  }

  .job-pill.workflow {
    color: var(--color-accent-amber);
  }

  .job-pill.strategy {
    color: var(--color-accent-green);
  }

  .job-pill.strategy.quarantined {
    color: var(--color-accent-red);
    border-color: color-mix(in srgb, var(--color-accent-red) 45%, transparent);
  }

  .job-pill.progress {
    color: var(--color-text-primary);
  }

  .job-result {
    margin-top: 8px;
    padding-top: 8px;
    border-top: 1px solid var(--color-border-subtle);
    font-size: 11px;
    color: var(--color-accent-green);
  }

  .job-actions {
    display: flex;
    justify-content: flex-start;
    margin-top: 10px;
  }

  .job-link {
    padding: 6px 10px;
    border-radius: 999px;
    border: 1px solid var(--color-border-subtle);
    background: color-mix(in srgb, var(--color-bg-elevated) 88%, transparent);
    color: var(--color-accent-cyan);
    font-size: 11px;
    font-weight: 600;
    cursor: pointer;
    transition: border-color 0.2s ease, background 0.2s ease, color 0.2s ease;
  }

  .job-link:hover {
    border-color: color-mix(in srgb, var(--color-accent-cyan) 42%, var(--color-border-subtle));
    background: color-mix(in srgb, var(--color-accent-cyan) 14%, var(--color-bg-elevated));
    color: var(--color-text-primary);
  }

  .job-error {
    margin-top: 8px;
    padding: 10px 12px;
    border-radius: 6px;
    background: color-mix(in srgb, var(--color-accent-red) 12%, transparent);
    border: 1px solid color-mix(in srgb, var(--color-accent-red) 35%, transparent);
    color: var(--color-accent-red);
    font-size: 11px;
    line-height: 1.4;
    word-break: break-word;
  }

  :global(.spin) {
    animation: spin 1s linear infinite;
  }

  @keyframes spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
  }
</style>
