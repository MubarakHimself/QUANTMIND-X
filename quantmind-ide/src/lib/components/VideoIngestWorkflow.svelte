<script lang="ts">
  import { onMount } from 'svelte';
  import { Loader2, Play, CheckCircle, XCircle, AlertCircle, Youtube, ListVideo, KeyRound } from 'lucide-svelte';
  import { getAuthStatus, getJobStatus, submitVideoJob } from '$lib/api/videoIngestApi';

  interface VideoIngestJob {
    id: string;
    status: string;
    youtube_url: string;
    created_at: string;
    result?: any;
    progress?: number;
    strategy_folder?: string;
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

  // Check authentication status on mount
  onMount(async () => {
    await checkAuth();
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

  async function submitJob() {
    if (!youtubeUrl) return;

    loading = true;
    error = '';

    try {
      const result = await submitVideoJob(
        youtubeUrl,
        extractVideoId(youtubeUrl) || 'video_ingest',
        isPlaylist
      );
      const jobIds = result.job_ids?.length ? result.job_ids : [result.job_id];
      jobs = [
        ...jobIds.map((jobId) => ({
          id: jobId,
          status: result.status,
          youtube_url: youtubeUrl,
          created_at: new Date().toISOString(),
          progress: 0,
          strategy_folder: result.strategy_folder
        })),
        ...jobs
      ];
      youtubeUrl = '';
    } catch (e: unknown) {
      error = e instanceof Error ? e.message : 'Failed to submit job';
    } finally {
      loading = false;
    }
  }

  async function refreshJobs() {
    // Poll for job status updates
    for (const job of jobs) {
      const normalizedStatus = job.status.toUpperCase();
      if (normalizedStatus === 'PENDING' || normalizedStatus === 'DOWNLOADING' || normalizedStatus === 'PROCESSING' || normalizedStatus === 'ANALYZING') {
        try {
          const updated = await getJobStatus(job.id);
          job.status = updated.status;
          job.progress = updated.progress;
          jobs = [...jobs];
        } catch {
          // Ignore polling errors
        }
      }
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
                {new Date(job.created_at).toLocaleString()}
              </span>
              {#if job.strategy_folder}
                <span class="job-folder">📁 {job.strategy_folder}</span>
              {/if}
            </div>
            {#if job.result?.timeline?.length}
              <div class="job-result">
                <span>{job.result.timeline.length} clips extracted</span>
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
    gap: 12px;
    font-size: 10px;
    color: var(--color-text-muted);
  }

  .job-folder {
    color: var(--color-accent-cyan);
  }

  .job-result {
    margin-top: 8px;
    padding-top: 8px;
    border-top: 1px solid var(--color-border-subtle);
    font-size: 11px;
    color: var(--color-accent-green);
  }

  :global(.spin) {
    animation: spin 1s linear infinite;
  }

  @keyframes spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
  }
</style>
