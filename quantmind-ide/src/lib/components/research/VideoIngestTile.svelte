<script lang="ts">
  /**
   * VideoIngestTile - Video Ingest UI Component
   *
   * Provides YouTube URL input, ingest trigger, and progress tracking.
   * Implements Frosted Terminal aesthetic with glass tiles and Lucide icons.
   */
  import GlassTile from '$lib/components/live-trading/GlassTile.svelte';
  import {
    Play,
    Loader2,
    CheckCircle,
    XCircle,
    Clock,
    ExternalLink,
    Search
  } from 'lucide-svelte';
  import {
    submitJob,
    currentJob,
    isProcessing,
    errorMessage,
    authStatus,
    loadAuthStatus,
    isValidYouTubeUrl,
    clearError,
    type VideoJob,
    PIPELINE_STAGES,
    type PipelineStage
  } from '$lib/stores/videoIngest';
  import { onMount } from 'svelte';

  // =============================================================================
  // State
  // =============================================================================

  let urlInput = '';
  let localError: string | null = null;
  let showAuthWarning = false;

  // =============================================================================
  // Lifecycle
  // =============================================================================

  onMount(async () => {
    await loadAuthStatus();
    showAuthWarning = !$authStatus.gemini && !$authStatus.qwen && $authStatus.checked;
  });

  // =============================================================================
  // Derived / Reactive
  // =============================================================================

  $: isProcessingJob = $isProcessing;
  $: currentVideoJob = $currentJob;
  $: errorMsg = $errorMessage || localError;

  $: stageIndex = currentVideoJob
    ? PIPELINE_STAGES.indexOf(currentVideoJob.currentStage)
    : -1;

  // =============================================================================
  // Methods
  // =============================================================================

  function validateAndSubmit() {
    localError = null;

    if (!urlInput.trim()) {
      localError = 'Please enter a YouTube URL';
      return;
    }

    if (!isValidYouTubeUrl(urlInput)) {
      localError = 'Invalid YouTube URL. Use youtube.com/watch, youtu.be, or playlist URL';
      return;
    }

    submitJob(urlInput.trim());
  }

  function handlePaste(event: ClipboardEvent) {
    const pastedText = event.clipboardData?.getData('text') || '';
    if (isValidYouTubeUrl(pastedText)) {
      urlInput = pastedText;
      validateAndSubmit();
    }
  }

  function handleKeyDown(event: KeyboardEvent) {
    if (event.key === 'Enter') {
      validateAndSubmit();
    }
  }

  function clearErrorState() {
    localError = null;
    clearError();
  }

  function getStageIcon(stage: PipelineStage, index: number): typeof CheckCircle {
    if (!currentVideoJob) return Clock;
    if (index < stageIndex) return CheckCircle;
    if (index === stageIndex) return Loader2;
    return Clock;
  }

  function getStageClass(index: number): string {
    if (!currentVideoJob) return 'pending';
    if (index < stageIndex) return 'completed';
    if (index === stageIndex) return 'in-progress';
    return 'pending';
  }
</script>

<GlassTile clickable={false}>
  <div class="video-ingest-container">
    <!-- Header -->
    <div class="ingest-header">
      <Search size={18} class="header-icon" />
      <span class="header-title">Video Ingest</span>
      {#if showAuthWarning}
        <span class="auth-warning">Setup required</span>
      {/if}
    </div>

    <!-- URL Input -->
    {#if !currentVideoJob || currentVideoJob.status === 'completed' || currentVideoJob.status === 'failed'}
      <div class="input-section">
        <input
          type="text"
          bind:value={urlInput}
          onpaste={handlePaste}
          onkeydown={handleKeyDown}
          placeholder="Paste YouTube URL..."
          class="url-input"
          disabled={isProcessingJob}
        />
        <button
          class="ingest-btn"
          onclick={validateAndSubmit}
          disabled={isProcessingJob || !urlInput.trim()}
        >
          <Play size={14} />
          <span>Ingest</span>
        </button>
      </div>
    {/if}

    <!-- Error Display -->
    {#if errorMsg}
      <div class="error-display">
        <XCircle size={14} class="error-icon" />
        <span class="error-text">{errorMsg}</span>
        <button class="clear-btn" onclick={clearErrorState}>×</button>
      </div>
    {/if}

    <!-- Progress Tracker -->
    {#if currentVideoJob && (currentVideoJob.status === 'processing' || currentVideoJob.status === 'pending')}
      <div class="progress-section">
        <div class="progress-header">
          <Loader2 size={14} class="spinner" />
          <span class="progress-label">Processing...</span>
        </div>

        <div class="pipeline-stages">
          {#each PIPELINE_STAGES.slice(0, -1) as stage, index}
            {@const stageClass = getStageClass(index)}
            {@const IconComponent = getStageIcon(stage, index)}
            <div class="stage-item {stageClass}">
              <div class="stage-icon">
                {#if stageClass === 'completed'}
                  <CheckCircle size={12} />
                {:else if stageClass === 'in-progress'}
                  <Loader2 size={12} class="spinning" />
                {:else if stageClass === 'failed'}
                  <XCircle size={12} />
                {:else}
                  <Clock size={12} />
                {/if}
              </div>
              <span class="stage-label">{stage}</span>
            </div>
            {#if index < PIPELINE_STAGES.length - 2}
              <div class="stage-connector {stageClass}"></div>
            {/if}
          {/each}
        </div>

        <!-- Job ID display -->
        <div class="job-id">
          <span class="job-id-label">Job:</span>
          <span class="job-id-value">{currentVideoJob.jobId.slice(0, 8)}...</span>
        </div>
      </div>
    {/if}

    <!-- Completed State -->
    {#if currentVideoJob && currentVideoJob.status === 'completed'}
      <div class="completed-section">
        <CheckCircle size={20} class="completed-icon" />
        <span class="completed-text">Video indexed successfully!</span>
        <a
          href={currentVideoJob.url}
          target="_blank"
          rel="noopener noreferrer"
          class="external-link"
        >
          <ExternalLink size={12} />
          View
        </a>
      </div>
    {/if}

    <!-- Failed State -->
    {#if currentVideoJob && currentVideoJob.status === 'failed'}
      <div class="failed-section">
        <XCircle size={20} class="failed-icon" />
        <span class="failed-text">Ingest failed</span>
        {#if currentVideoJob.error}
          <span class="failed-error">{currentVideoJob.error}</span>
        {/if}
        <button class="retry-btn" onclick={() => { clearErrorState(); }}>
          Try Again
        </button>
      </div>
    {/if}
  </div>
</GlassTile>

<style>
  .video-ingest-container {
    display: flex;
    flex-direction: column;
    gap: 10px;
    padding: 12px;
    min-width: 200px;
  }

  .ingest-header {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .header-icon {
    color: rgba(0, 212, 255, 0.6);
  }

  .header-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    font-weight: 500;
    color: rgba(224, 224, 224, 0.9);
  }

  .auth-warning {
    font-size: 10px;
    padding: 2px 6px;
    background: rgba(255, 71, 87, 0.2);
    border-radius: 4px;
    color: #ff4757;
    margin-left: auto;
  }

  .input-section {
    display: flex;
    gap: 8px;
    align-items: center;
  }

  .url-input {
    flex: 1;
    padding: 8px 12px;
    background: rgba(10, 15, 26, 0.6);
    border: 1px solid rgba(0, 212, 255, 0.15);
    border-radius: 6px;
    color: rgba(224, 224, 224, 0.9);
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    outline: none;
    transition: border-color 0.2s ease;
  }

  .url-input:focus {
    border-color: rgba(0, 212, 255, 0.3);
  }

  .url-input::placeholder {
    color: rgba(224, 224, 224, 0.4);
  }

  .url-input:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .ingest-btn {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 8px 14px;
    background: rgba(0, 212, 255, 0.15);
    border: 1px solid rgba(0, 212, 255, 0.25);
    border-radius: 6px;
    color: #00d4ff;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.2s ease;
  }

  .ingest-btn:hover:not(:disabled) {
    background: rgba(0, 212, 255, 0.25);
    border-color: rgba(0, 212, 255, 0.4);
  }

  .ingest-btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .error-display {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 8px 10px;
    background: rgba(255, 71, 87, 0.1);
    border: 1px solid rgba(255, 71, 87, 0.2);
    border-radius: 6px;
  }

  .error-icon {
    color: #ff4757;
    flex-shrink: 0;
  }

  .error-text {
    flex: 1;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: #ff4757;
  }

  .clear-btn {
    background: none;
    border: none;
    color: rgba(255, 71, 87, 0.6);
    font-size: 16px;
    cursor: pointer;
    padding: 0 4px;
  }

  .progress-section {
    display: flex;
    flex-direction: column;
    gap: 10px;
    padding: 10px;
    background: rgba(10, 15, 26, 0.4);
    border-radius: 8px;
  }

  .progress-header {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .spinner {
    color: #00d4ff;
    animation: spin 1s linear infinite;
  }

  @keyframes spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
  }

  .progress-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: #00d4ff;
    font-weight: 500;
  }

  .pipeline-stages {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 4px;
  }

  .stage-item {
    display: flex;
    align-items: center;
    gap: 4px;
    padding: 4px 8px;
    border-radius: 4px;
    transition: all 0.2s ease;
  }

  .stage-item.pending {
    color: rgba(100, 100, 100, 0.5);
  }

  .stage-item.in-progress {
    color: #00d4ff;
    background: rgba(0, 212, 255, 0.1);
    animation: pulse 2s ease-in-out infinite;
  }

  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.7; }
  }

  .stage-item.completed {
    color: #00c896;
  }

  .stage-item.failed {
    color: #ff4757;
  }

  .stage-icon {
    display: flex;
    align-items: center;
  }

  .spinning {
    animation: spin 1s linear infinite;
  }

  .stage-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    font-weight: 500;
  }

  .stage-connector {
    width: 16px;
    height: 1px;
    opacity: 0.3;
  }

  .stage-connector.pending { background: rgba(100, 100, 100, 0.5); }
  .stage-connector.completed { background: #00c896; }
  .stage-connector.in-progress { background: linear-gradient(90deg, #00c896, #00d4ff); }

  .job-id {
    display: flex;
    align-items: center;
    gap: 6px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: rgba(224, 224, 224, 0.4);
  }

  .job-id-value {
    color: rgba(0, 212, 255, 0.5);
  }

  .completed-section {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 10px;
    background: rgba(0, 200, 150, 0.1);
    border: 1px solid rgba(0, 200, 150, 0.2);
    border-radius: 8px;
  }

  .completed-icon {
    color: #00c896;
  }

  .completed-text {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    color: #00c896;
  }

  .external-link {
    display: flex;
    align-items: center;
    gap: 4px;
    margin-left: auto;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: rgba(0, 212, 255, 0.7);
    text-decoration: none;
  }

  .external-link:hover {
    color: #00d4ff;
  }

  .failed-section {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 8px;
    padding: 10px;
    background: rgba(255, 71, 87, 0.1);
    border: 1px solid rgba(255, 71, 87, 0.2);
    border-radius: 8px;
  }

  .failed-icon {
    color: #ff4757;
  }

  .failed-text {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    color: #ff4757;
  }

  .failed-error {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: rgba(255, 71, 87, 0.7);
    text-align: center;
  }

  .retry-btn {
    padding: 6px 12px;
    background: rgba(255, 71, 87, 0.15);
    border: 1px solid rgba(255, 71, 87, 0.25);
    border-radius: 4px;
    color: #ff4757;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    cursor: pointer;
    transition: all 0.2s ease;
  }

  .retry-btn:hover {
    background: rgba(255, 71, 87, 0.25);
  }
</style>