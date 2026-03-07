<script lang="ts">
  import { createEventDispatcher } from "svelte";
  import { X, AlertCircle, CheckCircle, Clock, Pause, Play, RefreshCw, ExternalLink, FileVideo, Zap } from "lucide-svelte";

  export let videoIngestUrl = "";
  export let videoIngestName = "";
  export let videoIngestQueue: Array<{
    id: string;
    name: string;
    status: string;
    progress: number;
    error?: string;
  }> = [];
  export let isOpen = false;

  const dispatch = createEventDispatcher();

  // Error state
  let submitError: string | null = null;
  let isSubmitting = false;

  function getStatusColor(status: string): string {
    const colors: Record<string, string> = {
      primal: "#10b981",
      ready: "#3b82f6",
      pending: "#f59e0b",
      quarantined: "#ef4444",
      paused: "#6b7280",
      processing: "#8b5cf6",
      completed: "#10b981",
      failed: "#ef4444",
    };
    return colors[status] || "#6b7280";
  }

  function getStatusIcon(status: string) {
    switch (status) {
      case 'primal':
      case 'completed':
        return CheckCircle;
      case 'processing':
        return Play;
      case 'pending':
        return Clock;
      case 'paused':
        return Pause;
      case 'quarantined':
      case 'failed':
        return AlertCircle;
      default:
        return Clock;
    }
  }

  function getProgressColor(progress: number, status: string): string {
    if (status === 'failed' || status === 'quarantined') return '#ef4444';
    if (progress === 100) return '#10b981';
    return 'var(--accent-primary)';
  }

  function formatProgress(progress: number): string {
    return `${Math.round(progress)}%`;
  }

  function handleClose() {
    submitError = null;
    dispatch("close");
  }

  async function handleSubmit() {
    submitError = null;

    if (!videoIngestUrl) {
      submitError = "Please enter a YouTube URL or playlist";
      return;
    }

    if (!videoIngestName) {
      submitError = "Please enter a strategy name";
      return;
    }

    // Validate URL format
    if (!videoIngestUrl.includes('youtube.com') && !videoIngestUrl.includes('youtu.be')) {
      submitError = "Please enter a valid YouTube URL";
      return;
    }

    isSubmitting = true;

    try {
      dispatch("submit", { url: videoIngestUrl, name: videoIngestName });
      // Reset form on successful submit
      videoIngestUrl = "";
      videoIngestName = "";
    } catch (e: any) {
      submitError = e.message || "Failed to start processing";
    } finally {
      isSubmitting = false;
    }
  }
</script>

{#if isOpen}
  <div class="modal-overlay">
    <div class="modal">
      <div class="modal-header">
        <h2>Video Ingest</h2>
        <button on:click={handleClose}><X size={20} /></button>
      </div>
      <div class="modal-body">
        {#if submitError}
          <div class="error-banner">
            <AlertCircle size={14} />
            <span>{submitError}</span>
            <button class="dismiss-btn" on:click={() => submitError = null}>x</button>
          </div>
        {/if}

        <div class="form-group">
          <label for="videoIngest-url">
            <FileVideo size={12} />
            YouTube URL or Playlist
          </label>
          <input
            id="videoIngest-url"
            type="text"
            placeholder="https://youtube.com/watch?v=..."
            bind:value={videoIngestUrl}
            class:error={submitError?.includes('URL')}
          />
        </div>
        <div class="form-group">
          <label for="videoIngest-name">
            <Zap size={12} />
            Strategy Name
          </label>
          <input
            id="videoIngest-name"
            type="text"
            placeholder="ICT Scalper v3"
            bind:value={videoIngestName}
            class:error={submitError?.includes('name')}
          />
        </div>
        {#if videoIngestQueue.length > 0}
          <div class="queue-section">
            <div class="queue-header">
              <h4>Processing Queue</h4>
              <span class="queue-count">{videoIngestQueue.length} item{videoIngestQueue.length > 1 ? 's' : ''}</span>
            </div>
            {#each videoIngestQueue as job}
              <div class="queue-item">
                <div class="queue-item-header">
                  <span class="job-name">{job.name}</span>
                  <span class="status" style="color: {getStatusColor(job.status)}">
                    <svelte:component this={getStatusIcon(job.status)} size={12} />
                    {job.status}
                  </span>
                </div>
                <div class="progress-container">
                  <div class="progress-bar">
                    <div
                      class="progress-fill"
                      style="width: {job.progress}%; background: {getProgressColor(job.progress, job.status)}"
                    ></div>
                  </div>
                  <span class="progress-text">{formatProgress(job.progress)}</span>
                </div>
                {#if job.error}
                  <div class="job-error">
                    <AlertCircle size={10} />
                    <span>{job.error}</span>
                  </div>
                {/if}
              </div>
            {/each}
          </div>
        {/if}
      </div>
      <div class="modal-footer">
        <button
          class="btn secondary"
          on:click={handleClose}
          disabled={isSubmitting}>Cancel</button
        >
        <button class="btn primary" on:click={handleSubmit} disabled={isSubmitting}>
          {#if isSubmitting}
            <RefreshCw size={14} class="spin" />
            Processing...
          {:else}
            <Play size={14} />
            Start Processing
          {/if}
        </button>
      </div>
    </div>
  </div>
{/if}

<style>
  .modal-overlay {
    position: fixed;
    inset: 0;
    background: rgba(0, 0, 0, 0.7);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 100;
  }

  .modal {
    background: var(--bg-secondary);
    border: 1px solid var(--border-strong);
    border-radius: 8px;
    width: 480px;
    max-width: 90%;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
  }

  .modal-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0 16px;
    height: 48px;
    border-bottom: 1px solid var(--border-subtle);
  }

  .modal-header h2 {
    margin: 0;
    font-size: 16px;
    color: var(--text-primary);
  }

  .modal-header button {
    background: none;
    border: none;
    color: var(--text-muted);
    cursor: pointer;
  }

  .modal-body {
    padding: 20px;
  }

  .form-group {
    margin-bottom: 16px;
  }

  .form-group label {
    display: flex;
    align-items: center;
    gap: 6px;
    margin-bottom: 6px;
    font-size: 12px;
    color: var(--text-muted);
  }

  .form-group input {
    width: 100%;
    padding: 10px 12px;
    background: var(--bg-tertiary);
    border: 1px solid var(--border-subtle);
    border-radius: 6px;
    color: var(--text-primary);
    font-size: 13px;
    transition: border-color 0.15s;
  }

  .form-group input:focus {
    outline: none;
    border-color: var(--accent-primary);
  }

  .form-group input.error {
    border-color: var(--accent-danger);
  }

  .error-banner {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 10px 12px;
    background: rgba(239, 68, 68, 0.1);
    border: 1px solid rgba(239, 68, 68, 0.3);
    border-radius: 6px;
    margin-bottom: 16px;
    font-size: 12px;
    color: var(--accent-danger);
  }

  .dismiss-btn {
    margin-left: auto;
    background: none;
    border: none;
    color: var(--text-muted);
    cursor: pointer;
    font-size: 14px;
  }

  .dismiss-btn:hover {
    color: var(--text-primary);
  }

  .queue-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
  }

  .queue-count {
    font-size: 11px;
    color: var(--text-muted);
    background: var(--bg-tertiary);
    padding: 2px 8px;
    border-radius: 10px;
  }

  .modal-footer {
    display: flex;
    justify-content: flex-end;
    gap: 8px;
    padding: 16px 20px;
    border-top: 1px solid var(--border-subtle);
  }

  .btn {
    padding: 8px 16px;
    border-radius: 6px;
    font-size: 13px;
    cursor: pointer;
  }

  .btn.secondary {
    background: var(--bg-tertiary);
    border: 1px solid var(--border-subtle);
    color: var(--text-secondary);
  }

  .btn.primary {
    background: var(--accent-primary);
    border: none;
    color: var(--bg-primary);
  }

  .queue-section {
    margin-top: 16px;
    padding-top: 16px;
    border-top: 1px solid var(--border-subtle);
  }

  .queue-section h4 {
    margin: 0 0 12px;
    font-size: 13px;
    color: var(--text-primary);
  }

  .queue-item {
    padding: 12px;
    background: var(--bg-tertiary);
    border-radius: 6px;
    margin-bottom: 8px;
  }

  .queue-item-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
  }

  .job-name {
    font-size: 13px;
    font-weight: 500;
    color: var(--text-primary);
  }

  .queue-item .status {
    display: flex;
    align-items: center;
    gap: 4px;
    font-size: 11px;
    text-transform: capitalize;
  }

  .progress-container {
    display: flex;
    align-items: center;
    gap: 12px;
  }

  .progress-bar {
    flex: 1;
    height: 6px;
    background: var(--border-subtle);
    border-radius: 3px;
    overflow: hidden;
  }

  .progress-fill {
    height: 100%;
    border-radius: 3px;
    transition: width 0.3s ease;
  }

  .progress-text {
    font-size: 11px;
    font-family: monospace;
    color: var(--text-muted);
    min-width: 40px;
    text-align: right;
  }

  .job-error {
    display: flex;
    align-items: center;
    gap: 6px;
    margin-top: 8px;
    padding: 6px 8px;
    background: rgba(239, 68, 68, 0.1);
    border-radius: 4px;
    font-size: 11px;
    color: var(--accent-danger);
  }

  .btn {
    display: flex;
    align-items: center;
    gap: 6px;
  }

  :global(.spin) {
    animation: spin 1s linear infinite;
  }

  @keyframes spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
  }
</style>
