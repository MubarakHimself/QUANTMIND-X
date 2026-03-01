<script lang="ts">
  import { onMount } from 'svelte';
  import { Loader2, Play, CheckCircle, XCircle, AlertCircle } from 'lucide-svelte';
  import { API_CONFIG } from '$lib/config/api';

  interface NPRDJob {
    id: string;
    status: string;
    youtube_url: string;
    created_at: string;
    result?: any;
  }

  const API_BASE = API_CONFIG.API_BASE;

  let youtubeUrl = '';
  let loading = false;
  let error = '';
  let jobs: NPRDJob[] = [];

  async function submitJob() {
    if (!youtubeUrl) return;

    loading = true;
    error = '';

    try {
      const response = await fetch(`${API_BASE}/nprd/process`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ youtube_url: youtubeUrl })
      });

      if (!response.ok) {
        throw new Error('Failed to submit job');
      }

      const result = await response.json();
      jobs = [result, ...jobs];
      youtubeUrl = '';
    } catch (e: any) {
      error = e.message || 'Failed to submit job';
    } finally {
      loading = false;
    }
  }

  async function refreshJobs() {
    // For now just show the submitted jobs
  }

  function getStatusIcon(status: string) {
    switch (status) {
      case 'completed': return CheckCircle;
      case 'failed': return XCircle;
      case 'processing': return Loader2;
      default: return AlertCircle;
    }
  }

  function getStatusColor(status: string): string {
    switch (status) {
      case 'completed': return 'var(--accent-success)';
      case 'failed': return 'var(--accent-danger)';
      case 'processing': return 'var(--accent-warning)';
      default: return 'var(--text-muted)';
    }
  }
</script>

<div class="nprd-workflow">
  <div class="header">
    <h3>NPRD Workflow</h3>
    <p>Process YouTube trading videos into Expert Advisors</p>
  </div>

  <!-- Submit Form -->
  <div class="submit-form">
    <input
      type="text"
      bind:value={youtubeUrl}
      placeholder="Enter YouTube URL..."
      disabled={loading}
      on:keydown={(e) => e.key === 'Enter' && submitJob()}
    />
    <button on:click={submitJob} disabled={loading || !youtubeUrl}>
      {#if loading}
        <Loader2 size={14} class="spin" />
        Processing...
      {:else}
        <Play size={14} />
        Generate EA
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
    <h4>Recent Jobs</h4>

    {#if jobs.length === 0}
      <div class="empty-state">
        <p>No jobs yet. Submit a YouTube URL to get started.</p>
      </div>
    {:else}
      <div class="jobs-list">
        {#each jobs as job}
          <div class="job-card">
            <div class="job-status" style="color: {getStatusColor(job.status)}">
              <svelte:component this={getStatusIcon(job.status)} size={16} />
              <span>{job.status}</span>
            </div>
            <div class="job-url">{job.youtube_url}</div>
            <div class="job-time">
              {new Date(job.created_at).toLocaleString()}
            </div>
          </div>
        {/each}
      </div>
    {/if}
  </div>
</div>

<style>
  .nprd-workflow {
    display: flex;
    flex-direction: column;
    gap: 20px;
  }

  .header h3 {
    margin: 0;
    font-size: 18px;
    color: var(--text-primary);
  }

  .header p {
    margin: 4px 0 0;
    font-size: 12px;
    color: var(--text-secondary);
  }

  .submit-form {
    display: flex;
    gap: 8px;
  }

  .submit-form input {
    flex: 1;
    padding: 10px 14px;
    background: var(--bg-tertiary);
    border: 1px solid var(--border-subtle);
    border-radius: 6px;
    color: var(--text-primary);
    font-size: 13px;
  }

  .submit-form input:focus {
    outline: none;
    border-color: var(--accent-primary);
  }

  .submit-form button {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 10px 16px;
    background: var(--accent-primary);
    color: var(--bg-primary);
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
    border: 1px solid var(--accent-danger);
    border-radius: 6px;
    color: var(--accent-danger);
    font-size: 12px;
  }

  .jobs-section h4 {
    margin: 0 0 12px;
    font-size: 14px;
    color: var(--text-primary);
  }

  .empty-state {
    padding: 24px;
    text-align: center;
    background: var(--bg-tertiary);
    border-radius: 8px;
    color: var(--text-muted);
    font-size: 12px;
  }

  .jobs-list {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .job-card {
    padding: 12px;
    background: var(--bg-tertiary);
    border: 1px solid var(--border-subtle);
    border-radius: 6px;
  }

  .job-status {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 12px;
    font-weight: 500;
    text-transform: capitalize;
    margin-bottom: 4px;
  }

  .job-url {
    font-size: 11px;
    color: var(--text-secondary);
    font-family: monospace;
    word-break: break-all;
  }

  .job-time {
    margin-top: 4px;
    font-size: 10px;
    color: var(--text-muted);
  }

  :global(.spin) {
    animation: spin 1s linear infinite;
  }

  @keyframes spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
  }
</style>
