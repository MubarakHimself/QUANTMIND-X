<script lang="ts">
  import { createBubbler, stopPropagation, self } from 'svelte/legacy';

  const bubble = createBubbler();
  import { onMount } from 'svelte';
  import {
    Clock,
    Play,
    Plus,
    Trash2,
    Edit3,
    RefreshCw,
    ToggleLeft,
    ToggleRight,
    CheckCircle,
    XCircle,
    Calendar,
    Timer,
    X,
    Save,
    AlertCircle,
  } from 'lucide-svelte';
  import { cronStore, type CronJob } from '$lib/stores/cronStore';
  import * as memoryApi from '$lib/api/memory';

  let { onClose = () => {} } = $props();

  let showAddModal = $state(false);
  let showEditModal = false;
  let editingJob: CronJob | null = null;

  let newJob = $state({
    name: '',
    schedule: '',
    command: '',
    description: ''
  });

  // Common cron schedule presets
  const schedulePresets = [
    { label: 'Every minute', value: '* * * * *' },
    { label: 'Every 5 minutes', value: '*/5 * * * *' },
    { label: 'Every hour', value: '0 * * * *' },
    { label: 'Every day at midnight', value: '0 0 * * *' },
    { label: 'Every day at noon', value: '0 12 * * *' },
    { label: 'Every week', value: '0 0 * * 0' },
    { label: 'Every month', value: '0 0 1 * *' }
  ];

  // Subscribe to store
  let jobs = $derived($cronStore.jobs);
  let loading = $derived($cronStore.loading);
  let error = $derived($cronStore.error);

  onMount(() => {
    loadJobs();
  });

  async function loadJobs() {
    cronStore.setLoading(true);
    cronStore.setError(null);
    try {
      const jobs = await memoryApi.listCronJobs();
      cronStore.setJobs(jobs);
    } catch (e) {
      cronStore.setError(e instanceof Error ? e.message : 'Failed to load cron jobs');
    } finally {
      cronStore.setLoading(false);
    }
  }

  async function handleToggleJob(job: CronJob) {
    try {
      await memoryApi.toggleCronJob(job.id, !job.enabled);
      cronStore.toggleJob(job.id);
    } catch (e) {
      cronStore.setError(e instanceof Error ? e.message : 'Failed to toggle job');
    }
  }

  async function handleRunJob(job: CronJob) {
    try {
      cronStore.updateJob(job.id, { status: 'running' });
      await memoryApi.runCronJob(job.id);
      cronStore.updateJob(job.id, {
        status: 'success',
        lastStatus: 'success',
        lastRun: new Date().toISOString()
      });
    } catch (e) {
      cronStore.updateJob(job.id, {
        status: 'failed',
        lastStatus: 'failed'
      });
      cronStore.setError(e instanceof Error ? e.message : 'Failed to run job');
    }
  }

  async function handleAddJob() {
    if (!newJob.name || !newJob.schedule || !newJob.command) return;

    cronStore.setLoading(true);
    try {
      const job = await memoryApi.addCronJob({
        ...newJob,
        enabled: true,
        status: 'idle'
      });

      cronStore.addJob(job);
      showAddModal = false;
      resetNewJob();
    } catch (e) {
      cronStore.setError(e instanceof Error ? e.message : 'Failed to add job');
    } finally {
      cronStore.setLoading(false);
    }
  }

  async function handleDeleteJob(job: CronJob) {
    if (!confirm(`Delete job "${job.name}"?`)) return;

    try {
      await memoryApi.deleteCronJob(job.id);
      cronStore.removeJob(job.id);
    } catch (e) {
      cronStore.setError(e instanceof Error ? e.message : 'Failed to delete job');
    }
  }

  function resetNewJob() {
    newJob = {
      name: '',
      schedule: '',
      command: '',
      description: ''
    };
  }

  function getStatusBadge(job: CronJob) {
    if (job.status === 'running') {
      return { class: 'running', icon: Timer, label: 'Running' };
    }
    if (job.lastStatus === 'success') {
      return { class: 'success', icon: CheckCircle, label: 'Success' };
    }
    if (job.lastStatus === 'failed') {
      return { class: 'failed', icon: XCircle, label: 'Failed' };
    }
    return { class: 'idle', icon: Clock, label: 'Idle' };
  }

  function formatDateTime(timestamp?: string): string {
    if (!timestamp) return 'N/A';
    return new Date(timestamp).toLocaleString();
  }

  function getNextRunInfo(job: CronJob): string {
    if (job.nextRun) {
      const diff = new Date(job.nextRun).getTime() - Date.now();
      if (diff < 60000) return 'in < 1 min';
      if (diff < 3600000) return `in ${Math.floor(diff / 60000)}m`;
      if (diff < 86400000) return `in ${Math.floor(diff / 3600000)}h`;
      return `in ${Math.floor(diff / 86400000)}d`;
    }
    return 'N/A';
  }

  function getLastRunInfo(job: CronJob): string {
    if (job.lastRun) {
      const diff = Date.now() - new Date(job.lastRun).getTime();
      if (diff < 60000) return '< 1 min ago';
      if (diff < 3600000) return `${Math.floor(diff / 60000)}m ago`;
      if (diff < 86400000) return `${Math.floor(diff / 3600000)}h ago`;
      return `${Math.floor(diff / 86400000)}d ago`;
    }
    return 'Never';
  }
</script>

<div class="cron-panel-overlay" onclick={onClose}>
  <div class="cron-panel" onclick={stopPropagation(bubble('click'))}>
    <!-- Header -->
    <div class="panel-header">
      <div class="header-left">
        <Clock size={20} />
        <div>
          <h2>Cron Job Manager</h2>
          <span class="subtitle">{jobs.length} scheduled jobs</span>
        </div>
      </div>
      <div class="header-actions">
        <button class="icon-btn" onclick={loadJobs} title="Refresh" disabled={loading}>
          <RefreshCw size={16} class={loading ? 'spinning' : ''} />
        </button>
        <button class="icon-btn primary" onclick={() => showAddModal = true} title="Add Job">
          <Plus size={16} />
        </button>
        <button class="icon-btn" onclick={onClose} title="Close">
          <X size={16} />
        </button>
      </div>
    </div>

    <!-- Error Display -->
    {#if error}
    <div class="error-banner">
      <AlertCircle size={16} />
      <span>{error}</span>
      <button onclick={() => cronStore.setError(null)}><X size={14} /></button>
    </div>
    {/if}

    <!-- Job List -->
    <div class="job-list">
      {#if loading}
      <div class="loading-state">
        <RefreshCw size={32} class="spinning" />
        <span>Loading cron jobs...</span>
      </div>
      {:else if jobs.length === 0}
      <div class="empty-state">
        <Clock size={48} />
        <p>No cron jobs configured</p>
        <button class="btn primary" onclick={() => showAddModal = true}>
          <Plus size={14} /> Add Job
        </button>
      </div>
      {:else}
      {#each jobs as job}
      {@const SvelteComponent = getStatusBadge(job).icon}
      <div class="job-item" class:disabled={!job.enabled}>
        <div class="job-header">
          <div class="job-name">
            <button
              class="toggle-btn"
              onclick={() => handleToggleJob(job)}
              title={job.enabled ? 'Disable' : 'Enable'}
            >
              {#if job.enabled}
              <ToggleRight size={16} />
              {:else}
              <ToggleLeft size={16} />
              {/if}
            </button>
            <span>{job.name}</span>
          </div>
          <div class="job-actions">
            <button
              class="icon-btn"
              onclick={() => handleRunJob(job)}
              title="Run now"
              disabled={job.status === 'running'}
            >
              <Play size={14} class={job.status === 'running' ? 'spinning' : ''} />
            </button>
            <button class="icon-btn danger" onclick={() => handleDeleteJob(job)} title="Delete">
              <Trash2 size={14} />
            </button>
          </div>
        </div>

        {#if job.description}
        <p class="job-description">{job.description}</p>
        {/if}

        <div class="job-details">
          <div class="job-schedule">
            <code>{job.schedule}</code>
            <span class="schedule-label">Schedule</span>
          </div>
          <div class="job-command">
            <code>{job.command}</code>
          </div>
        </div>

        <div class="job-status">
          <div class="status-item">
            <SvelteComponent size={12} class={getStatusBadge(job).class} />
            <span class="status-label">{getStatusBadge(job).label}</span>
          </div>
          <div class="status-item">
            <Calendar size={12} />
            <span title={formatDateTime(job.lastRun)}>Last: {getLastRunInfo(job)}</span>
          </div>
          <div class="status-item">
            <Timer size={12} />
            <span title={formatDateTime(job.nextRun)}>Next: {getNextRunInfo(job)}</span>
          </div>
          {#if job.executionTime}
          <div class="status-item">
            <span>Exec: {job.executionTime}ms</span>
          </div>
          {/if}
        </div>
      </div>
      {/each}
      {/if}
    </div>
  </div>
</div>

<!-- Add Job Modal -->
{#if showAddModal}
<div class="modal-overlay" onclick={self(() => showAddModal = false)}>
  <div class="modal">
    <div class="modal-header">
      <h3>Add Cron Job</h3>
      <button onclick={() => showAddModal = false}><X size={18} /></button>
    </div>
    <div class="modal-body">
      <div class="form-group">
        <label>Job Name *</label>
        <input type="text" bind:value={newJob.name} placeholder="my-scheduled-task" />
      </div>
      <div class="form-group">
        <label>Schedule (Cron Expression) *</label>
        <select onchange={(e) => newJob.schedule = e.target.value}>
          <option value="">Select a preset...</option>
          {#each schedulePresets as preset}
          <option value={preset.value}>{preset.label} ({preset.value})</option>
          {/each}
        </select>
        <input
          type="text"
          bind:value={newJob.schedule}
          placeholder="* * * * *"
          class="mt-2"
        />
        <small>Format: minute hour day month weekday</small>
      </div>
      <div class="form-group">
        <label>Command *</label>
        <input
          type="text"
          bind:value={newJob.command}
          placeholder="/path/to/script.sh"
        />
      </div>
      <div class="form-group">
        <label>Description</label>
        <textarea
          bind:value={newJob.description}
          rows="2"
          placeholder="What this job does..."
        ></textarea>
      </div>
    </div>
    <div class="modal-footer">
      <button class="btn secondary" onclick={() => showAddModal = false}>Cancel</button>
      <button class="btn primary" onclick={handleAddJob} disabled={!newJob.name || !newJob.schedule || !newJob.command || loading}>
        {#if loading}
        <RefreshCw size={14} class="spinning" />
        {:else}
        <Save size={14} />
        {/if}
        Add Job
      </button>
    </div>
  </div>
</div>
{/if}

<style>
  .cron-panel-overlay {
    position: fixed;
    inset: 0;
    background: rgba(0, 0, 0, 0.7);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 1000;
    padding: 20px;
  }

  .cron-panel {
    background: var(--color-bg-surface);
    border-radius: 12px;
    width: 800px;
    max-width: 100%;
    height: 80vh;
    max-height: 90vh;
    display: flex;
    flex-direction: column;
    box-shadow: 0 25px 50px rgba(0, 0, 0, 0.5);
    border: 1px solid var(--color-border-subtle);
  }

  .panel-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 16px 20px;
    border-bottom: 1px solid var(--color-border-subtle);
    background: var(--color-bg-base);
    border-radius: 12px 12px 0 0;
  }

  .header-left {
    display: flex;
    align-items: center;
    gap: 12px;
    color: var(--color-accent-cyan);
  }

  .header-left h2 {
    margin: 0;
    font-size: 16px;
    color: var(--color-text-primary);
  }

  .subtitle {
    font-size: 11px;
    color: var(--color-text-muted);
  }

  .header-actions {
    display: flex;
    gap: 6px;
  }

  .error-banner {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 20px;
    background: rgba(239, 68, 68, 0.1);
    border-bottom: 1px solid rgba(239, 68, 68, 0.3);
    color: var(--color-accent-red);
    font-size: 12px;
  }

  .error-banner button {
    margin-left: auto;
    background: none;
    border: none;
    color: inherit;
    cursor: pointer;
  }

  .job-list {
    flex: 1;
    overflow-y: auto;
    padding: 12px;
  }

  .loading-state,
  .empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 60px 20px;
    color: var(--color-text-muted);
    gap: 16px;
  }

  .job-item {
    padding: 14px;
    background: var(--color-bg-elevated);
    border: 1px solid var(--color-border-subtle);
    border-radius: 8px;
    margin-bottom: 10px;
    transition: all 0.15s;
  }

  .job-item:hover {
    border-color: var(--color-border-medium);
  }

  .job-item.disabled {
    opacity: 0.6;
  }

  .job-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
  }

  .job-name {
    display: flex;
    align-items: center;
    gap: 10px;
  }

  .toggle-btn {
    display: flex;
    align-items: center;
    background: none;
    border: none;
    color: var(--color-text-muted);
    cursor: pointer;
    padding: 0;
    transition: color 0.15s;
  }

  .toggle-btn:hover {
    color: var(--color-accent-cyan);
  }

  .job-name span {
    font-weight: 500;
    color: var(--color-text-primary);
  }

  .job-actions {
    display: flex;
    gap: 4px;
  }

  .job-description {
    margin: 0 0 10px 26px;
    font-size: 12px;
    color: var(--color-text-muted);
  }

  .job-details {
    margin: 0 0 10px 26px;
    display: flex;
    flex-direction: column;
    gap: 6px;
  }

  .job-schedule {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .job-schedule code,
  .job-command code {
    background: var(--color-bg-elevated);
    padding: 3px 8px;
    border-radius: 4px;
    font-size: 11px;
    color: var(--color-accent-amber);
  }

  .schedule-label {
    font-size: 10px;
    color: var(--color-text-muted);
    text-transform: uppercase;
  }

  .job-status {
    display: flex;
    gap: 16px;
    padding: 8px 12px;
    background: var(--color-bg-elevated);
    border-radius: 6px;
    font-size: 11px;
  }

  .status-item {
    display: flex;
    align-items: center;
    gap: 6px;
    color: var(--color-text-secondary);
  }

  .status-item .running {
    color: var(--color-accent-cyan);
    animation: pulse 1s infinite;
  }

  .status-item .success {
    color: var(--color-accent-green);
  }

  .status-item .failed {
    color: var(--color-accent-red);
  }

  .status-item .idle {
    color: var(--color-text-muted);
  }

  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
  }

  .icon-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 32px;
    height: 32px;
    background: transparent;
    border: none;
    border-radius: 6px;
    color: var(--color-text-muted);
    cursor: pointer;
    transition: all 0.15s;
  }

  .icon-btn:hover {
    background: var(--color-bg-elevated);
    color: var(--color-text-primary);
  }

  .icon-btn.primary {
    background: var(--color-accent-cyan);
    color: var(--color-bg-base);
  }

  .icon-btn.danger:hover {
    background: rgba(239, 68, 68, 0.2);
    color: var(--color-accent-red);
  }

  .btn {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 8px 16px;
    border-radius: 6px;
    font-size: 13px;
    font-weight: 500;
    border: none;
    cursor: pointer;
  }

  .btn.primary {
    background: var(--color-accent-cyan);
    color: var(--color-bg-base);
  }

  .btn.secondary {
    background: var(--color-bg-elevated);
    color: var(--color-text-secondary);
  }

  .spinning {
    animation: spin 1s linear infinite;
  }

  @keyframes spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
  }

  /* Modal */
  .modal-overlay {
    position: absolute;
    inset: 0;
    background: rgba(0, 0, 0, 0.7);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 10;
  }

  .modal {
    background: var(--color-bg-surface);
    border-radius: 12px;
    width: 480px;
    max-width: 90%;
  }

  .modal-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 16px 20px;
    border-bottom: 1px solid var(--color-border-subtle);
  }

  .modal-header h3 {
    margin: 0;
    font-size: 16px;
    color: var(--color-text-primary);
  }

  .modal-body {
    padding: 20px;
  }

  .form-group {
    margin-bottom: 16px;
  }

  .form-group label {
    display: block;
    margin-bottom: 6px;
    font-size: 12px;
    color: var(--color-text-muted);
  }

  .form-group input,
  .form-group select,
  .form-group textarea {
    width: 100%;
    padding: 8px 12px;
    background: var(--color-bg-elevated);
    border: 1px solid var(--color-border-subtle);
    border-radius: 6px;
    color: var(--color-text-primary);
    font-size: 13px;
  }

  .form-group textarea {
    min-height: 60px;
    resize: vertical;
  }

  .form-group small {
    display: block;
    margin-top: 4px;
    font-size: 10px;
    color: var(--color-text-muted);
  }

  .mt-2 {
    margin-top: 8px;
  }

  .modal-footer {
    display: flex;
    justify-content: flex-end;
    gap: 8px;
    padding: 16px 20px;
    border-top: 1px solid var(--color-border-subtle);
  }
</style>
