<script lang="ts">
  import { stopPropagation } from 'svelte/legacy';

  import { onMount, onDestroy } from 'svelte';
  import { createEventDispatcher } from 'svelte';
  import {
    Play, Pause, Square, RefreshCw, Plus, Trash2,
    CheckCircle, XCircle, Clock, AlertTriangle, List,
    ChevronDown, ChevronRight, X
  } from 'lucide-svelte';
  import { batchService, type BatchStatusResponse, type BatchStatsResponse, type BatchListItem, type BatchResultResponse } from '../services/batchService';

  const dispatch = createEventDispatcher();

  // State
  let stats: BatchStatsResponse | null = $state(null);
  let batches: BatchListItem[] = $state([]);
  let selectedBatchId: string | null = $state(null);
  let selectedBatchResults: BatchResultResponse | null = $state(null);
  let isLoading = $state(false);
  let error: string | null = $state(null);
  let refreshInterval: ReturnType<typeof setInterval> | null = null;
  let isExpanded = $state(true);

  // Submit form state
  let showSubmitForm = $state(false);
  let newBatchPayload = $state('');
  let newBatchPriority: 'LOW' | 'NORMAL' | 'HIGH' | 'CRITICAL' = $state('NORMAL');

  // Computed
  let selectedBatch = $derived(batches.find(b => b.batch_id === selectedBatchId));
  let statusColor = $derived((status: string) => {
    switch (status) {
      case 'completed': return 'var(--color-accent-green)';
      case 'processing': return 'var(--color-accent-cyan)';
      case 'pending': return 'var(--color-accent-amber)';
      case 'failed': return 'var(--color-accent-red)';
      case 'cancelled': return 'var(--color-text-muted)';
      default: return 'var(--color-text-muted)';
    }
  });

  let progressPercent = $derived(selectedBatch
    ? Math.round(((selectedBatch.completed_count + selectedBatch.failed_count) / selectedBatch.total_items) * 100)
    : 0);

  // Functions
  async function loadStats() {
    try {
      stats = await batchService.getStats();
    } catch (e) {
      console.error('Failed to load stats:', e);
    }
  }

  async function loadBatches() {
    try {
      const result = await batchService.listBatches();
      batches = result.batches.sort((a, b) => b.created_at - a.created_at);
    } catch (e) {
      console.error('Failed to load batches:', e);
    }
  }

  async function loadBatchResults(batchId: string) {
    try {
      selectedBatchResults = await batchService.getBatchResults(batchId);
    } catch (e) {
      console.error('Failed to load batch results:', e);
    }
  }

  async function refresh() {
    await Promise.all([loadStats(), loadBatches()]);
    if (selectedBatchId) {
      await loadBatchResults(selectedBatchId);
    }
  }

  async function handleStart() {
    isLoading = true;
    try {
      await batchService.start();
      await refresh();
    } catch (e: any) {
      error = e.message;
    } finally {
      isLoading = false;
    }
  }

  async function handleStop() {
    isLoading = true;
    try {
      await batchService.stop();
      await refresh();
    } catch (e: any) {
      error = e.message;
    } finally {
      isLoading = false;
    }
  }

  async function handleSubmitBatch() {
    if (!newBatchPayload.trim()) {
      error = 'Please enter payloads';
      return;
    }

    isLoading = true;
    error = null;

    try {
      // Parse payloads (JSON array or single item)
      let payloads: any[];
      try {
        payloads = JSON.parse(newBatchPayload);
        if (!Array.isArray(payloads)) {
          payloads = [payloads];
        }
      } catch {
        // If not JSON, treat as single string item
        payloads = [newBatchPayload];
      }

      await batchService.submitBatch({
        payloads,
        priority: newBatchPriority
      });

      newBatchPayload = '';
      showSubmitForm = false;
      await refresh();
    } catch (e: any) {
      error = e.message;
    } finally {
      isLoading = false;
    }
  }

  async function handleDeleteBatch(batchId: string) {
    try {
      await batchService.deleteBatch(batchId);
      if (selectedBatchId === batchId) {
        selectedBatchId = null;
        selectedBatchResults = null;
      }
      await refresh();
    } catch (e: any) {
      error = e.message;
    }
  }

  function selectBatch(batchId: string) {
    selectedBatchId = selectedBatchId === batchId ? null : batchId;
    if (selectedBatchId) {
      loadBatchResults(selectedBatchId);
    }
  }

  function formatTime(timestamp: number): string {
    return new Date(timestamp * 1000).toLocaleString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit'
    });
  }

  function formatDuration(ms: number): string {
    if (ms < 1000) return `${ms.toFixed(0)}ms`;
    if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
    return `${(ms / 60000).toFixed(1)}m`;
  }

  onMount(() => {
    refresh();
    refreshInterval = setInterval(refresh, 2000);
  });

  onDestroy(() => {
    if (refreshInterval) {
      clearInterval(refreshInterval);
    }
  });
</script>

<div class="batch-panel">
  <!-- Header -->
  <div class="panel-header" onclick={() => isExpanded = !isExpanded}>
    <div class="header-left">
      {#if isExpanded}
        <ChevronDown size={18} />
      {:else}
        <ChevronRight size={18} />
      {/if}
      <List size={18} />
      <h3>Batch Processing</h3>
      {#if stats?.running}
        <span class="status-badge running">Running</span>
      {:else}
        <span class="status-badge stopped">Stopped</span>
      {/if}
    </div>
    <div class="header-actions">
      <button class="icon-btn" onclick={stopPropagation(refresh)} title="Refresh">
        <span class:spinning={isLoading === true}><RefreshCw size={16} /></span>
      </button>
    </div>
  </div>

  {#if isExpanded}
    <!-- Controls -->
    <div class="controls-bar">
      <div class="control-group">
        {#if stats?.running}
          <button class="control-btn" onclick={handleStop} disabled={isLoading}>
            <Pause size={14} />
            <span>Pause</span>
          </button>
        {:else}
          <button class="control-btn primary" onclick={handleStart} disabled={isLoading}>
            <Play size={14} />
            <span>Start</span>
          </button>
        {/if}
      </div>
      <button class="control-btn" onclick={() => showSubmitForm = !showSubmitForm}>
        <Plus size={14} />
        <span>New Batch</span>
      </button>
    </div>

    <!-- Error Display -->
    {#if error}
      <div class="error-banner">
        <AlertTriangle size={16} />
        <span>{error}</span>
        <button class="icon-btn small" onclick={() => error = null}>
          <X size={14} />
        </button>
      </div>
    {/if}

    <!-- Submit Form -->
    {#if showSubmitForm}
      <div class="submit-form">
        <div class="form-row">
          <label>Payloads (JSON or text):</label>
          <textarea
            bind:value={newBatchPayload}
            placeholder='["item1", "item2", "item3"]'
            rows="3"
          ></textarea>
        </div>
        <div class="form-row">
          <label>Priority:</label>
          <select bind:value={newBatchPriority}>
            <option value="LOW">Low</option>
            <option value="NORMAL">Normal</option>
            <option value="HIGH">High</option>
            <option value="CRITICAL">Critical</option>
          </select>
        </div>
        <div class="form-actions">
          <button class="control-btn" onclick={() => showSubmitForm = false}>
            Cancel
          </button>
          <button class="control-btn primary" onclick={handleSubmitBatch} disabled={isLoading}>
            Submit Batch
          </button>
        </div>
      </div>
    {/if}

    <!-- Stats Overview -->
    {#if stats}
      <div class="stats-grid">
        <div class="stat-item">
          <span class="stat-label">Queue Size</span>
          <span class="stat-value">{stats.queue_size}</span>
        </div>
        <div class="stat-item">
          <span class="stat-label">Active</span>
          <span class="stat-value">{stats.active_items}</span>
        </div>
        <div class="stat-item">
          <span class="stat-label">Total Batches</span>
          <span class="stat-value">{stats.total_batches}</span>
        </div>
        <div class="stat-item">
          <span class="stat-label">Rate Limit</span>
          <span class="stat-value">{stats.rate_limit_rps} req/s</span>
        </div>
        <div class="stat-item">
          <span class="stat-label">Max Concurrent</span>
          <span class="stat-value">{stats.max_concurrent}</span>
        </div>
        <div class="stat-item">
          <span class="stat-label">Burst Size</span>
          <span class="stat-value">{stats.burst_size}</span>
        </div>
      </div>
    {/if}

    <!-- Batches List -->
    <div class="batches-list">
      <div class="list-header">
        <span>Recent Batches</span>
        <span class="batch-count">{batches.length}</span>
      </div>

      {#if batches.length === 0}
        <div class="empty-state">
          <Clock size={24} />
          <p>No batches yet</p>
        </div>
      {:else}
        <div class="batch-items">
          {#each batches as batch (batch.batch_id)}
            <div
              class="batch-item"
              class:selected={selectedBatchId === batch.batch_id}
              onclick={() => selectBatch(batch.batch_id)}
            >
              <div class="batch-main">
                <span class="batch-id">{batch.batch_id.slice(0, 8)}...</span>
                <span class="batch-status" style="color: {statusColor(batch.status)}">
                  {#if batch.status === 'completed'}
                    <CheckCircle size={14} />
                  {:else if batch.status === 'failed'}
                    <XCircle size={14} />
                  {:else if batch.status === 'processing'}
                    <RefreshCw size={14} class="spinning" />
                  {:else}
                    <Clock size={14} />
                  {/if}
                  {batch.status}
                </span>
                <span class="batch-time">{formatTime(batch.created_at)}</span>
              </div>
              <div class="batch-progress">
                <div class="progress-bar">
                  <div
                    class="progress-fill"
                    style="width: {((batch.completed_count + batch.failed_count) / batch.total_items) * 100}%"
                    class:failed={batch.failed_count > 0}
                  ></div>
                </div>
                <span class="progress-text">
                  {batch.completed_count + batch.failed_count} / {batch.total_items}
                </span>
                <button
                  class="icon-btn small danger"
                  onclick={stopPropagation(() => handleDeleteBatch(batch.batch_id))}
                  title="Delete batch"
                >
                  <Trash2 size={12} />
                </button>
              </div>
            </div>
          {/each}
        </div>
      {/if}
    </div>

    <!-- Batch Details -->
    {#if selectedBatchId && selectedBatchResults}
      <div class="batch-details">
        <div class="details-header">
          <h4>Batch Results: {selectedBatchId.slice(0, 8)}</h4>
          <span class="details-status" style="color: {statusColor(selectedBatchResults.metadata?.status || '')}">
            {selectedBatchResults.metadata?.status || 'unknown'}
          </span>
        </div>

        <div class="details-stats">
          <div class="detail-stat">
            <span class="label">Total</span>
            <span class="value">{selectedBatchResults.total}</span>
          </div>
          <div class="detail-stat success">
            <CheckCircle size={14} />
            <span class="label">Successful</span>
            <span class="value">{selectedBatchResults.successful}</span>
          </div>
          <div class="detail-stat danger">
            <XCircle size={14} />
            <span class="label">Failed</span>
            <span class="value">{selectedBatchResults.failed}</span>
          </div>
          <div class="detail-stat">
            <Clock size={14} />
            <span class="label">Duration</span>
            <span class="value">{formatDuration(selectedBatchResults.duration)}</span>
          </div>
        </div>

        {#if selectedBatchResults.errors.length > 0}
          <div class="errors-section">
            <h5>Errors ({selectedBatchResults.errors.length})</h5>
            <div class="errors-list">
              {#each selectedBatchResults.errors as err}
                <div class="error-item">
                  <span class="error-id">{err.id.slice(0, 8)}...</span>
                  <span class="error-msg">{err.error}</span>
                </div>
              {/each}
            </div>
          </div>
        {/if}

        {#if selectedBatchResults.results.length > 0}
          <div class="results-section">
            <h5>Results ({selectedBatchResults.results.length})</h5>
            <div class="results-list">
              {#each selectedBatchResults.results.slice(0, 5) as result}
                <div class="result-item">
                  <pre>{JSON.stringify(result, null, 2)}</pre>
                </div>
              {/each}
              {#if selectedBatchResults.results.length > 5}
                <div class="more-results">
                  + {selectedBatchResults.results.length - 5} more results
                </div>
              {/if}
            </div>
          </div>
        {/if}
      </div>
    {/if}
  {/if}
</div>

<style>
  .batch-panel {
    background: var(--color-bg-surface);
    border-radius: 8px;
    border: 1px solid var(--border-color);
    overflow: hidden;
  }

  .panel-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 12px 16px;
    background: var(--color-bg-elevated);
    cursor: pointer;
    user-select: none;
  }

  .panel-header:hover {
    background: var(--bg-hover);
  }

  .header-left {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .header-left h3 {
    margin: 0;
    font-size: 14px;
    font-weight: 600;
  }

  .status-badge {
    font-size: 10px;
    padding: 2px 8px;
    border-radius: 10px;
    text-transform: uppercase;
    font-weight: 600;
  }

  .status-badge.running {
    background: var(--color-accent-green);
    color: white;
  }

  .status-badge.stopped {
    background: var(--color-text-muted);
    color: white;
  }

  .header-actions {
    display: flex;
    gap: 4px;
  }

  .controls-bar {
    display: flex;
    justify-content: space-between;
    padding: 12px 16px;
    border-bottom: 1px solid var(--border-color);
    gap: 8px;
    flex-wrap: wrap;
  }

  .control-group {
    display: flex;
    gap: 8px;
  }

  .control-btn {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 6px 12px;
    border: 1px solid var(--border-color);
    border-radius: 4px;
    background: var(--color-bg-base);
    color: var(--color-text-primary);
    font-size: 12px;
    cursor: pointer;
    transition: all 0.2s;
  }

  .control-btn:hover:not(:disabled) {
    background: var(--bg-hover);
  }

  .control-btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .control-btn.primary {
    background: var(--color-accent-cyan);
    border-color: var(--color-accent-cyan);
    color: white;
  }

  .control-btn.primary:hover:not(:disabled) {
    background: var(--accent-primary-dark, var(--color-accent-cyan));
  }

  .icon-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 28px;
    height: 28px;
    border: none;
    border-radius: 4px;
    background: transparent;
    color: var(--color-text-secondary);
    cursor: pointer;
    transition: all 0.2s;
  }

  .icon-btn:hover {
    background: var(--bg-hover);
    color: var(--color-text-primary);
  }

  .icon-btn.small {
    width: 20px;
    height: 20px;
  }

  .icon-btn.danger:hover {
    background: var(--color-accent-red);
    color: white;
  }

  .error-banner {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 16px;
    background: var(--color-accent-red);
    color: white;
    font-size: 12px;
  }

  .submit-form {
    padding: 16px;
    border-bottom: 1px solid var(--border-color);
    background: var(--color-bg-elevated);
  }

  .form-row {
    margin-bottom: 12px;
  }

  .form-row label {
    display: block;
    font-size: 12px;
    font-weight: 500;
    margin-bottom: 4px;
    color: var(--color-text-secondary);
  }

  .form-row textarea,
  .form-row select {
    width: 100%;
    padding: 8px;
    border: 1px solid var(--border-color);
    border-radius: 4px;
    background: var(--color-bg-base);
    color: var(--color-text-primary);
    font-size: 12px;
    font-family: monospace;
  }

  .form-row textarea:focus,
  .form-row select:focus {
    outline: none;
    border-color: var(--color-accent-cyan);
  }

  .form-actions {
    display: flex;
    justify-content: flex-end;
    gap: 8px;
  }

  .stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(100px, 1fr));
    gap: 12px;
    padding: 16px;
    border-bottom: 1px solid var(--border-color);
  }

  .stat-item {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .stat-label {
    font-size: 10px;
    text-transform: uppercase;
    color: var(--color-text-muted);
  }

  .stat-value {
    font-size: 16px;
    font-weight: 600;
    color: var(--color-text-primary);
  }

  .batches-list {
    max-height: 300px;
    overflow-y: auto;
  }

  .list-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 16px;
    font-size: 12px;
    font-weight: 600;
    color: var(--color-text-secondary);
    border-bottom: 1px solid var(--border-color);
  }

  .batch-count {
    background: var(--color-bg-elevated);
    padding: 2px 8px;
    border-radius: 10px;
    font-size: 10px;
  }

  .empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 32px;
    color: var(--color-text-muted);
    gap: 8px;
  }

  .empty-state p {
    margin: 0;
    font-size: 12px;
  }

  .batch-items {
    display: flex;
    flex-direction: column;
  }

  .batch-item {
    padding: 12px 16px;
    border-bottom: 1px solid var(--border-color);
    cursor: pointer;
    transition: background 0.2s;
  }

  .batch-item:hover {
    background: var(--bg-hover);
  }

  .batch-item.selected {
    background: var(--color-bg-elevated);
  }

  .batch-main {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 8px;
  }

  .batch-id {
    font-family: monospace;
    font-size: 12px;
    color: var(--color-text-secondary);
  }

  .batch-status {
    display: flex;
    align-items: center;
    gap: 4px;
    font-size: 11px;
    font-weight: 500;
    text-transform: uppercase;
  }

  .batch-time {
    font-size: 11px;
    color: var(--color-text-muted);
    margin-left: auto;
  }

  .batch-progress {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .progress-bar {
    flex: 1;
    height: 4px;
    background: var(--color-bg-elevated);
    border-radius: 2px;
    overflow: hidden;
  }

  .progress-fill {
    height: 100%;
    background: var(--color-accent-green);
    transition: width 0.3s;
  }

  .progress-fill.failed {
    background: var(--color-accent-red);
  }

  .progress-text {
    font-size: 10px;
    color: var(--color-text-muted);
    min-width: 40px;
  }

  .batch-details {
    padding: 16px;
    background: var(--color-bg-elevated);
    border-top: 1px solid var(--border-color);
  }

  .details-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
  }

  .details-header h4 {
    margin: 0;
    font-size: 13px;
    font-weight: 600;
  }

  .details-status {
    font-size: 11px;
    font-weight: 500;
    text-transform: uppercase;
  }

  .details-stats {
    display: flex;
    gap: 16px;
    margin-bottom: 16px;
    flex-wrap: wrap;
  }

  .detail-stat {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 12px;
  }

  .detail-stat .label {
    color: var(--color-text-muted);
  }

  .detail-stat .value {
    font-weight: 600;
    color: var(--color-text-primary);
  }

  .detail-stat.success .value {
    color: var(--color-accent-green);
  }

  .detail-stat.danger .value {
    color: var(--color-accent-red);
  }

  .errors-section,
  .results-section {
    margin-top: 12px;
  }

  .errors-section h5,
  .results-section h5 {
    margin: 0 0 8px;
    font-size: 12px;
    font-weight: 600;
    color: var(--color-text-secondary);
  }

  .errors-list {
    display: flex;
    flex-direction: column;
    gap: 4px;
    max-height: 120px;
    overflow-y: auto;
  }

  .error-item {
    display: flex;
    gap: 8px;
    padding: 6px 8px;
    background: var(--color-bg-base);
    border-radius: 4px;
    font-size: 11px;
  }

  .error-id {
    font-family: monospace;
    color: var(--color-accent-red);
  }

  .error-msg {
    color: var(--color-text-secondary);
  }

  .results-list {
    display: flex;
    flex-direction: column;
    gap: 8px;
    max-height: 200px;
    overflow-y: auto;
  }

  .result-item {
    padding: 8px;
    background: var(--color-bg-base);
    border-radius: 4px;
    font-size: 11px;
  }

  .result-item pre {
    margin: 0;
    white-space: pre-wrap;
    word-break: break-all;
  }

  .more-results {
    text-align: center;
    font-size: 11px;
    color: var(--color-text-muted);
    padding: 8px;
  }

  :global(.spinning) {
    animation: spin 1s linear infinite;
  }

  @keyframes spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
  }
</style>
