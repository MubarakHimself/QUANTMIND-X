<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import { RefreshCw, Lock } from 'lucide-svelte';

  let { securitySettings = {
    secretKeyConfigured: false,
    secretKeyPrefix: ''
  } } = $props();

  const dispatch = createEventDispatcher();

  function generateNewKey() {
    dispatch('generateNewKey');
  }
</script>

<div class="panel">
  <h3>Security</h3>

  <div class="setting-group">
    <label>Secret Key Status</label>
    {#if securitySettings.secretKeyConfigured}
      <span class="status-badge success">Configured</span>
      <p class="hint">Key starts with: {securitySettings.secretKeyPrefix}***</p>
    {:else}
      <span class="status-badge warning">Not Configured</span>
      <p class="hint">Set SECRET_KEY environment variable</p>
    {/if}
  </div>

  <div class="setting-group">
    <button class="btn btn-secondary" onclick={generateNewKey}>
      <RefreshCw size={16} />
      Generate New Key
    </button>
  </div>
</div>

<style>
  /* Panel Header */
  .panel h3 {
    margin: 0 0 20px;
    font-size: 16px;
    font-weight: 600;
    color: var(--text-primary);
  }

  /* Setting Group */
  .setting-group {
    background: var(--bg-tertiary);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 12px;
  }

  .setting-group:last-child {
    margin-bottom: 0;
  }

  .setting-group > label {
    display: block;
    font-size: 13px;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 8px;
  }

  /* Status Badge */
  .status-badge {
    display: inline-flex;
    align-items: center;
    padding: 6px 12px;
    border-radius: 6px;
    font-size: 13px;
    font-weight: 500;
    margin-bottom: 8px;
  }

  .status-badge.success {
    background: rgba(16, 185, 129, 0.15);
    color: #10b981;
  }

  .status-badge.warning {
    background: rgba(245, 158, 11, 0.15);
    color: #f59e0b;
  }

  /* Hint Text */
  .hint {
    font-size: 12px;
    color: var(--text-muted);
    margin: 0;
    font-family: monospace;
  }

  /* Button */
  .btn {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 10px 16px;
    border-radius: 6px;
    font-size: 13px;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.15s;
    border: none;
  }

  .btn.secondary {
    background: var(--bg-primary);
    border: 1px solid var(--border-subtle);
    color: var(--text-secondary);
  }

  .btn.secondary:hover {
    background: var(--bg-surface);
    color: var(--text-primary);
  }
</style>
