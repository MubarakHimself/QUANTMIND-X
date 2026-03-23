<script lang="ts">
  import { onMount } from 'svelte';
  import { RefreshCw, Lock, Check, AlertCircle, Shield, Eye, EyeOff, Copy, Key } from 'lucide-svelte';
  import { apiFetch } from '$lib/api';

  interface SecurityStatus {
    secretKeyConfigured: boolean;
    secretKeyPrefix: string;
    jwtEnabled: boolean;
    corsOrigins: string[];
    rateLimitEnabled: boolean;
    requestsPerMinute: number;
  }

  let status: SecurityStatus = $state({
    secretKeyConfigured: false,
    secretKeyPrefix: '',
    jwtEnabled: false,
    corsOrigins: [],
    rateLimitEnabled: true,
    requestsPerMinute: 60
  });

  let isLoading = $state(false);
  let isSaving = $state(false);
  let isGenerating = $state(false);
  let error = $state<string | null>(null);
  let success = $state<string | null>(null);
  let showGeneratedKey = $state(false);
  let generatedKey = $state('');
  let newCorsOrigin = $state('');

  async function loadStatus() {
    isLoading = true;
    error = null;
    try {
      const data = await apiFetch<SecurityStatus>('/api/settings/security');
      status = { ...status, ...data };
    } catch (e) {
      console.warn('Security settings endpoint unavailable');
    } finally {
      isLoading = false;
    }
  }

  async function saveSettings() {
    isSaving = true;
    error = null;
    success = null;
    try {
      await apiFetch('/api/settings/security', {
        method: 'POST',
        body: JSON.stringify({
          jwtEnabled: status.jwtEnabled,
          corsOrigins: status.corsOrigins,
          rateLimitEnabled: status.rateLimitEnabled,
          requestsPerMinute: status.requestsPerMinute
        })
      });
      success = 'Security settings saved';
      setTimeout(() => success = null, 3000);
    } catch (e) {
      error = 'Failed to save security settings';
      console.error(e);
    } finally {
      isSaving = false;
    }
  }

  async function generateNewKey() {
    isGenerating = true;
    error = null;
    try {
      const result = await apiFetch<{ key: string; prefix: string }>('/api/settings/security/generate-key', {
        method: 'POST'
      });
      generatedKey = result.key;
      status.secretKeyPrefix = result.prefix;
      status.secretKeyConfigured = true;
      showGeneratedKey = true;
    } catch (e) {
      // Fallback: generate locally for display (server still needs to set it)
      const arr = new Uint8Array(32);
      crypto.getRandomValues(arr);
      generatedKey = Array.from(arr).map(b => b.toString(16).padStart(2, '0')).join('');
      showGeneratedKey = true;
      error = 'Note: Set this as your SECRET_KEY environment variable';
      console.warn(e);
    } finally {
      isGenerating = false;
    }
  }

  async function copyKey() {
    try {
      await navigator.clipboard.writeText(generatedKey);
      success = 'Key copied to clipboard';
      setTimeout(() => success = null, 2000);
    } catch (e) {
      console.error('Copy failed:', e);
    }
  }

  function addCorsOrigin() {
    if (newCorsOrigin.trim() && !status.corsOrigins.includes(newCorsOrigin.trim())) {
      status.corsOrigins = [...status.corsOrigins, newCorsOrigin.trim()];
      newCorsOrigin = '';
    }
  }

  function removeCorsOrigin(origin: string) {
    status.corsOrigins = status.corsOrigins.filter(o => o !== origin);
  }

  onMount(() => { loadStatus(); });
</script>

<div class="panel">
  <div class="panel-header">
    <h3>Security</h3>
    <div class="header-actions">
      <button class="icon-btn" onclick={loadStatus} title="Refresh" disabled={isLoading}>
        <RefreshCw size={16} class={isLoading ? 'spinning' : ''} />
      </button>
    </div>
  </div>

  {#if error}
    <div class="alert-error"><AlertCircle size={14} /> <span>{error}</span></div>
  {/if}
  {#if success}
    <div class="alert-success"><Check size={14} /> <span>{success}</span></div>
  {/if}

  <!-- Secret Key Status -->
  <div class="settings-section">
    <div class="section-title">Secret Key</div>
    <div class="key-status-row">
      <div class="key-status-info">
        <div class="status-indicator" class:configured={status.secretKeyConfigured}>
          <Lock size={13} />
          <span>{status.secretKeyConfigured ? 'Configured' : 'Not Configured'}</span>
        </div>
        {#if status.secretKeyConfigured && status.secretKeyPrefix}
          <span class="key-prefix">Prefix: <code>{status.secretKeyPrefix}•••</code></span>
        {:else if !status.secretKeyConfigured}
          <span class="hint">Set SECRET_KEY environment variable or generate below.</span>
        {/if}
      </div>
      <button class="btn secondary" onclick={generateNewKey} disabled={isGenerating}>
        <Key size={13} />
        {isGenerating ? 'Generating...' : 'Generate Key'}
      </button>
    </div>

    {#if showGeneratedKey && generatedKey}
      <div class="generated-key-box">
        <div class="generated-key-label">
          <AlertCircle size={12} />
          <span>Copy this key now — it will not be shown again.</span>
        </div>
        <div class="generated-key-row">
          {#if showGeneratedKey}
            <code class="generated-key-value">{generatedKey}</code>
          {:else}
            <code class="generated-key-value">{'•'.repeat(64)}</code>
          {/if}
          <button class="icon-btn" onclick={() => showGeneratedKey = !showGeneratedKey} title="Toggle visibility">
            {#if showGeneratedKey}
              <EyeOff size={12} />
            {:else}
              <Eye size={12} />
            {/if}
          </button>
          <button class="icon-btn accent" onclick={copyKey} title="Copy key">
            <Copy size={12} />
          </button>
        </div>
      </div>
    {/if}
  </div>

  <!-- Rate Limiting -->
  <div class="settings-section">
    <div class="section-title">Rate Limiting</div>
    <div class="setting-row">
      <span>Enable Rate Limiting</span>
      <label class="switch">
        <input type="checkbox" bind:checked={status.rateLimitEnabled} />
        <span class="slider"></span>
      </label>
    </div>
    {#if status.rateLimitEnabled}
      <div class="setting-row">
        <span>Requests per Minute</span>
        <input
          type="number"
          min="10"
          max="1000"
          step="10"
          class="number-input"
          bind:value={status.requestsPerMinute}
        />
      </div>
    {/if}
  </div>

  <!-- CORS Origins -->
  <div class="settings-section">
    <div class="section-title">CORS Origins</div>
    <div class="cors-input-row">
      <input
        type="text"
        class="text-input"
        placeholder="http://localhost:5173"
        bind:value={newCorsOrigin}
        onkeydown={(e) => e.key === 'Enter' && addCorsOrigin()}
      />
      <button class="btn secondary" onclick={addCorsOrigin}>Add</button>
    </div>
    {#if status.corsOrigins.length > 0}
      <div class="origins-list">
        {#each status.corsOrigins as origin}
          <div class="origin-item">
            <Shield size={11} />
            <code>{origin}</code>
            <button class="remove-btn" onclick={() => removeCorsOrigin(origin)} title="Remove">
              ×
            </button>
          </div>
        {/each}
      </div>
    {:else}
      <p class="hint">No CORS origins configured. All origins will be allowed.</p>
    {/if}
  </div>

  <div class="action-row">
    <button class="btn primary" onclick={saveSettings} disabled={isSaving}>
      <Shield size={14} />
      {isSaving ? 'Saving...' : 'Save Security Settings'}
    </button>
  </div>
</div>

<style>
  .panel-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 20px;
  }

  .panel-header h3 {
    margin: 0;
    font-size: 15px;
    font-weight: 600;
    color: var(--text-primary, #e8eaf0);
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }

  .header-actions { display: flex; gap: 8px; }

  .alert-error {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 10px 12px;
    background: rgba(255, 59, 59, 0.1);
    border: 1px solid rgba(255, 59, 59, 0.25);
    border-radius: 6px;
    color: #ff3b3b;
    font-size: 12px;
    margin-bottom: 12px;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }

  .alert-success {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 10px 12px;
    background: rgba(0, 200, 150, 0.08);
    border: 1px solid rgba(0, 200, 150, 0.2);
    border-radius: 6px;
    color: #00c896;
    font-size: 12px;
    margin-bottom: 12px;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }

  .settings-section {
    background: rgba(8, 13, 20, 0.35);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 12px;
  }

  .section-title {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: rgba(255, 255, 255, 0.5);
    margin-bottom: 14px;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }

  .key-status-row {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 12px;
  }

  .key-status-info { display: flex; flex-direction: column; gap: 4px; }

  .status-indicator {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 10px;
    border-radius: 5px;
    font-size: 12px;
    font-weight: 600;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    background: rgba(255, 59, 59, 0.1);
    border: 1px solid rgba(255, 59, 59, 0.2);
    color: #ff3b3b;
    width: fit-content;
  }

  .status-indicator.configured {
    background: rgba(0, 200, 150, 0.1);
    border-color: rgba(0, 200, 150, 0.2);
    color: #00c896;
  }

  .key-prefix {
    font-size: 11px;
    color: rgba(255, 255, 255, 0.4);
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }

  .key-prefix code {
    color: rgba(255, 255, 255, 0.6);
    background: rgba(255, 255, 255, 0.05);
    padding: 1px 5px;
    border-radius: 3px;
  }

  .generated-key-box {
    margin-top: 12px;
    padding: 12px;
    background: rgba(240, 165, 0, 0.06);
    border: 1px solid rgba(240, 165, 0, 0.2);
    border-radius: 6px;
  }

  .generated-key-label {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 11px;
    color: #f0a500;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    margin-bottom: 8px;
  }

  .generated-key-row {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .generated-key-value {
    flex: 1;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    font-size: 11px;
    color: #e8eaf0;
    word-break: break-all;
    background: rgba(8, 13, 20, 0.5);
    padding: 6px 8px;
    border-radius: 4px;
  }

  /* Setting Row */
  .setting-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 0;
    border-bottom: 1px solid rgba(255, 255, 255, 0.04);
    font-size: 12px;
    color: rgba(255, 255, 255, 0.6);
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }

  .setting-row:last-child { border-bottom: none; }

  .number-input {
    width: 80px;
    padding: 6px 10px;
    background: rgba(8, 13, 20, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 5px;
    color: #e8eaf0;
    font-size: 12px;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    text-align: right;
    -moz-appearance: textfield;
  }

  .number-input::-webkit-outer-spin-button,
  .number-input::-webkit-inner-spin-button { -webkit-appearance: none; margin: 0; }

  .number-input:focus {
    outline: none;
    border-color: rgba(0, 212, 255, 0.5);
    box-shadow: 0 0 0 3px rgba(0, 212, 255, 0.1);
  }

  /* Toggle Switch */
  .switch {
    position: relative;
    display: inline-block;
    width: 40px;
    height: 22px;
  }

  .switch input { opacity: 0; width: 0; height: 0; }

  .slider {
    position: absolute;
    cursor: pointer;
    inset: 0;
    background: rgba(255, 255, 255, 0.1);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 22px;
    transition: 0.2s;
  }

  .slider:before {
    position: absolute;
    content: "";
    height: 14px;
    width: 14px;
    left: 3px;
    bottom: 3px;
    background: rgba(255, 255, 255, 0.5);
    border-radius: 50%;
    transition: 0.2s;
  }

  input:checked + .slider {
    background: rgba(0, 212, 255, 0.25);
    border-color: rgba(0, 212, 255, 0.4);
  }

  input:checked + .slider:before {
    transform: translateX(18px);
    background: #00d4ff;
  }

  /* CORS */
  .cors-input-row {
    display: flex;
    gap: 8px;
    margin-bottom: 10px;
  }

  .cors-input-row .text-input { flex: 1; }

  .origins-list { display: flex; flex-direction: column; gap: 5px; }

  .origin-item {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 6px 10px;
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 5px;
    color: rgba(0, 212, 255, 0.7);
  }

  .origin-item code {
    flex: 1;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    font-size: 11px;
    color: rgba(255, 255, 255, 0.55);
  }

  .remove-btn {
    background: none;
    border: none;
    color: rgba(255, 59, 59, 0.6);
    cursor: pointer;
    font-size: 16px;
    line-height: 1;
    padding: 0 2px;
    transition: color 0.15s;
  }

  .remove-btn:hover { color: #ff3b3b; }

  .hint {
    font-size: 11px;
    color: rgba(255, 255, 255, 0.3);
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    margin: 4px 0 0;
  }

  .text-input {
    width: 100%;
    padding: 8px 12px;
    background: rgba(8, 13, 20, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 6px;
    color: #e8eaf0;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    font-size: 12px;
    box-sizing: border-box;
    transition: border-color 0.15s, box-shadow 0.15s;
  }

  .text-input:focus {
    outline: none;
    border-color: rgba(0, 212, 255, 0.5);
    box-shadow: 0 0 0 3px rgba(0, 212, 255, 0.1);
  }

  .text-input::placeholder { color: rgba(255, 255, 255, 0.25); }

  .action-row {
    display: flex;
    justify-content: flex-end;
    gap: 10px;
    margin-top: 4px;
  }

  .btn {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 7px 13px;
    border-radius: 6px;
    font-size: 12px;
    font-weight: 600;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    cursor: pointer;
    border: none;
    transition: all 0.15s;
  }

  .btn.primary {
    background: rgba(0, 212, 255, 0.15);
    border: 1px solid rgba(0, 212, 255, 0.4);
    color: #00d4ff;
  }

  .btn.primary:hover { background: rgba(0, 212, 255, 0.25); }
  .btn.primary:disabled { opacity: 0.45; cursor: not-allowed; }

  .btn.secondary {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    color: rgba(255, 255, 255, 0.65);
  }

  .btn.secondary:hover { background: rgba(255, 255, 255, 0.09); color: #fff; }
  .btn.secondary:disabled { opacity: 0.45; cursor: not-allowed; }

  .icon-btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 28px;
    height: 28px;
    border: none;
    border-radius: 5px;
    background: rgba(255, 255, 255, 0.04);
    color: rgba(255, 255, 255, 0.4);
    cursor: pointer;
    transition: all 0.15s;
  }

  .icon-btn:hover { background: rgba(255, 255, 255, 0.1); color: #e8eaf0; }
  .icon-btn:disabled { opacity: 0.45; cursor: not-allowed; }

  .icon-btn.accent {
    background: rgba(0, 212, 255, 0.12);
    color: #00d4ff;
    border: 1px solid rgba(0, 212, 255, 0.2);
  }

  .icon-btn.accent:hover { background: rgba(0, 212, 255, 0.22); }

  .spinning { animation: spin 1s linear infinite; }

  @keyframes spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
  }
</style>
