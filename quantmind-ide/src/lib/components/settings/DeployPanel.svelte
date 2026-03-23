<script lang="ts">
  import { onMount } from 'svelte';
  import { GitBranch, GitCommit, RefreshCw, Upload, RotateCcw, CheckCircle, XCircle, AlertTriangle, ChevronDown } from 'lucide-svelte';

  interface VersionInfo {
    commit: string;
    full_sha: string;
    message: string;
    branch: string;
    behind_origin: number;
    health: string;
    app_dir: string;
    error?: string;
  }

  interface DeployResult {
    status?: string;
    previous_commit?: string;
    new_commit?: string;
    rolled_back_to?: string;
    message?: string;
    error?: string;
  }

  let versionInfo: VersionInfo | null = $state(null);
  let isLoading = $state(true);
  let isDeploying = $state(false);
  let isRollingBack = $state(false);
  let error = $state<string | null>(null);
  let success = $state<string | null>(null);
  let rollbackRef = $state('');
  let recentCommits: Array<{ sha: string; message: string; time: string }> = $state([]);

  async function loadVersion() {
    isLoading = true;
    error = null;
    try {
      const res = await fetch('/api/settings/deploy/version');
      if (res.ok) {
        versionInfo = await res.json();
        if (versionInfo?.behind_origin && versionInfo.behind_origin > 0) {
          error = `${versionInfo.behind_origin} commit(s) behind origin/${versionInfo.branch}`;
        }
      } else {
        error = 'Failed to load version info';
      }
    } catch (e) {
      error = 'Server unreachable';
    } finally {
      isLoading = false;
    }
  }

  async function deployLatest() {
    if (!confirm('Pull latest from GitHub and restart the API?')) return;
    isDeploying = true;
    error = null;
    success = null;
    try {
      const res = await fetch('/api/settings/deploy/latest', { method: 'POST' });
      const data: DeployResult = await res.json();
      if (data.error) {
        error = data.error;
      } else {
        success = `Deployed ${data.new_commit}: ${data.message}`;
        await loadVersion();
      }
    } catch (e) {
      error = 'Deploy failed — server may be restarting';
    } finally {
      isDeploying = false;
    }
  }

  async function rollbackTo() {
    const ref = rollbackRef.trim();
    if (!ref) return;
    if (!confirm(`Rollback to "${ref}"? This will restart the API.`)) return;
    isRollingBack = true;
    error = null;
    success = null;
    try {
      const res = await fetch('/api/settings/deploy/rollback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ git_ref: ref }),
      });
      const data: DeployResult = await res.json();
      if (data.error) {
        error = data.error;
      } else {
        success = `Rolled back to ${data.rolled_back_to}: ${data.message}`;
        rollbackRef = '';
        await loadVersion();
      }
    } catch (e) {
      error = 'Rollback failed — server may be restarting';
    } finally {
      isRollingBack = false;
    }
  }

  function selectCommit(sha: string) {
    rollbackRef = sha;
  }

  onMount(() => {
    loadVersion();
  });
</script>

<div class="panel">
  <div class="panel-header">
    <h3>Deploy Control</h3>
    <div class="header-actions">
      <span class="last-updated">
        {#if versionInfo?.commit}
          <span class="commit-badge">{versionInfo.commit}</span>
        {/if}
      </span>
      <button class="icon-btn" onclick={loadVersion} title="Refresh" disabled={isLoading}>
        <RefreshCw size={16} class={isLoading ? 'spinning' : ''} />
      </button>
    </div>
  </div>

  {#if error && !isDeploying && !isRollingBack}
    <div class="alert error">
      <AlertTriangle size={14} />
      <span>{error}</span>
    </div>
  {/if}

  {#if success}
    <div class="alert success">
      <CheckCircle size={14} />
      <span>{success}</span>
    </div>
  {/if}

  {#if isLoading && !versionInfo}
    <div class="loading">
      <RefreshCw size={24} class="spinning" />
      <span>Loading version...</span>
    </div>
  {:else if versionInfo}
    <!-- Version Info -->
    <div class="version-card">
      <div class="version-row">
        <GitCommit size={16} />
        <span class="version-label">Commit</span>
        <code class="version-value">{versionInfo.commit}</code>
        <span class="version-msg">{versionInfo.message}</span>
      </div>
      <div class="version-row">
        <GitBranch size={16} />
        <span class="version-label">Branch</span>
        <code class="version-value">{versionInfo.branch}</code>
      </div>
      <div class="version-row">
        {#if versionInfo.health === 'healthy' || versionInfo.health.startsWith('{')}
          <CheckCircle size={16} class="icon-ok" />
          <span class="version-label">Health</span>
          <span class="status-badge ok">Healthy</span>
        {:else}
          <XCircle size={16} class="icon-fail" />
          <span class="version-label">Health</span>
          <span class="status-badge warn">{versionInfo.health}</span>
        {/if}
      </div>
    </div>

    <!-- Deploy Latest -->
    <div class="action-section">
      <div class="section-title">Deploy</div>
      <p class="section-desc">
        Pull the latest commit from GitHub <code>origin/{versionInfo.branch}</code>
        and restart the API service.
      </p>
      <button
        class="btn primary"
        onclick={deployLatest}
        disabled={isDeploying || isRollingBack || isLoading}
      >
        {#if isDeploying}
          <RefreshCw size={14} class="spinning" />
          Deploying...
        {:else}
          <Upload size={14} />
          Deploy Latest
        {/if}
      </button>
    </div>

    <!-- Rollback -->
    <div class="action-section">
      <div class="section-title">Rollback</div>
      <p class="section-desc">
        Enter a git ref (commit SHA, branch, or tag) to rollback to.
      </p>
      <div class="rollback-row">
        <input
          type="text"
          class="text-input"
          placeholder="e.g. abc1234 or origin/main"
          bind:value={rollbackRef}
          disabled={isDeploying || isRollingBack || isLoading}
        />
        <button
          class="btn danger"
          onclick={rollbackTo}
          disabled={isDeploying || isRollingBack || isLoading || !rollbackRef.trim()}
        >
          {#if isRollingBack}
            <RefreshCw size={14} class="spinning" />
            Rolling back...
          {:else}
            <RotateCcw size={14} />
            Rollback
          {/if}
        </button>
      </div>
    </div>

    <!-- Recent Commits -->
    {#if recentCommits.length > 0}
      <div class="action-section">
        <div class="section-title">Recent Commits</div>
        <div class="commits-list">
          {#each recentCommits as commit}
            <button
              class="commit-row"
              onclick={() => selectCommit(commit.sha)}
              title="Click to rollback to this commit"
            >
              <code class="commit-sha">{commit.sha}</code>
              <span class="commit-msg">{commit.message}</span>
              <span class="commit-time">{commit.time}</span>
            </button>
          {/each}
        </div>
      </div>
    {/if}
  {:else}
    <div class="loading">
      <span>Version info unavailable</span>
    </div>
  {/if}
</div>

<style>
  .panel {
    padding: 0;
  }

  .panel-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 16px;
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

  .header-actions {
    display: flex;
    align-items: center;
    gap: 12px;
  }

  .commit-badge {
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    font-size: 11px;
    padding: 3px 8px;
    background: rgba(0, 212, 255, 0.1);
    border: 1px solid rgba(0, 212, 255, 0.3);
    border-radius: 4px;
    color: #00d4ff;
  }

  .last-updated {
    font-size: 11px;
    color: rgba(255, 255, 255, 0.3);
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }

  .icon-btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 30px;
    height: 30px;
    border: none;
    border-radius: 6px;
    background: rgba(255, 255, 255, 0.04);
    color: rgba(255, 255, 255, 0.4);
    cursor: pointer;
    transition: all 0.15s;
  }

  .icon-btn:hover {
    background: rgba(255, 255, 255, 0.1);
    color: #e8eaf0;
  }

  .icon-btn:disabled {
    opacity: 0.45;
    cursor: not-allowed;
  }

  .spinning {
    animation: spin 1s linear infinite;
  }

  @keyframes spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
  }

  .alert {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 10px 12px;
    border-radius: 6px;
    font-size: 12px;
    margin-bottom: 12px;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }

  .alert.error {
    background: rgba(255, 59, 59, 0.1);
    border: 1px solid rgba(255, 59, 59, 0.25);
    color: #ff3b3b;
  }

  .alert.success {
    background: rgba(0, 200, 150, 0.08);
    border: 1px solid rgba(0, 200, 150, 0.2);
    color: #00c896;
  }

  .loading {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 40px;
    color: rgba(255, 255, 255, 0.3);
    gap: 12px;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    font-size: 12px;
  }

  .version-card {
    background: rgba(8, 13, 20, 0.35);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 8px;
    padding: 14px;
    margin-bottom: 16px;
    display: flex;
    flex-direction: column;
    gap: 10px;
  }

  .version-row {
    display: flex;
    align-items: center;
    gap: 8px;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    font-size: 12px;
    color: rgba(255, 255, 255, 0.6);
  }

  .version-row :global(.icon-ok) {
    color: #22c55e;
  }

  .version-row :global(.icon-fail) {
    color: #ff3b3b;
  }

  .version-label {
    color: rgba(255, 255, 255, 0.35);
    min-width: 56px;
  }

  .version-value {
    color: #00d4ff;
    font-size: 11px;
  }

  .version-msg {
    color: rgba(255, 255, 255, 0.5);
    margin-left: auto;
    font-size: 11px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    max-width: 300px;
  }

  .status-badge {
    font-size: 11px;
    padding: 2px 8px;
    border-radius: 4px;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }

  .status-badge.ok {
    background: rgba(34, 197, 94, 0.15);
    color: #22c55e;
  }

  .status-badge.warn {
    background: rgba(234, 179, 8, 0.15);
    color: #eab308;
  }

  .action-section {
    background: rgba(8, 13, 20, 0.35);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 8px;
    padding: 14px;
    margin-bottom: 12px;
  }

  .section-title {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: rgba(255, 255, 255, 0.5);
    margin-bottom: 8px;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }

  .section-desc {
    font-size: 11px;
    color: rgba(255, 255, 255, 0.4);
    margin: 0 0 12px;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    line-height: 1.5;
  }

  .section-desc code {
    color: rgba(255, 255, 255, 0.6);
    background: rgba(255, 255, 255, 0.05);
    padding: 1px 4px;
    border-radius: 3px;
  }

  .btn {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 8px 14px;
    border-radius: 6px;
    font-size: 12px;
    font-weight: 600;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    cursor: pointer;
    border: none;
    transition: all 0.15s;
  }

  .btn:disabled {
    opacity: 0.45;
    cursor: not-allowed;
  }

  .btn.primary {
    background: rgba(0, 212, 255, 0.15);
    border: 1px solid rgba(0, 212, 255, 0.4);
    color: #00d4ff;
  }

  .btn.primary:hover:not(:disabled) {
    background: rgba(0, 212, 255, 0.25);
  }

  .btn.danger {
    background: rgba(255, 59, 59, 0.1);
    border: 1px solid rgba(255, 59, 59, 0.35);
    color: #ff6b6b;
  }

  .btn.danger:hover:not(:disabled) {
    background: rgba(255, 59, 59, 0.2);
  }

  .rollback-row {
    display: flex;
    gap: 8px;
  }

  .text-input {
    flex: 1;
    padding: 8px 12px;
    background: rgba(8, 13, 20, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 6px;
    color: #e8eaf0;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    font-size: 12px;
    box-sizing: border-box;
    transition: border-color 0.15s;
  }

  .text-input:focus {
    outline: none;
    border-color: rgba(0, 212, 255, 0.5);
  }

  .text-input::placeholder {
    color: rgba(255, 255, 255, 0.25);
  }

  .text-input:disabled {
    opacity: 0.5;
  }

  .commits-list {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .commit-row {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 10px;
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 6px;
    cursor: pointer;
    transition: all 0.15s;
    text-align: left;
    width: 100%;
  }

  .commit-row:hover {
    background: rgba(255, 255, 255, 0.06);
    border-color: rgba(255, 59, 59, 0.3);
  }

  .commit-sha {
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    font-size: 11px;
    color: #ff6b6b;
    min-width: 60px;
  }

  .commit-msg {
    flex: 1;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    font-size: 11px;
    color: rgba(255, 255, 255, 0.5);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .commit-time {
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    font-size: 10px;
    color: rgba(255, 255, 255, 0.25);
  }
</style>
