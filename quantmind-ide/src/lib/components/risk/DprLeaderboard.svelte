<script lang="ts">
  import { onMount } from 'svelte';
  import { Trophy, Activity, AlertTriangle } from 'lucide-svelte';

  type DprHealth = {
    status?: string;
    module?: string;
    mode?: string;
  };

  let loading = $state(true);
  let error = $state<string | null>(null);
  let health = $state<DprHealth | null>(null);

  onMount(async () => {
    try {
      const response = await fetch('/api/dpr/health');
      if (!response.ok) {
        throw new Error(`DPR health check failed (${response.status})`);
      }
      health = await response.json();
    } catch (err) {
      error = err instanceof Error ? err.message : 'Failed to load DPR status';
    } finally {
      loading = false;
    }
  });
</script>

<div class="dpr-card">
  <div class="tile-header">
    <h3 class="tile-title">
      <Trophy size={16} />
      DPR Queue
    </h3>
    {#if health?.mode}
      <span class="mode-badge">{health.mode}</span>
    {/if}
  </div>

  <div class="tile-content">
    {#if loading}
      <div class="state-block muted">
        <Activity size={14} />
        <span>Checking DPR service...</span>
      </div>
    {:else if error}
      <div class="state-block error">
        <AlertTriangle size={14} />
        <span>{error}</span>
      </div>
    {:else}
      <div class="summary-grid">
        <div class="summary-item">
          <span class="summary-label">Module</span>
          <span class="summary-value">{health?.module ?? 'dpr'}</span>
        </div>
        <div class="summary-item">
          <span class="summary-label">Status</span>
          <span class="summary-value status-ok">{health?.status ?? 'ok'}</span>
        </div>
      </div>

      <div class="compat-note">
        DPR queue visuals are restored for frontend startup.
        Full leaderboard data is not wired yet in this worktree, so this panel exposes service readiness instead of fake rankings.
      </div>
    {/if}
  </div>
</div>

<style>
  .dpr-card {
    display: flex;
    flex-direction: column;
    gap: 0.9rem;
    padding: 1rem;
    border: 1px solid rgba(255, 59, 59, 0.16);
    border-radius: 16px;
    background:
      linear-gradient(180deg, rgba(13, 18, 24, 0.96), rgba(9, 13, 18, 0.98)),
      radial-gradient(circle at top right, rgba(255, 59, 59, 0.08), transparent 48%);
    box-shadow: 0 18px 40px rgba(0, 0, 0, 0.22);
    color: #d7e4f3;
  }

  .tile-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.75rem;
  }

  .tile-title {
    display: inline-flex;
    align-items: center;
    gap: 0.55rem;
    margin: 0;
    font-size: 0.95rem;
    font-weight: 600;
    letter-spacing: 0.02em;
  }

  .mode-badge {
    padding: 0.2rem 0.55rem;
    border-radius: 999px;
    background: rgba(255, 183, 0, 0.12);
    color: #ffd166;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
  }

  .tile-content {
    display: flex;
    flex-direction: column;
    gap: 0.9rem;
  }

  .state-block {
    display: inline-flex;
    align-items: center;
    gap: 0.55rem;
    padding: 0.8rem 0.9rem;
    border-radius: 12px;
    font-size: 0.88rem;
  }

  .state-block.muted {
    background: rgba(0, 212, 255, 0.08);
    color: #8ddff3;
  }

  .state-block.error {
    background: rgba(255, 59, 59, 0.12);
    color: #ff9f9f;
  }

  .summary-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 0.75rem;
  }

  .summary-item {
    display: flex;
    flex-direction: column;
    gap: 0.3rem;
    padding: 0.85rem 0.9rem;
    border-radius: 12px;
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.05);
  }

  .summary-label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #8ea1b5;
  }

  .summary-value {
    font-size: 0.95rem;
    font-weight: 600;
  }

  .status-ok {
    color: #7ce3b3;
  }

  .compat-note {
    padding: 0.85rem 0.9rem;
    border-radius: 12px;
    background: rgba(255, 183, 0, 0.08);
    color: #e7cf8f;
    font-size: 0.84rem;
    line-height: 1.45;
  }
</style>
