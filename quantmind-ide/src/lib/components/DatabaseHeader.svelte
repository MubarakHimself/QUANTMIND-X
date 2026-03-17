<script lang="ts">
  import { Database, RefreshCw, HardDrive } from "lucide-svelte";

  interface Props {
    onRefresh?: () => void;
    onLoadStats?: () => void;
  }

  let { onRefresh = () => {}, onLoadStats = () => {} }: Props = $props();
</script>

<div class="db-header">
  <div class="header-left">
    <Database size={24} class="db-icon" />
    <div>
      <h2>Database Manager</h2>
      <p>SQLite + DuckDB hybrid database with 15 tables</p>
    </div>
  </div>
  <div class="header-actions">
    <button class="btn" onclick={onRefresh}>
      <RefreshCw size={14} />
      <span>Refresh</span>
    </button>
    <button class="btn" onclick={onLoadStats}>
      <HardDrive size={14} />
      <span>Stats</span>
    </button>
  </div>
</div>

<style>
  .db-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 20px 24px;
    background: var(--bg-secondary);
    border-bottom: 1px solid var(--border-subtle);
  }

  .header-left {
    display: flex;
    align-items: center;
    gap: 12px;
  }

  .header-left h2 {
    margin: 0;
    font-size: 18px;
    color: var(--text-primary);
  }

  .header-left p {
    margin: 2px 0 0;
    font-size: 12px;
    color: var(--text-muted);
  }

  .header-actions {
    display: flex;
    gap: 8px;
  }

  .btn {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 8px 14px;
    background: var(--bg-tertiary);
    border: 1px solid var(--border-subtle);
    border-radius: 6px;
    color: var(--text-secondary);
    font-size: 12px;
    cursor: pointer;
    transition: all 0.15s;
  }

  .btn:hover:not(.disabled) {
    background: var(--bg-surface);
  }
</style>
