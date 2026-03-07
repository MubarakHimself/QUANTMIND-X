<script lang="ts">
  import { Database, HardDrive, FileText, Code } from "lucide-svelte";

  interface DatabaseStats {
    total_size_bytes: number;
    sqlite_path: string;
    duckdb_path: string;
    table_count: number;
  }

  export let stats: DatabaseStats | null = null;

  function formatBytes(bytes: number): string {
    if (bytes === 0) return "0 B";
    const k = 1024;
    const sizes = ["B", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + " " + sizes[i];
  }
</script>

{#if stats}
  <div class="stats-banner">
    <div class="stat-item">
      <Database size={14} />
      <span class="label">Tables:</span>
      <span class="value">{stats.table_count}</span>
    </div>
    <div class="stat-item">
      <HardDrive size={14} />
      <span class="label">Total Size:</span>
      <span class="value">{formatBytes(stats.total_size_bytes)}</span>
    </div>
    <div class="stat-item">
      <FileText size={14} />
      <span class="label">SQLite:</span>
      <span class="value path">{stats.sqlite_path}</span>
    </div>
    <div class="stat-item">
      <Code size={14} />
      <span class="label">DuckDB:</span>
      <span class="value path">{stats.duckdb_path}</span>
    </div>
  </div>
{/if}

<style>
  .stats-banner {
    display: flex;
    gap: 24px;
    padding: 12px 24px;
    background: var(--bg-tertiary);
    border-bottom: 1px solid var(--border-subtle);
    flex-wrap: wrap;
  }

  .stat-item {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 11px;
    color: var(--text-secondary);
  }

  .stat-item .label {
    color: var(--text-muted);
  }

  .stat-item .value {
    color: var(--text-primary);
    font-weight: 500;
  }

  .stat-item .value.path {
    font-family: "JetBrains Mono", monospace;
    font-size: 10px;
  }
</style>
