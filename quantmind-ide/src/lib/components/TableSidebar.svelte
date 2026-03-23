<script lang="ts">
  import { Table, Search, Filter, Database, RefreshCw } from "lucide-svelte";

  interface TableInfo {
    name: string;
    row_count: number;
    is_new: boolean;
    columns: ColumnInfo[];
    size_bytes: number;
    last_updated: string;
  }

  interface ColumnInfo {
    name: string;
    type: string;
    nullable: boolean;
    primary_key: boolean;
    default_value?: any;
  }

  interface Props {
    tables?: TableInfo[];
    filteredTables?: TableInfo[];
    selectedTable?: TableInfo | null;
    isLoading?: boolean;
    searchQuery?: string;
    tableTypeFilter?: "all" | "existing" | "new";
    onSelectTable?: (table: TableInfo) => void;
    onSearchChange?: () => void;
    onFilterChange?: () => void;
  }

  let {
    tables = [],
    filteredTables = [],
    selectedTable = null,
    isLoading = false,
    searchQuery = $bindable(""),
    tableTypeFilter = $bindable("all"),
    onSelectTable = () => {},
    onSearchChange = () => {},
    onFilterChange = () => {}
  }: Props = $props();

  function formatBytes(bytes: number): string {
    if (bytes === 0) return "0 B";
    const k = 1024;
    const sizes = ["B", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + " " + sizes[i];
  }
</script>

<div class="table-sidebar">
  <div class="sidebar-header">
    <h3>Tables</h3>
    <div class="table-count">{filteredTables.length}</div>
  </div>

  <!-- Search -->
  <div class="sidebar-search">
    <Search size={12} />
    <input
      type="text"
      placeholder="Search tables..."
      bind:value={searchQuery}
      oninput={onSearchChange}
    />
  </div>

  <!-- Filter -->
  <div class="sidebar-filter">
    <Filter size={12} />
    <select bind:value={tableTypeFilter} onchange={onFilterChange}>
      <option value="all">All Tables</option>
      <option value="existing">Existing</option>
      <option value="new">New</option>
    </select>
  </div>

  <!-- Table List -->
  <div class="table-list">
    {#if isLoading}
      <div class="loading-state">
        <RefreshCw size={20} class="spin" />
        <span>Loading tables...</span>
      </div>
    {:else}
      {#each filteredTables as table}
        <div
          class="table-item"
          class:selected={selectedTable?.name === table.name}
          class:new-table={table.is_new}
          onclick={() => onSelectTable(table)}
          onkeydown={(e) => e.key === "Enter" && onSelectTable(table)}
          role="button"
          tabindex="0"
          aria-label="Select table {table.name}"
        >
          <div class="table-info">
            <div class="table-name">
              <Table size={14} />
              <span>{table.name}</span>
            </div>
            <div class="table-meta">
              <span class="row-count"
                >{table.row_count.toLocaleString()} rows</span
              >
              <span class="table-size">{formatBytes(table.size_bytes)}</span
              >
            </div>
          </div>
          <div
            class="table-indicator"
            class:new-table={table.is_new}
            title={table.is_new ? "New table" : "Existing table"}
          ></div>
        </div>
      {/each}

      {#if filteredTables.length === 0}
        <div class="empty-state">
          <Database size={32} />
          <p>No tables found</p>
        </div>
      {/if}
    {/if}
  </div>
</div>

<style>
  .table-sidebar {
    width: 280px;
    background: var(--color-bg-surface);
    border-right: 1px solid var(--color-border-subtle);
    display: flex;
    flex-direction: column;
  }

  .sidebar-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 16px;
    border-bottom: 1px solid var(--color-border-subtle);
  }

  .sidebar-header h3 {
    margin: 0;
    font-size: 14px;
    color: var(--color-text-primary);
  }

  .table-count {
    padding: 2px 8px;
    background: var(--color-bg-elevated);
    border-radius: 10px;
    font-size: 11px;
    color: var(--color-text-muted);
  }

  .sidebar-search,
  .sidebar-filter {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 12px 16px;
    border-bottom: 1px solid var(--color-border-subtle);
  }

  .sidebar-search input,
  .sidebar-filter select {
    flex: 1;
    background: var(--color-bg-elevated);
    border: none;
    border-radius: 4px;
    padding: 6px 10px;
    font-size: 12px;
    color: var(--color-text-primary);
    outline: none;
  }

  .table-list {
    flex: 1;
    overflow-y: auto;
  }

  .table-item {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 12px 16px;
    cursor: pointer;
    transition: background 0.15s;
    border-bottom: 1px solid var(--color-border-subtle);
  }

  .table-item:hover {
    background: var(--color-bg-elevated);
  }

  .table-item.selected {
    background: var(--color-accent-cyan);
    color: var(--color-bg-base);
  }

  .table-item.selected .table-name,
  .table-item.selected .table-meta {
    color: var(--color-bg-base);
  }

  .table-info {
    flex: 1;
    min-width: 0;
  }

  .table-name {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 13px;
    font-weight: 500;
    color: var(--color-text-primary);
    margin-bottom: 4px;
  }

  .table-meta {
    display: flex;
    gap: 8px;
    font-size: 11px;
    color: var(--color-text-muted);
  }

  .table-indicator {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--color-text-muted);
    flex-shrink: 0;
  }

  .table-indicator.new-table {
    background: var(--color-accent-green);
  }

  .loading-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 40px;
    gap: 12px;
    color: var(--color-text-muted);
    font-size: 12px;
  }

  .empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 40px;
    gap: 12px;
    color: var(--color-text-muted);
  }

  .empty-state p {
    margin: 0;
    font-size: 13px;
  }

  :global(.spin) {
    animation: spin 1s linear infinite;
  }

  @keyframes spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
  }
</style>
