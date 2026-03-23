<script lang="ts">
  import {
    Table,
    Edit,
    Trash2,
    Plus,
    RefreshCw,
    Code,
    FileText,
    FileCode,
    Upload,
    Hash,
    ChevronDown,
  } from "lucide-svelte";


  
  interface Props {
    tableData?: any;
    selectedTable?: any;
    selectedRows?: Set<string>;
    currentPage?: number;
    rowsPerPage?: number;
    sortColumn?: string | null;
    sortDirection?: "asc" | "desc";
    isLoading?: boolean;
    showTableInfo?: boolean;
    // Function imports
    formatBytes: (bytes: number) => string;
    formatTimestamp: (dateStr: string) => string;
    getColumnTypeColor: (type: string) => string;
    isJsonColumn: (value: any) => boolean;
    previewJson: (data: any) => void;
    toggleRowSelection: (rowId: string) => void;
    toggleAllRows: () => void;
    sortTable: (column: string) => void;
    getPaginatedRows: () => any[];
    getTotalPages: () => number;
    loadTableData: (tableName: string, page: number) => Promise<void>;
    deleteSelectedRows: () => Promise<void>;
    openInsertModal: () => void;
    openEditModal: (row: any) => void;
    exportTable: (format: "csv" | "json") => Promise<void>;
    importTable: (file: File) => Promise<void>;
  }

  let {
    tableData = null,
    selectedTable = null,
    selectedRows = $bindable(new Set()),
    currentPage = $bindable(1),
    rowsPerPage = $bindable(25),
    sortColumn = null,
    sortDirection = "asc",
    isLoading = false,
    showTableInfo = $bindable(false),
    formatBytes,
    formatTimestamp,
    getColumnTypeColor,
    isJsonColumn,
    previewJson,
    toggleRowSelection,
    toggleAllRows,
    sortTable,
    getPaginatedRows,
    getTotalPages,
    loadTableData,
    deleteSelectedRows,
    openInsertModal,
    openEditModal,
    exportTable,
    importTable
  }: Props = $props();

  function handleDeleteRow(row: any) {
    if (confirm("Delete this row?")) {
      selectedRows = new Set([String(row.id || Object.values(row)[0])]);
      deleteSelectedRows();
    }
  }
</script>

<div class="data-grid-area">
  {#if selectedTable}
    <!-- Table Header -->
    <div class="grid-header">
      <div class="header-left">
        <Table size={16} />
        <div>
          <h3>{selectedTable.name}</h3>
          <span class="table-badge" class:new-table={selectedTable.is_new}>
            {selectedTable.is_new ? "NEW" : "EXISTING"}
          </span>
        </div>
      </div>
      <div class="header-actions">
        <button
          class="btn"
          onclick={() => (showTableInfo = !showTableInfo)}
          class:active={showTableInfo}
        >
          <FileText size={14} />
          <span>Schema</span>
        </button>
        <button class="btn" onclick={() => exportTable("csv")}>
          <FileText size={14} />
          <span>Export CSV</span>
        </button>
        <button class="btn" onclick={() => exportTable("json")}>
          <FileCode size={14} />
          <span>Export JSON</span>
        </button>
        <label class="btn">
          <Upload size={14} />
          <span>Import</span>
          <input
            type="file"
            accept=".csv,.json"
            onchange={(e) => {
              const input = e.currentTarget;
              const file = input.files?.[0];
              if (file) importTable(file);
            }}
            hidden
          />
        </label>
        <button class="btn" onclick={openInsertModal}>
          <Plus size={14} />
          <span>Insert</span>
        </button>
        <button
          class="btn danger"
          onclick={deleteSelectedRows}
          disabled={selectedRows.size === 0}
          class:disabled={selectedRows.size === 0}
        >
          <Trash2 size={14} />
          <span>Delete ({selectedRows.size})</span>
        </button>
      </div>
    </div>

    <!-- Table Schema Panel -->
    {#if showTableInfo}
      <div class="schema-panel">
        <h4>Table Schema</h4>
        <div class="schema-grid">
          {#each selectedTable.columns as column}
            <div class="schema-column">
              <div class="column-header">
                <Hash size={12} />
                <span class="column-name">{column.name}</span>
                {#if column.primary_key}
                  <span class="pk-badge">PK</span>
                {/if}
              </div>
              <div class="column-info">
                <span
                  class="column-type"
                  style="color: {getColumnTypeColor(column.type)}"
                >
                  {column.type}
                </span>
                {#if column.nullable}
                  <span class="nullable-badge">NULL</span>
                {:else}
                  <span class="not-null-badge">NOT NULL</span>
                {/if}
              </div>
              {#if column.default_value !== undefined}
                <div class="column-default">
                  <span class="label">Default:</span>
                  <span class="value">{column.default_value}</span>
                </div>
              {/if}
            </div>
          {/each}
        </div>
        <div class="schema-meta">
          <span class="meta-item"
            >Last updated: {formatTimestamp(
              selectedTable.last_updated,
            )}</span
          >
          <span class="meta-item"
            >Size: {formatBytes(selectedTable.size_bytes)}</span
          >
        </div>
      </div>
    {/if}

    <!-- Data Grid -->
    <div class="data-grid">
      {#if isLoading}
        <div class="loading-state">
          <RefreshCw size={32} class="spin" />
          <span>Loading data...</span>
        </div>
      {:else if tableData && tableData.rows.length > 0}
        <!-- Grid Header -->
        <div class="grid-header-row">
          <div class="grid-cell checkbox-cell">
            <input
              type="checkbox"
              checked={selectedRows.size === tableData.rows.length &&
                tableData.rows.length > 0}
              onchange={toggleAllRows}
            />
          </div>
          {#each tableData.columns as column}
            <div
              class="grid-cell header-cell {sortColumn === column
                ? 'sorted'
                : ''}"
              onclick={() => sortTable(column)}
              onkeydown={(e) => e.key === "Enter" && sortTable(column)}
              role="button"
              tabindex="0"
              aria-label="Sort by {column}"
            >
              <span>{column}</span>
              {#if sortColumn === column}
                <span class:rotate={sortDirection === "desc"}>
                  <ChevronDown size={12} />
                </span>
              {/if}
            </div>
          {/each}
          <div class="grid-cell actions-cell"></div>
        </div>

        <!-- Grid Rows -->
        <div class="grid-rows">
          {#each getPaginatedRows() as row}
            <div
              class="grid-row"
              class:selected={selectedRows.has(
                String(row.id || Object.values(row)[0]),
              )}
            >
              <div class="grid-cell checkbox-cell">
                <input
                  type="checkbox"
                  checked={selectedRows.has(
                    String(row.id || Object.values(row)[0]),
                  )}
                  onchange={() =>
                    toggleRowSelection(
                      String(row.id || Object.values(row)[0]),
                    )}
                />
              </div>
              {#each tableData.columns as column}
                <div class="grid-cell data-cell">
                  {#if isJsonColumn(row[column])}
                    <span
                      class="json-link"
                      onclick={() => previewJson(row[column])}
                      onkeydown={(e) =>
                        e.key === "Enter" && previewJson(row[column])}
                      role="button"
                      tabindex="0"
                      aria-label="View JSON data"
                    >
                      <Code size={10} />
                      <span>View JSON</span>
                    </span>
                  {:else}
                    <span class="cell-value" title={row[column]}>
                      {row[column] !== null && row[column] !== undefined
                        ? String(row[column]).slice(0, 50)
                        : "<NULL>"}
                    </span>
                  {/if}
                </div>
              {/each}
              <div class="grid-cell actions-cell">
                <button
                  class="icon-btn"
                  onclick={() => openEditModal(row)}
                  title="Edit row"
                >
                  <Edit size={12} />
                </button>
                <button
                  class="icon-btn danger"
                  onclick={() => handleDeleteRow(row)}
                  title="Delete row"
                >
                  <Trash2 size={12} />
                </button>
              </div>
            </div>
          {/each}
        </div>

        <!-- Pagination -->
        <div class="pagination">
          <div class="pagination-info">
            <span
              >Showing {(currentPage - 1) * rowsPerPage + 1} to {Math.min(
                currentPage * rowsPerPage,
                tableData.row_count,
              )} of {tableData.row_count} rows</span
            >
          </div>
          <div class="pagination-controls">
            <select
              bind:value={rowsPerPage}
              onchange={() =>
                selectedTable && loadTableData(selectedTable.name, 1)}
            >
              <option value="25">25 per page</option>
              <option value="50">50 per page</option>
              <option value="100">100 per page</option>
            </select>
            <button
              class="page-btn"
              onclick={() => {
                if (currentPage > 1) {
                  currentPage--;
                  selectedTable &&
                    loadTableData(selectedTable.name, currentPage);
                }
              }}
              disabled={currentPage === 1}
            >
              Previous
            </button>
            <span class="page-info"
              >Page {currentPage} of {getTotalPages()}</span
            >
            <button
              class="page-btn"
              onclick={() => {
                if (currentPage < getTotalPages()) {
                  currentPage++;
                  selectedTable &&
                    loadTableData(selectedTable.name, currentPage);
                }
              }}
              disabled={currentPage >= getTotalPages()}
            >
              Next
            </button>
          </div>
        </div>
      {:else}
        <div class="empty-state">
          <Table size={48} />
          <p>No data in this table</p>
          <button class="btn primary" onclick={openInsertModal}>
            <Plus size={14} />
            <span>Insert First Row</span>
          </button>
        </div>
      {/if}
    </div>
  {:else}
    <div class="empty-state select-table">
      <Table size={48} />
      <h3>Select a Table</h3>
      <p>Choose a table from the sidebar to view and edit its data</p>
    </div>
  {/if}
</div>

<style>
  .data-grid-area {
    flex: 1;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }

  .grid-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.75rem 1rem;
    background: var(--surface-2);
    border-bottom: 1px solid var(--border-color);
  }

  .header-left {
    display: flex;
    align-items: center;
    gap: 0.75rem;
  }

  .header-left h3 {
    margin: 0;
    font-size: 1rem;
    font-weight: 600;
  }

  .table-badge {
    font-size: 0.65rem;
    padding: 0.15rem 0.4rem;
    border-radius: 4px;
    background: var(--surface-3);
    color: var(--color-text-muted);
  }

  .table-badge.new-table {
    background: var(--success-color);
    color: white;
  }

  .header-actions {
    display: flex;
    gap: 0.5rem;
  }

  .btn {
    display: flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.4rem 0.6rem;
    font-size: 0.75rem;
    background: var(--surface-3);
    border: 1px solid var(--border-color);
    border-radius: 4px;
    cursor: pointer;
    transition: all 0.2s;
  }

  .btn:hover {
    background: var(--surface-4);
  }

  .btn.primary {
    background: var(--primary-color);
    color: white;
    border-color: var(--primary-color);
  }

  .btn.danger {
    color: var(--error-color);
  }

  .btn.danger:hover {
    background: var(--error-color);
    color: white;
  }

  .btn.disabled,
  .btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .btn.active {
    background: var(--primary-color);
    color: white;
  }

  .schema-panel {
    padding: 1rem;
    background: var(--surface-2);
    border-bottom: 1px solid var(--border-color);
    max-height: 200px;
    overflow-y: auto;
  }

  .schema-panel h4 {
    margin: 0 0 0.75rem 0;
    font-size: 0.9rem;
  }

  .schema-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
    gap: 0.75rem;
  }

  .schema-column {
    background: var(--surface-3);
    padding: 0.5rem;
    border-radius: 4px;
  }

  .column-header {
    display: flex;
    align-items: center;
    gap: 0.35rem;
    margin-bottom: 0.35rem;
  }

  .column-name {
    font-weight: 600;
    font-size: 0.8rem;
  }

  .pk-badge {
    font-size: 0.6rem;
    padding: 0.1rem 0.3rem;
    background: var(--warning-color);
    color: #000;
    border-radius: 2px;
  }

  .column-info {
    display: flex;
    gap: 0.5rem;
    font-size: 0.75rem;
    align-items: center;
  }

  .column-type {
    font-family: monospace;
  }

  .nullable-badge,
  .not-null-badge {
    font-size: 0.6rem;
    padding: 0.1rem 0.3rem;
    border-radius: 2px;
  }

  .nullable-badge {
    background: var(--surface-4);
    color: var(--color-text-muted);
  }

  .not-null-badge {
    background: var(--primary-color);
    color: white;
  }

  .column-default {
    font-size: 0.7rem;
    color: var(--color-text-muted);
    margin-top: 0.25rem;
  }

  .schema-meta {
    display: flex;
    gap: 1rem;
    margin-top: 0.75rem;
    font-size: 0.75rem;
    color: var(--color-text-muted);
  }

  .data-grid {
    flex: 1;
    overflow: auto;
    display: flex;
    flex-direction: column;
  }

  .loading-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 3rem;
    gap: 1rem;
    color: var(--color-text-muted);
  }

  .empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 3rem;
    gap: 1rem;
    color: var(--color-text-muted);
  }

  .empty-state h3 {
    margin: 0;
  }

  .grid-header-row {
    display: flex;
    background: var(--surface-2);
    border-bottom: 2px solid var(--border-color);
    position: sticky;
    top: 0;
    z-index: 1;
  }

  .grid-cell {
    padding: 0.5rem 0.75rem;
    font-size: 0.8rem;
    border-right: 1px solid var(--border-color);
  }

  .checkbox-cell {
    width: 40px;
    display: flex;
    align-items: center;
    justify-content: center;
  }

  .header-cell {
    font-weight: 600;
    cursor: pointer;
    min-width: 120px;
    display: flex;
    align-items: center;
    gap: 0.25rem;
    user-select: none;
  }

  .header-cell:hover {
    background: var(--surface-3);
  }

  .header-cell.sorted {
    color: var(--primary-color);
  }

  .rotate {
    transform: rotate(180deg);
  }

  .actions-cell {
    width: 80px;
    display: flex;
    justify-content: center;
    gap: 0.25rem;
  }

  .grid-rows {
    flex: 1;
  }

  .grid-row {
    display: flex;
    border-bottom: 1px solid var(--border-color);
  }

  .grid-row:hover {
    background: var(--surface-2);
  }

  .grid-row.selected {
    background: var(--primary-color-light);
  }

  .data-cell {
    min-width: 120px;
    max-width: 200px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .cell-value {
    display: block;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .json-link {
    display: flex;
    align-items: center;
    gap: 0.25rem;
    color: var(--primary-color);
    cursor: pointer;
    font-size: 0.7rem;
  }

  .json-link:hover {
    text-decoration: underline;
  }

  .icon-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 24px;
    height: 24px;
    background: transparent;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    color: var(--color-text-muted);
  }

  .icon-btn:hover {
    background: var(--surface-3);
    color: var(--text-color);
  }

  .icon-btn.danger:hover {
    background: var(--error-color);
    color: white;
  }

  .pagination {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.75rem 1rem;
    background: var(--surface-2);
    border-top: 1px solid var(--border-color);
  }

  .pagination-info {
    font-size: 0.8rem;
    color: var(--color-text-muted);
  }

  .pagination-controls {
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }

  .pagination-controls select {
    padding: 0.35rem;
    background: var(--surface-3);
    border: 1px solid var(--border-color);
    border-radius: 4px;
    font-size: 0.75rem;
  }

  .page-btn {
    padding: 0.35rem 0.75rem;
    background: var(--surface-3);
    border: 1px solid var(--border-color);
    border-radius: 4px;
    font-size: 0.75rem;
    cursor: pointer;
  }

  .page-btn:hover:not(:disabled) {
    background: var(--surface-4);
  }

  .page-btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .page-info {
    font-size: 0.8rem;
    color: var(--color-text-muted);
  }

  :global(.spin) {
    animation: spin 1s linear infinite;
  }

  @keyframes spin {
    from {
      transform: rotate(0deg);
    }
    to {
      transform: rotate(360deg);
    }
  }
</style>
