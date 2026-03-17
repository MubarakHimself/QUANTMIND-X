<script lang="ts">
  import { onMount } from "svelte";

  import DatabaseHeader from "./DatabaseHeader.svelte";
  import DatabaseStats from "./DatabaseStats.svelte";
  import TableSidebar from "./TableSidebar.svelte";
  import DataGrid from "./DataGrid.svelte";
  import QueryEditor from "./QueryEditor.svelte";
  import InsertRowModal from "./InsertRowModal.svelte";
  import EditRowModal from "./EditRowModal.svelte";
  import JsonPreviewModal from "./JsonPreviewModal.svelte";

  // ============================================================================
  // TYPE DEFINITIONS
  // ============================================================================

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

  interface QueryResult {
    columns: string[];
    rows: Array<Record<string, any>>;
    row_count: number;
    execution_time_ms: number;
    error?: string;
  }

  interface DatabaseStats {
    total_size_bytes: number;
    sqlite_path: string;
    duckdb_path: string;
    table_count: number;
  }

  // ============================================================================
  // STATE
  // ============================================================================

  // Table list
  let tables: TableInfo[] = $state([]);
  let filteredTables: TableInfo[] = $state([]);
  let selectedTable: TableInfo | null = $state(null);

  // Data grid
  let tableData: QueryResult | null = $state(null);
  let selectedRows: Set<string> = $state(new Set());
  let currentPage = $state(1);
  let rowsPerPage = 25;
  let sortColumn: string | null = $state(null);
  let sortDirection: "asc" | "desc" = $state("asc");

  // Query editor
  let queryInput = $state("");
  let queryHistory: string[] = $state([]);
  let queryResults: QueryResult | null = $state(null);
  let queryHistoryIndex = -1;

  // Table info
  let showTableInfo = $state(false);
  let tableSchema: ColumnInfo[] = [];

  // Database stats
  let dbStats: DatabaseStats | null = $state(null);

  // Modals
  let insertModalOpen = $state(false);
  let editModalOpen = $state(false);
  let jsonPreviewModalOpen = $state(false);
  let jsonPreviewData: any = $state(null);

  // Forms
  let newRowData: Record<string, any> = $state({});
  let editingRow: Record<string, any> | null = $state(null);

  // Loading states
  let isLoading = $state(false);
  let isQueryRunning = $state(false);

  // Search/filter
  let searchQuery = "";
  let tableTypeFilter: "all" | "existing" | "new" = "all";

  // ============================================================================
  // LIFECYCLE
  // ============================================================================

  onMount(() => {
    loadTables();
    loadDatabaseStats();
  });

  // ============================================================================
  // DATA LOADING
  // ============================================================================

  async function loadTables() {
    isLoading = true;
    try {
      const res = await fetch("http://localhost:8000/api/database/tables");
      if (res.ok) {
        const data = await res.json();
        tables = data.tables || [];
        applyFilters();
      } else {
        console.error("Failed to load tables:", res.statusText);
      }
    } catch (e) {
      console.error("Failed to load tables:", e);
    } finally {
      isLoading = false;
    }
  }


  async function loadDatabaseStats() {
    try {
      const res = await fetch("http://localhost:8000/api/database/stats");
      if (res.ok) {
        dbStats = await res.json();
      } else {
        console.error("Failed to load database stats:", res.statusText);
      }
    } catch (e) {
      console.error("Failed to load database stats:", e);
    }
  }

  // ============================================================================
  // TABLE OPERATIONS
  // ============================================================================

  async function selectTable(table: TableInfo) {
    selectedTable = table;
    currentPage = 1;
    selectedRows.clear();
    showTableInfo = false;
    await loadTableData(table.name);
  }

  async function loadTableData(
    tableName: string,
    page = 1,
    limit = rowsPerPage,
  ) {
    isLoading = true;
    try {
      const offset = (page - 1) * limit;
      const res = await fetch(
        `http://localhost:8000/api/database/table/${tableName}?limit=${limit}&offset=${offset}`,
      );
      if (res.ok) {
        tableData = await res.json();
      } else {
        console.error("Failed to load table data:", res.statusText);
      }
    } catch (e) {
      console.error("Failed to load table data:", e);
    } finally {
      isLoading = false;
    }
  }

  async function loadTableSchema(tableName: string) {
    try {
      const res = await fetch(
        `http://localhost:8000/api/database/schema/${tableName}`,
      );
      if (res.ok) {
        const data = await res.json();
        tableSchema = data.columns || [];
      } else {
        const table = tables.find((t) => t.name === tableName);
        tableSchema = table?.columns || [];
      }
    } catch (e) {
      const table = tables.find((t) => t.name === tableName);
      tableSchema = table?.columns || [];
    }
  }

  // ============================================================================
  // QUERY OPERATIONS
  // ============================================================================

  async function executeQuery() {
    if (!queryInput.trim()) return;

    isQueryRunning = true;
    const startTime = Date.now();

    try {
      const res = await fetch("http://localhost:8000/api/database/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: queryInput }),
      });

      const executionTime = Date.now() - startTime;

      if (res.ok) {
        queryResults = await res.json();
        queryResults!.execution_time_ms = executionTime;
        addToHistory(queryInput);
      } else {
        const error = await res.json();
        queryResults = {
          columns: [],
          rows: [],
          row_count: 0,
          execution_time_ms: executionTime,
          error: error.detail || "Query failed",
        };
        addToHistory(queryInput);
      }
    } catch (e) {
      queryResults = {
        columns: [],
        rows: [],
        row_count: 0,
        execution_time_ms: Date.now() - startTime,
        error: "Failed to connect to database",
      };
      addToHistory(queryInput);
    } finally {
      isQueryRunning = false;
    }
  }

  function addToHistory(query: string) {
    queryHistory = [query, ...queryHistory.filter((q) => q !== query)].slice(
      0,
      10,
    );
    queryHistoryIndex = -1;
  }

  function navigateHistory(direction: "up" | "down") {
    if (queryHistory.length === 0) return;

    if (direction === "up") {
      queryHistoryIndex = Math.min(
        queryHistoryIndex + 1,
        queryHistory.length - 1,
      );
    } else {
      queryHistoryIndex = Math.max(queryHistoryIndex - 1, -1);
    }

    if (queryHistoryIndex >= 0) {
      queryInput = queryHistory[queryHistoryIndex];
    }
  }

  // ============================================================================
  // DATA OPERATIONS
  // ============================================================================

  function openInsertModal() {
    if (!selectedTable) return;
    newRowData = {};
    selectedTable.columns.forEach((col) => {
      if (!col.primary_key) {
        newRowData[col.name] = col.default_value ?? "";
      }
    });
    insertModalOpen = true;
  }

  async function insertRow() {
    if (!selectedTable) return;

    try {
      const res = await fetch(
        `http://localhost:8000/api/database/table/${selectedTable.name}`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(newRowData),
        },
      );

      if (res.ok) {
        await loadTableData(selectedTable.name, currentPage);
        insertModalOpen = false;
      } else {
        alert("Failed to insert row");
      }
    } catch (e) {
      console.error("Failed to insert row:", e);
      // For development, add locally
      if (tableData) {
        const newRow = { id: tableData.rows.length + 1, ...newRowData };
        tableData.rows.push(newRow);
        tableData.row_count++;
      }
      insertModalOpen = false;
    }
  }

  function openEditModal(row: Record<string, any>) {
    editingRow = { ...row };
    editModalOpen = true;
  }

  async function updateRow() {
    if (!selectedTable || !editingRow) return;

    try {
      const res = await fetch(
        `http://localhost:8000/api/database/table/${selectedTable.name}`,
        {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(editingRow),
        },
      );

      if (res.ok) {
        await loadTableData(selectedTable.name, currentPage);
        editModalOpen = false;
      } else {
        alert("Failed to update row");
      }
    } catch (e) {
      console.error("Failed to update row:", e);
      // For development, update locally
      if (tableData) {
        const index = tableData.rows.findIndex((r) => r.id === editingRow?.id);
        if (index >= 0) {
          tableData.rows[index] = editingRow;
        }
      }
      editModalOpen = false;
    }
  }

  async function deleteSelectedRows() {
    if (!selectedTable || selectedRows.size === 0) return;

    if (!confirm(`Delete ${selectedRows.size} row(s)?`)) return;

    try {
      const res = await fetch(
        `http://localhost:8000/api/database/table/${selectedTable.name}`,
        {
          method: "DELETE",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ ids: Array.from(selectedRows) }),
        },
      );

      if (res.ok) {
        await loadTableData(selectedTable.name, currentPage);
        selectedRows.clear();
      } else {
        alert("Failed to delete rows");
      }
    } catch (e) {
      console.error("Failed to delete rows:", e);
      // For development, remove locally
      if (tableData) {
        tableData.rows = tableData.rows.filter(
          (r) => !selectedRows.has(String(r.id)),
        );
        tableData.row_count -= selectedRows.size;
      }
      selectedRows.clear();
    }
  }

  async function exportTable(format: "csv" | "json") {
    if (!selectedTable) return;

    try {
      const res = await fetch(
        `http://localhost:8000/api/database/export/${selectedTable.name}?format=${format}`,
      );
      if (res.ok) {
        const blob = await res.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `${selectedTable.name}.${format === "csv" ? "csv" : "json"}`;
        a.click();
        URL.revokeObjectURL(url);
      }
    } catch (e) {
      console.error("Failed to export table:", e);
      alert("Export failed");
    }
  }

  async function importTable(file: File) {
    if (!selectedTable) return;

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch(
        `http://localhost:8000/api/database/import/${selectedTable.name}`,
        {
          method: "POST",
          body: formData,
        },
      );

      if (res.ok) {
        await loadTableData(selectedTable.name, currentPage);
        await loadTables(); // Update row counts
      } else {
        alert("Import failed");
      }
    } catch (e) {
      console.error("Failed to import table:", e);
      alert("Import failed");
    }
  }

  // ============================================================================
  // UI HELPERS
  // ============================================================================

  function applyFilters() {
    filteredTables = tables.filter((table) => {
      if (tableTypeFilter === "existing" && table.is_new) return false;
      if (tableTypeFilter === "new" && !table.is_new) return false;
      if (searchQuery) {
        const query = searchQuery.toLowerCase();
        return table.name.toLowerCase().includes(query);
      }
      return true;
    });
  }

  function toggleRowSelection(rowId: string) {
    if (selectedRows.has(rowId)) {
      selectedRows.delete(rowId);
    } else {
      selectedRows.add(rowId);
    }
    selectedRows = new Set(selectedRows); // Trigger reactivity
  }

  function toggleAllRows() {
    if (!tableData) return;

    if (selectedRows.size === tableData.rows.length) {
      selectedRows.clear();
    } else {
      selectedRows = new Set(
        tableData.rows.map((r) => String(r.id || Object.values(r)[0])),
      );
    }
    selectedRows = new Set(selectedRows); // Trigger reactivity
  }

  function sortTable(column: string) {
    if (sortColumn === column) {
      sortDirection = sortDirection === "asc" ? "desc" : "asc";
    } else {
      sortColumn = column;
      sortDirection = "asc";
    }

    if (!tableData) return;

    tableData.rows.sort((a, b) => {
      const aVal = a[column];
      const bVal = b[column];

      if (aVal === bVal) return 0;

      const comparison = aVal < bVal ? -1 : 1;
      return sortDirection === "asc" ? comparison : -comparison;
    });
  }

  function getPaginatedRows() {
    if (!tableData) return [];
    const start = (currentPage - 1) * rowsPerPage;
    return tableData.rows.slice(start, start + rowsPerPage);
  }

  function getTotalPages() {
    if (!tableData) return 1;
    return Math.ceil(tableData.row_count / rowsPerPage);
  }

  function formatBytes(bytes: number) {
    if (bytes === 0) return "0 B";
    const k = 1024;
    const sizes = ["B", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + " " + sizes[i];
  }

  function formatTimestamp(dateStr: string) {
    const date = new Date(dateStr);
    return (
      date.toLocaleDateString() +
      " " +
      date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
    );
  }

  function getColumnTypeColor(type: string) {
    if (type.includes("INT")) return "#3b82f6";
    if (type.includes("REAL") || type.includes("FLOAT")) return "#10b981";
    if (type.includes("TEXT")) return "#f59e0b";
    if (type.includes("BLOB")) return "#8b5cf6";
    return "#6b7280";
  }

  function previewJson(data: any) {
    jsonPreviewData = data;
    jsonPreviewModalOpen = true;
  }

  function isJsonColumn(value: any) {
    if (typeof value === "string") {
      try {
        JSON.parse(value);
        return true;
      } catch {
        return false;
      }
    }
    return typeof value === "object" && value !== null;
  }
</script>

<div class="database-view">
  <!-- Header -->
  <DatabaseHeader
    onRefresh={loadTables}
    onLoadStats={loadDatabaseStats}
  />

  <!-- Database Stats Banner -->
  <DatabaseStats stats={dbStats} />

  <!-- Main Content -->
  <div class="db-content">
    <!-- Table List Sidebar -->
    <TableSidebar
      {tables}
      {filteredTables}
      {selectedTable}
      {isLoading}
      searchQuery={searchQuery}
      tableTypeFilter={tableTypeFilter}
      onSelectTable={selectTable}
      onSearchChange={applyFilters}
      onFilterChange={applyFilters}
    />

    <!-- Data Grid Area -->
    <div class="data-grid-area">
      <DataGrid
        {tableData}
        {selectedTable}
        {selectedRows}
        {currentPage}
        {rowsPerPage}
        {sortColumn}
        {sortDirection}
        {isLoading}
        {showTableInfo}
        formatBytes={formatBytes}
        formatTimestamp={formatTimestamp}
        getColumnTypeColor={getColumnTypeColor}
        {isJsonColumn}
        {previewJson}
        {toggleRowSelection}
        {toggleAllRows}
        {sortTable}
        {getPaginatedRows}
        {getTotalPages}
        loadTableData={loadTableData}
        {deleteSelectedRows}
        {openInsertModal}
        {openEditModal}
        {exportTable}
        {importTable}
      />
    </div>

    <!-- Query Editor Panel -->
    <QueryEditor
      bind:queryInput
      {queryHistory}
      {queryResults}
      {isQueryRunning}
      {executeQuery}
      {navigateHistory}
      {isJsonColumn}
      {previewJson}
    />
  </div>
</div>

<!-- Insert Modal -->
<InsertRowModal
  bind:open={insertModalOpen}
  {selectedTable}
  bind:newRowData
  {getColumnTypeColor}
  {insertRow}
/>

<!-- Edit Modal -->
<EditRowModal
  bind:open={editModalOpen}
  {editingRow}
  {selectedTable}
  {getColumnTypeColor}
  {updateRow}
  bind:jsonPreviewData
/>

<!-- JSON Preview Modal -->
<JsonPreviewModal
  bind:open={jsonPreviewModalOpen}
  jsonData={jsonPreviewData}
/>

<style>
  .database-view {
    display: flex;
    flex-direction: column;
    height: 100%;
    background: var(--bg-primary);
    overflow: hidden;
  }

  /* Main Content */
  .db-content {
    flex: 1;
    display: flex;
    overflow: hidden;
  }

  /* Data Grid Area */
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
    padding: 16px 20px;
    background: var(--bg-secondary);
    border-bottom: 1px solid var(--border-subtle);
  }

  .grid-header .header-left {
    display: flex;
    align-items: center;
    gap: 10px;
  }

  .grid-header h3 {
    margin: 0;
    font-size: 14px;
    color: var(--text-primary);
  }

  .table-badge {
    padding: 2px 8px;
    border-radius: 6px;
    font-size: 10px;
    font-weight: 500;
    text-transform: uppercase;
    background: var(--bg-tertiary);
    color: var(--text-muted);
  }

  .table-badge.new-table {
    background: var(--accent-success);
    color: var(--bg-primary);
  }

  .header-actions {
    display: flex;
    gap: 6px;
  }

  /* Schema Panel */
  .schema-panel {
    padding: 16px 20px;
    background: var(--bg-tertiary);
    border-bottom: 1px solid var(--border-subtle);
  }

  .schema-panel h4 {
    margin: 0 0 12px;
    font-size: 13px;
    color: var(--text-primary);
  }

  .schema-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 12px;
    margin-bottom: 12px;
  }

  .schema-column {
    background: var(--bg-secondary);
    border-radius: 8px;
    padding: 10px;
    border: 1px solid var(--border-subtle);
  }

  .column-header {
    display: flex;
    align-items: center;
    gap: 6px;
    margin-bottom: 6px;
  }

  .column-name {
    font-weight: 500;
    font-size: 12px;
    color: var(--text-primary);
    font-family: "JetBrains Mono", monospace;
  }

  .pk-badge {
    padding: 2px 6px;
    background: var(--accent-primary);
    color: var(--bg-primary);
    border-radius: 4px;
    font-size: 9px;
    font-weight: 600;
  }

  .column-info {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 11px;
  }

  .column-type {
    font-weight: 500;
  }

  .nullable-badge,
  .not-null-badge {
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 9px;
    font-weight: 500;
  }

  .nullable-badge {
    background: rgba(245, 158, 11, 0.2);
    color: #f59e0b;
  }

  .not-null-badge {
    background: rgba(107, 114, 128, 0.2);
    color: var(--text-muted);
  }

  .column-default {
    display: flex;
    gap: 6px;
    font-size: 10px;
    margin-top: 4px;
  }

  .column-default .label {
    color: var(--text-muted);
  }

  .column-default .value {
    color: var(--text-secondary);
    font-family: "JetBrains Mono", monospace;
  }

  .schema-meta {
    display: flex;
    gap: 16px;
    font-size: 11px;
    color: var(--text-muted);
  }

  /* Data Grid */
  .data-grid {
    flex: 1;
    overflow: auto;
    display: flex;
    flex-direction: column;
  }

  .grid-header-row {
    display: flex;
    position: sticky;
    top: 0;
    background: var(--bg-tertiary);
    z-index: 10;
    border-bottom: 1px solid var(--border-subtle);
  }

  .grid-cell {
    padding: 10px 12px;
    font-size: 12px;
    border-right: 1px solid var(--border-subtle);
    display: flex;
    align-items: center;
    min-width: 120px;
  }

  .checkbox-cell {
    width: 40px;
    min-width: 40px;
    justify-content: center;
  }

  .header-cell {
    font-weight: 500;
    color: var(--text-muted);
    cursor: pointer;
    user-select: none;
    gap: 4px;
  }

  .header-cell:hover {
    color: var(--text-primary);
  }

  .header-cell.sorted {
    color: var(--accent-primary);
  }

  .header-cell .rotate {
    transform: rotate(180deg);
  }

  .actions-cell {
    width: 80px;
    min-width: 80px;
    justify-content: center;
    gap: 4px;
  }

  .data-cell {
    color: var(--text-primary);
    font-family: "JetBrains Mono", monospace;
    font-size: 11px;
    overflow: hidden;
  }

  .cell-value {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .json-link {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 2px 6px;
    background: var(--bg-tertiary);
    border-radius: 4px;
    color: var(--accent-primary);
    cursor: pointer;
    font-size: 10px;
  }

  .json-link:hover {
    background: var(--accent-primary);
    color: var(--bg-primary);
  }

  .grid-rows {
    flex: 1;
  }

  .grid-row {
    display: flex;
    border-bottom: 1px solid var(--border-subtle);
    transition: background 0.15s;
  }

  .grid-row:hover {
    background: var(--bg-secondary);
  }

  .grid-row.selected {
    background: rgba(99, 102, 241, 0.1);
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
    color: var(--text-muted);
    cursor: pointer;
    transition: all 0.15s;
  }

  .icon-btn:hover {
    background: var(--bg-tertiary);
    color: var(--text-primary);
  }

  .icon-btn.danger:hover {
    background: rgba(239, 68, 68, 0.2);
    color: #ef4444;
  }

  /* Pagination */
  .pagination {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 20px;
    background: var(--bg-secondary);
    border-top: 1px solid var(--border-subtle);
  }

  .pagination-info {
    font-size: 12px;
    color: var(--text-muted);
  }

  .pagination-controls {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .pagination-controls select {
    padding: 6px 10px;
    background: var(--bg-tertiary);
    border: 1px solid var(--border-subtle);
    border-radius: 6px;
    color: var(--text-primary);
    font-size: 12px;
  }

  .page-btn {
    padding: 6px 12px;
    background: var(--bg-tertiary);
    border: 1px solid var(--border-subtle);
    border-radius: 6px;
    color: var(--text-secondary);
    font-size: 12px;
    cursor: pointer;
    transition: all 0.15s;
  }

  .page-btn:hover:not(:disabled) {
    background: var(--bg-surface);
  }

  .page-btn:disabled {
    opacity: 0.4;
    cursor: not-allowed;
  }

  .page-info {
    font-size: 12px;
    color: var(--text-muted);
  }

  /* Query Panel */
  .query-panel {
    width: 400px;
    background: var(--bg-secondary);
    border-left: 1px solid var(--border-subtle);
    display: flex;
    flex-direction: column;
  }

  .query-header {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 16px;
    border-bottom: 1px solid var(--border-subtle);
  }

  .query-header h3 {
    margin: 0;
    font-size: 14px;
    color: var(--text-primary);
  }

  .query-editor {
    flex: 0 0 200px;
    border-bottom: 1px solid var(--border-subtle);
  }

  .query-editor textarea {
    width: 100%;
    height: 100%;
    padding: 16px;
    background: var(--bg-tertiary);
    border: none;
    color: var(--text-primary);
    font-family: "JetBrains Mono", monospace;
    font-size: 12px;
    line-height: 1.6;
    resize: none;
    outline: none;
  }

  .query-actions {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 16px;
    background: var(--bg-tertiary);
    border-bottom: 1px solid var(--border-subtle);
  }

  .query-hint {
    display: flex;
    flex-direction: column;
    gap: 2px;
    font-size: 10px;
    color: var(--text-muted);
  }

  .query-history {
    padding: 16px;
    border-bottom: 1px solid var(--border-subtle);
    max-height: 200px;
    overflow-y: auto;
  }

  .query-history h4 {
    margin: 0 0 8px;
    font-size: 12px;
    color: var(--text-muted);
  }

  .history-list {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .history-item {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 6px 10px;
    background: var(--bg-tertiary);
    border-radius: 6px;
    cursor: pointer;
    transition: background 0.15s;
  }

  .history-item:hover {
    background: var(--bg-surface);
  }

  .history-item .query-text {
    font-family: "JetBrains Mono", monospace;
    font-size: 10px;
    color: var(--text-secondary);
  }

  .query-results {
    flex: 1;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }

  .results-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 16px;
    background: var(--bg-tertiary);
    border-bottom: 1px solid var(--border-subtle);
  }

  .results-header h4 {
    margin: 0;
    font-size: 12px;
    color: var(--text-primary);
  }

  .error-message {
    display: flex;
    align-items: center;
    gap: 6px;
    color: #ef4444;
    font-size: 11px;
  }

  .results-stats {
    font-size: 11px;
    color: var(--text-muted);
  }

  .results-grid {
    flex: 1;
    overflow: auto;
  }

  .results-header-row {
    display: flex;
    position: sticky;
    top: 0;
    background: var(--bg-surface);
    z-index: 5;
  }

  .results-cell {
    padding: 8px 12px;
    font-size: 11px;
    border-right: 1px solid var(--border-subtle);
    border-bottom: 1px solid var(--border-subtle);
    min-width: 100px;
  }

  .results-cell.header {
    font-weight: 500;
    color: var(--text-muted);
    background: var(--bg-surface);
  }

  .results-row {
    display: flex;
  }

  .results-row:hover .results-cell {
    background: var(--bg-secondary);
  }

  .empty-results {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 40px;
    color: var(--text-muted);
    text-align: center;
    gap: 12px;
  }

  /* Empty States */
  .empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 60px 20px;
    color: var(--text-muted);
    text-align: center;
    gap: 16px;
  }

  .empty-state p {
    margin: 0;
    font-size: 13px;
  }

  .loading-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 60px 20px;
    color: var(--text-muted);
    text-align: center;
    gap: 16px;
  }

  @keyframes spin {
    from {
      transform: rotate(0deg);
    }
    to {
      transform: rotate(360deg);
    }
  }

  /* Modal */
  .modal-overlay {
    position: fixed;
    inset: 0;
    background: rgba(0, 0, 0, 0.7);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 100;
  }

  .modal {
    background: var(--bg-secondary);
    border-radius: 12px;
    width: 500px;
    max-width: 90%;
    max-height: 85vh;
    overflow: hidden;
    display: flex;
    flex-direction: column;
  }

  .modal.large {
    width: 700px;
  }

  .modal-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    padding: 16px 20px;
    border-bottom: 1px solid var(--border-subtle);
  }

  .modal-header h3 {
    margin: 0;
    font-size: 16px;
    color: var(--text-primary);
  }

  .modal-subtitle {
    margin: 4px 0 0;
    font-size: 11px;
    color: var(--text-muted);
  }

  .modal-content {
    flex: 1;
    overflow-y: auto;
    padding: 20px;
  }

  .form-group {
    margin-bottom: 16px;
  }

  .form-group label {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 6px;
    font-size: 12px;
    font-weight: 500;
    color: var(--text-primary);
  }

  .type-badge {
    font-size: 10px;
    font-family: "JetBrains Mono", monospace;
    font-weight: 400;
  }

  .form-group input,
  .form-group textarea {
    width: 100%;
    padding: 10px 12px;
    background: var(--bg-tertiary);
    border: 1px solid var(--border-subtle);
    border-radius: 6px;
    color: var(--text-primary);
    font-size: 13px;
    font-family: "JetBrains Mono", monospace;
    outline: none;
    transition: border-color 0.15s;
  }

  .form-group input:focus,
  .form-group textarea:focus {
    border-color: var(--accent-primary);
  }

  .modal-actions {
    display: flex;
    justify-content: flex-end;
    gap: 8px;
    padding-top: 16px;
    border-top: 1px solid var(--border-subtle);
  }

  .json-preview {
    background: var(--bg-tertiary);
    padding: 16px;
    border-radius: 8px;
    font-family: "JetBrains Mono", monospace;
    font-size: 11px;
    line-height: 1.6;
    color: var(--text-primary);
    overflow: auto;
    max-height: 500px;
  }
</style>
