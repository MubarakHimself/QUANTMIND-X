<script lang="ts">
  import { onMount } from 'svelte';
  import {
    Database, Table, Search, Filter, Plus, Edit, Trash2, Eye,
    RefreshCw, Download, Upload, Play, FileText, Code, Hash,
    Clock, HardDrive, ChevronRight, ChevronDown, X, Check,
    AlertCircle, FileCode, Terminal, Settings as SettingsIcon
  } from 'lucide-svelte';

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
  let tables: TableInfo[] = [];
  let filteredTables: TableInfo[] = [];
  let selectedTable: TableInfo | null = null;

  // Data grid
  let tableData: QueryResult | null = null;
  let selectedRows: Set<string> = new Set();
  let currentPage = 1;
  let rowsPerPage = 25;
  let sortColumn: string | null = null;
  let sortDirection: 'asc' | 'desc' = 'asc';

  // Query editor
  let queryInput = '';
  let queryHistory: string[] = [];
  let queryResults: QueryResult | null = null;
  let queryHistoryIndex = -1;

  // Table info
  let showTableInfo = false;
  let tableSchema: ColumnInfo[] = [];

  // Database stats
  let dbStats: DatabaseStats | null = null;

  // Modals
  let insertModalOpen = false;
  let editModalOpen = false;
  let jsonPreviewModalOpen = false;
  let jsonPreviewData: any = null;

  // Forms
  let newRowData: Record<string, any> = {};
  let editingRow: Record<string, any> | null = null;

  // Loading states
  let isLoading = false;
  let isQueryRunning = false;

  // Search/filter
  let searchQuery = '';
  let tableTypeFilter: 'all' | 'existing' | 'new' = 'all';

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
      const res = await fetch('http://localhost:8000/api/database/tables');
      if (res.ok) {
        const data = await res.json();
        tables = data.tables || [];
        applyFilters();
      } else {
        loadMockTables();
      }
    } catch (e) {
      console.error('Failed to load tables:', e);
      loadMockTables();
    } finally {
      isLoading = false;
    }
  }

  function loadMockTables() {
    tables = [
      // Existing tables
      {
        name: 'prop_firm_accounts',
        row_count: 12,
        is_new: false,
        size_bytes: 8192,
        last_updated: '2024-01-20T10:30:00Z',
        columns: [
          { name: 'id', type: 'INTEGER', nullable: false, primary_key: true },
          { name: 'firm_name', type: 'TEXT', nullable: false, primary_key: false },
          { name: 'account_id', type: 'INTEGER', nullable: false, primary_key: false },
          { name: 'balance', type: 'REAL', nullable: false, primary_key: false },
          { name: 'equity', type: 'REAL', nullable: false, primary_key: false },
          { name: 'max_drawdown', type: 'REAL', nullable: true, primary_key: false },
          { name: 'created_at', type: 'TEXT', nullable: false, primary_key: false }
        ]
      },
      {
        name: 'daily_snapshots',
        row_count: 342,
        is_new: false,
        size_bytes: 131072,
        last_updated: '2024-01-20T23:59:00Z',
        columns: [
          { name: 'id', type: 'INTEGER', nullable: false, primary_key: true },
          { name: 'account_id', type: 'INTEGER', nullable: false, primary_key: false },
          { name: 'date', type: 'TEXT', nullable: false, primary_key: false },
          { name: 'balance', type: 'REAL', nullable: false, primary_key: false },
          { name: 'equity', type: 'REAL', nullable: false, primary_key: false },
          { name: 'floating_pnl', type: 'REAL', nullable: true, primary_key: false },
          { name: 'margin_used', type: 'REAL', nullable: false, primary_key: false }
        ]
      },
      {
        name: 'trade_proposals',
        row_count: 89,
        is_new: false,
        size_bytes: 65536,
        last_updated: '2024-01-20T15:22:00Z',
        columns: [
          { name: 'id', type: 'INTEGER', nullable: false, primary_key: true },
          { name: 'strategy_name', type: 'TEXT', nullable: false, primary_key: false },
          { name: 'symbol', type: 'TEXT', nullable: false, primary_key: false },
          { name: 'action', type: 'TEXT', nullable: false, primary_key: false },
          { name: 'entry_price', type: 'REAL', nullable: true, primary_key: false },
          { name: 'confidence', type: 'REAL', nullable: false, primary_key: false },
          { name: 'status', type: 'TEXT', nullable: false, primary_key: false },
          { name: 'created_at', type: 'TEXT', nullable: false, primary_key: false }
        ]
      },
      {
        name: 'agent_tasks',
        row_count: 256,
        is_new: false,
        size_bytes: 98304,
        last_updated: '2024-01-20T16:45:00Z',
        columns: [
          { name: 'id', type: 'TEXT', nullable: false, primary_key: true },
          { name: 'agent_type', type: 'TEXT', nullable: false, primary_key: false },
          { name: 'task_type', type: 'TEXT', nullable: false, primary_key: false },
          { name: 'status', type: 'TEXT', nullable: false, primary_key: false },
          { name: 'input_data', type: 'TEXT', nullable: true, primary_key: false },
          { name: 'result', type: 'TEXT', nullable: true, primary_key: false },
          { name: 'created_at', type: 'TEXT', nullable: false, primary_key: false }
        ]
      },
      {
        name: 'strategy_performance',
        row_count: 45,
        is_new: false,
        size_bytes: 49152,
        last_updated: '2024-01-20T14:30:00Z',
        columns: [
          { name: 'id', type: 'INTEGER', nullable: false, primary_key: true },
          { name: 'strategy_name', type: 'TEXT', nullable: false, primary_key: false },
          { name: 'total_trades', type: 'INTEGER', nullable: false, primary_key: false },
          { name: 'win_rate', type: 'REAL', nullable: false, primary_key: false },
          { name: 'total_profit', type: 'REAL', nullable: false, primary_key: false },
          { name: 'max_drawdown', type: 'REAL', nullable: true, primary_key: false },
          { name: 'sharpe_ratio', type: 'REAL', nullable: true, primary_key: false }
        ]
      },
      {
        name: 'risk_tier_transitions',
        row_count: 128,
        is_new: false,
        size_bytes: 57344,
        last_updated: '2024-01-20T17:15:00Z',
        columns: [
          { name: 'id', type: 'INTEGER', nullable: false, primary_key: true },
          { name: 'account_id', type: 'INTEGER', nullable: false, primary_key: false },
          { name: 'from_tier', type: 'TEXT', nullable: false, primary_key: false },
          { name: 'to_tier', type: 'TEXT', nullable: false, primary_key: false },
          { name: 'reason', type: 'TEXT', nullable: true, primary_key: false },
          { name: 'timestamp', type: 'TEXT', nullable: false, primary_key: false }
        ]
      },
      {
        name: 'crypto_trades',
        row_count: 67,
        is_new: false,
        size_bytes: 40960,
        last_updated: '2024-01-20T18:00:00Z',
        columns: [
          { name: 'id', type: 'INTEGER', nullable: false, primary_key: true },
          { name: 'symbol', type: 'TEXT', nullable: false, primary_key: false },
          { name: 'side', type: 'TEXT', nullable: false, primary_key: false },
          { name: 'quantity', type: 'REAL', nullable: false, primary_key: false },
          { name: 'price', type: 'REAL', nullable: false, primary_key: false },
          { name: 'timestamp', type: 'TEXT', nullable: false, primary_key: false }
        ]
      },
      // New tables
      {
        name: 'strategy_folders',
        row_count: 8,
        is_new: true,
        size_bytes: 4096,
        last_updated: '2024-01-19T10:00:00Z',
        columns: [
          { name: 'id', type: 'INTEGER', nullable: false, primary_key: true },
          { name: 'name', type: 'TEXT', nullable: false, primary_key: false },
          { name: 'type', type: 'TEXT', nullable: false, primary_key: false },
          { name: 'path', type: 'TEXT', nullable: false, primary_key: false },
          { name: 'created_at', type: 'TEXT', nullable: false, primary_key: false }
        ]
      },
      {
        name: 'backtest_results',
        row_count: 23,
        is_new: true,
        size_bytes: 81920,
        last_updated: '2024-01-19T14:30:00Z',
        columns: [
          { name: 'id', type: 'TEXT', nullable: false, primary_key: true },
          { name: 'strategy_name', type: 'TEXT', nullable: false, primary_key: false },
          { name: 'backtest_type', type: 'TEXT', nullable: false, primary_key: false },
          { name: 'start_date', type: 'TEXT', nullable: false, primary_key: false },
          { name: 'end_date', type: 'TEXT', nullable: false, primary_key: false },
          { name: 'total_trades', type: 'INTEGER', nullable: false, primary_key: false },
          { name: 'profit_factor', type: 'REAL', nullable: true, primary_key: false },
          { name: 'results_json', type: 'TEXT', nullable: true, primary_key: false }
        ]
      },
      {
        name: 'shared_assets',
        row_count: 15,
        is_new: true,
        size_bytes: 24576,
        last_updated: '2024-01-18T09:15:00Z',
        columns: [
          { name: 'id', type: 'TEXT', nullable: false, primary_key: true },
          { name: 'name', type: 'TEXT', nullable: false, primary_key: false },
          { name: 'category', type: 'TEXT', nullable: false, primary_key: false },
          { name: 'version', type: 'TEXT', nullable: false, primary_key: false },
          { name: 'filesystem_path', type: 'TEXT', nullable: false, primary_key: false },
          { name: 'checksum', type: 'TEXT', nullable: false, primary_key: false },
          { name: 'created_by', type: 'TEXT', nullable: false, primary_key: false }
        ]
      },
      {
        name: 'bot_registry',
        row_count: 5,
        is_new: true,
        size_bytes: 8192,
        last_updated: '2024-01-20T11:00:00Z',
        columns: [
          { name: 'bot_id', type: 'TEXT', nullable: false, primary_key: true },
          { name: 'name', type: 'TEXT', nullable: false, primary_key: false },
          { name: 'strategy_type', type: 'TEXT', nullable: false, primary_key: false },
          { name: 'status', type: 'TEXT', nullable: false, primary_key: false },
          { name: 'deployed_at', type: 'TEXT', nullable: true, primary_key: false },
          { name: 'config_json', type: 'TEXT', nullable: true, primary_key: false }
        ]
      },
      {
        name: 'bot_circuit_breaker',
        row_count: 3,
        is_new: true,
        size_bytes: 4096,
        last_updated: '2024-01-20T12:30:00Z',
        columns: [
          { name: 'bot_id', type: 'TEXT', nullable: false, primary_key: true },
          { name: 'daily_loss', type: 'REAL', nullable: false, primary_key: false },
          { name: 'threshold', type: 'REAL', nullable: false, primary_key: false },
          { name: 'is_tripped', type: 'INTEGER', nullable: false, primary_key: false },
          { name: 'trip_count', type: 'INTEGER', nullable: false, primary_key: false },
          { name: 'last_trip_time', type: 'TEXT', nullable: true, primary_key: false }
        ]
      },
      {
        name: 'trade_journal',
        row_count: 54,
        is_new: true,
        size_bytes: 36864,
        last_updated: '2024-01-20T16:20:00Z',
        columns: [
          { name: 'id', type: 'INTEGER', nullable: false, primary_key: true },
          { name: 'trade_id', type: 'INTEGER', nullable: false, primary_key: false },
          { name: 'entry_reason', type: 'TEXT', nullable: true, primary_key: false },
          { name: 'exit_reason', type: 'TEXT', nullable: true, primary_key: false },
          { name: 'lessons_learned', type: 'TEXT', nullable: true, primary_key: false },
          { name: 'emotion_score', type: 'INTEGER', nullable: true, primary_key: false },
          { name: 'created_at', type: 'TEXT', nullable: false, primary_key: false }
        ]
      },
      {
        name: 'broker_registry',
        row_count: 4,
        is_new: true,
        size_bytes: 4096,
        last_updated: '2024-01-15T08:00:00Z',
        columns: [
          { name: 'id', type: 'INTEGER', nullable: false, primary_key: true },
          { name: 'name', type: 'TEXT', nullable: false, primary_key: false },
          { name: 'type', type: 'TEXT', nullable: false, primary_key: false },
          { name: 'server', type: 'TEXT', nullable: false, primary_key: false },
          { name: 'timezone', type: 'TEXT', nullable: true, primary_key: false },
          { name: 'spreads_json', type: 'TEXT', nullable: true, primary_key: false }
        ]
      },
      {
        name: 'house_money_state',
        row_count: 20,
        is_new: true,
        size_bytes: 8192,
        last_updated: '2024-01-20T23:50:00Z',
        columns: [
          { name: 'id', type: 'INTEGER', nullable: false, primary_key: true },
          { name: 'date', type: 'TEXT', nullable: false, primary_key: false },
          { name: 'daily_profit', type: 'REAL', nullable: false, primary_key: false },
          { name: 'house_money_amount', type: 'REAL', nullable: false, primary_key: false },
          { name: 'threshold_percent', type: 'REAL', nullable: false, primary_key: false },
          { name: 'mode', type: 'TEXT', nullable: false, primary_key: false }
        ]
      }
    ];
    applyFilters();
  }

  async function loadDatabaseStats() {
    try {
      const res = await fetch('http://localhost:8000/api/database/stats');
      if (res.ok) {
        dbStats = await res.json();
      } else {
        loadMockStats();
      }
    } catch (e) {
      console.error('Failed to load database stats:', e);
      loadMockStats();
    }
  }

  function loadMockStats() {
    const totalSize = tables.reduce((sum, t) => sum + t.size_bytes, 0);
    dbStats = {
      total_size_bytes: totalSize,
      sqlite_path: '/data/quantmind.db',
      duckdb_path: '/data/analytics.duckdb',
      table_count: tables.length
    };
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

  async function loadTableData(tableName: string, page = 1, limit = rowsPerPage) {
    isLoading = true;
    try {
      const offset = (page - 1) * limit;
      const res = await fetch(
        `http://localhost:8000/api/database/table/${tableName}?limit=${limit}&offset=${offset}`
      );
      if (res.ok) {
        tableData = await res.json();
      } else {
        loadMockTableData(tableName);
      }
    } catch (e) {
      console.error('Failed to load table data:', e);
      loadMockTableData(tableName);
    } finally {
      isLoading = false;
    }
  }

  function loadMockTableData(tableName: string) {
    // Generate mock data based on table
    const table = tables.find(t => t.name === tableName);
    if (!table) return;

    const columns = table.columns.map(c => c.name);
    const mockRows: Array<Record<string, any>> = [];

    for (let i = 0; i < Math.min(rowsPerPage, table.row_count); i++) {
      const row: Record<string, any> = {};
      table.columns.forEach(col => {
        if (col.primary_key) {
          row[col.name] = i + 1;
        } else if (col.type === 'INTEGER') {
          row[col.name] = Math.floor(Math.random() * 1000);
        } else if (col.type === 'REAL') {
          row[col.name] = Math.random() * 100;
        } else if (col.name.includes('_at') || col.name.includes('date') || col.name.includes('time')) {
          row[col.name] = new Date(Date.now() - Math.random() * 86400000).toISOString().slice(0, 19);
        } else if (col.name.includes('status')) {
          const statuses = ['active', 'pending', 'completed', 'failed'];
          row[col.name] = statuses[Math.floor(Math.random() * statuses.length)];
        } else if (col.name.includes('type')) {
          const types = ['NPRD', 'TRD', 'EA'];
          row[col.name] = types[Math.floor(Math.random() * types.length)];
        } else {
          row[col.name] = `Sample ${col.name} ${i + 1}`;
        }
      });
      mockRows.push(row);
    }

    tableData = {
      columns,
      rows: mockRows,
      row_count: table.row_count,
      execution_time_ms: Math.random() * 50
    };
  }

  async function loadTableSchema(tableName: string) {
    try {
      const res = await fetch(`http://localhost:8000/api/database/schema/${tableName}`);
      if (res.ok) {
        const data = await res.json();
        tableSchema = data.columns || [];
      } else {
        const table = tables.find(t => t.name === tableName);
        tableSchema = table?.columns || [];
      }
    } catch (e) {
      const table = tables.find(t => t.name === tableName);
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
      const res = await fetch('http://localhost:8000/api/database/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: queryInput })
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
          error: error.detail || 'Query failed'
        };
        addToHistory(queryInput);
      }
    } catch (e) {
      queryResults = {
        columns: [],
        rows: [],
        row_count: 0,
        execution_time_ms: Date.now() - startTime,
        error: 'Failed to connect to database'
      };
      addToHistory(queryInput);
    } finally {
      isQueryRunning = false;
    }
  }

  function addToHistory(query: string) {
    queryHistory = [query, ...queryHistory.filter(q => q !== query)].slice(0, 10);
    queryHistoryIndex = -1;
  }

  function navigateHistory(direction: 'up' | 'down') {
    if (queryHistory.length === 0) return;

    if (direction === 'up') {
      queryHistoryIndex = Math.min(queryHistoryIndex + 1, queryHistory.length - 1);
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
    selectedTable.columns.forEach(col => {
      if (!col.primary_key) {
        newRowData[col.name] = col.default_value ?? '';
      }
    });
    insertModalOpen = true;
  }

  async function insertRow() {
    if (!selectedTable) return;

    try {
      const res = await fetch(`http://localhost:8000/api/database/table/${selectedTable.name}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(newRowData)
      });

      if (res.ok) {
        await loadTableData(selectedTable.name, currentPage);
        insertModalOpen = false;
      } else {
        alert('Failed to insert row');
      }
    } catch (e) {
      console.error('Failed to insert row:', e);
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
      const res = await fetch(`http://localhost:8000/api/database/table/${selectedTable.name}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(editingRow)
      });

      if (res.ok) {
        await loadTableData(selectedTable.name, currentPage);
        editModalOpen = false;
      } else {
        alert('Failed to update row');
      }
    } catch (e) {
      console.error('Failed to update row:', e);
      // For development, update locally
      if (tableData) {
        const index = tableData.rows.findIndex(r => r.id === editingRow?.id);
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
      const res = await fetch(`http://localhost:8000/api/database/table/${selectedTable.name}`, {
        method: 'DELETE',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ids: Array.from(selectedRows) })
      });

      if (res.ok) {
        await loadTableData(selectedTable.name, currentPage);
        selectedRows.clear();
      } else {
        alert('Failed to delete rows');
      }
    } catch (e) {
      console.error('Failed to delete rows:', e);
      // For development, remove locally
      if (tableData) {
        tableData.rows = tableData.rows.filter(r => !selectedRows.has(String(r.id)));
        tableData.row_count -= selectedRows.size;
      }
      selectedRows.clear();
    }
  }

  async function exportTable(format: 'csv' | 'json') {
    if (!selectedTable) return;

    try {
      const res = await fetch(`http://localhost:8000/api/database/export/${selectedTable.name}?format=${format}`);
      if (res.ok) {
        const blob = await res.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${selectedTable.name}.${format === 'csv' ? 'csv' : 'json'}`;
        a.click();
        URL.revokeObjectURL(url);
      }
    } catch (e) {
      console.error('Failed to export table:', e);
      alert('Export failed');
    }
  }

  async function importTable(file: File) {
    if (!selectedTable) return;

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch(`http://localhost:8000/api/database/import/${selectedTable.name}`, {
        method: 'POST',
        body: formData
      });

      if (res.ok) {
        await loadTableData(selectedTable.name, currentPage);
        await loadTables(); // Update row counts
      } else {
        alert('Import failed');
      }
    } catch (e) {
      console.error('Failed to import table:', e);
      alert('Import failed');
    }
  }

  // ============================================================================
  // UI HELPERS
  // ============================================================================

  function applyFilters() {
    filteredTables = tables.filter(table => {
      if (tableTypeFilter === 'existing' && table.is_new) return false;
      if (tableTypeFilter === 'new' && !table.is_new) return false;
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
      selectedRows = new Set(tableData.rows.map(r => String(r.id || Object.values(r)[0])));
    }
    selectedRows = new Set(selectedRows); // Trigger reactivity
  }

  function sortTable(column: string) {
    if (sortColumn === column) {
      sortDirection = sortDirection === 'asc' ? 'desc' : 'asc';
    } else {
      sortColumn = column;
      sortDirection = 'asc';
    }

    if (!tableData) return;

    tableData.rows.sort((a, b) => {
      const aVal = a[column];
      const bVal = b[column];

      if (aVal === bVal) return 0;

      const comparison = aVal < bVal ? -1 : 1;
      return sortDirection === 'asc' ? comparison : -comparison;
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
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
  }

  function formatTimestamp(dateStr: string) {
    const date = new Date(dateStr);
    return date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  }

  function getColumnTypeColor(type: string) {
    if (type.includes('INT')) return '#3b82f6';
    if (type.includes('REAL') || type.includes('FLOAT')) return '#10b981';
    if (type.includes('TEXT')) return '#f59e0b';
    if (type.includes('BLOB')) return '#8b5cf6';
    return '#6b7280';
  }

  function previewJson(data: any) {
    jsonPreviewData = data;
    jsonPreviewModalOpen = true;
  }

  function isJsonColumn(value: any) {
    if (typeof value === 'string') {
      try {
        JSON.parse(value);
        return true;
      } catch {
        return false;
      }
    }
    return typeof value === 'object' && value !== null;
  }
</script>

<div class="database-view">
  <!-- Header -->
  <div class="db-header">
    <div class="header-left">
      <Database size={24} class="db-icon" />
      <div>
        <h2>Database Manager</h2>
        <p>SQLite + DuckDB hybrid database with 15 tables</p>
      </div>
    </div>
    <div class="header-actions">
      <button class="btn" on:click={loadTables}>
        <RefreshCw size={14} />
        <span>Refresh</span>
      </button>
      <button class="btn" on:click={loadDatabaseStats}>
        <HardDrive size={14} />
        <span>Stats</span>
      </button>
    </div>
  </div>

  <!-- Database Stats Banner -->
  {#if dbStats}
    <div class="stats-banner">
      <div class="stat-item">
        <Database size={14} />
        <span class="label">Tables:</span>
        <span class="value">{dbStats.table_count}</span>
      </div>
      <div class="stat-item">
        <HardDrive size={14} />
        <span class="label">Total Size:</span>
        <span class="value">{formatBytes(dbStats.total_size_bytes)}</span>
      </div>
      <div class="stat-item">
        <FileText size={14} />
        <span class="label">SQLite:</span>
        <span class="value path">{dbStats.sqlite_path}</span>
      </div>
      <div class="stat-item">
        <Code size={14} />
        <span class="label">DuckDB:</span>
        <span class="value path">{dbStats.duckdb_path}</span>
      </div>
    </div>
  {/if}

  <!-- Main Content -->
  <div class="db-content">
    <!-- Table List Sidebar -->
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
          on:input={applyFilters}
        />
      </div>

      <!-- Filter -->
      <div class="sidebar-filter">
        <Filter size={12} />
        <select bind:value={tableTypeFilter} on:change={applyFilters}>
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
              on:click={() => selectTable(table)}
              on:keydown={(e) => e.key === 'Enter' && selectTable(table)}
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
                  <span class="row-count">{table.row_count.toLocaleString()} rows</span>
                  <span class="table-size">{formatBytes(table.size_bytes)}</span>
                </div>
              </div>
              <div class="table-indicator" class:new-table={table.is_new} title={table.is_new ? 'New table' : 'Existing table'}></div>
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

    <!-- Data Grid Area -->
    <div class="data-grid-area">
      {#if selectedTable}
        <!-- Table Header -->
        <div class="grid-header">
          <div class="header-left">
            <Table size={16} />
            <div>
              <h3>{selectedTable.name}</h3>
              <span class="table-badge" class:new-table={selectedTable.is_new}>
                {selectedTable.is_new ? 'NEW' : 'EXISTING'}
              </span>
            </div>
          </div>
          <div class="header-actions">
            <button class="btn" on:click={() => showTableInfo = !showTableInfo} class:active={showTableInfo}>
              <FileText size={14} />
              <span>Schema</span>
            </button>
            <button class="btn" on:click={() => exportTable('csv')}>
              <FileText size={14} />
              <span>Export CSV</span>
            </button>
            <button class="btn" on:click={() => exportTable('json')}>
              <FileCode size={14} />
              <span>Export JSON</span>
            </button>
            <label class="btn">
              <Upload size={14} />
              <span>Import</span>
              <input type="file" accept=".csv,.json" on:change={(e) => {
                const input = e.currentTarget;
                const file = input.files?.[0];
                if (file) importTable(file);
              }} hidden />
            </label>
            <button class="btn" on:click={openInsertModal}>
              <Plus size={14} />
              <span>Insert</span>
            </button>
            <button
              class="btn danger"
              on:click={deleteSelectedRows}
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
                    <span class="column-type" style="color: {getColumnTypeColor(column.type)}">
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
              <span class="meta-item">Last updated: {formatTimestamp(selectedTable.last_updated)}</span>
              <span class="meta-item">Size: {formatBytes(selectedTable.size_bytes)}</span>
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
                  checked={selectedRows.size === tableData.rows.length && tableData.rows.length > 0}
                  on:change={toggleAllRows}
                />
              </div>
              {#each tableData.columns as column}
                <div
                  class="grid-cell header-cell {sortColumn === column ? 'sorted' : ''}"
                  on:click={() => sortTable(column)}
                  on:keydown={(e) => e.key === 'Enter' && sortTable(column)}
                  role="button"
                  tabindex="0"
                  aria-label="Sort by {column}"
                >
                  <span>{column}</span>
                  {#if sortColumn === column}
                    <span class:rotate={sortDirection === 'desc'}>
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
                <div class="grid-row" class:selected={selectedRows.has(String(row.id || Object.values(row)[0]))}>
                  <div class="grid-cell checkbox-cell">
                    <input
                      type="checkbox"
                      checked={selectedRows.has(String(row.id || Object.values(row)[0]))}
                      on:change={() => toggleRowSelection(String(row.id || Object.values(row)[0]))}
                    />
                  </div>
                  {#each tableData.columns as column}
                    <div class="grid-cell data-cell">
                      {#if isJsonColumn(row[column])}
                        <span class="json-link" on:click={() => previewJson(row[column])} on:keydown={(e) => e.key === 'Enter' && previewJson(row[column])} role="button" tabindex="0" aria-label="View JSON data">
                          <Code size={10} />
                          <span>View JSON</span>
                        </span>
                      {:else}
                        <span class="cell-value" title={row[column]}>
                          {row[column] !== null && row[column] !== undefined ? String(row[column]).slice(0, 50) : '<NULL>'}
                        </span>
                      {/if}
                    </div>
                  {/each}
                  <div class="grid-cell actions-cell">
                    <button class="icon-btn" on:click={() => openEditModal(row)} title="Edit row">
                      <Edit size={12} />
                    </button>
                    <button
                      class="icon-btn danger"
                      on:click={() => {
                        if (confirm('Delete this row?')) {
                          selectedRows = new Set([String(row.id || Object.values(row)[0])]);
                          deleteSelectedRows();
                        }
                      }}
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
                <span>Showing {(currentPage - 1) * rowsPerPage + 1} to {Math.min(currentPage * rowsPerPage, tableData.row_count)} of {tableData.row_count} rows</span>
              </div>
              <div class="pagination-controls">
                <select bind:value={rowsPerPage} on:change={() => selectedTable && loadTableData(selectedTable.name, 1)}>
                  <option value="25">25 per page</option>
                  <option value="50">50 per page</option>
                  <option value="100">100 per page</option>
                </select>
                <button class="page-btn" on:click={() => {
                  if (currentPage > 1) {
                    currentPage--;
                    selectedTable && loadTableData(selectedTable.name, currentPage);
                  }
                }} disabled={currentPage === 1}>
                  Previous
                </button>
                <span class="page-info">Page {currentPage} of {getTotalPages()}</span>
                <button class="page-btn" on:click={() => {
                  if (currentPage < getTotalPages()) {
                    currentPage++;
                    selectedTable && loadTableData(selectedTable.name, currentPage);
                  }
                }} disabled={currentPage >= getTotalPages()}>
                  Next
                </button>
              </div>
            </div>
          {:else}
            <div class="empty-state">
              <Table size={48} />
              <p>No data in this table</p>
              <button class="btn primary" on:click={openInsertModal}>
                <Plus size={14} />
                <span>Insert First Row</span>
              </button>
            </div>
          {/if}
        </div>
      {:else}
        <div class="empty-state select-table">
          <Database size={48} />
          <h3>Select a Table</h3>
          <p>Choose a table from the sidebar to view and edit its data</p>
        </div>
      {/if}
    </div>

    <!-- Query Editor Panel -->
    <div class="query-panel">
      <div class="query-header">
        <Terminal size={14} />
        <h3>SQL Query Editor</h3>
      </div>

      <div class="query-editor">
        <textarea
          placeholder="Enter SQL query...
Examples:
  SELECT * FROM prop_firm_accounts LIMIT 10
  SELECT COUNT(*) FROM trade_proposals WHERE status = 'active'
  SELECT strategy_name, AVG(profit) FROM crypto_trades GROUP BY strategy_name"
          bind:value={queryInput}
          on:keydown={(e) => {
            if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
              e.preventDefault();
              executeQuery();
            }
            if (e.key === 'ArrowUp') {
              e.preventDefault();
              navigateHistory('up');
            }
            if (e.key === 'ArrowDown') {
              e.preventDefault();
              navigateHistory('down');
            }
          }}
        ></textarea>
      </div>

      <div class="query-actions">
        <div class="query-hint">
          <span>Ctrl+Enter to run</span>
          <span>Use ↑/↓ for history</span>
        </div>
        <button class="btn primary" on:click={executeQuery} disabled={isQueryRunning || !queryInput.trim()}>
          {#if isQueryRunning}
            <RefreshCw size={14} class="spin" />
            <span>Running...</span>
          {:else}
            <Play size={14} />
            <span>Run Query</span>
          {/if}
        </button>
      </div>

      <!-- Query History -->
      {#if queryHistory.length > 0}
        <div class="query-history">
          <h4>Recent Queries</h4>
          <div class="history-list">
            {#each queryHistory.slice(0, 5) as query}
              <div class="history-item" on:click={() => queryInput = query} on:keydown={(e) => e.key === 'Enter' && (queryInput = query)} role="button" tabindex="0" aria-label="Use query: {query.slice(0, 30)}...">
                <Code size={10} />
                <span class="query-text">{query.slice(0, 60)}...</span>
              </div>
            {/each}
          </div>
        </div>
      {/if}

      <!-- Query Results -->
      {#if queryResults}
        <div class="query-results">
          <div class="results-header">
            <h4>Results</h4>
            {#if queryResults.error}
              <div class="error-message">
                <AlertCircle size={12} />
                <span>{queryResults.error}</span>
              </div>
            {:else}
              <span class="results-stats">
                {queryResults.row_count} rows in {queryResults.execution_time_ms.toFixed(2)}ms
              </span>
            {/if}
          </div>

          {#if queryResults.rows.length > 0}
            <div class="results-grid">
              <div class="results-header-row">
                {#each queryResults.columns as column}
                  <div class="results-cell header">{column}</div>
                {/each}
              </div>
              {#each queryResults.rows.slice(0, 50) as row}
                <div class="results-row">
                  {#each queryResults.columns as column}
                    <div class="results-cell">
                      {#if isJsonColumn(row[column])}
                        <span class="json-link" on:click={() => previewJson(row[column])} on:keydown={(e) => e.key === 'Enter' && previewJson(row[column])} role="button" tabindex="0" aria-label="View JSON data">
                          <Code size={8} />
                          <span>JSON</span>
                        </span>
                      {:else}
                        <span>{row[column] !== null ? String(row[column]) : '<NULL>'}</span>
                      {/if}
                    </div>
                  {/each}
                </div>
              {/each}
            </div>
          {:else if !queryResults.error}
            <div class="empty-results">
              <FileText size={24} />
              <p>Query executed successfully but returned no rows</p>
            </div>
          {/if}
        </div>
      {/if}
    </div>
  </div>
</div>

<!-- Insert Modal -->
{#if insertModalOpen && selectedTable}
  <div class="modal-overlay" on:click|self={() => insertModalOpen = false}>
    <div class="modal">
      <div class="modal-header">
        <div>
          <h3>Insert Row</h3>
          <p class="modal-subtitle">{selectedTable.name}</p>
        </div>
        <button class="icon-btn" on:click={() => insertModalOpen = false}>
          <X size={18} />
        </button>
      </div>

      <div class="modal-content">
        {#each selectedTable.columns as column}
          {#if !column.primary_key}
            <div class="form-group">
              <label for="insert-{column.name}">
                {column.name}
                <span class="type-badge" style="color: {getColumnTypeColor(column.type)}">{column.type}</span>
              </label>
              {#if column.type.includes('INT') || column.type.includes('REAL')}
                <input type="number" id="insert-{column.name}" bind:value={newRowData[column.name]} placeholder={column.name} />
              {:else if column.type.includes('TEXT')}
                <textarea id="insert-{column.name}" bind:value={newRowData[column.name]} placeholder={column.name} rows="2"></textarea>
              {:else}
                <input type="text" id="insert-{column.name}" bind:value={newRowData[column.name]} placeholder={column.name} />
              {/if}
            </div>
          {/if}
        {/each}

        <div class="modal-actions">
          <button class="btn" on:click={() => insertModalOpen = false}>Cancel</button>
          <button class="btn primary" on:click={insertRow}>
            <Plus size={14} />
            <span>Insert Row</span>
          </button>
        </div>
      </div>
    </div>
  </div>
{/if}

<!-- Edit Modal -->
{#if editModalOpen && editingRow}
  <div class="modal-overlay" on:click|self={() => editModalOpen = false}>
    <div class="modal">
      <div class="modal-header">
        <div>
          <h3>Edit Row</h3>
          <p class="modal-subtitle">ID: {editingRow.id || editingRow[Object.keys(editingRow)[0]]}</p>
        </div>
        <button class="icon-btn" on:click={() => editModalOpen = false}>
          <X size={18} />
        </button>
      </div>

      <div class="modal-content">
        {#if selectedTable}
          {#each selectedTable.columns as column}
            <div class="form-group">
              <label for="edit-{column.name}">
                {column.name}
                <span class="type-badge" style="color: {getColumnTypeColor(column.type)}">{column.type}</span>
              </label>
              {#if column.type.includes('INT') || column.type.includes('REAL')}
                <input type="number" id="edit-{column.name}" bind:value={editingRow[column.name]} />
              {:else if column.type.includes('TEXT')}
                <textarea id="edit-{column.name}" bind:value={editingRow[column.name]} rows="2"></textarea>
              {:else}
                <input type="text" id="edit-{column.name}" bind:value={editingRow[column.name]} />
              {/if}
            </div>
          {/each}
        {/if}

        <div class="modal-actions">
          <button class="btn" on:click={() => editModalOpen = false}>Cancel</button>
          <button class="btn primary" on:click={updateRow}>
            <Check size={14} />
            <span>Save Changes</span>
          </button>
        </div>
      </div>
    </div>
  </div>
{/if}

<!-- JSON Preview Modal -->
{#if jsonPreviewModalOpen}
  <div class="modal-overlay" on:click|self={() => jsonPreviewModalOpen = false}>
    <div class="modal large">
      <div class="modal-header">
        <div>
          <h3>JSON Preview</h3>
          <p class="modal-subtitle">View complex data structure</p>
        </div>
        <button class="icon-btn" on:click={() => jsonPreviewModalOpen = false}>
          <X size={18} />
        </button>
      </div>

      <div class="modal-content">
        <pre class="json-preview">{JSON.stringify(jsonPreviewData, null, 2)}</pre>
      </div>
    </div>
  </div>
{/if}

<style>
  .database-view {
    display: flex;
    flex-direction: column;
    height: 100%;
    background: var(--bg-primary);
    overflow: hidden;
  }

  /* Header */
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

  .db-icon {
    color: var(--accent-primary);
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

  .btn.active {
    background: var(--accent-primary);
    border-color: var(--accent-primary);
    color: var(--bg-primary);
  }

  .btn.primary {
    background: var(--accent-primary);
    border-color: var(--accent-primary);
    color: var(--bg-primary);
  }

  .btn.primary:hover {
    opacity: 0.9;
  }

  .btn.danger:hover:not(.disabled) {
    background: rgba(239, 68, 68, 0.2);
    border-color: #ef4444;
    color: #ef4444;
  }

  .btn.disabled {
    opacity: 0.4;
    cursor: not-allowed;
  }

  /* Stats Banner */
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
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
  }

  /* Main Content */
  .db-content {
    flex: 1;
    display: flex;
    overflow: hidden;
  }

  /* Table Sidebar */
  .table-sidebar {
    width: 280px;
    background: var(--bg-secondary);
    border-right: 1px solid var(--border-subtle);
    display: flex;
    flex-direction: column;
  }

  .sidebar-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 16px;
    border-bottom: 1px solid var(--border-subtle);
  }

  .sidebar-header h3 {
    margin: 0;
    font-size: 14px;
    color: var(--text-primary);
  }

  .table-count {
    padding: 2px 8px;
    background: var(--bg-tertiary);
    border-radius: 10px;
    font-size: 11px;
    color: var(--text-muted);
  }

  .sidebar-search,
  .sidebar-filter {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 12px 16px;
    border-bottom: 1px solid var(--border-subtle);
  }

  .sidebar-search input,
  .sidebar-filter select {
    flex: 1;
    background: var(--bg-tertiary);
    border: none;
    border-radius: 4px;
    padding: 6px 10px;
    font-size: 12px;
    color: var(--text-primary);
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
    border-bottom: 1px solid var(--border-subtle);
  }

  .table-item:hover {
    background: var(--bg-tertiary);
  }

  .table-item.selected {
    background: var(--accent-primary);
    color: var(--bg-primary);
  }

  .table-item.selected .table-name,
  .table-item.selected .table-meta {
    color: var(--bg-primary);
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
    color: var(--text-primary);
    margin-bottom: 4px;
  }

  .table-meta {
    display: flex;
    gap: 8px;
    font-size: 11px;
    color: var(--text-muted);
  }

  .table-indicator {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--text-muted);
    flex-shrink: 0;
  }

  .table-indicator.new-table {
    background: var(--accent-success);
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
    font-family: 'JetBrains Mono', monospace;
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
    font-family: 'JetBrains Mono', monospace;
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
    font-family: 'JetBrains Mono', monospace;
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
    font-family: 'JetBrains Mono', monospace;
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
    font-family: 'JetBrains Mono', monospace;
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

  .spin {
    animation: spin 1s linear infinite;
  }

  @keyframes spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
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
    font-family: 'JetBrains Mono', monospace;
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
    font-family: 'JetBrains Mono', monospace;
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
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    line-height: 1.6;
    color: var(--text-primary);
    overflow: auto;
    max-height: 500px;
  }
</style>
