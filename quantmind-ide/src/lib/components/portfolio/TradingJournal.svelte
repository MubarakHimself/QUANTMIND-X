<script lang="ts">
  /**
   * Trading Journal Component
   *
   * Portfolio canvas sub-page for viewing trade logs with annotation capability.
   * Story 9-5: Trading Journal Component
   *
   * Features:
   * - Filterable trade log (entry time, exit time, symbol, direction, P&L, session, hold duration, EA name)
   * - Trade detail view with annotation
   * - CSV export
   */
  import { onMount } from 'svelte';
  import { X, Download, Search, Filter, ChevronDown, ChevronUp, Clock, Tag, TrendingUp, TrendingDown } from 'lucide-svelte';

  // Props
  let { onClose }: { onClose: () => void } = $props();

  // State
  let trades = $state<any[]>([]);
  let loading = $state(true);
  let selectedTrade = $state<any>(null);
  let showDetail = $state(false);

  // Filters
  let filters = $state({
    symbol: '',
    direction: '',
    session: '',
    eaName: ''
  });

  // Sort state
  let sortColumn = $state('entryTime');
  let sortAsc = $state(false);

  // Annotation
  let editingNote = $state(false);
  let noteText = $state('');
  let savingNote = $state(false);

  onMount(async () => {
    await loadTrades();
  });

  async function loadTrades() {
    loading = true;
    try {
      const params = new URLSearchParams();
      if (filters.symbol) params.append('symbol', filters.symbol);
      if (filters.session && filters.session !== 'all') params.append('mode', filters.session);

      const response = await fetch(`/api/journal/trades?${params.toString()}`);
      if (response.ok) {
        trades = await response.json();
      }
    } catch (e) {
      console.error('Failed to load trades:', e);
    } finally {
      loading = false;
    }
  }

  function applyFilters() {
    loadTrades();
  }

  function sortBy(column: string) {
    if (sortColumn === column) {
      sortAsc = !sortAsc;
    } else {
      sortColumn = column;
      sortAsc = true;
    }
  }

  function getSortedTrades() {
    return [...trades].sort((a, b) => {
      const aVal = a[sortColumn] ?? '';
      const bVal = b[sortColumn] ?? '';
      if (sortAsc) {
        return aVal > bVal ? 1 : -1;
      }
      return aVal < bVal ? 1 : -1;
    });
  }

  function openTradeDetail(trade: any) {
    selectedTrade = trade;
    noteText = trade.note || '';
    showDetail = true;
    editingNote = false;
  }

  function closeDetail() {
    showDetail = false;
    selectedTrade = null;
    editingNote = false;
  }

  async function saveNote() {
    if (!selectedTrade) return;
    savingNote = true;
    try {
      const response = await fetch(`/api/journal/trades/${selectedTrade.id}/annotation`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ note: noteText })
      });
      if (response.ok) {
        const result = await response.json();
        selectedTrade.note = result.note;
        selectedTrade.annotatedAt = result.annotated_at;
        // Update in the main list too
        const idx = trades.findIndex(t => t.id === selectedTrade.id);
        if (idx >= 0) {
          trades[idx].note = result.note;
          trades[idx].annotatedAt = result.annotated_at;
        }
        editingNote = false;
      }
    } catch (e) {
      console.error('Failed to save note:', e);
    } finally {
      savingNote = false;
    }
  }

  async function exportCSV() {
    const params = new URLSearchParams();
    if (filters.symbol) params.append('symbol', filters.symbol);
    if (filters.direction) params.append('direction', filters.direction);
    if (filters.session && filters.session !== 'all') params.append('session', filters.session);
    if (filters.eaName) params.append('ea_name', filters.eaName);

    window.open(`/api/journal/trades/export/csv?${params.toString()}`, '_blank');
  }

  function formatTime(iso: string) {
    if (!iso) return '-';
    try {
      const date = new Date(iso);
      return date.toLocaleString('en-US', {
        timeZone: 'UTC',
        year: 'numeric',
        month: '2-digit',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit'
      }) + ' UTC';
    } catch {
      return iso;
    }
  }

  function formatPnL(value: number) {
    const sign = value >= 0 ? '+' : '';
    return sign + value.toFixed(2);
  }
</script>

<div class="trading-journal">
  <header class="journal-header">
    <div class="header-left">
      <h2>Trading Journal</h2>
      <span class="trade-count">{trades.length} trades</span>
    </div>
    <div class="header-actions">
      <button class="export-btn" onclick={exportCSV}>
        <Download size={14} />
        <span>Export CSV</span>
      </button>
      <button class="close-btn" onclick={onClose}>
        <X size={18} />
      </button>
    </div>
  </header>

  <div class="filters-bar">
    <div class="filter-group">
      <Search size={14} />
      <input
        type="text"
        placeholder="Symbol..."
        bind:value={filters.symbol}
        onchange={applyFilters}
      />
    </div>
    <div class="filter-group">
      <Filter size={14} />
      <select bind:value={filters.direction} onchange={applyFilters}>
        <option value="">Direction</option>
        <option value="BUY">Buy</option>
        <option value="SELL">Sell</option>
      </select>
    </div>
    <div class="filter-group">
      <select bind:value={filters.session} onchange={applyFilters}>
        <option value="">Session</option>
        <option value="all">All</option>
        <option value="demo">Demo</option>
        <option value="live">Live</option>
      </select>
    </div>
    <div class="filter-group">
      <input
        type="text"
        placeholder="EA Name..."
        bind:value={filters.eaName}
        onchange={applyFilters}
      />
    </div>
  </div>

  <div class="trade-table-container">
    {#if loading}
      <div class="loading">Loading trades...</div>
    {:else if trades.length === 0}
      <div class="empty">No trades found</div>
    {:else}
      <table class="trade-table">
        <thead>
          <tr>
            <th class="sortable" onclick={() => sortBy('entryTime')}>
              Entry Time {sortColumn === 'entryTime' ? (sortAsc ? '↑' : '↓') : ''}
            </th>
            <th class="sortable" onclick={() => sortBy('symbol')}>
              Symbol {sortColumn === 'symbol' ? (sortAsc ? '↑' : '↓') : ''}
            </th>
            <th>Direction</th>
            <th class="sortable" onclick={() => sortBy('pnl')}>
              P&L {sortColumn === 'pnl' ? (sortAsc ? '↑' : '↓') : ''}
            </th>
            <th>Session</th>
            <th>Hold</th>
            <th>EA Name</th>
            <th>Note</th>
          </tr>
        </thead>
        <tbody>
          {#each getSortedTrades() as trade (trade.id)}
            <tr onclick={() => openTradeDetail(trade)} class="trade-row">
              <td>{formatTime(trade.entryTime)}</td>
              <td class="symbol">{trade.symbol}</td>
              <td class="direction" class:buy={trade.direction === 'BUY'} class:sell={trade.direction === 'SELL'}>
                {trade.direction === 'BUY' ? '↑' : '↓'}
              </td>
              <td class="pnl" class:positive={trade.pnl > 0} class:negative={trade.pnl < 0}>
                {formatPnL(trade.pnl)}
              </td>
              <td class="session">{trade.session}</td>
              <td>{trade.holdDuration ? `${trade.holdDuration}m` : '-'}</td>
              <td class="ea-name">{trade.eaName}</td>
              <td class="has-note">
                {#if trade.note}
                  <Tag size={12} />
                {:else}
                  -
                {/if}
              </td>
            </tr>
          {/each}
        </tbody>
      </table>
    {/if}
  </div>

  <!-- Trade Detail Drawer -->
  {#if showDetail && selectedTrade}
    <div class="detail-overlay" onclick={closeDetail}></div>
    <div class="detail-drawer">
      <div class="detail-header">
        <h3>Trade Details</h3>
        <button class="close-btn" onclick={closeDetail}>
          <X size={18} />
        </button>
      </div>

      <div class="detail-content">
        <div class="detail-row">
          <span class="label">Symbol</span>
          <span class="value symbol">{selectedTrade.symbol}</span>
        </div>
        <div class="detail-row">
          <span class="label">Direction</span>
          <span class="value direction" class:buy={selectedTrade.direction === 'BUY'} class:sell={selectedTrade.direction === 'SELL'}>
            {selectedTrade.direction}
          </span>
        </div>
        <div class="detail-row">
          <span class="label">Entry Price</span>
          <span class="value">{selectedTrade.entryPrice || '-'}</span>
        </div>
        <div class="detail-row">
          <span class="label">Exit Price</span>
          <span class="value">{selectedTrade.exitPrice || '-'}</span>
        </div>
        <div class="detail-row">
          <span class="label">Spread at Entry</span>
          <span class="value">{selectedTrade.spreadAtEntry ? `${selectedTrade.spreadAtEntry} pips` : '-'}</span>
        </div>
        <div class="detail-row">
          <span class="label">Slippage</span>
          <span class="value">{selectedTrade.slippage || '0'} pips</span>
        </div>
        <div class="detail-row">
          <span class="label">Strategy Version</span>
          <span class="value">{selectedTrade.strategyVersion || 'N/A'}</span>
        </div>
        <div class="detail-row">
          <span class="label">P&L</span>
          <span class="value pnl" class:positive={selectedTrade.pnl > 0} class:negative={selectedTrade.pnl < 0}>
            {formatPnL(selectedTrade.pnl)}
          </span>
        </div>
        <div class="detail-row">
          <span class="label">Hold Duration</span>
          <span class="value">{selectedTrade.holdDuration ? `${selectedTrade.holdDuration} minutes` : '-'}</span>
        </div>

        <div class="annotation-section">
          <div class="annotation-header">
            <h4>Annotation</h4>
            {#if !editingNote}
              <button class="edit-btn" onclick={() => editingNote = true}>
                {selectedTrade.note ? 'Edit Note' : 'Add Note'}
              </button>
            {/if}
          </div>

          {#if editingNote}
            <div class="annotation-editor">
              <textarea
                bind:value={noteText}
                placeholder="Add your trade notes here..."
                rows="4"
              ></textarea>
              <div class="annotation-actions">
                <button class="cancel-btn" onclick={() => editingNote = false}>Cancel</button>
                <button class="save-btn" onclick={saveNote} disabled={savingNote}>
                  {savingNote ? 'Saving...' : 'Save'}
                </button>
              </div>
            </div>
          {:else if selectedTrade.note}
            <div class="annotation-display">
              <p>{selectedTrade.note}</p>
              {#if selectedTrade.annotatedAt}
                <span class="annotated-time">Last edited: {formatTime(selectedTrade.annotatedAt)}</span>
              {/if}
            </div>
          {:else}
            <p class="no-annotation">No annotation yet. Click "Add Note" to add one.</p>
          {/if}
        </div>
      </div>
    </div>
  {/if}
</div>

<style>
  .trading-journal {
    display: flex;
    flex-direction: column;
    height: 100%;
    background: rgba(10, 15, 26, 0.98);
    backdrop-filter: blur(16px);
    position: relative;
  }

  .journal-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 16px 20px;
    border-bottom: 1px solid rgba(0, 212, 255, 0.1);
    background: rgba(8, 13, 20, 0.6);
  }

  .header-left {
    display: flex;
    align-items: center;
    gap: 12px;
  }

  .header-left h2 {
    font-family: 'JetBrains Mono', monospace;
    font-size: 16px;
    font-weight: 600;
    color: #00d4ff;
    margin: 0;
  }

  .trade-count {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    color: rgba(255, 255, 255, 0.4);
    padding: 2px 8px;
    background: rgba(0, 212, 255, 0.1);
    border-radius: 4px;
  }

  .header-actions {
    display: flex;
    gap: 8px;
  }

  .export-btn {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 6px 12px;
    background: rgba(34, 197, 94, 0.1);
    border: 1px solid rgba(34, 197, 94, 0.3);
    border-radius: 6px;
    color: #22c55e;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    cursor: pointer;
    transition: all 0.2s ease;
  }

  .export-btn:hover {
    background: rgba(34, 197, 94, 0.2);
  }

  .close-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 32px;
    height: 32px;
    background: transparent;
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 6px;
    color: rgba(255, 255, 255, 0.5);
    cursor: pointer;
    transition: all 0.2s ease;
  }

  .close-btn:hover {
    background: rgba(239, 68, 68, 0.1);
    border-color: rgba(239, 68, 68, 0.3);
    color: #ef4444;
  }

  .filters-bar {
    display: flex;
    gap: 12px;
    padding: 12px 20px;
    background: rgba(8, 13, 20, 0.4);
    border-bottom: 1px solid rgba(0, 212, 255, 0.05);
  }

  .filter-group {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 6px 12px;
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 6px;
    color: rgba(255, 255, 255, 0.5);
  }

  .filter-group input,
  .filter-group select {
    background: transparent;
    border: none;
    color: rgba(255, 255, 255, 0.8);
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    outline: none;
  }

  .filter-group input::placeholder {
    color: rgba(255, 255, 255, 0.3);
  }

  .filter-group select {
    cursor: pointer;
  }

  .filter-group select option {
    background: #0a0f1a;
  }

  .trade-table-container {
    flex: 1;
    overflow: auto;
    padding: 0 20px 20px;
  }

  .loading,
  .empty {
    display: flex;
    align-items: center;
    justify-content: center;
    height: 200px;
    font-family: 'JetBrains Mono', monospace;
    color: rgba(255, 255, 255, 0.4);
  }

  .trade-table {
    width: 100%;
    border-collapse: collapse;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
  }

  .trade-table th {
    text-align: left;
    padding: 12px 16px;
    color: rgba(255, 255, 255, 0.5);
    font-weight: 500;
    border-bottom: 1px solid rgba(0, 212, 255, 0.1);
    position: sticky;
    top: 0;
    background: rgba(10, 15, 26, 0.98);
    z-index: 1;
  }

  .trade-table th.sortable {
    cursor: pointer;
    user-select: none;
  }

  .trade-table th.sortable:hover {
    color: rgba(255, 255, 255, 0.8);
  }

  .trade-table td {
    padding: 10px 16px;
    color: rgba(255, 255, 255, 0.7);
    border-bottom: 1px solid rgba(255, 255, 255, 0.05);
  }

  .trade-row {
    cursor: pointer;
    transition: background 0.15s ease;
  }

  .trade-row:hover {
    background: rgba(0, 212, 255, 0.05);
  }

  .symbol {
    color: #00d4ff;
    font-weight: 500;
  }

  .direction.buy {
    color: #22c55e;
  }

  .direction.sell {
    color: #ef4444;
  }

  .pnl.positive {
    color: #22c55e;
  }

  .pnl.negative {
    color: #ef4444;
  }

  .session {
    text-transform: capitalize;
  }

  .ea-name {
    color: rgba(255, 255, 255, 0.6);
  }

  .has-note {
    color: #f59e0b;
  }

  /* Detail Drawer */
  .detail-overlay {
    position: absolute;
    inset: 0;
    background: rgba(0, 0, 0, 0.5);
    z-index: 10;
  }

  .detail-drawer {
    position: absolute;
    right: 0;
    top: 0;
    bottom: 0;
    width: 400px;
    background: rgba(10, 15, 26, 0.98);
    backdrop-filter: blur(20px);
    border-left: 1px solid rgba(0, 212, 255, 0.1);
    z-index: 20;
    display: flex;
    flex-direction: column;
    animation: slideIn 0.2s ease;
  }

  @keyframes slideIn {
    from {
      transform: translateX(100%);
    }
    to {
      transform: translateX(0);
    }
  }

  .detail-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 16px 20px;
    border-bottom: 1px solid rgba(0, 212, 255, 0.1);
  }

  .detail-header h3 {
    font-family: 'JetBrains Mono', monospace;
    font-size: 14px;
    font-weight: 600;
    color: #00d4ff;
    margin: 0;
  }

  .detail-content {
    flex: 1;
    overflow: auto;
    padding: 20px;
  }

  .detail-row {
    display: flex;
    justify-content: space-between;
    padding: 8px 0;
    border-bottom: 1px solid rgba(255, 255, 255, 0.05);
  }

  .detail-row .label {
    color: rgba(255, 255, 255, 0.5);
    font-size: 12px;
  }

  .detail-row .value {
    color: rgba(255, 255, 255, 0.9);
    font-size: 12px;
  }

  .detail-row .value.symbol {
    color: #00d4ff;
    font-weight: 500;
  }

  .detail-row .value.direction.buy {
    color: #22c55e;
  }

  .detail-row .value.direction.sell {
    color: #ef4444;
  }

  .detail-row .value.pnl.positive {
    color: #22c55e;
  }

  .detail-row .value.pnl.negative {
    color: #ef4444;
  }

  .annotation-section {
    margin-top: 24px;
    padding-top: 16px;
    border-top: 1px solid rgba(0, 212, 255, 0.1);
  }

  .annotation-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
  }

  .annotation-header h4 {
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    font-weight: 600;
    color: #f59e0b;
    margin: 0;
  }

  .edit-btn {
    padding: 4px 8px;
    background: rgba(245, 158, 11, 0.1);
    border: 1px solid rgba(245, 158, 11, 0.3);
    border-radius: 4px;
    color: #f59e0b;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    cursor: pointer;
    transition: all 0.2s ease;
  }

  .edit-btn:hover {
    background: rgba(245, 158, 11, 0.2);
  }

  .annotation-editor textarea {
    width: 100%;
    padding: 12px;
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(245, 158, 11, 0.2);
    border-radius: 6px;
    color: rgba(255, 255, 255, 0.9);
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    resize: vertical;
    outline: none;
  }

  .annotation-editor textarea:focus {
    border-color: rgba(245, 158, 11, 0.5);
  }

  .annotation-actions {
    display: flex;
    gap: 8px;
    margin-top: 8px;
  }

  .cancel-btn {
    padding: 6px 12px;
    background: transparent;
    border: 1px solid rgba(255, 255, 255, 0.2);
    border-radius: 4px;
    color: rgba(255, 255, 255, 0.6);
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    cursor: pointer;
  }

  .save-btn {
    padding: 6px 12px;
    background: rgba(245, 158, 11, 0.2);
    border: 1px solid rgba(245, 158, 11, 0.4);
    border-radius: 4px;
    color: #f59e0b;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    cursor: pointer;
  }

  .save-btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .annotation-display {
    padding: 12px;
    background: rgba(245, 158, 11, 0.05);
    border: 1px solid rgba(245, 158, 11, 0.1);
    border-radius: 6px;
  }

  .annotation-display p {
    margin: 0;
    color: rgba(255, 255, 255, 0.9);
    font-size: 12px;
    line-height: 1.5;
  }

  .annotated-time {
    display: block;
    margin-top: 8px;
    font-size: 10px;
    color: rgba(255, 255, 255, 0.4);
  }

  .no-annotation {
    color: rgba(255, 255, 255, 0.4);
    font-size: 12px;
    font-style: italic;
  }
</style>