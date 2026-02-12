<script lang="ts">
  import { createEventDispatcher, onMount } from 'svelte';
  import {
    BookOpen, Filter, Search, Download, RefreshCw, Eye, ChevronRight,
    ChevronDown, ChevronUp, X, TrendingUp, TrendingDown, Activity, Shield, DollarSign,
    Clock, Calendar, Target, Award, AlertTriangle, CheckCircle, XCircle,
    Zap, Layers, BarChart3, FileText, Globe, Compass, ExternalLink,
    Tag, Package, Settings as SettingsIcon, Maximize2, Minimize2
  } from 'lucide-svelte';

  const dispatch = createEventDispatcher();

  // Trade Journal Data
  let trades: Array<any> = [];
  let filteredTrades: Array<any> = [];
  let selectedTrade: any = null;

  // Filters
  let filters = {
    symbol: 'all',
    status: 'all',
    dateFrom: null as Date | null,
    dateTo: null as Date | null,
    minProfit: null as number | null,
    maxProfit: null as number | null
  };

  let searchQuery = '';

  // View state
  let expandedTrade: string | null = null;
  let detailViewOpen = false;
  let sortField = 'timestamp';
  let sortDirection = 'desc';

  // Statistics
  let stats = {
    total: 0,
    wins: 0,
    losses: 0,
    winRate: 0,
    totalProfit: 0,
    avgProfit: 0,
    largestWin: 0,
    largestLoss: 0
  };

  onMount(() => {
    loadTrades();
  });

  async function loadTrades() {
    try {
      const res = await fetch('http://localhost:8000/api/journal/trades');
      if (res.ok) {
        trades = await res.json();
        applyFilters();
        calculateStats();
      }
    } catch (e) {
      console.error('Failed to load trades:', e);
    }
  }

  function applyFilters() {
    filteredTrades = trades.filter(trade => {
      if (filters.symbol !== 'all' && trade.symbol !== filters.symbol) return false;
      if (filters.status !== 'all' && trade.status !== filters.status) return false;
      if (filters.dateFrom && new Date(trade.timestamp) < filters.dateFrom) return false;
      if (filters.dateTo && new Date(trade.timestamp) > filters.dateTo) return false;
      if (filters.minProfit && trade.profit < filters.minProfit) return false;
      if (filters.maxProfit && trade.profit > filters.maxProfit) return false;
      if (searchQuery) {
        const query = searchQuery.toLowerCase();
        return (
          trade.strategy?.toLowerCase().includes(query) ||
          trade.symbol?.toLowerCase().includes(query) ||
          trade.reason?.toLowerCase().includes(query)
        );
      }
      return true;
    });

    sortTrades();
  }

  function sortTrades() {
    filteredTrades.sort((a, b) => {
      let aVal = a[sortField];
      let bVal = b[sortField];

      if (sortField === 'timestamp') {
        aVal = new Date(aVal).getTime();
        bVal = new Date(bVal).getTime();
      }

      if (sortDirection === 'asc') {
        return aVal > bVal ? 1 : -1;
      } else {
        return aVal < bVal ? 1 : -1;
      }
    });
  }

  function calculateStats() {
    const relevantTrades = filters.status === 'all'
      ? filteredTrades
      : filteredTrades.filter(t => t.status === filters.status);

    stats.total = relevantTrades.length;
    stats.wins = relevantTrades.filter(t => t.profit > 0).length;
    stats.losses = relevantTrades.filter(t => t.profit < 0).length;
    stats.winRate = stats.total > 0 ? (stats.wins / stats.total) * 100 : 0;
    stats.totalProfit = relevantTrades.reduce((sum, t) => sum + t.profit, 0);
    stats.avgProfit = stats.total > 0 ? stats.totalProfit / stats.total : 0;
    stats.largestWin = Math.max(...relevantTrades.map(t => t.profit), 0);
    stats.largestLoss = Math.min(...relevantTrades.map(t => t.profit), 0);
  }

  function toggleExpanded(tradeId: string) {
    expandedTrade = expandedTrade === tradeId ? null : tradeId;
  }

  function selectTrade(trade: any) {
    selectedTrade = trade;
    detailViewOpen = true;
  }

  function handleSort(field: string) {
    if (sortField === field) {
      sortDirection = sortDirection === 'asc' ? 'desc' : 'asc';
    } else {
      sortField = field;
      sortDirection = 'desc';
    }
    sortTrades();
  }

  function exportJournal() {
    const data = filteredTrades.map(t => ({
      timestamp: t.timestamp,
      symbol: t.symbol,
      type: t.type,
      lots: t.lots,
      openPrice: t.openPrice,
      closePrice: t.closePrice,
      profit: t.profit,
      strategy: t.strategy,
      reason: t.reason,
      regime: t.context?.regime?.quality,
      chaos: t.context?.chaos,
      kelly: t.context?.kelly?.fraction
    }));

    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `trade-journal-${new Date().toISOString().split('T')[0]}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }

  function formatProfit(profit: number) {
    const formatted = profit.toFixed(2);
    if (profit > 0) return `+$${formatted}`;
    if (profit < 0) return `-$${Math.abs(profit).toFixed(2)}`;
    return '$0.00';
  }

  function getProfitColor(profit: number) {
    if (profit > 0) return '#10b981';
    if (profit < 0) return '#ef4444';
    return '#6b7280';
  }

  function getRegimeColor(quality: number) {
    if (quality >= 0.7) return '#10b981';
    if (quality >= 0.4) return '#f59e0b';
    return '#ef4444';
  }

  function getChaosColor(chaos: number) {
    if (chaos <= 20) return '#10b981';
    if (chaos <= 40) return '#f59e0b';
    return '#ef4444';
  }

  function formatDate(dateStr: string) {
    const date = new Date(dateStr);
    return date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  }

  // Mock data for development
  if (trades.length === 0) {
    trades = [
      {
        id: 'trade-1',
        timestamp: new Date(Date.now() - 3600000).toISOString(),
        symbol: 'EURUSD',
        type: 'BUY',
        lots: 0.05,
        openPrice: 1.0876,
        closePrice: 1.0912,
        profit: 18.00,
        strategy: 'ICT Scalper',
        status: 'closed',
        reason: 'FVG setup confirmed with ATR filter',
        context: {
          regime: { quality: 0.82, trend: 'bullish', chaos: 18.5 },
          kelly: { fraction: 0.025, edge: 0.52, odds: 2.0 },
          houseMoney: { enabled: true, amount: 87.15 },
          sentiment: 'London session, low spread, FVG at key level'
        }
      },
      {
        id: 'trade-2',
        timestamp: new Date(Date.now() - 7200000).toISOString(),
        symbol: 'GBPUSD',
        type: 'SELL',
        lots: 0.04,
        openPrice: 1.2654,
        closePrice: 1.2678,
        profit: -9.60,
        strategy: 'SMC Reversal',
        status: 'closed',
        reason: 'Order block fail, volatility spike',
        context: {
          regime: { quality: 0.65, trend: 'ranging', chaos: 35.2 },
          kelly: { fraction: 0.02, edge: 0.45, odds: 1.8 },
          houseMoney: { enabled: true, amount: 77.55 },
          sentiment: 'Late session, widening spreads'
        }
      },
      {
        id: 'trade-3',
        timestamp: new Date(Date.now() - 10800000).toISOString(),
        symbol: 'USDJPY',
        type: 'BUY',
        lots: 0.03,
        openPrice: 149.85,
        closePrice: 150.12,
        profit: 8.10,
        strategy: 'ICT Scalper',
        status: 'closed',
        reason: 'Premium discount zone with RSI divergence',
        context: {
          regime: { quality: 0.78, trend: 'bullish', chaos: 22.1 },
          kelly: { fraction: 0.022, edge: 0.50, odds: 1.9 },
          houseMoney: { enabled: true, amount: 87.15 },
          sentiment: 'Asian session, clean move'
        }
      }
    ];
    filteredTrades = trades;
    calculateStats();
  }
</script>

<div class="journal-view">
  <!-- Header -->
  <div class="journal-header">
    <div class="header-left">
      <BookOpen size={24} class="journal-icon" />
      <div>
        <h2>Trade Journal</h2>
        <p>Complete trade history with context and reasoning</p>
      </div>
    </div>
    <div class="header-actions">
      <button class="btn" on:click={loadTrades}>
        <RefreshCw size={14} />
        <span>Refresh</span>
      </button>
      <button class="btn" on:click={exportJournal}>
        <Download size={14} />
        <span>Export</span>
      </button>
    </div>
  </div>

  <!-- Statistics Cards -->
  <div class="stats-row">
    <div class="stat-card">
      <div class="stat-icon">
        <Activity size={18} />
      </div>
      <div class="stat-info">
        <span class="stat-value">{stats.total}</span>
        <span class="stat-label">Total Trades</span>
      </div>
    </div>
    <div class="stat-card">
      <div class="stat-icon success">
        <CheckCircle size={18} />
      </div>
      <div class="stat-info">
        <span class="stat-value">{stats.wins}</span>
        <span class="stat-label">Wins</span>
      </div>
    </div>
    <div class="stat-card">
      <div class="stat-icon danger">
        <XCircle size={18} />
      </div>
      <div class="stat-info">
        <span class="stat-value">{stats.losses}</span>
        <span class="stat-label">Losses</span>
      </div>
    </div>
    <div class="stat-card">
      <div class="stat-icon">
        <TrendingUp size={18} />
      </div>
      <div class="stat-info">
        <span class="stat-value">{stats.winRate.toFixed(1)}%</span>
        <span class="stat-label">Win Rate</span>
      </div>
    </div>
    <div class="stat-card">
      <div class="stat-icon" class:positive={stats.totalProfit > 0}>
        <DollarSign size={18} />
      </div>
      <div class="stat-info">
        <span class="stat-value" class:positive={stats.totalProfit > 0} class:negative={stats.totalProfit < 0}>
          {formatProfit(stats.totalProfit)}
        </span>
        <span class="stat-label">Total P&L</span>
      </div>
    </div>
    <div class="stat-card">
      <div class="stat-icon">
        <BarChart3 size={18} />
      </div>
      <div class="stat-info">
        <span class="stat-value">{formatProfit(stats.avgProfit)}</span>
        <span class="stat-label">Avg Profit</span>
      </div>
    </div>
  </div>

  <!-- Filters -->
  <div class="filters-bar">
    <div class="filter-group">
      <Search size={14} />
      <input
        type="text"
        placeholder="Search by symbol, strategy, reason..."
        bind:value={searchQuery}
        on:input={applyFilters}
      />
    </div>

    <div class="filter-group">
      <Filter size={14} />
      <select bind:value={filters.symbol} on:change={applyFilters}>
        <option value="all">All Symbols</option>
        <option value="EURUSD">EURUSD</option>
        <option value="GBPUSD">GBPUSD</option>
        <option value="USDJPY">USDJPY</option>
      </select>
    </div>

    <div class="filter-group">
      <select bind:value={filters.status} on:change={applyFilters}>
        <option value="all">All Status</option>
        <option value="closed">Closed</option>
        <option value="open">Open</option>
        <option value="pending">Pending</option>
      </select>
    </div>

    <div class="filter-group date-range">
      <Calendar size={14} />
      <input
        type="date"
        bind:value={filters.dateFrom}
        on:change={applyFilters}
      />
      <span>to</span>
      <input
        type="date"
        bind:value={filters.dateTo}
        on:change={applyFilters}
      />
    </div>
  </div>

  <!-- Trades Table -->
  <div class="trades-table-container">
    <div class="table-header">
      <div class="header-cell sortable" on:click={() => handleSort('timestamp')}>
        <span>Time</span>
        <span class:active={sortField === 'timestamp' && sortDirection === 'desc'}><ChevronDown size={12} /></span>
      </div>
      <div class="header-cell sortable" on:click={() => handleSort('symbol')}>
        <span>Symbol</span>
      </div>
      <div class="header-cell">Type</div>
      <div class="header-cell">Strategy</div>
      <div class="header-cell sortable" on:click={() => handleSort('profit')}>
        <span>Profit</span>
        <span class:active={sortField === 'profit' && sortDirection === 'desc'}><ChevronDown size={12} /></span>
      </div>
      <div class="header-cell">Regime</div>
      <div class="header-cell">Kelly</div>
      <div class="header-cell"></div>
    </div>

    <div class="table-body">
      {#each filteredTrades as trade}
        <div class="table-row" class:expanded={expandedTrade === trade.id}>
          <div class="cell time">
            <div class="time-main">{formatDate(trade.timestamp)}</div>
          </div>

          <div class="cell symbol">
            <span class="symbol-badge">{trade.symbol}</span>
          </div>

          <div class="cell type">
            <span class="type-badge" class:buy={trade.type === 'BUY'} class:sell={trade.type === 'SELL'}>
              {trade.type}
            </span>
          </div>

          <div class="cell strategy">
            <span class="strategy-name">{trade.strategy}</span>
          </div>

          <div class="cell profit">
            <span class="profit-value" style="color: {getProfitColor(trade.profit)}">
              {formatProfit(trade.profit)}
            </span>
          </div>

          <div class="cell regime">
            <div class="regime-indicator" style="background: {getRegimeColor(trade.context?.regime?.quality || 0)}">
              {(trade.context?.regime?.quality || 0).toFixed(2)}
            </div>
          </div>

          <div class="cell kelly">
            <span class="kelly-value">
              {((trade.context?.kelly?.fraction || 0) * 100).toFixed(1)}%
            </span>
          </div>

          <div class="cell actions">
            <button class="icon-btn" on:click={() => toggleExpanded(trade.id)} title="Toggle details">
              {#if expandedTrade === trade.id}
                <ChevronUp size={14} />
              {:else}
                <ChevronDown size={14} />
              {/if}
            </button>
            <button class="icon-btn" on:click={() => selectTrade(trade)} title="View full details">
              <Eye size={14} />
            </button>
          </div>
        </div>

        <!-- Expanded Row -->
        {#if expandedTrade === trade.id}
          <div class="expanded-row">
            <div class="expanded-content">
              <!-- Trade Context -->
              <div class="context-section">
                <h4><Target size={14} /> Trade Context</h4>
                <div class="context-grid">
                  <div class="context-item">
                    <span class="label">Reason</span>
                    <span class="value">{trade.reason}</span>
                  </div>
                  <div class="context-item">
                    <span class="label">Sentiment</span>
                    <span class="value">{trade.context?.sentiment || 'N/A'}</span>
                  </div>
                  <div class="context-item">
                    <span class="label">Lots</span>
                    <span class="value">{trade.lots}</span>
                  </div>
                  <div class="context-item">
                    <span class="label">Entry</span>
                    <span class="value">{trade.openPrice}</span>
                  </div>
                  <div class="context-item">
                    <span class="label">Exit</span>
                    <span class="value">{trade.closePrice}</span>
                  </div>
                </div>
              </div>

              <!-- Regime Details -->
              <div class="context-section">
                <h4><Activity size={14} /> Market Regime</h4>
                <div class="regime-details">
                  <div class="regime-metric">
                    <span class="metric-label">Quality</span>
                    <div class="metric-bar">
                      <div class="metric-fill" style="width: {trade.context?.regime?.quality * 100}%; background: {getRegimeColor(trade.context?.regime?.quality || 0)}"></div>
                    </div>
                    <span class="metric-value">{((trade.context?.regime?.quality || 0) * 100).toFixed(1)}%</span>
                  </div>
                  <div class="regime-metric">
                    <span class="metric-label">Chaos</span>
                    <div class="metric-bar">
                      <div class="metric-fill" style="width: {Math.min(trade.context?.regime?.chaos || 0, 100)}%; background: {getChaosColor(trade.context?.regime?.chaos || 0)}"></div>
                    </div>
                    <span class="metric-value">{(trade.context?.regime?.chaos || 0).toFixed(1)}</span>
                  </div>
                  <div class="regime-metric">
                    <span class="metric-label">Trend</span>
                    <span class="metric-badge {trade.context?.regime?.trend}">
                      {trade.context?.regime?.trend || 'N/A'}
                    </span>
                  </div>
                </div>
              </div>

              <!-- Kelly Sizing -->
              <div class="context-section">
                <h4><Shield size={14} /> Position Sizing (Kelly)</h4>
                <div class="kelly-details">
                  <div class="kelly-item">
                    <span class="label">Fraction</span>
                    <span class="value">{((trade.context?.kelly?.fraction || 0) * 100).toFixed(2)}%</span>
                  </div>
                  <div class="kelly-item">
                    <span class="label">Edge</span>
                    <span class="value">{((trade.context?.kelly?.edge || 0) * 100).toFixed(1)}%</span>
                  </div>
                  <div class="kelly-item">
                    <span class="label">Odds</span>
                    <span class="value">{trade.context?.kelly?.odds || 0}:1</span>
                  </div>
                </div>
              </div>

              <!-- House Money -->
              {#if trade.context?.houseMoney?.enabled}
                <div class="context-section">
                  <h4><DollarSign size={14} /> House Money Effect</h4>
                  <div class="house-money-details">
                    <div class="hm-item">
                      <span class="label">House Money</span>
                      <span class="value success">${(trade.context.houseMoney.amount || 0).toFixed(2)}</span>
                    </div>
                  </div>
                </div>
              {/if}
            </div>
          </div>
        {/if}
      {/each}

      {#if filteredTrades.length === 0}
        <div class="empty-state">
          <BookOpen size={32} />
          <p>No trades found matching your filters</p>
        </div>
      {/if}
    </div>
  </div>

  <!-- Detail Modal -->
  {#if detailViewOpen && selectedTrade}
    <div class="modal-overlay" on:click|self={() => detailViewOpen = false}>
      <div class="modal">
        <div class="modal-header">
          <div>
            <h3>Trade Details</h3>
            <p class="trade-id">{selectedTrade.id}</p>
          </div>
          <button class="icon-btn" on:click={() => detailViewOpen = false}>
            <X size={18} />
          </button>
        </div>

        <div class="modal-content">
          <!-- Trade Summary -->
          <div class="detail-section">
            <h4>Trade Summary</h4>
            <div class="summary-grid">
              <div class="summary-item">
                <span class="label">Symbol</span>
                <span class="value">{selectedTrade.symbol}</span>
              </div>
              <div class="summary-item">
                <span class="label">Type</span>
                <span class="value">{selectedTrade.type}</span>
              </div>
              <div class="summary-item">
                <span class="label">Strategy</span>
                <span class="value">{selectedTrade.strategy}</span>
              </div>
              <div class="summary-item">
                <span class="label">Profit</span>
                <span class="value" style="color: {getProfitColor(selectedTrade.profit)}">
                  {formatProfit(selectedTrade.profit)}
                </span>
              </div>
            </div>
          </div>

          <!-- Full Context -->
          <div class="detail-section">
            <h4>Why This Trade?</h4>
            <div class="reason-box">
              <p>{selectedTrade.reason}</p>
              {#if selectedTrade.context?.sentiment}
                <p class="sentiment"><em>"{selectedTrade.context.sentiment}"</em></p>
              {/if}
            </div>
          </div>

          <!-- Action Buttons -->
          <div class="detail-actions">
            <button class="btn" on:click={() => window.open(`/api/journal/export/${selectedTrade.id}`, '_blank')}>
              <Download size={14} />
              <span>Export Trade</span>
            </button>
            <button class="btn" on:click={() => dispatch('viewInBacktest', { trade: selectedTrade })}>
              <BarChart3 size={14} />
              <span>View in Backtest</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  {/if}
</div>

<style>
  .journal-view {
    display: flex;
    flex-direction: column;
    height: 100%;
    background: var(--bg-primary);
    overflow: hidden;
  }

  /* Header */
  .journal-header {
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

  .journal-icon {
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
  }

  .btn:hover {
    background: var(--bg-surface);
  }

  .icon-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 28px;
    height: 28px;
    background: transparent;
    border: none;
    border-radius: 4px;
    color: var(--text-muted);
    cursor: pointer;
  }

  .icon-btn:hover {
    background: var(--bg-tertiary);
    color: var(--text-primary);
  }

  /* Stats Row */
  .stats-row {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 12px;
    padding: 16px 24px;
    background: var(--bg-secondary);
    border-bottom: 1px solid var(--border-subtle);
  }

  .stat-card {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 12px;
    background: var(--bg-tertiary);
    border-radius: 8px;
  }

  .stat-icon {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 36px;
    height: 36px;
    background: var(--bg-surface);
    border-radius: 8px;
    color: var(--text-muted);
  }

  .stat-icon.success {
    background: rgba(16, 185, 129, 0.2);
    color: #10b981;
  }

  .stat-icon.danger {
    background: rgba(239, 68, 68, 0.2);
    color: #ef4444;
  }

  .stat-icon.positive {
    background: rgba(16, 185, 129, 0.2);
    color: #10b981;
  }

  .stat-info {
    display: flex;
    flex-direction: column;
  }

  .stat-value {
    font-size: 16px;
    font-weight: 600;
    color: var(--text-primary);
  }

  .stat-value.positive {
    color: #10b981;
  }

  .stat-value.negative {
    color: #ef4444;
  }

  .stat-label {
    font-size: 11px;
    color: var(--text-muted);
  }

  /* Filters */
  .filters-bar {
    display: flex;
    gap: 12px;
    padding: 16px 24px;
    background: var(--bg-primary);
    border-bottom: 1px solid var(--border-subtle);
    flex-wrap: wrap;
  }

  .filter-group {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 12px;
    background: var(--bg-secondary);
    border: 1px solid var(--border-subtle);
    border-radius: 6px;
    font-size: 12px;
  }

  .filter-group input,
  .filter-group select {
    background: transparent;
    border: none;
    color: var(--text-primary);
    font-size: 12px;
    outline: none;
  }

  .filter-group input[type="text"] {
    width: 200px;
  }

  .filter-group.date-range {
    gap: 6px;
  }

  .filter-group.date-range span {
    color: var(--text-muted);
  }

  /* Table */
  .trades-table-container {
    flex: 1;
    overflow-y: auto;
  }

  .table-header {
    display: grid;
    grid-template-columns: 140px 80px 60px 1fr 100px 60px 60px 60px;
    gap: 8px;
    padding: 12px 24px;
    background: var(--bg-tertiary);
    font-size: 11px;
    font-weight: 500;
    color: var(--text-muted);
    position: sticky;
    top: 0;
    z-index: 10;
  }

  .header-cell {
    display: flex;
    align-items: center;
    gap: 4px;
  }

  .header-cell.sortable {
    cursor: pointer;
  }

  .header-cell.sortable:hover {
    color: var(--text-primary);
  }

  .header-cell :global(svg) {
    opacity: 0;
  }

  .header-cell.sortable :global(span.active svg) {
    opacity: 1;
  }

  .table-row {
    display: grid;
    grid-template-columns: 140px 80px 60px 1fr 100px 60px 60px 60px;
    gap: 8px;
    padding: 12px 24px;
    border-bottom: 1px solid var(--border-subtle);
    font-size: 12px;
    transition: background 0.15s;
  }

  .table-row:hover {
    background: var(--bg-secondary);
  }

  .table-row.expanded {
    background: var(--bg-secondary);
    border-bottom: none;
  }

  .cell {
    display: flex;
    align-items: center;
    color: var(--text-primary);
  }

  .cell.time .time-main {
    font-size: 12px;
  }

  .symbol-badge {
    padding: 4px 8px;
    background: var(--bg-tertiary);
    border-radius: 4px;
    font-weight: 500;
  }

  .type-badge {
    padding: 4px 8px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 500;
  }

  .type-badge.buy {
    background: rgba(16, 185, 129, 0.2);
    color: #10b981;
  }

  .type-badge.sell {
    background: rgba(239, 68, 68, 0.2);
    color: #ef4444;
  }

  .strategy-name {
    color: var(--text-primary);
  }

  .profit-value {
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
  }

  .regime-indicator {
    padding: 4px 8px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 600;
    color: white;
    text-align: center;
    min-width: 40px;
  }

  .kelly-value {
    font-family: 'JetBrains Mono', monospace;
    color: var(--text-muted);
  }

  /* Expanded Row */
  .expanded-row {
    grid-column: 1 / -1;
    background: var(--bg-surface);
    border-bottom: 1px solid var(--border-subtle);
  }

  .expanded-content {
    padding: 16px 24px;
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 16px;
  }

  .context-section {
    background: var(--bg-secondary);
    border-radius: 8px;
    padding: 12px;
  }

  .context-section h4 {
    display: flex;
    align-items: center;
    gap: 6px;
    margin: 0 0 12px;
    font-size: 12px;
    color: var(--text-muted);
  }

  .context-grid {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .context-item {
    display: flex;
    justify-content: space-between;
    font-size: 11px;
  }

  .context-item .label {
    color: var(--text-muted);
  }

  .context-item .value {
    color: var(--text-primary);
  }

  .regime-details {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .regime-metric {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 11px;
  }

  .metric-label {
    color: var(--text-muted);
    min-width: 50px;
  }

  .metric-bar {
    flex: 1;
    height: 6px;
    background: var(--bg-tertiary);
    border-radius: 3px;
    overflow: hidden;
  }

  .metric-fill {
    height: 100%;
    transition: width 0.3s;
  }

  .metric-value {
    min-width: 40px;
    text-align: right;
    font-family: 'JetBrains Mono', monospace;
    color: var(--text-primary);
  }

  .metric-badge {
    padding: 4px 8px;
    border-radius: 4px;
    font-size: 10px;
    text-transform: capitalize;
  }

  .metric-badge.bullish {
    background: rgba(16, 185, 129, 0.2);
    color: #10b981;
  }

  .metric-badge.bearish {
    background: rgba(239, 68, 68, 0.2);
    color: #ef4444;
  }

  .metric-badge.ranging {
    background: rgba(245, 158, 11, 0.2);
    color: #f59e0b;
  }

  .kelly-details {
    display: flex;
    flex-direction: column;
    gap: 6px;
  }

  .kelly-item {
    display: flex;
    justify-content: space-between;
    font-size: 11px;
  }

  .kelly-item .label {
    color: var(--text-muted);
  }

  .kelly-item .value {
    font-family: 'JetBrains Mono', monospace;
    color: var(--text-primary);
  }

  .house-money-details {
    display: flex;
    justify-content: space-between;
    font-size: 11px;
  }

  .hm-item .label {
    color: var(--text-muted);
  }

  .hm-item .value.success {
    color: #10b981;
  }

  /* Empty State */
  .empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 60px 24px;
    color: var(--text-muted);
    text-align: center;
  }

  .empty-state p {
    margin-top: 12px;
    font-size: 13px;
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
    width: 600px;
    max-width: 90%;
    max-height: 80vh;
    overflow: hidden;
    display: flex;
    flex-direction: column;
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

  .trade-id {
    margin: 4px 0 0;
    font-size: 11px;
    color: var(--text-muted);
    font-family: 'JetBrains Mono', monospace;
  }

  .modal-content {
    flex: 1;
    overflow-y: auto;
    padding: 20px;
  }

  .detail-section {
    margin-bottom: 20px;
  }

  .detail-section h4 {
    margin: 0 0 12px;
    font-size: 13px;
    color: var(--text-primary);
  }

  .summary-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 12px;
  }

  .summary-item {
    display: flex;
    justify-content: space-between;
    padding: 8px 12px;
    background: var(--bg-tertiary);
    border-radius: 6px;
    font-size: 12px;
  }

  .summary-item .label {
    color: var(--text-muted);
  }

  .summary-item .value {
    color: var(--text-primary);
    font-weight: 500;
  }

  .reason-box {
    padding: 12px;
    background: var(--bg-tertiary);
    border-radius: 8px;
    font-size: 13px;
    color: var(--text-primary);
  }

  .reason-box p {
    margin: 0;
  }

  .reason-box .sentiment {
    margin-top: 8px;
    color: var(--text-muted);
  }

  .detail-actions {
    display: flex;
    gap: 8px;
    padding-top: 16px;
    border-top: 1px solid var(--border-subtle);
  }

  .detail-actions .btn {
    flex: 1;
    justify-content: center;
  }
</style>
