<script lang="ts">
  import { onMount } from 'svelte';
  import {
    X,
    TrendingUp,
    TrendingDown,
    AlertTriangle,
    Shield,
    Activity,
    Calendar,
    Clock,
    BarChart3,
    Award
  } from 'lucide-svelte';
  import { API_BASE } from '$lib/constants';

  const apiBase = API_BASE || '';

  // Report data
  let report = $state<{
    date: string;
    total_trades: number;
    wins: number;
    losses: number;
    win_rate: number;
    net_pnl: number;
    best_trade: number;
    worst_trade: number;
    circuit_breaker_events: Array<{
      timestamp: string;
      event_type: string;
      severity: string;
      description: string;
      affected_bots: string[];
    }>;
    force_close_reasons: Array<{
      timestamp: string;
      event_type: string;
      severity: string;
      description: string;
      affected_bots: string[];
    }>;
    session_breakdown: {
      london: { pnl: number; trades: number };
      ny: { pnl: number; trades: number };
    };
    regime_at_close: string;
    timestamp: string;
  } | null>(null);

  let availableReports = $state<Array<{
    report_id: string;
    trading_date: string;
    total_trades: number;
    wins: number;
    losses: number;
    win_rate: number;
    net_pnl: number;
    regime_at_close: string;
    generated_at: string;
  }>>([]);

  let loading = $state(true);
  let error = $state<string | null>(null);
  let selectedDate = $state<string>('latest');
  let showReportSelector = $state(false);

  // Available dates for selector
  let reportDates = $derived(
    availableReports.map(r => r.trading_date).sort((a, b) => b.localeCompare(a))
  );

  async function fetchLatestReport() {
    try {
      loading = true;
      error = null;
      const baseUrl = apiBase || window.location.origin;
      const response = await fetch(`${baseUrl}/api/dead-zone/eod-report/latest`);

      if (!response.ok) {
        if (response.status === 404) {
          error = 'No EOD report available yet';
        } else {
          throw new Error(`Failed to fetch: ${response.status}`);
        }
        report = null;
        return;
      }

      report = await response.json();
      selectedDate = report?.date || 'latest';
    } catch (e) {
      console.error('Failed to fetch latest EOD report:', e);
      error = e instanceof Error ? e.message : 'Unknown error';
      report = null;
    } finally {
      loading = false;
    }
  }

  async function fetchReportByDate(date: string) {
    try {
      loading = true;
      error = null;
      const baseUrl = apiBase || window.location.origin;
      const response = await fetch(`${baseUrl}/api/dead-zone/eod-report/${date}`);

      if (!response.ok) {
        throw new Error(`Failed to fetch: ${response.status}`);
      }

      report = await response.json();
      selectedDate = date;
      showReportSelector = false;
    } catch (e) {
      console.error('Failed to fetch EOD report:', e);
      error = e instanceof Error ? e.message : 'Unknown error';
      report = null;
    } finally {
      loading = false;
    }
  }

  async function fetchAvailableReports() {
    try {
      const baseUrl = apiBase || window.location.origin;
      const response = await fetch(`${baseUrl}/api/dead-zone/eod-reports`);

      if (!response.ok) {
        console.warn('Failed to fetch available reports:', response.status);
        return;
      }

      availableReports = await response.json();
    } catch (e) {
      console.warn('Failed to fetch available reports:', e);
    }
  }

  function formatDate(dateStr: string): string {
    if (!dateStr) return '';
    const date = new Date(dateStr);
    return date.toLocaleDateString('en-US', {
      weekday: 'short',
      year: 'numeric',
      month: 'short',
      day: 'numeric'
    });
  }

  function formatTime(timestamp: string): string {
    if (!timestamp) return '';
    const date = new Date(timestamp);
    return date.toLocaleTimeString('en-US', {
      hour: '2-digit',
      minute: '2-digit'
    });
  }

  function formatCurrency(value: number): string {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2
    }).format(value);
  }

  function formatPercent(value: number): string {
    return `${(value * 100).toFixed(1)}%`;
  }

  function getSeverityColor(severity: string): string {
    switch (severity.toUpperCase()) {
      case 'CRITICAL': return '#ef4444';
      case 'HIGH': return '#f97316';
      case 'MEDIUM': return '#f59e0b';
      case 'LOW': return '#10b981';
      default: return '#6b7280';
    }
  }

  function getRegimeColor(regime: string): string {
    switch (regime.toUpperCase()) {
      case 'TREND': return '#3b82f6';
      case 'RANGE': return '#10b981';
      case 'BREAKOUT': return '#f59e0b';
      case 'CHAOS': return '#ef4444';
      default: return '#6b7280';
    }
  }

  onMount(() => {
    fetchLatestReport();
    fetchAvailableReports();
  });
</script>

<div class="eod-report-viewer">
  <div class="report-header">
    <div class="header-title">
      <BarChart3 size={18} />
      <h3>EOD Report</h3>
    </div>
    <div class="header-actions">
      {#if reportDates.length > 1}
        <button
          class="selector-btn"
          onclick={() => showReportSelector = !showReportSelector}
          title="Select report date"
        >
          <Calendar size={14} />
          <span>{selectedDate === 'latest' ? 'Latest' : formatDate(selectedDate)}</span>
        </button>
      {/if}
    </div>
  </div>

  {#if showReportSelector && reportDates.length > 0}
    <div class="report-selector">
      {#each reportDates as date}
        <button
          class="date-option"
          class:active={date === selectedDate}
          onclick={() => fetchReportByDate(date)}
        >
          {formatDate(date)}
        </button>
      {/each}
    </div>
  {/if}

  {#if error}
    <div class="error-state">
      <AlertTriangle size={20} />
      <span>{error}</span>
    </div>
  {:else if loading}
    <div class="loading-state">
      <div class="spinner"></div>
      <span>Loading EOD report...</span>
    </div>
  {:else if report}
    <div class="report-content">
      <!-- Summary Stats Row -->
      <div class="summary-stats">
        <div class="stat-card">
          <div class="stat-icon trades">
            <Activity size={16} />
          </div>
          <div class="stat-value">{report.total_trades}</div>
          <div class="stat-label">Total Trades</div>
        </div>

        <div class="stat-card">
          <div class="stat-icon win-rate" class:positive={report.win_rate >= 0.5}>
            <Award size={16} />
          </div>
          <div class="stat-value">{formatPercent(report.win_rate)}</div>
          <div class="stat-label">Win Rate</div>
        </div>

        <div class="stat-card">
          <div class="stat-icon pnl" class:positive={report.net_pnl >= 0} class:negative={report.net_pnl < 0}>
            {#if report.net_pnl >= 0}
              <TrendingUp size={16} />
            {:else}
              <TrendingDown size={16} />
            {/if}
          </div>
          <div class="stat-value" class:positive={report.net_pnl >= 0} class:negative={report.net_pnl < 0}>
            {formatCurrency(report.net_pnl)}
          </div>
          <div class="stat-label">Net PnL</div>
        </div>

        <div class="stat-card">
          <div class="stat-icon best">
            <TrendingUp size={16} />
          </div>
          <div class="stat-value positive">{formatCurrency(report.best_trade)}</div>
          <div class="stat-label">Best Trade</div>
        </div>

        <div class="stat-card">
          <div class="stat-icon worst">
            <TrendingDown size={16} />
          </div>
          <div class="stat-value negative">{formatCurrency(report.worst_trade)}</div>
          <div class="stat-label">Worst Trade</div>
        </div>
      </div>

      <!-- Wins/Losses Bar -->
      <div class="wl-bar-container">
        <div class="wl-bar">
          <div
            class="wl-win"
            style="width: {report.win_rate * 100}%"
          ></div>
          <div
            class="wl-loss"
            style="width: {(1 - report.win_rate) * 100}%"
          ></div>
        </div>
        <div class="wl-labels">
          <span class="wins">{report.wins} Wins</span>
          <span class="losses">{report.losses} Losses</span>
        </div>
      </div>

      <!-- Session Breakdown -->
      <div class="section">
        <h4 class="section-title">
          <Clock size={14} />
          Session Breakdown
        </h4>
        <div class="session-table">
          <div class="session-row">
            <div class="session-label london">London</div>
            <div class="session-pnl" class:positive={report.session_breakdown.london.pnl >= 0} class:negative={report.session_breakdown.london.pnl < 0}>
              {formatCurrency(report.session_breakdown.london.pnl)}
            </div>
            <div class="session-trades">{report.session_breakdown.london.trades} trades</div>
          </div>
          <div class="session-row">
            <div class="session-label ny">New York</div>
            <div class="session-pnl" class:positive={report.session_breakdown.ny.pnl >= 0} class:negative={report.session_breakdown.ny.pnl < 0}>
              {formatCurrency(report.session_breakdown.ny.pnl)}
            </div>
            <div class="session-trades">{report.session_breakdown.ny.trades} trades</div>
          </div>
        </div>
        <div class="regime-badge" style="background: {getRegimeColor(report.regime_at_close)}20; color: {getRegimeColor(report.regime_at_close)}">
          Regime at Close: {report.regime_at_close}
        </div>
      </div>

      <!-- Circuit Breaker Events -->
      {#if report.circuit_breaker_events.length > 0}
        <div class="section">
          <h4 class="section-title warning">
            <Shield size={14} />
            Circuit Breaker Events ({report.circuit_breaker_events.length})
          </h4>
          <div class="events-list">
            {#each report.circuit_breaker_events as event}
              <div class="event-item">
                <div class="event-header">
                  <span class="event-type">{event.event_type}</span>
                  <span class="event-severity" style="color: {getSeverityColor(event.severity)}">
                    {event.severity}
                  </span>
                </div>
                <div class="event-description">{event.description}</div>
                <div class="event-meta">
                  <span class="event-time">{formatTime(event.timestamp)}</span>
                  {#if event.affected_bots.length > 0}
                    <span class="event-bots">Bots: {event.affected_bots.join(', ')}</span>
                  {/if}
                </div>
              </div>
            {/each}
          </div>
        </div>
      {/if}

      <!-- Force Close Reasons -->
      {#if report.force_close_reasons.length > 0}
        <div class="section">
          <h4 class="section-title danger">
            <AlertTriangle size={14} />
            Force Close Reasons ({report.force_close_reasons.length})
          </h4>
          <div class="events-list">
            {#each report.force_close_reasons as event}
              <div class="event-item">
                <div class="event-header">
                  <span class="event-type">{event.event_type}</span>
                  <span class="event-severity" style="color: {getSeverityColor(event.severity)}">
                    {event.severity}
                  </span>
                </div>
                <div class="event-description">{event.description}</div>
                <div class="event-meta">
                  <span class="event-time">{formatTime(event.timestamp)}</span>
                  {#if event.affected_bots.length > 0}
                    <span class="event-bots">Bots: {event.affected_bots.join(', ')}</span>
                  {/if}
                </div>
              </div>
            {/each}
          </div>
        </div>
      {/if}

      <!-- Timestamp -->
      <div class="report-footer">
        <Clock size={12} />
        <span>Report generated: {formatDate(report.timestamp)} at {formatTime(report.timestamp)}</span>
      </div>
    </div>
  {:else}
    <div class="empty-state">
      <BarChart3 size={32} />
      <span>No EOD report data available</span>
    </div>
  {/if}
</div>

<style>
  .eod-report-viewer {
    background: rgba(8, 8, 12, 0.85);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 16px;
    color: #e4e4e7;
    max-height: 600px;
    overflow-y: auto;
  }

  .report-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 16px;
    padding-bottom: 12px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
  }

  .header-title {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .header-title h3 {
    font-size: 14px;
    font-weight: 600;
    margin: 0;
  }

  .header-actions {
    display: flex;
    gap: 8px;
  }

  .selector-btn {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 6px 12px;
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 6px;
    color: #9ca3af;
    font-size: 12px;
    cursor: pointer;
    transition: all 0.2s;
  }

  .selector-btn:hover {
    background: rgba(255, 255, 255, 0.1);
    color: #e4e4e7;
  }

  .report-selector {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    padding: 12px;
    background: rgba(255, 255, 255, 0.03);
    border-radius: 8px;
    margin-bottom: 16px;
  }

  .date-option {
    padding: 4px 10px;
    background: transparent;
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 4px;
    color: #9ca3af;
    font-size: 11px;
    cursor: pointer;
    transition: all 0.2s;
  }

  .date-option:hover {
    background: rgba(255, 255, 255, 0.05);
    color: #e4e4e7;
  }

  .date-option.active {
    background: rgba(16, 185, 129, 0.2);
    border-color: rgba(16, 185, 129, 0.4);
    color: #10b981;
  }

  .loading-state,
  .error-state,
  .empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 12px;
    padding: 40px;
    color: #6b7280;
  }

  .error-state {
    color: #ef4444;
  }

  .spinner {
    width: 24px;
    height: 24px;
    border: 2px solid rgba(255, 255, 255, 0.1);
    border-top-color: #10b981;
    border-radius: 50%;
    animation: spin 1s linear infinite;
  }

  @keyframes spin {
    to { transform: rotate(360deg); }
  }

  .report-content {
    display: flex;
    flex-direction: column;
    gap: 16px;
  }

  .summary-stats {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(100px, 1fr));
    gap: 10px;
  }

  .stat-card {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 8px;
    padding: 12px;
    text-align: center;
  }

  .stat-icon {
    width: 32px;
    height: 32px;
    margin: 0 auto 8px;
    border-radius: 6px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: rgba(107, 114, 128, 0.2);
    color: #9ca3af;
  }

  .stat-icon.trades { background: rgba(59, 130, 246, 0.2); color: #3b82f6; }
  .stat-icon.win-rate { background: rgba(107, 114, 128, 0.2); color: #9ca3af; }
  .stat-icon.win-rate.positive { background: rgba(16, 185, 129, 0.2); color: #10b981; }
  .stat-icon.pnl { background: rgba(107, 114, 128, 0.2); color: #9ca3af; }
  .stat-icon.pnl.positive { background: rgba(16, 185, 129, 0.2); color: #10b981; }
  .stat-icon.pnl.negative { background: rgba(239, 68, 68, 0.2); color: #ef4444; }
  .stat-icon.best { background: rgba(16, 185, 129, 0.2); color: #10b981; }
  .stat-icon.worst { background: rgba(239, 68, 68, 0.2); color: #ef4444; }

  .stat-value {
    font-size: 18px;
    font-weight: 600;
    color: #e4e4e7;
    margin-bottom: 2px;
  }

  .stat-value.positive { color: #10b981; }
  .stat-value.negative { color: #ef4444; }

  .stat-label {
    font-size: 10px;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  .wl-bar-container {
    padding: 12px;
    background: rgba(255, 255, 255, 0.03);
    border-radius: 8px;
  }

  .wl-bar {
    display: flex;
    height: 8px;
    border-radius: 4px;
    overflow: hidden;
    margin-bottom: 8px;
  }

  .wl-win {
    background: #10b981;
    transition: width 0.3s ease;
  }

  .wl-loss {
    background: #ef4444;
    transition: width 0.3s ease;
  }

  .wl-labels {
    display: flex;
    justify-content: space-between;
    font-size: 11px;
  }

  .wl-labels .wins { color: #10b981; }
  .wl-labels .losses { color: #ef4444; }

  .section {
    padding: 12px;
    background: rgba(255, 255, 255, 0.03);
    border-radius: 8px;
  }

  .section-title {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 12px;
    font-weight: 600;
    color: #e4e4e7;
    margin: 0 0 10px 0;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  .section-title.warning { color: #f59e0b; }
  .section-title.danger { color: #ef4444; }

  .session-table {
    display: flex;
    flex-direction: column;
    gap: 6px;
  }

  .session-row {
    display: grid;
    grid-template-columns: 80px 1fr 80px;
    align-items: center;
    gap: 10px;
    padding: 8px;
    background: rgba(255, 255, 255, 0.02);
    border-radius: 6px;
  }

  .session-label {
    font-size: 12px;
    font-weight: 500;
  }

  .session-label.london { color: #3b82f6; }
  .session-label.ny { color: #8b5cf6; }

  .session-pnl {
    font-size: 13px;
    font-weight: 600;
    color: #9ca3af;
  }

  .session-pnl.positive { color: #10b981; }
  .session-pnl.negative { color: #ef4444; }

  .session-trades {
    font-size: 11px;
    color: #6b7280;
    text-align: right;
  }

  .regime-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    margin-top: 10px;
    padding: 4px 10px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 500;
  }

  .events-list {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .event-item {
    padding: 10px;
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 6px;
  }

  .event-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 4px;
  }

  .event-type {
    font-size: 11px;
    font-weight: 600;
    color: #e4e4e7;
  }

  .event-severity {
    font-size: 10px;
    font-weight: 500;
    text-transform: uppercase;
  }

  .event-description {
    font-size: 12px;
    color: #9ca3af;
    margin-bottom: 6px;
  }

  .event-meta {
    display: flex;
    gap: 12px;
    font-size: 10px;
    color: #6b7280;
  }

  .report-footer {
    display: flex;
    align-items: center;
    gap: 6px;
    padding-top: 12px;
    border-top: 1px solid rgba(255, 255, 255, 0.08);
    font-size: 11px;
    color: #6b7280;
  }
</style>
