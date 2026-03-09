<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { Bot, DollarSign, Percent } from 'lucide-svelte';
  import { API_CONFIG } from '$lib/config/api';
  import { getAllSessions, getMarketState, getTradingMetrics } from '$lib/api';

  // State
  let loading = true;
  let sessions: Record<string, { active: boolean; name: string }> = {};
  let regime = 'UNKNOWN';
  let activeBots = 0;
  let dailyPnl = 0;
  let winRate = 0;

  let refreshInterval: ReturnType<typeof setInterval>;

  // Use centralized API configuration
  const API_BASE = API_CONFIG.API_BASE;

  // Explicit session order for deterministic display
  const SESSION_ORDER = ['ASIAN', 'LONDON', 'NEW_YORK', 'OVERLAP'];

  onMount(async () => {
    await fetchData();
    loading = false;
    refreshInterval = setInterval(fetchData, 5000); // Refresh every 5 seconds
  });

  onDestroy(() => {
    if (refreshInterval) clearInterval(refreshInterval);
  });

  async function fetchData() {
    try {
      // Fetch sessions using helper
      const sessionsData = await getAllSessions();
      if (sessionsData) sessions = sessionsData;

      // Fetch market state using helper
      const marketData = await getMarketState();
      if (marketData) {
        // Extract regime from the nested structure
        if (marketData.regime) {
          regime = determineRegime(marketData.regime);
        }
      }

      // Fetch trading metrics using helper
      const metricsData = await getTradingMetrics();
      if (metricsData) {
        activeBots = metricsData.active_bots || 0;
        dailyPnl = metricsData.daily_pnl || 0;
        winRate = (metricsData.win_rate || 0) * 100;
      }
    } catch (e) {
      console.error('StatusBand: Failed to fetch data', e);
      loading = false;
    }
  }

  // Determine regime string from market data
  function determineRegime(regimeData: { trend?: string; volatility?: string; chaos?: number }): string {
    if (!regimeData) return 'UNKNOWN';

    const { trend, volatility, chaos } = regimeData;

    if (chaos && chaos > 50) return 'HIGH_CHAOS';
    if (volatility === 'low' && trend) return 'RANGE_STABLE';
    if (volatility === 'high') return 'HIGH_CHAOS';
    if (trend === 'bullish' || trend === 'bearish') return 'TREND_STABLE';
    if (volatility === 'medium') return 'BREAKOUT_PRIME';

    return 'UNCERTAIN';
  }

  // Helper: Get session color
  function getSessionColor(active: boolean): string {
    return active ? 'var(--accent-success, #10b981)' : 'var(--text-muted, #6b7280)';
  }

  // Helper: Get regime color
  function getRegimeColor(r: string): string {
    const colors: Record<string, string> = {
      'TREND_STABLE': '#10b981',
      'RANGE_STABLE': '#3b82f6',
      'HIGH_CHAOS': '#ef4444',
      'BREAKOUT_PRIME': '#f59e0b',
      'UNCERTAIN': '#6b7280'
    };
    return colors[r] || '#6b7280';
  }

  // Helper: Format P&L
  function formatPnl(pnl: number): string {
    const sign = pnl >= 0 ? '+' : '';
    return `${sign}$${pnl.toFixed(2)}`;
  }
</script>

<div class="status-band">
  {#if loading}
    <span class="loading">Loading...</span>
  {:else}
    <div class="sessions">
      {#each SESSION_ORDER as sessionKey}
        {#if sessions[sessionKey] && sessionKey !== 'CLOSED'}
          <div class="session-item" class:active={sessions[sessionKey].active}>
            <span class="dot" style="background: {getSessionColor(sessions[sessionKey].active)}"></span>
            <span class="session-name">{sessionKey}</span>
          </div>
        {/if}
      {/each}
    </div>

  <div class="divider">|</div>

  <div class="regime">
    <span class="regime-dot" style="background: {getRegimeColor(regime)}"></span>
    <span>{regime}</span>
  </div>

  <div class="divider">|</div>

  <div class="metrics">
    <div class="metric">
      <Bot size={14} />
      <span>{activeBots} Bots</span>
    </div>
    <div class="metric" class:profit={dailyPnl >= 0} class:loss={dailyPnl < 0}>
      <DollarSign size={14} />
      <span>{formatPnl(dailyPnl)}</span>
    </div>
    <div class="metric">
      <Percent size={14} />
      <span>{winRate.toFixed(0)}% WR</span>
    </div>
  </div>
  {/if}
</div>

<style>
  .status-band {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 8px 16px;
    background: var(--bg-secondary, #1e293b);
    border-bottom: 1px solid var(--border-subtle, #374151);
    font-size: 12px;
    color: var(--text-secondary, #d1d5db);
    overflow-x: auto;
  }

  .loading {
    color: var(--text-muted, #6b7280);
    font-style: italic;
  }

  .sessions {
    display: flex;
    gap: 12px;
  }

  .session-item {
    display: flex;
    align-items: center;
    gap: 4px;
  }

  .session-item.active {
    color: var(--text-primary, #fff);
  }

  .dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
  }

  .divider {
    color: var(--border-subtle, #374151);
  }

  .regime {
    display: flex;
    align-items: center;
    gap: 6px;
  }

  .regime-dot {
    width: 8px;
    height: 8px;
    border-radius: 2px;
  }

  .metrics {
    display: flex;
    gap: 16px;
    margin-left: auto;
  }

  .metric {
    display: flex;
    align-items: center;
    gap: 4px;
  }

  .metric.profit {
    color: var(--accent-success, #10b981);
  }

  .metric.loss {
    color: var(--accent-danger, #ef4444);
  }
</style>
