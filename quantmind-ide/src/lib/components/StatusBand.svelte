<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { Bot, DollarSign, Percent, Shield, Route, TrendingUp, Activity, Target, Clock } from 'lucide-svelte';
  import { getAllSessions, getMarketState, getTradingMetrics, getRiskSettings, getRouterSettings } from '$lib/api';
  import { navigationStore } from '../stores/navigationStore';

  // State
  let loading = true;
  let sessions: Record<string, { active: boolean; name: string }> = {};
  let regime = 'UNKNOWN';
  let activeBots = 0;
  let dailyPnl = 0;
  let winRate = 0;
  let openPositions = 0;
  let tradesToday = 0;
  let currentSession = 'CLOSED';

  // Risk and Router settings
  let riskMode: 'fixed' | 'dynamic' | 'conservative' = 'dynamic';
  let routerMode: 'auction' | 'priority' | 'round-robin' = 'auction';

  let refreshInterval: ReturnType<typeof setInterval>;

  // Explicit session order for deterministic display
  const SESSION_ORDER = ['ASIAN', 'LONDON', 'NEW_YORK', 'OVERLAP'];

  // Helper: Get current active trading session
  function getCurrentSession(): string {
    for (const sessionKey of SESSION_ORDER) {
      if (sessions[sessionKey]?.active) {
        return sessionKey;
      }
    }
    return 'CLOSED';
  }

  onMount(async () => {
    try {
      await fetchData();
    } finally {
      loading = false;
    }
    refreshInterval = setInterval(fetchData, 5000); // Refresh every 5 seconds
  });

  onDestroy(() => {
    if (refreshInterval) clearInterval(refreshInterval);
  });

  async function fetchData() {
    try {
      // Fetch sessions using helper
      const sessionsData = await getAllSessions();
      if (sessionsData) {
        sessions = sessionsData;
        currentSession = getCurrentSession();
      }

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
        openPositions = metricsData.active_positions || 0;
        tradesToday = metricsData.total_trades || 0;
      }

      // Fetch risk settings
      try {
        const riskData = await getRiskSettings();
        if (riskData && riskData.riskMode) {
          riskMode = riskData.riskMode;
        }
      } catch (e) {
        // Risk settings may not be available, use default
        console.debug('StatusBand: Risk settings not available');
      }

      // Fetch router settings
      try {
        const routerData = await getRouterSettings();
        if (routerData && routerData.mode) {
          routerMode = routerData.mode;
        }
      } catch (e) {
        // Router settings may not be available, use default
        console.debug('StatusBand: Router settings not available');
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

  // Helper: Format risk mode for display
  function formatRiskMode(mode: string): string {
    const labels: Record<string, string> = {
      fixed: 'Fixed',
      dynamic: 'Dynamic',
      conservative: 'Conservative'
    };
    return labels[mode] || mode;
  }

  // Helper: Format router mode for display
  function formatRouterMode(mode: string): string {
    const labels: Record<string, string> = {
      auction: 'Auction',
      priority: 'Priority',
      'round-robin': 'Round Robin'
    };
    return labels[mode] || mode;
  }
</script>

<div class="status-band">
  {#if loading}
    <span class="loading">Loading...</span>
  {:else}
    <div class="ticker-wrapper">
      <!-- Current Trading Session - prominently displayed -->
      <div class="current-session clickable" on:click={() => navigationStore.navigateToView('live')}>
        <Clock size={16} />
        <span class="label">Trading:</span>
        <span class="session-name current">{currentSession}</span>
      </div>

      <div class="divider">|</div>

      <!-- All sessions with status -->
      <div class="sessions">
        {#each SESSION_ORDER as sessionKey}
          {#if sessions[sessionKey]}
            <div class="session-item" class:active={sessions[sessionKey].active}>
              <span class="dot" style="background: {getSessionColor(sessions[sessionKey].active)}"></span>
              <span class="session-name">{sessionKey}</span>
            </div>
          {/if}
        {/each}
      </div>

      <div class="divider">|</div>

      <div class="regime clickable" on:click={() => navigationStore.navigateToView('live')}>
        <span class="regime-dot" style="background: {getRegimeColor(regime)}"></span>
        <span>{regime}</span>
      </div>

      <div class="divider">|</div>

      <div class="metrics">
        <div class="metric clickable" on:click={() => navigationStore.navigateToView('live')}>
          <Bot size={16} />
          <span>{activeBots} Bots</span>
        </div>
        <div class="metric clickable" class:profit={dailyPnl >= 0} class:loss={dailyPnl < 0} on:click={() => navigationStore.navigateToView('live')}>
          <DollarSign size={16} />
          <span>{formatPnl(dailyPnl)}</span>
        </div>
        <div class="metric clickable" on:click={() => navigationStore.navigateToView('live')}>
          <Percent size={16} />
          <span>{winRate.toFixed(0)}% WR</span>
        </div>
        <div class="metric clickable" on:click={() => navigationStore.navigateToView('live')}>
          <Target size={16} />
          <span>{openPositions} Pos</span>
        </div>
        <div class="metric clickable" on:click={() => navigationStore.navigateToView('live')}>
          <Activity size={16} />
          <span>{tradesToday} Trades</span>
        </div>
      </div>

      <div class="divider">|</div>

      <div class="mode-indicators clickable" on:click={() => navigationStore.navigateToView('settings')}>
        <div class="mode-item" title="Risk Mode">
          <Shield size={16} />
          <span>Risk: {formatRiskMode(riskMode)}</span>
        </div>
        <div class="mode-item" title="Router Mode">
          <Route size={16} />
          <span>Router: {formatRouterMode(routerMode)}</span>
        </div>
      </div>

      <!-- Duplicate for seamless loop -->
      <!-- Current Trading Session -->
      <div class="current-session clickable" on:click={() => navigationStore.navigateToView('live')}>
        <Clock size={16} />
        <span class="label">Trading:</span>
        <span class="session-name current">{currentSession}</span>
      </div>

      <div class="divider">|</div>

      <!-- All sessions -->
      <div class="sessions">
        {#each SESSION_ORDER as sessionKey}
          {#if sessions[sessionKey]}
            <div class="session-item" class:active={sessions[sessionKey].active}>
              <span class="dot" style="background: {getSessionColor(sessions[sessionKey].active)}"></span>
              <span class="session-name">{sessionKey}</span>
            </div>
          {/if}
        {/each}
      </div>

      <div class="divider">|</div>

      <div class="regime clickable" on:click={() => navigationStore.navigateToView('live')}>
        <span class="regime-dot" style="background: {getRegimeColor(regime)}"></span>
        <span>{regime}</span>
      </div>

      <div class="divider">|</div>

      <div class="metrics">
        <div class="metric clickable" on:click={() => navigationStore.navigateToView('live')}>
          <Bot size={16} />
          <span>{activeBots} Bots</span>
        </div>
        <div class="metric clickable" class:profit={dailyPnl >= 0} class:loss={dailyPnl < 0} on:click={() => navigationStore.navigateToView('live')}>
          <DollarSign size={16} />
          <span>{formatPnl(dailyPnl)}</span>
        </div>
        <div class="metric clickable" on:click={() => navigationStore.navigateToView('live')}>
          <Target size={16} />
          <span>{openPositions} Pos</span>
        </div>
        <div class="metric clickable" on:click={() => navigationStore.navigateToView('trading')}>
          <Activity size={16} />
          <span>{tradesToday} Trades</span>
        </div>
      </div>

      <div class="divider">|</div>

      <div class="mode-indicators clickable" on:click={() => navigationStore.navigateToView('settings')}>
        <div class="mode-item" title="Risk Mode">
          <Shield size={16} />
          <span>Risk: {formatRiskMode(riskMode)}</span>
        </div>
        <div class="mode-item" title="Router Mode">
          <Route size={16} />
          <span>Router: {formatRouterMode(routerMode)}</span>
        </div>
      </div>
    </div>
  {/if}
</div>

<style>
  .status-band {
    display: flex;
    align-items: center;
    gap: 24px;
    padding: 12px 20px;
    background: var(--bg-secondary, #1e293b);
    border-bottom: 1px solid var(--border-subtle, #374151);
    font-size: 13px;
    font-weight: 500;
    color: var(--text-secondary, #d1d5db);
    overflow-x: hidden;
    min-height: 48px;
  }

  .ticker-wrapper {
    display: flex;
    gap: 24px;
    animation: ticker-scroll 20s linear infinite;
    white-space: nowrap;
    will-change: transform;
  }

  .ticker-wrapper > * {
    flex-shrink: 0;
  }

  .ticker-wrapper:hover {
    animation-play-state: paused;
  }

  @keyframes ticker-scroll {
    0% {
      transform: translateX(0);
    }
    100% {
      transform: translateX(-50%);
    }
  }

  .loading {
    color: var(--text-muted, #6b7280);
    font-style: italic;
  }

  .clickable {
    cursor: pointer;
    transition: opacity 0.2s, color 0.2s;
  }

  .clickable:hover {
    opacity: 0.7;
    color: var(--text-primary, #fff);
  }

  .current-session {
    display: flex;
    align-items: center;
    gap: 8px;
    font-weight: 600;
  }

  .current-session .label {
    color: var(--text-muted, #9ca3af);
  }

  .current-session .current {
    color: var(--accent-success, #10b981);
  }

  .sessions {
    display: flex;
    gap: 20px;
  }

  .session-item {
    display: flex;
    align-items: center;
    gap: 6px;
  }

  .session-item.active {
    color: var(--text-primary, #fff);
  }

  .dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
  }

  .divider {
    color: var(--border-subtle, #374151);
  }

  .regime {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .regime-dot {
    width: 10px;
    height: 10px;
    border-radius: 2px;
  }

  .metrics {
    display: flex;
    gap: 24px;
    margin-left: auto;
  }

  .metric {
    display: flex;
    align-items: center;
    gap: 6px;
  }

  .metric.profit {
    color: var(--accent-success, #10b981);
  }

  .metric.loss {
    color: var(--accent-danger, #ef4444);
  }

  .mode-indicators {
    display: flex;
    gap: 24px;
  }

  .mode-item {
    display: flex;
    align-items: center;
    gap: 6px;
    color: var(--text-secondary, #d1d5db);
  }
</style>
