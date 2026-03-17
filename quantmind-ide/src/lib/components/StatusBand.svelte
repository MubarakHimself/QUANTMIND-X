<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { Bot, DollarSign, Percent, Shield, Route, TrendingUp, Activity, Target, Clock } from 'lucide-svelte';
  import { getAllSessions, getCurrentSessionInfo, getMarketState, getTradingMetrics, getRiskSettings, getRouterSettings } from '$lib/api';
  import { navigationStore } from '../stores/navigationStore';

  // State
  let loading = true;
  let sessions: Record<string, { active: boolean; name: string }> = {};
  let sessionsError = false;
  let regime = 'UNKNOWN';
  let activeBots = 0;
  let dailyPnl = 0;
  let winRate = 0;
  let openPositions = 0;
  let tradesToday = 0;
  let currentSession = 'CLOSED';
  let currentTime = '';

  // Risk and Router settings
  let riskMode: 'fixed' | 'dynamic' | 'conservative' = 'dynamic';
  let routerMode: 'auction' | 'priority' | 'round-robin' = 'auction';

  let refreshInterval: ReturnType<typeof setInterval>;
  let timeInterval: ReturnType<typeof setInterval>;

  // Explicit session order for deterministic display
  const SESSION_ORDER = ['ASIAN', 'LONDON', 'NEW_YORK', 'OVERLAP'];

  // Session times in UTC
  const SESSION_TIMES = {
    ASIAN: { start: '00:00', end: '08:00', label: '00:00-08:00 UTC' },
    LONDON: { start: '08:00', end: '16:00', label: '08:00-16:00 UTC' },
    NEW_YORK: { start: '13:00', end: '21:00', label: '13:00-21:00 UTC' },
    OVERLAP: { start: '13:00', end: '16:00', label: '13:00-16:00 UTC' }
  };

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
    // Update time every second
    currentTime = getCurrentTime();
    timeInterval = setInterval(() => {
      currentTime = getCurrentTime();
    }, 1000);
    // Refresh data every 5 seconds
    refreshInterval = setInterval(fetchData, 5000);
  });

  onDestroy(() => {
    if (refreshInterval) clearInterval(refreshInterval);
    if (timeInterval) clearInterval(timeInterval);
  });

  async function fetchData() {
    try {
      // Fetch current session info with timing
      try {
        const sessionInfo = await getCurrentSessionInfo();
        console.log('[StatusBand] Session info:', sessionInfo);
        if (sessionInfo) {
          currentSession = sessionInfo.session;
        }
      } catch (e) {
        console.log('[StatusBand] Could not get session info, using fallback');
      }

      // Fetch sessions using helper
      const sessionsData = await getAllSessions();
      console.log('[StatusBand] Sessions data:', sessionsData);
      if (sessionsData && Object.keys(sessionsData).length > 0) {
        sessions = sessionsData;
        if (!currentSession || currentSession === 'CLOSED') {
          currentSession = getCurrentSession();
        }
        sessionsError = false;
      } else {
        console.log('[StatusBand] No sessions data, using fallback');
        // Fallback mock data for development
        sessions = {
          ASIAN: { active: false, name: 'Asian Session' },
          LONDON: { active: true, name: 'London Session' },
          NEW_YORK: { active: false, name: 'New York Session' },
          OVERLAP: { active: false, name: 'London/NY Overlap' },
          CLOSED: { active: false, name: 'Market Closed' }
        };
        currentSession = 'LONDON';
      }

      // Fetch market state using helper
      try {
        const marketData = await getMarketState();
        if (marketData) {
          // Extract regime from the nested structure
          if (marketData.regime) {
            regime = determineRegime(marketData.regime);
          }
        }
      } catch (e) {
        console.log('[StatusBand] Market data not available');
        regime = 'UNCERTAIN';
      }

      // Fetch trading metrics using helper
      try {
        const metricsData = await getTradingMetrics();
        if (metricsData) {
          activeBots = metricsData.active_bots || 0;
          dailyPnl = metricsData.daily_pnl || 0;
          winRate = (metricsData.win_rate || 0) * 100;
          openPositions = metricsData.active_positions || 0;
          tradesToday = metricsData.total_trades || 0;
        }
      } catch (e) {
        console.log('[StatusBand] Trading metrics not available');
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

  // Helper: Format regime for user-friendly display
  function formatRegime(r: string): string {
    const labels: Record<string, string> = {
      'TREND_STABLE': 'Trending',
      'RANGE_STABLE': 'Range Bound',
      'HIGH_CHAOS': 'High Volatility',
      'BREAKOUT_PRIME': 'Breakout',
      'UNCERTAIN': 'Uncertain'
    };
    return labels[r] || r;
  }

  // Helper: Format session name for display
  function formatSessionName(session: string): string {
    const labels: Record<string, string> = {
      'ASIAN': 'Asian Session',
      'LONDON': 'London Session',
      'NEW_YORK': 'New York Session',
      'OVERLAP': 'London/NY Overlap',
      'CLOSED': 'Market Closed'
    };
    return labels[session] || session;
  }

  // Helper: Get session time display
  function getSessionTimeDisplay(session: string): string {
    return SESSION_TIMES[session as keyof typeof SESSION_TIMES]?.label || '';
  }

  // Helper: Get current time formatted
  function getCurrentTime(): string {
    const now = new Date();
    return now.toLocaleTimeString('en-US', {
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
      hour12: false
    });
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
      <!-- CURRENT SESSION BLOCK -->
      <div class="info-block">
        <div class="current-session clickable" on:click={() => navigationStore.navigateToView('journal')}>
          <Clock size={14} />
          <span class="label">Trading:</span>
          <span class="session-name current">{formatSessionName(currentSession)}</span>
        </div>
        <div class="current-time">
          <Clock size={14} />
          <span class="time-display">{currentTime}</span>
          <span class="utc-label">UTC</span>
        </div>
      </div>

      <div class="divider">|</div>

      <!-- ALL SESSIONS WITH TIMES -->
      <div class="sessions-block">
        {#each SESSION_ORDER as sessionKey}
          {#if sessions[sessionKey]}
            <div class="session-item" class:active={sessions[sessionKey].active}>
              <span class="dot" style="background: {getSessionColor(sessions[sessionKey].active)}"></span>
              <span class="session-name">{formatSessionName(sessionKey)}</span>
              <span class="session-time-label">{getSessionTimeDisplay(sessionKey)}</span>
            </div>
          {:else}
            <div class="session-item">
              <span class="dot" style="background: #6b7280"></span>
              <span class="session-name">{formatSessionName(sessionKey)}</span>
              <span class="session-time-label">{getSessionTimeDisplay(sessionKey)}</span>
            </div>
          {/if}
        {/each}
      </div>

      <div class="divider">|</div>

      <!-- REGIME -->
      <div class="regime clickable" on:click={() => navigationStore.navigateToView('journal')}>
        <span class="regime-dot" style="background: {getRegimeColor(regime)}"></span>
        <span>{formatRegime(regime)}</span>
      </div>

      <div class="divider">|</div>

      <!-- METRICS -->
      <div class="metrics">
        <div class="metric clickable" on:click={() => navigationStore.navigateToView('journal')}>
          <Bot size={14} />
          <span>{activeBots} Bots</span>
        </div>
        <div class="metric clickable" class:profit={dailyPnl >= 0} class:loss={dailyPnl < 0} on:click={() => navigationStore.navigateToView('journal')}>
          <DollarSign size={14} />
          <span>{formatPnl(dailyPnl)}</span>
        </div>
        <div class="metric clickable" on:click={() => navigationStore.navigateToView('journal')}>
          <Percent size={14} />
          <span>{winRate.toFixed(0)}% WR</span>
        </div>
        <div class="metric clickable" on:click={() => navigationStore.navigateToView('journal')}>
          <Target size={14} />
          <span>{openPositions} Pos</span>
        </div>
        <div class="metric clickable" on:click={() => navigationStore.navigateToView('journal')}>
          <Activity size={14} />
          <span>{tradesToday} Trades</span>
        </div>
      </div>

      <div class="divider">|</div>

      <!-- MODE INDICATORS -->
      <div class="mode-indicators clickable" on:click={() => navigationStore.navigateToView('settings')}>
        <div class="mode-item" title="Risk Mode">
          <Shield size={14} />
          <span>Risk: {formatRiskMode(riskMode)}</span>
        </div>
        <div class="mode-item" title="Router Mode">
          <Route size={14} />
          <span>Router: {formatRouterMode(routerMode)}</span>
        </div>
      </div>

      <!-- DUPLICATE FOR SEAMLESS LOOP -->
      <div class="divider">|</div>

      <!-- CURRENT SESSION BLOCK -->
      <div class="info-block">
        <div class="current-session clickable" on:click={() => navigationStore.navigateToView('journal')}>
          <Clock size={14} />
          <span class="label">Trading:</span>
          <span class="session-name current">{formatSessionName(currentSession)}</span>
        </div>
        <div class="current-time">
          <Clock size={14} />
          <span class="time-display">{currentTime}</span>
          <span class="utc-label">UTC</span>
        </div>
      </div>

      <div class="divider">|</div>

      <!-- ALL SESSIONS WITH TIMES -->
      <div class="sessions-block">
        {#each SESSION_ORDER as sessionKey}
          {#if sessions[sessionKey]}
            <div class="session-item" class:active={sessions[sessionKey].active}>
              <span class="dot" style="background: {getSessionColor(sessions[sessionKey].active)}"></span>
              <span class="session-name">{formatSessionName(sessionKey)}</span>
              <span class="session-time-label">{getSessionTimeDisplay(sessionKey)}</span>
            </div>
          {:else}
            <div class="session-item">
              <span class="dot" style="background: #6b7280"></span>
              <span class="session-name">{formatSessionName(sessionKey)}</span>
              <span class="session-time-label">{getSessionTimeDisplay(sessionKey)}</span>
            </div>
          {/if}
        {/each}
      </div>

      <div class="divider">|</div>

      <!-- REGIME -->
      <div class="regime clickable" on:click={() => navigationStore.navigateToView('journal')}>
        <span class="regime-dot" style="background: {getRegimeColor(regime)}"></span>
        <span>{formatRegime(regime)}</span>
      </div>

      <div class="divider">|</div>

      <!-- METRICS -->
      <div class="metrics">
        <div class="metric clickable" on:click={() => navigationStore.navigateToView('journal')}>
          <Bot size={14} />
          <span>{activeBots} Bots</span>
        </div>
        <div class="metric clickable" class:profit={dailyPnl >= 0} class:loss={dailyPnl < 0} on:click={() => navigationStore.navigateToView('journal')}>
          <DollarSign size={14} />
          <span>{formatPnl(dailyPnl)}</span>
        </div>
        <div class="metric clickable" on:click={() => navigationStore.navigateToView('journal')}>
          <Target size={14} />
          <span>{openPositions} Pos</span>
        </div>
        <div class="metric clickable" on:click={() => navigationStore.navigateToView('trading')}>
          <Activity size={14} />
          <span>{tradesToday} Trades</span>
        </div>
      </div>

      <div class="divider">|</div>

      <!-- MODE INDICATORS -->
      <div class="mode-indicators clickable" on:click={() => navigationStore.navigateToView('settings')}>
        <div class="mode-item" title="Risk Mode">
          <Shield size={14} />
          <span>Risk: {formatRiskMode(riskMode)}</span>
        </div>
        <div class="mode-item" title="Router Mode">
          <Route size={14} />
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
    padding: 8px 0;
    background: var(--bg-secondary, #1e293b);
    border-bottom: 1px solid var(--border-subtle, #374151);
    font-size: 12px;
    font-weight: 500;
    color: var(--text-secondary, #d1d5db);
    overflow: hidden;
    min-height: 40px;
    height: 40px;
  }

  .ticker-wrapper {
    display: flex;
    align-items: center;
    gap: 16px;
    animation: ticker-scroll 40s linear infinite;
    white-space: nowrap;
    will-change: transform;
    padding-left: 100%;
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
      transform: translateX(-100%);
    }
  }

  .loading {
    color: var(--text-muted, #6b7280);
    font-style: italic;
    padding-left: 20px;
  }

  .clickable {
    cursor: pointer;
    transition: opacity 0.2s, color 0.2s;
  }

  .clickable:hover {
    opacity: 0.7;
    color: var(--text-primary, #fff);
  }

  .info-block {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .current-session {
    display: flex;
    align-items: center;
    gap: 4px;
    font-weight: 600;
  }

  .current-session .label {
    color: var(--text-muted, #9ca3af);
    font-size: 11px;
  }

  .current-session .current {
    color: var(--accent-success, #10b981);
    font-size: 11px;
  }

  .current-time {
    display: flex;
    align-items: center;
    gap: 4px;
    font-weight: 600;
    font-family: monospace;
  }

  .current-time .time-display {
    font-size: 12px;
    color: var(--text-primary, #fff);
  }

  .current-time .utc-label {
    font-size: 9px;
    color: var(--text-muted, #9ca3af);
    font-weight: 400;
  }

  .sessions-block {
    display: flex;
    gap: 12px;
  }

  .session-item {
    display: flex;
    align-items: center;
    gap: 4px;
  }

  .session-item .session-name {
    font-size: 11px;
    font-weight: 500;
  }

  .session-item .session-time-label {
    font-size: 9px;
    color: var(--text-muted, #6b7280);
  }

  .session-item.active .session-name {
    color: var(--text-primary, #fff);
  }

  .dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    flex-shrink: 0;
  }

  .divider {
    color: var(--border-subtle, #374151);
    flex-shrink: 0;
    font-size: 10px;
  }

  .regime {
    display: flex;
    align-items: center;
    gap: 4px;
    font-size: 11px;
  }

  .regime-dot {
    width: 8px;
    height: 8px;
    border-radius: 2px;
  }

  .metrics {
    display: flex;
    gap: 12px;
  }

  .metric {
    display: flex;
    align-items: center;
    gap: 4px;
    font-size: 11px;
  }

  .metric.profit {
    color: var(--accent-success, #10b981);
  }

  .metric.loss {
    color: var(--accent-danger, #ef4444);
  }

  .mode-indicators {
    display: flex;
    gap: 12px;
  }

  .mode-item {
    display: flex;
    align-items: center;
    gap: 4px;
    font-size: 11px;
    color: var(--text-secondary, #d1d5db);
  }
</style>
