<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import {
    Bot,
    DollarSign,
    Shield,
    Route,
    Sun,
    Moon,
    WifiOff,
    TrendingUp,
    Gauge,
    Target,
    CheckCircle2,
    Circle
  } from 'lucide-svelte';
  import {
    getAllSessions,
    getCurrentSessionInfo,
    getMarketState,
    getTradingMetrics,
    getRiskSettings,
    getRouterSettings
  } from '$lib/api';
  import { activeCanvasStore } from '../stores/canvasStore';

  // Svelte 5 Runes
  let loading = $state(true);
  let sessions = $state<Record<string, { active: boolean; name: string }>>({});
  let sessionsError = $state(false);
  let regime = $state('UNKNOWN');
  let activeBots = $state(0);
  let dailyPnl = $state(0);
  let dailyPnlPrev = $state(0);
  let winRate = $state(0);
  let openPositions = $state(0);
  let tradesToday = $state(0);
  let currentSession = $state('CLOSED');
  let currentTime = $state('');

  // Risk and Router settings
  let riskMode = $state<'fixed' | 'dynamic' | 'conservative'>('dynamic');
  let routerMode = $state<'auction' | 'priority' | 'round-robin'>('auction');

  // Node health
  let nodeHealth = $state({
    cloudzy: { status: 'online', lastSeen: new Date() },
    contabo: { status: 'online', lastSeen: new Date() },
    local: { status: 'online', lastSeen: new Date() }
  });
  let contaboDegraded = $state(false);

  // Workflow & Challenge
  let workflowCount = $state(0);
  let challengeProgress = $state({ phase: 'qualifying', target: 1000, current: 0 });

  // Animation state
  let pnlFlash = $state<'positive' | 'negative' | null>(null);

  let refreshInterval: ReturnType<typeof setInterval>;
  let timeInterval: ReturnType<typeof setInterval>;

  // Explicit session order for deterministic display (Tokyo/London/NY)
  const SESSION_ORDER = ['ASIAN', 'LONDON', 'NEW_YORK'];

  // Session times in UTC
  const SESSION_TIMES = {
    ASIAN: { start: '00:00', end: '08:00', label: 'Tokyo', timezone: 'JST' },
    LONDON: { start: '08:00', end: '16:00', label: 'London', timezone: 'GMT' },
    NEW_YORK: { start: '13:00', end: '21:00', label: 'NY', timezone: 'EST' }
  };

  // Session city labels
  const SESSION_CITIES = {
    ASIAN: 'Tokyo',
    LONDON: 'London',
    NEW_YORK: 'NY'
  };

  // Derived: formatted P&L
  let formattedPnl = $derived((() => {
    const sign = dailyPnl >= 0 ? '+' : '';
    return `${sign}$${dailyPnl.toFixed(2)}`;
  })());

  // Watch for P&L changes and trigger flash
  $effect(() => {
    if (dailyPnl !== dailyPnlPrev && !loading) {
      pnlFlash = dailyPnl >= 0 ? 'positive' : 'negative';
      dailyPnlPrev = dailyPnl;
      // Clear flash after 100ms
      setTimeout(() => {
        pnlFlash = null;
      }, 100);
    }
  });

  // Helper: Get current active trading session
  function getCurrentSession(): string {
    for (const sessionKey of SESSION_ORDER) {
      if (sessions[sessionKey]?.active) {
        return sessionKey;
      }
    }
    return 'CLOSED';
  }

  // Get session local time
  function getSessionLocalTime(session: string): string {
    const now = new Date();
    const offsets: Record<string, number> = {
      ASIAN: 9,    // JST = UTC+9
      LONDON: 0,   // GMT = UTC+0 (winter)
      NEW_YORK: -5 // EST = UTC-5 (winter)
    };
    const offset = offsets[session] ?? 0;
    const localTime = new Date(now.getTime() + offset * 60 * 60 * 1000);
    return localTime.toLocaleTimeString('en-US', {
      hour: '2-digit',
      minute: '2-digit',
      hour12: false
    });
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
        if (sessionInfo) {
          currentSession = sessionInfo.session;
        }
      } catch (e) {
        // Using fallback
      }

      // Fetch sessions using helper
      const sessionsData = await getAllSessions();
      if (sessionsData && Object.keys(sessionsData).length > 0) {
        sessions = sessionsData;
        if (!currentSession || currentSession === 'CLOSED') {
          currentSession = getCurrentSession();
        }
        sessionsError = false;
      } else {
        // No mock data — indicate error state
        sessions = {};
        sessionsError = true;
      }

      // Fetch market state using helper
      try {
        const marketData = await getMarketState();
        if (marketData?.regime) {
          regime = determineRegime(marketData.regime);
        }
      } catch (e) {
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
        // Metrics may not be available
      }

      // Fetch risk settings
      try {
        const riskData = await getRiskSettings();
        if (riskData?.riskMode) {
          riskMode = riskData.riskMode;
        }
      } catch (e) {
        // Risk settings may not be available
      }

      // Fetch router settings
      try {
        const routerData = await getRouterSettings();
        if (routerData?.mode) {
          routerMode = routerData.mode;
        }
      } catch (e) {
        // Router settings may not be available
      }

      // Simulate node health check (in real app, this would be an API call)
      // For now, simulate node_backend sometimes being unreachable
      const randomCheck = Math.random();
      if (randomCheck < 0.1) {
        contaboDegraded = true;
        nodeHealth.contabo.status = 'degraded';
      } else {
        contaboDegraded = false;
        nodeHealth.contabo.status = 'online';
      }
      nodeHealth.contabo.lastSeen = new Date();

      // Simulate workflow count
      workflowCount = Math.floor(Math.random() * 5) + 1;

      // Simulate challenge progress
      challengeProgress = {
        phase: 'qualifying',
        target: 1000,
        current: Math.floor(Math.random() * 500) + 100
      };
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
      'HIGH_CHAOS': 'High Vol',
      'BREAKOUT_PRIME': 'Breakout',
      'UNCERTAIN': 'Uncertain'
    };
    return labels[r] || r;
  }

  // Helper: Get session color
  function getSessionColor(active: boolean): string {
    return active ? '#00c896' : '#4a5568';
  }

  // Helper: Get regime color
  function getRegimeColor(r: string): string {
    const colors: Record<string, string> = {
      'TREND_STABLE': '#00c896',
      'RANGE_STABLE': '#00d4ff',
      'HIGH_CHAOS': '#ff3b3b',
      'BREAKOUT_PRIME': '#f0a500',
      'UNCERTAIN': '#4a5568'
    };
    return colors[r] || '#4a5568';
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

  // Helper: Format risk mode for display
  function formatRiskMode(mode: string): string {
    const labels: Record<string, string> = {
      fixed: 'Fixed',
      dynamic: 'Dynamic',
      conservative: 'Conservative'
    };
    return labels[mode] || mode;
  }

  // Navigation handlers — use activeCanvasStore as single source of truth (AC 12-5-7)
  function navigateToLiveTrading() {
    activeCanvasStore.setActiveCanvas('live-trading');
  }

  function navigateToPortfolio() {
    activeCanvasStore.setActiveCanvas('portfolio');
  }

  function navigateToRisk() {
    activeCanvasStore.setActiveCanvas('risk');
  }

  // Router mode label → risk canvas (AC 12-5-7)
  function navigateToRouter() {
    activeCanvasStore.setActiveCanvas('risk');
  }

  function showNodeStatus() {
    // In a full implementation, this would open a node status overlay
    // For now, we'll just log and could trigger a modal
    console.log('Node status:', nodeHealth);
  }

  // Get node status indicator
  function getNodeStatusColor(status: string): string {
    switch (status) {
      case 'online': return '#00c896';
      case 'degraded': return '#f0a500';
      case 'offline': return '#ff3b3b';
      default: return '#4a5568';
    }
  }
</script>

<div class="status-band" role="status" aria-live="polite" aria-atomic="false">
  {#if loading}
    <span class="loading">Loading...</span>
  {:else}
    <div class="ticker-wrapper">
      <!-- ===== SESSION CLOCKS (Tokyo/London/NY) ===== -->
      <div class="segment session-clocks clickable" onclick={navigateToLiveTrading} role="button" tabindex="0" onkeypress={(e) => e.key === 'Enter' && navigateToLiveTrading()}>
        <span class="segment-label">Sessions</span>
        {#each SESSION_ORDER as sessionKey}
          {@const sessionData = sessions[sessionKey]}
          {@const isActive = sessionData?.active ?? false}
          <div class="session-clock" class:active={isActive}>
            <span class="city">{SESSION_CITIES[sessionKey as keyof typeof SESSION_CITIES]}</span>
            <span class="time">{getSessionLocalTime(sessionKey)}</span>
            {#if isActive}
              <Sun size={10} class="session-icon sun" />
            {:else}
              <Moon size={10} class="session-icon moon" />
            {/if}
          </div>
        {/each}
      </div>

      <div class="divider">|</div>

      <!-- ===== ACTIVE BOTS ===== -->
      <div class="segment bots clickable" onclick={navigateToPortfolio} role="button" tabindex="0" onkeypress={(e) => e.key === 'Enter' && navigateToPortfolio()}>
        <Bot size={12} />
        <span class="metric-value">{activeBots}</span>
        <span class="metric-label">Bots</span>
      </div>

      <div class="divider">|</div>

      <!-- ===== DAILY P&L ===== -->
      <div class="segment pnl clickable" onclick={navigateToPortfolio} role="button" tabindex="0" onkeypress={(e) => e.key === 'Enter' && navigateToPortfolio()}>
        <DollarSign size={12} />
        <span
          class="metric-value"
          class:profit={dailyPnl >= 0}
          class:loss={dailyPnl < 0}
          class:flash-positive={pnlFlash === 'positive'}
          class:flash-negative={pnlFlash === 'negative'}
        >
          {formattedPnl}
        </span>
        {#if contaboDegraded}
          <span class="stale-label">[stale]</span>
        {/if}
      </div>

      <div class="divider">|</div>

      <!-- ===== NODE HEALTH DOTS ===== -->
      <div class="segment nodes clickable" onclick={showNodeStatus} role="button" tabindex="0" onkeypress={(e) => e.key === 'Enter' && showNodeStatus()}>
        <span class="segment-label">Nodes</span>
        <div class="node-dots">
          <!-- node_trading -->
          <div class="node-dot" title="node_trading: Trading Node">
            <Circle size={8} fill={getNodeStatusColor(nodeHealth.cloudzy.status)} stroke="none" />
          </div>
          <span class="node-sep">·</span>
          <!-- node_backend -->
          <div class="node-dot" class:degraded={contaboDegraded} title={contaboDegraded ? 'node_backend: Unreachable' : 'node_backend: Agent Node'}>
            {#if contaboDegraded}
              <WifiOff size={10} class="node-icon degraded" />
            {:else}
              <Circle size={8} fill={getNodeStatusColor(nodeHealth.contabo.status)} stroke="none" />
            {/if}
          </div>
          <span class="node-sep">·</span>
          <!-- Local -->
          <div class="node-dot" title="Local: Development">
            <Circle size={8} fill={getNodeStatusColor(nodeHealth.local.status)} stroke="none" />
          </div>
        </div>
      </div>

      <div class="divider">|</div>

      <!-- ===== WORKFLOW COUNT ===== -->
      <div class="segment workflow">
        <Gauge size={12} />
        <span class="metric-value">{workflowCount}</span>
        <span class="metric-label">Workflows</span>
      </div>

      <div class="divider">|</div>

      <!-- ===== CHALLENGE PROGRESS ===== -->
      <div class="segment challenge">
        <Target size={12} />
        <span class="metric-value">{challengeProgress.current}</span>
        <span class="metric-label">/ {challengeProgress.target}</span>
        <CheckCircle2 size={10} class="challenge-icon" />
      </div>

      <div class="divider">|</div>

      <!-- ===== REGIME ===== -->
      <div class="segment regime clickable" onclick={navigateToLiveTrading} role="button" tabindex="0" onkeypress={(e) => e.key === 'Enter' && navigateToLiveTrading()}>
        <TrendingUp size={12} />
        <span class="regime-dot" style="background: {getRegimeColor(regime)}"></span>
        <span class="metric-value">{formatRegime(regime)}</span>
      </div>

      <div class="divider">|</div>

      <!-- ===== RISK MODE ===== -->
      <div class="segment risk clickable" onclick={navigateToRisk} role="button" tabindex="0" onkeypress={(e) => e.key === 'Enter' && navigateToRisk()}>
        <Shield size={12} />
        <span class="metric-label">Risk:</span>
        <span class="metric-value">{formatRiskMode(riskMode)}</span>
      </div>

      <div class="divider">|</div>

      <!-- ===== ROUTER MODE (AC 12-5-7) ===== -->
      <div class="segment router clickable" onclick={navigateToRouter} role="button" tabindex="0" onkeypress={(e) => e.key === 'Enter' && navigateToRouter()}>
        <Route size={12} />
        <span class="metric-label">Router:</span>
        <span class="metric-value">{routerMode}</span>
      </div>

      <!-- ===== DUPLICATE FOR SEAMLESS LOOP ===== -->
      <div class="divider">|</div>

      <!-- ===== SESSION CLOCKS (Tokyo/London/NY) ===== -->
      <div class="segment session-clocks clickable" onclick={navigateToLiveTrading} role="button" tabindex="0" onkeypress={(e) => e.key === 'Enter' && navigateToLiveTrading()}>
        <span class="segment-label">Sessions</span>
        {#each SESSION_ORDER as sessionKey}
          {@const sessionData = sessions[sessionKey]}
          {@const isActive = sessionData?.active ?? false}
          <div class="session-clock" class:active={isActive}>
            <span class="city">{SESSION_CITIES[sessionKey as keyof typeof SESSION_CITIES]}</span>
            <span class="time">{getSessionLocalTime(sessionKey)}</span>
            {#if isActive}
              <Sun size={10} class="session-icon sun" />
            {:else}
              <Moon size={10} class="session-icon moon" />
            {/if}
          </div>
        {/each}
      </div>

      <div class="divider">|</div>

      <!-- ===== ACTIVE BOTS ===== -->
      <div class="segment bots clickable" onclick={navigateToPortfolio} role="button" tabindex="0" onkeypress={(e) => e.key === 'Enter' && navigateToPortfolio()}>
        <Bot size={12} />
        <span class="metric-value">{activeBots}</span>
        <span class="metric-label">Bots</span>
      </div>

      <div class="divider">|</div>

      <!-- ===== DAILY P&L ===== -->
      <div class="segment pnl clickable" onclick={navigateToPortfolio} role="button" tabindex="0" onkeypress={(e) => e.key === 'Enter' && navigateToPortfolio()}>
        <DollarSign size={12} />
        <span
          class="metric-value"
          class:profit={dailyPnl >= 0}
          class:loss={dailyPnl < 0}
          class:flash-positive={pnlFlash === 'positive'}
          class:flash-negative={pnlFlash === 'negative'}
        >
          {formattedPnl}
        </span>
        {#if contaboDegraded}
          <span class="stale-label">[stale]</span>
        {/if}
      </div>

      <div class="divider">|</div>

      <!-- ===== NODE HEALTH DOTS ===== -->
      <div class="segment nodes clickable" onclick={showNodeStatus} role="button" tabindex="0" onkeypress={(e) => e.key === 'Enter' && showNodeStatus()}>
        <span class="segment-label">Nodes</span>
        <div class="node-dots">
          <!-- node_trading -->
          <div class="node-dot" title="node_trading: Trading Node">
            <Circle size={8} fill={getNodeStatusColor(nodeHealth.cloudzy.status)} stroke="none" />
          </div>
          <span class="node-sep">·</span>
          <!-- node_backend -->
          <div class="node-dot" class:degraded={contaboDegraded} title={contaboDegraded ? 'node_backend: Unreachable' : 'node_backend: Agent Node'}>
            {#if contaboDegraded}
              <WifiOff size={10} class="node-icon degraded" />
            {:else}
              <Circle size={8} fill={getNodeStatusColor(nodeHealth.contabo.status)} stroke="none" />
            {/if}
          </div>
          <span class="node-sep">·</span>
          <!-- Local -->
          <div class="node-dot" title="Local: Development">
            <Circle size={8} fill={getNodeStatusColor(nodeHealth.local.status)} stroke="none" />
          </div>
        </div>
      </div>

      <div class="divider">|</div>

      <!-- ===== WORKFLOW COUNT ===== -->
      <div class="segment workflow">
        <Gauge size={12} />
        <span class="metric-value">{workflowCount}</span>
        <span class="metric-label">Workflows</span>
      </div>

      <div class="divider">|</div>

      <!-- ===== CHALLENGE PROGRESS ===== -->
      <div class="segment challenge">
        <Target size={12} />
        <span class="metric-value">{challengeProgress.current}</span>
        <span class="metric-label">/ {challengeProgress.target}</span>
        <CheckCircle2 size={10} class="challenge-icon" />
      </div>

      <div class="divider">|</div>

      <!-- ===== REGIME ===== -->
      <div class="segment regime clickable" onclick={navigateToLiveTrading} role="button" tabindex="0" onkeypress={(e) => e.key === 'Enter' && navigateToLiveTrading()}>
        <TrendingUp size={12} />
        <span class="regime-dot" style="background: {getRegimeColor(regime)}"></span>
        <span class="metric-value">{formatRegime(regime)}</span>
      </div>

      <div class="divider">|</div>

      <!-- ===== RISK MODE ===== -->
      <div class="segment risk clickable" onclick={navigateToRisk} role="button" tabindex="0" onkeypress={(e) => e.key === 'Enter' && navigateToRisk()}>
        <Shield size={12} />
        <span class="metric-label">Risk:</span>
        <span class="metric-value">{formatRiskMode(riskMode)}</span>
      </div>

      <div class="divider">|</div>

      <!-- ===== ROUTER MODE (AC 12-5-7) ===== -->
      <div class="segment router clickable" onclick={navigateToRouter} role="button" tabindex="0" onkeypress={(e) => e.key === 'Enter' && navigateToRouter()}>
        <Route size={12} />
        <span class="metric-label">Router:</span>
        <span class="metric-value">{routerMode}</span>
      </div>
    </div>
  {/if}
</div>

<style>
  .status-band {
    grid-area: statusband;
    height: 32px;
    display: flex;
    align-items: center;
    /* Tier 1 Frosted Terminal Glass */
    background: rgba(8, 13, 20, 0.08);
    backdrop-filter: blur(24px) saturate(160%);
    -webkit-backdrop-filter: blur(24px) saturate(160%);
    border-bottom: 1px solid rgba(0, 212, 255, 0.1);
    font-family: 'JetBrains Mono', 'Fragment Mono', 'Fira Code', monospace;
    font-size: 11px;
    font-weight: 400;
    color: #e2e8f0;
    overflow: hidden;
    z-index: 90;
  }

  .ticker-wrapper {
    display: flex;
    align-items: center;
    gap: 12px;
    animation: ticker-scroll 60s linear infinite;
    white-space: nowrap;
    will-change: transform;
    padding-left: 100%;
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
    color: #4a5568;
    font-style: italic;
    padding-left: 20px;
  }

  .segment {
    display: flex;
    align-items: center;
    gap: 4px;
    flex-shrink: 0;
  }

  .segment-label {
    color: #718096;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-right: 4px;
  }

  .clickable {
    cursor: pointer;
    transition: opacity 0.2s ease;
  }

  .clickable:hover {
    opacity: 0.7;
  }

  .clickable:active {
    opacity: 0.5;
  }

  /* Session Clocks */
  .session-clocks {
    gap: 8px;
  }

  .session-clock {
    display: flex;
    align-items: center;
    gap: 3px;
    padding: 2px 6px;
    background: rgba(0, 0, 0, 0.2);
    border-radius: 4px;
  }

  .session-clock.active {
    background: rgba(0, 200, 150, 0.15);
    border: 1px solid rgba(0, 200, 150, 0.3);
  }

  .session-clock .city {
    color: #a0aec0;
    font-size: 10px;
  }

  .session-clock .time {
    color: #e2e8f0;
    font-size: 11px;
    font-weight: 500;
  }

  .session-clock.active .time {
    color: #00c896;
  }

  .session-icon {
    flex-shrink: 0;
  }

  .session-icon.sun {
    color: #f0a500;
  }

  .session-icon.moon {
    color: #4a5568;
  }

  /* Metrics */
  .metric-value {
    color: #e2e8f0;
    font-weight: 500;
  }

  .metric-label {
    color: #718096;
    font-size: 10px;
  }

  .segment.pnl .metric-value.profit {
    color: #00c896;
  }

  .segment.pnl .metric-value.loss {
    color: #ff3b3b;
  }

  .stale-label {
    color: #f0a500;
    font-size: 9px;
    font-style: italic;
  }

  /* Flash animations */
  .flash-positive {
    animation: flash-green 100ms ease-out;
  }

  .flash-negative {
    animation: flash-red 100ms ease-out;
  }

  @keyframes flash-green {
    0% {
      color: #00c896;
      text-shadow: 0 0 8px rgba(0, 200, 150, 0.8);
    }
    100% {
      color: #00c896;
      text-shadow: none;
    }
  }

  @keyframes flash-red {
    0% {
      color: #ff3b3b;
      text-shadow: 0 0 8px rgba(255, 59, 59, 0.8);
    }
    100% {
      color: #ff3b3b;
      text-shadow: none;
    }
  }

  /* Node Health Dots */
  .node-dots {
    display: flex;
    align-items: center;
    gap: 2px;
  }

  .node-dot {
    display: flex;
    align-items: center;
  }

  .node-dot.degraded .node-icon.degraded {
    color: #ff3b3b;
  }

  .node-sep {
    color: #4a5568;
    margin: 0 2px;
  }

  /* Challenge */
  .challenge-icon {
    color: #00c896;
    margin-left: 2px;
  }

  /* Regime */
  .regime-dot {
    width: 6px;
    height: 6px;
    border-radius: 2px;
    flex-shrink: 0;
  }

  .divider {
    color: rgba(0, 212, 255, 0.2);
    flex-shrink: 0;
    font-size: 10px;
  }

  /* Accessibility */
  .clickable:focus {
    outline: 1px solid #00d4ff;
    outline-offset: 2px;
  }

  .clickable:focus:not(:focus-visible) {
    outline: none;
  }
</style>
