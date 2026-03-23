<script lang="ts">
  /**
   * Live Trading Canvas — Bloomberg-style morning dashboard
   *
   * Dense tile grid layout. No right-side agent panel.
   * Each tile is clickable → canvas-local sub-page routing.
   * Session clocks update every second via setInterval.
   */
  import { onMount, onDestroy } from 'svelte';
  import BotDetailPage from '$lib/components/live-trading/BotDetailPage.svelte';
  import MorningDigestCard from '$lib/components/live-trading/MorningDigestCard.svelte';
  import NewsFeedTile from '$lib/components/live-trading/NewsFeedTile.svelte';
  import LiveTradingMailPage from '$lib/components/live-trading/LiveTradingMailPage.svelte';
  import DepartmentKanban from '$lib/components/department-kanban/DepartmentKanban.svelte';
  import { canvasContextService } from '$lib/services/canvasContextService';
  import {
    selectedBotId,
    connectTradingWS,
    disconnectTradingWS,
    fetchActiveBots,
    wsConnected,
    wsError,
    activeBots,
    botCount
  } from '$lib/stores/trading';
  import { startHealthMonitoring, stopHealthMonitoring, nodeHealthState } from '$lib/stores/node-health';
  import {
    Activity,
    Clock,
    TrendingUp,
    DollarSign,
    Wifi,
    WifiOff,
    Newspaper,
    RefreshCw,
    ArrowLeft,
    AlertCircle,
    Server,
    Kanban,
    Mail
  } from 'lucide-svelte';

  // ─── Sub-page routing (canvas-local, no URL change) ───────────────────────
  type SubPage = 'bot-detail' | 'positions' | 'news' | 'dept-kanban' | 'dept-mail' | null;
  let currentSubPage = $state<SubPage>(null);

  // ─── Session clock state ──────────────────────────────────────────────────
  let clockInterval: ReturnType<typeof setInterval> | null = null;
  let tokyoTime = $state('--:--:--');
  let londonTime = $state('--:--:--');
  let newYorkTime = $state('--:--:--');
  let tokyoOpen = $state(false);
  let londonOpen = $state(false);
  let newYorkOpen = $state(false);

  // ─── Misc UI state ────────────────────────────────────────────────────────
  let refreshing = $state(false);
  let isMorning = $state(false);

  // ─── Derived display values from trading store ───────────────────────────
  let totalPnl = $derived(
    $activeBots.reduce((sum, b) => sum + (b.current_pnl ?? 0), 0)
  );
  let openPositions = $derived(
    $activeBots.reduce((sum, b) => sum + (b.open_positions ?? 0), 0)
  );
  let largestBot = $derived(
    $activeBots.length > 0
      ? $activeBots.reduce((a, b) => (Math.abs(b.current_pnl) > Math.abs(a.current_pnl) ? b : a))
      : null
  );

  // ─── Helper: format session clock ─────────────────────────────────────────
  function getSessionTime(offsetHours: number): string {
    const now = new Date();
    const utcMs = now.getTime() + now.getTimezoneOffset() * 60_000;
    const local = new Date(utcMs + offsetHours * 3_600_000);
    return local.toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false });
  }

  function isDSTActive(): boolean {
    const jan = new Date(new Date().getFullYear(), 0, 1).getTimezoneOffset();
    const jul = new Date(new Date().getFullYear(), 6, 1).getTimezoneOffset();
    return new Date().getTimezoneOffset() < Math.max(jan, jul);
  }

  function tickClocks() {
    const dst = isDSTActive();
    // Tokyo: JST = UTC+9 (no DST)
    tokyoTime = getSessionTime(9);
    // London: UTC+1 in summer (BST), UTC+0 in winter
    const londonOffset = dst ? 1 : 0;
    londonTime = getSessionTime(londonOffset);
    // New York: EDT = UTC-4 in summer, EST = UTC-5 in winter
    const nyOffset = dst ? -4 : -5;
    newYorkTime = getSessionTime(nyOffset);

    // Session open/closed windows (UTC-based hours)
    const utcHour = new Date().getUTCHours();
    tokyoOpen = utcHour >= 0 && utcHour < 9;
    londonOpen = utcHour >= 8 && utcHour < 17;
    newYorkOpen = utcHour >= 13 && utcHour < 22;

    // Morning: before noon UTC
    isMorning = utcHour >= 0 && utcHour < 12;
  }

  // ─── Lifecycle ────────────────────────────────────────────────────────────
  onMount(async () => {
    try {
      await canvasContextService.loadCanvasContext('live-trading');
    } catch (e) {
      console.error('[LiveTradingCanvas] Failed to load canvas context:', e);
    }

    connectTradingWS();
    await fetchActiveBots();
    startHealthMonitoring();

    tickClocks();
    clockInterval = setInterval(tickClocks, 1000);
  });

  onDestroy(() => {
    disconnectTradingWS();
    stopHealthMonitoring();
    if (clockInterval) clearInterval(clockInterval);
  });

  // ─── Actions ──────────────────────────────────────────────────────────────
  async function refresh() {
    refreshing = true;
    await fetchActiveBots();
    refreshing = false;
  }

  function goBack() {
    currentSubPage = null;
  }

  function navigateTo(page: SubPage) {
    currentSubPage = page;
  }

  function handleBotTileClick() {
    navigateTo('bot-detail');
  }

  // ─── Formatting helpers ───────────────────────────────────────────────────
  function formatPnl(v: number): string {
    const sign = v >= 0 ? '+' : '';
    return `${sign}$${Math.abs(v).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
  }

  function pnlColor(v: number): string {
    return v >= 0 ? '#00c896' : '#ff3b3b';
  }
</script>

<div class="live-trading-canvas" data-dept="trading">

  <!-- ── Canvas Header ───────────────────────────────────────────────────── -->
  <header class="canvas-header">
    <div class="header-left">
      {#if currentSubPage}
        <button class="back-btn" onclick={goBack} title="Back to dashboard">
          <ArrowLeft size={14} />
          <span>Back</span>
        </button>
      {/if}
      <h1>Live Trading</h1>
    </div>

    <div class="header-right">
      <!-- WS status indicator -->
      <div class="ws-indicator" class:connected={$wsConnected} class:error={!!$wsError}>
        {#if $wsConnected}
          <Wifi size={13} />
          <span>Live</span>
        {:else if $wsError}
          <WifiOff size={13} />
          <span>Error</span>
        {:else}
          <span class="icon-spin"><RefreshCw size={13} /></span>
          <span>Connecting</span>
        {/if}
      </div>

      <!-- Dept tasks button (header-only, not a tile) -->
      <button
        class="dept-btn"
        onclick={() => navigateTo('dept-kanban')}
        title="Department Tasks"
      >
        <Kanban size={13} />
        <span>Dept Tasks</span>
      </button>

      <!-- Dept mail button -->
      <button
        class="dept-btn"
        onclick={() => navigateTo('dept-mail')}
        title="Department Mail — Trading Inbox"
      >
        <Mail size={13} />
        <span>Mail</span>
      </button>

      <!-- Refresh -->
      <button class="icon-btn" onclick={refresh} disabled={refreshing} title="Refresh">
        <span class:spinning={refreshing}><RefreshCw size={13} /></span>
      </button>
    </div>
  </header>

  <!-- ── Sub-page: Dept Kanban ───────────────────────────────────────────── -->
  {#if currentSubPage === 'dept-kanban'}
    <div class="subpage-wrap">
      <DepartmentKanban department="trading" onClose={goBack} />
    </div>

  <!-- ── Sub-page: Dept Mail ─────────────────────────────────────────────── -->
  {:else if currentSubPage === 'dept-mail'}
    <div class="subpage-wrap">
      <LiveTradingMailPage />
    </div>

  <!-- ── Sub-page: Bot Detail ────────────────────────────────────────────── -->
  {:else if currentSubPage === 'bot-detail'}
    <div class="subpage-wrap">
      <BotDetailPage />
    </div>

  <!-- ── Sub-page: Positions ─────────────────────────────────────────────── -->
  {:else if currentSubPage === 'positions'}
    <div class="subpage-wrap placeholder-page">
      <div class="placeholder-inner">
        <Activity size={32} strokeWidth={1} />
        <h2>Open Positions</h2>
        <p class="muted">Full positions detail view — coming soon.</p>
      </div>
    </div>

  <!-- ── Sub-page: News ──────────────────────────────────────────────────── -->
  {:else if currentSubPage === 'news'}
    <div class="subpage-wrap">
      <div class="news-expanded">
        <NewsFeedTile />
      </div>
    </div>

  <!-- ── Main Dashboard Grid ─────────────────────────────────────────────── -->
  {:else}
    <div class="canvas-scroll">

      <!-- Morning Digest (conditional on time of day) -->
      {#if isMorning}
        <div class="digest-row">
          <MorningDigestCard />
        </div>
      {/if}

      <!-- Tile Grid -->
      <div class="tile-grid">

        <!-- ① Session Clocks ─────────────────────────────────────────────── -->
        <div class="tile tile--clocks">
          <div class="tile-header">
            <Clock size={14} />
            <span class="tile-title">Market Sessions</span>
          </div>
          <div class="clocks-body">
            <div class="clock-row">
              <div class="clock-label">Tokyo</div>
              <div class="clock-time">{tokyoTime}</div>
              <div class="session-badge" class:open={tokyoOpen} class:closed={!tokyoOpen}>
                {tokyoOpen ? 'OPEN' : 'CLOSED'}
              </div>
            </div>
            <div class="clock-row">
              <div class="clock-label">London</div>
              <div class="clock-time">{londonTime}</div>
              <div class="session-badge" class:open={londonOpen} class:closed={!londonOpen}>
                {londonOpen ? 'OPEN' : 'CLOSED'}
              </div>
            </div>
            <div class="clock-row">
              <div class="clock-label">New York</div>
              <div class="clock-time">{newYorkTime}</div>
              <div class="session-badge" class:open={newYorkOpen} class:closed={!newYorkOpen}>
                {newYorkOpen ? 'OPEN' : 'CLOSED'}
              </div>
            </div>
          </div>
        </div>

        <!-- ② Bot Status ─────────────────────────────────────────────────── -->
        <button class="tile tile--clickable tile--bots" onclick={handleBotTileClick}>
          <div class="tile-header">
            <Activity size={14} />
            <span class="tile-title">Active Bots</span>
            <span class="tile-badge cyan">{$botCount}</span>
          </div>
          <div class="tile-body">
            <div class="big-number cyan">{$botCount}</div>
            <div class="tile-sub">bots running</div>
            {#if $activeBots.length > 0}
              <div class="bot-pairs">
                {#each $activeBots.slice(0, 4) as bot}
                  <span class="pair-chip" class:active={bot.session_active}>{bot.symbol}</span>
                {/each}
                {#if $activeBots.length > 4}
                  <span class="pair-chip muted-chip">+{$activeBots.length - 4}</span>
                {/if}
              </div>
            {:else}
              <div class="empty-hint">No active bots</div>
            {/if}
            <div class="tile-pnl" style="color: {pnlColor(totalPnl)}">
              {formatPnl(totalPnl)}
              <span class="pnl-label">total P&L</span>
            </div>
          </div>
          <div class="tile-cta">View detail →</div>
        </button>

        <!-- ③ Daily P&L ───────────────────────────────────────────────────── -->
        <div class="tile tile--pnl">
          <div class="tile-header">
            <DollarSign size={14} />
            <span class="tile-title">Daily P&L</span>
          </div>
          <div class="tile-body">
            <div class="pnl-hero" style="color: {pnlColor(totalPnl)}">
              {formatPnl(totalPnl)}
            </div>
            {#if $activeBots.length > 0}
              <div class="pnl-breakdown">
                {#each $activeBots.slice(0, 3) as bot}
                  <div class="pnl-row">
                    <span class="pnl-symbol">{bot.symbol}</span>
                    <span class="pnl-val" style="color: {pnlColor(bot.current_pnl)}">
                      {formatPnl(bot.current_pnl)}
                    </span>
                  </div>
                {/each}
              </div>
            {:else}
              <div class="empty-hint">No positions today</div>
            {/if}
          </div>
        </div>

        <!-- ④ Open Positions ─────────────────────────────────────────────── -->
        <button class="tile tile--clickable tile--positions" onclick={() => navigateTo('positions')}>
          <div class="tile-header">
            <TrendingUp size={14} />
            <span class="tile-title">Open Positions</span>
            <span class="tile-badge amber">{openPositions}</span>
          </div>
          <div class="tile-body">
            <div class="big-number amber">{openPositions}</div>
            <div class="tile-sub">open positions</div>
            {#if largestBot}
              <div class="largest-pos">
                <span class="muted-text">Largest:</span>
                <span class="pos-symbol">{largestBot.symbol}</span>
                <span style="color: {pnlColor(largestBot.current_pnl)}">
                  {formatPnl(largestBot.current_pnl)}
                </span>
              </div>
            {/if}
          </div>
          <div class="tile-cta">View positions →</div>
        </button>

        <!-- ⑤ Node Health ─────────────────────────────────────────────────── -->
        <div class="tile tile--health">
          <div class="tile-header">
            <Server size={14} />
            <span class="tile-title">Node Health</span>
            {#if $nodeHealthState.isDegraded}
              <span class="tile-badge red">DEGRADED</span>
            {:else}
              <span class="tile-badge green">OK</span>
            {/if}
          </div>
          <div class="tile-body">
            <div class="node-rows">
              <div class="node-row">
                <div class="node-dot" class:dot-green={$nodeHealthState.contabo.status === 'connected'} class:dot-amber={$nodeHealthState.contabo.status === 'reconnecting'} class:dot-red={$nodeHealthState.contabo.status === 'disconnected'}></div>
                <span class="node-name">Contabo (CZ)</span>
                <span class="node-latency">
                  {$nodeHealthState.contabo.latency_ms > 0 ? `${$nodeHealthState.contabo.latency_ms}ms` : '—'}
                </span>
                <span class="node-status-text" class:text-green={$nodeHealthState.contabo.status === 'connected'} class:text-red={$nodeHealthState.contabo.status !== 'connected'}>
                  {$nodeHealthState.contabo.status.toUpperCase()}
                </span>
              </div>
              <div class="node-row">
                <div class="node-dot" class:dot-green={$nodeHealthState.cloudzy.status === 'connected'} class:dot-amber={$nodeHealthState.cloudzy.status === 'reconnecting'} class:dot-red={$nodeHealthState.cloudzy.status === 'disconnected'}></div>
                <span class="node-name">Cloudzy (LO)</span>
                <span class="node-latency">
                  {$nodeHealthState.cloudzy.latency_ms > 0 ? `${$nodeHealthState.cloudzy.latency_ms}ms` : '—'}
                </span>
                <span class="node-status-text" class:text-green={$nodeHealthState.cloudzy.status === 'connected'} class:text-red={$nodeHealthState.cloudzy.status !== 'connected'}>
                  {$nodeHealthState.cloudzy.status.toUpperCase()}
                </span>
              </div>
            </div>
          </div>
        </div>

        <!-- ⑥ News Feed ───────────────────────────────────────────────────── -->
        <button class="tile tile--clickable tile--news tile--tall" onclick={() => navigateTo('news')}>
          <div class="tile-header">
            <Newspaper size={14} />
            <span class="tile-title">Live News</span>
          </div>
          <div class="news-preview-wrap">
            <NewsFeedTile compact={true} />
          </div>
          <div class="tile-cta">Expand news →</div>
        </button>

        <!-- ⑦ Morning Digest (tile version — always visible as a tile) ───── -->
        <div class="tile tile--digest tile--tall">
          <div class="tile-header">
            <AlertCircle size={14} />
            <span class="tile-title">Morning Digest</span>
          </div>
          <div class="digest-tile-body">
            <MorningDigestCard />
          </div>
        </div>

      </div><!-- /tile-grid -->
    </div><!-- /canvas-scroll -->
  {/if}

</div><!-- /live-trading-canvas -->

<style>
  /* ── Root ─────────────────────────────────────────────────────────────── */
  .live-trading-canvas {
    display: flex;
    flex-direction: column;
    height: 100%;
    width: 100%;
    background: transparent;
    overflow: hidden;
  }

  /* ── Header ───────────────────────────────────────────────────────────── */
  .canvas-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 10px 20px;
    border-bottom: 1px solid rgba(0, 212, 255, 0.1);
    flex-shrink: 0;
    gap: 12px;
  }

  .header-left {
    display: flex;
    align-items: center;
    gap: 10px;
  }

  .header-right {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  h1 {
    font-family: 'JetBrains Mono', monospace;
    font-size: 16px;
    font-weight: 600;
    color: #e8edf5;
    margin: 0;
    letter-spacing: 0.02em;
  }

  /* ── WS status ─────────────────────────────────────────────────────────── */
  .ws-indicator {
    display: flex;
    align-items: center;
    gap: 5px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: #5a6a80;
    padding: 3px 8px;
    border-radius: 4px;
    background: rgba(0, 0, 0, 0.25);
  }

  .ws-indicator.connected {
    color: #00c896;
    background: rgba(0, 200, 150, 0.1);
  }

  .ws-indicator.error {
    color: #ff3b3b;
    background: rgba(255, 59, 59, 0.1);
  }

  /* ── Header buttons ───────────────────────────────────────────────────── */
  .dept-btn {
    display: flex;
    align-items: center;
    gap: 5px;
    padding: 4px 10px;
    background: rgba(0, 212, 255, 0.07);
    border: 1px solid rgba(0, 212, 255, 0.18);
    border-radius: 5px;
    color: #00d4ff;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    cursor: pointer;
    transition: background 0.15s, border-color 0.15s;
  }

  .dept-btn:hover {
    background: rgba(0, 212, 255, 0.14);
    border-color: rgba(0, 212, 255, 0.35);
  }

  .icon-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 26px;
    height: 26px;
    background: rgba(0, 212, 255, 0.07);
    border: 1px solid rgba(0, 212, 255, 0.18);
    border-radius: 4px;
    color: #00d4ff;
    cursor: pointer;
    transition: background 0.15s;
  }

  .icon-btn:hover:not(:disabled) {
    background: rgba(0, 212, 255, 0.15);
  }

  .icon-btn:disabled {
    opacity: 0.45;
    cursor: not-allowed;
  }

  .back-btn {
    display: flex;
    align-items: center;
    gap: 5px;
    padding: 4px 10px;
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 5px;
    color: #e8edf5;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    cursor: pointer;
    transition: background 0.15s;
  }

  .back-btn:hover {
    background: rgba(255, 255, 255, 0.1);
  }

  /* ── Sub-page wrapper ─────────────────────────────────────────────────── */
  .subpage-wrap {
    flex: 1;
    overflow-y: auto;
    padding: 12px 16px;
    width: 100%;
    min-width: 0;
    box-sizing: border-box;
  }

  /* Kanban sub-page: no padding so board fills edge-to-edge; overflow hidden */
  .subpage-wrap:has(.department-kanban) {
    padding: 0;
    overflow: hidden;
    display: flex;
    flex-direction: column;
  }

  .news-expanded {
    max-width: 900px;
    margin: 0 auto;
  }

  .placeholder-page {
    display: flex;
    align-items: center;
    justify-content: center;
  }

  .placeholder-inner {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 12px;
    color: #5a6a80;
    text-align: center;
  }

  .placeholder-inner h2 {
    font-family: 'JetBrains Mono', monospace;
    font-size: 18px;
    font-weight: 500;
    color: #e8edf5;
    margin: 0;
  }

  /* ── Canvas scroll area ───────────────────────────────────────────────── */
  .canvas-scroll {
    flex: 1;
    overflow-y: auto;
    padding: 12px 16px;
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  /* ── Morning digest row (full-width strip) ────────────────────────────── */
  .digest-row {
    width: 100%;
  }

  /* ── Tile grid ────────────────────────────────────────────────────────── */
  .tile-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
    gap: 12px;
    align-items: start;
  }

  /* ── Base tile ────────────────────────────────────────────────────────── */
  .tile {
    background: rgba(8, 13, 20, 0.35);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 8px;
    padding: 16px;
    display: flex;
    flex-direction: column;
    gap: 10px;
    text-align: left;
    color: #e8edf5;
    position: relative;
  }

  .tile--clickable {
    cursor: pointer;
    transition: border-color 0.15s, background 0.15s;
    /* Reset button styles */
    font-family: inherit;
    font-size: inherit;
  }

  .tile--clickable:hover {
    background: rgba(8, 13, 20, 0.5);
    border-color: rgba(0, 212, 255, 0.25);
  }

  .tile--tall {
    grid-row: span 2;
  }

  /* ── Tile header ──────────────────────────────────────────────────────── */
  .tile-header {
    display: flex;
    align-items: center;
    gap: 7px;
    color: #5a6a80;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
  }

  .tile-title {
    flex: 1;
    color: #5a6a80;
  }

  .tile-badge {
    font-size: 10px;
    padding: 2px 6px;
    border-radius: 4px;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0;
    text-transform: none;
  }

  .tile-badge.cyan {
    background: rgba(0, 212, 255, 0.12);
    color: #00d4ff;
    border: 1px solid rgba(0, 212, 255, 0.25);
  }

  .tile-badge.amber {
    background: rgba(240, 165, 0, 0.12);
    color: #f0a500;
    border: 1px solid rgba(240, 165, 0, 0.25);
  }

  .tile-badge.green {
    background: rgba(0, 200, 150, 0.12);
    color: #00c896;
    border: 1px solid rgba(0, 200, 150, 0.25);
  }

  .tile-badge.red {
    background: rgba(255, 59, 59, 0.12);
    color: #ff3b3b;
    border: 1px solid rgba(255, 59, 59, 0.25);
  }

  .tile-body {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .tile-cta {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: #00d4ff;
    opacity: 0.6;
    margin-top: auto;
    padding-top: 4px;
  }

  .tile--clickable:hover .tile-cta {
    opacity: 1;
  }

  /* ── Big numbers ──────────────────────────────────────────────────────── */
  .big-number {
    font-family: 'JetBrains Mono', monospace;
    font-size: 36px;
    font-weight: 700;
    line-height: 1;
  }

  .big-number.cyan { color: #00d4ff; }
  .big-number.amber { color: #f0a500; }

  .tile-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: #5a6a80;
    margin-top: -4px;
  }

  /* ── Session clocks ───────────────────────────────────────────────────── */
  .clocks-body {
    display: flex;
    flex-direction: column;
    gap: 10px;
  }

  .clock-row {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .clock-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: #5a6a80;
    width: 68px;
    flex-shrink: 0;
  }

  .clock-time {
    font-family: 'JetBrains Mono', monospace;
    font-size: 15px;
    font-weight: 600;
    color: #e8edf5;
    flex: 1;
    letter-spacing: 0.05em;
  }

  .session-badge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px;
    font-weight: 700;
    letter-spacing: 0.08em;
    padding: 2px 6px;
    border-radius: 3px;
  }

  .session-badge.open {
    background: rgba(0, 200, 150, 0.15);
    color: #00c896;
    border: 1px solid rgba(0, 200, 150, 0.3);
  }

  .session-badge.closed {
    background: rgba(90, 106, 128, 0.15);
    color: #5a6a80;
    border: 1px solid rgba(90, 106, 128, 0.2);
  }

  /* ── Bot pairs ────────────────────────────────────────────────────────── */
  .bot-pairs {
    display: flex;
    flex-wrap: wrap;
    gap: 5px;
    margin-top: 2px;
  }

  .pair-chip {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    padding: 2px 7px;
    border-radius: 3px;
    background: rgba(0, 212, 255, 0.1);
    color: #00d4ff;
    border: 1px solid rgba(0, 212, 255, 0.2);
  }

  .pair-chip.active {
    background: rgba(0, 200, 150, 0.12);
    color: #00c896;
    border-color: rgba(0, 200, 150, 0.25);
  }

  .muted-chip {
    background: rgba(90, 106, 128, 0.1);
    color: #5a6a80;
    border-color: rgba(90, 106, 128, 0.2);
  }

  .tile-pnl {
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    font-weight: 600;
    display: flex;
    align-items: baseline;
    gap: 6px;
    margin-top: 2px;
  }

  .pnl-label {
    font-size: 10px;
    color: #5a6a80;
    font-weight: 400;
  }

  /* ── P&L tile ─────────────────────────────────────────────────────────── */
  .pnl-hero {
    font-family: 'JetBrains Mono', monospace;
    font-size: 28px;
    font-weight: 700;
    line-height: 1;
    letter-spacing: -0.01em;
  }

  .pnl-breakdown {
    display: flex;
    flex-direction: column;
    gap: 5px;
    margin-top: 4px;
  }

  .pnl-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
  }

  .pnl-symbol {
    color: #5a6a80;
  }

  .pnl-val {
    font-weight: 500;
  }

  /* ── Positions tile ───────────────────────────────────────────────────── */
  .largest-pos {
    display: flex;
    align-items: center;
    gap: 6px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    margin-top: 2px;
  }

  .muted-text {
    color: #5a6a80;
  }

  .pos-symbol {
    color: #e8edf5;
    font-weight: 600;
  }

  /* ── Node health tile ─────────────────────────────────────────────────── */
  .node-rows {
    display: flex;
    flex-direction: column;
    gap: 10px;
  }

  .node-row {
    display: flex;
    align-items: center;
    gap: 8px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
  }

  .node-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
  }

  .dot-green { background: #00c896; box-shadow: 0 0 5px rgba(0, 200, 150, 0.6); }
  .dot-amber { background: #f0a500; box-shadow: 0 0 5px rgba(240, 165, 0, 0.6); }
  .dot-red   { background: #ff3b3b; box-shadow: 0 0 5px rgba(255, 59, 59, 0.6); }

  .node-name {
    flex: 1;
    color: #e8edf5;
  }

  .node-latency {
    color: #5a6a80;
    font-size: 10px;
    min-width: 36px;
    text-align: right;
  }

  .node-status-text {
    font-size: 9px;
    font-weight: 700;
    letter-spacing: 0.06em;
    min-width: 72px;
    text-align: right;
  }

  .text-green { color: #00c896; }
  .text-red   { color: #ff3b3b; }

  /* ── News tile inner ──────────────────────────────────────────────────── */
  .news-preview-wrap {
    flex: 1;
    overflow: hidden;
    /* Let NewsFeedTile render inside; clip overflow for compact view */
    min-height: 0;
  }

  /* ── Digest tile inner ────────────────────────────────────────────────── */
  .digest-tile-body {
    flex: 1;
    overflow: hidden;
    min-height: 0;
  }

  /* ── Misc ─────────────────────────────────────────────────────────────── */
  .empty-hint {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: #5a6a80;
  }

  .muted {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    color: #5a6a80;
    margin: 0;
  }

  /* ── Animations ───────────────────────────────────────────────────────── */
  :global(.spinning) {
    animation: spin 0.8s linear infinite;
    display: inline-flex;
  }

  .icon-spin {
    animation: spin 1s linear infinite;
    display: inline-flex;
  }

  @keyframes spin {
    from { transform: rotate(0deg); }
    to   { transform: rotate(360deg); }
  }

  /* ── Scrollbar ────────────────────────────────────────────────────────── */
  .canvas-scroll::-webkit-scrollbar {
    width: 4px;
  }

  .canvas-scroll::-webkit-scrollbar-track {
    background: transparent;
  }

  .canvas-scroll::-webkit-scrollbar-thumb {
    background: rgba(0, 212, 255, 0.15);
    border-radius: 2px;
  }
</style>
