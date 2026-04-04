<script lang="ts">
  /**
   * BotStatusCard - Individual Bot Status Display
   *
   * Shows: EA name, symbol, P&L, position count, regime, session status
   * P&L flash animation on change
   * Includes 3-dot menu for position close action and cross-canvas navigation
   */
  import GlassTile from './GlassTile.svelte';
  import { pnlFlash, selectBot, activeBots, type BotStatus } from '$lib/stores/trading';
  import { activeCanvasStore } from '$lib/stores/canvasStore';
  import { navigationStore } from '$lib/stores/navigationStore';
  import { TrendingUp, TrendingDown, Circle, Activity, Clock, MoreHorizontal, XCircle, Code, BarChart3, History, AlertCircle, FileText } from 'lucide-svelte';
  import { apiFetch } from '$lib/api';
  import PositionCloseModal from './PositionCloseModal.svelte';
  import { goto } from '$app/navigation';
  import { browser } from '$app/environment';

  interface Props {
    bot: BotStatus;
  }

  let { bot }: Props = $props();

  // Menu state
  let showMenu = $state(false);
  let showCloseModal = $state(false);

  // SSL state
  let sslData = $state<{ consecutive_losses: number; ssl_state: string; recovery_win_count: number } | null>(null);

  $effect(() => {
    if (!browser) return;  // SSR guard
    if (bot.bot_id) {
      fetchSSLState(bot.bot_id);
      const interval = setInterval(() => fetchSSLState(bot.bot_id), 10000);
      return () => clearInterval(interval);
    }
  });

  async function fetchSSLState(botId: string) {
    try {
      const res = await apiFetch<any>(`/api/ssl/state/${botId}`);
      sslData = res;
      // Also update the store
      activeBots.update((bots) => {
        const idx = bots.findIndex((b) => b.bot_id === botId);
        if (idx >= 0) {
          bots[idx] = {
            ...bots[idx],
            consecutive_losses: res.consecutive_losses,
            ssl_state: res.ssl_state as any
          };
        }
        return [...bots];
      });
    } catch (e) {
      // Silently fail — SSL data is supplementary
    }
  }

  let flashColor = $derived($pnlFlash.get(bot.bot_id));
  let pnlClass = $derived(bot.current_pnl > 0 ? 'positive' : bot.current_pnl < 0 ? 'negative' : 'neutral');
  let isFlashing = $derived(flashColor === 'green' || flashColor === 'red');

  function handleClick() {
    selectBot(bot.bot_id);
  }

  function formatPnl(value: number): string {
    return value >= 0 ? `+$${value.toFixed(2)}` : `-$${Math.abs(value).toFixed(2)}`;
  }

  function getRegimeLabel(regime: string): string {
    const labels: Record<string, string> = {
      'ASIAN': 'Asian',
      'LONDON': 'London',
      'NEW_YORK': 'New York',
      'OVERLAP': 'Overlap',
      'CLOSED': 'Closed'
    };
    return labels[regime] || regime;
  }

  function toggleMenu(e: MouseEvent) {
    e.stopPropagation();
    showMenu = !showMenu;
  }

  function closeMenu() {
    showMenu = false;
  }

  function handleClosePosition(e: MouseEvent) {
    e.stopPropagation();
    showMenu = false;
    showCloseModal = true;
  }

  function handleModalClose() {
    showCloseModal = false;
  }

  // Cross-canvas navigation handlers
  async function handleViewCode(e: MouseEvent) {
    e.stopPropagation();
    showMenu = false;
    activeCanvasStore.setActiveCanvas('development');
    navigationStore.navigateToSubPage(`bot-${bot.bot_id}`, bot.ea_name);
    await goto(`/development`);
  }

  async function handleViewPerformance(e: MouseEvent) {
    e.stopPropagation();
    showMenu = false;
    activeCanvasStore.setActiveCanvas('trading');
    navigationStore.navigateToSubPage(`bot-${bot.bot_id}`, bot.ea_name);
    await goto(`/trading`);
  }

  async function handleViewHistory(e: MouseEvent) {
    e.stopPropagation();
    showMenu = false;
    activeCanvasStore.setActiveCanvas('portfolio');
    navigationStore.navigateToSubPage(`bot-${bot.bot_id}`, bot.ea_name);
    await goto(`/portfolio`);
  }
</script>

<GlassTile clickable on:click={handleClick}>
  <div class="bot-card">
    <div class="header">
      <span class="ea-name" title={bot.ea_name}>{bot.ea_name}</span>
      {#if bot.ssl_state === 'paper'}
        <span class="ssl-counter paper">
          <FileText size={10} /> Paper Recovery (Win {(sslData?.recovery_win_count ?? 0)}/2)
        </span>
      {:else if (bot.consecutive_losses ?? 0) > 0}
        <span class="ssl-counter" class:amber={(bot.consecutive_losses ?? 0) >= 2 && (bot.consecutive_losses ?? 0) < 3} class:red={(bot.consecutive_losses ?? 0) >= 3}>
          <AlertCircle size={10} /> {bot.consecutive_losses} consec. losses
        </span>
      {/if}
      <div class="header-actions">
        <span class="symbol">{bot.symbol}</span>
        <button class="menu-btn" onclick={toggleMenu} aria-label="More options">
          <MoreHorizontal size={14} />
        </button>
      </div>

      {#if showMenu}
        <!-- svelte-ignore a11y_click_events_have_key_events a11y_no_static_element_interactions -->
        <div class="dropdown-menu" onclick={closeMenu}>
          <button class="menu-item" onclick={handleViewCode}>
            <Code size={14} />
            <span>View Code</span>
          </button>
          <button class="menu-item" onclick={handleViewPerformance}>
            <BarChart3 size={14} />
            <span>View Performance</span>
          </button>
          <button class="menu-item" onclick={handleViewHistory}>
            <History size={14} />
            <span>View History</span>
          </button>
          <div class="menu-divider"></div>
          <button class="menu-item danger" onclick={handleClosePosition} disabled={bot.open_positions === 0}>
            <XCircle size={14} />
            <span>Close Position</span>
          </button>
        </div>
      {/if}
    </div>

    <div class="metrics">
      <div class="metric pnl" class:positive={pnlClass === 'positive'} class:negative={pnlClass === 'negative'} class:flashing={isFlashing} class:flash-green={flashColor === 'green'} class:flash-red={flashColor === 'red'}>
        {#if bot.current_pnl > 0}
          <TrendingUp size={14} />
        {:else if bot.current_pnl < 0}
          <TrendingDown size={14} />
        {/if}
        <span class="value">{formatPnl(bot.current_pnl)}</span>
      </div>

      <div class="metric positions">
        <Activity size={14} />
        <span class="value">{bot.open_positions}</span>
      </div>
    </div>

    <div class="footer">
      <div class="regime" class:active={bot.session_active}>
        <Circle size={8} fill={bot.session_active ? '#00c896' : '#666'} stroke="none" />
        <span>{getRegimeLabel(bot.regime)}</span>
      </div>

      <div class="session-status" class:active={bot.session_active}>
        <Clock size={12} />
        <span>{bot.session_active ? 'Active' : 'Inactive'}</span>
      </div>
    </div>

    <div class="timestamp">
      {bot.last_update.replace('_utc', '').replace('T', ' ').slice(0, 16)}
    </div>
  </div>
</GlassTile>

{#if showCloseModal}
  <PositionCloseModal
    position={{
      ticket: 0, // Would come from actual position data
      bot_id: bot.bot_id,
      symbol: bot.symbol,
      direction: 'buy', // Would come from actual position data
      lot: 0, // Would come from actual position data
      current_pnl: bot.current_pnl
    }}
    onClose={handleModalClose}
  />
{/if}

<style>
  .bot-card {
    display: flex;
    flex-direction: column;
    gap: 12px;
    min-width: 0;
  }

  .header {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .ea-name {
    font-family: 'JetBrains Mono', 'Fragment Mono', 'Fira Code', monospace;
    font-size: 13px;
    font-weight: 600;
    color: #e0e0e0;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 60%;
  }

  .symbol {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: #00d4ff;
    background: rgba(0, 212, 255, 0.1);
    padding: 2px 6px;
    border-radius: 4px;
  }

  .ssl-counter {
    display: flex;
    align-items: center;
    gap: 3px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px;
    color: #888;
  }
  .ssl-counter.amber { color: #F59E0B; }
  .ssl-counter.red { color: #EF4444; }
  .ssl-counter.paper { color: #3B82F6; }

  .metrics {
    display: flex;
    gap: 16px;
  }

  .metric {
    display: flex;
    align-items: center;
    gap: 6px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
  }

  .pnl.positive {
    color: #00c896;
  }

  .pnl.negative {
    color: #ff3b3b;
  }

  .pnl.neutral {
    color: #888;
  }

  .pnl.flashing {
    animation: flash 0.1s ease;
  }

  .pnl.flash-green {
    background: rgba(0, 200, 150, 0.2);
  }

  .pnl.flash-red {
    background: rgba(255, 59, 59, 0.2);
  }

  @keyframes flash {
    0%, 100% {
      opacity: 1;
    }
    50% {
      opacity: 0.7;
    }
  }

  .positions {
    color: #aaa;
  }

  .footer {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .regime, .session-status {
    display: flex;
    align-items: center;
    gap: 4px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: #888;
    text-transform: uppercase;
  }

  .regime.active, .session-status.active {
    color: #00c896;
  }

  .timestamp {
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px;
    color: #555;
    text-align: right;
  }

  .header-actions {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .menu-btn {
    background: transparent;
    border: none;
    color: #888;
    cursor: pointer;
    padding: 4px;
    border-radius: 4px;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: all 0.15s ease;
  }

  .menu-btn:hover {
    background: rgba(255, 255, 255, 0.08);
    color: #e0e0e0;
  }

  .dropdown-menu {
    position: absolute;
    top: 32px;
    right: 8px;
    background: rgba(13, 17, 23, 0.95);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 8px;
    padding: 4px;
    z-index: 100;
    min-width: 140px;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
  }

  .menu-item {
    display: flex;
    align-items: center;
    gap: 8px;
    width: 100%;
    padding: 8px 12px;
    border: none;
    background: transparent;
    color: #e0e0e0;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    cursor: pointer;
    border-radius: 4px;
    transition: all 0.15s ease;
    text-align: left;
  }

  .menu-item:hover:not(:disabled) {
    background: rgba(255, 255, 255, 0.08);
  }

  .menu-item:disabled {
    opacity: 0.4;
    cursor: not-allowed;
  }

  .menu-item.danger {
    color: #ff3b3b;
  }

  .menu-item.danger:hover:not(:disabled) {
    background: rgba(255, 59, 59, 0.1);
  }

  .menu-divider {
    height: 1px;
    background: rgba(255, 255, 255, 0.1);
    margin: 4px 0;
  }
</style>
