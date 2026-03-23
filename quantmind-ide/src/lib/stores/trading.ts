/**
 * Live Trading State Store
 *
 * Manages real-time state for the Live Trading canvas:
 * - Active bots and their status
 * - WebSocket connection for streaming updates
 * - Bot detail view state
 * - Position close actions
 */

import { writable, derived, get } from 'svelte/store';
import { nodeHealthState, isContaboDegraded, isCloudzyDegraded } from './node-health';

// Bot status types
export type Regime = 'ASIAN' | 'LONDON' | 'NEW_YORK' | 'OVERLAP' | 'CLOSED';

export interface BotStatus {
  bot_id: string;
  ea_name: string;
  symbol: string;
  current_pnl: number;
  open_positions: number;
  regime: Regime;
  session_active: boolean;
  last_update: string; // ISO timestamp with _utc suffix
}

export interface BotDetail extends BotStatus {
  session_mask: number[]; // 24-element array for hours
  force_close_hour: number | null;
  overnight_hold: boolean;
  daily_loss_cap: number;
  current_loss_pct: number;
  equity_exposure: number;
}

export interface TradingUpdate {
  type: 'position_update' | 'pnl_update' | 'regime_change' | 'bot_status_change';
  bot_id: string;
  timestamp_utc: string;
  data: Record<string, any>;
}

// Position close types
export interface PositionInfo {
  ticket: number;
  bot_id: string;
  symbol: string;
  direction: 'buy' | 'sell';
  lot: number;
  current_pnl: number;
}

export interface CloseResult {
  success: boolean;
  filled_price?: number;
  slippage?: number;
  final_pnl?: number;
  message: string;
}

export interface CloseAllResult {
  success: boolean;
  results: Array<{
    position_ticket: number;
    status: 'filled' | 'partial' | 'rejected';
    filled_price?: number;
    slippage?: number;
    final_pnl?: number;
    message?: string;
  }>;
}

// WebSocket connection state
export const wsConnected = writable(false);
export const wsError = writable<string | null>(null);

// Active bots state
export const activeBots = writable<BotStatus[]>([]);

// Bot detail state
export const selectedBotId = writable<string | null>(null);
export const botDetails = writable<Map<string, BotDetail>>(new Map());

// Flash animation state for P&L changes
export const pnlFlash = writable<Map<string, 'green' | 'red'>>(new Map());

// Loading states
export const isLoading = writable(true);

// Close position states
export const closeLoading = writable(false);
export const closeError = writable<string | null>(null);

// Derived stores
export const selectedBot = derived(
  [selectedBotId, botDetails],
  ([$selectedBotId, $botDetails]) => {
    if (!$selectedBotId) return null;
    return $botDetails.get($selectedBotId) || null;
  }
);

export const botCount = derived(activeBots, ($activeBots) => $activeBots.length);

// Degraded mode awareness - derived from node-health store
export const tradingDegraded = derived(
  [isContaboDegraded, isCloudzyDegraded],
  ([$isContaboDegraded, $isCloudzyDegraded]) => ({
    contabo: $isContaboDegraded,
    cloudzy: $isCloudzyDegraded,
    hasDegradation: $isContaboDegraded || $isCloudzyDegraded
  })
);

// WebSocket connection
let ws: WebSocket | null = null;
let reconnectAttempts = 0;
const MAX_RECONNECT_ATTEMPTS = 10;
const BASE_RECONNECT_DELAY = 1000;

/**
 * Connect to trading WebSocket
 */
export function connectTradingWS(url?: string) {
  const wsUrl = url || `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/ws/trading`;

  if (ws?.readyState === WebSocket.OPEN) {
    console.log('[LiveTradingWS] Already connected');
    return;
  }

  try {
    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      console.log('[LiveTradingWS] Connected');
      wsConnected.set(true);
      wsError.set(null);
      reconnectAttempts = 0;

      // Subscribe to trading events
      ws?.send(JSON.stringify({
        type: 'subscribe',
        topic: 'trading',
        events: ['position_update', 'pnl_update', 'regime_change', 'bot_status_change']
      }));

      // Request initial state
      ws?.send(JSON.stringify({
        type: 'get_initial_state'
      }));
    };

    ws.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data);
        handleTradingUpdate(message);
      } catch (e) {
        console.error('[LiveTradingWS] Parse error:', e);
      }
    };

    ws.onerror = (error) => {
      console.error('[LiveTradingWS] Error:', error);
      wsError.set('WebSocket connection error');
    };

    ws.onclose = () => {
      console.log('[LiveTradingWS] Disconnected');
      wsConnected.set(false);
      ws = null;

      // Exponential backoff reconnect
      if (reconnectAttempts < MAX_RECONNECT_ATTEMPTS) {
        reconnectAttempts++;
        const delay = BASE_RECONNECT_DELAY * Math.pow(2, reconnectAttempts - 1);
        console.log(`[LiveTradingWS] Reconnecting in ${delay}ms (attempt ${reconnectAttempts})`);
        setTimeout(() => connectTradingWS(url), delay);
      }
    };
  } catch (e) {
    console.error('[LiveTradingWS] Connection failed:', e);
    wsError.set('Failed to connect');
  }
}

/**
 * Disconnect WebSocket
 */
export function disconnectTradingWS() {
  if (ws) {
    ws.close();
    ws = null;
  }
  wsConnected.set(false);
}

/**
 * Handle incoming trading updates
 */
function handleTradingUpdate(message: any) {
  const update = message as TradingUpdate;

  switch (update.type) {
    case 'position_update':
      handlePositionUpdate(update);
      break;
    case 'pnl_update':
      handlePnlUpdate(update);
      break;
    case 'regime_change':
      handleRegimeChange(update);
      break;
    case 'bot_status_change':
      handleBotStatusChange(update);
      break;
    case 'initial_state':
      handleInitialState(message);
      break;
  }
}

function handlePositionUpdate(update: TradingUpdate) {
  const { bot_id, data } = update;

  activeBots.update((bots) => {
    const index = bots.findIndex((b) => b.bot_id === bot_id);
    if (index >= 0) {
      bots[index] = {
        ...bots[index],
        open_positions: data.open_positions,
        last_update: update.timestamp_utc
      };
    }
    return [...bots];
  });
}

function handlePnlUpdate(update: TradingUpdate) {
  const { bot_id, data, timestamp_utc } = update;

  // Trigger flash animation
  const previousPnl = get(activeBots).find((b) => b.bot_id === bot_id)?.current_pnl || 0;
  const newPnl = data.current_pnl;

  if (newPnl > previousPnl) {
    triggerFlash(bot_id, 'green');
  } else if (newPnl < previousPnl) {
    triggerFlash(bot_id, 'red');
  }

  activeBots.update((bots) => {
    const index = bots.findIndex((b) => b.bot_id === bot_id);
    if (index >= 0) {
      bots[index] = {
        ...bots[index],
        current_pnl: newPnl,
        last_update: timestamp_utc
      };
    }
    return [...bots];
  });
}

function handleRegimeChange(update: TradingUpdate) {
  const { bot_id, data, timestamp_utc } = update;

  activeBots.update((bots) => {
    const index = bots.findIndex((b) => b.bot_id === bot_id);
    if (index >= 0) {
      bots[index] = {
        ...bots[index],
        regime: data.regime as Regime,
        last_update: timestamp_utc
      };
    }
    return [...bots];
  });
}

function handleBotStatusChange(update: TradingUpdate) {
  const { bot_id, data, timestamp_utc } = update;

  activeBots.update((bots) => {
    const index = bots.findIndex((b) => b.bot_id === bot_id);
    if (index >= 0) {
      bots[index] = {
        ...bots[index],
        session_active: data.session_active,
        last_update: timestamp_utc
      };
    }
    return [...bots];
  });
}

function handleInitialState(message: any) {
  if (message.bots) {
    activeBots.set(message.bots);
  }
  if (message.bots_detail) {
    const detailsMap = new Map<string, BotDetail>();
    message.bots_detail.forEach((detail: BotDetail) => {
      detailsMap.set(detail.bot_id, detail);
    });
    botDetails.set(detailsMap);
  }
  isLoading.set(false);
}

/**
 * Trigger P&L flash animation
 */
function triggerFlash(botId: string, color: 'green' | 'red') {
  pnlFlash.update((flash) => {
    flash.set(botId, color);
    return new Map(flash);
  });

  // Clear flash after 100ms
  setTimeout(() => {
    pnlFlash.update((flash) => {
      flash.delete(botId);
      return new Map(flash);
    });
  }, 100);
}

/**
 * Select a bot to view details
 */
export function selectBot(botId: string | null) {
  selectedBotId.set(botId);
}

/**
 * Load bot details from API
 */
export async function loadBotDetails(botId: string): Promise<BotDetail | null> {
  try {
    const response = await fetch(`/api/v1/trading/bots/${botId}/params`);
    if (!response.ok) {
      throw new Error('Failed to load bot details');
    }
    const data = await response.json();

    const detail: BotDetail = {
      bot_id: botId,
      ea_name: data.ea_name,
      symbol: data.symbol,
      current_pnl: data.current_pnl || 0,
      open_positions: data.open_positions || 0,
      regime: data.current_regime || 'CLOSED',
      session_active: data.session_active || false,
      last_update: data.last_update || new Date().toISOString(),
      session_mask: data.session_mask || [],
      force_close_hour: data.force_close_hour,
      overnight_hold: data.overnight_hold || false,
      daily_loss_cap: data.daily_loss_cap || 0,
      current_loss_pct: data.current_loss_pct || 0,
      equity_exposure: data.equity_exposure || 0
    };

    botDetails.update((details) => {
      const newDetails = new Map(details);
      newDetails.set(botId, detail);
      return newDetails;
    });

    return detail;
  } catch (error) {
    console.error('[LiveTrading] Failed to load bot details:', error);
    return null;
  }
}

/**
 * Fetch all active bots from API
 */
export async function fetchActiveBots(): Promise<BotStatus[]> {
  try {
    const response = await fetch('/api/v1/trading/bots');
    if (!response.ok) {
      throw new Error('Failed to fetch bots');
    }
    const data = await response.json();
    activeBots.set(data.bots || []);
    isLoading.set(false);
    return data.bots || [];
  } catch (error) {
    console.error('[LiveTrading] Failed to fetch bots:', error);
    isLoading.set(false);
    return [];
  }
}

/**
 * Close a single position
 */
export async function closePosition(ticket: number, botId: string): Promise<CloseResult | null> {
  closeLoading.set(true);
  closeError.set(null);

  try {
    const response = await fetch('/api/v1/trading/close', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        position_ticket: ticket,
        bot_id: botId
      })
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.message || 'Failed to close position');
    }

    const result = await response.json() as CloseResult;

    // Trigger P&L flash for the bot
    triggerFlash(botId, result.final_pnl && result.final_pnl > 0 ? 'green' : 'red');

    // Update bot's position count
    activeBots.update((bots) => {
      const index = bots.findIndex((b) => b.bot_id === botId);
      if (index >= 0) {
        bots[index] = {
          ...bots[index],
          open_positions: Math.max(0, bots[index].open_positions - 1),
          current_pnl: result.final_pnl ?? bots[index].current_pnl,
          last_update: new Date().toISOString() + '_utc'
        };
      }
      return [...bots];
    });

    return result;
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unknown error';
    closeError.set(message);
    console.error('[LiveTrading] Failed to close position:', error);
    return null;
  } finally {
    closeLoading.set(false);
  }
}

/**
 * Close all positions for a bot or all bots
 */
export async function closeAllPositions(botId?: string): Promise<CloseAllResult | null> {
  closeLoading.set(true);
  closeError.set(null);

  try {
    const response = await fetch('/api/v1/trading/close-all', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        bot_id: botId || null
      })
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.message || 'Failed to close positions');
    }

    const result = await response.json() as CloseAllResult;

    // Trigger P&L flash for affected bots
    const affectedBots = new Set<string>();
    result.results.forEach((r) => {
      const bot = get(activeBots).find((b) => b.open_positions > 0);
      if (bot) affectedBots.add(bot.bot_id);
    });
    affectedBots.forEach((bid) => {
      const filledCount = result.results.filter((r) => r.status === 'filled').length;
      triggerFlash(bid, filledCount > 0 ? 'green' : 'red');
    });

    // Refresh bot list
    await fetchActiveBots();

    return result;
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unknown error';
    closeError.set(message);
    console.error('[LiveTrading] Failed to close all positions:', error);
    return null;
  } finally {
    closeLoading.set(false);
  }
}
