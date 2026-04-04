/**
 * Session Kelly Store
 *
 * Svelte store for Session Kelly modifier state management.
 * Provides per-session Kelly fraction data, house money status,
 * and London-NY overlap special indicators for the UI.
 *
 * Part of Story 4.10: Session-Scoped Kelly Modifiers
 */

import { writable, derived, type Readable } from 'svelte/store';
import { API_BASE } from '$lib/constants';

const apiBase = API_BASE || '';

// ============================================================================
// Types
// ============================================================================

export interface SessionKellyData {
  name: string;
  is_active: boolean;
  is_premium: boolean;
  premium_assault: string | null;
  kelly_fraction: number;
  kelly_dollar: number;
  house_money_threshold: number;
  status: 'NORMAL' | 'WARNING' | 'STRESS' | 'CRITICAL';
  hmm_multiplier: number;
  reverse_hmm_multiplier: number;
  session_kelly_multiplier: number;
  session_loss_counter: number;
  daily_pnl_pct: number;
}

export interface KellyCurrentState {
  sessions: SessionKellyData[];
  current_session: string;
  timestamp: string;
}

export interface KellyHistoryEntry {
  timestamp: string;
  current_session: string;
  hmm_multiplier: number;
  reverse_hmm_multiplier: number;
  session_kelly_multiplier: number;
  is_house_money_active: boolean;
  is_preservation_mode: boolean;
  daily_pnl_pct: number;
  session_loss_counter: number;
  premium_boost_active: boolean;
  is_premium_session: boolean;
  premium_assault: string | null;
}

export interface KellyHistoryState {
  history: KellyHistoryEntry[];
  count: number;
  date: string;
}

export type KellyStatus = 'NORMAL' | 'WARNING' | 'STRESS' | 'CRITICAL';

// ============================================================================
// State
// ============================================================================

interface SessionKellyState {
  current: KellyCurrentState | null;
  history: KellyHistoryState | null;
  isLoading: boolean;
  error: string | null;
  lastUpdate: number;
}

const initialState: SessionKellyState = {
  current: null,
  history: null,
  isLoading: false,
  error: null,
  lastUpdate: 0,
};

function createSessionKellyStore() {
  const { subscribe, set, update } = writable<SessionKellyState>(initialState);

  // ============================================================================
  // API Calls
  // ============================================================================

  async function fetchCurrentKelly(accountId: string = 'default'): Promise<KellyCurrentState> {
    const response = await fetch(`${apiBase}/api/risk/kelly/current?account_id=${accountId}`);
    if (!response.ok) {
      throw new Error(`Failed to fetch current Kelly: ${response.statusText}`);
    }
    return await response.json();
  }

  async function fetchKellyHistory(
    accountId: string = 'default',
    limit: number = 100
  ): Promise<KellyHistoryState> {
    const response = await fetch(
      `${apiBase}/api/risk/kelly/history?account_id=${accountId}&limit=${limit}`
    );
    if (!response.ok) {
      throw new Error(`Failed to fetch Kelly history: ${response.statusText}`);
    }
    return await response.json();
  }

  async function recordTradeResult(
    accountId: string = 'default',
    dailyPnlPct: number,
    isWin: boolean
  ): Promise<void> {
    const response = await fetch(
      `${apiBase}/api/risk/kelly/record?account_id=${accountId}&daily_pnl_pct=${dailyPnlPct}&is_win=${isWin}`,
      { method: 'POST' }
    );
    if (!response.ok) {
      throw new Error(`Failed to record trade result: ${response.statusText}`);
    }
  }

  async function notifySessionClose(accountId: string = 'default'): Promise<void> {
    const response = await fetch(
      `${apiBase}/api/risk/kelly/session-close?account_id=${accountId}`,
      { method: 'POST' }
    );
    if (!response.ok) {
      throw new Error(`Failed to notify session close: ${response.statusText}`);
    }
  }

  // ============================================================================
  // Actions
  // ============================================================================

  async function loadCurrentKelly(accountId: string = 'default') {
    update(state => ({ ...state, isLoading: true, error: null }));
    try {
      const current = await fetchCurrentKelly(accountId);
      update(state => ({
        ...state,
        current,
        isLoading: false,
        lastUpdate: Date.now(),
      }));
    } catch (error) {
      update(state => ({
        ...state,
        isLoading: false,
        error: error instanceof Error ? error.message : 'Failed to load current Kelly',
      }));
    }
  }

  async function loadHistory(accountId: string = 'default', limit: number = 100) {
    update(state => ({ ...state, isLoading: true, error: null }));
    try {
      const history = await fetchKellyHistory(accountId, limit);
      update(state => ({
        ...state,
        history,
        isLoading: false,
        lastUpdate: Date.now(),
      }));
    } catch (error) {
      update(state => ({
        ...state,
        isLoading: false,
        error: error instanceof Error ? error.message : 'Failed to load Kelly history',
      }));
    }
  }

  async function recordTrade(accountId: string = 'default', dailyPnlPct: number, isWin: boolean) {
    try {
      await recordTradeResult(accountId, dailyPnlPct, isWin);
      await loadCurrentKelly(accountId); // Reload after recording
    } catch (error) {
      update(state => ({
        ...state,
        error: error instanceof Error ? error.message : 'Failed to record trade',
      }));
    }
  }

  async function sessionClosed(accountId: string = 'default') {
    try {
      await notifySessionClose(accountId);
      await loadCurrentKelly(accountId); // Reload after session close
    } catch (error) {
      update(state => ({
        ...state,
        error: error instanceof Error ? error.message : 'Failed to notify session close',
      }));
    }
  }

  function reset() {
    set(initialState);
  }

  // ============================================================================
  // Derived Stores
  // ============================================================================

  /**
   * Get the active session's Kelly data
   */
  const activeSessionKelly: Readable<SessionKellyData | null> = derived(
    { subscribe },
    ($state: SessionKellyState) => {
      if (!$state.current) return null;
      return $state.current.sessions.find(s => s.is_active) || null;
    }
  );

  /**
   * Get all premium sessions
   */
  const premiumSessions: Readable<SessionKellyData[]> = derived(
    { subscribe },
    ($state: SessionKellyState) => {
      if (!$state.current) return [];
      return $state.current.sessions.filter(s => s.is_premium);
    }
  );

  /**
   * Get London-NY overlap session if active
   */
  const londonNyOverlap: Readable<SessionKellyData | null> = derived(
    { subscribe },
    ($state: SessionKellyState) => {
      if (!$state.current) return null;
      return $state.current.sessions.find(s => s.name === 'OVERLAP' && s.is_active) || null;
    }
  );

  /**
   * Get status color based on current Kelly status
   */
  const statusColor: Readable<string> = derived(
    { subscribe },
    ($state: SessionKellyState) => {
      const active = $state.current?.sessions.find(s => s.is_active);
      if (!active) return 'rgba(255, 255, 255, 0.5)';

      const colors: Record<KellyStatus, string> = {
        NORMAL: '#00ff88',     // green - House Money active or baseline
        WARNING: '#f59e0b',    // amber - 2-3 consecutive losses
        STRESS: '#ef4444',     // red - 4+ consecutive losses
        CRITICAL: '#dc2626',  // dark red - Preservation mode
      };
      return colors[active.status] || colors.NORMAL;
    }
  );

  /**
   * Get house money threshold display text
   */
  const thresholdText: Readable<string> = derived(
    { subscribe },
    ($state: SessionKellyState) => {
      const active = $state.current?.sessions.find(s => s.is_active);
      if (!active) return '+8%';

      // Premium sessions have lowered threshold (+4% vs normal +8%)
      if (active.is_premium) {
        return `+${(active.house_money_threshold * 100).toFixed(0)}% (Premium)`;
      }
      return `+${(active.house_money_threshold * 100).toFixed(0)}%`;
    }
  );

  /**
   * Get composite Kelly multiplier for display
   */
  const compositeMultiplier: Readable<string> = derived(
    { subscribe },
    ($state: SessionKellyState) => {
      const active = $state.current?.sessions.find(s => s.is_active);
      if (!active) return '1.00x';

      const composite = active.session_kelly_multiplier;
      if (composite >= 1.0) {
        return `${composite.toFixed(2)}x`;
      }
      return `${composite.toFixed(2)}x`;
    }
  );

  /**
   * Check if House Money Mode is active
   */
  const isHouseMoneyActive: Readable<boolean> = derived(
    { subscribe },
    ($state: SessionKellyState) => {
      const active = $state.current?.sessions.find(s => s.is_active);
      return active?.status === 'NORMAL' && active?.hmm_multiplier > 1.0;
    }
  );

  /**
   * Check if in Preservation mode
   */
  const isPreservationMode: Readable<boolean> = derived(
    { subscribe },
    ($state: SessionKellyState) => {
      const active = $state.current?.sessions.find(s => s.is_active);
      return active?.status === 'CRITICAL' || active?.hmm_multiplier < 1.0;
    }
  );

  // Auto-refresh every 30 seconds
  let refreshInterval: ReturnType<typeof setInterval> | null = null;

  function startAutoRefresh(intervalMs: number = 30000) {
    if (refreshInterval) clearInterval(refreshInterval);
    refreshInterval = setInterval(() => {
      loadCurrentKelly();
      loadHistory();
    }, intervalMs);
  }

  function stopAutoRefresh() {
    if (refreshInterval) {
      clearInterval(refreshInterval);
      refreshInterval = null;
    }
  }

  return {
    subscribe,
    loadCurrentKelly,
    loadHistory,
    recordTrade,
    sessionClosed,
    reset,
    startAutoRefresh,
    stopAutoRefresh,
    // Derived stores
    activeSessionKelly,
    premiumSessions,
    londonNyOverlap,
    statusColor,
    thresholdText,
    compositeMultiplier,
    isHouseMoneyActive,
    isPreservationMode,
  };
}

export const sessionKellyStore = createSessionKellyStore();
export default sessionKellyStore;
