/**
 * Economic Calendar Store
 *
 * Manages polling for economic calendar data from the risk API:
 * - Today's economic events with impact levels
 * - Events for specific date
 * - Active blackout windows
 *
 * Polls every 30 seconds (economic events don't change as frequently).
 */

import { writable, derived } from 'svelte/store';
import { wsClient } from '$lib/ws-client';
import type { WebSocketMessage } from '$lib/ws-client';

// Types for economic calendar data
export interface EconomicEvent {
  time: string;
  currency: string;
  event_name: string;
  impact: 'high' | 'medium' | 'low';
  previous: string | null;
  forecast: string | null;
  actual: string | null;
  is_blackout: boolean;
}

export interface BlackoutWindow {
  start: string;
  end: string;
  currency: string;
  reason: string | null;
}

export interface EconomicCalendarData {
  date: string;
  events: EconomicEvent[];
  blackouts: BlackoutWindow[];
}

export interface EconomicCalendarState {
  data: EconomicCalendarData | null;
  loading: boolean;
  error: string | null;
  lastUpdated: Date | null;
}

// Initial state
const initialState: EconomicCalendarState = {
  data: null,
  loading: false,
  error: null,
  lastUpdated: null
};

// Create the store
function createEconomicCalendarStore() {
  const { subscribe, set, update } = writable<EconomicCalendarState>(initialState);

  let pollingInterval: ReturnType<typeof setInterval> | null = null;
  let wsHandler: ((msg: WebSocketMessage) => void) | null = null;

  function connectWebSocket() {
    try {
      // Handler for news_blackout_update messages from NewsBlackoutService
      wsHandler = (msg: WebSocketMessage) => {
        if (msg.type === 'news_blackout_update') {
          // Re-fetch calendar data when news state changes
          fetchToday();
        }
      };
      wsClient.on('news_blackout_update', wsHandler);
    } catch (e) {
      console.warn('Failed to connect economic calendar to WebSocket:', e);
    }
  }

  async function fetchToday() {
    update(state => ({ ...state, loading: true, error: null }));

    try {
      const response = await fetch('/api/risk/economic-calendar');

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data: EconomicCalendarData = await response.json();

      update(state => ({
        ...state,
        data,
        loading: false,
        lastUpdated: new Date()
      }));
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to fetch economic calendar data';
      update(state => ({
        ...state,
        loading: false,
        error: errorMessage
      }));
    }
  }

  async function fetchByDate(date: string) {
    update(state => ({ ...state, loading: true, error: null }));

    try {
      const response = await fetch(`/api/risk/economic-calendar/${date}`);

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data: EconomicCalendarData = await response.json();

      update(state => ({
        ...state,
        data,
        loading: false,
        lastUpdated: new Date()
      }));
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to fetch economic calendar data';
      update(state => ({
        ...state,
        loading: false,
        error: errorMessage
      }));
    }
  }

  async function fetchBlackouts() {
    update(state => ({ ...state, loading: true, error: null }));

    try {
      const response = await fetch('/api/risk/economic-calendar/blackouts');

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const { blackouts } = await response.json();

      update(state => ({
        ...state,
        data: state.data ? { ...state.data, blackouts } : { date: '', events: [], blackouts },
        loading: false,
        lastUpdated: new Date()
      }));
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to fetch blackouts';
      update(state => ({
        ...state,
        loading: false,
        error: errorMessage
      }));
    }
  }

  function startPolling(intervalMs: number = 30000) {
    // Connect WebSocket for real-time updates
    connectWebSocket();

    // Fetch immediately
    fetchToday();

    // Then poll at interval
    if (pollingInterval) {
      clearInterval(pollingInterval);
    }
    pollingInterval = setInterval(fetchToday, intervalMs);
  }

  function stopPolling() {
    if (pollingInterval) {
      clearInterval(pollingInterval);
      pollingInterval = null;
    }
  }

  return {
    subscribe,
    fetchToday,
    fetchByDate,
    fetchBlackouts,
    startPolling,
    stopPolling,
    reset: () => set(initialState)
  };
}

// Export the store
export const economicCalendarStore = createEconomicCalendarStore();

// Derived stores for individual data
export const economicCalendarData = derived(economicCalendarStore, $store => $store.data);
export const economicCalendarEvents = derived(economicCalendarStore, $store => $store.data?.events ?? []);
export const economicCalendarBlackouts = derived(economicCalendarStore, $store => $store.data?.blackouts ?? []);
export const economicCalendarLoading = derived(economicCalendarStore, $store => $store.loading);
export const economicCalendarError = derived(economicCalendarStore, $store => $store.error);
export const economicCalendarLastUpdated = derived(economicCalendarStore, $store => $store.lastUpdated);

// Helper: Get impact color
export function getImpactColor(impact: 'high' | 'medium' | 'low'): string {
  switch (impact) {
    case 'high':
      return '#ff3b3b';
    case 'medium':
      return '#ffb700';
    case 'low':
      return '#00d4ff';
    default:
      return '#888888';
  }
}

// Helper: Get currency flag/emoji
export function getCurrencyBadge(currency: string): string {
  const flags: Record<string, string> = {
    USD: '$',
    EUR: '\u20ac',
    GBP: '\u00a3',
    JPY: '\u00a5',
    AUD: 'A$',
    CAD: 'C$',
    CHF: 'CHF',
    NZD: 'NZ$',
    CNY: '\u00a5',
    HKD: 'HK$',
  };
  return flags[currency] || currency;
}

// Helper: Format time until event
export function getTimeUntilEvent(eventTime: string): string {
  const now = new Date();
  const event = new Date(eventTime);
  const diff = event.getTime() - now.getTime();

  if (diff < 0) {
    return 'Past';
  }

  const minutes = Math.floor(diff / 60000);
  const hours = Math.floor(minutes / 60);
  const days = Math.floor(hours / 24);

  if (days > 0) {
    return `${days}d ${hours % 24}h`;
  }
  if (hours > 0) {
    return `${hours}h ${minutes % 60}m`;
  }
  return `${minutes}m`;
}

// Helper: Check if blackout is currently active
export function isBlackoutActive(blackout: BlackoutWindow): boolean {
  const now = new Date();
  const start = new Date(blackout.start);
  const end = new Date(blackout.end);
  return now >= start && now <= end;
}

// Helper: Get countdown to next event
export function getNextEvent(events: EconomicEvent[]): EconomicEvent | null {
  const now = new Date();
  const futureEvents = events
    .filter(e => new Date(e.time) > now)
    .sort((a, b) => new Date(a.time).getTime() - new Date(b.time).getTime());
  return futureEvents[0] || null;
}
