/**
 * News State Store
 *
 * Manages state for news feed on Live Trading and Research canvases:
 * - News items with real-time updates
 * - Filtering by impact tier, symbol, date range
 * - WebSocket subscription for real-time alerts
 */

import { writable, derived, get } from 'svelte/store';
import { getNewsFeed, type NewsFeedItem } from '$lib/api/newsApi';
import { wsClient, type WebSocketMessage } from '$lib/ws-client';
import { API_CONFIG } from '$lib/config/api';

// =============================================================================
// Types
// =============================================================================

export type NewsSeverity = 'LOW' | 'MEDIUM' | 'HIGH' | null;
export type NewsActionType = 'MONITOR' | 'ALERT' | 'FAST_TRACK' | null;

export interface NewsFilter {
  severity: NewsSeverity | 'all';
  symbols: string[];
  dateFrom: string | null;
  dateTo: string | null;
}

export interface NewsState {
  items: NewsFeedItem[];
  isLoading: boolean;
  error: string | null;
  filter: NewsFilter;
  lastUpdate: string | null;
  wsConnected: boolean;
}

export interface NewsAlertMessage {
  type: string;
  data: {
    item_id: string;
    headline: string;
    severity: 'HIGH' | 'MEDIUM' | 'LOW';
    action_type: 'ALERT' | 'FAST_TRACK' | 'MONITOR';
    affected_symbols: string[];
    published_utc: string;
  };
}

// =============================================================================
// Stores
// =============================================================================

const defaultFilter: NewsFilter = {
  severity: 'all',
  symbols: [],
  dateFrom: null,
  dateTo: null
};

function createNewsStore() {
  const { subscribe, set, update } = writable<NewsState>({
    items: [],
    isLoading: false,
    error: null,
    filter: { ...defaultFilter },
    lastUpdate: null,
    wsConnected: false
  });

  let pollInterval: ReturnType<typeof setInterval> | null = null;
  let wsConnected = false;
  let wsHandler: ((msg: WebSocketMessage) => void) | null = null;
  const POLL_INTERVAL_MS = 60000; // 60 seconds

  return {
    subscribe,

    /**
     * Fetch news from API
     */
    async fetchNews(limit: number = 20) {
      update(state => ({ ...state, isLoading: true, error: null }));

      try {
        const items = await getNewsFeed(limit);
        update(state => ({
          ...state,
          items,
          isLoading: false,
          lastUpdate: new Date().toISOString()
        }));
      } catch (e) {
        const errorMessage = e instanceof Error ? e.message : 'Failed to fetch news';
        update(state => ({
          ...state,
          isLoading: false,
          error: errorMessage
        }));
      }
    },

    /**
     * Handle WebSocket news alert
     */
    handleNewsAlert(message: NewsAlertMessage) {
      const alertData = message.data;

      // Convert WebSocket message to NewsFeedItem
      const newsItem: NewsFeedItem = {
        item_id: alertData.item_id,
        headline: alertData.headline,
        published_utc: alertData.published_utc,
        related_instruments: alertData.affected_symbols,
        severity: alertData.severity,
        action_type: alertData.action_type
      };

      // Add to store
      update(state => {
        // Check if item already exists
        const exists = state.items.some(i => i.item_id === newsItem.item_id);
        if (exists) {
          // Update existing item
          return {
            ...state,
            items: state.items.map(i => i.item_id === newsItem.item_id ? newsItem : i),
            lastUpdate: new Date().toISOString()
          };
        }
        // Add new item at the beginning
        return {
          ...state,
          items: [newsItem, ...state.items],
          lastUpdate: new Date().toISOString()
        };
      });
    },

    /**
     * Connect to WebSocket for real-time news
     */
    async connectWebSocket() {
      if (wsConnected) return;

      try {
        const baseUrl = API_CONFIG.API_URL;
        const wsUrl = `${baseUrl.replace('http://', 'ws://').replace('https://', 'wss://')}/ws`;

        if (!wsClient.isConnected()) {
          await wsClient.connect();
        }

        wsClient.subscribe('news');

        wsHandler = (msg: WebSocketMessage) => {
          if (msg.type === 'news_alert') {
            this.handleNewsAlert(msg as unknown as NewsAlertMessage);
          }
        };

        wsClient.on('news_alert', wsHandler);
        wsConnected = true;
        update(state => ({ ...state, wsConnected: true }));
      } catch (e) {
        console.error('Failed to connect to news WebSocket:', e);
        // Don't fail - polling will still work
      }
    },

    /**
     * Disconnect from WebSocket
     */
    disconnectWebSocket() {
      if (!wsConnected) return;

      if (wsHandler) {
        wsClient.off('news_alert', wsHandler);
        wsHandler = null;
      }

      wsClient.unsubscribe('news');
      wsConnected = false;
      update(state => ({ ...state, wsConnected: false }));
    },

    /**
     * Add a new news item from WebSocket alert
     */
    addItem(item: NewsFeedItem) {
      update(state => {
        // Check if item already exists
        const exists = state.items.some(i => i.item_id === item.item_id);
        if (exists) {
          // Update existing item
          return {
            ...state,
            items: state.items.map(i => i.item_id === item.item_id ? item : i),
            lastUpdate: new Date().toISOString()
          };
        }
        // Add new item at the beginning
        return {
          ...state,
          items: [item, ...state.items],
          lastUpdate: new Date().toISOString()
        };
      });
    },

    /**
     * Update filter
     */
    setFilter(filter: Partial<NewsFilter>) {
      update(state => ({
        ...state,
        filter: { ...state.filter, ...filter }
      }));
    },

    /**
     * Clear filter
     */
    clearFilter() {
      update(state => ({
        ...state,
        filter: { ...defaultFilter }
      }));
    },

    /**
     * Start polling and WebSocket for updates
     */
    startPolling(limit: number = 20) {
      this.fetchNews(limit); // Fetch immediately

      if (pollInterval) {
        clearInterval(pollInterval);
      }

      pollInterval = setInterval(() => {
        this.fetchNews(limit);
      }, POLL_INTERVAL_MS);

      // Also connect to WebSocket for real-time alerts
      this.connectWebSocket();
    },

    /**
     * Stop polling and WebSocket
     */
    stopPolling() {
      if (pollInterval) {
        clearInterval(pollInterval);
        pollInterval = null;
      }

      this.disconnectWebSocket();
    },

    /**
     * Clear error
     */
    clearError() {
      update(state => ({ ...state, error: null }));
    }
  };
}

export const newsStore = createNewsStore();

// =============================================================================
// Derived Stores
// =============================================================================

/**
 * Filtered news items based on current filter
 */
export const filteredNews = derived(newsStore, ($news) => {
  const { items, filter } = $news;

  return items.filter(item => {
    // Filter by severity
    if (filter.severity !== 'all' && item.severity !== filter.severity) {
      return false;
    }

    // Filter by symbols
    if (filter.symbols.length > 0) {
      const hasMatchingSymbol = item.related_instruments.some(symbol =>
        filter.symbols.includes(symbol)
      );
      if (!hasMatchingSymbol) {
        return false;
      }
    }

    // Filter by date range
    if (filter.dateFrom || filter.dateTo) {
      const itemDate = new Date(item.published_utc);
      if (filter.dateFrom && itemDate < new Date(filter.dateFrom)) {
        return false;
      }
      if (filter.dateTo && itemDate > new Date(filter.dateTo)) {
        return false;
      }
    }

    return true;
  });
});

/**
 * Latest 5 news items (for Live Trading tile)
 */
export const latestNews = derived(newsStore, ($news) => {
  return $news.items.slice(0, 5);
});

/**
 * High severity items
 */
export const highSeverityNews = derived(newsStore, ($news) => {
  return $news.items.filter(item => item.severity === 'HIGH');
});

/**
 * Loading state
 */
export const newsLoading = derived(newsStore, ($news) => $news.isLoading);

/**
 * Error state
 */
export const newsError = derived(newsStore, ($news) => $news.error);

/**
 * Current filter
 */
export const currentNewsFilter = derived(newsStore, ($news) => $news.filter);

// =============================================================================
// Helper Functions
// =============================================================================

/**
 * Initialize news store with polling
 */
export async function initNewsStore() {
  newsStore.startPolling(20);
}

/**
 * Cleanup news store
 */
export function cleanupNewsStore() {
  newsStore.stopPolling();
}

/**
 * Get exposure count for a symbol
 * This is a placeholder - actual implementation would match against active strategies
 */
export function getExposureCount(symbols: string[]): number {
  // TODO: Implement actual exposure calculation based on active strategies
  // For now, return a mock count for demonstration
  return Math.floor(Math.random() * 10) + 1;
}
