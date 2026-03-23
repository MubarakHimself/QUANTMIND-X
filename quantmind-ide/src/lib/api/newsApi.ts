/**
 * News API Client
 * Provides frontend API functions for news feed management
 *
 * Connects to the backend endpoints defined in src/api/news_endpoints.py
 */

import { API_CONFIG } from '$lib/config/api';

// =============================================================================
// Types
// =============================================================================

export interface NewsFeedItem {
  item_id: string;
  headline: string;
  summary?: string;
  source?: string;
  published_utc: string;        // ISO 8601 string
  url?: string;
  related_instruments: string[];
  severity?: 'LOW' | 'MEDIUM' | 'HIGH' | null;
  action_type?: 'MONITOR' | 'ALERT' | 'FAST_TRACK' | null;
}

export interface NewsFeedResponse {
  success: boolean;
  items: NewsFeedItem[];
}

export interface NewsAlertRequest {
  item_id: string;
  headline: string;
  severity: 'HIGH' | 'MEDIUM' | 'LOW';
  action_type: 'ALERT' | 'FAST_TRACK' | 'MONITOR';
  affected_symbols: string[];
  published_utc: string;
}

export interface NewsAlertResponse {
  stored: boolean;
  broadcast: boolean;
  item_id: string;
}

// =============================================================================
// API Functions
// =============================================================================

/**
 * Generic fetch wrapper with error handling
 */
async function apiFetch<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_CONFIG.API_BASE}${endpoint}`, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options?.headers
    }
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`API Error: ${response.status} ${response.statusText} - ${errorText}`);
  }

  return response.json();
}

/**
 * Get the latest news feed items
 * @param limit - Number of items to fetch (default: 20)
 */
export async function getNewsFeed(limit?: number): Promise<NewsFeedItem[]> {
  const params = limit ? `?limit=${encodeURIComponent(limit)}` : '';
  return apiFetch<NewsFeedItem[]>(`/news/feed${params}`);
}

/**
 * Post a news alert to be stored and broadcast
 */
export async function postNewsAlert(alert: NewsAlertRequest): Promise<NewsAlertResponse> {
  return apiFetch<NewsAlertResponse>('/news/alert', {
    method: 'POST',
    body: JSON.stringify(alert)
  });
}
