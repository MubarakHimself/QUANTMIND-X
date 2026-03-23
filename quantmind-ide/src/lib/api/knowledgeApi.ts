/**
 * Knowledge API Client
 * Provides frontend API functions for knowledge base search and source status
 *
 * Connects to the backend endpoints:
 * - POST /api/knowledge/search - Full-text search across knowledge base
 * - GET /api/knowledge/sources - Get status of knowledge sources
 */

import { API_CONFIG } from '$lib/config/api';

// API base - use centralized config
const API_BASE = API_CONFIG.API_BASE;

/**
 * Generic fetch wrapper with error handling (mirrors skillsApi.ts pattern)
 */
async function apiFetch<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${endpoint}`, {
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

// =============================================================================
// Types
// =============================================================================

/**
 * Knowledge source status from GET /api/knowledge/sources
 */
export interface KnowledgeSourceStatus {
  id: string;              // 'articles' | 'books' | 'logs'
  type: string;
  status: string;          // 'online' | 'offline'
  document_count: number;
}

/**
 * Provenance metadata for a search result
 */
export interface KnowledgeProvenance {
  source_url: string | null;
  source_type: string;
  indexed_at_utc: string | null;
}

/**
 * Single search result from POST /api/knowledge/search
 */
export interface KnowledgeSearchResult {
  source_type: string;      // 'articles' | 'books' | 'logs'
  title: string;            // filename (last path segment after '/')
  excerpt: string;          // first 300 chars of content
  relevance_score: number;  // 0.0–1.0
  provenance: KnowledgeProvenance;
}

/**
 * Search request body
 */
export interface KnowledgeSearchRequest {
  query: string;           // min 1, max 2000 chars
  sources?: string[];      // subset of ['articles','books','logs']; null = all 3
  limit?: number;          // 1-100, default 5
}

/**
 * Search response from POST /api/knowledge/search
 */
export interface KnowledgeSearchResponse {
  results: KnowledgeSearchResult[];
  total: number;
  query: string;
  warnings: string[];     // offline instances reported here
}

// =============================================================================
// API Functions
// =============================================================================

/**
 * Get knowledge source status
 * Returns the status (online/offline) and document counts for each knowledge source
 */
export async function getKnowledgeSources(): Promise<KnowledgeSourceStatus[]> {
  return apiFetch<KnowledgeSourceStatus[]>('/knowledge/sources');
}

/**
 * Search the knowledge base
 * @param query - Search query string (1-2000 chars)
 * @param sources - Optional array of source types to filter ('articles', 'books', 'logs')
 * @param limit - Maximum number of results (1-100, default 10)
 */
export async function searchKnowledge(
  query: string,
  sources?: string[],
  limit: number = 10
): Promise<KnowledgeSearchResponse> {
  return apiFetch<KnowledgeSearchResponse>('/knowledge/search', {
    method: 'POST',
    body: JSON.stringify({
      query,
      sources: sources ?? null,
      limit
    })
  });
}

// =============================================================================
// Constants
// =============================================================================

/**
 * Source type badge colors for UI rendering
 */
export const SOURCE_BADGE_COLORS: Record<string, string> = {
  articles: '#00d4ff',   // cyan
  books:    '#00c896',   // emerald
  logs:     '#f0a500',   // amber
  personal: '#a78bfa',   // violet
  youtube:  '#00d4ff',   // cyan — matches Frosted Terminal accent
};

/**
 * Available filter options for source filtering
 */
export type SourceFilter = 'all' | 'articles' | 'books' | 'logs' | 'personal' | 'youtube';

export const SOURCE_FILTERS: { value: SourceFilter; label: string }[] = [
  { value: 'all', label: 'All' },
  { value: 'articles', label: 'Articles' },
  { value: 'books', label: 'Books' },
  { value: 'logs', label: 'Logs' },
  { value: 'personal', label: 'Personal' },
  { value: 'youtube', label: 'YouTube' },
];
