/**
 * Memory API Client
 * Provides frontend API functions for memory management endpoints
 *
 * Wraps the backend endpoints defined in src/api/memory_endpoints.py
 */

const API_BASE = 'http://localhost:8000/api';

/**
 * Generic fetch wrapper with error handling
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
// Store-Compatible Types (for Svelte stores)
// =============================================================================

/**
 * Store-compatible memory entry
 */
export interface StoreMemoryEntry {
  id: string;
  content: string;
  namespace: string;
  key: string;
  timestamp: string;
  decay?: number;
  agent?: string;
  tags?: string[];
  embedding_model?: string;
}

/**
 * Store-compatible memory stats
 */
export interface StoreMemoryStats {
  total_count: number;
  namespace_counts: Record<string, number>;
  last_sync: string;
  embedding_model: string;
  oldest_memory?: string;
  newest_memory?: string;
}

/**
 * Store-compatible cron job
 */
export interface StoreCronJob {
  id: string;
  name: string;
  enabled: boolean;
  schedule: string;
  command: string;
  lastRun?: string;
  nextRun?: string;
  status: 'idle' | 'running' | 'success' | 'failed';
  lastStatus?: 'success' | 'failed';
  executionTime?: number;
  description?: string;
}

/**
 * Store-compatible hook
 */
export interface StoreHook {
  name: string;
  description: string;
  enabled: boolean;
  lastExecuted?: string;
  executionCount: number;
  avgExecutionTime?: number;
  priority?: number;
}

// =============================================================================
// Type Definitions
// =============================================================================

/**
 * Memory entry representing a stored memory
 */
export interface MemoryEntry {
  id: string | null;
  content: string;
  source: string; // 'memory' | 'sessions' | etc.
  agent_id: string | null;
  metadata: Record<string, unknown>;
  created_at: string | null;
  updated_at: string | null;
  relevance_score: number | null;
}

/**
 * Memory system statistics
 */
export interface MemoryStats {
  total_memories: number;
  total_sessions: number;
  embedding_model: string;
  last_sync: string | null;
  vector_dimensions: number | null;
  sources: string[];
}

/**
 * Request body for memory search
 */
export interface MemorySearchRequest {
  query: string;
  limit?: number;
  source?: string;
  agent_id?: string;
  min_relevance?: number;
  use_temporal_decay?: boolean;
}

/**
 * Response from memory search
 */
export interface MemorySearchResponse {
  results: MemoryEntry[];
  total: number;
  query: string;
  elapsed_ms: number;
}

/**
 * Request body for memory sync
 */
export interface MemorySyncRequest {
  force?: boolean;
  sources?: string[];
}

/**
 * Hook definition
 */
export interface MemoryHook {
  id: string;
  name: string;
  enabled: boolean;
  priority: number;
}

/**
 * Cron job definition
 */
export interface MemoryCronJob {
  id: string;
  name: string;
  schedule: string;
  enabled: boolean;
  last_run: string | null;
  next_run: string | null;
}

/**
 * Response for hooks list
 */
export interface HooksListResponse {
  hooks: MemoryHook[];
}

/**
 * Response for cron jobs list
 */
export interface CronJobsListResponse {
  jobs: MemoryCronJob[];
}

/**
 * Generic status response
 */
export interface StatusResponse {
  status: string;
  [key: string]: unknown;
}

/**
 * Hook log entry for execution history
 */
export interface HookLogEntry {
  id: string;
  hook_name: string;
  executed_at: string;
  status: 'success' | 'error' | 'pending';
  duration_ms: number | null;
  result: string | null;
  error: string | null;
}

/**
 * Cron job definition (simplified for frontend use)
 */
export interface CronJob {
  id: string;
  name: string;
  schedule: string;
  enabled: boolean;
  command?: string;
  last_run: string | null;
  next_run: string | null;
}

/**
 * Response for hook logs list
 */
export interface HookLogsResponse {
  logs: HookLogEntry[];
  total: number;
}

// =============================================================================
// Memory CRUD Operations
// =============================================================================

/**
 * Get memory system statistics
 */
export async function getMemoryStats(): Promise<MemoryStats> {
  return apiFetch<MemoryStats>('/memory/stats');
}

/**
 * Search memories using semantic search
 * @param request - Search parameters
 */
export async function searchMemories(request: MemorySearchRequest): Promise<MemorySearchResponse> {
  return apiFetch<MemorySearchResponse>('/memory/search', {
    method: 'POST',
    body: JSON.stringify(request)
  });
}

/**
 * Add a new memory entry
 * @param entry - Memory entry to add
 */
export async function addMemory(
  entry: Omit<MemoryEntry, 'id' | 'created_at' | 'updated_at'>
): Promise<MemoryEntry> {
  return apiFetch<MemoryEntry>('/memory/add', {
    method: 'POST',
    body: JSON.stringify(entry)
  });
}

/**
 * Get a specific memory by ID
 * @param memoryId - The memory ID to retrieve
 */
export async function getMemory(memoryId: string): Promise<MemoryEntry> {
  return apiFetch<MemoryEntry>(`/memory/${memoryId}`);
}

/**
 * Delete a memory entry
 * @param memoryId - The memory ID to delete
 */
export async function deleteMemory(memoryId: string): Promise<StatusResponse> {
  return apiFetch<StatusResponse>(`/memory/${memoryId}`, {
    method: 'DELETE'
  });
}

/**
 * Sync memory index with files and sessions
 * @param request - Sync options
 */
export async function syncMemories(request?: MemorySyncRequest): Promise<StatusResponse> {
  const defaultRequest: MemorySyncRequest = {
    force: false,
    sources: ['memory', 'sessions']
  };
  return apiFetch<StatusResponse>('/memory/sync', {
    method: 'POST',
    body: JSON.stringify(request ?? defaultRequest)
  });
}

/**
 * Clear memories, optionally filtered by source or agent
 * @param source - Optional source filter
 * @param agentId - Optional agent ID filter
 */
export async function clearMemories(source?: string, agentId?: string): Promise<StatusResponse> {
  const params = new URLSearchParams();
  if (source) params.append('source', source);
  if (agentId) params.append('agent_id', agentId);

  const queryString = params.toString();
  const endpoint = queryString ? `/memory/clear?${queryString}` : '/memory/clear';

  return apiFetch<StatusResponse>(endpoint, {
    method: 'POST'
  });
}

// =============================================================================
// Hook Management
// =============================================================================

/**
 * List all registered hooks
 */
export async function listHooks(): Promise<HooksListResponse> {
  return apiFetch<HooksListResponse>('/memory/hooks');
}

/**
 * Enable or disable a hook
 * @param hookId - The hook ID to toggle
 * @param enabled - Whether to enable or disable
 */
export async function toggleHook(hookId: string, enabled: boolean): Promise<StatusResponse> {
  return apiFetch<StatusResponse>(`/memory/hooks/${hookId}/toggle?enabled=${enabled}`, {
    method: 'POST'
  });
}

// =============================================================================
// Cron Job Management
// =============================================================================

/**
 * List all scheduled cron jobs
 */
export async function listCronJobs(): Promise<CronJobsListResponse> {
  return apiFetch<CronJobsListResponse>('/memory/cron');
}

/**
 * Manually trigger a cron job
 * @param jobId - The cron job ID to run
 */
export async function runCronJob(jobId: string): Promise<StatusResponse> {
  return apiFetch<StatusResponse>(`/memory/cron/${jobId}/run`, {
    method: 'POST'
  });
}

/**
 * Enable or disable a cron job
 * @param jobId - The cron job ID to toggle
 * @param enabled - Whether to enable or disable
 */
export async function toggleCronJob(jobId: string, enabled: boolean): Promise<StatusResponse> {
  return apiFetch<StatusResponse>(`/memory/cron/${jobId}/toggle?enabled=${enabled}`, {
    method: 'POST'
  });
}

// =============================================================================
// List Operations (MemoryPanel)
// =============================================================================

/**
 * List memories with optional namespace filter and limit
 * @param namespace - Optional namespace to filter by
 * @param limit - Maximum number of memories to return
 */
export async function listMemories(
  namespace?: string,
  limit?: number
): Promise<{ memories: MemoryEntry[] }> {
  const params = new URLSearchParams();
  if (namespace) params.append('namespace', namespace);
  if (limit) params.append('limit', String(limit));

  const queryString = params.toString();
  const endpoint = queryString ? `/memory/list?${queryString}` : '/memory/list';

  return apiFetch<{ memories: MemoryEntry[] }>(endpoint);
}

// =============================================================================
// Hook Logs & Execution (HookViewer)
// =============================================================================

/**
 * Get hook execution logs
 * @param limit - Maximum number of log entries to return
 */
export async function getHookLogs(limit?: number): Promise<HookLogEntry[]> {
  const params = limit ? `?limit=${limit}` : '';
  const response = await apiFetch<HookLogsResponse>(`/memory/hooks/logs${params}`);
  return response.logs;
}

/**
 * Execute a hook by name
 * @param hookName - The name of the hook to execute
 */
export async function executeHook(hookName: string): Promise<StatusResponse> {
  return apiFetch<StatusResponse>(`/memory/hooks/${encodeURIComponent(hookName)}/execute`, {
    method: 'POST'
  });
}

/**
 * Clear all hook logs
 */
export async function clearHookLogs(): Promise<StatusResponse> {
  return apiFetch<StatusResponse>('/memory/hooks/logs/clear', {
    method: 'POST'
  });
}

// =============================================================================
// Cron Job CRUD (CronJobManager)
// =============================================================================

/**
 * Add a new cron job
 * @param job - Cron job configuration (without id)
 */
export async function addCronJob(job: Omit<CronJob, 'id'>): Promise<CronJob> {
  return apiFetch<CronJob>('/memory/cron', {
    method: 'POST',
    body: JSON.stringify(job)
  });
}

/**
 * Delete a cron job by ID
 * @param jobId - The cron job ID to delete
 */
export async function deleteCronJob(jobId: string): Promise<StatusResponse> {
  return apiFetch<StatusResponse>(`/memory/cron/${jobId}`, {
    method: 'DELETE'
  });
}

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * Create a new memory entry with defaults
 * @param content - The memory content
 * @param options - Optional parameters
 */
export function createMemoryEntry(
  content: string,
  options?: {
    source?: string;
    agent_id?: string;
    metadata?: Record<string, unknown>;
  }
): Omit<MemoryEntry, 'id' | 'created_at' | 'updated_at' | 'relevance_score'> {
  return {
    content,
    source: options?.source ?? 'memory',
    agent_id: options?.agent_id ?? null,
    metadata: options?.metadata ?? {}
  };
}

/**
 * Format memory stats for display
 * @param stats - Memory statistics
 */
export function formatMemoryStats(stats: MemoryStats): string {
  const lines = [
    `Total Memories: ${stats.total_memories}`,
    `Total Sessions: ${stats.total_sessions}`,
    `Embedding Model: ${stats.embedding_model}`,
    `Vector Dimensions: ${stats.vector_dimensions ?? 'N/A'}`,
    `Last Sync: ${stats.last_sync ? new Date(stats.last_sync).toLocaleString() : 'Never'}`,
    `Sources: ${stats.sources.join(', ')}`
  ];
  return lines.join('\n');
}

/**
 * Format search results for display
 * @param response - Search response
 */
export function formatSearchResults(response: MemorySearchResponse): string {
  if (response.results.length === 0) {
    return `No results found for "${response.query}" (${response.elapsed_ms.toFixed(2)}ms)`;
  }

  const header = `Found ${response.total} results for "${response.query}" (${response.elapsed_ms.toFixed(2)}ms)\n`;
  const results = response.results
    .map((entry, index) => {
      const relevance = entry.relevance_score ? ` (relevance: ${entry.relevance_score.toFixed(3)})` : '';
      const source = entry.source ? ` [${entry.source}]` : '';
      const preview = entry.content.length > 100 ? entry.content.substring(0, 100) + '...' : entry.content;
      return `${index + 1}. ${preview}${source}${relevance}`;
    })
    .join('\n');

  return header + results;
}

// =============================================================================
// Store-Compatible API Functions (with adapters)
// =============================================================================

/**
 * Convert API MemoryEntry to Store-compatible format
 */
function toStoreMemory(entry: MemoryEntry): StoreMemoryEntry {
  return {
    id: entry.id ?? '',
    content: entry.content,
    namespace: (entry.metadata?.namespace as string) ?? entry.source ?? 'default',
    key: (entry.metadata?.key as string) ?? entry.id ?? '',
    timestamp: entry.created_at ?? new Date().toISOString(),
    decay: entry.metadata?.decay as number | undefined,
    agent: entry.agent_id ?? undefined,
    tags: entry.metadata?.tags as string[] | undefined,
    embedding_model: entry.metadata?.embedding_model as string | undefined
  };
}

/**
 * Convert API MemoryStats to Store-compatible format
 */
function toStoreStats(stats: MemoryStats): StoreMemoryStats {
  return {
    total_count: stats.total_memories,
    namespace_counts: (stats.sources || []).reduce((acc, source) => {
      acc[source] = 1;
      return acc;
    }, {} as Record<string, number>),
    last_sync: stats.last_sync ?? new Date().toISOString(),
    embedding_model: stats.embedding_model,
    oldest_memory: undefined,
    newest_memory: undefined
  };
}

/**
 * Convert API MemoryCronJob to Store-compatible format
 */
function toStoreCronJob(job: MemoryCronJob): StoreCronJob {
  return {
    id: job.id,
    name: job.name,
    enabled: job.enabled,
    schedule: job.schedule,
    command: '',
    lastRun: job.last_run ?? undefined,
    nextRun: job.next_run ?? undefined,
    status: 'idle',
    description: ''
  };
}

/**
 * Convert API MemoryHook to Store-compatible format
 */
function toStoreHook(hook: MemoryHook): StoreHook {
  return {
    name: hook.name,
    description: '',
    enabled: hook.enabled,
    lastExecuted: undefined,
    executionCount: 0,
    priority: hook.priority
  };
}

/**
 * List memories with optional namespace filter (returns store-compatible format)
 */
export async function listMemoriesForStore(
  namespace?: string,
  limit?: number
): Promise<{ memories: StoreMemoryEntry[] }> {
  const result = await listMemories(namespace, limit);
  return {
    memories: result.memories.map(toStoreMemory)
  };
}

/**
 * Get memory stats (returns store-compatible format)
 */
export async function getMemoryStatsForStore(): Promise<StoreMemoryStats> {
  const stats = await getMemoryStats();
  return toStoreStats(stats);
}

/**
 * List cron jobs (returns store-compatible format)
 */
export async function listCronJobsForStore(): Promise<{ jobs: StoreCronJob[] }> {
  const result = await listCronJobs();
  return {
    jobs: result.jobs.map(toStoreCronJob)
  };
}

/**
 * List hooks (returns store-compatible format)
 */
export async function listHooksForStore(): Promise<{ hooks: StoreHook[] }> {
  const result = await listHooks();
  return {
    hooks: result.hooks.map(toStoreHook)
  };
}

// =============================================================================
// Agent Memory Integration (AgentDB-style with SQLite backend)
// =============================================================================

/**
 * Store a memory entry with explicit namespace (for AgentMemory backend)
 * This provides direct integration with the AgentMemory SQLite backend
 */
export async function storeAgentMemory(
  key: string,
  value: string,
  namespace: string = "default",
  tags: string[] = [],
  metadata: Record<string, unknown> = {}
): Promise<MemoryEntry> {
  return apiFetch<MemoryEntry>('/memory/add', {
    method: 'POST',
    body: JSON.stringify({
      content: value,
      source: namespace,
      agent_id: null,
      metadata: { key, tags, ...metadata }
    })
  });
}

/**
 * Retrieve a specific memory by key from AgentMemory backend
 */
export async function retrieveAgentMemory(
  key: string,
  namespace: string = "default"
): Promise<MemoryEntry | null> {
  try {
    const result = await apiFetch<MemoryEntry>(`/memory/${key}`);
    return result;
  } catch (error) {
    // Return null if not found
    return null;
  }
}

/**
 * Search memories with advanced options
 */
export async function searchAgentMemory(
  query: string,
  namespace?: string,
  limit: number = 10
): Promise<MemorySearchResponse> {
  return apiFetch<MemorySearchResponse>('/memory/search', {
    method: 'POST',
    body: JSON.stringify({
      query,
      limit,
      source: namespace,
      agent_id: null,
      min_relevance: 0.0,
      use_temporal_decay: true
    })
  });
}

/**
 * Clear all memories from a specific namespace
 */
export async function clearNamespaceMemories(namespace: string): Promise<StatusResponse> {
  return clearMemories(namespace);
}

/**
 * Get memory statistics with detailed breakdown
 */
export async function getDetailedMemoryStats(): Promise<MemoryStats> {
  return getMemoryStats();
}
