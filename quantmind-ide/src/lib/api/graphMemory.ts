/**
 * Graph Memory API Client
 * Provides frontend API functions for graph-based memory management
 *
 * Connects to the backend endpoints defined in src/api/graph_memory_endpoints.py
 */

const API_BASE = '/api/graph-memory';

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
// Types
// =============================================================================

export interface GraphMemoryNode {
  id: string;
  content: string;
  node_type: string;
  category: string;
  department: string | null;
  agent_id: string | null;
  session_id: string | null;
  importance: number;
  tags: string[];
  tier: string;
  created_at: string;
  updated_at: string;
  access_count: number;
  relevance_score: number | null;
}

export interface GraphMemoryEdge {
  id: string;
  source_id: string;
  target_id: string;
  relation_type: string;
  strength: number;
  created_at: string;
}

export interface GraphMemoryStats {
  total_nodes: number;
  hot: number;
  warm: number;
  cold: number;
}

export interface CompactionStatus {
  should_compact: boolean;
  current_percent: number;
  threshold_percent: number;
  hot_count: number;
  warm_count: number;
  cold_count: number;
  threshold: number;
}

export interface RecallResult {
  success: boolean;
  nodes: GraphMemoryNode[];
  count: number;
}

export interface ReflectResult {
  success: boolean;
  answer: string;
  sources: string[];
  synthesized: boolean;
}

export interface LinkResult {
  success: boolean;
  edges_created: number;
  edge_ids: string[];
}

export interface RetainRequest {
  content: string;
  source?: string;
  department?: string;
  agent_id?: string;
  session_id?: string;
  importance?: number;
  tags?: string[];
  related_to?: string[];
}

export interface RecallRequest {
  query: string;
  department?: string;
  agent_id?: string;
  tags?: string[];
  node_types?: string[];
  categories?: string[];
  min_importance?: number;
  limit?: number;
  cursor?: string;
}

export interface ReflectRequest {
  query: string;
  department?: string;
  agent_id?: string;
  context?: string;
}

export interface LinkRequest {
  source_id: string;
  target_ids: string[];
  relation_type?: string;
  strength?: number;
}

// =============================================================================
// Memory Operations
// =============================================================================

/**
 * Store a new memory in the graph
 */
export async function retainMemory(request: RetainRequest): Promise<{ success: boolean; node_id: string }> {
  return apiFetch('/retain', {
    method: 'POST',
    body: JSON.stringify(request)
  });
}

/**
 * Recall memories based on query and filters
 */
export async function recallMemories(request: RecallRequest): Promise<RecallResult> {
  return apiFetch('/recall', {
    method: 'POST',
    body: JSON.stringify(request)
  });
}

/**
 * Synthesize answer from memories (REFLECT operation)
 */
export async function reflectOnMemories(request: ReflectRequest): Promise<ReflectResult> {
  return apiFetch('/reflect', {
    method: 'POST',
    body: JSON.stringify(request)
  });
}

/**
 * Create relationship edges between memory nodes
 */
export async function linkMemories(request: LinkRequest): Promise<LinkResult> {
  return apiFetch('/link', {
    method: 'POST',
    body: JSON.stringify(request)
  });
}

// =============================================================================
// Stats & Status
// =============================================================================

/**
 * Get memory statistics
 */
export async function getMemoryStats(): Promise<GraphMemoryStats> {
  return apiFetch('/stats');
}

/**
 * Check compaction status
 */
export async function getCompactionStatus(contextPercent: number = 0): Promise<CompactionStatus> {
  return apiFetch(`/compaction?context_percent=${contextPercent}`);
}

/**
 * Trigger manual compaction
 */
export async function triggerCompaction(): Promise<{ success: boolean; nodes_archived: number }> {
  return apiFetch('/compact', {
    method: 'POST'
  });
}

// =============================================================================
// Tier Management
// =============================================================================

/**
 * Get hot (recent) memory nodes
 */
export async function getHotNodes(limit: number = 50): Promise<GraphMemoryNode[]> {
  return apiFetch(`/nodes/hot?limit=${limit}`);
}

/**
 * Get warm memory nodes
 */
export async function getWarmNodes(limit: number = 100): Promise<GraphMemoryNode[]> {
  return apiFetch(`/nodes/warm?limit=${limit}`);
}

/**
 * Get cold (archived) memory nodes
 */
export async function getColdNodes(limit: number = 100): Promise<GraphMemoryNode[]> {
  return apiFetch(`/nodes/cold?limit=${limit}`);
}

/**
 * Move a node to hot tier
 */
export async function moveNodeToHot(nodeId: string): Promise<{ success: boolean; node_id: string; new_tier: string }> {
  return apiFetch(`/nodes/${nodeId}/move-to-hot`, {
    method: 'POST'
  });
}

/**
 * Move a node to warm tier
 */
export async function moveNodeToWarm(nodeId: string): Promise<{ success: boolean; node_id: string; new_tier: string }> {
  return apiFetch(`/nodes/${nodeId}/move-to-warm`, {
    method: 'POST'
  });
}

/**
 * Move a node to cold tier
 */
export async function moveNodeToCold(nodeId: string): Promise<{ success: boolean; node_id: string; new_tier: string }> {
  return apiFetch(`/nodes/${nodeId}/move-to-cold`, {
    method: 'POST'
  });
}

/**
 * Delete a memory node
 */
export async function deleteMemoryNode(nodeId: string): Promise<{ success: boolean; node_id: string }> {
  return apiFetch(`/nodes/${nodeId}`, {
    method: 'DELETE'
  });
}

// =============================================================================
// Convenience Functions
// =============================================================================

/**
 * Search memories with common defaults
 */
export async function searchMemories(query: string, limit: number = 20): Promise<RecallResult> {
  return recallMemories({ query, limit });
}

/**
 * Get all nodes across tiers
 */
export async function getAllNodes(): Promise<{
  hot: GraphMemoryNode[];
  warm: GraphMemoryNode[];
  cold: GraphMemoryNode[];
}> {
  const [hot, warm, cold] = await Promise.all([
    getHotNodes(50),
    getWarmNodes(100),
    getColdNodes(100)
  ]);
  return { hot, warm, cold };
}
