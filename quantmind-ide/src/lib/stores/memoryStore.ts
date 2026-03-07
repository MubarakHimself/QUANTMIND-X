import { writable, derived, get } from 'svelte/store';
import * as memoryApi from '$lib/api/memory';

export interface MemoryEntry {
  id: string;
  content: string;
  namespace: string;
  key: string;
  timestamp: string;
  decay?: number; // 0-1, temporal decay factor
  agent?: string;
  tags?: string[];
  embedding_model?: string;
}

export interface MemoryStats {
  total_count: number;
  namespace_counts: Record<string, number>;
  last_sync: string;
  embedding_model: string;
  oldest_memory?: string;
  newest_memory?: string;
}

export interface MemoryFilters {
  namespace: 'all' | 'default' | 'patterns' | 'solutions' | 'sessions' | 'tasks';
  agent: string;
  searchQuery: string;
  minDecay: number;
}

function createMemoryStore() {
  const initialState = {
    memories: [] as MemoryEntry[],
    filteredMemories: [] as MemoryEntry[],
    selectedMemory: null as MemoryEntry | null,
    stats: null as MemoryStats | null,
    loading: false,
    error: null as string | null,
    filters: {
      namespace: 'all' as const,
      agent: '',
      searchQuery: '',
      minDecay: 0
    } as MemoryFilters
  };

  const { subscribe, update, set } = writable(initialState);

  return {
    subscribe,

    // Load all memories from backend
    loadMemories: async (namespace?: string, limit: number = 100) => {
      update(state => ({ ...state, loading: true, error: null }));
      try {
        const result = await memoryApi.listMemoriesForStore(namespace, limit);
        update(state => {
          const newState = {
            ...state,
            memories: result.memories,
            loading: false
          };
          newState.filteredMemories = applyFilters(newState.memories, state.filters);
          return newState;
        });
      } catch (e) {
        update(state => ({
          ...state,
          loading: false,
          error: e instanceof Error ? e.message : 'Failed to load memories'
        }));
      }
    },

    // Add a new memory entry
    addMemory: async (
      content: string,
      namespace: string = 'default',
      tags: string[] = []
    ) => {
      update(state => ({ ...state, loading: true, error: null }));
      try {
        const entry = await memoryApi.addMemory({
          content,
          source: namespace,
          agent_id: null,
          metadata: { tags }
        });
        // Reload memories after adding
        const store = memoryStore;
        await store.loadMemories();
        return entry;
      } catch (e) {
        update(state => ({
          ...state,
          loading: false,
          error: e instanceof Error ? e.message : 'Failed to add memory'
        }));
        throw e;
      }
    },

    // Delete a memory entry
    deleteMemory: async (memoryId: string) => {
      update(state => ({ ...state, loading: true, error: null }));
      try {
        await memoryApi.deleteMemory(memoryId);
        // Reload memories after deletion
        const store = memoryStore;
        await store.loadMemories();
      } catch (e) {
        update(state => ({
          ...state,
          loading: false,
          error: e instanceof Error ? e.message : 'Failed to delete memory'
        }));
        throw e;
      }
    },

    // Search memories
    searchMemories: async (query: string, namespace?: string) => {
      update(state => ({ ...state, loading: true, error: null }));
      try {
        const result = await memoryApi.searchAgentMemory(query, namespace, 20);
        const memories = result.results.map(r => ({
          id: r.id || '',
          content: r.content,
          namespace: r.source || namespace || 'default',
          key: (r.metadata?.key as string) || r.id || '',
          timestamp: r.created_at || new Date().toISOString(),
          tags: (r.metadata?.tags as string[]) || []
        }));
        update(state => {
          const newState = {
            ...state,
            filteredMemories: memories,
            loading: false
          };
          return newState;
        });
        return memories;
      } catch (e) {
        update(state => ({
          ...state,
          loading: false,
          error: e instanceof Error ? e.message : 'Search failed'
        }));
        throw e;
      }
    },

    // Load stats from backend
    loadStats: async () => {
      try {
        const stats = await memoryApi.getMemoryStatsForStore();
        update(state => ({ ...state, stats }));
      } catch (e) {
        console.error('Failed to load memory stats:', e);
      }
    },

    // Clear all memories
    clearMemories: async (namespace?: string) => {
      update(state => ({ ...state, loading: true, error: null }));
      try {
        await memoryApi.clearMemories(namespace);
        // Reload after clearing
        const store = memoryStore;
        await store.loadMemories();
        await store.loadStats();
      } catch (e) {
        update(state => ({
          ...state,
          loading: false,
          error: e instanceof Error ? e.message : 'Failed to clear memories'
        }));
        throw e;
      }
    },

    setMemories: (memories: MemoryEntry[]) => update(state => {
      const newState = { ...state, memories };
      newState.filteredMemories = applyFilters(newState.memories, state.filters);
      return newState;
    }),

    setSelectedMemory: (memory: MemoryEntry | null) => update(state => ({
      ...state,
      selectedMemory: memory
    })),

    setStats: (stats: MemoryStats) => update(state => ({ ...state, stats })),

    setLoading: (loading: boolean) => update(state => ({ ...state, loading })),

    setError: (error: string | null) => update(state => ({ ...state, error })),

    setFilters: (filters: Partial<MemoryFilters>) => update(state => {
      const newFilters = { ...state.filters, ...filters };
      const newState = { ...state, filters: newFilters };
      newState.filteredMemories = applyFilters(state.memories, newFilters);
      return newState;
    }),

    reset: () => set(initialState)
  };
}

function applyFilters(memories: MemoryEntry[], filters: MemoryFilters): MemoryEntry[] {
  let filtered = [...memories];

  // Namespace filter
  if (filters.namespace !== 'all') {
    filtered = filtered.filter(m => m.namespace === filters.namespace);
  }

  // Agent filter
  if (filters.agent) {
    filtered = filtered.filter(m => m.agent === filters.agent);
  }

  // Search query filter
  if (filters.searchQuery) {
    const query = filters.searchQuery.toLowerCase();
    filtered = filtered.filter(m =>
      m.content.toLowerCase().includes(query) ||
      m.key.toLowerCase().includes(query) ||
      m.tags?.some(t => t.toLowerCase().includes(query))
    );
  }

  // Decay filter (only show memories with decay >= minDecay)
  filtered = filtered.filter(m => (m.decay ?? 1) >= filters.minDecay);

  // Sort by timestamp (newest first)
  filtered.sort((a, b) =>
    new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
  );

  return filtered;
}

export const memoryStore = createMemoryStore();

// Derived stores for convenience
export const memories = derived(memoryStore, $store => $store.memories);
export const filteredMemories = derived(memoryStore, $store => $store.filteredMemories);
export const selectedMemory = derived(memoryStore, $store => $store.selectedMemory);
export const memoryStats = derived(memoryStore, $store => $store.stats);
export const memoryLoading = derived(memoryStore, $store => $store.loading);
export const memoryError = derived(memoryStore, $store => $store.error);
export const memoryFilters = derived(memoryStore, $store => $store.filters);

// Re-export store methods for convenience
export const {
  loadMemories,
  addMemory,
  deleteMemory,
  searchMemories,
  loadStats,
  clearMemories
} = memoryStore;
