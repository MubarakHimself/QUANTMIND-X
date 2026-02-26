import { writable, derived } from 'svelte/store';

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
