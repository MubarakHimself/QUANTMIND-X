import { writable, derived } from 'svelte/store';

export interface Hook {
  name: string;
  description: string;
  enabled: boolean;
  lastExecuted?: string;
  executionCount: number;
  avgExecutionTime?: number;
  priority?: number;
}

export interface HookLogEntry {
  id: string;
  hookName: string;
  timestamp: string;
  duration: number;
  status: 'success' | 'failed';
  message?: string;
}

function createHooksStore() {
  const initialState = {
    hooks: [] as Hook[],
    logs: [] as HookLogEntry[],
    loading: false,
    error: null as string | null,
    selectedHook: null as Hook | null
  };

  const { subscribe, update, set } = writable(initialState);

  return {
    subscribe,

    setHooks: (hooks: Hook[]) => update(state => ({ ...state, hooks })),

    setLogs: (logs: HookLogEntry[]) => update(state => ({ ...state, logs })),

    addLog: (log: HookLogEntry) => update(state => ({
      ...state,
      logs: [log, ...state.logs].slice(0, 100) // Keep last 100 logs
    })),

    toggleHook: (name: string) => update(state => ({
      ...state,
      hooks: state.hooks.map(hook =>
        hook.name === name ? { ...hook, enabled: !hook.enabled } : hook
      )
    })),

    updateHook: (name: string, updates: Partial<Hook>) => update(state => ({
      ...state,
      hooks: state.hooks.map(hook =>
        hook.name === name ? { ...hook, ...updates } : hook
      )
    })),

    setSelectedHook: (hook: Hook | null) => update(state => ({
      ...state,
      selectedHook: hook
    })),

    setLoading: (loading: boolean) => update(state => ({ ...state, loading })),

    setError: (error: string | null) => update(state => ({ ...state, error })),

    clearLogs: () => update(state => ({ ...state, logs: [] })),

    reset: () => set(initialState)
  };
}

export const hooksStore = createHooksStore();

// Derived stores
export const hooks = derived(hooksStore, $store => $store.hooks);
export const enabledHooks = derived(hooksStore, $store =>
  $store.hooks.filter(h => h.enabled)
);
export const hookLogs = derived(hooksStore, $store => $store.logs);
export const hooksLoading = derived(hooksStore, $store => $store.loading);
