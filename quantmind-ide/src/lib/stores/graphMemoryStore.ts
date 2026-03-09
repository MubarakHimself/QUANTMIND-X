import { writable } from 'svelte/store';

interface GraphMemoryState {
  isOpen: boolean;
}

function createGraphMemoryStore() {
  const { subscribe, set, update } = writable<GraphMemoryState>({
    isOpen: false
  });

  return {
    subscribe,
    open: () => update(state => ({ ...state, isOpen: true })),
    close: () => update(state => ({ ...state, isOpen: false })),
    toggle: () => update(state => ({ ...state, isOpen: !state.isOpen })),
    reset: () => set({ isOpen: false })
  };
}

export const graphMemoryStore = createGraphMemoryStore();
