// Canvas context store for tracking current canvas
// Story 5.7: NL System Commands & Context-Aware Canvas Binding

import { writable } from 'svelte/store';

export interface CanvasContext {
  canvas: 'live_trading' | 'risk' | 'portfolio' | 'workshop' | 'research' | 'development';
  session_id: string;
  entity?: string;
}

function createCanvasContextStore() {
  const { subscribe, set, update } = writable<CanvasContext>({
    canvas: 'workshop',
    session_id: ''
  });

  return {
    subscribe,
    setCanvas: (canvas: CanvasContext['canvas']) => update(ctx => ({ ...ctx, canvas })),
    setSessionId: (session_id: string) => update(ctx => ({ ...ctx, session_id })),
    setContext: (context: Partial<CanvasContext>) => update(ctx => ({ ...ctx, ...context })),
    reset: () => set({ canvas: 'workshop', session_id: '' })
  };
}

export const canvasContextStore = createCanvasContextStore();
