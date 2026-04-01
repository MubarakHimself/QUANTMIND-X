// Canvas Store - Manages active canvas state
import { writable } from 'svelte/store';

export interface Canvas {
  id: string;
  name: string;
  route: string;
  epic: number;
  epicName: string;
}

// 9 Canvases defined in UX spec
export const CANVASES: Canvas[] = [
  { id: 'live-trading', name: 'Live Trading', route: 'live-trading', epic: 3, epicName: 'Live Trading Command Center' },
  { id: 'research', name: 'Research', route: 'research', epic: 6, epicName: 'Knowledge & Research Engine' },
  { id: 'development', name: 'Development', route: 'development', epic: 8, epicName: 'Alpha Forge — Strategy Factory' },
  { id: 'risk', name: 'Risk', route: 'risk', epic: 4, epicName: 'Risk Management & Compliance' },
  { id: 'trading', name: 'Trading', route: 'trading', epic: 3, epicName: 'Live Trading Command Center' },
  { id: 'portfolio', name: 'Portfolio', route: 'portfolio', epic: 9, epicName: 'Portfolio & Multi-Broker Management' },
  { id: 'shared-assets', name: 'Shared Assets', route: 'shared-assets', epic: 6, epicName: 'Knowledge & Research Engine' },
  { id: 'workshop', name: 'Workshop', route: 'workshop', epic: 5, epicName: 'Unified Memory & Copilot Core' },
  { id: 'flowforge', name: 'FlowForge', route: 'flowforge', epic: 8, epicName: 'Alpha Forge — Strategy Factory' },
];

// Keyboard shortcuts mapping (1-9)
export const CANVAS_SHORTCUTS: Record<string, string> = {
  '1': 'live-trading',
  '2': 'research',
  '3': 'development',
  '4': 'risk',
  '5': 'trading',
  '6': 'portfolio',
  '7': 'shared-assets',
  '8': 'workshop',
  '9': 'flowforge',
};

// Canvas store using Svelte writable store
function createCanvasStore() {
  const { subscribe, set, update } = writable<string>('live-trading'); // Default to live trading

  return {
    subscribe,
    setActiveCanvas: (canvasId: string) => set(canvasId),
    getCanvas: (canvasId: string) => CANVASES.find(c => c.id === canvasId),
    getCanvasByRoute: (route: string) => CANVASES.find(c => c.route === route),
  };
}

export const activeCanvasStore = createCanvasStore();
