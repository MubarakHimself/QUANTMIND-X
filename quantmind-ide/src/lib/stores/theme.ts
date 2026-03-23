// Theme Preset Store for QUANTMINDX
// Manages theme presets and wallpaper configuration

import { writable } from 'svelte/store';
import { browser } from '$app/environment';

export type ThemePreset = 'frosted-terminal' | 'ghost-panel' | 'open-air' | 'breathing-space';

export interface ThemeConfig {
  id: ThemePreset;
  name: string;
  description: string;
  colorScheme: string;
}

export const THEME_PRESETS: ThemeConfig[] = [
  {
    id: 'frosted-terminal',
    name: 'Frosted Terminal',
    description: 'Balanced glass aesthetic with deep navy undertones',
    colorScheme: 'midnight-finance'
  },
  {
    id: 'ghost-panel',
    name: 'Ghost Panel',
    description: 'Kanagawa-inspired with ocean wave accents',
    colorScheme: 'kanagawa'
  },
  {
    id: 'open-air',
    name: 'Open Air',
    description: 'Tokyo Night theme with ethereal clarity',
    colorScheme: 'tokyo-night'
  },
  {
    id: 'breathing-space',
    name: 'Breathing Space',
    description: 'Catppuccin Mocha for comfortable extended use',
    colorScheme: 'catppuccin-mocha'
  }
];

const DEFAULT_THEME: ThemePreset = 'frosted-terminal';

function createThemeStore() {
  const stored = browser ? (localStorage.getItem('theme') as ThemePreset) : null;
  const initial = stored || DEFAULT_THEME;

  const { subscribe, set, update } = writable<ThemePreset>(initial);

  // Apply theme to document on initialization
  if (browser) {
    document.documentElement.setAttribute('data-theme', initial);
  }

  return {
    subscribe,
    set: (theme: ThemePreset) => {
      if (browser) {
        localStorage.setItem('theme', theme);
        document.documentElement.setAttribute('data-theme', theme);
      }
      set(theme);
    },
    reset: () => {
      if (browser) {
        localStorage.removeItem('theme');
        document.documentElement.setAttribute('data-theme', DEFAULT_THEME);
      }
      set(DEFAULT_THEME);
    }
  };
}

function createWallpaperStore() {
  const stored = browser ? localStorage.getItem('wallpaper') : null;

  const { subscribe, set, update } = writable<string | null>(stored);

  return {
    subscribe,
    set: (wallpaper: string | null) => {
      if (browser) {
        if (wallpaper) {
          localStorage.setItem('wallpaper', wallpaper);
        } else {
          localStorage.removeItem('wallpaper');
        }
      }
      set(wallpaper);
    },
    clear: () => {
      if (browser) {
        localStorage.removeItem('wallpaper');
      }
      set(null);
    }
  };
}

function createScanLineStore() {
  const stored = browser ? localStorage.getItem('scanlines') : null;
  const initial = stored !== 'false'; // Default true

  const { subscribe, set, update } = writable<boolean>(initial);

  return {
    subscribe,
    set: (enabled: boolean) => {
      if (browser) {
        localStorage.setItem('scanlines', String(enabled));
      }
      set(enabled);
    },
    toggle: () => {
      update(enabled => {
        const newValue = !enabled;
        if (browser) {
          localStorage.setItem('scanlines', String(newValue));
        }
        return newValue;
      });
    }
  };
}

export const theme = createThemeStore();
export const wallpaper = createWallpaperStore();
export const scanlines = createScanLineStore();

// Helper to get preset by id
export function getPreset(id: ThemePreset): ThemeConfig | undefined {
  return THEME_PRESETS.find(p => p.id === id);
}
