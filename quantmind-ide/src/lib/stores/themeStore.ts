import { writable, derived } from 'svelte/store';

export type ThemeName = 'trading-terminal' | 'monokai' | 'ambient' | 'cyberpunk' | 'matrix' | 'dark-pro' | 'bloomberg' | 'crypto-quant' | 'gold-futures' | 'forex-pro' | 'midnight-quant' | 'ocean-blue' | 'deep-space' | 'nordic-frost';

export interface Theme {
  name: ThemeName;
  displayName: string;
  description: string;
  colors: {
    bg: {
      primary: string;
      secondary: string;
      tertiary: string;
      glass: string;
    };
    text: {
      primary: string;
      secondary: string;
      muted: string;
      accent: string;
    };
    border: {
      subtle: string;
      medium: string;
      accent: string;
    };
    accent: {
      primary: string;
      secondary: string;
      success: string;
      warning: string;
      danger: string;
    };
    syntax: {
      keyword: string;
      string: string;
      number: string;
      comment: string;
      function: string;
      variable: string;
      operator: string;
      background: string;
    };
  };
  effects: {
    glass: boolean;
    glow: boolean;
    scanlines: boolean;
    animated: boolean;
    gradients: boolean;
  };
  wallpaper?: {
    type: 'gradient' | 'pattern' | 'animated';
    value: string;
    gradient?: string;
  };
}

export const themes: Record<ThemeName, Theme> = {
  'trading-terminal': {
    name: 'trading-terminal',
    displayName: 'Trading Terminal',
    description: 'Retro-futuristic terminal with CRT effects',
    colors: {
      bg: {
        primary: '#0a0a0a',
        secondary: '#0f0f0f',
        tertiary: '#141414',
        glass: 'rgba(0, 255, 0, 0.05)'
      },
      text: {
        primary: '#00ff00',
        secondary: '#00cc00',
        muted: '#008800',
        accent: '#00ff88'
      },
      border: {
        subtle: '#003300',
        medium: '#006600',
        accent: '#00ff00'
      },
      accent: {
        primary: '#00ff00',
        secondary: '#00cc00',
        success: '#00ff00',
        warning: '#ffaa00',
        danger: '#ff0000'
      },
      syntax: {
        keyword: '#00ff88',
        string: '#00ff00',
        number: '#ffff00',
        comment: '#008800',
        function: '#00ffff',
        variable: '#00ff00',
        operator: '#ff00ff',
        background: '#0a0a0a'
      }
    },
    effects: {
      glass: true,
      glow: true,
      scanlines: true,
      animated: true,
      gradients: false
    },
    wallpaper: {
      type: 'pattern',
      value: 'terminal-grid'
    }
  },
  
  'monokai': {
    name: 'monokai',
    displayName: 'Monokai Pro',
    description: 'Classic dark theme with vibrant colors',
    colors: {
      bg: {
        primary: '#272822',
        secondary: '#1e1f1c',
        tertiary: '#2d2e2a',
        glass: 'rgba(39, 40, 34, 0.8)'
      },
      text: {
        primary: '#f8f8f2',
        secondary: '#e6e6dc',
        muted: '#75715e',
        accent: '#66d9ef'
      },
      border: {
        subtle: '#3e3d32',
        medium: '#49483e',
        accent: '#75715e'
      },
      accent: {
        primary: '#66d9ef',
        secondary: '#a6e22e',
        success: '#a6e22e',
        warning: '#fd971f',
        danger: '#f92672'
      },
      syntax: {
        keyword: '#f92672',
        string: '#e6db74',
        number: '#ae81ff',
        comment: '#75715e',
        function: '#66d9ef',
        variable: '#f8f8f2',
        operator: '#f92672',
        background: '#272822'
      }
    },
    effects: {
      glass: false,
      glow: false,
      scanlines: false,
      animated: false,
      gradients: false
    }
  },
  
  'ambient': {
    name: 'ambient',
    displayName: 'Ambient Lighting',
    description: 'Soft ambient colors with warm lighting',
    colors: {
      bg: {
        primary: '#1a1a1a',
        secondary: '#1f1f1f',
        tertiary: '#242424',
        glass: 'rgba(255, 147, 41, 0.05)'
      },
      text: {
        primary: '#ffffff',
        secondary: '#e0e0e0',
        muted: '#999999',
        accent: '#ff9329'
      },
      border: {
        subtle: '#333333',
        medium: '#4a4a4a',
        accent: '#ff9329'
      },
      accent: {
        primary: '#ff9329',
        secondary: '#ff6b35',
        success: '#4caf50',
        warning: '#ff9329',
        danger: '#f44336'
      },
      syntax: {
        keyword: '#ff9329',
        string: '#4caf50',
        number: '#2196f3',
        comment: '#666666',
        function: '#9c27b0',
        variable: '#ffffff',
        operator: '#ff9329',
        background: '#1a1a1a'
      }
    },
    effects: {
      glass: true,
      glow: true,
      scanlines: false,
      animated: true,
      gradients: true
    },
    wallpaper: {
      type: 'gradient',
      value: 'linear-gradient(135deg, #1a1a1a 0%, #2d1b1b 100%)'
    }
  },
  
  'cyberpunk': {
    name: 'cyberpunk',
    displayName: 'Cyberpunk',
    description: 'Neon cyberpunk theme with vibrant colors',
    colors: {
      bg: {
        primary: '#0d0221',
        secondary: '#1a0033',
        tertiary: '#26004d',
        glass: 'rgba(255, 0, 255, 0.1)'
      },
      text: {
        primary: '#ffffff',
        secondary: '#e0e0ff',
        muted: '#9999cc',
        accent: '#00ffff'
      },
      border: {
        subtle: '#1a0033',
        medium: '#330066',
        accent: '#ff00ff'
      },
      accent: {
        primary: '#00ffff',
        secondary: '#ff00ff',
        success: '#00ff00',
        warning: '#ffff00',
        danger: '#ff0066'
      },
      syntax: {
        keyword: '#ff00ff',
        string: '#00ffff',
        number: '#ffff00',
        comment: '#666699',
        function: '#00ff00',
        variable: '#ffffff',
        operator: '#ff00ff',
        background: '#0d0221'
      }
    },
    effects: {
      glass: true,
      glow: true,
      scanlines: false,
      animated: true,
      gradients: true
    },
    wallpaper: {
      type: 'gradient',
      value: 'linear-gradient(45deg, #0d0221 0%, #1a0033 50%, #26004d 100%)'
    }
  },
  
  'matrix': {
    name: 'matrix',
    displayName: 'Matrix',
    description: 'Matrix-style green terminal theme',
    colors: {
      bg: {
        primary: '#000000',
        secondary: '#0a0a0a',
        tertiary: '#0f0f0f',
        glass: 'rgba(0, 255, 0, 0.03)'
      },
      text: {
        primary: '#00ff00',
        secondary: '#00cc00',
        muted: '#008800',
        accent: '#00ff88'
      },
      border: {
        subtle: '#001100',
        medium: '#003300',
        accent: '#00ff00'
      },
      accent: {
        primary: '#00ff00',
        secondary: '#00cc00',
        success: '#00ff00',
        warning: '#ffaa00',
        danger: '#ff0000'
      },
      syntax: {
        keyword: '#00ff88',
        string: '#00ff00',
        number: '#ffff00',
        comment: '#006600',
        function: '#00ffff',
        variable: '#00ff00',
        operator: '#ff00ff',
        background: '#000000'
      }
    },
    effects: {
      glass: true,
      glow: true,
      scanlines: true,
      animated: true,
      gradients: false
    },
    wallpaper: {
      type: 'pattern',
      value: 'matrix-rain'
    }
  },
  
  'dark-pro': {
    name: 'dark-pro',
    displayName: 'Dark Professional',
    description: 'Clean professional dark theme',
    colors: {
      bg: {
        primary: '#1e1e1e',
        secondary: '#252526',
        tertiary: '#2d2d30',
        glass: 'rgba(56, 56, 56, 0.4)'
      },
      text: {
        primary: '#cccccc',
        secondary: '#9d9d9d',
        muted: '#6d6d6d',
        accent: '#007acc'
      },
      border: {
        subtle: '#3e3e42',
        medium: '#464647',
        accent: '#007acc'
      },
      accent: {
        primary: '#007acc',
        secondary: '#4fc1ff',
        success: '#4ec9b0',
        warning: '#ce9178',
        danger: '#f44747'
      },
      syntax: {
        keyword: '#569cd6',
        string: '#ce9178',
        number: '#b5cea8',
        comment: '#6a9955',
        function: '#dcdcaa',
        variable: '#9cdcfe',
        operator: '#d4d4d4',
        background: '#1e1e1e'
      }
    },
    effects: {
      glass: true,
      glow: false,
      scanlines: false,
      animated: false,
      gradients: false
    }
  },

  // ========== FINANCE THEMES ==========

  'bloomberg': {
    name: 'bloomberg',
    displayName: 'Bloomberg Terminal',
    description: 'Classic financial terminal with amber accents',
    colors: {
      bg: {
        primary: '#0a0a0a',
        secondary: '#121212',
        tertiary: '#1a1a1a',
        glass: 'rgba(255, 153, 0, 0.05)'
      },
      text: {
        primary: '#ff9900',
        secondary: '#cc7a00',
        muted: '#664000',
        accent: '#ffcc00'
      },
      border: {
        subtle: '#1a1a1a',
        medium: '#333333',
        accent: '#ff9900'
      },
      accent: {
        primary: '#ff9900',
        secondary: '#ffb300',
        success: '#00ff00',
        warning: '#ffcc00',
        danger: '#ff3300'
      },
      syntax: {
        keyword: '#ff9900',
        string: '#ffcc00',
        number: '#00ff00',
        comment: '#664000',
        function: '#00ccff',
        variable: '#ff9900',
        operator: '#ff6600',
        background: '#0a0a0a'
      }
    },
    effects: {
      glass: true,
      glow: false,
      scanlines: false,
      animated: false,
      gradients: false
    },
    wallpaper: {
      type: 'gradient',
      value: 'bloomberg-grid',
      gradient: 'linear-gradient(180deg, #0a0a0a 0%, #1a1208 100%)'
    }
  },

  'crypto-quant': {
    name: 'crypto-quant',
    displayName: 'Crypto Quant',
    description: 'Modern crypto trading with purple neon accents',
    colors: {
      bg: {
        primary: '#0d0d1a',
        secondary: '#151525',
        tertiary: '#1e1e35',
        glass: 'rgba(138, 43, 226, 0.08)'
      },
      text: {
        primary: '#e8e8ff',
        secondary: '#a0a0cc',
        muted: '#606090',
        accent: '#bf00ff'
      },
      border: {
        subtle: '#2a2a45',
        medium: '#404060',
        accent: '#8b5cf6'
      },
      accent: {
        primary: '#a855f7',
        secondary: '#c084fc',
        success: '#22c55e',
        warning: '#eab308',
        danger: '#ef4444'
      },
      syntax: {
        keyword: '#c084fc',
        string: '#22c55e',
        number: '#fbbf24',
        comment: '#606090',
        function: '#38bdf8',
        variable: '#e8e8ff',
        operator: '#f472b6',
        background: '#0d0d1a'
      }
    },
    effects: {
      glass: true,
      glow: true,
      scanlines: false,
      animated: true,
      gradients: true
    },
    wallpaper: {
      type: 'gradient',
      value: 'crypto-gradient',
      gradient: 'linear-gradient(135deg, #0d0d1a 0%, #1a0a2e 50%, #0d1a1a 100%)'
    }
  },

  'gold-futures': {
    name: 'gold-futures',
    displayName: 'Gold Futures',
    description: 'Premium commodities trading with gold accents',
    colors: {
      bg: {
        primary: '#0c0a07',
        secondary: '#14100c',
        tertiary: '#1c1812',
        glass: 'rgba(212, 175, 55, 0.06)'
      },
      text: {
        primary: '#f5e6c8',
        secondary: '#c9b896',
        muted: '#6b5c48',
        accent: '#d4af37'
      },
      border: {
        subtle: '#2a2318',
        medium: '#3d3423',
        accent: '#d4af37'
      },
      accent: {
        primary: '#d4af37',
        secondary: '#e6c158',
        success: '#22c55e',
        warning: '#f59e0b',
        danger: '#dc2626'
      },
      syntax: {
        keyword: '#d4af37',
        string: '#fbbf24',
        number: '#fb923c',
        comment: '#6b5c48',
        function: '#38bdf8',
        variable: '#f5e6c8',
        operator: '#d4af37',
        background: '#0c0a07'
      }
    },
    effects: {
      glass: true,
      glow: true,
      scanlines: false,
      animated: false,
      gradients: true
    },
    wallpaper: {
      type: 'gradient',
      value: 'gold-shimmer',
      gradient: 'linear-gradient(135deg, #0c0a07 0%, #1a1510 50%, #0d0a05 100%)'
    }
  },

  'forex-pro': {
    name: 'forex-pro',
    displayName: 'Forex Pro',
    description: 'Professional forex trading with green/red indicators',
    colors: {
      bg: {
        primary: '#0a0f0a',
        secondary: '#0f140f',
        tertiary: '#151a15',
        glass: 'rgba(34, 197, 94, 0.05)'
      },
      text: {
        primary: '#e8f5e8',
        secondary: '#a8c8a8',
        muted: '#486048',
        accent: '#22c55e'
      },
      border: {
        subtle: '#1a201a',
        medium: '#2a352a',
        accent: '#22c55e'
      },
      accent: {
        primary: '#22c55e',
        secondary: '#4ade80',
        success: '#22c55e',
        warning: '#eab308',
        danger: '#ef4444'
      },
      syntax: {
        keyword: '#22c55e',
        string: '#86efac',
        number: '#fcd34d',
        comment: '#486048',
        function: '#38bdf8',
        variable: '#e8f5e8',
        operator: '#22c55e',
        background: '#0a0f0a'
      }
    },
    effects: {
      glass: true,
      glow: false,
      scanlines: false,
      animated: false,
      gradients: false
    },
    wallpaper: {
      type: 'gradient',
      value: 'forex-forest',
      gradient: 'linear-gradient(180deg, #0a0f0a 0%, #0a140a 100%)'
    }
  },

  'midnight-quant': {
    name: 'midnight-quant',
    displayName: 'Midnight Quant',
    description: 'Modern dark theme with teal accents for quant traders',
    colors: {
      bg: {
        primary: '#0a0f14',
        secondary: '#0f161c',
        tertiary: '#151d26',
        glass: 'rgba(20, 184, 166, 0.06)'
      },
      text: {
        primary: '#e0f2f1',
        secondary: '#99b8b4',
        muted: '#4a6662',
        accent: '#14b8a6'
      },
      border: {
        subtle: '#1a242c',
        medium: '#2a343c',
        accent: '#14b8a6'
      },
      accent: {
        primary: '#14b8a6',
        secondary: '#2dd4bf',
        success: '#22c55e',
        warning: '#f59e0b',
        danger: '#f43f5e'
      },
      syntax: {
        keyword: '#2dd4bf',
        string: '#34d399',
        number: '#fbbf24',
        comment: '#4a6662',
        function: '#38bdf8',
        variable: '#e0f2f1',
        operator: '#14b8a6',
        background: '#0a0f14'
      }
    },
    effects: {
      glass: true,
      glow: true,
      scanlines: false,
      animated: true,
      gradients: true
    },
    wallpaper: {
      type: 'gradient',
      value: 'midnight-teal',
      gradient: 'linear-gradient(135deg, #0a0f14 0%, #0c1418 50%, #0a1012 100%)'
    }
  },

  // ========== WALLPAPER THEMES ==========

  'ocean-blue': {
    name: 'ocean-blue',
    displayName: 'Ocean Blue',
    description: 'Deep ocean gradient for focus',
    colors: {
      bg: { primary: '#0a1628', secondary: '#0f1d32', tertiary: '#142540', glass: 'rgba(59, 130, 246, 0.08)' },
      text: { primary: '#e0f2fe', secondary: '#7dd3fc', muted: '#0369a1', accent: '#38bdf8' },
      border: { subtle: '#1e3a5f', medium: '#2d4a6f', accent: '#38bdf8' },
      accent: { primary: '#38bdf8', secondary: '#7dd3fc', success: '#22c55e', warning: '#f59e0b', danger: '#ef4444' },
      syntax: { keyword: '#38bdf8', string: '#34d399', number: '#fbbf24', comment: '#0369a1', function: '#c084fc', variable: '#e0f2fe', operator: '#38bdf8', background: '#0a1628' }
    },
    effects: { glass: true, glow: true, scanlines: false, animated: true, gradients: true },
    wallpaper: { type: 'gradient', value: 'ocean-gradient', gradient: 'linear-gradient(180deg, #0a1628 0%, #0e2744 50%, #0a1628 100%)' }
  },

  'deep-space': {
    name: 'deep-space',
    displayName: 'Deep Space',
    description: 'Cosmic dark theme for night trading',
    colors: {
      bg: { primary: '#050510', secondary: '#0a0a1a', tertiary: '#0f0f25', glass: 'rgba(139, 92, 246, 0.06)' },
      text: { primary: '#e9d5ff', secondary: '#c4b5fd', muted: '#6b21a8', accent: '#a855f7' },
      border: { subtle: '#1a1a35', medium: '#2a2a45', accent: '#a855f7' },
      accent: { primary: '#a855f7', secondary: '#c084fc', success: '#22c55e', warning: '#eab308', danger: '#f43f5e' },
      syntax: { keyword: '#c084fc', string: '#a855f7', number: '#fbbf24', comment: '#6b21a8', function: '#38bdf8', variable: '#e9d5ff', operator: '#a855f7', background: '#050510' }
    },
    effects: { glass: true, glow: true, scanlines: false, animated: true, gradients: true },
    wallpaper: { type: 'gradient', value: 'space-gradient', gradient: 'radial-gradient(ellipse at top, #1a0a2e 0%, #050510 50%, #0a0a1a 100%)' }
  },

  'nordic-frost': {
    name: 'nordic-frost',
    displayName: 'Nordic Frost',
    description: 'Clean cool theme for clarity',
    colors: {
      bg: { primary: '#0c1220', secondary: '#111a2a', tertiary: '#162235', glass: 'rgba(148, 163, 184, 0.06)' },
      text: { primary: '#f1f5f9', secondary: '#94a3b8', muted: '#475569', accent: '#94a3b8' },
      border: { subtle: '#1e293b', medium: '#334155', accent: '#64748b' },
      accent: { primary: '#64748b', secondary: '#94a3b8', success: '#10b981', warning: '#f59e0b', danger: '#ef4444' },
      syntax: { keyword: '#38bdf8', string: '#34d399', number: '#fbbf24', comment: '#475569', function: '#818cf8', variable: '#f1f5f9', operator: '#64748b', background: '#0c1220' }
    },
    effects: { glass: true, glow: false, scanlines: false, animated: false, gradients: false },
    wallpaper: { type: 'gradient', value: 'frost-gradient', gradient: 'linear-gradient(135deg, #0c1220 0%, #162235 50%, #0c1220 100%)' }
  }
};

// ========== SEPARATE WALLPAPERS ==========
export interface Wallpaper {
  id: string;
  name: string;
  type: 'gradient' | 'pattern' | 'image';
  value: string;
  gradient?: string;
  imageUrl?: string;
}

// Helper to extract wallpapers from themes
function getThemeWallpapers(): Wallpaper[] {
  const themeWalls: Wallpaper[] = [];
  Object.values(themes).forEach(theme => {
    if (theme.wallpaper?.gradient) {
      themeWalls.push({
        id: `theme-${theme.name}`,
        name: `${theme.displayName}`,
        type: theme.wallpaper.type,
        value: theme.wallpaper.value,
        gradient: theme.wallpaper.gradient
      });
    }
  });
  return themeWalls;
}

export const wallpapers: Wallpaper[] = [
  // Theme Wallpapers (from existing themes)
  ...getThemeWallpapers(),

  // Gradients
  { id: 'midnight', name: 'Midnight', type: 'gradient', value: 'midnight', gradient: 'linear-gradient(180deg, #0a0a0f 0%, #1a1a2e 50%, #0f0f1a 100%)' },
  { id: 'ocean', name: 'Ocean', type: 'gradient', value: 'ocean', gradient: 'linear-gradient(180deg, #0a1628 0%, #0e2744 50%, #0a1628 100%)' },
  { id: 'sunset', name: 'Sunset', type: 'gradient', value: 'sunset', gradient: 'linear-gradient(180deg, #1a0a0a 0%, #2e1a1a 50%, #1a0f0f 100%)' },
  { id: 'forest', name: 'Forest', type: 'gradient', value: 'forest', gradient: 'linear-gradient(180deg, #0a1a0a 0%, #1a2e1a 50%, #0a1a0f 100%)' },
  { id: 'purple-haze', name: 'Purple Haze', type: 'gradient', value: 'purple-haze', gradient: 'linear-gradient(135deg, #1a0a2e 0%, #2e1a4e 50%, #1a1a2e 100%)' },
  { id: 'gold-shimmer', name: 'Gold Shimmer', type: 'gradient', value: 'gold-shimmer', gradient: 'linear-gradient(135deg, #1a1508 0%, #2e2510 50%, #1a1808 100%)' },
  { id: 'steel-blue', name: 'Steel Blue', type: 'gradient', value: 'steel-blue', gradient: 'linear-gradient(180deg, #0a1520 0%, #152535 50%, #0a1820 100%)' },
  { id: 'emerald', name: 'Emerald', type: 'gradient', value: 'emerald', gradient: 'linear-gradient(180deg, #0a1a15 0%, #152e25 50%, #0a1a10 100%)' },
  { id: 'cosmic', name: 'Cosmic', type: 'gradient', value: 'cosmic', gradient: 'radial-gradient(ellipse at top, #1a0a2e 0%, #050510 50%, #0a0a1a 100%)' },
  { id: 'nebula', name: 'Nebula', type: 'gradient', value: 'nebula', gradient: 'radial-gradient(ellipse at bottom, #1a0a1a 0%, #0a0510 50%, #0a0a15 100%)' },
  { id: 'arctic', name: 'Arctic', type: 'gradient', value: 'arctic', gradient: 'linear-gradient(135deg, #0a1520 0%, #152535 30%, #1a2a3a 60%, #0a1520 100%)' },
  { id: 'volcano', name: 'Volcano', type: 'gradient', value: 'volcano', gradient: 'linear-gradient(180deg, #1a0a05 0%, #2e1508 50%, #1a0f05 100%)' },

  // Patterns
  { id: 'grid', name: 'Grid', type: 'pattern', value: 'grid', gradient: 'linear-gradient(rgba(255,255,255,0.03) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.03) 1px, transparent 1px)' },
  { id: 'dots', name: 'Dots', type: 'pattern', value: 'dots', gradient: 'radial-gradient(rgba(255,255,255,0.15) 1px, transparent 1px)' },
  { id: 'cross', name: 'Cross', type: 'pattern', value: 'cross', gradient: 'linear-gradient(45deg, rgba(255,255,255,0.03) 25%, transparent 25%), linear-gradient(-45deg, rgba(255,255,255,0.03) 25%, transparent 25%), linear-gradient(45deg, transparent 75%, rgba(255,255,255,0.03) 75%), linear-gradient(-45deg, transparent 75%, rgba(255,255,255,0.03) 75%)' },
  { id: 'waves', name: 'Waves', type: 'pattern', value: 'waves', gradient: 'repeating-linear-gradient(0deg, transparent, transparent 50px, rgba(255,255,255,0.02) 50px, rgba(255,255,255,0.02) 51px)' },

  // Image Wallpapers (Unsplash)
  { id: 'city-night', name: 'City Night', type: 'image', value: 'city-night', imageUrl: 'https://images.unsplash.com/photo-1519501025264-65ba15a82390?w=1920&q=80' },
  { id: 'abstract-blue', name: 'Abstract Blue', type: 'image', value: 'abstract-blue', imageUrl: 'https://images.unsplash.com/photo-1557682250-33bd709cbe85?w=1920&q=80' },
  { id: 'neon Tokyo', name: 'Neon Tokyo', type: 'image', value: 'neon-tokyo', imageUrl: 'https://images.unsplash.com/photo-1540959733332-eab4deabeeaf?w=1920&q=80' },
  { id: 'stock-chart', name: 'Stock Chart', type: 'image', value: 'stock-chart', imageUrl: 'https://images.unsplash.com/photo-1611974789855-9c2a0a7236a3?w=1920&q=80' },
  { id: 'trading-floor', name: 'Trading Floor', type: 'image', value: 'trading-floor', imageUrl: 'https://images.unsplash.com/photo-1611974789855-9c2a0a7236a3?w=1920&q=80' },
  { id: 'data-center', name: 'Data Center', type: 'image', value: 'data-center', imageUrl: 'https://images.unsplash.com/photo-1558494949-ef010cbdcc31?w=1920&q=80' },
  { id: 'bitcoin', name: 'Bitcoin', type: 'image', value: 'bitcoin', imageUrl: 'https://images.unsplash.com/photo-1518546305927-5a555bb7020d?w=1920&q=80' },
  { id: 'glass-building', name: 'Glass Building', type: 'image', value: 'glass-building', imageUrl: 'https://images.unsplash.com/photo-1486406146926-c627a92ad1ab?w=1920&q=80' },
  { id: 'cyberpunk', name: 'Cyberpunk City', type: 'image', value: 'cyberpunk', imageUrl: 'https://images.unsplash.com/photo-1533147670608-2a2f9775d3a4?w=1920&q=80' },
  { id: 'aurora', name: 'Aurora Borealis', type: 'image', value: 'aurora', imageUrl: 'https://images.unsplash.com/photo-1531366936337-7c912a4589a7?w=1920&q=80' },
];

// ========== FONTS ==========
export interface FontOption {
  id: string;
  name: string;
  family: string;
  category: 'mono' | 'sans' | 'serif';
}

export const fonts: FontOption[] = [
  { id: 'inter', name: 'Inter', family: "'Inter', system-ui, sans-serif", category: 'sans' },
  { id: 'system', name: 'System', family: "system-ui, -apple-system, sans-serif", category: 'sans' },
  { id: 'roboto', name: 'Roboto', family: "'Roboto', sans-serif", category: 'sans' },
  { id: 'open-sans', name: 'Open Sans', family: "'Open Sans', sans-serif", category: 'sans' },
  { id: 'poppins', name: 'Poppins', family: "'Poppins', sans-serif", category: 'sans' },
  { id: 'fira-code', name: 'Fira Code', family: "'Fira Code', 'JetBrains Mono', monospace", category: 'mono' },
  { id: 'jetbrains', name: 'JetBrains Mono', family: "'JetBrains Mono', monospace", category: 'mono' },
  { id: 'source-code', name: 'Source Code Pro', family: "'Source Code Pro', monospace", category: 'mono' },
  { id: 'ibm-plex', name: 'IBM Plex Mono', family: "'IBM Plex Mono', monospace", category: 'mono' },
  { id: 'crimson', name: 'Crimson Pro', family: "'Crimson Pro', serif", category: 'serif' },
  { id: 'merriweather', name: 'Merriweather', family: "'Merriweather', serif", category: 'serif' },
  { id: 'playfair', name: 'Playfair Display', family: "'Playfair Display', serif", category: 'serif' },
];

// Current font store
export const currentFont = writable<string>('inter');

export function setFont(fontId: string) {
  const font = fonts.find(f => f.id === fontId);
  if (font) {
    currentFont.set(fontId);
    localStorage.setItem('quantmind-font', fontId);
    document.documentElement.style.setProperty('--font-family', font.family);
  }
}

export function loadSavedFont(): string {
  const saved = localStorage.getItem('quantmind-font');
  if (saved && fonts.find(f => f.id === saved)) {
    return saved;
  }
  return 'inter';
}

// Theme store
export const currentTheme = writable<ThemeName>('trading-terminal');
export const customWallpaper = writable<string>('');

// Derived store for current theme object
export const theme = derived(
  [currentTheme, customWallpaper],
  ([$currentTheme, $customWallpaper]) => {
    const baseTheme = themes[$currentTheme];
    
    // Apply custom wallpaper if set
    if ($customWallpaper) {
      return {
        ...baseTheme,
        wallpaper: {
          type: 'gradient' as const,
          value: $customWallpaper
        }
      };
    }
    
    return baseTheme;
  }
);

// Theme utilities
export function applyTheme(themeName: ThemeName) {
  currentTheme.set(themeName);
  localStorage.setItem('quantmind-theme', themeName);
}

// Helper function to apply theme CSS variables to document
function applyThemeVariables(themeObj: Theme) {
  const root = document.documentElement;

  // Background colors
  root.style.setProperty('--bg-primary', themeObj.colors.bg.primary);
  root.style.setProperty('--bg-secondary', themeObj.colors.bg.secondary);
  root.style.setProperty('--bg-tertiary', themeObj.colors.bg.tertiary);
  root.style.setProperty('--bg-glass', themeObj.colors.bg.glass);

  // Text colors
  root.style.setProperty('--text-primary', themeObj.colors.text.primary);
  root.style.setProperty('--text-secondary', themeObj.colors.text.secondary);
  root.style.setProperty('--text-muted', themeObj.colors.text.muted);
  root.style.setProperty('--text-accent', themeObj.colors.text.accent);

  // Border colors
  root.style.setProperty('--border-subtle', themeObj.colors.border.subtle);
  root.style.setProperty('--border-medium', themeObj.colors.border.medium);
  root.style.setProperty('--border-accent', themeObj.colors.border.accent);

  // Accent colors
  root.style.setProperty('--accent-primary', themeObj.colors.accent.primary);
  root.style.setProperty('--accent-secondary', themeObj.colors.accent.secondary);
  root.style.setProperty('--accent-success', themeObj.colors.accent.success);
  root.style.setProperty('--accent-warning', themeObj.colors.accent.warning);
  root.style.setProperty('--accent-danger', themeObj.colors.accent.danger);

  // Syntax colors
  root.style.setProperty('--syntax-keyword', themeObj.colors.syntax.keyword);
  root.style.setProperty('--syntax-string', themeObj.colors.syntax.string);
  root.style.setProperty('--syntax-number', themeObj.colors.syntax.number);
  root.style.setProperty('--syntax-comment', themeObj.colors.syntax.comment);
  root.style.setProperty('--syntax-function', themeObj.colors.syntax.function);
  root.style.setProperty('--syntax-variable', themeObj.colors.syntax.variable);
  root.style.setProperty('--syntax-operator', themeObj.colors.syntax.operator);
  root.style.setProperty('--syntax-background', themeObj.colors.syntax.background);

  // Apply effects as data attributes
  root.dataset.glass = String(themeObj.effects.glass);
  root.dataset.glow = String(themeObj.effects.glow);
  root.dataset.scanlines = String(themeObj.effects.scanlines);
  root.dataset.animated = String(themeObj.effects.animated);
  root.dataset.gradients = String(themeObj.effects.gradients);
}

// Helper function to apply wallpaper
function applyWallpaper(themeObj: Theme) {
  const root = document.documentElement;
  const wallpaper = themeObj.wallpaper;

  if (wallpaper?.gradient) {
    root.style.setProperty('--wallpaper', wallpaper.gradient);
    root.style.setProperty('--wallpaper-type', wallpaper.type);
  } else {
    root.style.setProperty('--wallpaper', 'none');
  }
}

// Reactive effect to apply theme when currentTheme changes
let previousTheme: ThemeName | null = null;

currentTheme.subscribe(themeName => {
  if (themeName !== previousTheme) {
    const themeObj = themes[themeName];
    if (themeObj) {
      applyThemeVariables(themeObj);
      applyWallpaper(themeObj);
      localStorage.setItem('quantmind-theme', themeName);
    }
    previousTheme = themeName;
  }
});

customWallpaper.subscribe(wallpaper => {
  if (previousTheme) {
    const themeObj = themes[previousTheme];
    if (themeObj) {
      applyWallpaper(themeObj);
    }
  }
  localStorage.setItem('quantmind-wallpaper', wallpaper);
});

export function loadSavedTheme(): ThemeName {
  const saved = localStorage.getItem('quantmind-theme');
  return (saved as ThemeName) || 'trading-terminal';
}

export function setCustomWallpaper(wallpaper: string) {
  customWallpaper.set(wallpaper);
  localStorage.setItem('quantmind-wallpaper', wallpaper);

  // Apply wallpaper to document
  const root = document.documentElement;
  if (wallpaper.startsWith('http')) {
    // Image URL
    root.style.setProperty('--wallpaper', `url(${wallpaper})`);
    root.style.setProperty('--wallpaper-type', 'image');
  } else if (wallpaper.includes('gradient') || wallpaper.includes('linear') || wallpaper.includes('radial')) {
    // Gradient
    root.style.setProperty('--wallpaper', wallpaper);
    root.style.setProperty('--wallpaper-type', 'gradient');
  } else {
    // Pattern
    root.style.setProperty('--wallpaper', wallpaper);
    root.style.setProperty('--wallpaper-type', 'pattern');
  }
}

export function loadSavedWallpaper(): string {
  return localStorage.getItem('quantmind-wallpaper') || '';
}

// Initialize theme on load
const savedTheme = loadSavedTheme();
currentTheme.set(savedTheme);
customWallpaper.set(loadSavedWallpaper());
