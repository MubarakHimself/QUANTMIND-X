import { writable, derived } from 'svelte/store';

export type ThemeName = 'trading-terminal' | 'monokai' | 'ambient' | 'cyberpunk' | 'matrix' | 'dark-pro';

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
  }
};

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

export function loadSavedTheme(): ThemeName {
  const saved = localStorage.getItem('quantmind-theme');
  return (saved as ThemeName) || 'trading-terminal';
}

export function setCustomWallpaper(wallpaper: string) {
  customWallpaper.set(wallpaper);
  localStorage.setItem('quantmind-wallpaper', wallpaper);
}

export function loadSavedWallpaper(): string {
  return localStorage.getItem('quantmind-wallpaper') || '';
}

// Initialize theme on load
const savedTheme = loadSavedTheme();
currentTheme.set(savedTheme);
customWallpaper.set(loadSavedWallpaper());
