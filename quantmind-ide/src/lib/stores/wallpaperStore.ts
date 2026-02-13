import { writable, derived } from 'svelte/store';

export type WallpaperType = 'gradient' | 'anime' | 'pattern' | 'custom';

export interface Wallpaper {
  id: string;
  name: string;
  type: WallpaperType;
  url?: string;
  gradient?: string;
  category?: string;
  thumbnail?: string;
}

// Predefined anime wallpapers
export const animeWallpapers: Wallpaper[] = [
  {
    id: 'anime-cyberpunk',
    name: 'Cyberpunk City',
    type: 'anime',
    url: 'https://images.unsplash.com/photo-1518709268805-4e9042af2176?w=1920&h=1080&fit=crop',
    category: 'Cyberpunk',
    thumbnail: 'https://images.unsplash.com/photo-1518709268805-4e9042af2176?w=400&h=225&fit=crop'
  },
  {
    id: 'anime-samurai',
    name: 'Samurai Warrior',
    type: 'anime',
    url: 'https://images.unsplash.com/photo-1578662996442-48f60103fc96?w=1920&h=1080&fit=crop',
    category: 'Action',
    thumbnail: 'https://images.unsplash.com/photo-1578662996442-48f60103fc96?w=400&h=225&fit=crop'
  },
  {
    id: 'anime-nature',
    name: 'Mystic Forest',
    type: 'anime',
    url: 'https://images.unsplash.com/photo-1540206395-68808572332f?w=1920&h=1080&fit=crop',
    category: 'Nature',
    thumbnail: 'https://images.unsplash.com/photo-1540206395-68808572332f?w=400&h=225&fit=crop'
  },
  {
    id: 'anime-space',
    name: 'Space Station',
    type: 'anime',
    url: 'https://images.unsplash.com/photo-1446776811953-b23d57921c34?w=1920&h=1080&fit=crop',
    category: 'Sci-Fi',
    thumbnail: 'https://images.unsplash.com/photo-1446776811953-b23d57921c34?w=400&h=225&fit=crop'
  },
  {
    id: 'anime-dragon',
    name: 'Dragon Spirit',
    type: 'anime',
    url: 'https://images.unsplash.com/photo-1578662996442-48f60103fc96?w=1920&h=1080&fit=crop',
    category: 'Fantasy',
    thumbnail: 'https://images.unsplash.com/photo-1578662996442-48f60103fc96?w=400&h=225&fit=crop'
  },
  {
    id: 'anime-ocean',
    name: 'Ocean Waves',
    type: 'anime',
    url: 'https://images.unsplash.com/photo-1544551763-46a013bb70d5?w=1920&h=1080&fit=crop',
    category: 'Nature',
    thumbnail: 'https://images.unsplash.com/photo-1544551763-46a013bb70d5?w=400&h=225&fit=crop'
  }
];

// Gradient wallpapers
export const gradientWallpapers: Wallpaper[] = [
  {
    id: 'gradient-sunset',
    name: 'Sunset Vibes',
    type: 'gradient',
    gradient: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)'
  },
  {
    id: 'gradient-ocean',
    name: 'Ocean Blue',
    type: 'gradient',
    gradient: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)'
  },
  {
    id: 'gradient-forest',
    name: 'Forest Green',
    type: 'gradient',
    gradient: 'linear-gradient(135deg, #43e97b 0%, #38f9d7 100%)'
  },
  {
    id: 'gradient-fire',
    name: 'Fire Orange',
    type: 'gradient',
    gradient: 'linear-gradient(135deg, #fa709a 0%, #fee140 100%)'
  },
  {
    id: 'gradient-galaxy',
    name: 'Galaxy Purple',
    type: 'gradient',
    gradient: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)'
  },
  {
    id: 'gradient-aurora',
    name: 'Aurora Borealis',
    type: 'gradient',
    gradient: 'linear-gradient(135deg, #00d2ff 0%, #3a7bd5 50%, #00d2ff 100%)'
  }
];

// Pattern wallpapers
export const patternWallpapers: Wallpaper[] = [
  {
    id: 'pattern-terminal',
    name: 'Terminal Grid',
    type: 'pattern',
    gradient: 'repeating-linear-gradient(0deg, transparent, transparent 50px, rgba(0, 255, 0, 0.03) 50px, rgba(0, 255, 0, 0.03) 51px)'
  },
  {
    id: 'pattern-matrix',
    name: 'Matrix Rain',
    type: 'pattern',
    gradient: 'repeating-linear-gradient(0deg, transparent, transparent 20px, rgba(0, 255, 0, 0.05) 20px, rgba(0, 255, 0, 0.05) 21px)'
  },
  {
    id: 'pattern-circuit',
    name: 'Circuit Board',
    type: 'pattern',
    gradient: 'repeating-linear-gradient(90deg, transparent, transparent 30px, rgba(0, 255, 255, 0.03) 30px, rgba(0, 255, 255, 0.03) 31px)'
  },
  {
    id: 'pattern-hexagon',
    name: 'Hexagon Grid',
    type: 'pattern',
    gradient: 'repeating-linear-gradient(60deg, transparent, transparent 40px, rgba(255, 0, 255, 0.02) 40px, rgba(255, 0, 255, 0.02) 41px)'
  }
];

// All wallpapers
export const allWallpapers = [...animeWallpapers, ...gradientWallpapers, ...patternWallpapers];

// Store
export const currentWallpaper = writable<Wallpaper | null>(null);
export const wallpaperEnabled = writable<boolean>(false);

// Derived store for CSS background
export const wallpaperBackground = derived(
  [currentWallpaper, wallpaperEnabled],
  ([$currentWallpaper, $wallpaperEnabled]) => {
    if (!$wallpaperEnabled || !$currentWallpaper) return 'none';
    
    if ($currentWallpaper.type === 'gradient' || $currentWallpaper.type === 'pattern') {
      return $currentWallpaper.gradient || 'none';
    }
    
    if ($currentWallpaper.type === 'anime' && $currentWallpaper.url) {
      return `url(${$currentWallpaper.url})`;
    }
    
    return 'none';
  }
);

// Functions
export function setWallpaper(wallpaper: Wallpaper | null) {
  currentWallpaper.set(wallpaper);
  if (wallpaper) {
    localStorage.setItem('quantmind-wallpaper', JSON.stringify(wallpaper));
  } else {
    localStorage.removeItem('quantmind-wallpaper');
  }
}

export function enableWallpaper(enabled: boolean) {
  wallpaperEnabled.set(enabled);
  localStorage.setItem('quantmind-wallpaper-enabled', enabled.toString());
}

export function loadSavedWallpaper(): Wallpaper | null {
  const saved = localStorage.getItem('quantmind-wallpaper');
  return saved ? JSON.parse(saved) : null;
}

export function loadWallpaperEnabled(): boolean {
  const saved = localStorage.getItem('quantmind-wallpaper-enabled');
  return saved === 'true';
}

// Initialize
currentWallpaper.set(loadSavedWallpaper());
wallpaperEnabled.set(loadWallpaperEnabled());
