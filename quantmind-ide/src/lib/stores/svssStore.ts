/**
 * SVSS Store - Shared Volume Session Service Polling
 *
 * Manages polling for SVSS indicator data from the backend API.
 * Polls every 15 seconds per story requirements.
 *
 * Endpoints:
 * - GET /api/svss/vwap/{symbol} — VWAP + Volume Profile data
 * - GET /api/svss/rvol/{symbol} — Relative Volume data
 * - GET /api/svss/profile/{symbol} — Volume Profile data
 * - GET /api/svss/mfi/{symbol} — Money Flow Index data
 * - GET /api/svss/summary — Summary of all tracked symbols
 */

import { writable, derived } from 'svelte/store';

// Types for SVSS data
export interface VWAPData {
  symbol: string;
  vwap: number;
  poc: number | null;
  vah: number | null;
  val: number | null;
  timestamp: string;
}

export interface RVOLData {
  symbol: string;
  rvol: number;
  quality_score: number;
  timestamp: string;
}

export interface VolumeProfileData {
  symbol: string;
  poc: number | null;
  vah: number | null;
  val: number | null;
  volume_bid: number;
  volume_ask: number;
  timestamp: string;
}

export interface MFIData {
  symbol: string;
  mfi: number;
  zone: 'overbought' | 'oversold' | 'neutral';
  timestamp: string;
}

export interface SVSSSummary {
  symbols: string[];
  indicators: string[];
  timestamp: string;
}

export interface SVSSState {
  // Per-symbol data
  vwap: Record<string, VWAPData>;
  rvol: Record<string, RVOLData>;
  profile: Record<string, VolumeProfileData>;
  mfi: Record<string, MFIData>;
  summary: SVSSSummary | null;

  // Tracking state
  trackedSymbols: string[];
  selectedSymbol: string | null;

  // Loading and error states
  loading: boolean;
  error: string | null;
  lastUpdated: Date | null;
}

// Initial state
const initialState: SVSSState = {
  vwap: {},
  rvol: {},
  profile: {},
  mfi: {},
  summary: null,
  trackedSymbols: [],
  selectedSymbol: null,
  loading: false,
  error: null,
  lastUpdated: null
};

// Create the store
function createSVSSStore() {
  const { subscribe, set, update } = writable<SVSSState>(initialState);

  let pollingInterval: ReturnType<typeof setInterval> | null = null;

  /**
   * Fetch summary to get list of tracked symbols
   */
  async function fetchSummary(): Promise<string[]> {
    try {
      const response = await fetch('/api/svss/summary');
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      const data: SVSSSummary = await response.json();
      update(state => ({
        ...state,
        summary: data,
        trackedSymbols: data.symbols
      }));
      return data.symbols;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to fetch SVSS summary';
      update(state => ({
        ...state,
        error: errorMessage
      }));
      return [];
    }
  }

  /**
   * Fetch VWAP data for a symbol
   */
  async function fetchVWAP(symbol: string): Promise<VWAPData | null> {
    try {
      const response = await fetch(`/api/svss/vwap/${symbol}`);
      if (!response.ok) {
        if (response.status === 404) {
          return null;  // Symbol not tracked
        }
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      return await response.json();
    } catch (err) {
      logger.warn(`Failed to fetch VWAP for ${symbol}:`, err);
      return null;
    }
  }

  /**
   * Fetch RVOL data for a symbol
   */
  async function fetchRVOL(symbol: string): Promise<RVOLData | null> {
    try {
      const response = await fetch(`/api/svss/rvol/${symbol}`);
      if (!response.ok) {
        if (response.status === 404) {
          return null;
        }
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      return await response.json();
    } catch (err) {
      logger.warn(`Failed to fetch RVOL for ${symbol}:`, err);
      return null;
    }
  }

  /**
   * Fetch Volume Profile data for a symbol
   */
  async function fetchProfile(symbol: string): Promise<VolumeProfileData | null> {
    try {
      const response = await fetch(`/api/svss/profile/${symbol}`);
      if (!response.ok) {
        if (response.status === 404) {
          return null;
        }
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      return await response.json();
    } catch (err) {
      logger.warn(`Failed to fetch profile for ${symbol}:`, err);
      return null;
    }
  }

  /**
   * Fetch MFI data for a symbol
   */
  async function fetchMFI(symbol: string): Promise<MFIData | null> {
    try {
      const response = await fetch(`/api/svss/mfi/${symbol}`);
      if (!response.ok) {
        if (response.status === 404) {
          return null;
        }
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      return await response.json();
    } catch (err) {
      logger.warn(`Failed to fetch MFI for ${symbol}:`, err);
      return null;
    }
  }

  /**
   * Fetch all SVSS data for tracked symbols
   */
  async function fetchAllData() {
    update(state => ({ ...state, loading: true, error: null }));

    try {
      // First get the list of tracked symbols
      const symbols = await fetchSummary();

      if (symbols.length === 0) {
        update(state => ({
          ...state,
          loading: false,
          lastUpdated: new Date()
        }));
        return;
      }

      // Fetch all indicators for each symbol in parallel
      const vwapPromises = symbols.map(s => fetchVWAP(s));
      const rvolPromises = symbols.map(s => fetchRVOL(s));
      const profilePromises = symbols.map(s => fetchProfile(s));
      const mfiPromises = symbols.map(s => fetchMFI(s));

      const [vwapResults, rvolResults, profileResults, mfiResults] = await Promise.all([
        Promise.all(vwapPromises),
        Promise.all(rvolPromises),
        Promise.all(profilePromises),
        Promise.all(mfiPromises)
      ]);

      // Build updated records
      const newVwap: Record<string, VWAPData> = {};
      const newRvol: Record<string, RVOLData> = {};
      const newProfile: Record<string, VolumeProfileData> = {};
      const newMfi: Record<string, MFIData> = {};

      symbols.forEach((symbol, index) => {
        if (vwapResults[index]) newVwap[symbol] = vwapResults[index];
        if (rvolResults[index]) newRvol[symbol] = rvolResults[index];
        if (profileResults[index]) newProfile[symbol] = profileResults[index];
        if (mfiResults[index]) newMfi[symbol] = mfiResults[index];
      });

      update(state => ({
        ...state,
        vwap: newVwap,
        rvol: newRvol,
        profile: newProfile,
        mfi: newMfi,
        loading: false,
        lastUpdated: new Date()
      }));

    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to fetch SVSS data';
      update(state => ({
        ...state,
        loading: false,
        error: errorMessage
      }));
    }
  }

  /**
   * Fetch data for a specific symbol only
   */
  async function fetchSymbolData(symbol: string) {
    update(state => ({ ...state, loading: true, error: null }));

    try {
      const [vwap, rvol, profile, mfi] = await Promise.all([
        fetchVWAP(symbol),
        fetchRVOL(symbol),
        fetchProfile(symbol),
        fetchMFI(symbol)
      ]);

      update(state => {
        const newVwap = { ...state.vwap };
        const newRvol = { ...state.rvol };
        const newProfile = { ...state.profile };
        const newMfi = { ...state.mfi };

        if (vwap) newVwap[symbol] = vwap;
        if (rvol) newRvol[symbol] = rvol;
        if (profile) newProfile[symbol] = profile;
        if (mfi) newMfi[symbol] = mfi;

        // Add to tracked symbols if not already there
        const trackedSymbols = state.trackedSymbols.includes(symbol)
          ? state.trackedSymbols
          : [...state.trackedSymbols, symbol];

        return {
          ...state,
          vwap: newVwap,
          rvol: newRvol,
          profile: newProfile,
          mfi: newMfi,
          trackedSymbols,
          loading: false,
          lastUpdated: new Date()
        };
      });

    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to fetch symbol data';
      update(state => ({
        ...state,
        loading: false,
        error: errorMessage
      }));
    }
  }

  /**
   * Start polling at specified interval
   */
  function startPolling(intervalMs: number = 15000) {
    // Fetch immediately
    fetchAllData();

    // Then poll at interval
    if (pollingInterval) {
      clearInterval(pollingInterval);
    }
    pollingInterval = setInterval(fetchAllData, intervalMs);
  }

  /**
   * Stop polling
   */
  function stopPolling() {
    if (pollingInterval) {
      clearInterval(pollingInterval);
      pollingInterval = null;
    }
  }

  /**
   * Select a symbol for focused display
   */
  function selectSymbol(symbol: string | null) {
    update(state => ({ ...state, selectedSymbol: symbol }));
  }

  /**
   * Reset store to initial state
   */
  function reset() {
    stopPolling();
    set(initialState);
  }

  return {
    subscribe,
    fetchAll: fetchAllData,
    fetchSymbol: fetchSymbolData,
    startPolling,
    stopPolling,
    selectSymbol,
    reset
  };
}

// Simple logger fallback
const logger = {
  warn: (msg: string, ...args: unknown[]) => console.warn(msg, ...args),
  error: (msg: string, ...args: unknown[]) => console.error(msg, ...args)
};

// Export the store
export const svssStore = createSVSSStore();

// Derived stores for convenience
export const svssVWAP = derived(svssStore, $store => $store.vwap);
export const svssRVOL = derived(svssStore, $store => $store.rvol);
export const svssProfile = derived(svssStore, $store => $store.profile);
export const svssMFI = derived(svssStore, $store => $store.mfi);
export const svssSymbols = derived(svssStore, $store => $store.trackedSymbols);
export const svssSelectedSymbol = derived(svssStore, $store => $store.selectedSymbol);
export const svssLoading = derived(svssStore, $store => $store.loading);
export const svssError = derived(svssStore, $store => $store.error);
export const svssLastUpdated = derived(svssStore, $store => $store.lastUpdated);

// Helper to get VWAP data for selected symbol
export const selectedSymbolVWAP = derived(
  svssStore,
  $store => $store.selectedSymbol ? $store.vwap[$store.selectedSymbol] : null
);

// Helper to get RVOL data for selected symbol
export const selectedSymbolRVOL = derived(
  svssStore,
  $store => $store.selectedSymbol ? $store.rvol[$store.selectedSymbol] : null
);

// Helper to get MFI zone color
export function getMFIZoneColor(zone: string): string {
  switch (zone) {
    case 'overbought':
      return '#ff3b3b';  // red
    case 'oversold':
      return '#22c55e'; // green
    default:
      return '#f0a500'; // yellow/neutral
  }
}

// Helper to get RVOL color based on value
export function getRVOLColor(rvol: number): string {
  if (rvol >= 1.5) return '#22c55e';    // green - high volume
  if (rvol >= 0.8) return '#f0a500';    // yellow - normal
  if (rvol >= 0.5) return '#ff9500';    // orange - below normal
  return '#ff3b3b';                      // red - very low
}

// Helper to interpret RVOL value
export function getRVOLInterpretation(rvol: number): string {
  if (rvol >= 2.0) return 'Very High';
  if (rvol >= 1.5) return 'High';
  if (rvol >= 1.0) return 'Normal';
  if (rvol >= 0.5) return 'Low';
  return 'Very Low';
}
