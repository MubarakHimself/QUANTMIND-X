/**
 * Shared Assets State Store
 *
 * Manages state for shared assets canvas:
 * - Asset categories and items
 * - Loading states
 * - Selected asset for detail view
 */

import { writable, derived, get } from 'svelte/store';
import {
  listAssetsByType,
  getAssetDetail,
  getAssetCounts,
  type SharedAsset,
  type AssetType
} from '$lib/api/sharedAssetsApi';

// =============================================================================
// Types
// =============================================================================

export interface AssetFilter {
  type: AssetType | 'all';
}

export interface SharedAssetsState {
  assets: Record<AssetType, SharedAsset[]>;
  isLoading: boolean;
  error: string | null;
  selectedAsset: SharedAsset | null;
  selectedType: AssetType | null;
  assetCounts: Record<AssetType, number>;
}

// =============================================================================
// Store
// =============================================================================

const defaultState: SharedAssetsState = {
  assets: {
    'docs': [],
    'strategy-templates': [],
    'indicators': [],
    'skills': [],
    'flow-components': [],
    'mcp-configs': [],
    'strategies': []
  },
  isLoading: true,
  error: null,
  selectedAsset: null,
  selectedType: null,
  assetCounts: {
    'docs': 0,
    'strategy-templates': 0,
    'indicators': 0,
    'skills': 0,
    'flow-components': 0,
    'mcp-configs': 0,
    'strategies': 0
  }
};

function createSharedAssetsStore() {
  const { subscribe, set, update } = writable<SharedAssetsState>({ ...defaultState });

  return {
    subscribe,

    /**
     * Fetch all assets
     */
    async fetchAssets() {
      update(state => ({ ...state, isLoading: true, error: null }));

      try {
        const counts = await getAssetCounts();

        update(state => ({
          ...state,
          assetCounts: counts,
          isLoading: false
        }));
      } catch (e) {
        const errorMessage = e instanceof Error ? e.message : 'Failed to fetch assets';
        update(state => ({
          ...state,
          isLoading: false,
          error: errorMessage
        }));
      }
    },

    /**
     * Fetch assets for a specific type
     */
    async fetchAssetsByType(type: AssetType) {
      update(state => ({ ...state, isLoading: true, error: null, selectedType: type }));

      try {
        const assets = await listAssetsByType(type);
        let refreshedCounts: Record<AssetType, number> | null = null;
        try {
          refreshedCounts = await getAssetCounts();
        } catch {
          // Keep UI responsive even if one category count fetch fails.
        }

        update(state => ({
          ...state,
          assets: { ...state.assets, [type]: assets },
          assetCounts: refreshedCounts
            ? refreshedCounts
            : { ...state.assetCounts, [type]: assets.length },
          isLoading: false
        }));
      } catch (e) {
        const errorMessage = e instanceof Error ? e.message : 'Failed to fetch assets';
        update(state => ({
          ...state,
          isLoading: false,
          error: errorMessage
        }));
      }
    },

    /**
     * Select an asset for detail view
     */
    async selectAsset(asset: SharedAsset) {
      update(state => ({ ...state, selectedAsset: asset }));

      // Optionally fetch full detail if not already loaded
      if (!asset.content) {
        try {
          const detail = await getAssetDetail(asset.id, asset.type);
          update(state => ({ ...state, selectedAsset: detail }));
        } catch (e) {
          console.error('Failed to load asset detail:', e);
        }
      }
    },

    /**
     * Clear selected asset
     */
    clearSelection() {
      update(state => ({
        ...state,
        selectedAsset: null,
        selectedType: null
      }));
    },

    /**
     * Clear only the selected asset while preserving the active type/category.
     */
    clearSelectedAsset() {
      update(state => ({
        ...state,
        selectedAsset: null
      }));
    },

    /**
     * Set selected type (category)
     */
    setSelectedType(type: AssetType | null) {
      update(state => ({ ...state, selectedType: type }));
    },

    /**
     * Clear error
     */
    clearError() {
      update(state => ({ ...state, error: null }));
    },

    /**
     * Update asset content in store
     */
    updateAssetContent(assetId: string, content: string) {
      update(state => {
        // Find and update the asset in assets record
        const newAssets = { ...state.assets };
        for (const type of Object.keys(newAssets) as AssetType[]) {
          const assets = newAssets[type];
          const index = assets.findIndex(a => a.id === assetId);
          if (index !== -1) {
            newAssets[type] = [
              ...assets.slice(0, index),
              { ...assets[index], content },
              ...assets.slice(index + 1)
            ];
            break;
          }
        }

        // Also update selectedAsset if it matches
        let updatedSelectedAsset = state.selectedAsset;
        if (state.selectedAsset?.id === assetId) {
          updatedSelectedAsset = { ...state.selectedAsset, content };
        }

        return {
          ...state,
          assets: newAssets,
          selectedAsset: updatedSelectedAsset
        };
      });
    },

    /**
     * Reset store to initial state
     */
    reset() {
      set({ ...defaultState });
    },

    async fetchAssetCounts() {
      update(state => ({ ...state, isLoading: true, error: null }));

      try {
        const counts = await getAssetCounts();

        update(state => ({
          ...state,
          assetCounts: counts,
          isLoading: false
        }));
      } catch (e) {
        const errorMessage = e instanceof Error ? e.message : 'Failed to fetch asset counts';
        update(state => ({
          ...state,
          isLoading: false,
          error: errorMessage
        }));
      }
    }
  };
}

export const sharedAssetsStore = createSharedAssetsStore();

// =============================================================================
// Derived Stores
// =============================================================================

/**
 * Current assets based on selected type
 */
export const currentAssets = derived(sharedAssetsStore, ($store) => {
  if (!$store.selectedType) {
    // Return all assets flattened
    return Object.values($store.assets).flat();
  }
  return $store.assets[$store.selectedType] || [];
});

/**
 * Loading state
 */
export const assetsLoading = derived(sharedAssetsStore, ($store) => $store.isLoading);

/**
 * Error state
 */
export const assetsError = derived(sharedAssetsStore, ($store) => $store.error);

/**
 * Selected asset
 */
export const selectedAsset = derived(sharedAssetsStore, ($store) => $store.selectedAsset);

 /**
 * Asset counts
 */
export const assetCounts = derived(sharedAssetsStore, ($store) => $store.assetCounts);

/**
 * Has any assets
 */
export const hasAssets = derived(sharedAssetsStore, ($store) => {
  return Object.values($store.assets).some(arr => arr.length > 0);
});

// =============================================================================
// Helper Functions
// =============================================================================

/**
 * Get assets for a specific type from store
 */
export function getAssetsByType(type: AssetType): SharedAsset[] {
  const state = get(sharedAssetsStore);
  return state.assets[type] || [];
}

/**
 * Initialize store with data
 */
export async function initSharedAssetsStore() {
  await sharedAssetsStore.fetchAssets();
}
