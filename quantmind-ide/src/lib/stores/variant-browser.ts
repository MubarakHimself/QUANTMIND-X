/**
 * Variant Browser Store
 *
 * State management for EA variant browser on Development canvas.
 */

import { writable, derived } from 'svelte/store';

// Types
export interface BacktestSummary {
    total_pnl: number;
    sharpe_ratio: number;
    max_drawdown: number;
    trade_count: number;
    win_rate: number;
    profit_factor: number;
    period: string;
    last_updated?: string;
}

export interface VariantInfo {
    variant_type: string;
    version_tag: string;
    improvement_cycle: number;
    author: string;
    created_at: string;
    is_active: boolean;
    backtest?: BacktestSummary;
    promotion_status: string;
}

export interface StrategyVariants {
    strategy_id: string;
    strategy_name: string;
    variants: VariantInfo[];
    variant_counts: Record<string, number>;
}

export interface VariantDetail {
    strategy_id: string;
    strategy_name: string;
    variant: VariantInfo;
    version_timeline: Array<{
        version_tag: string;
        created_at: string;
        author: string;
        improvement_cycle: number;
        is_active: boolean;
    }>;
    code_content?: string;
}

export interface VariantBrowserState {
    strategies: StrategyVariants[];
    selectedStrategy: string | null;
    selectedVariant: string | null;
    currentDetail: VariantDetail | null;
    isLoading: boolean;
    error: string | null;
}

// Initial state
const initialState: VariantBrowserState = {
    strategies: [],
    selectedStrategy: null,
    selectedVariant: null,
    currentDetail: null,
    isLoading: false,
    error: null,
};

// Create the store
function createVariantBrowserStore() {
    const { subscribe, set, update } = writable<VariantBrowserState>(initialState);

    return {
        subscribe,

        // Load all variants
        async loadVariants() {
            update(state => ({ ...state, isLoading: true, error: null }));
            try {
                const response = await fetch('/api/variant-browser');
                if (!response.ok) throw new Error('Failed to fetch variants');
                const data = await response.json();
                update(state => ({
                    ...state,
                    strategies: data.strategies || [],
                    isLoading: false
                }));
            } catch (error) {
                update(state => ({
                    ...state,
                    isLoading: false,
                    error: error instanceof Error ? error.message : 'Unknown error'
                }));
            }
        },

        // Select a strategy
        selectStrategy(strategyId: string | null) {
            update(state => ({
                ...state,
                selectedStrategy: strategyId,
                selectedVariant: null,
                currentDetail: null
            }));
        },

        // Select a variant and load detail
        async selectVariant(strategyId: string, variantType: string) {
            update(state => ({
                ...state,
                selectedStrategy: strategyId,
                selectedVariant: variantType,
                isLoading: true,
                error: null
            }));
            try {
                const response = await fetch(`/api/variant-browser/${strategyId}/${variantType}`);
                if (!response.ok) throw new Error('Failed to fetch variant detail');
                const data = await response.json();
                update(state => ({
                    ...state,
                    currentDetail: data,
                    isLoading: false
                }));
            } catch (error) {
                update(state => ({
                    ...state,
                    isLoading: false,
                    error: error instanceof Error ? error.message : 'Unknown error'
                }));
            }
        },

        // Clear selection
        clearSelection() {
            update(state => ({
                ...state,
                selectedStrategy: null,
                selectedVariant: null,
                currentDetail: null
            }));
        },

        // Reset store
        reset() {
            set(initialState);
        }
    };
}

// Export store
export const variantBrowserStore = createVariantBrowserStore();

// Derived stores for convenience
export const strategiesList = derived(
    variantBrowserStore,
    $store => $store.strategies
);

export const currentDetail = derived(
    variantBrowserStore,
    $store => $store.currentDetail
);

export const isLoading = derived(
    variantBrowserStore,
    $store => $store.isLoading
);