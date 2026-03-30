/**
 * Portfolio Store - Manages portfolio state for multi-account dashboard
 *
 * Extended for Story 9.4: Attribution, Correlation Matrix & Performance
 */

import { writable, derived, get } from 'svelte/store';
import { API_BASE } from '$lib/constants';

const apiBase = API_BASE || '';

export interface BrokerAccount {
  broker_id: string;
  broker_name: string;
  account_id: string;
  server: string;
  account_type: string;
  balance: number;
  equity: number;
  margin?: number;
  leverage?: number;
  currency: string;
  connected: boolean;
  is_active?: boolean;
  status?: string;
  drawdown?: number;
  exposure?: number;
}

export interface PortfolioSummary {
  totalEquity: number;
  dailyPnL: number;
  totalDrawdown: number;
  drawdownPercent: number;
}

export interface RoutingRule {
  strategy_id: string;
  strategy_name: string;
  account_id: string;
  enabled: boolean;
  regime_filter?: string;
  strategy_type_filter?: string;
}

export interface Strategy {
  id: string;
  name: string;
  type: 'SCALPER' | 'HFT' | 'STRUCTURAL' | 'SWING';
  current_account?: string;
}

/**
 * Attribution data for each strategy (Story 9.4 - AC #1)
 */
export interface StrategyAttribution {
  strategy_id: string;
  strategy_name: string;
  equity_contribution: number;
  pnl_contribution: number;
  drawdown_contribution: number;
  portfolio_percent: number;
  broker_account: string;
  broker_name: string;
}

/**
 * Correlation matrix cell (Story 9.4 - AC #2, #3)
 */
export interface CorrelationCell {
  strategy_a: string;
  strategy_b: string;
  correlation: number;
  data_period: string;
}

/**
 * Performance metrics (Story 9.4 - AC all)
 */
export interface PerformanceMetrics {
  total_return: number;
  sharpe_ratio: number;
  max_drawdown: number;
  win_rate: number;
  profit_factor: number;
  avg_trade: number;
  total_trades: number;
  profitable_trades: number;
  losing_trades: number;
}

interface PortfolioStoreState {
  accounts: BrokerAccount[];
  portfolioSummary: PortfolioSummary | null;
  routingRules: RoutingRule[];
  strategies: Strategy[];
  attribution: StrategyAttribution[];
  correlationMatrix: CorrelationCell[];
  performance: PerformanceMetrics | null;
  loading: boolean;
  error: string | null;
  drawdownAlert: { active: boolean; percent: number } | null;
}

const initialState: PortfolioStoreState = {
  accounts: [],
  portfolioSummary: null,
  routingRules: [],
  strategies: [],
  attribution: [],
  correlationMatrix: [],
  performance: null,
  loading: false,
  error: null,
  drawdownAlert: null
};

function createPortfolioStore() {
  const { subscribe, set, update } = writable<PortfolioStoreState>(initialState);

  return {
    subscribe,

    /**
     * Fetch all broker accounts
     */
    async fetchAccounts() {
      update(state => ({ ...state, loading: true, error: null }));

      try {
        const response = await fetch(`${apiBase}/api/brokers/accounts`);
        if (!response.ok) {
          throw new Error(`Failed to fetch accounts: ${response.statusText}`);
        }
        const accounts = await response.json();

        // Calculate portfolio summary
        const totalEquity = accounts.reduce((sum: number, acc: BrokerAccount) => sum + (acc.equity || 0), 0);
        const totalBalance = accounts.reduce((sum: number, acc: BrokerAccount) => sum + (acc.balance || 0), 0);
        const dailyPnL = totalEquity - totalBalance;

        // Calculate total drawdown (assuming equity starts at balance)
        const totalDrawdown = Math.max(0, totalBalance - totalEquity);
        const drawdownPercent = totalBalance > 0 ? (totalDrawdown / totalBalance) * 100 : 0;

        const portfolioSummary: PortfolioSummary = {
          totalEquity,
          dailyPnL,
          totalDrawdown,
          drawdownPercent
        };

        // Check for drawdown alert
        let drawdownAlert = null;
        if (drawdownPercent > 10) {
          drawdownAlert = { active: true, percent: drawdownPercent };
        }

        update(state => ({
          ...state,
          accounts,
          portfolioSummary,
          drawdownAlert,
          loading: false
        }));
      } catch (error) {
        // No mock data — fail gracefully with empty state
        update(state => ({
          ...state,
          accounts: [],
          portfolioSummary: { totalEquity: 0, dailyPnL: 0, totalDrawdown: 0, drawdownPercent: 0 },
          drawdownAlert: null,
          loading: false
        }));
      }
    },

    /**
     * Fetch routing matrix
     */
    async fetchRoutingMatrix() {
      update(state => ({ ...state, loading: true, error: null }));

      try {
        const response = await fetch(`${apiBase}/api/routing-matrix`);
        if (!response.ok) {
          throw new Error(`Failed to fetch routing matrix: ${response.statusText}`);
        }
        const data = await response.json();

        update(state => ({
          ...state,
          routingRules: data.rules || [],
          strategies: data.strategies || [],
          loading: false
        }));
      } catch (error) {
        // No mock data — fail gracefully with empty state
        update(state => ({
          ...state,
          routingRules: [],
          strategies: [],
          loading: false
        }));
      }
    },

    /**
     * Toggle routing rule
     */
    async toggleRoutingRule(strategyId: string, enabled: boolean) {
      try {
        await fetch(`${apiBase}/api/routing-matrix/rules`, {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ strategy_id: strategyId, enabled })
        });
      } catch (error) {
        console.error('[Portfolio] Failed to toggle routing rule:', error);
      }

      update(state => ({
        ...state,
        routingRules: state.routingRules.map(rule =>
          rule.strategy_id === strategyId ? { ...rule, enabled } : rule
        )
      }));
    },

    /**
     * Dismiss drawdown alert
     */
    dismissDrawdownAlert() {
      update(state => ({
        ...state,
        drawdownAlert: null
      }));
    },

    /**
     * Fetch strategy attribution data (Story 9.4 - AC #1)
     */
    async fetchAttribution() {
      update(state => ({ ...state, loading: true, error: null }));

      try {
        const response = await fetch(`${apiBase}/api/portfolio/attribution`);
        if (!response.ok) {
          throw new Error(`Failed to fetch attribution: ${response.statusText}`);
        }
        const data = await response.json();

        // Map API response shape {by_strategy, by_broker} to StrategyAttribution[]
        const attribution: StrategyAttribution[] = (data.by_strategy || []).map((s: {
          strategy: string;
          pnl: number;
          percentage: number;
          equity_contribution: number;
        }) => ({
          strategy_id: s.strategy,
          strategy_name: s.strategy,
          equity_contribution: s.equity_contribution ?? 0,
          pnl_contribution: s.pnl ?? 0,
          drawdown_contribution: 0,
          portfolio_percent: s.percentage ?? 0,
          broker_account: '',
          broker_name: ''
        }));

        update(state => ({
          ...state,
          attribution,
          loading: false
        }));
      } catch (error) {
        // No mock data — fail gracefully with empty state
        update(state => ({
          ...state,
          attribution: [],
          loading: false
        }));
      }
    },

    /**
     * Fetch correlation matrix data (Story 9.4 - AC #2, #3)
     */
    async fetchCorrelation() {
      update(state => ({ ...state, loading: true, error: null }));

      try {
        const response = await fetch(`${apiBase}/api/portfolio/correlation`);
        if (!response.ok) {
          throw new Error(`Failed to fetch correlation: ${response.statusText}`);
        }
        const data = await response.json();

        // API returns { matrix: [...], high_correlation_threshold, period_days, generated_at }
        // Map to CorrelationCell[] by filling in data_period from period_days
        const period = `${data.period_days ?? 30} days`;
        const correlationMatrix: CorrelationCell[] = (data.matrix || []).map((c: {
          strategy_a: string;
          strategy_b: string;
          correlation: number;
          period_days: number;
        }) => ({
          strategy_a: c.strategy_a,
          strategy_b: c.strategy_b,
          correlation: c.correlation,
          data_period: `${c.period_days ?? data.period_days ?? 30} days`
        }));

        update(state => ({
          ...state,
          correlationMatrix,
          loading: false
        }));
      } catch (error) {
        // No mock data — fail gracefully with empty state
        update(state => ({
          ...state,
          correlationMatrix: [],
          loading: false
        }));
      }
    },

    /**
     * Fetch performance metrics (Story 9.4 - AC all)
     */
    async fetchPerformance() {
      update(state => ({ ...state, loading: true, error: null }));

      try {
        const response = await fetch(`${apiBase}/api/portfolio/summary`);
        if (!response.ok) {
          throw new Error(`Failed to fetch performance: ${response.statusText}`);
        }
        const data = await response.json();

        update(state => ({
          ...state,
          performance: data.performance || null,
          loading: false
        }));
      } catch (error) {
        // No mock data — fail gracefully with empty state
        update(state => ({
          ...state,
          performance: null,
          loading: false
        }));
      }
    },

    /**
     * Initialize store
     */
    async initialize() {
      await Promise.all([
        this.fetchAccounts(),
        this.fetchRoutingMatrix()
      ]);
    }
  };
}

export const portfolioStore = createPortfolioStore();

// Derived stores for convenience
export const accounts = derived(portfolioStore, $store => $store.accounts);
export const portfolioSummary = derived(portfolioStore, $store => $store.portfolioSummary);
export const routingRules = derived(portfolioStore, $store => $store.routingRules);
export const strategies = derived(portfolioStore, $store => $store.strategies);
export const portfolioLoading = derived(portfolioStore, $store => $store.loading);
export const portfolioError = derived(portfolioStore, $store => $store.error);
export const drawdownAlert = derived(portfolioStore, $store => $store.drawdownAlert);

// Story 9.4 derived stores
export const attribution = derived(portfolioStore, $store => $store.attribution);
export const correlationMatrix = derived(portfolioStore, $store => $store.correlationMatrix);
export const performance = derived(portfolioStore, $store => $store.performance);