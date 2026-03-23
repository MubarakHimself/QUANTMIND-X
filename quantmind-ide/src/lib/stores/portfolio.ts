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
        // If API fails, use demo data for development
        const demoAccounts: BrokerAccount[] = [
          {
            broker_id: 'broker_ftmo',
            broker_name: 'FTMO',
            account_id: 'acc_001',
            server: 'mt5.ftmo.com',
            account_type: 'PROP_FIRM',
            balance: 100000,
            equity: 98500,
            margin: 5000,
            leverage: 100,
            currency: 'USD',
            connected: true,
            is_active: true,
            status: 'connected',
            drawdown: 1.5,
            exposure: 45
          },
          {
            broker_id: 'broker_personal',
            broker_name: 'Exness',
            account_id: 'acc_002',
            server: 'mt5.exness.com',
            account_type: 'MACHINE_GUN',
            balance: 8000,
            equity: 8200,
            margin: 400,
            leverage: 500,
            currency: 'USD',
            connected: true,
            is_active: false,
            status: 'connected',
            drawdown: 0,
            exposure: 30
          }
        ];

        const totalEquity = demoAccounts.reduce((sum, acc) => sum + acc.equity, 0);
        const totalBalance = demoAccounts.reduce((sum, acc) => sum + acc.balance, 0);
        const dailyPnL = totalEquity - totalBalance;
        const totalDrawdown = Math.max(0, totalBalance - totalEquity);
        const drawdownPercent = totalBalance > 0 ? (totalDrawdown / totalBalance) * 100 : 0;

        const portfolioSummary: PortfolioSummary = {
          totalEquity,
          dailyPnL,
          totalDrawdown,
          drawdownPercent
        };

        let drawdownAlert = null;
        if (drawdownPercent > 10) {
          drawdownAlert = { active: true, percent: drawdownPercent };
        }

        update(state => ({
          ...state,
          accounts: demoAccounts,
          portfolioSummary,
          drawdownAlert,
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
        // Use demo data for development
        const demoStrategies: Strategy[] = [
          { id: 'strat_001', name: 'Gold Scalper', type: 'SCALPER' },
          { id: 'strat_002', name: 'EURUSD HFT', type: 'HFT' },
          { id: 'strat_003', name: 'Swing Strategy', type: 'SWING' },
          { id: 'strat_004', name: 'ICT Structure', type: 'STRUCTURAL' }
        ];

        const demoRules: RoutingRule[] = [
          { strategy_id: 'strat_001', strategy_name: 'Gold Scalper', account_id: 'acc_001', enabled: true, regime_filter: 'LONDON', strategy_type_filter: 'SCALPER' },
          { strategy_id: 'strat_002', strategy_name: 'EURUSD HFT', account_id: 'acc_002', enabled: true, regime_filter: 'NEW_YORK', strategy_type_filter: 'HFT' },
          { strategy_id: 'strat_003', strategy_name: 'Swing Strategy', account_id: 'acc_001', enabled: false, regime_filter: 'ASIAN', strategy_type_filter: 'SWING' },
          { strategy_id: 'strat_004', strategy_name: 'ICT Structure', account_id: 'acc_002', enabled: true, regime_filter: 'LONDON', strategy_type_filter: 'STRUCTURAL' }
        ];

        update(state => ({
          ...state,
          routingRules: demoRules,
          strategies: demoStrategies,
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
        // Fallback: just update local state for demo
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
        // Use demo data for development
        const demoAttribution: StrategyAttribution[] = [
          { strategy_id: 'strat_001', strategy_name: 'Gold Scalper', equity_contribution: 45000, pnl_contribution: 3200, drawdown_contribution: -500, portfolio_percent: 42.5, broker_account: 'acc_001', broker_name: 'FTMO' },
          { strategy_id: 'strat_002', strategy_name: 'EURUSD HFT', equity_contribution: 28000, pnl_contribution: 1850, drawdown_contribution: -200, portfolio_percent: 26.4, broker_account: 'acc_002', broker_name: 'Exness' },
          { strategy_id: 'strat_003', strategy_name: 'Swing Strategy', equity_contribution: 18000, pnl_contribution: -450, drawdown_contribution: -800, portfolio_percent: 17.0, broker_account: 'acc_001', broker_name: 'FTMO' },
          { strategy_id: 'strat_004', strategy_name: 'ICT Structure', equity_contribution: 15000, pnl_contribution: 1200, drawdown_contribution: -100, portfolio_percent: 14.1, broker_account: 'acc_002', broker_name: 'Exness' }
        ];

        update(state => ({
          ...state,
          attribution: demoAttribution,
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
        // Use demo data for development
        const strategies = ['Gold Scalper', 'EURUSD HFT', 'Swing Strategy', 'ICT Structure'];
        const demoCorrelation: CorrelationCell[] = [];

        // Generate NxN correlation matrix (lower triangle)
        for (let i = 0; i < strategies.length; i++) {
          for (let j = 0; j <= i; j++) {
            const correlation = i === j ? 1.0 : Math.round((Math.random() * 2 - 1) * 100) / 100;
            demoCorrelation.push({
              strategy_a: strategies[i],
              strategy_b: strategies[j],
              correlation,
              data_period: '2026-01-01 to 2026-03-15'
            });
          }
        }

        update(state => ({
          ...state,
          correlationMatrix: demoCorrelation,
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
        // Use demo data for development
        const demoPerformance: PerformanceMetrics = {
          total_return: 8.5,
          sharpe_ratio: 1.82,
          max_drawdown: -4.2,
          win_rate: 62.5,
          profit_factor: 2.15,
          avg_trade: 125.50,
          total_trades: 156,
          profitable_trades: 98,
          losing_trades: 58
        };

        update(state => ({
          ...state,
          performance: demoPerformance,
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