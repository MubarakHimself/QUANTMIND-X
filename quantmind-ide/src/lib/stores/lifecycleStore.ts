/**
 * Bot Lifecycle Store
 *
 * Manages state for the Bot Lifecycle Stage Report Viewer.
 * Handles fetching and caching of lifecycle data for bots.
 */

import { writable, derived, get } from 'svelte/store';
import { apiFetch } from '../api';
import { getBaseUrl } from '../config/api';

// =============================================================================
// Types
// =============================================================================

export type LifecycleStage = 'Born' | 'Backtest' | 'Paper' | 'Live' | 'Review';

export type DeclineRecoveryStatus = 'none' | 'declining' | 'recovering' | 'recovered';

export interface QAAnswer {
  question_id: string;
  question: string;
  answer: unknown;
  passed?: boolean;
}

export interface StageMetrics {
  win_rate?: number;
  drawdown?: number;
  pnl?: number;
  sharpe_ratio?: number;
  profit_factor?: number;
  total_trades?: number;
  consecutive_losses?: number;
  avg_win?: number;
  avg_loss?: number;
  recovery_factor?: number;
  max_drawdown_duration?: number;
}

export interface StageReport {
  stage: LifecycleStage;
  entered_at: string;
  exited_at?: string;
  q1_q20_answers: QAAnswer[];
  metrics: StageMetrics;
  decline_recovery_status?: DeclineRecoveryStatus;
  notes?: string;
}

export interface BotLifecycle {
  bot_id: string;
  current_stage: LifecycleStage;
  stage_history: StageReport[];
  current_report: StageReport;
  created_at?: string;
  updated_at?: string;
}

export interface LifecycleStats {
  total_bots: number;
  bots_by_stage: Record<string, number>;
  promotions_today: number;
  demotions_today: number;
  in_recovery: number;
  next_check: string;
}

export interface BotLifecycleSummary {
  bot_id: string;
  current_stage: LifecycleStage;
  created_at?: string;
  updated_at?: string;
  summary: {
    total_trades: number;
    win_rate?: number;
    pnl?: number;
    decline_recovery_status?: DeclineRecoveryStatus;
  };
}

export interface LifecycleListResponse {
  bots: BotLifecycleSummary[];
  total: number;
  limit: number;
  offset: number;
}

// =============================================================================
// Store State
// =============================================================================

interface LifecycleState {
  /** Map of bot_id to full lifecycle data */
  lifecycles: Map<string, BotLifecycle>;
  /** Currently selected bot ID for detailed view */
  selectedBotId: string | null;
  /** Currently selected stage for detailed report view */
  selectedStage: LifecycleStage | null;
  /** Loading state */
  loading: boolean;
  /** Error message if any */
  error: string | null;
  /** Stats overview */
  stats: LifecycleStats | null;
  /** List of bot summaries */
  botList: BotLifecycleSummary[];
  /** Total count for pagination */
  totalBots: number;
}

// Initial state
const initialState: LifecycleState = {
  lifecycles: new Map(),
  selectedBotId: null,
  selectedStage: null,
  loading: false,
  error: null,
  stats: null,
  botList: [],
  totalBots: 0
};

// =============================================================================
// Store Creation
// =============================================================================

function createLifecycleStore() {
  const { subscribe, set, update } = writable<LifecycleState>(initialState);

  return {
    subscribe,

    /**
     * Fetch full lifecycle data for a specific bot
     */
    async fetchBotLifecycle(botId: string): Promise<BotLifecycle | null> {
      update(state => ({ ...state, loading: true, error: null }));

      try {
        const lifecycle = await apiFetch<BotLifecycle>(
          `/api/bots/lifecycle/${botId}`
        );

        update(state => {
          const newLifecycles = new Map(state.lifecycles);
          newLifecycles.set(botId, lifecycle);
          return {
            ...state,
            lifecycles: newLifecycles,
            loading: false
          };
        });

        return lifecycle;
      } catch (e) {
        const message = e instanceof Error ? e.message : 'Failed to fetch lifecycle';
        update(state => ({ ...state, loading: false, error: message }));
        console.error('[LifecycleStore] Failed to fetch bot lifecycle:', e);
        return null;
      }
    },

    /**
     * Fetch lifecycle stats overview
     */
    async fetchStats(): Promise<LifecycleStats | null> {
      try {
        const stats = await apiFetch<LifecycleStats>(
          `/api/bots/lifecycle/stats/overview`
        );

        update(state => ({ ...state, stats }));
        return stats;
      } catch (e) {
        console.error('[LifecycleStore] Failed to fetch stats:', e);
        return null;
      }
    },

    /**
     * Fetch list of bots with lifecycle summary
     */
    async fetchBotList(
      stage?: LifecycleStage,
      limit = 50,
      offset = 0
    ): Promise<BotLifecycleSummary[]> {
      update(state => ({ ...state, loading: true, error: null }));

      try {
        let url = `/api/bots/lifecycle/?limit=${limit}&offset=${offset}`;
        if (stage) {
          url += `&stage=${encodeURIComponent(stage)}`;
        }

        const response = await apiFetch<LifecycleListResponse>(url);

        update(state => ({
          ...state,
          botList: response.bots,
          totalBots: response.total,
          loading: false
        }));

        return response.bots;
      } catch (e) {
        const message = e instanceof Error ? e.message : 'Failed to fetch bot list';
        update(state => ({ ...state, loading: false, error: message }));
        console.error('[LifecycleStore] Failed to fetch bot list:', e);
        return [];
      }
    },

    /**
     * Select a bot for detailed view
     */
    selectBot(botId: string | null) {
      update(state => ({
        ...state,
        selectedBotId: botId,
        selectedStage: null // Reset stage selection when changing bots
      }));

      // Fetch lifecycle if not already cached
      if (botId) {
        const currentState = get({ subscribe });
        const cached = currentState.lifecycles.get(botId);
        if (!cached) {
          this.fetchBotLifecycle(botId);
        }
      }
    },

    /**
     * Select a stage for detailed report view
     */
    selectStage(stage: LifecycleStage | null) {
      update(state => ({ ...state, selectedStage: stage }));
    },

    /**
     * Clear error state
     */
    clearError() {
      update(state => ({ ...state, error: null }));
    },

    /**
     * Reset store to initial state
     */
    reset() {
      set(initialState);
    }
  };
}

// Create and export the store instance
export const lifecycleStore = createLifecycleStore();

// =============================================================================
// Derived Stores
// =============================================================================

/** Currently selected bot's lifecycle data */
export const selectedBotLifecycle = derived(
  lifecycleStore,
  $state => {
    if (!$state.selectedBotId) return null;
    return $state.lifecycles.get($state.selectedBotId) || null;
  }
);

/** Currently selected stage's report */
export const selectedStageReport = derived(
  [lifecycleStore, selectedBotLifecycle],
  ([$state, $lifecycle]) => {
    if (!$state.selectedStage || !$lifecycle) return null;

    // First check stage_history
    const historyReport = $lifecycle.stage_history.find(
      r => r.stage === $state.selectedStage
    );
    if (historyReport) return historyReport;

    // Then check current_report if it matches
    if ($lifecycle.current_report.stage === $state.selectedStage) {
      return $lifecycle.current_report;
    }

    return null;
  }
);

/** Loading state */
export const lifecycleLoading = derived(
  lifecycleStore,
  $state => $state.loading
);

/** Error message */
export const lifecycleError = derived(
  lifecycleStore,
  $state => $state.error
);

/** Lifecycle stats */
export const lifecycleStats = derived(
  lifecycleStore,
  $state => $state.stats
);

/** List of bots */
export const lifecycleBotList = derived(
  lifecycleStore,
  $state => $state.botList
);

/** Total bot count */
export const lifecycleTotalBots = derived(
  lifecycleStore,
  $state => $state.totalBots
);

/** Stage progress (5 stages with current position) */
export const stageProgress = derived(
  selectedBotLifecycle,
  $lifecycle => {
    if (!$lifecycle) return null;

    const stages: LifecycleStage[] = ['Born', 'Backtest', 'Paper', 'Live', 'Review'];
    const currentIndex = stages.indexOf($lifecycle.current_stage);

    return {
      stages,
      currentIndex,
      currentStage: $lifecycle.current_stage,
      isCompleted: (index: number) => index < currentIndex,
      isCurrent: (index: number) => index === currentIndex,
      isPending: (index: number) => index > currentIndex
    };
  }
);

/** Check if bot is in decline/recovery */
export const isInRecovery = derived(
  selectedBotLifecycle,
  $lifecycle => {
    if (!$lifecycle) return false;
    const status = $lifecycle.current_report.decline_recovery_status;
    return status === 'declining' || status === 'recovering';
  }
);

/** Failed questions from Q1-Q20 */
export const failedQuestions = derived(
  selectedStageReport,
  $report => {
    if (!$report) return [];
    return $report.q1_q20_answers.filter(q => q.passed === false);
  }
);

/** Passed questions from Q1-Q20 */
export const passedQuestions = derived(
  selectedStageReport,
  $report => {
    if (!$report) return [];
    return $report.q1_q20_answers.filter(q => q.passed === true);
  }
);

// =============================================================================
// Utility Functions
// =============================================================================

/** Format a timestamp for display */
export function formatTimestamp(isoString: string | undefined): string {
  if (!isoString) return 'N/A';
  const date = new Date(isoString);
  return date.toLocaleString('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit'
  });
}

/** Format a duration in days */
export function formatDurationDays(days: number | undefined): string {
  if (days === undefined || days === null) return 'N/A';
  if (days === 1) return '1 day';
  return `${days} days`;
}

/** Get stage color CSS variable name */
export function getStageColor(stage: LifecycleStage): string {
  switch (stage) {
    case 'Born':
      return 'var(--stage-born, #6366f1)';
    case 'Backtest':
      return 'var(--stage-backtest, #f59e0b)';
    case 'Paper':
      return 'var(--stage-paper, #10b981)';
    case 'Live':
      return 'var(--stage-live, #3b82f6)';
    case 'Review':
      return 'var(--stage-review, #8b5cf6)';
    default:
      return 'var(--text-secondary, #94a3b8)';
  }
}

/** Get decline/recovery status color */
export function getRecoveryStatusColor(status: DeclineRecoveryStatus | undefined): string {
  switch (status) {
    case 'declining':
      return 'var(--accent-danger, #ef4444)';
    case 'recovering':
      return 'var(--accent-warning, #f59e0b)';
    case 'recovered':
      return 'var(--accent-success, #10b981)';
    case 'none':
    default:
      return 'var(--text-secondary, #94a3b8)';
  }
}

/** Format P&L with sign and currency */
export function formatPnL(pnl: number | undefined): string {
  if (pnl === undefined || pnl === null) return 'N/A';
  const sign = pnl >= 0 ? '+' : '';
  return `${sign}$${pnl.toLocaleString('en-US', { minimumFractionDigits: 2 })}`;
}

/** Format percentage */
export function formatPercent(value: number | undefined, decimals = 1): string {
  if (value === undefined || value === null) return 'N/A';
  return `${value.toFixed(decimals)}%`;
}

export default lifecycleStore;
