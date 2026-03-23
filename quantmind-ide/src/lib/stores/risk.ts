/**
 * Risk Physics Sensor Store
 *
 * Manages polling for physics sensor data from the risk API:
 * - Ising Model sensor
 * - Lyapunov Exponent sensor
 * - HMM regime sensor
 * - Kelly Engine
 *
 * Polls every 5 seconds per story requirements.
 */

import { writable, derived, get } from 'svelte/store';

// Types for physics sensor data
export interface PhysicsIsingData {
  magnetization: number;
  correlation_matrix: Record<string, number> | null;
  alert: 'normal' | 'warning' | 'critical';
}

export interface PhysicsLyapunovData {
  exponent_value: number;
  divergence_rate: number | null;
  alert: 'normal' | 'warning' | 'critical';
}

export interface PhysicsHMMData {
  current_state: string | null;
  transition_probabilities: Record<string, number> | null;
  is_shadow_mode: boolean;
  alert: 'normal' | 'warning' | 'critical';
}

export interface PhysicsKellyData {
  fraction: number;
  multiplier: number;
  house_of_money: boolean;
  kelly_fraction_setting: number;
}

export interface PhysicsSensorData {
  ising: PhysicsIsingData;
  lyapunov: PhysicsLyapunovData;
  hmm: PhysicsHMMData;
  kelly: PhysicsKellyData;
}

export interface PhysicsSensorState {
  data: PhysicsSensorData | null;
  loading: boolean;
  error: string | null;
  lastUpdated: Date | null;
}

// Initial state
const initialState: PhysicsSensorState = {
  data: null,
  loading: false,
  error: null,
  lastUpdated: null
};

// Create the store
function createPhysicsSensorStore() {
  const { subscribe, set, update } = writable<PhysicsSensorState>(initialState);

  let pollingInterval: ReturnType<typeof setInterval> | null = null;

  async function fetchData() {
    update(state => ({ ...state, loading: true, error: null }));

    try {
      const response = await fetch('/api/risk/physics');

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data: PhysicsSensorData = await response.json();

      update(state => ({
        ...state,
        data,
        loading: false,
        lastUpdated: new Date()
      }));
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to fetch physics sensor data';
      update(state => ({
        ...state,
        loading: false,
        error: errorMessage
      }));
    }
  }

  function startPolling(intervalMs: number = 5000) {
    // Fetch immediately
    fetchData();

    // Then poll at interval
    if (pollingInterval) {
      clearInterval(pollingInterval);
    }
    pollingInterval = setInterval(fetchData, intervalMs);
  }

  function stopPolling() {
    if (pollingInterval) {
      clearInterval(pollingInterval);
      pollingInterval = null;
    }
  }

  return {
    subscribe,
    fetch: fetchData,
    startPolling,
    stopPolling,
    reset: () => set(initialState)
  };
}

// Export the store
export const physicsSensorStore = createPhysicsSensorStore();

// Derived stores for individual sensors
export const isingData = derived(physicsSensorStore, $store => $store.data?.ising ?? null);
export const lyapunovData = derived(physicsSensorStore, $store => $store.data?.lyapunov ?? null);
export const hmmData = derived(physicsSensorStore, $store => $store.data?.hmm ?? null);
export const kellyData = derived(physicsSensorStore, $store => $store.data?.kelly ?? null);

// Loading and error states
export const physicsLoading = derived(physicsSensorStore, $store => $store.loading);
export const physicsError = derived(physicsSensorStore, $store => $store.error);
export const physicsLastUpdated = derived(physicsSensorStore, $store => $store.lastUpdated);


// =============================================================================
// Compliance Store (Story 4.6)
// =============================================================================

export interface AccountTagCompliance {
  tag: string;
  circuit_breaker_state: 'normal' | 'warning' | 'triggered';
  drawdown_pct: number;
  daily_halt_triggered: boolean;
  paused_strategies: number;
  last_check_utc: string;
}

export interface IslamicComplianceStatus {
  countdown_seconds: number;
  force_close_at: string | null;
  is_within_60min_window: boolean;
  is_within_30min_window: boolean;
  current_time_utc: string;
  active_positions_count: number;
}

export interface ComplianceData {
  account_tags: AccountTagCompliance[];
  islamic: IslamicComplianceStatus;
  overall_status: 'compliant' | 'warning' | 'critical';
}

export interface ComplianceState {
  data: ComplianceData | null;
  loading: boolean;
  error: string | null;
  lastUpdated: Date | null;
}

const initialComplianceState: ComplianceState = {
  data: null,
  loading: false,
  error: null,
  lastUpdated: null
};

function createComplianceStore() {
  const { subscribe, set, update } = writable<ComplianceState>(initialComplianceState);

  let pollingInterval: ReturnType<typeof setInterval> | null = null;

  async function fetchData() {
    update(state => ({ ...state, loading: true, error: null }));

    try {
      const response = await fetch('/api/risk/compliance');

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data: ComplianceData = await response.json();

      update(state => ({
        ...state,
        data,
        loading: false,
        lastUpdated: new Date()
      }));
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to fetch compliance data';
      update(state => ({
        ...state,
        loading: false,
        error: errorMessage
      }));
    }
  }

  function startPolling(intervalMs: number = 5000) {
    fetchData();
    if (pollingInterval) {
      clearInterval(pollingInterval);
    }
    pollingInterval = setInterval(fetchData, intervalMs);
  }

  function stopPolling() {
    if (pollingInterval) {
      clearInterval(pollingInterval);
      pollingInterval = null;
    }
  }

  return {
    subscribe,
    fetch: fetchData,
    startPolling,
    stopPolling,
    reset: () => set(initialComplianceState)
  };
}

export const complianceStore = createComplianceStore();
export const complianceData = derived(complianceStore, $store => $store.data);
export const complianceLoading = derived(complianceStore, $store => $store.loading);
export const complianceError = derived(complianceStore, $store => $store.error);
export const complianceLastUpdated = derived(complianceStore, $store => $store.lastUpdated);


// =============================================================================
// Prop Firm Store (Story 4.6)
// =============================================================================

export interface PropFirm {
  id: number;
  firm_name: string;
  account_id: string;
  daily_loss_limit_pct: number;
  target_profit_pct: number;
  risk_mode: string;
  account_type: string;
  created_at: string;
  updated_at: string;
}

export interface PropFirmState {
  firms: PropFirm[];
  loading: boolean;
  error: string | null;
}

const initialPropFirmState: PropFirmState = {
  firms: [],
  loading: false,
  error: null
};

function createPropFirmStore() {
  const { subscribe, set, update } = writable<PropFirmState>(initialPropFirmState);

  async function fetchFirms() {
    update(state => ({ ...state, loading: true, error: null }));

    try {
      const response = await fetch('/api/risk/prop-firms');

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const firms: PropFirm[] = await response.json();

      update(state => ({
        ...state,
        firms,
        loading: false
      }));
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to fetch prop firms';
      update(state => ({
        ...state,
        loading: false,
        error: errorMessage
      }));
    }
  }

  async function updatePropFirm(id: number, data: Partial<PropFirm>) {
    update(state => ({ ...state, loading: true, error: null }));

    try {
      const response = await fetch(`/api/risk/prop-firms/${id}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const updated: PropFirm = await response.json();

      update(state => ({
        ...state,
        firms: state.firms.map(f => f.id === id ? updated : f),
        loading: false
      }));

      return updated;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to update prop firm';
      update(state => ({
        ...state,
        loading: false,
        error: errorMessage
      }));
      throw err;
    }
  }

  async function createPropFirm(data: Omit<PropFirm, 'id' | 'created_at' | 'updated_at'>) {
    update(state => ({ ...state, loading: true, error: null }));

    try {
      const response = await fetch('/api/risk/prop-firms', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const created: PropFirm = await response.json();

      update(state => ({
        ...state,
        firms: [...state.firms, created],
        loading: false
      }));

      return created;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to create prop firm';
      update(state => ({
        ...state,
        loading: false,
        error: errorMessage
      }));
      throw err;
    }
  }

  async function deletePropFirm(id: number) {
    update(state => ({ ...state, loading: true, error: null }));

    try {
      const response = await fetch(`/api/risk/prop-firms/${id}`, {
        method: 'DELETE'
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      update(state => ({
        ...state,
        firms: state.firms.filter(f => f.id !== id),
        loading: false
      }));
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to delete prop firm';
      update(state => ({
        ...state,
        loading: false,
        error: errorMessage
      }));
      throw err;
    }
  }

  return {
    subscribe,
    fetch: fetchFirms,
    update: updatePropFirm,
    create: createPropFirm,
    delete: deletePropFirm,
    reset: () => set(initialPropFirmState)
  };
}

export const propFirmStore = createPropFirmStore();
export const propFirms = derived(propFirmStore, $store => $store.firms);
export const propFirmLoading = derived(propFirmStore, $store => $store.loading);
export const propFirmError = derived(propFirmStore, $store => $store.error);


// =============================================================================
// Calendar Gate Store (Story 4.6)
// =============================================================================

export interface CalendarEvent {
  event_id: string;
  title: string;
  event_type: string;
  impact: 'high' | 'medium' | 'low';
  event_time: string;
  currencies: string[];
  description?: string;
}

export interface CalendarBlackout {
  start_utc: string;
  end_utc: string;
  affected_strategies: string[];
  account_tag?: string;
  reason?: string;
}

export interface CalendarGateData {
  events: CalendarEvent[];
  blackouts: CalendarBlackout[];
}

export interface CalendarGateState {
  data: CalendarGateData | null;
  loading: boolean;
  error: string | null;
  lastUpdated: Date | null;
}

const initialCalendarState: CalendarGateState = {
  data: null,
  loading: false,
  error: null,
  lastUpdated: null
};

function createCalendarGateStore() {
  const { subscribe, set, update } = writable<CalendarGateState>(initialCalendarState);

  let pollingInterval: ReturnType<typeof setInterval> | null = null;

  async function fetchData() {
    update(state => ({ ...state, loading: true, error: null }));

    try {
      const response = await fetch('/api/risk/calendar/blackout');

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data: CalendarGateData = await response.json();

      update(state => ({
        ...state,
        data,
        loading: false,
        lastUpdated: new Date()
      }));
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to fetch calendar data';
      update(state => ({
        ...state,
        loading: false,
        error: errorMessage
      }));
    }
  }

  function startPolling(intervalMs: number = 5000) {
    fetchData();
    if (pollingInterval) {
      clearInterval(pollingInterval);
    }
    pollingInterval = setInterval(fetchData, intervalMs);
  }

  function stopPolling() {
    if (pollingInterval) {
      clearInterval(pollingInterval);
      pollingInterval = null;
    }
  }

  return {
    subscribe,
    fetch: fetchData,
    startPolling,
    stopPolling,
    reset: () => set(initialCalendarState)
  };
}

export const calendarGateStore = createCalendarGateStore();
export const calendarGateData = derived(calendarGateStore, $store => $store.data);
export const calendarGateLoading = derived(calendarGateStore, $store => $store.loading);
export const calendarGateError = derived(calendarGateStore, $store => $store.error);
export const calendarGateLastUpdated = derived(calendarGateStore, $store => $store.lastUpdated);


// =============================================================================
// Backtest Results Store (Story 4.6)
// =============================================================================

export interface BacktestSummary {
  id: string;
  ea_name: string;
  mode: string;
  run_at_utc: string;
  net_pnl: number;
  sharpe: number;
  max_drawdown: number;
  win_rate: number;
}

export interface BacktestDetail extends BacktestSummary {
  equity_curve: Array<{ timestamp: string; equity: number }>;
  trade_distribution: Array<{ bin: string; count: number }>;
  mode_params: Record<string, unknown>;
  report_type?: string;
  total_trades: number;
  profit_factor: number;
  avg_trade_pnl: number;
}

export interface BacktestState {
  list: BacktestSummary[];
  selectedDetail: BacktestDetail | null;
  loading: boolean;
  error: string | null;
}

const initialBacktestState: BacktestState = {
  list: [],
  selectedDetail: null,
  loading: false,
  error: null
};

function createBacktestStore() {
  const { subscribe, set, update } = writable<BacktestState>(initialBacktestState);

  async function fetchList() {
    update(state => ({ ...state, loading: true, error: null }));

    try {
      const response = await fetch('/api/backtests');

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const list: BacktestSummary[] = await response.json();

      update(state => ({
        ...state,
        list,
        loading: false
      }));
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to fetch backtests';
      update(state => ({
        ...state,
        loading: false,
        error: errorMessage
      }));
    }
  }

  async function fetchDetail(id: string) {
    update(state => ({ ...state, loading: true, error: null }));

    try {
      const response = await fetch(`/api/backtests/${id}`);

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const detail: BacktestDetail = await response.json();

      update(state => ({
        ...state,
        selectedDetail: detail,
        loading: false
      }));

      return detail;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to fetch backtest detail';
      update(state => ({
        ...state,
        loading: false,
        error: errorMessage
      }));
      throw err;
    }
  }

  function clearDetail() {
    update(state => ({ ...state, selectedDetail: null }));
  }

  return {
    subscribe,
    fetchList,
    fetchDetail,
    clearDetail,
    reset: () => set(initialBacktestState)
  };
}

export const backtestStore = createBacktestStore();
export const backtestList = derived(backtestStore, $store => $store.list);
export const selectedBacktest = derived(backtestStore, $store => $store.selectedDetail);
export const backtestLoading = derived(backtestStore, $store => $store.loading);
export const backtestError = derived(backtestStore, $store => $store.error);
