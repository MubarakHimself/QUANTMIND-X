/**
 * Tests for Risk Physics Sensor Store
 *
 * Tests the physicsSensorStore polling, fetch functions, and derived stores.
 * Story 4-5: Risk Canvas — Physics Sensor Tiles & Live Dashboard
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { get } from 'svelte/store';
import {
	physicsSensorStore,
	isingData,
	lyapunovData,
	hmmData,
	kellyData,
	physicsLoading,
	physicsError,
	physicsLastUpdated,
	complianceStore,
	complianceData,
	complianceLoading,
	complianceError,
	complianceLastUpdated,
	propFirmStore,
	propFirms,
	propFirmLoading,
	propFirmError,
	calendarGateStore,
	calendarGateData,
	calendarGateLoading,
	calendarGateError,
	calendarGateLastUpdated,
	backtestStore,
	backtestList,
	selectedBacktest,
	backtestLoading,
	backtestError,
	type PhysicsSensorData,
	type ComplianceData,
	type PropFirm,
	type CalendarGateData,
	type BacktestSummary,
	type BacktestDetail
} from './risk';

// Mock fetch globally
global.fetch = vi.fn();

describe('Physics Sensor Store', () => {
	beforeEach(() => {
		vi.clearAllMocks();
		physicsSensorStore.reset();
	});

	afterEach(() => {
		vi.restoreAllMocks();
		physicsSensorStore.stopPoling?.();
	});

	describe('fetchData', () => {
		it('should set loading state to true when fetching', async () => {
			// Arrange
			const mockData: PhysicsSensorData = {
				ising: { magnetization: 0.5, correlation_matrix: null, alert: 'normal' },
				lyapunov: { exponent_value: 0.1, divergence_rate: 0.01, alert: 'normal' },
				hmm: { current_state: 'TREND', transition_probabilities: {}, is_shadow_mode: false, alert: 'normal' },
				kelly: { fraction: 0.25, multiplier: 1.0, house_of_money: false, kelly_fraction_setting: 0.25 }
			};

			(global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
				ok: true,
				json: async () => mockData
			});

			// Act
			const promise = physicsSensorStore.fetch();

			// Assert - loading should be true immediately
			expect(get(physicsLoading)).toBe(true);

			await promise;
		});

		it('should call /api/risk/physics endpoint', async () => {
			// Arrange
			const mockData: PhysicsSensorData = {
				ising: { magnetization: 0.5, correlation_matrix: null, alert: 'normal' },
				lyapunov: { exponent_value: 0.1, divergence_rate: 0.01, alert: 'normal' },
				hmm: { current_state: 'TREND', transition_probabilities: {}, is_shadow_mode: false, alert: 'normal' },
				kelly: { fraction: 0.25, multiplier: 1.0, house_of_money: false, kelly_fraction_setting: 0.25 }
			};

			(global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
				ok: true,
				json: async () => mockData
			});

			// Act
			await physicsSensorStore.fetch();

			// Assert
			expect(global.fetch).toHaveBeenCalledWith('/api/risk/physics');
		});

		it('should update store with fetched data on success', async () => {
			// Arrange
			const mockData: PhysicsSensorData = {
				ising: { magnetization: 0.75, correlation_matrix: { A: 0.5 }, alert: 'warning' },
				lyapunov: { exponent_value: 0.3, divergence_rate: 0.02, alert: 'warning' },
				hmm: { current_state: 'RANGE', transition_probabilities: { TREND: 0.3, RANGE: 0.7 }, is_shadow_mode: true, alert: 'normal' },
				kelly: { fraction: 0.5, multiplier: 1.2, house_of_money: true, kelly_fraction_setting: 0.5 }
			};

			(global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
				ok: true,
				json: async () => mockData
			});

			// Act
			await physicsSensorStore.fetch();

			// Assert - check derived stores
			const ising = get(isingData);
			expect(ising).not.toBeNull();
			expect(ising?.magnetization).toBe(0.75);
			expect(ising?.alert).toBe('warning');

			const lyapunov = get(lyapunovData);
			expect(lyapunov?.exponent_value).toBe(0.3);

			const hmm = get(hmmData);
			expect(hmm?.current_state).toBe('RANGE');
			expect(hmm?.is_shadow_mode).toBe(true);

			const kelly = get(kellyData);
			expect(kelly?.house_of_money).toBe(true);
		});

		it('should set error state when API returns error', async () => {
			// Arrange
			(global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
				ok: false,
				status: 500,
				statusText: 'Internal Server Error'
			});

			// Act
			await physicsSensorStore.fetch();

			// Assert
			const error = get(physicsError);
			expect(error).toContain('500');
		});

		it('should set error state when fetch throws', async () => {
			// Arrange
			(global.fetch as ReturnType<typeof vi.fn>).mockRejectedValueOnce(
				new Error('Network error')
			);

			// Act
			await physicsSensorStore.fetch();

			// Assert
			const error = get(physicsError);
			expect(error).toBe('Network error');
		});

		it('should reset loading state after completion', async () => {
			// Arrange
			const mockData: PhysicsSensorData = {
				ising: { magnetization: 0.5, correlation_matrix: null, alert: 'normal' },
				lyapunov: { exponent_value: 0.1, divergence_rate: 0.01, alert: 'normal' },
				hmm: { current_state: 'TREND', transition_probabilities: {}, is_shadow_mode: false, alert: 'normal' },
				kelly: { fraction: 0.25, multiplier: 1.0, house_of_money: false, kelly_fraction_setting: 0.25 }
			};

			(global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
				ok: true,
				json: async () => mockData
			});

			// Act
			await physicsSensorStore.fetch();

			// Assert
			expect(get(physicsLoading)).toBe(false);
		});

		it('should update lastUpdated timestamp on success', async () => {
			// Arrange
			const mockData: PhysicsSensorData = {
				ising: { magnetization: 0.5, correlation_matrix: null, alert: 'normal' },
				lyapunov: { exponent_value: 0.1, divergence_rate: 0.01, alert: 'normal' },
				hmm: { current_state: 'TREND', transition_probabilities: {}, is_shadow_mode: false, alert: 'normal' },
				kelly: { fraction: 0.25, multiplier: 1.0, house_of_money: false, kelly_fraction_setting: 0.25 }
			};

			(global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
				ok: true,
				json: async () => mockData
			});

			// Act
			const beforeFetch = new Date();
			await physicsSensorStore.fetch();
			const afterFetch = new Date();

			// Assert
			const lastUpdated = get(physicsLastUpdated);
			expect(lastUpdated).not.toBeNull();
			expect(lastUpdated!.getTime()).toBeGreaterThanOrEqual(beforeFetch.getTime());
			expect(lastUpdated!.getTime()).toBeLessThanOrEqual(afterFetch.getTime());
		});
	});

	describe('Derived Stores', () => {
		beforeEach(() => {
			physicsSensorStore.reset();
		});

		it('isingData should return null when no data', () => {
			expect(get(isingData)).toBeNull();
		});

		it('lyapunovData should return null when no data', () => {
			expect(get(lyapunovData)).toBeNull();
		});

		it('hmmData should return null when no data', () => {
			expect(get(hmmData)).toBeNull();
		});

		it('kellyData should return null when no data', () => {
			expect(get(kellyData)).toBeNull();
		});

		it('physicsLoading should default to false', () => {
			expect(get(physicsLoading)).toBe(false);
		});

		it('physicsError should default to null', () => {
			expect(get(physicsError)).toBeNull();
		});

		it('physicsLastUpdated should default to null', () => {
			expect(get(physicsLastUpdated)).toBeNull();
		});
	});

	describe('Polling', () => {
		it('should have startPolling function', () => {
			expect(typeof physicsSensorStore.startPolling).toBe('function');
		});

		it('should have stopPolling function', () => {
			expect(typeof physicsSensorStore.stopPolling).toBe('function');
		});

		it('should have reset function', () => {
			expect(typeof physicsSensorStore.reset).toBe('function');
		});
	});
});

describe('Alert State Rendering', () => {
	beforeEach(() => {
		vi.clearAllMocks();
		physicsSensorStore.reset();
	});

	afterEach(() => {
		vi.restoreAllMocks();
	});

	it('should correctly render normal alert state', async () => {
		// Arrange
		const mockData: PhysicsSensorData = {
			ising: { magnetization: 0.8, correlation_matrix: null, alert: 'normal' },
			lyapunov: { exponent_value: 0.1, divergence_rate: 0.01, alert: 'normal' },
			hmm: { current_state: 'TREND', transition_probabilities: {}, is_shadow_mode: false, alert: 'normal' },
			kelly: { fraction: 0.25, multiplier: 1.0, house_of_money: false, kelly_fraction_setting: 0.25 }
		};

		(global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
			ok: true,
			json: async () => mockData
		});

		// Act
		await physicsSensorStore.fetch();

		// Assert
		const ising = get(isingData);
		expect(ising?.alert).toBe('normal');
	});

	it('should correctly render warning alert state', async () => {
		// Arrange
		const mockData: PhysicsSensorData = {
			ising: { magnetization: 0.4, correlation_matrix: null, alert: 'warning' },
			lyapunov: { exponent_value: 0.35, divergence_rate: 0.02, alert: 'warning' },
			hmm: { current_state: 'UNKNOWN', transition_probabilities: {}, is_shadow_mode: false, alert: 'warning' },
			kelly: { fraction: 0.2, multiplier: 0.7, house_of_money: false, kelly_fraction_setting: 0.25 }
		};

		(global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
			ok: true,
			json: async () => mockData
		});

		// Act
		await physicsSensorStore.fetch();

		// Assert
		expect(get(isingData)?.alert).toBe('warning');
		expect(get(lyapunovData)?.alert).toBe('warning');
	});

	it('should correctly render critical alert state', async () => {
		// Arrange
		const mockData: PhysicsSensorData = {
			ising: { magnetization: 0.1, correlation_matrix: null, alert: 'critical' },
			lyapunov: { exponent_value: 0.6, divergence_rate: 0.05, alert: 'critical' },
			hmm: { current_state: 'UNKNOWN', transition_probabilities: null, is_shadow_mode: false, alert: 'critical' },
			kelly: { fraction: 0.1, multiplier: 0.4, house_of_money: false, kelly_fraction_setting: 0.25 }
		};

		(global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
			ok: true,
			json: async () => mockData
		});

		// Act
		await physicsSensorStore.fetch();

		// Assert
		expect(get(isingData)?.alert).toBe('critical');
		expect(get(lyapunovData)?.alert).toBe('critical');
		expect(get(hmmData)?.alert).toBe('critical');
	});
});

describe('Independent Failure Isolation (NFR-R1)', () => {
	beforeEach(() => {
		vi.clearAllMocks();
		physicsSensorStore.reset();
	});

	afterEach(() => {
		vi.restoreAllMocks();
	});

	it('should handle partial data gracefully', async () => {
		// Arrange - API returns some null fields
		const mockData = {
			ising: { magnetization: 0.5, correlation_matrix: null, alert: 'normal' },
			lyapunov: null, // Missing
			hmm: { current_state: 'TREND', transition_probabilities: {}, is_shadow_mode: false, alert: 'normal' },
			kelly: { fraction: 0.25, multiplier: 1.0, house_of_money: false, kelly_fraction_setting: 0.25 }
		};

		(global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
			ok: true,
			json: async () => mockData
		});

		// Act
		await physicsSensorStore.fetch();

		// Assert - Should not throw, null fields handled
		expect(get(isingData)?.magnetization).toBe(0.5);
		expect(get(lyapunovData)).toBeNull(); // Missing field
		expect(get(hmmData)?.current_state).toBe('TREND');
	});
});

describe('Compliance Store (Story 4-6)', () => {
	beforeEach(() => {
		vi.clearAllMocks();
	});

	afterEach(() => {
		vi.restoreAllMocks();
	});

	describe('fetchData', () => {
		it('should call /api/risk/compliance endpoint', async () => {
			const mockData: ComplianceData = {
				account_tags: [
					{ tag: 'FTMO', circuit_breaker_state: 'active', drawdown_pct: 5.2, daily_halt_triggered: false }
				],
				islamic: { countdown_seconds: 3600, force_close_at: '21:45:00' },
				overall_status: 'compliant'
			};

			(global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
				ok: true,
				json: async () => mockData
			});

			await complianceStore.fetch();

			expect(global.fetch).toHaveBeenCalledWith('/api/risk/compliance');
		});

		it('should update store with compliance data', async () => {
			const mockData: ComplianceData = {
				account_tags: [
					{ tag: 'FTMO', circuit_breaker_state: 'active', drawdown_pct: 8.5, daily_halt_triggered: true }
				],
				islamic: { countdown_seconds: 1800, force_close_at: '21:45:00' },
				overall_status: 'warning'
			};

			(global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
				ok: true,
				json: async () => mockData
			});

			await complianceStore.fetch();

			const data = get(complianceData);
			expect(data).not.toBeNull();
			expect(data?.overall_status).toBe('warning');
			expect(data?.account_tags[0].daily_halt_triggered).toBe(true);
		});

		it('should set loading state correctly', async () => {
			(global.fetch as ReturnType<typeof vi.fn>).mockImplementation(() => new Promise(() => {}));

			const promise = complianceStore.fetch();
			expect(get(complianceLoading)).toBe(true);

			promise?.catch(() => {}); // Catch to avoid unhandled rejection
			global.fetch.mockClear();
		});

		it('should handle API errors', async () => {
			(global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
				ok: false,
				status: 500
			});

			await complianceStore.fetch();

			expect(get(complianceError)).not.toBeNull();
		});
	});

	describe('Derived Stores', () => {
		beforeEach(() => {
			complianceStore.reset();
		});

		it('complianceData should default to null', () => {
			expect(get(complianceData)).toBeNull();
		});

		it('complianceLoading should default to false', () => {
			expect(get(complianceLoading)).toBe(false);
		});

		it('complianceError should default to null', () => {
			expect(get(complianceError)).toBeNull();
		});

		it('complianceLastUpdated should default to null', () => {
			expect(get(complianceLastUpdated)).toBeNull();
		});
	});
});

describe('PropFirm Store (Story 4-6)', () => {
	beforeEach(() => {
		vi.clearAllMocks();
	});

	afterEach(() => {
		vi.restoreAllMocks();
	});

	describe('fetchData', () => {
		it('should call /api/risk/prop-firms endpoint', async () => {
			const mockData: PropFirm[] = [
				{ id: 1, firm_name: 'FTMO', account_id: 'acc-001', daily_loss_limit_pct: 5, target_profit_pct: 10, risk_mode: 'normal', account_type: 'demo', created_at: '2026-01-01', updated_at: '2026-01-01' }
			];

			(global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
				ok: true,
				json: async () => mockData
			});

			await propFirmStore.fetch();

			expect(global.fetch).toHaveBeenCalledWith('/api/risk/prop-firms');
		});

		it('should update store with prop firm data', async () => {
			const mockData: PropFirm[] = [
				{ id: 1, firm_name: 'FTMO', account_id: 'acc-001', daily_loss_limit_pct: 5, target_profit_pct: 10, risk_mode: 'normal', account_type: 'demo', created_at: '2026-01-01', updated_at: '2026-01-01' },
				{ id: 2, firm_name: 'ELEV8', account_id: 'acc-002', daily_loss_limit_pct: 4, target_profit_pct: 8, risk_mode: 'aggressive', account_type: 'live', created_at: '2026-01-01', updated_at: '2026-01-01' }
			];

			(global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
				ok: true,
				json: async () => mockData
			});

			await propFirmStore.fetch();

			const firms = get(propFirms);
			expect(firms).toHaveLength(2);
			expect(firms[0].firm_name).toBe('FTMO');
		});

		it('should handle errors', async () => {
			(global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
				ok: false,
				status: 404
			});

			await propFirmStore.fetch();

			expect(get(propFirmError)).not.toBeNull();
		});
	});

	describe('Derived Stores', () => {
		beforeEach(() => {
			propFirmStore.reset();
		});

		it('propFirms should default to empty array', () => {
			expect(get(propFirms)).toEqual([]);
		});

		it('propFirmLoading should default to false', () => {
			expect(get(propFirmLoading)).toBe(false);
		});

		it('propFirmError should default to null', () => {
			expect(get(propFirmError)).toBeNull();
		});
	});
});

describe('Calendar Gate Store (Story 4-6)', () => {
	beforeEach(() => {
		vi.clearAllMocks();
	});

	afterEach(() => {
		vi.restoreAllMocks();
	});

	describe('fetchData', () => {
		it('should call /api/risk/calendar/blackout endpoint', async () => {
			const mockData: CalendarGateData = {
				events: [
					{ event_name: 'NFP', impact: 'high', datetime_utc: '2026-03-21T13:30:00Z', blackout_minutes: 60 }
				],
				blackouts: []
			};

			(global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
				ok: true,
				json: async () => mockData
			});

			await calendarGateStore.fetch();

			expect(global.fetch).toHaveBeenCalledWith('/api/risk/calendar/blackout');
		});

		it('should update store with calendar gate data', async () => {
			const mockData: CalendarGateData = {
				events: [
					{ event_name: 'ECB Rate Decision', impact: 'high', datetime_utc: '2026-03-20T12:45:00Z', blackout_minutes: 120 }
				],
				blackouts: [
					{ start_utc: '2026-03-20T10:45:00Z', end_utc: '2026-03-20T14:45:00Z', affected_strategies: ['scalper'] }
				]
			};

			(global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
				ok: true,
				json: async () => mockData
			});

			await calendarGateStore.fetch();

			const data = get(calendarGateData);
			expect(data).not.toBeNull();
			expect(data?.events).toHaveLength(1);
			expect(data?.blackouts).toHaveLength(1);
		});

		it('should handle errors', async () => {
			(global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
				ok: false,
				status: 500
			});

			await calendarGateStore.fetch();

			expect(get(calendarGateError)).not.toBeNull();
		});
	});

	describe('Derived Stores', () => {
		beforeEach(() => {
			calendarGateStore.reset();
		});

		it('calendarGateData should default to null', () => {
			expect(get(calendarGateData)).toBeNull();
		});

		it('calendarGateLoading should default to false', () => {
			expect(get(calendarGateLoading)).toBe(false);
		});

		it('calendarGateError should default to null', () => {
			expect(get(calendarGateError)).toBeNull();
		});

		it('calendarGateLastUpdated should default to null', () => {
			expect(get(calendarGateLastUpdated)).toBeNull();
		});
	});
});

describe('Backtest Store (Story 4-6)', () => {
	beforeEach(() => {
		vi.clearAllMocks();
	});

	afterEach(() => {
		vi.restoreAllMocks();
	});

	describe('fetchList', () => {
		it('should call /api/backtests endpoint', async () => {
			const mockData: BacktestSummary[] = [
				{ id: '1', ea_name: 'TrendScaper', mode: 'VANILLA', run_at_utc: '2026-03-15T10:00:00Z', net_pnl: 2500, sharpe: 1.8, max_drawdown: 3.2, win_rate: 62 }
			];

			(global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
				ok: true,
				json: async () => mockData
			});

			await backtestStore.fetchList();

			expect(global.fetch).toHaveBeenCalledWith('/api/backtests');
		});

		it('should update store with backtest list', async () => {
			const mockData: BacktestSummary[] = [
				{ id: '1', ea_name: 'TrendScaper', mode: 'VANILLA', run_at_utc: '2026-03-15T10:00:00Z', net_pnl: 2500, sharpe: 1.8, max_drawdown: 3.2, win_rate: 62 },
				{ id: '2', ea_name: 'TrendScaper', mode: 'SPICED', run_at_utc: '2026-03-15T11:00:00Z', net_pnl: 3200, sharpe: 2.1, max_drawdown: 2.8, win_rate: 65 }
			];

			(global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
				ok: true,
				json: async () => mockData
			});

			await backtestStore.fetchList();

			const list = get(backtestList);
			expect(list).toHaveLength(2);
		});

		it('should handle errors', async () => {
			(global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
				ok: false,
				status: 500
			});

			await backtestStore.fetchList();

			expect(get(backtestError)).not.toBeNull();
		});
	});

	describe('fetchDetail', () => {
		it('should call /api/backtests/{id} endpoint', async () => {
			const mockDetail: BacktestDetail = {
				id: '1',
				ea_name: 'TrendScaper',
				mode: 'VANILLA',
				run_at_utc: '2026-03-15T10:00:00Z',
				net_pnl: 2500,
				sharpe: 1.8,
				max_drawdown: 3.2,
				win_rate: 62,
				equity_curve: [{ timestamp: '2026-03-01T00:00:00Z', equity: 10000 }, { timestamp: '2026-03-15T00:00:00Z', equity: 12500 }]
			};

			(global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
				ok: true,
				json: async () => mockDetail
			});

			await backtestStore.fetchDetail('1');

			expect(global.fetch).toHaveBeenCalledWith('/api/backtests/1');
		});

		it('should update selectedDetail with backtest detail', async () => {
			const mockDetail: BacktestDetail = {
				id: '1',
				ea_name: 'TrendScaper',
				mode: 'VANILLA',
				run_at_utc: '2026-03-15T10:00:00Z',
				net_pnl: 2500,
				sharpe: 1.8,
				max_drawdown: 3.2,
				win_rate: 62,
				equity_curve: [{ timestamp: '2026-03-01T00:00:00Z', equity: 10000 }]
			};

			(global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
				ok: true,
				json: async () => mockDetail
			});

			await backtestStore.fetchDetail('1');

			const detail = get(selectedBacktest);
			expect(detail).not.toBeNull();
			expect(detail?.equity_curve).toHaveLength(1);
		});
	});

	describe('Derived Stores', () => {
		beforeEach(() => {
			backtestStore.reset();
		});

		it('backtestList should default to empty array', () => {
			expect(get(backtestList)).toEqual([]);
		});

		it('selectedBacktest should default to null', () => {
			expect(get(selectedBacktest)).toBeNull();
		});

		it('backtestLoading should default to false', () => {
			expect(get(backtestLoading)).toBe(false);
		});

		it('backtestError should default to null', () => {
			expect(get(backtestError)).toBeNull();
		});
	});
});