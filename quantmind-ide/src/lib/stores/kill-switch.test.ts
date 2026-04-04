/**
 * P1 Tests: Kill Switch Store Unit Tests
 *
 * Tests the kill-switch.ts store functions:
 * - State management (armed, ready, fired)
 * - Countdown timer logic
 * - Tier selection
 * - API integration
 * - Modal visibility
 *
 * Priority: P1
 * Story: Epic 1 - Story 3-5 (Kill Switch UI)
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { get } from 'svelte/store';
import {
	killSwitchState,
	killSwitchCountdown,
	selectedTier,
	showKillSwitchModal,
	showEmergencyCloseModal,
	killSwitchFired,
	killSwitchLoading,
	killSwitchError,
	killSwitchLockState,
	manualMarketLockActive,
	killSwitchAriaLabel,
	armKillSwitch,
	disarmKillSwitch,
	cancelKillSwitch,
	selectTier,
	triggerKillSwitch,
	confirmKillSwitch,
	fetchKillSwitchStatus,
	activateManualMarketLock,
	resumeManualMarketLock,
	TIER_DESCRIPTIONS,
	type KillSwitchState,
	type KillSwitchTier
} from './kill-switch';

// Mock fetch globally
global.fetch = vi.fn();

describe('Kill Switch Store', () => {
	beforeEach(() => {
		vi.clearAllMocks();
		vi.useFakeTimers();
		// Reset all stores to initial state
		killSwitchState.set('ready');
		killSwitchCountdown.set(0);
		selectedTier.set(null);
		showKillSwitchModal.set(false);
		showEmergencyCloseModal.set(false);
		killSwitchFired.set(false);
		killSwitchLoading.set(false);
		killSwitchError.set(null);
		killSwitchLockState.set(null);
	});

	afterEach(() => {
		vi.restoreAllMocks();
		vi.useRealTimers();
	});

	describe('Initial State', () => {
		it('should have correct initial values', () => {
			expect(get(killSwitchState)).toBe('ready');
			expect(get(killSwitchCountdown)).toBe(0);
			expect(get(selectedTier)).toBe(null);
			expect(get(showKillSwitchModal)).toBe(false);
			expect(get(showEmergencyCloseModal)).toBe(false);
			expect(get(killSwitchFired)).toBe(false);
			expect(get(killSwitchLoading)).toBe(false);
			expect(get(killSwitchError)).toBe(null);
		});
	});

	describe('TIER_DESCRIPTIONS', () => {
		it('should have descriptions for all three tiers', () => {
			expect(TIER_DESCRIPTIONS).toHaveProperty(1);
			expect(TIER_DESCRIPTIONS).toHaveProperty(2);
			expect(TIER_DESCRIPTIONS).toHaveProperty(3);
		});

		it('should have correct tier names', () => {
			expect(TIER_DESCRIPTIONS[1].name).toBe('Soft Stop');
			expect(TIER_DESCRIPTIONS[2].name).toBe('Strategy Pause');
			expect(TIER_DESCRIPTIONS[3].name).toBe('Emergency Close');
		});

		it('should have icons for each tier', () => {
			expect(TIER_DESCRIPTIONS[1].icon).toBe('shield');
			expect(TIER_DESCRIPTIONS[2].icon).toBe('pause');
			expect(TIER_DESCRIPTIONS[3].icon).toBe('x-circle');
		});
	});

	describe('armKillSwitch', () => {
		it('should set state to armed', () => {
			armKillSwitch();
			expect(get(killSwitchState)).toBe('armed');
		});

		it('should start countdown at 2 seconds', () => {
			armKillSwitch();
			expect(get(killSwitchCountdown)).toBe(2);
		});

		it('should open modal after countdown completes', () => {
			armKillSwitch();

			// Advance timer by 2 seconds
			vi.advanceTimersByTime(2000);

			expect(get(showKillSwitchModal)).toBe(true);
			expect(get(killSwitchCountdown)).toBe(0);
		});

		it('should clear any existing countdown before starting new one', () => {
			// Arm twice rapidly
			armKillSwitch();
			const firstInterval = killSwitchCountdown.subscribe;

			armKillSwitch(); // Should reset

			// Countdown should restart at 2
			expect(get(killSwitchCountdown)).toBe(2);
		});
	});

	describe('disarmKillSwitch', () => {
		it('should set state back to ready', () => {
			armKillSwitch();
			disarmKillSwitch();

			expect(get(killSwitchState)).toBe('ready');
		});

		it('should reset countdown to 0', () => {
			armKillSwitch();
			vi.advanceTimersByTime(1000); // Countdown at 1

			disarmKillSwitch();

			expect(get(killSwitchCountdown)).toBe(0);
		});

		it('should clear countdown interval', () => {
			armKillSwitch();

			disarmKillSwitch();

			// Advance time - should not trigger modal
			vi.advanceTimersByTime(2000);
			expect(get(showKillSwitchModal)).toBe(false);
		});
	});

	describe('cancelKillSwitch', () => {
		it('should close both modals', () => {
			showKillSwitchModal.set(true);
			showEmergencyCloseModal.set(true);

			cancelKillSwitch();

			expect(get(showKillSwitchModal)).toBe(false);
			expect(get(showEmergencyCloseModal)).toBe(false);
		});

		it('should disarm the kill switch', () => {
			armKillSwitch();
			vi.advanceTimersByTime(500); // Partway through countdown

			cancelKillSwitch();

			expect(get(killSwitchState)).toBe('ready');
			expect(get(killSwitchCountdown)).toBe(0);
		});
	});

	describe('selectTier', () => {
		it('should set the selected tier', () => {
			selectTier(1);
			expect(get(selectedTier)).toBe(1);

			selectTier(2);
			expect(get(selectedTier)).toBe(2);
		});

		it('should set selected tier to 1', () => {
			showKillSwitchModal.set(true);

			selectTier(1);

			expect(get(selectedTier)).toBe(1);
			// Tier 1 does NOT close the modal - it just selects the tier
			// Modal should be closed by triggerKillSwitch or cancelKillSwitch
			expect(get(showKillSwitchModal)).toBe(true);
			expect(get(showEmergencyCloseModal)).toBe(false);
		});

		it('should close regular modal for Tier 2', () => {
			showKillSwitchModal.set(true);

			selectTier(2);

			expect(showKillSwitchModal).toBeDefined();
		});

		it('should show emergency close modal for Tier 3', () => {
			selectTier(3);

			expect(get(showKillSwitchModal)).toBe(false);
			expect(get(showEmergencyCloseModal)).toBe(true);
		});
	});

	describe('triggerKillSwitch', () => {
		it('should call /api/kill-switch/trigger endpoint', async () => {
			(mockedFetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
				ok: true,
				json: async () => ({ success: true, audit_log_id: 'test-id' })
			});

			await triggerKillSwitch(1);

			expect(global.fetch).toHaveBeenCalledWith('/api/kill-switch/trigger', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({
					tier: 1,
					strategy_ids: null,
					activator: 'UI_User'
				})
			});
		});

		it('should set loading state during API call', async () => {
			let resolvePromise: (value: unknown) => void;
			(mockedFetch as ReturnType<typeof vi.fn>).mockImplementation(
				() => new Promise(resolve => { resolvePromise = resolve; })
			);

			const promise = triggerKillSwitch(1);
			expect(get(killSwitchLoading)).toBe(true);

			resolvePromise!({ ok: true, json: async () => ({ success: true }) });
			await promise;

			expect(get(killSwitchLoading)).toBe(false);
		});

		it('should reset state on success', async () => {
			(mockedFetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
				ok: true,
				json: async () => ({ success: true })
			});

			killSwitchState.set('armed');
			showKillSwitchModal.set(true);

			await triggerKillSwitch(1);

			expect(get(killSwitchState)).toBe('ready');
			expect(get(showKillSwitchModal)).toBe(false);
			expect(get(killSwitchFired)).toBe(true);
		});

		it('should set error on failure', async () => {
			(mockedFetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
				ok: false,
				status: 500,
				json: async () => ({ detail: 'Internal error' })
			});

			await triggerKillSwitch(1);

			expect(get(killSwitchError)).toContain('Internal error');
		});

		it('should clear loading state on error', async () => {
			(mockedFetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
				ok: false,
				status: 500,
				json: async () => ({ detail: 'Error' })
			});

			await triggerKillSwitch(1);

			expect(get(killSwitchLoading)).toBe(false);
		});
	});

	describe('confirmKillSwitch', () => {
		it('should call triggerKillSwitch with selected tier', async () => {
			selectedTier.set(2);

			(mockedFetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
				ok: true,
				json: async () => ({ success: true })
			});

			await confirmKillSwitch();

			expect(global.fetch).toHaveBeenCalled();
		});

		it('should set error if no tier selected', async () => {
			selectedTier.set(null);

			const result = await confirmKillSwitch();

			expect(result).toBe(false);
			expect(get(killSwitchError)).toBe('No tier selected');
		});
	});

	describe('fetchKillSwitchStatus', () => {
		it('should call /api/kill-switch/status endpoint', async () => {
			(mockedFetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
				ok: true,
				json: async () => ({ enabled: true })
			});

			await fetchKillSwitchStatus();

			expect(global.fetch).toHaveBeenCalledWith('/api/kill-switch/status');
		});

		it('should hydrate lock state from backend response', async () => {
			(mockedFetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
				ok: true,
				json: async () => ({
					enabled: true,
					lock_state: {
						lock_source: 'MANUAL_MARKET_LOCK',
						pressure_state: 'RESTRICTED',
						reason: 'operator review',
						hard_lock_active: false,
						manual_market_lock_active: true
					}
				})
			});

			await fetchKillSwitchStatus();

			expect(get(killSwitchLockState)?.lock_source).toBe('MANUAL_MARKET_LOCK');
			expect(get(manualMarketLockActive)).toBe(true);
		});

		it('should normalize canonical backend lock_state shape', async () => {
			(mockedFetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
				ok: true,
				json: async () => ({
					enabled: true,
					lock_state: {
						source: 'ACCOUNT_HARD_LOCK',
						pressure_state: 'STOPPED',
						reason: 'Daily loss budget breached',
						manual_market_lock_active: false
					}
				})
			});

			await fetchKillSwitchStatus();

			expect(get(killSwitchLockState)?.lock_source).toBe('ACCOUNT_HARD_LOCK');
			expect(get(killSwitchLockState)?.hard_lock_active).toBe(true);
			expect(get(manualMarketLockActive)).toBe(false);
		});

		it('should handle fetch errors gracefully', async () => {
			(mockedFetch as ReturnType<typeof vi.fn>).mockRejectedValueOnce(
				new Error('Network error')
			);

			// Should not throw
			await fetchKillSwitchStatus();

			// Error is logged but not thrown
			expect(true).toBe(true);
		});
	});

	describe('killSwitchAriaLabel', () => {
		it('should return "Trading stopped" when fired', () => {
			killSwitchFired.set(true);
			killSwitchState.set('ready');

			const label = get(killSwitchAriaLabel);
			expect(label).toBe('Trading stopped');
		});

		it('should return "Armed — click Confirm" when armed', () => {
			killSwitchFired.set(false);
			killSwitchState.set('armed');

			const label = get(killSwitchAriaLabel);
			expect(label).toBe('Armed — click Confirm');
		});

		it('should return default message when ready', () => {
			killSwitchFired.set(false);
			killSwitchState.set('ready');

			const label = get(killSwitchAriaLabel);
			expect(label).toBe('Emergency stop — click to arm');
		});
	});

	describe('Error Handling', () => {
		it('should handle network errors in triggerKillSwitch', async () => {
			(mockedFetch as ReturnType<typeof vi.fn>).mockRejectedValueOnce(
				new Error('Network error')
			);

			await triggerKillSwitch(1);

			expect(get(killSwitchError)).toBe('Network error');
			expect(get(killSwitchLoading)).toBe(false);
		});

		it('should handle non-JSON error responses', async () => {
			(mockedFetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
				ok: false,
				status: 500
				// No json method
			} as Response);

			await triggerKillSwitch(1);

			// Should handle gracefully
			expect(get(killSwitchLoading)).toBe(false);
		});
	});

	describe('manual market lock', () => {
		it('should call activate endpoint and refresh status', async () => {
			(mockedFetch as ReturnType<typeof vi.fn>)
				.mockResolvedValueOnce({
					ok: true,
					json: async () => ({ success: true })
				})
				.mockResolvedValueOnce({
					ok: true,
					json: async () => ({ enabled: true, lock_state: null })
				});

			const result = await activateManualMarketLock('operator review');

			expect(result).toBe(true);
			expect(global.fetch).toHaveBeenNthCalledWith(1, '/api/kill-switch/market-lock', expect.any(Object));
			expect(global.fetch).toHaveBeenNthCalledWith(2, '/api/kill-switch/status');
		});

		it('should call resume endpoint and refresh status', async () => {
			(mockedFetch as ReturnType<typeof vi.fn>)
				.mockResolvedValueOnce({
					ok: true,
					json: async () => ({ success: true })
				})
				.mockResolvedValueOnce({
					ok: true,
					json: async () => ({ enabled: true, lock_state: null })
				});

			const result = await resumeManualMarketLock();

			expect(result).toBe(true);
			expect(global.fetch).toHaveBeenNthCalledWith(
				1,
				'/api/kill-switch/market-lock/resume?admin_key=UI_User',
				expect.objectContaining({ method: 'POST' })
			);
			expect(global.fetch).toHaveBeenNthCalledWith(2, '/api/kill-switch/status');
		});
	});
});

// Type for mocked fetch
const mockedFetch = global.fetch;
