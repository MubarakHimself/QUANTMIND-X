/**
 * P2 Tests: Kill Switch Component Tests
 *
 * Tests the kill switch TopBar component UI behavior:
 * - ShieldAlert icon rendering by state
 * - Modal visibility by tier selection
 * - Button disabled states
 * - State transitions
 *
 * Note: These are UI logic tests (not rendered component tests).
 * Full rendered tests require Playwright component testing infrastructure.
 *
 * Priority: P2
 * Story: Epic 1 - Story 3-5 (Kill Switch UI)
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { get } from 'svelte/store';
import {
	killSwitchState,
	killSwitchFired,
	selectedTier,
	showKillSwitchModal,
	showEmergencyCloseModal,
	killSwitchLoading,
	killSwitchError,
	killSwitchAriaLabel,
	armKillSwitch,
	disarmKillSwitch,
	cancelKillSwitch,
	selectTier,
	confirmKillSwitch,
	type KillSwitchTier
} from '../../stores/kill-switch';

// Mock fetch
global.fetch = vi.fn();

describe('Kill Switch Component Logic', () => {
	beforeEach(() => {
		vi.clearAllMocks();
		vi.useFakeTimers();

		// Reset all stores
		killSwitchState.set('ready');
		killSwitchCountdown.set(0);
		selectedTier.set(null);
		showKillSwitchModal.set(false);
		showEmergencyCloseModal.set(false);
		killSwitchFired.set(false);
		killSwitchLoading.set(false);
		killSwitchError.set(null);
	});

	afterEach(() => {
		vi.restoreAllMocks();
		vi.useRealTimers();
	});

	describe('ShieldAlert Icon State', () => {
		it('should show normal icon when state is ready', () => {
			killSwitchState.set('ready');
			killSwitchFired.set(false);

			const label = get(killSwitchAriaLabel);
			expect(label).toBe('Emergency stop — click to arm');
		});

		it('should show armed state when armed', () => {
			killSwitchState.set('armed');
			killSwitchFired.set(false);

			const label = get(killSwitchAriaLabel);
			expect(label).toBe('Armed — click Confirm');
		});

		it('should show fired state when triggered', () => {
			killSwitchFired.set(true);
			killSwitchState.set('ready');

			const label = get(killSwitchAriaLabel);
			expect(label).toBe('Trading stopped');
		});
	});

	describe('Modal Visibility Logic', () => {
		it('should not show modal when ready', () => {
			killSwitchState.set('ready');

			expect(get(showKillSwitchModal)).toBe(false);
			expect(get(showEmergencyCloseModal)).toBe(false);
		});

		it('should not show modal when armed (countdown ongoing)', () => {
			armKillSwitch();
			// At this point, modal is NOT yet visible (countdown is 2 seconds)

			expect(get(showKillSwitchModal)).toBe(false);
		});

		it('should show modal after countdown completes', () => {
			armKillSwitch();
			vi.advanceTimersByTime(2000);

			expect(get(showKillSwitchModal)).toBe(true);
		});

		it('should show emergency modal for Tier 3 selection', () => {
			selectTier(3);

			expect(get(showKillSwitchModal)).toBe(false);
			expect(get(showEmergencyCloseModal)).toBe(true);
		});

		it('should close modal on cancel', () => {
			showKillSwitchModal.set(true);
			showEmergencyCloseModal.set(true);

			cancelKillSwitch();

			expect(get(showKillSwitchModal)).toBe(false);
			expect(get(showEmergencyCloseModal)).toBe(false);
		});
	});

	describe('Tier Selection Logic', () => {
		it('should select Tier 1 without opening emergency modal', () => {
			selectTier(1);

			expect(get(selectedTier)).toBe(1);
			expect(get(showKillSwitchModal)).toBe(false);
			expect(get(showEmergencyCloseModal)).toBe(false);
		});

		it('should select Tier 2 without opening emergency modal', () => {
			selectTier(2);

			expect(get(selectedTier)).toBe(2);
			expect(get(showEmergencyCloseModal)).toBe(false);
		});

		it('should route Tier 3 to emergency modal', () => {
			selectTier(3);

			expect(get(selectedTier)).toBe(3);
			expect(get(showEmergencyCloseModal)).toBe(true);
			expect(get(showKillSwitchModal)).toBe(false);
		});
	});

	describe('Button Disabled States', () => {
		it('should have no tier selected initially', () => {
			expect(get(selectedTier)).toBe(null);
		});

		it('should have tier selected after selectTier', () => {
			selectTier(1);
			expect(get(selectedTier)).toBe(1);
		});

		it('should clear selected tier after cancel', () => {
			selectTier(2);
			cancelKillSwitch();

			expect(get(selectedTier)).toBe(null);
		});

		it('should clear selected tier after successful trigger', async () => {
			(mockedFetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
				ok: true,
				json: async () => ({ success: true })
			});

			selectTier(1);
			await confirmKillSwitch();

			expect(get(selectedTier)).toBe(null);
		});
	});

	describe('Loading State During API Calls', () => {
		it('should not be loading initially', () => {
			expect(get(killSwitchLoading)).toBe(false);
		});

		it('should be loading during trigger', async () => {
			// First select a tier so the trigger proceeds
			selectTier(1);

			let resolve: (value: unknown) => void;
			(mockedFetch as ReturnType<typeof vi.fn>).mockImplementation(
				() => new Promise(r => { resolve = r; })
			);

			const promise = confirmKillSwitch();
			expect(get(killSwitchLoading)).toBe(true);

			resolve!({ ok: true, json: async () => ({ success: true }) });
			await promise;

			expect(get(killSwitchLoading)).toBe(false);
		});

		it('should not be loading after error', async () => {
			(mockedFetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
				ok: false,
				status: 500
			});

			await confirmKillSwitch();

			expect(get(killSwitchLoading)).toBe(false);
		});
	});

	describe('Error State Display', () => {
		it('should have no error initially', () => {
			expect(get(killSwitchError)).toBe(null);
		});

		it('should show error after failed trigger', async () => {
			// First select a tier
			selectTier(1);

			(mockedFetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
				ok: false,
				status: 500,
				json: async () => ({ detail: 'Server error' })
			});

			await confirmKillSwitch();

			expect(get(killSwitchError)).toBe('Server error');
		});

		it('should clear error on new arm', () => {
			killSwitchError.set('Previous error');

			armKillSwitch();

			expect(get(killSwitchError)).toBe(null);
		});

		it('should clear error on cancel', () => {
			killSwitchError.set('Some error');

			cancelKillSwitch();

			expect(get(killSwitchError)).toBe(null);
		});
	});

	describe('Fired State Persistence', () => {
		it('should persist fired state', () => {
			killSwitchFired.set(true);

			// Navigate away (simulated by changing other stores)
			killSwitchState.set('ready');

			expect(get(killSwitchFired)).toBe(true);
		});

		it('should show correct aria label when fired', () => {
			killSwitchFired.set(true);

			const label = get(killSwitchAriaLabel);
			expect(label).toBe('Trading stopped');
		});
	});
});

// Type for mocked fetch
const mockedFetch = global.fetch;

// Helper to reset countdown store
import { killSwitchCountdown } from '../../stores/kill-switch';
