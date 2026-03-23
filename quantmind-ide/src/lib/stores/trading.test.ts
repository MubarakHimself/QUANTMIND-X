/**
 * Tests for Trading Store - Position Close Functions
 *
 * Tests the closePosition and closeAllPositions functions
 * with mocked API calls.
 *
 * Story 3-6: Manual Trade Controls UI
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { get } from 'svelte/store';
import {
	closePosition,
	closeAllPositions,
	closeLoading,
	closeError,
	activeBots,
	type CloseResult,
	type CloseAllResult
} from './trading';

// Mock fetch globally
global.fetch = vi.fn();

describe('Trading Store - closePosition', () => {
	beforeEach(() => {
		vi.clearAllMocks();
		closeLoading.set(false);
		closeError.set(null);
		activeBots.set([]);
	});

	afterEach(() => {
		vi.restoreAllMocks();
	});

	it('should set loading state to true when calling closePosition', async () => {
		// Arrange
		const mockResponse: CloseResult = {
			success: true,
			filled_price: 1.0850,
			slippage: 0.5,
			final_pnl: 25.50,
			message: 'Position closed successfully'
		};

		(global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
			ok: true,
			json: async () => mockResponse
		});

		// Act
		const promise = closePosition(12345, 'test-bot-001');

		// Assert - loading should be true immediately after call
		expect(get(closeLoading)).toBe(true);

		// Wait for resolution
		await promise;
	});

	it('should call API with correct parameters', async () => {
		// Arrange
		const mockResponse: CloseResult = {
			success: true,
			filled_price: 1.0850,
			slippage: 0.5,
			final_pnl: 25.50,
			message: 'Position closed successfully'
		};

		(global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
			ok: true,
			json: async () => mockResponse
		});

		// Act
		await closePosition(12345, 'test-bot-001');

		// Assert
		expect(global.fetch).toHaveBeenCalledWith('/api/v1/trading/close', {
			method: 'POST',
			headers: {
				'Content-Type': 'application/json'
			},
			body: JSON.stringify({
				position_ticket: 12345,
				bot_id: 'test-bot-001'
			})
		});
	});

	it('should return CloseResult on successful close', async () => {
		// Arrange
		const mockResponse: CloseResult = {
			success: true,
			filled_price: 1.0850,
			slippage: 0.5,
			final_pnl: 25.50,
			message: 'Position closed successfully'
		};

		(global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
			ok: true,
			json: async () => mockResponse
		});

		// Act
		const result = await closePosition(12345, 'test-bot-001');

		// Assert
		expect(result).not.toBeNull();
		expect(result?.success).toBe(true);
		expect(result?.filled_price).toBe(1.0850);
		expect(result?.final_pnl).toBe(25.50);
	});

	it('should set error state when API returns error', async () => {
		// Arrange
		(global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
			ok: false,
			json: async () => ({ message: 'Position not found' })
		});

		// Act
		const result = await closePosition(99999, 'test-bot-001');

		// Assert
		expect(result).toBeNull();
		expect(get(closeError)).toBe('Position not found');
	});

	it('should set error state when fetch throws', async () => {
		// Arrange
		(global.fetch as ReturnType<typeof vi.fn>).mockRejectedValueOnce(
			new Error('Network error')
		);

		// Act
		const result = await closePosition(12345, 'test-bot-001');

		// Assert
		expect(result).toBeNull();
		expect(get(closeError)).toBe('Network error');
	});

	it('should reset loading state after completion (success)', async () => {
		// Arrange
		const mockResponse: CloseResult = {
			success: true,
			filled_price: 1.0850,
			slippage: 0.5,
			final_pnl: 25.50,
			message: 'Position closed successfully'
		};

		(global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
			ok: true,
			json: async () => mockResponse
		});

		// Act
		await closePosition(12345, 'test-bot-001');

		// Assert
		expect(get(closeLoading)).toBe(false);
	});

	it('should reset loading state after completion (error)', async () => {
		// Arrange
		(global.fetch as ReturnType<typeof vi.fn>).mockRejectedValueOnce(
			new Error('Network error')
		);

		// Act
		await closePosition(12345, 'test-bot-001');

		// Assert
		expect(get(closeLoading)).toBe(false);
	});
});

describe('Trading Store - closeAllPositions', () => {
	beforeEach(() => {
		vi.clearAllMocks();
		closeLoading.set(false);
		closeError.set(null);
		activeBots.set([]);
	});

	afterEach(() => {
		vi.restoreAllMocks();
	});

	it('should set loading state to true when calling closeAllPositions', async () => {
		// Arrange
		const mockResponse: CloseAllResult = {
			success: true,
			results: [
				{
					position_ticket: 1001,
					status: 'filled',
					filled_price: 1.0850,
					slippage: 0.3,
					final_pnl: 25.50
				}
			]
		};

		(global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
			ok: true,
			json: async () => mockResponse
		});

		// Also mock the fetchActiveBots call
		(global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
			ok: true,
			json: async () => ({ bots: [] })
		});

		// Act
		const promise = closeAllPositions();

		// Assert
		expect(get(closeLoading)).toBe(true);

		await promise;
	});

	it('should call API without bot_id when none provided', async () => {
		// Arrange
		const mockResponse: CloseAllResult = {
			success: true,
			results: []
		};

		(global.fetch as ReturnType<typeof vi.fn>)
			.mockResolvedValueOnce({
				ok: true,
				json: async () => mockResponse
			})
			.mockResolvedValueOnce({
				ok: true,
				json: async () => ({ bots: [] })
			});

		// Act
		await closeAllPositions();

		// Assert - first call should be close-all
		expect(global.fetch).toHaveBeenCalledWith(
			'/api/v1/trading/close-all',
			expect.objectContaining({
				method: 'POST',
				body: JSON.stringify({ bot_id: null })
			})
		);
	});

	it('should call API with bot_id when provided', async () => {
		// Arrange
		const mockResponse: CloseAllResult = {
			success: true,
			results: []
		};

		(global.fetch as ReturnType<typeof vi.fn>)
			.mockResolvedValueOnce({
				ok: true,
				json: async () => mockResponse
			})
			.mockResolvedValueOnce({
				ok: true,
				json: async () => ({ bots: [] })
			});

		// Act
		await closeAllPositions('specific-bot');

		// Assert
		expect(global.fetch).toHaveBeenCalledWith(
			'/api/v1/trading/close-all',
			expect.objectContaining({
				body: JSON.stringify({ bot_id: 'specific-bot' })
			})
		);
	});

	it('should return CloseAllResult on success', async () => {
		// Arrange
		const mockResponse: CloseAllResult = {
			success: true,
			results: [
				{
					position_ticket: 1001,
					status: 'filled',
					filled_price: 1.0850,
					slippage: 0.3,
					final_pnl: 25.50
				},
				{
					position_ticket: 1002,
					status: 'rejected',
					message: 'Insufficient margin'
				}
			]
		};

		(global.fetch as ReturnType<typeof vi.fn>)
			.mockResolvedValueOnce({
				ok: true,
				json: async () => mockResponse
			})
			.mockResolvedValueOnce({
				ok: true,
				json: async () => ({ bots: [] })
			});

		// Act
		const result = await closeAllPositions();

		// Assert
		expect(result).not.toBeNull();
		expect(result?.success).toBe(true);
		expect(result?.results).toHaveLength(2);
		expect(result?.results[0].status).toBe('filled');
		expect(result?.results[1].status).toBe('rejected');
	});

	it('should set error state when API returns error', async () => {
		// Arrange
		(global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
			ok: false,
			json: async () => ({ message: 'Failed to close positions' })
		});

		// Act
		const result = await closeAllPositions();

		// Assert
		expect(result).toBeNull();
		expect(get(closeError)).toBe('Failed to close positions');
	});

	it('should reset loading state after completion', async () => {
		// Arrange
		const mockResponse: CloseAllResult = {
			success: true,
			results: []
		};

		(global.fetch as ReturnType<typeof vi.fn>)
			.mockResolvedValueOnce({
				ok: true,
				json: async () => mockResponse
			})
			.mockResolvedValueOnce({
				ok: true,
				json: async () => ({ bots: [] })
			});

		// Act
		await closeAllPositions();

		// Assert
		expect(get(closeLoading)).toBe(false);
	});
});

describe('Trading Store - State Management', () => {
	beforeEach(() => {
		vi.clearAllMocks();
		closeLoading.set(false);
		closeError.set(null);
		activeBots.set([]);
	});

	afterEach(() => {
		vi.restoreAllMocks();
	});

	it('should have closeLoading as writable store', () => {
		expect(typeof closeLoading.subscribe).toBe('function');
	});

	it('should have closeError as writable store', () => {
		expect(typeof closeError.subscribe).toBe('function');
	});

	it('should reset error when starting new close operation', async () => {
		// Arrange - set initial error
		closeError.set('Previous error');

		const mockResponse: CloseResult = {
			success: true,
			filled_price: 1.0850,
			slippage: 0.5,
			final_pnl: 25.50,
			message: 'Position closed successfully'
		};

		(global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
			ok: true,
			json: async () => mockResponse
		});

		// Act
		await closePosition(12345, 'test-bot-001');

		// Assert - error should be cleared
		expect(get(closeError)).toBeNull();
	});
});
