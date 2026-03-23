/**
 * P1 Tests: Canvas Navigation Store Tests
 *
 * Tests the canvas.ts store functions:
 * - Canvas type transitions (live_trading, risk, portfolio, workshop, research, development)
 * - Session ID management
 * - Context updates
 * - Store reset
 *
 * Priority: P1
 * Story: Epic 1 - Canvas Navigation (Story 1-6-9)
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { get } from 'svelte/store';
import {
	canvasContextStore,
	type CanvasContext
} from './canvas';

describe('Canvas Navigation Store', () => {
	beforeEach(() => {
		// Reset store to initial state
		canvasContextStore.reset();
	});

	describe('Initial State', () => {
		it('should have workshop as default canvas', () => {
			const ctx = get(canvasContextStore);
			expect(ctx.canvas).toBe('workshop');
		});

		it('should have empty session_id initially', () => {
			const ctx = get(canvasContextStore);
			expect(ctx.session_id).toBe('');
		});

		it('should have no entity initially', () => {
			const ctx = get(canvasContextStore);
			expect(ctx.entity).toBeUndefined();
		});
	});

	describe('setCanvas', () => {
		it('should transition to live_trading canvas', () => {
			canvasContextStore.setCanvas('live_trading');

			const ctx = get(canvasContextStore);
			expect(ctx.canvas).toBe('live_trading');
		});

		it('should transition to risk canvas', () => {
			canvasContextStore.setCanvas('risk');

			const ctx = get(canvasContextStore);
			expect(ctx.canvas).toBe('risk');
		});

		it('should transition to portfolio canvas', () => {
			canvasContextStore.setCanvas('portfolio');

			const ctx = get(canvasContextStore);
			expect(ctx.canvas).toBe('portfolio');
		});

		it('should transition to workshop canvas', () => {
			canvasContextStore.setCanvas('workshop');

			const ctx = get(canvasContextStore);
			expect(ctx.canvas).toBe('workshop');
		});

		it('should transition to research canvas', () => {
			canvasContextStore.setCanvas('research');

			const ctx = get(canvasContextStore);
			expect(ctx.canvas).toBe('research');
		});

		it('should transition to development canvas', () => {
			canvasContextStore.setCanvas('development');

			const ctx = get(canvasContextStore);
			expect(ctx.canvas).toBe('development');
		});

		it('should preserve session_id when changing canvas', () => {
			canvasContextStore.setSessionId('session-123');
			canvasContextStore.setCanvas('live_trading');

			const ctx = get(canvasContextStore);
			expect(ctx.canvas).toBe('live_trading');
			expect(ctx.session_id).toBe('session-123');
		});
	});

	describe('setSessionId', () => {
		it('should set the session ID', () => {
			canvasContextStore.setSessionId('session-456');

			const ctx = get(canvasContextStore);
			expect(ctx.session_id).toBe('session-456');
		});

		it('should update existing session ID', () => {
			canvasContextStore.setSessionId('session-001');
			canvasContextStore.setSessionId('session-002');

			const ctx = get(canvasContextStore);
			expect(ctx.session_id).toBe('session-002');
		});

		it('should preserve canvas when changing session ID', () => {
			canvasContextStore.setCanvas('risk');
			canvasContextStore.setSessionId('session-789');

			const ctx = get(canvasContextStore);
			expect(ctx.canvas).toBe('risk');
			expect(ctx.session_id).toBe('session-789');
		});
	});

	describe('setContext', () => {
		it('should update canvas via setContext', () => {
			canvasContextStore.setContext({ canvas: 'live_trading' });

			const ctx = get(canvasContextStore);
			expect(ctx.canvas).toBe('live_trading');
		});

		it('should update session_id via setContext', () => {
			canvasContextStore.setContext({ session_id: 'session-abc' });

			const ctx = get(canvasContextStore);
			expect(ctx.session_id).toBe('session-abc');
		});

		it('should update entity via setContext', () => {
			canvasContextStore.setContext({ entity: 'STRAT-001' });

			const ctx = get(canvasContextStore);
			expect(ctx.entity).toBe('STRAT-001');
		});

		it('should update multiple fields at once', () => {
			canvasContextStore.setContext({
				canvas: 'portfolio',
				session_id: 'session-xyz',
				entity: 'EA-001'
			});

			const ctx = get(canvasContextStore);
			expect(ctx.canvas).toBe('portfolio');
			expect(ctx.session_id).toBe('session-xyz');
			expect(ctx.entity).toBe('EA-001');
		});

		it('should preserve existing values when updating partial context', () => {
			canvasContextStore.setCanvas('risk');
			canvasContextStore.setSessionId('session-old');

			canvasContextStore.setContext({ entity: 'STRAT-NEW' });

			const ctx = get(canvasContextStore);
			expect(ctx.canvas).toBe('risk');
			expect(ctx.session_id).toBe('session-old');
			expect(ctx.entity).toBe('STRAT-NEW');
		});
	});

	describe('reset', () => {
		it('should reset canvas to workshop', () => {
			canvasContextStore.setCanvas('live_trading');
			canvasContextStore.setSessionId('session-123');
			canvasContextStore.setContext({ entity: 'STRAT-001' });

			canvasContextStore.reset();

			const ctx = get(canvasContextStore);
			expect(ctx.canvas).toBe('workshop');
		});

		it('should reset session_id to empty', () => {
			canvasContextStore.setSessionId('session-123');

			canvasContextStore.reset();

			const ctx = get(canvasContextStore);
			expect(ctx.session_id).toBe('');
		});

		it('should clear entity', () => {
			canvasContextStore.setContext({ entity: 'STRAT-001' });

			canvasContextStore.reset();

			const ctx = get(canvasContextStore);
			expect(ctx.entity).toBeUndefined();
		});
	});

	describe('CanvasContext Type', () => {
		it('should accept valid canvas types', () => {
			const validCanvases: CanvasContext['canvas'][] = [
				'live_trading',
				'risk',
				'portfolio',
				'workshop',
				'research',
				'development'
			];

			for (const canvas of validCanvases) {
				canvasContextStore.setCanvas(canvas);
				const ctx = get(canvasContextStore);
				expect(ctx.canvas).toBe(canvas);
			}
		});
	});

	describe('Immutability', () => {
		it('should create new context object on setCanvas', () => {
			const ctx1 = get(canvasContextStore);
			canvasContextStore.setCanvas('risk');
			const ctx2 = get(canvasContextStore);

			expect(ctx1).not.toBe(ctx2);
			expect(ctx1.canvas).toBe('workshop');
			expect(ctx2.canvas).toBe('risk');
		});

		it('should create new context object on setContext', () => {
			const ctx1 = get(canvasContextStore);
			canvasContextStore.setContext({ entity: 'TEST' });
			const ctx2 = get(canvasContextStore);

			expect(ctx1).not.toBe(ctx2);
		});
	});
});
