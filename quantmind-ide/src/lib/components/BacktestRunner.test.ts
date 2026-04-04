// BacktestRunner Component Tests
// Uses Vitest + @testing-library/svelte
// NOTE: These tests are skipped due to Svelte 5 + @testing-library/svelte incompatibility
// The component uses Svelte 5 runes ($state, $props) which require proper Svelte 5 testing setup

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, waitFor } from '@testing-library/svelte';
import BacktestRunner from './BacktestRunner.svelte';
import { createBacktestClient } from '$lib/ws-client';

// Mock WebSocket client
vi.mock('$lib/ws-client', () => ({
  createBacktestClient: vi.fn()
}));

describe.skip('BacktestRunner', () => {
  it('connects to WebSocket on component mount', async () => {
    const { component } = render(BacktestRunner, { props: { baseUrl: 'ws://test' } });
    
    // Assert WebSocket connection established
    await waitFor(() => {
      expect(createBacktestClient).toHaveBeenCalled();
    });
  });

  it('updates progress bar on backtest_progress message', async () => {
    const mockClient = { on: vi.fn(), subscribe: vi.fn() };
    (createBacktestClient as any).mockResolvedValue(mockClient);
    
    const view = render(BacktestRunner) as any;
    const progressBar = view.getByTestId('progress-bar');
    
    // Simulate progress message
    mockClient.on.mock.calls[0][1]({ type: 'backtest_progress', data: { progress: 50 } });
    
    await waitFor(() => {
      expect(progressBar.style.width).toBe('50%');
    });
  });

  it('displays logs in real-time', async () => {
    const mockClient = { on: vi.fn() };
    (createBacktestClient as any).mockResolvedValue(mockClient);
    
    const view = render(BacktestRunner) as any;
    const logContainer = view.getByTestId('logs-container');
    
    // Simulate log message
    mockClient.on.mock.calls.find((call: [string, Function]) => call[0] === 'log_entry')![1]({
      type: 'log_entry',
      data: { level: 'INFO', message: 'Test log' }
    });
    
    await waitFor(() => {
      expect(logContainer.textContent).toContain('Test log');
    });
  });

  it('displays results on backtest completion', async () => {
    const mockClient = { on: vi.fn() };
    (createBacktestClient as any).mockResolvedValue(mockClient);
    
    const view = render(BacktestRunner) as any;
    const resultsSection = view.getByTestId('results-section');
    
    // Simulate complete message
    mockClient.on.mock.calls.find((call: [string, Function]) => call[0] === 'backtest_complete')![1]({
      type: 'backtest_complete',
      data: { final_balance: 12500, total_trades: 45 }
    });
    
    await waitFor(() => {
      expect(resultsSection).toBeDefined();
      expect(resultsSection.textContent).toContain('12500');
    });
  });
});
