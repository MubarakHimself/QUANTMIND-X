<script lang="ts">
  import { onMount, onDestroy, createEventDispatcher } from 'svelte';
  import { 
    WebSocketClient, 
    createBacktestClient
  } from '../ws-client';
  import type { WebSocketMessage } from '../ws-client';
  
  // Props
  export let baseUrl: string = 'http://localhost:8000';
  export let strategyCode: string = '';
  export let strategyName: string = 'MyStrategy';
  
  // Backtest parameters
  let symbol = 'EURUSD';
  let timeframe = 'H1';
  let variant: 'vanilla' | 'spiced' | 'vanilla_full' | 'spiced_full' = 'spiced';
  let startDate = new Date(Date.now() - 30 * 24 * 60 * 60 * 1000).toISOString().split('T')[0];
  let endDate = new Date().toISOString().split('T')[0];
  
  // State
  let isRunning = false;
  let progress = 0;
  let status = '';
  let logs: Array<{ timestamp: string; level: string; message: string }> = [];
  let results: Record<string, unknown> | null = null;
  let error: string | null = null;
  
  // WebSocket
  let wsClient: WebSocketClient | null = null;
  let wsConnected = false;
  
  // Options
  const symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'XAUUSD', 'BTCUSD'];
  const timeframes = ['M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1'];
  const variants = [
    { value: 'vanilla', label: 'Vanilla' },
    { value: 'spiced', label: 'Spiced' },
    { value: 'vanilla_full', label: 'Vanilla + Walk-Forward' },
    { value: 'spiced_full', label: 'Spiced + Walk-Forward' }
  ];
  
  // Log container reference for auto-scroll
  let logContainer: HTMLDivElement;
  
  const dispatch = createEventDispatcher();
  
  onMount(async () => {
    await connectWebSocket();
  });
  
  onDestroy(() => {
    disconnectWebSocket();
  });
  
  async function connectWebSocket() {
    try {
      wsClient = await createBacktestClient(baseUrl);
      wsConnected = true;
      
      // Register event handlers
      wsClient.on('backtest_start', handleBacktestStart);
      wsClient.on('backtest_progress', handleBacktestProgress);
      wsClient.on('backtest_complete', handleBacktestComplete);
      wsClient.on('backtest_error', handleBacktestError);
      wsClient.on('log_entry', handleLogEntry);
      
    } catch (e) {
      console.error('Failed to connect WebSocket:', e);
      wsConnected = false;
      addLog('ERROR', `Failed to connect to WebSocket: ${e}`);
    }
  }
  
  function disconnectWebSocket() {
    if (wsClient) {
      wsClient.off('backtest_start', handleBacktestStart);
      wsClient.off('backtest_progress', handleBacktestProgress);
      wsClient.off('backtest_complete', handleBacktestComplete);
      wsClient.off('backtest_error', handleBacktestError);
      wsClient.off('log_entry', handleLogEntry);
      wsClient.disconnect();
      wsClient = null;
      wsConnected = false;
    }
  }
  
  function handleBacktestStart(message: WebSocketMessage) {
    // Unwrap data property from WebSocket message
    const payload = message.data as Record<string, unknown> | undefined;
    const data = (payload || message) as Record<string, unknown>;

    isRunning = true;
    progress = 0;
    status = `Starting ${(data.variant as string) || 'backtest'} for ${(data.symbol as string) || ''}`;
    logs = [];
    results = null;
    error = null;
    addLog('INFO', `Backtest started: ${data.variant} on ${data.symbol} ${data.timeframe}`);
  }
  
  function handleBacktestProgress(message: WebSocketMessage) {
    // Unwrap data property from WebSocket message
    const payload = message.data as Record<string, unknown> | undefined;
    const data = (payload || message) as Record<string, unknown>;

    progress = (data.progress as number) || 0;
    status = (data.status as string) || '';
    addLog('PROGRESS', `[${(data.progress as number)?.toFixed(1) || '0'}%] ${data.status}`);
  }
  
  function handleBacktestComplete(message: WebSocketMessage) {
    // Unwrap data property from WebSocket message
    const payload = message.data as Record<string, unknown> | undefined;
    const data = (payload || message) as Record<string, unknown>;

    isRunning = false;
    progress = 100;
    status = 'Backtest completed';
    results = data;
    addLog('COMPLETE', `Backtest complete! Final balance: ${(data.final_balance as number)?.toFixed(2)}, Trades: ${data.total_trades}`);
    dispatch('complete', results);
  }
  
  function handleBacktestError(message: WebSocketMessage) {
    // Unwrap data property from WebSocket message
    const payload = message.data as Record<string, unknown> | undefined;
    const data = (payload || message) as Record<string, unknown>;

    isRunning = false;
    error = (data.error as string) || 'Unknown error';
    addLog('ERROR', `Backtest error: ${data.error}`);
    dispatch('error', data);
  }
  
  function handleLogEntry(message: WebSocketMessage) {
    // Unwrap data property from WebSocket message
    const payload = message.data as Record<string, unknown> | undefined;
    const data = (payload || message) as Record<string, unknown>;

    addLog((data.level as string) || 'INFO', (data.message as string) || '');
  }
  
  function addLog(level: string, message: string) {
    const logEntry = {
      timestamp: new Date().toISOString(),
      level,
      message
    };
    logs = [...logs, logEntry];
    
    // Auto-scroll to bottom
    setTimeout(() => {
      if (logContainer) {
        logContainer.scrollTop = logContainer.scrollHeight;
      }
    }, 10);
  }
  
  async function runBacktest() {
    if (isRunning || !wsConnected) return;
    
    isRunning = true;
    progress = 0;
    status = 'Initializing...';
    logs = [];
    results = null;
    error = null;
    
    addLog('INFO', 'Preparing backtest request...');
    
    try {
      const response = await fetch(`${baseUrl}/api/v1/backtest/run`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          symbol,
          timeframe,
          variant,
          start_date: startDate,
          end_date: endDate,
          strategy_code: strategyCode,
          strategy_name: strategyName
        })
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error: ${response.status}`);
      }
      
      const data = await response.json();
      addLog('INFO', `Backtest started with ID: ${data.backtest_id}`);
      
    } catch (e) {
      isRunning = false;
      error = `Failed to start backtest: ${e}`;
      addLog('ERROR', error);
    }
  }
  
  function stopBacktest() {
    isRunning = false;
    status = 'Stopped';
    addLog('WARN', 'Backtest stopped by user');
  }
  
  function clearLogs() {
    logs = [];
  }
  
  function downloadResults() {
    if (!results) return;
    
    const dataStr = JSON.stringify(results, null, 2);
    const blob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `backtest_results_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }
  
  function getLogColor(level: string): string {
    switch (level) {
      case 'INFO': return '#3b82f6';
      case 'WARNING': case 'WARN': return '#f59e0b';
      case 'ERROR': return '#ef4444';
      case 'PROGRESS': return '#10b981';
      case 'COMPLETE': return '#8b5cf6';
      default: return '#6b7280';
    }
  }
</script>

<div class="backtest-runner">
  <div class="header">
    <h2>Backtest Runner</h2>
    <div class="connection-status" class:connected={wsConnected}>
      {wsConnected ? 'Connected' : 'Disconnected'}
    </div>
  </div>
  
  <div class="controls">
    <div class="control-group">
      <label for="symbol">Symbol</label>
      <select id="symbol" bind:value={symbol} disabled={isRunning}>
        {#each symbols as s}
          <option value={s}>{s}</option>
        {/each}
      </select>
    </div>
    
    <div class="control-group">
      <label for="timeframe">Timeframe</label>
      <select id="timeframe" bind:value={timeframe} disabled={isRunning}>
        {#each timeframes as tf}
          <option value={tf}>{tf}</option>
        {/each}
      </select>
    </div>
    
    <div class="control-group">
      <label for="variant">Variant</label>
      <select id="variant" bind:value={variant} disabled={isRunning}>
        {#each variants as v}
          <option value={v.value}>{v.label}</option>
        {/each}
      </select>
    </div>
    
    <div class="control-group">
      <label for="startDate">Start Date</label>
      <input type="date" id="startDate" bind:value={startDate} disabled={isRunning} />
    </div>
    
    <div class="control-group">
      <label for="endDate">End Date</label>
      <input type="date" id="endDate" bind:value={endDate} disabled={isRunning} />
    </div>
  </div>
  
  <div class="actions">
    {#if isRunning}
      <button class="btn btn-stop" on:click={stopBacktest} disabled={!isRunning}>
        Stop
      </button>
    {:else}
      <button class="btn btn-run" on:click={runBacktest} disabled={!wsConnected}>
        Run Backtest
      </button>
    {/if}
    
    <button class="btn btn-clear" on:click={clearLogs} disabled={isRunning}>
      Clear Logs
    </button>
    
    {#if results}
      <button class="btn btn-download" on:click={downloadResults}>
        Download Results
      </button>
    {/if}
  </div>
  
  {#if isRunning || progress > 0}
    <div class="progress-section">
      <div class="progress-bar-container">
        <div class="progress-bar" style="width: {progress}%"></div>
      </div>
      <div class="progress-text">
        {progress.toFixed(1)}% - {status}
      </div>
    </div>
  {/if}
  
  {#if error}
    <div class="error-display">
      <strong>Error:</strong> {error}
    </div>
  {/if}
  
  <div class="logs-section">
    <h3>Logs</h3>
    <div class="logs-container" bind:this={logContainer}>
      {#if logs.length === 0}
        <div class="empty-logs">No logs yet</div>
      {:else}
        {#each logs as log}
          <div class="log-entry">
            <span class="log-timestamp">{new Date(log.timestamp).toLocaleTimeString()}</span>
            <span class="log-level" style="color: {getLogColor(log.level)}">{log.level}</span>
            <span class="log-message">{log.message}</span>
          </div>
        {/each}
      {/if}
    </div>
  </div>
  
  {#if results}
    <div class="results-section">
      <h3>Results</h3>
      <div class="results-grid">
        <div class="result-card">
          <div class="result-label">Final Balance</div>
          <div class="result-value" class:positive={Number(results.final_balance) > 10000} class:negative={Number(results.final_balance) < 10000}>
            {Number(results.final_balance)?.toFixed(2)}
          </div>
        </div>
        
        <div class="result-card">
          <div class="result-label">Total Trades</div>
          <div class="result-value">{results.total_trades}</div>
        </div>
        
        {#if results.win_rate !== undefined}
          <div class="result-card">
            <div class="result-label">Win Rate</div>
            <div class="result-value" class:positive={Number(results.win_rate) > 50} class:negative={Number(results.win_rate) < 50}>
              {Number(results.win_rate)?.toFixed(1)}%
            </div>
          </div>
        {/if}
        
        {#if results.sharpe_ratio !== undefined}
          <div class="result-card">
            <div class="result-label">Sharpe Ratio</div>
            <div class="result-value" class:positive={Number(results.sharpe_ratio) > 1} class:negative={Number(results.sharpe_ratio) < 1}>
              {Number(results.sharpe_ratio)?.toFixed(2)}
            </div>
          </div>
        {/if}
        
        {#if results.drawdown !== undefined}
          <div class="result-card">
            <div class="result-label">Max Drawdown</div>
            <div class="result-value negative">
              {Number(results.drawdown)?.toFixed(2)}%
            </div>
          </div>
        {/if}
        
        {#if results.return_pct !== undefined}
          <div class="result-card">
            <div class="result-label">Return</div>
            <div class="result-value" class:positive={Number(results.return_pct) > 0} class:negative={Number(results.return_pct) < 0}>
              {typeof results.return_pct === 'number' ? results.return_pct.toFixed(2) : 'N/A'}%
            </div>
          </div>
        {/if}
      </div>
    </div>
  {/if}
</div>

<style>
  .backtest-runner {
    padding: 1rem;
    background: #1e1e1e;
    border-radius: 8px;
    color: #e0e0e0;
  }
  
  .header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1rem;
  }
  
  .header h2 {
    margin: 0;
    font-size: 1.25rem;
    font-weight: 600;
  }
  
  .connection-status {
    font-size: 0.75rem;
    padding: 0.25rem 0.5rem;
    border-radius: 4px;
    background: #374151;
  }
  
  .connection-status.connected {
    background: #065f46;
    color: #10b981;
  }
  
  .controls {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 1rem;
    margin-bottom: 1rem;
  }
  
  .control-group {
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
  }
  
  .control-group label {
    font-size: 0.75rem;
    color: #9ca3af;
  }
  
  select, input {
    padding: 0.5rem;
    background: #2d2d2d;
    border: 1px solid #4b5563;
    border-radius: 4px;
    color: #e0e0e0;
    font-size: 0.875rem;
  }
  
  select:disabled, input:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
  
  .actions {
    display: flex;
    gap: 0.5rem;
    margin-bottom: 1rem;
  }
  
  .btn {
    padding: 0.5rem 1rem;
    border: none;
    border-radius: 4px;
    font-size: 0.875rem;
    cursor: pointer;
    transition: background 0.2s;
  }
  
  .btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
  
  .btn-run {
    background: #10b981;
    color: white;
  }
  
  .btn-run:hover:not(:disabled) {
    background: #059669;
  }
  
  .btn-stop {
    background: #ef4444;
    color: white;
  }
  
  .btn-stop:hover:not(:disabled) {
    background: #dc2626;
  }
  
  .btn-clear {
    background: #6b7280;
    color: white;
  }
  
  .btn-clear:hover:not(:disabled) {
    background: #4b5563;
  }
  
  .btn-download {
    background: #3b82f6;
    color: white;
  }
  
  .btn-download:hover:not(:disabled) {
    background: #2563eb;
  }
  
  .progress-section {
    margin-bottom: 1rem;
  }
  
  .progress-bar-container {
    height: 8px;
    background: #2d2d2d;
    border-radius: 4px;
    overflow: hidden;
    margin-bottom: 0.5rem;
  }
  
  .progress-bar {
    height: 100%;
    background: linear-gradient(90deg, #10b981, #3b82f6);
    transition: width 0.3s ease;
  }
  
  .progress-text {
    font-size: 0.875rem;
    color: #9ca3af;
    text-align: center;
  }
  
  .error-display {
    padding: 0.75rem;
    background: #7f1d1d;
    border-radius: 4px;
    margin-bottom: 1rem;
    color: #fecaca;
  }
  
  .logs-section {
    margin-bottom: 1rem;
  }
  
  .logs-section h3 {
    margin: 0 0 0.5rem 0;
    font-size: 1rem;
    font-weight: 500;
  }
  
  .logs-container {
    height: 200px;
    background: #2d2d2d;
    border-radius: 4px;
    padding: 0.5rem;
    overflow-y: auto;
    font-family: Monaco, Menlo, Ubuntu Mono, monospace;
    font-size: 0.75rem;
  }
  
  .empty-logs {
    color: #6b7280;
    text-align: center;
    padding: 2rem;
  }
  
  .log-entry {
    display: flex;
    gap: 0.5rem;
    margin-bottom: 0.25rem;
    padding: 0.25rem 0;
    border-bottom: 1px solid #374151;
  }
  
  .log-timestamp {
    color: #6b7280;
    white-space: nowrap;
  }
  
  .log-level {
    font-weight: 600;
    width: 60px;
    flex-shrink: 0;
  }
  
  .log-message {
    color: #d1d5db;
    word-break: break-word;
  }
  
  .results-section {
    margin-bottom: 1rem;
  }
  
  .results-section h3 {
    margin: 0 0 0.5rem 0;
    font-size: 1rem;
    font-weight: 500;
  }
  
  .results-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
    gap: 1rem;
  }
  
  .result-card {
    background: #2d2d2d;
    border-radius: 4px;
    padding: 0.75rem;
    text-align: center;
  }
  
  .result-label {
    font-size: 0.75rem;
    color: #9ca3af;
    margin-bottom: 0.25rem;
  }
  
  .result-value {
    font-size: 1.125rem;
    font-weight: 600;
  }
  
  .result-value.positive {
    color: #10b981;
  }
  
  .result-value.negative {
    color: #ef4444;
  }
</style>
