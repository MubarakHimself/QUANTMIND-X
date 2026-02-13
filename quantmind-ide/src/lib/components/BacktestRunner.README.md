# BacktestRunner Component

A Svelte component for running backtests with real-time WebSocket streaming of progress, logs, and results.

## Usage

### Basic Usage

```svelte
<script>
  import BacktestRunner from './lib/components/BacktestRunner.svelte';
</script>

<BacktestRunner 
  baseUrl="http://localhost:8000" 
  strategyCode="def on_bar(tester): pass"
  strategyName="MyStrategy"
/>
```

### With Event Handlers

```svelte
<script>
  import BacktestRunner from './lib/components/BacktestRunner.svelte';
  
  function handleComplete(event) {
    console.log('Backtest complete:', event.detail);
  }
  
  function handleError(event) {
    console.error('Backtest error:', event.detail);
  }
</script>

<BacktestRunner 
  on:complete={handleComplete}
  on:error={handleError}
/>
```

## Props

| Prop | Type | Default | Description |
|------|------|---------|-------------|
| `baseUrl` | string | `'http://localhost:8000'` | Backend API base URL |
| `strategyCode` | string | `''` | Python strategy code with `on_bar(tester)` function |
| `strategyName` | string | `'MyStrategy'` | Strategy name for logging |

## Events

| Event | Detail | Description |
|-------|--------|-------------|
| `complete` | `BacktestResult` | Fired when backtest completes successfully |
| `error` | `{ error: string }` | Fired when backtest encounters an error |

### BacktestResult Object

```typescript
interface BacktestResult {
  backtest_id: string;
  final_balance: number;
  total_trades: number;
  win_rate?: number;
  sharpe_ratio?: number;
  drawdown?: number;
  return_pct?: number;
  duration_seconds?: number;
  results?: Record<string, unknown>;
}
```

## Features

### Backtest Configuration

- **Symbol Selection**: Choose from predefined symbols (EURUSD, GBPUSD, USDJPY, XAUUSD, BTCUSD)
- **Timeframe**: Select chart timeframe (M1, M5, M15, M30, H1, H4, D1)
- **Variant**: Choose backtest mode:
  - `vanilla`: Historical backtest with static parameters
  - `spiced`: Vanilla + regime filtering
  - `vanilla_full`: Vanilla + Walk-Forward optimization
  - `spiced_full`: Spiced + Walk-Forward optimization
- **Date Range**: Set start and end dates for backtest period

### Real-Time Updates

- **Progress Bar**: Visual progress indicator with percentage
- **Status Messages**: Current backtest status
- **Live Logs**: Real-time log streaming with timestamps

### Log Management

- **Log Level Filtering**: Filter logs by level (ALL, INFO, WARNING, ERROR, PROGRESS, COMPLETE)
- **Auto-Scroll**: Automatically scroll to newest logs (toggleable)
- **Export Logs**: Download logs as `.log` file

### Results Display

- Final Balance
- Total Trades
- Win Rate
- Sharpe Ratio
- Maximum Drawdown
- Return Percentage

## WebSocket Integration

The component automatically connects to the WebSocket endpoint and subscribes to the `backtest` topic on mount.

### Connection Status

The header displays connection status:
- **Connected** (green): WebSocket connected and ready
- **Disconnected** (gray): WebSocket not connected

### Event Flow

1. User clicks "Run Backtest"
2. POST request to `/api/v1/backtest/run`
3. WebSocket receives `backtest_start` event
4. Progress updates via `backtest_progress` events
5. Log entries via `log_entry` events
6. Final results via `backtest_complete` or `backtest_error`

## Styling

The component uses a dark theme with the following color scheme:

- Background: `#1e1e1e`
- Cards: `#2d2d2d`
- Text: `#e0e0e0`
- Success: `#10b981`
- Error: `#ef4444`
- Warning: `#f59e0b`
- Info: `#3b82f6`

### CSS Customization

Override styles by targeting the component's CSS classes:

```css
.backtest-runner {
  /* Override background */
  background: #your-color;
}

.btn-run {
  /* Override run button */
  background: #your-color;
}
```

## Dependencies

- `svelte` - Component framework
- `../ws-client` - WebSocket client module

## Example: Full Integration

```svelte
<script>
  import BacktestRunner from './lib/components/BacktestRunner.svelte';
  
  const strategyCode = `
def on_bar(tester):
    """Simple moving average crossover strategy."""
    symbol = tester.symbol
    if not symbol:
        return
    
    close = tester.iClose(symbol, 0, 0)
    if close is None:
        return
    
    ma_fast = tester.iMA(symbol, 0, 0, 20, 0)
    ma_slow = tester.iMA(symbol, 0, 0, 50, 0)
    
    if ma_fast is None or ma_slow is None:
        return
    
    if ma_fast > ma_slow:
        tester.buy(symbol, 0.1)
    elif ma_fast < ma_slow:
        tester.sell(symbol, 0.1)
  `;
  
  let results = null;
  
  function handleComplete(event) {
    results = event.detail;
    console.log('Backtest completed:', results);
  }
  
  function handleError(event) {
    console.error('Backtest failed:', event.detail.error);
  }
</script>

<div class="app">
  <h1>QuantMindX Backtest</h1>
  
  <BacktestRunner 
    baseUrl="http://localhost:8000"
    strategyCode={strategyCode}
    strategyName="MA_Crossover"
    on:complete={handleComplete}
    on:error={handleError}
  />
  
  {#if results}
    <div class="results-summary">
      <h2>Results Summary</h2>
      <p>Return: {results.return_pct?.toFixed(2)}%</p>
      <p>Sharpe: {results.sharpe_ratio?.toFixed(2)}</p>
    </div>
  {/if}
</div>
```

## Troubleshooting

### WebSocket Not Connecting

1. Ensure backend server is running
2. Check `baseUrl` prop is correct
3. Verify WebSocket endpoint is available at `/ws`

### No Progress Updates

1. Confirm WebSocket is connected (green status)
2. Check browser console for errors
3. Verify backend is broadcasting events

### Logs Not Appearing

1. Check log level filter is set correctly
2. Verify `log_entry` events are being received
3. Check for JavaScript errors in console

### Export Not Working

1. Ensure browser allows file downloads
2. Check logs array is not empty
3. Verify popup blocker is not blocking

## Testing

Unit tests are available in `BacktestRunner.test.ts`:

```bash
npm run test -- BacktestRunner
```

## See Also

- [WebSocket API Documentation](../../docs/api/websocket-endpoints.md)
- [WebSocket Architecture](../../docs/architecture/websocket-streaming.md)
- [ws-client Module](../ws-client.ts)
