# Real-Time Backtest Monitoring

## Overview
Watch your backtests run in real-time with live progress updates, logs, and results.

## Getting Started

1. Open the Backtest Runner in the IDE
2. Configure backtest parameters (symbol, timeframe, variant)
3. Click \"Run Backtest\"
4. Watch real-time updates:
   - Progress bar shows completion percentage
   - Logs display regime transitions and trade decisions
   - Results appear automatically on completion

## Features

### Live Progress Tracking
- Real-time progress bar (0-100%)
- Current bar being processed
- Number of trades executed
- Current P&L

### Real-Time Logs
- Regime transitions
- Trade executions
- Filter decisions
- Error messages

### Auto-Scroll
Logs automatically scroll to show latest entries

### Results Display
- Final balance
- Total trades
- Win rate
- Sharpe ratio
- Max drawdown
- Return percentage

## Troubleshooting

### Connection Issues
If \"Disconnected\" appears:
1. Check server is running on port 8000
2. Verify WebSocket endpoint at ws://localhost:8000/ws
3. Check browser console for errors

### Missing Updates
If progress stops updating:
1. Check server logs for errors
2. Verify backtest is still running
3. Try refreshing the page
