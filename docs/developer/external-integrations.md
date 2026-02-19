# External Integrations Architecture

This document describes the external integrations added to QuantMind, including TradingView webhooks, GitHub EA sync, and LEAN-style backtesting models.

## Table of Contents

1. [TradingView Integration](#tradingview-integration)
2. [GitHub EA Sync](#github-ea-sync)
3. [LEAN Backtesting Models](#lean-backtesting-models)
4. [API Reference](#api-reference)
5. [Configuration](#configuration)

---

## TradingView Integration

### Webhook Alerts

The TradingView webhook endpoint receives alerts from TradingView and triggers bot execution.

**Endpoint:** `POST /api/tradingview/webhook`

**Request Body:**
```json
{
  "symbol": "EURUSD",
  "action": "buy",
  "price": 1.0850,
  "strategy": "ICT Silver Bullet",
  "timeframe": "M15",
  "timestamp": "2024-01-15T10:30:00Z",
  "signature": "sha256_hash"
}
```

**Security Features:**
- HMAC-SHA256 signature validation
- Timestamp validation (60-second window)
- IP whitelist for TradingView servers
- Rate limiting (100 requests/minute per IP)

### Lightweight Charts

Professional charting component using TradingView's Lightweight Charts library.

**Features:**
- Candlestick and OHLCV display
- Volume histogram overlay
- Trade markers (entry/exit points)
- Regime overlays (TRENDING/RANGING/VOLATILE)
- Multiple timeframes (M1 to D1)
- Real-time updates via WebSocket

**Usage in Svelte:**
```svelte
<script>
  import TradingViewChart from '$lib/components/TradingViewChart.svelte';
</script>

<TradingViewChart 
  symbol="EURUSD"
  data={ohlcData}
  trades={tradeMarkers}
  regimes={regimeData}
/>
```

### Pine Script Agent

LangGraph agent for generating Pine Script strategies from natural language.

**Endpoint:** `POST /api/chat/pinescript`

**Request:**
```json
{
  "query": "Create an ICT Silver Bullet strategy with 15-minute timeframe"
}
```

**Response:**
```json
{
  "pine_script": "//@version=5\nstrategy('ICT Silver Bullet', ...)",
  "status": "success",
  "errors": []
}
```

---

## GitHub EA Sync

### Overview

Automatically synchronizes Expert Advisors from a GitHub repository to QuantMind's bot management system.

### Configuration

Set the following environment variables:

```bash
GITHUB_EA_REPO_URL=https://github.com/your-org/quantmind-eas
GITHUB_EA_BRANCH=main
GITHUB_ACCESS_TOKEN=ghp_xxxx  # Optional for private repos
EA_LIBRARY_PATH=/data/ea-library
GITHUB_EA_SYNC_INTERVAL_HOURS=24
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/github/status` | GET | Get sync status and next scheduled run |
| `/api/github/sync` | POST | Trigger manual sync |
| `/api/github/eas` | GET | List all available EAs |
| `/api/github/eas/{id}` | GET | Get EA details |
| `/api/github/import` | POST | Import selected EAs |
| `/api/github/imported` | GET | List imported EAs |

### EA Parsing

The sync service automatically extracts from `.mq5` files:
- Input parameters (type, name, default value)
- Strategy type from comments or filename
- Timeframe from `PERIOD_*` constants
- Symbol compatibility
- Version and checksum for change detection

### Scheduled Sync

Sync runs automatically every 24 hours (configurable). Use the scheduler module:

```python
from src.integrations.github_ea_scheduler import start_scheduler

# Start the scheduler
start_scheduler()
```

---

## LEAN Backtesting Models

Inspired by QuantConnect's LEAN engine, providing realistic backtesting with slippage and commission models.

### Slippage Models

Located in `src/backtesting/lean_slippage.py`

| Model | Description | Formula |
|-------|-------------|---------|
| `ConstantSlippage` | Fixed slippage per trade | `slippage_pips × pip_value` |
| `VolumeBasedSlippage` | Scales with order volume | `base × √(order_vol / avg_vol)` |
| `VolatilityBasedSlippage` | Scales with ATR | `base × (current_atr / avg_atr)` |
| `MarketImpactSlippage` | Almgren-Chriss model | `k × (vol / daily_vol)^0.6 × price` |

**Usage:**
```python
from src.backtesting.lean_slippage import (
    SlippageModelFactory,
    ConstantSlippage,
    VolumeBasedSlippage
)

# Create via factory
slippage = SlippageModelFactory.create('volume_based', base_slippage_pips=0.5)

# Apply to price
adjusted_price = slippage.apply_slippage(
    price=1.0850,
    side='buy',
    order_volume=1.0,
    avg_volume=10.0
)
```

### Commission Models

Located in `src/backtesting/lean_commission.py`

| Model | Description | Use Case |
|-------|-------------|----------|
| `PerTradeCommission` | Fixed fee per trade | Flat-fee brokers |
| `PerLotCommission` | Fee per lot traded | Forex brokers |
| `PerShareCommission` | Fee per share | Equity trading |
| `TieredCommission` | Volume-based tiers | Professional forex |
| `SpreadBasedCommission` | Spread cost conversion | Market makers |
| `MakerTakerCommission` | Different rates | Crypto exchanges |

**Usage:**
```python
from src.backtesting.lean_commission import (
    CommissionModelFactory,
    PerLotCommission,
    TieredCommission
)

# Create via factory
commission = CommissionModelFactory.create('per_lot', cost_per_lot=7.0)

# Calculate commission
fee = commission.calculate_commission(
    lots=1.0,
    symbol='EURUSD',
    price=1.0850,
    side='buy'
)
```

### Multi-Asset Framework

Located in `src/backtesting/multi_asset_engine.py`

Supports multiple asset classes with specific handling:

| Asset Class | Handler | Features |
|-------------|---------|----------|
| Forex | `ForexHandler` | 24/5, pip-based, leverage |
| Stocks | `StockHandler` | Market hours, per-share, dividends |
| Crypto | `CryptoHandler` | 24/7, maker/taker, funding rates |
| Futures | `FuturesHandler` | Contract specs, margins, expiration |

**Usage:**
```python
from src.backtesting.multi_asset_engine import (
    AssetHandlerFactory,
    AssetClass
)

# Get handler for symbol
handler = AssetHandlerFactory.get_handler_for_symbol('EURUSD')

# Calculate asset-specific values
pip_value = handler.calculate_pip_value('EURUSD', 1.0850, 1.0)
margin = handler.calculate_margin('EURUSD', 1.0850, 1.0)
is_open = handler.is_market_open(datetime.now())
```

---

## API Reference

### TradingView Endpoints

```
POST /api/tradingview/webhook
  - Receives TradingView alerts
  - Validates signature and timestamp
  - Triggers bot execution

GET /api/tradingview/charts/{symbol}
  - Get chart data for symbol
  - Supports timeframe parameter
```

### GitHub Endpoints

```
GET /api/github/status
  - Returns scheduler status
  - Last sync time, next scheduled run

POST /api/github/sync
  - Trigger manual repository sync
  - Returns sync results

GET /api/github/eas
  - List all discovered EAs
  - Filter by status or strategy type

POST /api/github/import
  - Import selected EAs
  - Generate BotManifests
```

---

## Configuration

### Environment Variables

Add to `.env`:

```bash
# TradingView
TRADINGVIEW_WEBHOOK_SECRET=your-secret-key
TRADINGVIEW_IP_WHITELIST=52.89.214.238,34.212.75.30

# GitHub EA Sync
GITHUB_EA_REPO_URL=https://github.com/org/repo
GITHUB_EA_BRANCH=main
GITHUB_ACCESS_TOKEN=ghp_xxx
EA_LIBRARY_PATH=/data/ea-library
GITHUB_EA_SYNC_INTERVAL_HOURS=24

# LEAN Models
DEFAULT_SLIPPAGE_MODEL=volume_based
DEFAULT_COMMISSION_MODEL=per_lot
DEFAULT_COMMISSION_PER_LOT=7.0
```

### Database Tables

New tables added:

1. **webhook_logs** - TradingView webhook audit trail
2. **imported_eas** - GitHub EA tracking

---

## Troubleshooting

### Webhook Not Working

1. Check webhook secret matches TradingView configuration
2. Verify IP is whitelisted
3. Check timestamp is within 60 seconds
4. Review logs in `webhook_logs` table

### GitHub Sync Failing

1. Verify repository URL is accessible
2. Check access token has read permissions
3. Ensure `EA_LIBRARY_PATH` directory exists
4. Review sync errors via `/api/github/status`

### Chart Not Loading

1. Verify `lightweight-charts` package is installed
2. Check WebSocket connection is established
3. Ensure data format matches expected schema
4. Check browser console for errors

---

## Architecture Diagram

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  TradingView    │────▶│  Webhook API     │────▶│  Commander      │
│  Alerts         │     │  /api/tradingview│     │  (Bot Trigger)  │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                                        │
┌─────────────────┐     ┌──────────────────┐           │
│  GitHub Repo    │────▶│  EA Sync Service │           ▼
│  (.mq5 files)   │     │  (Scheduled)     │     ┌─────────────────┐
└─────────────────┘     └──────────────────┘     │  MT5 Terminal   │
                              │                   │  (Execution)    │
                              ▼                   └─────────────────┘
                        ┌──────────────────┐
                        │  BotManifest     │
                        │  Generator       │
                        └──────────────────┘
                              │
┌─────────────────┐           │
│  Backtest Engine│◀──────────┘
│  + Slippage     │
│  + Commission   │
│  + Multi-Asset  │
└─────────────────┘