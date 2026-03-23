# QUANTMINDX — Strategy Router (The Sentient Loop)

**Generated:** 2026-03-11

---

## Overview

The Strategy Router is the real-time trading orchestration engine. It continuously ingests market ticks, classifies market regime, authorizes risk, selects strategies, and dispatches trade orders. It runs as part of the main FastAPI server process.

```
Tick Data (ZMQ / MT5 Bridge)
        ↓
    [Sentinel]          — regime classification
        ↓
   RegimeReport
        ↓
    [Governor]          — risk scalar authorization
        ↓
   RiskMandate
        ↓
   [Commander]          — strategy auction
        ↓
   Selected Bots
        ↓
   [RoutingMatrix]      — bot-to-account assignment
        ↓
   MT5 Bridge API       — order execution
        ↓
   TradeLogger          — trade context logging
```

---

## Core Components

### 1. StrategyRouter Engine (`src/router/engine.py`)

The top-level coordinator. Integrates all router components.

**Key integrations:**
- `Sentinel` — market regime classifier
- `Governor` / `EnhancedGovernor` — risk compliance
- `Commander` — bot selection
- `MultiTimeframeSentinel` — multi-timeframe analysis
- `ProgressiveKillSwitch` — tiered circuit breaker
- `RegimeFetcher` — polls Contabo HMM API for regime data every 5 minutes

**RegimeFetcher fallback chain:**
1. Contabo API cache (if reachable within 15 minutes)
2. Local HMM model via `HMMVersionControl`
3. `ISING_ONLY` mode as last resort

---

### 2. Sentinel (`src/router/sentinel.py`)

Market intelligence engine. Aggregates sensor data into `RegimeReport`.

**Sensors:**
| Sensor | File | What it detects |
|--------|------|----------------|
| ChaosSensor | `sensors/chaos.py` | Volatility / chaos score |
| RegimeSensor | `sensors/regime.py` | Trend/range/breakout classification |
| CorrelationSensor | `sensors/correlation.py` | Cross-asset correlation |
| NewsSensor | `sensors/news.py` | High-impact news KILL_ZONE detection |
| HMMRegimeSensor | `src/risk/physics/hmm_sensor.py` | HMM-based regime (optional) |

**Regimes:**
```
TREND_STABLE    → Directional trend with low chaos
RANGE_STABLE    → Sideways market
BREAKOUT_PRIME  → Imminent breakout
HIGH_CHAOS      → Chaotic, no trading authorized
```

---

### 3. Governor (`src/router/governor.py`)

Compliance layer. Maps regime to a risk scalar.

**See [Risk Management](./risk-management.md) for full details.**

### 4. Commander (`src/router/commander.py`)

Bot selection via **strategy auction**.

**Routing modes:**
| Mode | Description |
|------|-------------|
| `auction` | All @primal-tagged bots score against regime; highest scores selected (default) |
| `priority` | Bots ordered by priority, first match wins |
| `round_robin` | Round-robin through eligible bots |

**Regime → Strategy Type mapping:**
```python
TREND_STABLE    → ["SCALPER", "STRUCTURAL", "HFT"]
RANGE_STABLE    → ["SCALPER", "STRUCTURAL"]
BREAKOUT_PRIME  → ["STRUCTURAL", "SWING"]
HIGH_CHAOS      → []  # No bots authorized
```

**@primal tag:** Only bots with the `@primal` manifest tag participate in the auction. This prevents experimental bots from trading live.

**Global limit:** Maximum 50 active bots enforced.

---

### 5. BotManifest & BotRegistry (`src/router/bot_manifest.py`)

**BotManifest** defines a bot's trading profile:
```python
class BotManifest:
    bot_id: str
    strategy_type: StrategyType  # SCALPER | STRUCTURAL | SWING | HFT
    symbols: List[str]
    timeframes: List[str]
    tags: List[str]              # @primal, @paper_only, etc.
    max_lot_size: float
    max_spread: float
```

**BotRegistry** is an in-memory registry + SQLite-backed persistence of all bot manifests.

---

### 6. RoutingMatrix (`src/router/routing_matrix.py`)

Assigns bots to broker accounts based on:
- Bot's symbol list vs. account's allowed symbols
- Account tier (prop firm evaluation phase vs. funded)
- Account risk limits

```python
@dataclass
class RoutingDecision:
    bot_id: str
    account_id: str
    broker_id: str
    reason: str
```

---

### 7. BrokerRegistry (`src/router/broker_registry.py`)

Manages broker profiles with fee structures:
```python
class BrokerRegistryManager:
    # broker_id → BrokerRegistry (SQLAlchemy model)
    # pip_values per symbol (default: EURUSD=10.0, XAUUSD=1.0, etc.)
    # commission_per_lot, spread_avg, lot_step, min_lot, max_lot
```

**Script to seed:** `scripts/populate_broker_registry.py`

---

### 8. Kill Switches (`src/router/kill_switch.py`, `progressive_kill_switch.py`)

**SmartKillSwitch** — regime-aware:
- `soft_kill()` → stops new trades, lets positions close naturally
- `hard_kill()` → closes all open positions immediately

**ProgressiveKillSwitch** — loss-based tiered:
- Tier 1 (soft): Reduce position sizing
- Tier 2 (medium): Halt new entries
- Tier 3 (hard): Full stop + close all

---

### 9. MultiTimeframeSentinel (`src/router/multi_timeframe_sentinel.py`)

Runs Sentinel analysis across multiple timeframes simultaneously:
- `Timeframe.M1`, `M5`, `M15`, `H1`, `H4`, `D1`
- Aggregates regime signals across timeframes for confirmation

---

### 10. HMM Version Control (`src/router/hmm_version_control.py`)

Manages HMM model versioning for staged production deployment:
- `shadow` version — runs in parallel, not used for trading
- `cloudzy` version — deployed on Contabo VPS for live regime signals
- Version promotion: shadow → production

---

### 11. Lifecycle Manager (`src/router/lifecycle_manager.py`)

Manages bot lifecycle events:
- Bot promotion (paper → live trading)
- Bot demotion (underperforming live → paper)
- Scheduled lifecycle checks via cron (`scripts/schedule_lifecycle_check.py`)

**Promotion criteria:** Configurable win rate, profit factor, Sharpe ratio thresholds over minimum trade count.

---

### 12. Market Scanner (`src/router/market_scanner.py` / `scanners/`)

Background scanner that monitors:
- `market_scanner.py` — general market condition monitoring
- `scanners/symbol_scanner.py` — per-symbol opportunity detection
- `scanners/trend_scanner.py` — trend continuation signals
- Results feed into `market_opportunities` table

---

### 13. Supporting Utilities

| File | Purpose |
|------|---------|
| `bot_circuit_breaker.py` | Per-bot consecutive loss quarantine |
| `bot_cloner.py` | Clone a working bot to a new symbol |
| `dynamic_bot_limits.py` | Adjust per-account bot count dynamically |
| `fee_monitor.py` | Daily fee burn tracking |
| `file_watcher.py` | Hot-reload strategy configs on file change |
| `github_sync.py` | Sync EA files from/to GitHub |
| `house_money.py` | House money position scaling (trade profits only) |
| `hmm_deployment.py` | HMM deployment mode management |
| `promotion_manager.py` | Strategy promotion pipeline |
| `session_detector.py` | Forex session detection (London/NY/Tokyo/Sydney) |
| `trade_logger.py` | Enhanced trade context logging |
| `trd_watcher.py` | Watch `data/trd/` directory for new strategy files |
| `video_to_ea_workflow.py` | Video → EA conversion workflow |
| `virtual_balance.py` | Virtual balance tracking for paper trading |
| `workflow_orchestrator.py` | EA creation workflow orchestration |

---

## Data Flow: Trade Execution

1. **Market tick arrives** via ZMQ socket or MT5 bridge polling
2. **Sentinel** processes tick → updates ChaosSensor, RegimeSensor, CorrelationSensor, NewsSensor → emits `RegimeReport`
3. **Governor** receives `RegimeReport` + active trade proposal → calculates `RiskMandate`
4. **Commander** receives `RegimeReport` + `RiskMandate` → runs strategy auction → selects authorized bots
5. **RoutingMatrix** assigns each selected bot to a broker account
6. **MT5 Bridge** (`mt5-bridge/server.py`) receives HTTP trade requests → executes via MetaTrader 5 SDK
7. **TradeLogger** logs full context (regime, chaos score, Kelly fraction, bot_id, account_id)

---

## API Endpoints (Router-Related)

| Endpoint | File | Purpose |
|----------|------|---------|
| `GET /api/router/status` | `router_endpoints.py` | Router engine status |
| `POST /api/router/tick` | `router_endpoints.py` | Submit tick data |
| `GET /api/broker/*` | `broker_endpoints.py` | Broker management |
| `GET/POST /api/trading/*` | `trading_endpoints.py` | Trading operations |
| `GET /api/trading-floor/*` | `trading_floor_endpoints.py` | Trading floor control |
| `GET /api/floor-manager/*` | `floor_manager_endpoints.py` | AI floor manager |
| `POST /api/kill-switch/*` | `kill_switch_endpoints.py` | Kill switch control |
| `GET /api/lifecycle-scanner/*` | `lifecycle_scanner_endpoints.py` | Bot lifecycle |
| `GET /api/hmm/*` | `hmm_endpoints.py` | HMM model management |
| `WS /ws/broker/{broker_id}` | `broker_endpoints.py` | Broker WebSocket stream |
| `WS /ws/monte-carlo` | `monte_carlo_ws.py` | Monte Carlo WebSocket |
