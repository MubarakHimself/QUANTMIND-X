# 09 — cTrader Boundary Plan

**Version:** 1.0
**Date:** 2026-04-08
**Purpose:** Define the minimum cTrader-aware boundary for QuantMindLib V1

---

## 1. Platform Context

### cTrader Environment
- **Platform:** cTrader (not MT5)
- **First broker:** IC Markets cTrader Raw (ECN-style, raw spreads)
- **API:** cTrader Open API (Python SDK available)
- **Algo:** cTrader Algo (Python and C#)
- **Node OS:** Trading node (T1 on Kamatera) runs Windows for cTrader terminal

### Official Documentation Links (from memo Appendix A)
- cTrader Algo overview: https://help.ctrader.com/ctrader-algo/
- cTrader Open API: https://help.ctrader.com/open-api/
- cTrader Python SDK: https://help.ctrader.com/open-api/python-SDK/python-sdk-index/
- cTrader MarketData: https://help.ctrader.com/ctrader-algo/references/MarketData/
- cTrader Ticks: https://help.ctrader.com/ctrader-algo/references/MarketData/Ticks/Ticks/
- cTrader MarketDepth: https://help.ctrader.com/ctrader-algo/id/references/MarketData/MarketDepth/MarketDepth/
- cTrader DOM: https://help.ctrader.com/trading-with-ctrader/depth-of-market/
- cTrader Network Access: https://help.ctrader.com/ctrader-algo/guides/network-access/
- cTrader Web (Linux): https://help.ctrader.com/ctrader-web/
- IC Markets cTrader Raw: https://www.icmarkets.com/global/en/trading-accounts/ctrader-raw

### Key cTrader API Concepts for Library Design
- **Ticks:** Real-time price updates, bid/ask/last
- **Bars:** OHLCV aggregation (M1/M5/M15/H1/H4/D1)
- **MarketDepth:** Order book depth (DOM), level 2
- **Network Access:** HTTP/WebSocket for external data feeds
- **Algo:** Python/C# algos that can subscribe to market data and submit orders

---

## 2. cTrader Adapter Architecture

### Adapter Layer Structure

```
src/library/adapters/ctrader/
├── client.py                  # CTraderClient — cTrader Open API Python SDK wrapper
├── market_adapter.py          # CTraderMarketAdapter (IMarketDataAdapter impl)
├── execution_adapter.py        # CTraderExecutionAdapter (IExecutionAdapter impl)
├── types.py                   # cTrader-specific type conversions
└── converter.py              # OHLCV, tick, depth converters
```

### Adapter Interface Contracts

All platform-specific code lives behind these interfaces:

```python
class IMarketDataAdapter(ABC):
    @async
    def tick_stream(self, symbols: List[str]) -> AsyncIterator[TickEvent]: ...
    @async
    def bar_stream(self, symbols: List[str], timeframe: str) -> AsyncIterator[BarEvent]: ...
    @async
    def depth_stream(self, symbol: str) -> AsyncIterator[DepthEvent]: ...
    def get_historical(self, symbol: str, timeframe: str, start: datetime, end: datetime) -> List[Bar]: ...
    def get_account_state(self) -> AccountState: ...

class IExecutionAdapter(ABC):
    def send_order(self, directive: ExecutionDirective) -> ExecutionResult: ...
    def cancel_order(self, order_id: str) -> bool: ...
    def get_positions(self, symbol: Optional[str]) -> List[Position]: ...
    def get_orders(self, symbol: Optional[str]) -> List[Order]: ...
```

### cTrader-Specific Implementations

**CTraderMarketAdapter:**
- Wraps cTrader Open API Python SDK
- Converts cTrader tick format → library `TickEvent`
- Converts cTrader bar format → library `BarEvent`
- Converts cTrader depth format → library `DepthEvent`
- Handles cTrader-specific reconnect logic
- Implements heartbeat/keepalive for WebSocket connections

**CTraderExecutionAdapter:**
- Converts library `ExecutionDirective` → cTrader order
- Maps cTrader order types (market, limit, stop)
- Translates cTrader execution reports → library `ExecutionResult`
- Handles cTrader-specific error codes and retry logic

---

## 3. What Stays Platform-Agnostic (Library Core)

These must NEVER reference cTrader directly:

| Layer | What is Platform-Neutral |
|-------|------------------------|
| **BotSpec** | All fields are platform-neutral strings |
| **MarketContext** | RegimeType enum, chaos_score, regime_quality — no cTrader types |
| **TradeIntent** | direction, confidence, size_requested — no broker types |
| **RiskEnvelope** | allocation_scalar, position_size, risk_mode — no broker types |
| **FeatureVector** | Dict[str, float] of computed features — no broker types |
| **FeatureEvaluator** | Evaluates feature stack from cached market context |
| **IntentEmitter** | Emits TradeIntent from BotSpec + FeatureVector |
| **All bridge adapters** | Translate to/from existing systems (sentinel, risk, registry) |

---

## 4. What is cTrader-Specific (Adapter Layer)

These are allowed to be cTrader-specific because they live in the adapter layer:

| Component | Why cTrader-Specific |
|-----------|---------------------|
| **CTraderClient** | Wraps cTrader Open API SDK directly |
| **cTrader type converters** | Translates cTrader wire format → library domain |
| **Account state mapping** | cTrader account fields → library AccountState |
| **Order type mapping** | library execution types → cTrader order types |
| **Position mapping** | cTrader position format → library Position |
| **Error code translation** | cTrader error codes → library errors |
| **Reconnection logic** | cTrader-specific retry/backoff |

---

## 5. Live Tick Path (cTrader → Library)

```
cTrader Terminal (live tick stream)
    │
    ▼
CTrader Open API WebSocket (tick events)
    │
    ▼
CTraderClient.tick_stream() (async iterator)
    │
    ▼
CTraderMarketAdapter.tick_stream() → TickEvent normalization
    │
    ├──► Feature workers (async)
    │         │
    │         ▼
    │    FeatureEvaluator (sync read cached state)
    │
    ├──► SentinelBridge (async)
    │         │
    │         ▼
    │    RegimeReport → MarketContext (cached)
    │
    └──► BotStateManager (async write, sync read)
              │
              ▼
         Library core (FeatureVector, TradeIntent, RiskEnvelope)
```

**Key insight:** The tick path from cTrader to library is fully async. Library core does sync reads from cached state.

---

## 6. Historical Data Path

```
Library request: get_historical(symbol, timeframe, start, end)
    │
    ▼
CTraderMarketAdapter.get_historical()
    │
    ├─► CTraderClient.get_bars() (cTrader Open API)
    │         │
    │         ▼
    │    cTrader bar format [timestamp, open, high, low, close, volume]
    │
    ▼
Converter.bars_to_library(bars) → List[Bar]
    │
    ▼
Return List[Bar] to caller (backtest, data manager)
```

---

## 7. Execution Path (Library → cTrader)

```
TradeIntent (from IntentEmitter)
    │
    ▼
RiskBridge → Governor → RiskEnvelope (sync)
    │
    ▼
ExecutionDirective (approved/rejected with size)
    │
    ▼
CTraderExecutionAdapter.send_order(directive)
    │
    ▼
Convert to cTrader order format
    │
    ├─► Market order: direction, symbol, volume, stop, take-profit
    ├─► Limit order: direction, symbol, volume, price, stop, take-profit
    └─► Stop order: direction, symbol, volume, trigger_price, stop, take-profit
    │
    ▼
CTraderClient.send_order() → cTrader order submission
    │
    ▼
cTrader execution report
    │
    ▼
CTraderExecutionAdapter.convert_report() → ExecutionResult
    │
    ▼
Library ExecutionResult → DPRBridge (for scoring) + JournalBridge (for audit)
```

---

## 8. Order Flow Data Availability

Per trading_platform_decision_memo §5: Order flow quality-aware design.

### cTrader → Order Flow Data Available

| Data Source | Available | Quality | Notes |
|-------------|-----------|---------|-------|
| **Tick stream** | Yes | HIGH | Real-time bid/ask/last ticks |
| **OHLCV bars** | Yes | HIGH | All standard timeframes |
| **Market Depth (DOM)** | Yes | HIGH | Level 2 order book via cTrader Open API |
| **Spread behavior** | Yes | HIGH | Derived from tick stream |
| **Volume** | Yes | MEDIUM | cTrader volume is relative (not absolute) |
| **Delta (buy/sell volume)** | PARTIAL | LOW | cTrader doesn't expose raw buy/sell volume directly |
| **Imbalance at levels** | PARTIAL | MEDIUM | Can be derived from DOM |

### Quality Tagging Strategy

All order flow features must include `FeatureConfidence`:
```python
@dataclass
class FeatureConfidence:
    source: str       # "ctrader_native" / "ctrader_proxied" / "approximated"
    quality: float    # 0.0-1.0
    latency_ms: float
    feed_quality_tag: str  # HIGH / MEDIUM / LOW
```

Order flow features should degrade gracefully:
- HIGH: cTrader DOM tick-level data
- MEDIUM: OHLCV-derived approximations
- LOW: Session-relative proxies (e.g., current volume vs session average)

---

## 9. MT5 Migration Surface

These areas currently use MT5 and must be migrated to cTrader adapters:

| Current MT5 Usage | Migration Target | Risk |
|------------------|-------------------|------|
| `src/data/brokers/mt5_socket_adapter.py` | `adapters/ctrader/market_adapter.py` | HIGH — live data feed |
| `src/risk/integrations/mt5/` | `adapters/ctrader/execution_adapter.py` | HIGH — execution |
| `src/backtesting/mt5_engine.py` | `adapters/ctrader/backtest_engine.py` | HIGH — evaluation pipeline |
| `src/mql5/` (EA code generation) | `adapters/ctrader/ea_generator.py` | HIGH — code generation |
| `src/risk/pipeline/layer3_kill_switch.py` | `adapters/ctrader/kill_switch_adapter.py` | CRITICAL — forced exits |

### Migration Priority
1. **Market data adapter** — must work before anything else
2. **Backtest engine** — must produce same schemas as MT5BacktestResult
3. **Execution adapter** — must support all order types
4. **Kill switch adapter** — must integrate with cTrader order management
5. **EA code generator** — Phase 2 (cTrader Algo Python instead of MQL5)

### Backtest Engine Schema Compatibility

Critical: `CTraderBacktestResult` must have the same schema as `MT5BacktestResult`:
```python
# Both must produce this structure:
@dataclass
class BacktestResult:
    sharpe: float
    return_pct: float
    drawdown: float
    trades: int
    initial_cash: float
    final_cash: float
    equity_curve: List[float]
    trade_history: List[Dict[str, Any]]
    win_rate: float
    profit_factor: float
    kelly_score: float
```

This ensures the evaluation pipeline (`FullBacktestPipeline`) doesn't need to change when switching from MT5 to cTrader backtest engine.

---

## 10. cTrader-specific Configuration

All cTrader-specific configuration must be in adapter config, not in library core:

```python
@dataclass
class CTraderAdapterConfig:
    api_url: str                             # cTrader Open API URL
    broker_id: str                           # IC Markets
    account_id: str                         # Account number
    symbols: List[str]                      # Symbols to subscribe
    timeframes: List[str]                   # M1, M5, M15, H1, etc.
    network_access_enabled: bool            # cTrader Network Access feature
    # IC Markets Raw specific
    raw_spread_account: bool = True         # Raw spread account flag
    commission_per_lot: float = 7.0         # IC Markets Raw commission
    pip_value: Dict[str, float]             # Symbol-specific pip values
    min_lot: Dict[str, float]               # Symbol-specific minimum lot
    lot_step: Dict[str, float]               # Symbol-specific lot step
```

---

## 11. Network Access Feature

cTrader Algo supports Network Access (HTTP/WebSocket for external data). This enables:
- Economic calendar integration (fetch from external API)
- News feed integration (Finnhub via cTrader Network Access)
- Custom data sources

This is relevant because it changes how market intelligence data reaches the system: currently Finnhub polling is direct; with cTrader, it could go through cTrader Network Access.

Recommendation: Keep Finnhub integration as-is (direct API) for V1. Network Access integration is Phase 2.

---

## 12. cTrader Web on Linux

cTrader Web enables browser-based cTrader access on Linux. This is relevant because:
- The backend node (Linux) could use cTrader Web for data access
- No Windows cTrader installation needed for back-end data fetching

For V1: cTrader Open API is the primary interface. cTrader Web is a fallback/future path.

---

## 13. Sync/Async Design for cTrader

| Operation | Mode | Reason |
|-----------|------|--------|
| tick_stream | Async | WebSocket stream, event-driven |
| bar_stream | Async | WebSocket stream or polling |
| depth_stream | Async | WebSocket stream |
| get_historical | Sync | One-shot request/response |
| send_order | Sync | Order submission must be confirmed |
| get_positions | Sync | One-shot snapshot |
| cancel_order | Sync | Order cancellation confirmation |

---

## 14. Error Handling for cTrader

| Error Type | Handling |
|------------|----------|
| Connection lost | Reconnect with exponential backoff, max 5 retries |
| Symbol not found | Log error, skip symbol, continue with others |
| Order rejected | Return ExecutionResult with rejection, log reason |
| Rate limit | Back off, queue requests, retry with delay |
| Account disconnected | Emergency: trigger KillSwitch, log critical |

---

## 15. Platform Isolation Checklist

These are the items that must be verified before claiming "platform-agnostic core":

- [ ] No direct imports of cTrader SDK types in library core
- [ ] No direct imports of MT5 types in library core
- [ ] All broker-specific types converted at adapter boundary
- [ ] FeatureEvaluator reads only from library domain objects
- [ ] IntentEmitter produces only library domain objects
- [ ] RiskBridge translates only between library and existing systems
- [ ] No broker-specific enums (RegimeType, TradeDirection are library enums)
- [ ] All adapter implementations are in `src/library/adapters/`