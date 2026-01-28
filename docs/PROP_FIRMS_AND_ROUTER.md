# QuantMindX: Prop Firms, Strategy Router & Risk Architecture

> **Purpose:** Complete specification of prop firm compatibility, Strategy Router/Pilot Model, and risk management architecture  
> **Last Updated:** 2026-01-26

---

## Table of Contents

1. [Prop Firm Strategy](#1-prop-firm-strategy)
2. [Strategy Router / Pilot Model](#2-strategy-router--pilot-model)
3. [Risk Management Architecture](#3-risk-management-architecture)
4. [Multi-Strategy Portfolio Design](#4-multi-strategy-portfolio-design)
5. [HFT-Friendly Brokers](#5-hft-friendly-brokers)
6. [Deployment Path to Live Trading](#6-deployment-path-to-live-trading)
7. [MT5 MCP Server Configuration](#7-mt5-mcp-server-configuration)
8. [Modular Trading Architecture](#8-modular-trading-architecture-hummingbot-inspired)

---

## 1. Prop Firm Strategy

### Why Prop Firms?

From the Kimi conversation, the user's goal is to use **funded accounts** from prop firms rather than personal capital:

- **Lower personal risk** - Prop firm capital, not yours
- **Scaling opportunity** - Pass challenge → Get larger account
- **Discipline requirement** - Forces adhering to rules (daily loss limits, max drawdown)

### Prop Firm Compatibility Requirements

Most prop firms require:

| Requirement | QuantMindX Approach |
|-------------|---------------------|
| **Max Daily Loss: 5%** | Risk Governor enforces 2% per strategy, 5% total |
| **Max Total Drawdown: 10-12%** | Circuit breaker at 8% (buffer before limit) |
| **No Martingale** | Static position sizing only |
| **Consistent Trading** | Strategy Router ensures activity |
| **MT4 or MT5** | MT5 primary (better features) |

### Challenge Phase vs Funded Phase

**Challenge Phase** (proving edge):
- Run conservative (1% risk per trade)
- Aim for stability, not maximum returns
- Meet minimum trading days requirement
- Stay well under drawdown limits

**Funded Phase** (live capital):
- Slightly increased risk (1.5-2% per trade)
- Focus on consistency over time
- Monitor for strategy decay
- Regular profit splits

### Implementation Notes

From Kimi discussion:
> "Paper trade for 6-12 months before funded. 100+ approved code changes and a track record of *human-validated* improvements."

**Graduation Criteria:**
1. 200+ trades on paper with consistent edge
2. Win rate >55% with Sharpe >1.0
3. Max drawdown <10% sustained
4. Edge proven across 3+ market regimes

---

## 2. Strategy Router / Pilot Model

### Core Concept

From the Jan 19 Discussion Summary:

> **Pilot Model / Strategy Router** - Auto-delegates EAs to pairs based on:
> - Time (London/NY/Asia session opens)
> - Market condition (trending/ranging/volatility)
> - Correlation across timeframes (15M with lower TFs)
> 
> Each EA has built-in drawdown limits and self-deactivates on failure.

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     STRATEGY ROUTER                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  INPUTS:                                                         │
│  ├─ Current time (session: London/NY/Asia/Off-hours)            │
│  ├─ Market regime per pair (Trending/Ranging/Volatile)          │
│  ├─ Correlation matrix (cross-pair, cross-timeframe)            │
│  └─ Each EA's performance history + current status               │
│                                                                  │
│  LOGIC:                                                          │
│  1. Classify current market conditions                           │
│  2. Match conditions to EA capabilities                          │
│  3. Check portfolio correlation limits                           │
│  4. Allocate capital across qualifying EAs                       │
│  5. Activate/deactivate EAs based on rules                       │
│                                                                  │
│  OUTPUTS:                                                        │
│  ├─ EA activation commands (enable/disable/modify)               │
│  ├─ Capital allocation per EA                                    │
│  └─ Risk adjustment signals                                      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Session-Based Routing

| Session | Time (UTC) | Pairs to Favor | Strategy Types |
|---------|------------|----------------|----------------|
| **Asia** | 00:00-08:00 | USDJPY, AUDJPY, NZDUSD | Range strategies |
| **London** | 08:00-16:00 | EURUSD, GBPUSD, EURGBP | Breakout, momentum |
| **New York** | 13:00-21:00 | EURUSD, USDCAD, Gold | Trend continuation |
| **Overlap (London+NY)** | 13:00-16:00 | Majors | High volatility strategies |
| **Off-hours** | 21:00-00:00 | None | No trading (skip) |

### Regime-Based Routing

```python
class MarketRegime(Enum):
    TRENDING_UP = "trending_up"      # ADX > 25, price > MA50
    TRENDING_DOWN = "trending_down"  # ADX > 25, price < MA50
    RANGING = "ranging"              # ADX < 20
    VOLATILE = "volatile"            # ATR > 2σ of 20-day mean
    TRANSITIONING = "transitioning"  # Mixed signals

# Strategy-Regime Matching
REGIME_STRATEGY_MAP = {
    "TRENDING_UP": ["momentum", "breakout", "trend_following"],
    "TRENDING_DOWN": ["momentum", "breakout", "trend_following"],
    "RANGING": ["mean_reversion", "support_resistance", "scalping"],
    "VOLATILE": ["reduced_size", "wider_stops", "avoid_new_entries"],
    "TRANSITIONING": ["wait_for_clarity", "small_tests_only"]
}
```

### Correlation Management

The Router prevents over-exposure:

```python
# Before activating a new EA
def check_correlation_limits(new_ea, active_eas):
    """Prevent correlated risk concentration."""
    
    # Rule 1: Max 3 EAs on same pair
    same_pair_count = sum(1 for ea in active_eas if ea.pair == new_ea.pair)
    if same_pair_count >= 3:
        return False, "Max 3 EAs per pair"
    
    # Rule 2: Max 5 EAs in same direction across correlated pairs
    correlated_pairs = get_correlated_pairs(new_ea.pair)  # e.g., EURUSD ↔ GBPUSD
    same_direction_count = sum(
        1 for ea in active_eas 
        if ea.pair in correlated_pairs and ea.direction == new_ea.direction
    )
    if same_direction_count >= 5:
        return False, "Correlated exposure limit"
    
    # Rule 3: Total portfolio correlation < 0.7
    if calculate_portfolio_correlation(active_eas + [new_ea]) > 0.7:
        return False, "Portfolio correlation too high"
    
    return True, "OK"
```

### Self-Deactivation Rules

Each EA has built-in circuit breakers:

| Condition | Action |
|-----------|--------|
| 3 consecutive losses | Pause for current session |
| Daily loss > 2% (EA level) | Pause until next day |
| Win rate drops 15% vs backtest | Flag for review |
| 10 trades with <1.0 profit factor | Quarantine |

---

## 3. Risk Management Architecture

### The Risk Governor (Hardcoded, Immutable)

From Kimi discussion - these rules **cannot be overridden by AI**:

```python
# risk_governor.py - NEVER LET AI MODIFY THIS FILE

class RiskGovernor:
    """Immutable risk limits enforced at execution layer."""
    
    # HARDCODED LIMITS (NEVER CHANGE DURING RUNTIME)
    MAX_DAILY_LOSS_PER_STRATEGY = 0.02  # 2%
    MAX_DAILY_LOSS_TOTAL = 0.05         # 5%
    MAX_POSITION_SIZE = 0.01            # 1% of account per trade
    MAX_OPEN_TRADES_PER_STRATEGY = 3
    MAX_OPEN_TRADES_TOTAL = 10
    MAX_DRAWDOWN_TOTAL = 0.08           # 8% (buffer before prop firm limit)
    
    def validate_trade(self, trade_request):
        """Block any trade that violates limits."""
        
        if self.daily_loss >= self.MAX_DAILY_LOSS_TOTAL:
            raise TradeBlocked("Daily loss limit reached")
        
        if self.current_drawdown >= self.MAX_DRAWDOWN_TOTAL:
            raise TradeBlocked("Drawdown limit reached - trading halted")
        
        if trade_request.risk_amount > self.account_balance * self.MAX_POSITION_SIZE:
            raise TradeBlocked("Position size exceeds limit")
        
        return True
```

### Three-Tier Risk Controls

```
┌─────────────────────────────────────────────────────────────────┐
│  TIER 1: STRATEGY-LEVEL CONTROLS (Per EA)                       │
│  ├─ Max 2% daily loss per strategy                              │
│  ├─ Max 3 open trades per strategy                              │
│  ├─ SL/TP derived from ATR (not arbitrary)                      │
│  └─ Self-pause on 3 consecutive losses                          │
├─────────────────────────────────────────────────────────────────┤
│  TIER 2: PORTFOLIO-LEVEL CONTROLS (Across all EAs)              │
│  ├─ Max 5% total daily loss                                     │
│  ├─ Max 10 open trades system-wide                              │
│  ├─ Max 0.7 portfolio correlation                               │
│  └─ Correlation monitoring per session                          │
├─────────────────────────────────────────────────────────────────┤
│  TIER 3: ACCOUNT-LEVEL CONTROLS (Emergency)                     │
│  ├─ Max 8% total drawdown → HALT ALL TRADING                    │
│  ├─ VIX > 50 → Close all positions                              │
│  ├─ Broker disconnect > 30s → Mark all positions UNKNOWN        │
│  └─ Human notification required to resume                       │
└─────────────────────────────────────────────────────────────────┘
```

### Dynamic Stop Loss / Take Profit (ATR-Based, Not AI-Adjusted)

From Kimi's critique of AI-adjusted risk:
> "Your example adjusts stops based on recent win rate—pure superstition. Recent P&L has zero predictive value."

**Correct Approach:**

```python
def calculate_sl_tp(pair, timeframe, strategy_type):
    """Market-derived risk levels, never performance-based."""
    
    atr = get_atr(pair, timeframe, period=14)
    
    # Strategy-specific multipliers (fixed, backtested)
    SL_MULTIPLIERS = {
        "scalping": 1.0,          # Tight stops
        "breakout": 1.5,          # Room for retest
        "trend_following": 2.0,   # Wider for pullbacks
        "mean_reversion": 1.2     # Moderate
    }
    
    TP_MULTIPLIERS = {
        "scalping": 1.5,          # Quick profits
        "breakout": 2.5,          # Let runners run
        "trend_following": 3.0,   # Maximize trends
        "mean_reversion": 1.8     # Target mean
    }
    
    sl_pips = atr * SL_MULTIPLIERS[strategy_type]
    tp_pips = atr * TP_MULTIPLIERS[strategy_type]
    
    return sl_pips, tp_pips
```

---

## 4. Multi-Strategy Portfolio Design

### Portfolio Construction Theory

From Kimi conversation:

> "Multiple *uncorrelated* strategies smooth equity curves. Combined portfolio: Overall win rate ~55%, Drawdown drops from 12% to 6% (diversification effect), Sharpe ratio improves 40%."

### Target Portfolio Structure

| Strategy Type | Count | Pairs | Regime | Risk Allocation |
|---------------|-------|-------|--------|-----------------|
| **Breakout** | 1-2 | EURUSD, GBPUSD | Trending | 25% |
| **Mean Reversion** | 1-2 | USDJPY, EURGBP | Ranging | 25% |
| **Momentum** | 1-2 | EURUSD, XAUUSD | Strong Trend | 25% |
| **Scalping** | 1-2 | Any Majors | Low volatility | 25% |

**Critical Rule:** Strategies must be **uncorrelated**. Two breakout variants = 1 strategy.

### Strategy Lifecycle Tags

From original TRD:

| Tag | Meaning | Requirements to Graduate |
|-----|---------|-------------------------|
| `@primal` | New, in paper testing | 50+ paper trades |
| `@pending` | Live but unproven | 100+ trades, >50% WR |
| `@perfect` | Consistent performer | 100+ trades, >60% WR, Sharpe >1.0 |
| `@quarantine` | Showing problems | Under investigation |
| `@dead` | Killed | Moved to graveyard |

### Saturday Evolution Cycle

Weekly improvement process (from TRD):

1. **00:01-00:15:** Performance data collection
2. **00:15-00:45:** Pattern analysis across all EAs
3. **00:45-02:00:** Cross-breeding top performers (paper only)
4. **02:00-04:00:** Parallel backtesting of mutations
5. **04:00:** Human approval queue
6. **04:30-05:00:** Deploy approved changes to paper trading

**Key Rule:** Evolution happens in paper only. Human promotes to live.

---

## 5. HFT-Friendly Brokers

### Research from Jan 19 Discussion

User mentioned: "0.0% spreads for 10000 pips"

**Brokers identified:**

| Broker | Spreads | Latency | MT5 Support | Notes |
|--------|---------|---------|-------------|-------|
| **StarTrader** | 0.6 pips | Fast | ✅ | $3/lot commission |
| **VT Markets** | Low | NY4 Equinix | ✅ | cTrader also supported |
| **FP Markets** | Low | Equinix DCs | ✅ | ASIC regulated |

### Key Considerations

1. **Scalping focus** - Need tight spreads and fast execution
2. **VPS location** - Should be near broker servers
3. **Prop firm compatibility** - Check if broker is allowed
4. **Demo account quality** - Must mirror live execution

### MT4 vs MT5 Decision

From discussion:
> "MT5 is more powerful (21 timeframes, multi-asset, better tester). MT5 supports stocks, futures, crypto. Some prop firms still require MT4. **Decision: Prioritize MT5**"

---

## 6. Deployment Path to Live Trading

### Graduated Approach (from Kimi)

| Phase | Capital | Agent Control | Human Approval | Duration |
|-------|---------|---------------|----------------|----------|
| 1 | $0 (paper) | Proposes only | Every change | 90+ days |
| 2 | $500 (micro live) | Proposes + executes | Weekly review | 30+ days |
| 3 | $2,000 (small live) | Proposes + executes | Monthly review | 90+ days |
| 4 | Prop firm challenge | Proposes + executes | Quarterly audit | As needed |
| 5 | Funded account | Proposes + executes | Ongoing monitoring | Indefinite |

### Validation Requirements Before Each Phase

**Before Phase 2 (Paper → Small Live):**
- 200+ trades completed
- Win rate >55%
- Profit factor >1.5
- Max drawdown <10%
- Edge proven in 3+ regimes

**Before Phase 4 (Small Live → Prop Challenge):**
- 500+ trades completed
- Consistent monthly returns
- Risk governor never triggered
- No catastrophic bugs
- Human validated all code

### Circuit Breakers for Live Trading

```python
# Emergency protocols (from TRD)

class DisasterProtocol:
    
    def market_crash_protocol(self):
        """Trigger: VIX > 50 OR Account drawdown > 10% in 1 hour"""
        self.close_all_positions(order_type="MARKET")
        self.pause_all_strategies()
        self.send_sms("EMERGENCY: All trading halted")
        self.require_manual_restart()
    
    def broker_outage_protocol(self):
        """Trigger: MT5 connection fails for 30+ seconds"""
        self.attempt_reconnection(retries=3)
        if not self.connected:
            self.mark_positions_unknown()
            self.send_sms("BROKER OUTAGE: Manual intervention needed")
    
    def strategy_decay_detected(self, strategy_id):
        """Trigger: Win rate drops >15% vs backtest over 20+ trades"""
        self.auto_pause_strategy(strategy_id)
        self.add_to_investigation_queue(strategy_id)
        self.send_alert(f"Strategy {strategy_id} paused - decay detected")
```

---

## Summary

This document covers the missing pieces from the HANDOFF:

1. **Prop Firm Strategy:** Requirements, challenge vs funded phases, graduation criteria

2. **Strategy Router / Pilot Model:** Session-based routing, regime-based routing, correlation management, self-deactivation rules

3. **Risk Architecture:** Three-tier controls, hardcoded limits, ATR-based (not AI-adjusted) risk

4. **Multi-Strategy Portfolio:** Uncorrelated strategies, lifecycle tags, Saturday evolution

5. **HFT Brokers:** StarTrader, VT Markets, FP Markets; MT5 prioritized

6. **Deployment Path:** Graduated approach from paper → micro live → prop firm

---

**Key Philosophy (from Kimi):**
> "Build the AI-Augmented Trader, not the AI Trader. AI is an auditor and advisor, never the trader or coder."

---

## 7. MT5 MCP Server Configuration

### Repository Location

The MT5 MCP server has been cloned to:
```
/home/mubarkahimself/Desktop/QUANTMINDX/mcp-metatrader5-server/
```

**Source:** [Qoyyuum/mcp-metatrader5-server](https://github.com/Qoyyuum/mcp-metatrader5-server)

### Available Tools (30+)

The MCP server exposes these tools via the FastMCP framework:

#### Connection Management
| Tool | Description |
|------|-------------|
| `initialize(path)` | Initialize MT5 terminal (must call first) |
| `login(login, password, server)` | Login to trading account |
| `shutdown()` | Close MT5 connection |

#### Market Data Functions
| Tool | Description |
|------|-------------|
| `get_symbols()` | Get all available symbols |
| `get_symbols_by_group(group)` | Filter symbols (e.g., "EUR*") |
| `get_symbol_info(symbol)` | Get symbol details (spread, volume, etc.) |
| `get_symbol_info_tick(symbol)` | Get latest tick (bid/ask) |
| `symbol_select(symbol, visible)` | Add symbol to Market Watch |
| `copy_rates_from_pos(symbol, timeframe, start_pos, count)` | Get historical bars |
| `copy_rates_from_date(symbol, timeframe, date_from, count)` | Get bars from date |
| `copy_rates_range(symbol, timeframe, date_from, date_to)` | Get bars in date range |
| `copy_ticks_from_pos(symbol, start_time, count, flags)` | Get tick data |
| `get_account_info()` | Account balance, equity, margin |
| `get_terminal_info()` | Terminal state and version |
| `get_version()` | MT5 version info |

#### Trading Functions
| Tool | Description |
|------|-------------|
| `order_send(request)` | Send trade order |
| `order_check(request)` | Validate order before sending |
| `positions_get(symbol, group)` | Get open positions |
| `positions_get_by_ticket(ticket)` | Get position by ticket ID |
| `orders_get(symbol, group)` | Get pending orders |
| `orders_get_by_ticket(ticket)` | Get order by ticket ID |
| `history_orders_get(...)` | Get historical orders |
| `history_deals_get(...)` | Get historical deals |

### Timeframe Constants

Use these INTEGER values for `timeframe` parameters:

| Value | Meaning | Value | Meaning |
|-------|---------|-------|---------|
| 1 | M1 (1 min) | 60 | H1 (1 hour) |
| 5 | M5 (5 min) | 240 | H4 (4 hours) |
| 15 | M15 (15 min) | 1440 | D1 (1 day) |
| 30 | M30 (30 min) | 10080 | W1 (1 week) |

### Order Types (INTEGER Required)

```python
# action (Trade operation type)
1 = TRADE_ACTION_DEAL      # Execute immediately
2 = TRADE_ACTION_PENDING   # Place pending order
5 = TRADE_ACTION_SLTP      # Modify SL/TP
6 = TRADE_ACTION_MODIFY    # Modify pending order
8 = TRADE_ACTION_REMOVE    # Remove pending order

# type (Order type)
0 = ORDER_TYPE_BUY         # Buy market order
1 = ORDER_TYPE_SELL        # Sell market order
2 = ORDER_TYPE_BUY_LIMIT   # Buy limit
3 = ORDER_TYPE_SELL_LIMIT  # Sell limit
4 = ORDER_TYPE_BUY_STOP    # Buy stop
5 = ORDER_TYPE_SELL_STOP   # Sell stop

# type_filling (Order filling type)
0 = ORDER_FILLING_FOK      # Fill or Kill
1 = ORDER_FILLING_IOC      # Immediate or Cancel
2 = ORDER_FILLING_RETURN   # Return remaining
```

### Gemini CLI Configuration

Add to your MCP client config (e.g., `settings.json`):

```json
{
  "mcpServers": {
    "mcp-metatrader5-server": {
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/Qoyyuum/mcp-metatrader5-server",
        "mt5mcp"
      ]
    }
  }
}
```

**Alternative (local installation):**
```bash
cd /home/mubarkahimself/Desktop/QUANTMINDX/mcp-metatrader5-server
uv run fastmcp install gemini-cli src/mcp_mt5/main.py
```

### Pydantic AI Integration Pattern

The MCP server supports integration with Pydantic AI agents:

```python
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio

# Setup MCP server for MetaTrader 5
mt5_server = MCPServerStdio(
    'uvx',
    args=['--from', 'mcp-metatrader5-server', 'mt5mcp'],
    timeout=30
)

# Create trading agent with MT5 tools
trading_agent = Agent(
    model,
    system_prompt="You are an expert trading analyst...",
    toolsets=[mt5_server],
    retries=2
)

# Use the agent
async with trading_agent:
    result = await trading_agent.run("Analyze EURUSD on H1 timeframe...")
```

### Planned Extensions (Not Yet Implemented)

From our discussions, the following extensions are planned:

1. **EA Management** - Deploy/start/stop Expert Advisors remotely
2. **Real-Time Monitoring** - Stream account state changes
3. **Per-EA Performance** - Track individual bot P&L
4. **Multi-Account Support** - Manage multiple demo/live accounts
5. **Alert System** - Push notifications for drawdown, broker disconnect

---

## 8. Modular Trading Architecture (Hummingbot-Inspired)

### Design Philosophy

From Kimi conversation discussions about building a modular trading system:

> "For live trading, you need **direct broker APIs** (MT5, CCXT). Every millisecond counts."
> 
> "OpenBB adds abstraction layers. Use direct APIs instead."

### Broker Abstraction Layer

Inspired by how Hummingbot and similar frameworks handle multi-exchange support:

```python
# Unified interface for different brokers
class BrokerConnector:
    """Abstract broker interface for unified order management."""
    
    def __init__(self, broker_config):
        self.broker = broker_config['type']  # 'mt5' or 'ccxt'
        
        if self.broker == 'mt5':
            import MetaTrader5 as mt5
            mt5.initialize(broker_config['path'])
            mt5.login(**broker_config['credentials'])
        elif self.broker == 'ccxt':
            import ccxt
            self.exchange = ccxt.binance(broker_config['keys'])
    
    def get_balance(self):
        if self.broker == 'mt5':
            return mt5.account_info().balance
        else:
            return self.exchange.fetch_balance()['total']['USDT']
    
    def place_order(self, symbol, side, volume, price=None):
        if self.broker == 'mt5':
            return self._place_mt5_order(symbol, side, volume, price)
        else:
            return self._place_ccxt_order(symbol, side, volume, price)
```

### Latency Comparison (From Kimi Research)

| Method | Latency | Use Case |
|--------|---------|----------|
| **MT5 Python API** | 1-10ms | Forex, CFDs, live trading |
| **CCXT** | 50-100ms | Crypto, backtesting |
| **OpenBB** | 100-500ms | Analysis only, not trading |

### Multi-Asset Support Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    STRATEGY LAYER                                │
│  (EAs, Python strategies, signals)                              │
├─────────────────────────────────────────────────────────────────┤
│                    BROKER ROUTER                                 │
│  Routes orders to appropriate API based on asset class          │
├──────────────────────┬──────────────────────────────────────────┤
│    MT5 CONNECTOR     │         CCXT CONNECTOR                    │
│  └─ Forex, CFDs      │  └─ Crypto (Binance, Coinbase, etc.)     │
│  └─ Gold, Indices    │                                          │
│  └─ Low latency      │                                          │
├──────────────────────┼──────────────────────────────────────────┤
│    MCP SERVER        │       WEBSOCKET FEEDS                     │
│  (AI integration)    │  (Real-time price streaming)              │
└──────────────────────┴──────────────────────────────────────────┘
```

### Data Feed Architecture

From Kimi discussion on data sources:

```python
# data_feeds/mt5_feed.py
import MetaTrader5 as mt5

class MT5DataFeed:
    """Real-time and historical data from MT5."""
    
    def get_ohlcv(self, symbol, timeframe, count):
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
        return pd.DataFrame(rates)
    
    def stream_ticks(self, symbol, callback):
        """Stream live ticks to callback function."""
        while True:
            tick = mt5.symbol_info_tick(symbol)
            callback(tick)
            time.sleep(0.1)

# data_feeds/ccxt_feed.py  
import ccxt

class CCXTDataFeed:
    """Crypto data via CCXT for backtesting."""
    
    def get_ohlcv(self, symbol, timeframe, limit=1000):
        binance = ccxt.binance()
        return binance.fetch_ohlcv(symbol, timeframe, limit=limit)
```

### Order Manager Design

```python
class OrderManager:
    """Unified order management across brokers."""
    
    def __init__(self, connectors: dict):
        # {"forex": MT5Connector, "crypto": CCXTConnector}
        self.connectors = connectors
    
    def route_order(self, order_request):
        """Route to appropriate connector based on symbol."""
        
        if self._is_forex_symbol(order_request.symbol):
            connector = self.connectors['forex']
        else:
            connector = self.connectors['crypto']
        
        # Validate through Risk Governor first
        risk_governor.validate_trade(order_request)
        
        return connector.place_order(
            symbol=order_request.symbol,
            side=order_request.side,
            volume=order_request.volume,
            sl=order_request.sl,
            tp=order_request.tp
        )
```

### Key Design Principles (From Kimi)

| Principle | Implementation |
|-----------|----------------|
| **Direct APIs** | MT5/CCXT, not abstraction layers |
| **No Experimental Tech** | FastAPI, Redis, Qdrant - all mature |
| **Modular Connectors** | Each broker is a separate module |
| **Type Safety** | Pydantic models for all requests/responses |
| **Human-in-the-Loop** | AI proposes, human approves before live |

### Future Crypto Integration (Low Priority)

While MT5/Forex is the current priority, the architecture supports future crypto:

```python
# Example future extension
BROKER_CONFIG = {
    "forex": {
        "type": "mt5",
        "path": "C:\\Program Files\\MetaTrader 5\\terminal64.exe",
        "credentials": {"login": 12345, "password": "...", "server": "..."}
    },
    "crypto": {
        "type": "ccxt",
        "exchange": "binance",
        "keys": {"apiKey": "...", "secret": "..."}
    }
}

# Latency targets
# MT5: ~5ms (acceptable for scalping)
# CCXT: ~50-100ms (acceptable for swing trading)
```

---

## Summary

This document covers:

1. **Prop Firm Strategy:** Requirements, challenge vs funded phases, graduation criteria

2. **Strategy Router / Pilot Model:** Session-based routing, regime-based routing, correlation management, self-deactivation rules

3. **Risk Architecture:** Three-tier controls, hardcoded limits, ATR-based (not AI-adjusted) risk

4. **Multi-Strategy Portfolio:** Uncorrelated strategies, lifecycle tags, Saturday evolution

5. **HFT Brokers:** StarTrader, VT Markets, FP Markets; MT5 prioritized

6. **Deployment Path:** Graduated approach from paper → micro live → prop firm

7. **MT5 MCP Server:** Complete tool reference, Gemini CLI config, Pydantic AI integration

8. **Modular Architecture:** Broker abstraction layer, CCXT integration patterns, multi-asset support design
