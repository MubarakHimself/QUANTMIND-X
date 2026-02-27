# Hummingbot Trading Strategy Implementation

**Research Documentation** - Exploring how Hummingbot implements trading strategies

**Date**: February 2026
**Sources**: [Hummingbot Docs](https://hummingbot.org/docs/), [GitHub](https://github.com/hummingbot/hummingbot)

---

## Overview

Hummingbot is an open-source Python framework for building automated market making and algorithmic trading bots. It is designed to be modular and extensible, allowing users to automate any trading strategy on any exchange and blockchain. Users have generated over $34 billion in trading volume across 140+ unique trading venues.

**License**: Apache 2.0
**Language**: Python (with some Cython for performance)

---

## 1. Strategy Framework Architecture

### Event-Driven Design

Hummingbot uses an **Event-Driven Architecture (EDA)**, which is highly modular and efficient for high-frequency trading:

| Component | Description |
|-----------|-------------|
| **Event System** | Core of the architecture, handles market events and trading state changes |
| **Event Listeners** | Built into each strategy for real-time market response |
| **Market Events** | Complete event type system covering all market activities |

**Advantages over traditional polling:**
- Real-time response to market changes
- Reduced API calls (avoids rate limits)
- Higher efficiency for HFT strategies

### Class Hierarchy

```
StrategyBase (Cython - lowest level)
    |
    v
StrategyPyBase (Python-friendly interface)
    |
    v
ScriptStrategyBase (simplified strategy development)
    |
    v
StrategyV2Base (V2 framework with Executors support)
```

---

## 2. Strategy Definition and Configuration (V2 Framework)

Since version 2.0, Hummingbot offers **two ways to create strategies**:

### A. Scripts

Scripts serve as the **entry point for all Hummingbot strategies**:

- The `on_tick` method defines actions taken each clock tick
- Provides access to core Hummingbot components like connectors
- Can range from simple Python files containing all strategy logic to launcher scripts that spawn multiple Controllers
- Each controller defined in a script can represent a separate sub-strategy

**Key Lifecycle Methods:**

```python
def on_tick(self):
    """
    The code here will be executed on every tick
    which is every second by default (configurable down to 0.1s).
    This is a timer-based tick, not a trade tick.
    """
    # Get current market price
    mid_price = self.get_mid_price()
    # Strategy logic here

async def on_stop(self):
    """
    Callback when strategy is stopped.
    Clean up resources, cancel tasks, close connections.
    """
    # Stop all data sources
    # Cancel pending orders
    # Close connectors properly

def on_status(self):
    """Called when user issues 'status' command"""
    return "Strategy status information"

def on_buy_order_completed(self, event):
    """Called when a buy order is fully filled"""

def on_sell_order_completed(self, event):
    """Called when a sell order is fully filled"""
```

### B. Controllers

Controllers define **modularized strategies** using components:

- **Executors**: For order execution management
- Support for **backtesting** strategies
- Facilitate **multi-bot deployment** via Dashboard
- More flexible and customizable than legacy V1 strategies

**Controller Architecture:**

| Component | Role | Description |
|-----------|------|-------------|
| **Script** | Entry Point | Entry point for all strategies; defines `on_tick` method |
| **Controller** | Strategy Logic | Analyzes market data, generates trading signals |
| **Executor** | Order Execution | Manages orders and positions |
| **Market Data Provider** | Data Layer | Provides candles, order book, and trade data |

---

## 3. Strategy Base Classes and Interfaces

### Core Event Listeners

```python
# Built-in event listeners in StrategyBase
self._sb_create_buy_order_listener = BuyOrderCreatedListener(self)
self._sb_create_sell_order_listener = SellOrderCreatedListener(self)
self._sb_fill_order_listener = OrderFilledListener(self)
self._sb_fail_order_listener = OrderFailedListener(self)
self._sb_cancel_order_listener = OrderCancelledListener(self)
```

### Available Base Classes

| Class | Purpose |
|-------|---------|
| `StrategyBase` | Lowest-level Cython base class |
| `StrategyPyBase` | Python-friendly interface |
| `ScriptStrategyBase` | Simplified strategy development with `on_tick` |
| `StrategyV2Base` | V2 framework with Executors support |
| `DirectionalStrategyBase` | For directional trading strategies |
| `MarketMakingStrategyBase` | For market making strategies |

### Simple Script Strategy Example

```python
from hummingbot.strategy.script_strategy_base import ScriptStrategyBase

class CustomStrategy(ScriptStrategyBase):
    markets = {"binance": {"BTC-USDT"}}

    def on_tick(self):
        # Strategy logic executed every tick
        mid_price = self.get_mid_price("binance", "BTC-USDT")
        # Place orders, check positions, etc.
```

---

## 4. Built-in Strategies

### Strategy Templates

Hummingbot provides four main strategy templates:

| Strategy | Description | Profit Source |
|----------|-------------|---------------|
| **Pure Market Making (PMM)** | Places buy/sell limit orders in a single market | Earns bid-ask spread |
| **Arbitrage** | Low buys, high sells across different exchanges | Price discrepancy between venues |
| **Cross-Exchange Market Making (XEMM)** | Combines market making and arbitrage | Spread between maker/taker exchanges |
| **AMM Arbitrage** | Optimizes liquidity pool returns on DEXs | DEX price differences |

### Pure Market Making (PMM)

| Feature | Description |
|---------|-------------|
| **Purpose** | Provide liquidity on a single exchange |
| **Mechanism** | Places buy and sell limit orders around a mid-price |
| **Profit Source** | Earns bid-ask spread |
| **Customization** | Adjustable bid/ask spread, order refresh frequency, min/max trade volume, risk parameters |

### Cross-Exchange Market Making (XEMM)

| Parameter | Description | Example |
|-----------|-------------|---------|
| `maker_platform` | Exchange for placing orders | Binance, KuCoin |
| `taker_platform` | Exchange for hedging | Bybit, OKX |
| `order_amount` | Order quantity | 0.1 BTC |
| `spread_bps` | Spread in basis points | 10-50 bps |
| `min_spread_bps` | Minimum spread threshold | 5 bps |
| `max_order_age` | Maximum order duration | 120 seconds |

---

## 5. Adding Custom Strategies

### Directory Structure

```
hummingbot/
├── controllers/              # Controller code for V2 strategies
├── scripts/                  # Script files for automation
│   └── community/            # Community strategy templates
├── hummingbot/
│   └── strategy/             # Legacy V1 strategy implementations
│   └── strategy_v2/
│       └── executors/        # Executor implementations
│           └── executor_base.py
│           └── position_executor.py
│           └── dca_executor.py
│           └── grid_executor.py
└── conf/
    └── strategies/           # Strategy configuration files
```

### Creating a Custom Strategy

1. **Create strategy folder:**
   ```bash
   cd hummingbot/strategy
   mkdir my_custom_strategy
   cd my_custom_strategy
   ```

2. **Key files needed:**
   - Strategy implementation (Python)
   - Config map (defines parameters)
   - Start function

3. **For V2 strategies, create a Controller:**

```python
# controllers/my_controller.py
from hummingbot.strategy_v2.controllers.directional_trading_controller import (
    DirectionalTradingController,
    DirectionalTradingControllerConfig
)

class MyControllerConfig(DirectionalTradingControllerConfig):
    # Define custom parameters
    custom_parameter: float = 0.01

class MyController(DirectionalTradingController):
    def __init__(self, config: MyControllerConfig):
        super().__init__(config)

    def on_tick(self):
        # Custom strategy logic
        pass
```

### Available Executor Types

| Executor | Purpose |
|----------|---------|
| **PositionExecutor** | Standard position management with stop-loss logic |
| **DCAExecutor** | Dollar-cost averaging execution |
| **GridExecutor** | Grid trading execution |

---

## 6. Strategy Lifecycle

### Lifecycle States

| State | Description | Operations |
|-------|-------------|------------|
| **CREATED** | Strategy configuration created, not running | Stored in database |
| **RUNNING** | Strategy actively executing | Listening to market data, executing trades |
| **PAUSED** | Temporarily halted | Stops execution but preserves state/memory |
| **RESUMED** | Restored from paused state | Continues from previous state |
| **STOPPED** | Permanently terminated | Resources released, logs written |
| **ERROR** | Exception occurred | Data loss, risk control failure, etc. |

### Core States (from network_iterator.pyx)

- **STOPPED**: Strategy/iterator has been stopped
- **NOT_CONNECTED**: Network not connected (intermediate state)
- **CONNECTED**: Network connected and running

### Strategy Execution Flow

1. **Initialization**: `__init__` - load config, create controllers
2. **Market preparation**: `on_tick` runs but `ready_to_trade = False` until enough historical data
3. **Running phase**: `ready_to_trade = True`, full decision cycle each tick
4. **Stop**: `on_stop` called for cleanup

### Start/Stop Commands

- Configuration parameter `initial_state` can be set to `"running"`, `"paused"`, or `"stopped"`
- If set to `"stopped"`, must explicitly start via command `/start`

---

## 7. Configuration Format

### YAML Configuration (Primary Format)

**strategy.yaml example:**
```yaml
# conf/strategies/cloud_market_making.yml
strategy: pure_market_making
exchange: binance
market: BTC-USDT

bid_spread: 0.001
ask_spread: 0.001
order_refresh_time: 15
order_amount: 0.01

# Advanced parameters
max_order_age: 1800
order_levels: 3
order_level_spread: 0.002
order_level_amount: 0.005

# Risk control settings
inventory_skew_enabled: true
inventory_target_base_pct: 0.5
filled_order_delay: 5.0
```

**Cross-Exchange Market Making Config:**
```yaml
strategy: 'cross_exchange_market_making'
market:
  base_asset: 'ETH'
  quote_asset: 'USDT'
  base_asset_ticker: 'ETHUSDT'
  quote_asset_ticker: 'USDT'
```

### Key Configuration Files

| File | Purpose |
|------|---------|
| `strategy.yaml` | Defines trading strategy parameters (pairs, order types, prices) |
| `connector.yaml` | Exchange connector settings (API keys, wallet addresses) |
| `conf/strategies/` | Directory where all strategy configs are stored |
| `scripts/community/` | Community strategy templates |

### JSON/Python Dict Format (for API/Dashboard)

```python
instances = {
    "arbitrage_bot": {
        "strategy": "cross_exchange_market_making",
        "exchanges": ["binance", "okx"],
        "pairs": ["BTC-USDT", "ETH-USDT"]
    },
    "market_making_bot": {
        "strategy": "pure_market_making",
        "exchange": "gate_io",
        "pairs": ["XRP-USDT", "ADA-USDT"]
    }
}
```

### Docker Compose Configuration

```yaml
version: '3.8'
services:
  hummingbot:
    container_name: hummingbot-cloud
    image: hummingbot/hummingbot:latest
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 16G
    volumes:
      - ./conf:/home/hummingbot/conf
      - ./logs:/home/hummingbot/logs
      - ./data:/home/hummingbot/data
      - ./scripts:/home/hummingbot/scripts
    environment:
      - CONFIG_PASSWORD=${CONFIG_PASSWORD}
      - CONFIG_FILE_NAME=cloud_strategy.py
    restart: unless-stopped
```

---

## 8. Connector Architecture

### Connector Types

| Type | Description | Connection Method |
|------|-------------|-------------------|
| **CLOB CEX** | Centralized exchanges with central limit order books | API keys |
| **CLOB DEX** | Decentralized exchanges with on-chain order books | Wallet keys |
| **AMM DEX** | Automated Market Maker protocols | Gateway middleware |

### Exchange Sponsors (Major Connectors)

| Exchange | Type | Connector ID(s) |
|----------|------|-----------------|
| Binance | CLOB CEX | `binance`, `binance_perpetual` |
| Bybit | CLOB CEX | `bybit`, `bybit_perpetual` |
| OKX | CLOB CEX | `okx`, `okx_perpetual` |
| Gate.io | CLOB CEX | `gate_io`, `gate_io_perpetual` |
| KuCoin | CLOB CEX | `kucoin`, `kucoin_perpetual` |
| Hyperliquid | CLOB DEX | `hyperliquid`, `hyperliquid_perpetual` |
| dYdX | CLOB DEX | `dydx_v4_perpetual` |
| Uniswap | AMM DEX | `uniswap` |
| Jupiter | AMM DEX | `jupiter` |

---

## 9. Order Types and Executors

### Supported Order Types

| Order Type | Description |
|------------|-------------|
| **Market Order** | Immediate execution at best available price |
| **Limit Order** | Execution at specified price or better |
| **Iceberg Order** | Hidden order showing only partial size |
| **TWAP Order** | Time-weighted execution over intervals |
| **Conditional Order** | Triggered by price/conditions |
| **OCO Order** | One-cancels-the-other |

### Algorithm Execution Types

| Algorithm | Purpose |
|-----------|---------|
| **TWAP** | Minimize market impact via time-based execution |
| **VWAP** | Match volume-weighted average market price |
| **Iceberg** | Hide full order size to prevent detection |
| **Sniper** | Aggressive execution targeting best prices |
| **BestLimit** | Maintain best bid/ask positions |

### Core Executor Components

- **Order Scheduler** - Determines when/how to split orders
- **Order Tracker** - Monitors order status and fills
- **Data Handler** - Manages market data feeds
- **Strategy Engine** - Executes trading logic
- **Risk Manager** - Controls exposure and limits

---

## 10. Hummingbot Ecosystem

### Companion Modules

| Module | Description |
|--------|-------------|
| **Hummingbot Client** | Core trading engine with CLI interface |
| **Gateway** | Middleware for DEX connections (TypeScript-based) |
| **Hummingbot API** | API server for trading execution and deployment |
| **Dashboard** | Web-based GUI for managing multiple bot instances |
| **Hummingbot MCP** | Model Context Protocol for AI assistants |
| **Quants Lab** | Python framework for backtesting and research |
| **Condor** | Telegram interface for Hummingbot |

### API Deployment Methods

```python
# Deploy V2 script
deploy_v2_script(name, profile, script, config)

# Deploy V2 controllers
deploy_v2_controllers(name, profile, controllers)
```

---

## 11. Key Takeaways for Implementation

### Strengths to Emulate

1. **Modular Architecture**: Clean separation between strategy logic, execution, and connectors
2. **Event-Driven Design**: Efficient real-time response to market events
3. **Configuration-Driven**: Strategies defined by YAML configs, logic in code
4. **Executor Pattern**: Composable executors for different order types (DCA, Grid, Position)
5. **Controller/Script Separation**: Scripts as entry points, controllers for modularity
6. **Backtesting Support**: V2 framework designed with backtesting in mind

### Architecture Patterns

```
+------------------+     +------------------+     +------------------+
|     Script       |     |   Controller     |     |    Executor      |
|  (Entry Point)   |---->| (Strategy Logic) |---->| (Order Mgmt)     |
+------------------+     +------------------+     +------------------+
         |                       |                        |
         v                       v                        v
+------------------+     +------------------+     +------------------+
| Market Data      |     | Signal Generator |     | Order Tracker    |
| Provider         |     |                  |     |                  |
+------------------+     +------------------+     +------------------+
```

---

## Sources

- [Hummingbot Official Documentation](https://hummingbot.org/docs/)
- [Hummingbot GitHub Repository](https://github.com/hummingbot/hummingbot)
- [Hummingbot Chinese Documentation](https://www.wuzao.com/hummingbot/document/docs/index.html)
- [Hummingbot Strategy Guides](https://www.wuzao.com/p/hummingbot/document/guides/index.html)
- [Hummingbot V2 Framework Tutorial](https://www.wuzao.com/p/hummingbot/tutorial/v2-framework)
- [Hummingbot Dashboard Tutorial](https://hummingbot.org/blog/dashboard-tutorial-pmm-simple-v2/)

---

*This document is for research purposes only.*
