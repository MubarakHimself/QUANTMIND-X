# Hummingbot Exchange Connectors Documentation

This document summarizes all supported crypto exchanges in Hummingbot, including spot and perpetual/futures connectors, paper trading support, 1x leverage availability, and guidance for adding new exchange connectors.

**Source Repository**: `/home/mubarkahimself/Desktop/hummingbot-research`

---

## Table of Contents

1. [Spot Exchange Connectors](#spot-exchange-connectors)
2. [Perpetual/Futures Exchange Connectors](#perpetualfutures-exchange-connectors)
3. [1x Leverage Support (Islamic Finance)](#1x-leverage-support-islamic-finance)
4. [Paper Trading / Testnet Support](#paper-trading--testnet-support)
5. [How to Add a New Exchange Connector](#how-to-add-a-new-exchange-connector)

---

## Spot Exchange Connectors

Hummingbot supports **26 spot exchange connectors** across CLOB CEX, CLOB DEX, and AMM DEX types.

### CLOB CEX (Centralized Exchanges)

| Exchange | Connector ID | Type | Notes |
|----------|--------------|------|-------|
| Binance | `binance` | CLOB CEX | Sponsor - 10% fee discount |
| BitMart | `bitmart` | CLOB CEX | Sponsor |
| Bitget | `bitget` | CLOB CEX | Sponsor |
| BingX | `bing_x` | CLOB CEX | - |
| Bitrue | `bitrue` | CLOB CEX | - |
| Bitstamp | `bitstamp` | CLOB CEX | - |
| BTC Markets | `btc_markets` | CLOB CEX | - |
| Bybit | `bybit` | CLOB CEX | - |
| Coinbase | `coinbase_advanced_trade` | CLOB CEX | - |
| Cube | `cube` | CLOB CEX | - |
| Gate.io | `gate_io` | CLOB CEX | Sponsor - 20% fee discount |
| HTX (Huobi) | `htx` | CLOB CEX | Sponsor - 20% fee discount |
| Kraken | `kraken` | CLOB CEX | - |
| KuCoin | `kucoin` | CLOB CEX | Sponsor - 20% fee discount |
| MEXC | `mexc` | CLOB CEX | - |
| NDAX | `ndax` | CLOB CEX | - |
| OKX | `okx` | CLOB CEX | Sponsor - 20% fee discount |
| AscendEx | `ascend_ex` | CLOB CEX | - |

### CLOB DEX (Decentralized Exchanges)

| Exchange | Connector ID | Type | Notes |
|----------|--------------|------|-------|
| Derive | `derive` | CLOB DEX | Sponsor |
| Dexalot | `dexalot` | CLOB DEX | - |
| Hyperliquid | `hyperliquid` | CLOB DEX | - |
| Injective Helix | `injective_v2` | CLOB DEX | - |
| Vertex | `vertex` | CLOB DEX | - |
| XRP Ledger | `xrpl` | CLOB DEX | - |

### AMM DEX (via Gateway)

| Exchange | Connector ID | Type | Notes |
|----------|--------------|------|-------|
| 0x Protocol | `0x` | AMM DEX (Router) | Via Gateway |
| Balancer | `balancer` | AMM DEX | Via Gateway |
| Curve | `curve` | AMM DEX | Via Gateway |
| Jupiter | `jupiter` | AMM DEX (Router) | Via Gateway |
| Meteora | `meteora` | AMM DEX (CLMM) | Via Gateway |
| PancakeSwap | `pancakeswap` | AMM DEX | Via Gateway |
| QuickSwap | `quickswap` | AMM DEX | Via Gateway |
| Raydium | `raydium` | AMM DEX (AMM, CLMM) | Via Gateway |
| SushiSwap | `sushiswap` | AMM DEX | Via Gateway |
| Trader Joe | `traderjoe` | AMM DEX | Via Gateway |
| Uniswap | `uniswap` | AMM DEX (Router, AMM, CLMM) | Via Gateway |

---

## Perpetual/Futures Exchange Connectors

Hummingbot supports **12 perpetual/futures connectors** for derivatives trading.

### CLOB CEX Perpetual

| Exchange | Connector ID | Type | Testnet Available | Notes |
|----------|--------------|------|-------------------|-------|
| Binance Futures | `binance_perpetual` | CLOB CEX | Yes (`binance_perpetual_testnet`) | Sponsor |
| Bitget Perpetual | `bitget_perpetual` | CLOB CEX | - | Sponsor |
| BitMart Perpetual | `bitmart_perpetual` | CLOB CEX | - | Sponsor |
| Bybit Perpetual | `bybit_perpetual` | CLOB CEX | Yes (`bybit_perpetual_testnet`) | - |
| Gate.io Perpetual | `gate_io_perpetual` | CLOB CEX | - | Sponsor - 20% fee discount |
| KuCoin Perpetual | `kucoin_perpetual` | CLOB CEX | - | Sponsor |
| OKX Perpetual | `okx_perpetual` | CLOB CEX | Yes (`okx_perpetual_demo`) | Sponsor - 20% fee discount |

### CLOB DEX Perpetual

| Exchange | Connector ID | Type | Notes |
|----------|--------------|------|-------|
| Derive Perpetual | `derive_perpetual` | CLOB DEX | Sponsor |
| dYdX v4 Perpetual | `dydx_v4_perpetual` | CLOB DEX | - |
| Hyperliquid Perpetual | `hyperliquid_perpetual` | CLOB DEX | - |
| Injective Perpetual | `injective_v2_perpetual` | CLOB DEX | - |

---

## 1x Leverage Support (Islamic Finance)

**Important for Islamic Finance / Sharia-compliant trading**: All perpetual connectors in Hummingbot support **1x leverage** by default.

### Implementation Details

The default leverage in Hummingbot is set to **1x** across all perpetual connectors:

```python
# From /hummingbot/connector/perpetual_trading.py
self._leverage: Dict[str, int] = defaultdict(lambda: 1)  # Default is 1x

def set_leverage(self, trading_pair: str, leverage: int = 1):
    """
    Sets leverage level, e.g. 2x, 10x, etc..
    """
    self._leverage[trading_pair] = leverage
```

### Exchanges with Verified 1x Leverage Support

| Exchange | Connector ID | 1x Leverage Support | Leverage Range |
|----------|--------------|---------------------|----------------|
| Binance Perpetual | `binance_perpetual` | Yes | 1x - 125x |
| Bybit Perpetual | `bybit_perpetual` | Yes | 1x - 100x |
| OKX Perpetual | `okx_perpetual` | Yes | 1x - 125x |
| Gate.io Perpetual | `gate_io_perpetual` | Yes | 1x - 100x |
| Bitget Perpetual | `bitget_perpetual` | Yes | 1x - 125x |
| KuCoin Perpetual | `kucoin_perpetual` | Yes | 1x - 100x |
| BitMart Perpetual | `bitmart_perpetual` | Yes | 1x - 100x |
| Hyperliquid Perpetual | `hyperliquid_perpetual` | Yes | Variable |
| dYdX v4 Perpetual | `dydx_v4_perpetual` | Yes | Variable (margin-based) |
| Derive Perpetual | `derive_perpetual` | Yes | Variable |
| Injective Perpetual | `injective_v2_perpetual` | Yes | Variable |

### Setting 1x Leverage

To ensure 1x leverage in your strategy:

```python
# In your strategy configuration
connector.set_leverage(trading_pair="BTC-USDT", leverage=1)
```

The leverage setting is handled via the `_set_trading_pair_leverage` method in each connector, which calls the exchange's API to set the leverage for a specific trading pair.

---

## Paper Trading / Testnet Support

### Built-in Paper Trade Connector

Hummingbot includes a **Paper Trade Exchange** (`paper_trade`) that simulates trading without real orders:

- **Location**: `/hummingbot/connector/exchange/paper_trade/`
- **Features**:
  - Simulates order execution with configurable delay (5 seconds default)
  - Uses real order book data from any connected exchange
  - Supports both market and limit orders
  - Calculates realistic fees based on exchange fee schedules
  - No API keys required

**Usage**:
```python
# Configure paper trading in your strategy
connector_name: str = "paper_trade"
```

### Exchange Testnets

The following exchanges provide testnet/demo environments:

| Exchange | Connector ID | Testnet Domain | Testnet URL |
|----------|--------------|----------------|-------------|
| Binance Perpetual | `binance_perpetual` | `binance_perpetual_testnet` | `https://testnet.binancefuture.com/fapi/` |
| Bybit Perpetual | `bybit_perpetual` | `bybit_perpetual_testnet` | `https://api-testnet.bybit.com/` |
| OKX Perpetual | `okx_perpetual` | `okx_perpetual_demo` | `https://www.okx.com` (demo mode) |

### Using Testnets

To use a testnet, configure the `domain` parameter when initializing the connector:

```python
# Binance Perpetual Testnet
connector = BinancePerpetualDerivative(
    binance_perpetual_api_key="your_testnet_api_key",
    binance_perpetual_api_secret="your_testnet_secret",
    domain="binance_perpetual_testnet"
)

# Bybit Perpetual Testnet
connector = BybitPerpetualDerivative(
    bybit_perpetual_api_key="your_testnet_api_key",
    bybit_perpetual_secret_key="your_testnet_secret",
    domain="bybit_perpetual_testnet"
)

# OKX Demo Trading
connector = OKXPerpetualDerivative(
    okx_perpetual_api_key="your_demo_api_key",
    okx_perpetual_secret_key="your_demo_secret",
    okx_perpetual_passphrase="your_passphrase",
    domain="okx_perpetual_demo"
)
```

---

## How to Add a New Exchange Connector

### Overview

Adding a new exchange connector requires creating several components that implement Hummingbot's connector interfaces. The process varies slightly between spot and perpetual connectors.

### Directory Structure

```
hummingbot/connector/
  exchange/                      # For spot connectors
    <exchange_name>/
      __init__.py
      <exchange_name>_exchange.py        # Main connector class
      <exchange_name>_auth.py            # Authentication
      <exchange_name>_api_order_book_data_source.py
      <exchange_name>_api_user_stream_data_source.py
      <exchange_name>_constants.py       # API URLs, rate limits
      <exchange_name>_utils.py           # Utility functions
      <exchange_name>_web_utils.py       # Web request helpers

  derivative/                    # For perpetual connectors
    <exchange_name>_perpetual/
      __init__.py
      <exchange_name>_perpetual_derivative.py
      <exchange_name>_perpetual_auth.py
      <exchange_name>_perpetual_api_order_book_data_source.py
      <exchange_name>_perpetual_user_stream_data_source.py
      <exchange_name>_perpetual_constants.py
      <exchange_name>_perpetual_utils.py
      <exchange_name>_perpetual_web_utils.py
```

### Step-by-Step Process

#### 1. Study an Existing Connector

The best way to learn is by studying existing implementations:

- **Spot**: `/hummingbot/connector/exchange/binance/`
- **Perpetual**: `/hummingbot/connector/derivative/binance_perpetual/`

#### 2. Create the Constants File

Define all API endpoints, rate limits, and constants:

```python
# <exchange_name>_constants.py
from hummingbot.core.api_throttler.data_types import RateLimit
from hummingbot.core.data_type.in_flight_order import OrderState

EXCHANGE_NAME = "your_exchange"
DEFAULT_DOMAIN = "your_exchange_main"
TESTNET_DOMAIN = "your_exchange_testnet"  # If applicable

REST_URLS = {
    DEFAULT_DOMAIN: "https://api.yourexchange.com",
    TESTNET_DOMAIN: "https://testnet.yourexchange.com"
}

WS_URLS = {
    DEFAULT_DOMAIN: "wss://ws.yourexchange.com",
    TESTNET_DOMAIN: "wss://testnet-ws.yourexchange.com"
}

# API Endpoints
ORDER_URL = "v1/order"
ACCOUNT_INFO_URL = "v1/account"
# ... other endpoints

# Rate Limits
RATE_LIMITS = [
    RateLimit(limit_id="default", limit=1200, time_interval=60),
    # ... other rate limits
]

# Order States
ORDER_STATE = {
    "NEW": OrderState.OPEN,
    "FILLED": OrderState.FILLED,
    # ... other states
}
```

#### 3. Implement Authentication

```python
# <exchange_name>_auth.py
import hmac
import hashlib
from hummingbot.core.web_assistant.auth import AuthBase

class YourExchangeAuth(AuthBase):
    def __init__(self, api_key: str, secret_key: str, time_synchronizer):
        self.api_key = api_key
        self.secret_key = secret_key
        self.time_synchronizer = time_synchronizer

    async def rest_authenticate(self, request):
        # Add authentication headers/params to REST requests
        headers = {}
        headers["X-API-KEY"] = self.api_key
        # Add signature, timestamp, etc.
        request.headers = {**request.headers, **headers}
        return request

    async def ws_authenticate(self, request):
        # Add authentication for WebSocket connections
        return request
```

#### 4. Implement the Order Book Data Source

```python
# <exchange_name>_api_order_book_data_source.py
from hummingbot.core.data_type.order_book_tracker_data_source import OrderBookTrackerDataSource

class YourExchangeAPIOrderBookDataSource(OrderBookTrackerDataSource):
    async def get_new_order_book(self, trading_pair: str):
        # Fetch and return order book snapshot
        pass

    async def listen_for_subscriptions(self):
        # Subscribe to order book updates via WebSocket
        pass

    async def listen_for_trades(self):
        # Listen for trade updates
        pass
```

#### 5. Implement the User Stream Data Source (for trading)

```python
# <exchange_name>_api_user_stream_data_source.py
from hummingbot.core.data_type.user_stream_tracker_data_source import UserStreamTrackerDataSource

class YourExchangeUserStreamDataSource(UserStreamTrackerDataSource):
    async def listen_for_user_stream(self):
        # Listen for order updates, balance changes, etc.
        pass
```

#### 6. Implement the Main Connector Class

For **Spot Connectors**, inherit from `ExchangePyBase`:

```python
# <exchange_name>_exchange.py
from hummingbot.connector.exchange_py_base import ExchangePyBase

class YourExchangeExchange(ExchangePyBase):
    def __init__(self, ...):
        super().__init__(...)

    @property
    def name(self) -> str:
        return "your_exchange"

    def _create_web_assistants_factory(self):
        # Return WebAssistantsFactory with auth
        pass

    def _create_order_book_data_source(self):
        return YourExchangeAPIOrderBookDataSource(...)

    def _create_user_stream_data_source(self):
        return YourExchangeUserStreamDataSource(...)

    async def _place_order(self, order_id, trading_pair, amount, trade_type, order_type, price, **kwargs):
        # Submit order to exchange
        pass

    async def _place_cancel(self, order_id, tracked_order):
        # Cancel order
        pass

    async def _update_balances(self):
        # Fetch and update account balances
        pass

    async def _update_order_status(self):
        # Update status of in-flight orders
        pass

    def _get_fee(self, base_currency, quote_currency, order_type, order_side, amount, price, **kwargs):
        # Calculate trading fees
        pass
```

For **Perpetual Connectors**, inherit from `PerpetualDerivativePyBase`:

```python
# <exchange_name>_perpetual_derivative.py
from hummingbot.connector.perpetual_derivative_py_base import PerpetualDerivativePyBase

class YourExchangePerpetualDerivative(PerpetualDerivativePyBase):
    # Same as spot, plus:

    def supported_position_modes(self) -> List[PositionMode]:
        return [PositionMode.ONEWAY, PositionMode.HEDGE]

    async def _set_trading_pair_leverage(self, trading_pair: str, leverage: int) -> Tuple[bool, str]:
        # Set leverage for trading pair
        pass

    async def _update_positions(self):
        # Update position information
        pass

    async def _trading_pair_position_mode_set(self, mode: PositionMode, trading_pair: str) -> Tuple[bool, str]:
        # Set position mode (hedge/one-way)
        pass

    async def _fetch_last_fee_payment(self, trading_pair: str) -> Tuple[float, Decimal, Decimal]:
        # Fetch funding fee information
        pass

    def get_buy_collateral_token(self, trading_pair: str) -> str:
        # Return collateral token for buy orders
        pass

    def get_sell_collateral_token(self, trading_pair: str) -> str:
        # Return collateral token for sell orders
        pass
```

#### 7. Register the Connector

Add your connector to the connector manager and settings files.

#### 8. Write Tests

Create tests in `/test/hummingbot/connector/exchange/<exchange_name>/` or `/test/hummingbot/connector/derivative/<exchange_name>_perpetual/`:

```python
# test_<exchange_name>_exchange.py
import unittest
from unittest.mock import MagicMock, AsyncMock

class TestYourExchangeExchange(unittest.TestCase):
    def test_name(self):
        connector = YourExchangeExchange(...)
        self.assertEqual(connector.name, "your_exchange")

    # Add more tests for each method
```

### Testing Your Connector

1. **Unit Tests**: Test individual methods in isolation
2. **Integration Tests**: Test with exchange testnet (if available)
3. **Paper Trade**: Test with Hummingbot's paper trade mode first
4. **Small Real Trades**: Start with minimal amounts on mainnet

### Resources

- [Hummingbot Developer Docs](https://hummingbot.org/developers/)
- [Connector Development Guide](https://hummingbot.org/developers/connectors/)
- [Example Connectors in Codebase](https://github.com/hummingbot/hummingbot/tree/master/hummingbot/connector)

### Proposal Process

To have your connector merged into the official Hummingbot codebase:

1. Submit a **New Connector Proposal** via Hummingbot governance
2. Hold HBOT tokens in your Ethereum wallet for voting
3. Get community approval
4. Submit a Pull Request following the [contribution guidelines](https://hummingbot.org/about/proposals/)

---

## Summary Tables

### Quick Reference: All Connectors by Type

| Category | Count | Examples |
|----------|-------|----------|
| CLOB CEX Spot | 18 | binance, bybit, kucoin, okx |
| CLOB CEX Perpetual | 7 | binance_perpetual, bybit_perpetual |
| CLOB DEX Spot | 6 | derive, hyperliquid, injective_v2 |
| CLOB DEX Perpetual | 5 | dydx_v4_perpetual, hyperliquid_perpetual |
| AMM DEX (Gateway) | 11 | uniswap, jupiter, raydium |
| **Total** | **47+** | - |

### Quick Reference: 1x Leverage & Testnet Support

| Exchange | 1x Leverage | Testnet | Paper Trade |
|----------|-------------|---------|-------------|
| Binance | Yes | Yes | Yes (built-in) |
| Bybit | Yes | Yes | Yes (built-in) |
| OKX | Yes | Yes (demo) | Yes (built-in) |
| Gate.io | Yes | - | Yes (built-in) |
| Bitget | Yes | - | Yes (built-in) |
| KuCoin | Yes | - | Yes (built-in) |
| BitMart | Yes | - | Yes (built-in) |
| Hyperliquid | Yes | - | Yes (built-in) |
| dYdX | Yes | - | Yes (built-in) |
| Derive | Yes | - | Yes (built-in) |
| Injective | Yes | - | Yes (built-in) |

---

*Last Updated: February 2026*
*Source: Hummingbot Repository Analysis*
