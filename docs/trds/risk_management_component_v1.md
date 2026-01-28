# Technical Requirements Document: Risk Management Component v1.0

**Project**: QuantMindX
**Component**: Risk Management Service (RiskMind)
**Version**: 1.0
**Status**: Draft
**Created**: 2026-01-27

---

## Executive Summary

**Purpose**: Build a centralized, Python-based Risk Management Component for QuantMindX that enforces risk limits across 200+ trading bots, supports both prop firm and personal account modes, and provides real-time monitoring via WebSocket to the QuantMind IDE.

**Key Goals**:
1. Prevent prop firm challenge failures (5% daily drawdown limit)
2. Enable multi-bot portfolio risk management
3. Provide real-time per-bot risk monitoring in QuantMind IDE
4. Support dynamic position sizing based on account size
5. Scalable to 200+ concurrent bots

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Component Design](#2-component-design)
3. [Risk Management Logic](#3-risk-management-logic)
4. [Bot Identification & Tagging](#4-bot-identification--tagging)
5. [API Specification](#5-api-specification)
6. [WebSocket Protocol](#6-websocket-protocol)
7. [Integration Points](#7-integration-points)
8. [Database Schema](#8-database-schema)
9. [Configuration](#9-configuration)
10. [Implementation Phases](#10-implementation-phases)

---

## 1. Architecture Overview

### 1.1 System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        QUANTMINDX RISK ARCHITECTURE                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐  │
│  │  Trading Bots    │      │  RiskMind        │      │  QuantMind IDE   │  │
│  │  (MT5 EAs)       │─────│  (Python Service) │─────│  (Dashboard)      │  │
│  │                  │ API  │                  │ WS   │                  │  │
│  │ • Analyst Agent  │      │ • Risk Engine    │      │ • Per-bot Stats  │  │
│  │ • QuantCode      │      │ • Position Sizing│      │ • Traffic Lights │  │
│  │ • Manual Bots    │      │ • Limit Checker  │      │ • Real-time P&L  │  │
│  │ • Paper Trading  │      │ • Event Logger   │      │ • Risk Alerts    │  │
│  └──────────────────┘      └──────────────────┘      └──────────────────┘  │
│                                      │                                     │
│                              ┌───────┴───────┐                             │
│                              ▼               ▼                             │
│                       ┌─────────────┐  ┌─────────────┐                      │
│                       │ PostgreSQL  │  │  Redis      │                      │
│                       │ (Persistence)│  │ (Cache)     │                      │
│                       └─────────────┘  └─────────────┘                      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Technology Stack

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| **Risk Service** | Python 3.11+ | Better logic structure, async I/O, ML-ready |
| **MT5 Integration** | MetaTrader5 Python API | Official library, bidirectional communication |
| **API Framework** | FastAPI | Async, auto-docs, WebSocket support |
| **Database** | PostgreSQL 15+ | ACID compliance, JSONB for flexible schemas |
| **Cache Layer** | Redis 7+ | Sub-ms latency for real-time checks |
| **WebSocket** | FastAPI WebSockets | Built-in, efficient streaming |
| **ORM** | SQLAlchemy 2.0 | Type-safe, async support |
| **Validation** | Pydantic v2 | Request/response validation |

### 1.3 Deployment Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    VPS (MT5 Terminal)                           │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  MT5 Terminal                                            │   │
│  │  ├─ Account 1: Prop Firm ($100k)                        │   │
│  │  ├─ Account 2: Personal ($50k)                          │   │
│  │  └─ Account 3: Paper Trading ($100k demo)               │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              ▲                                  │
│                              │ localhost:8000                   │
│  ┌──────────────────────────────┴──────────────────────────┐   │
│  │  RiskMind Service (Python)                              │   │
│  │  ├─ API Server (FastAPI)                                │   │
│  │  ├─ Risk Engine (async)                                 │   │
│  │  ├─ WebSocket Manager                                   │   │
│  │  └─ Scheduler (daily resets)                            │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              ▲                                  │
│                              │ WebSocket                        │
│  ┌──────────────────────────────┴──────────────────────────┐   │
│  │  QuantMind IDE (Remote/Local)                           │   │
│  │  └─ Risk Dashboard (real-time updates)                  │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Component Design

### 2.1 RiskMind Service Structure

```
riskmind/
├── __init__.py
├── main.py                    # FastAPI application entry point
├── config.py                  # Configuration management
├── database/
│   ├── __init__.py
│   ├── models.py              # SQLAlchemy ORM models
│   ├── crud.py                # Database operations
│   └── session.py             # Database session management
├── core/
│   ├── __init__.py
│   ├── risk_engine.py         # Core risk calculation logic
│   ├── position_sizer.py      # Dynamic position sizing
│   ├── limit_checker.py       # Pre/post trade risk validation
│   ├── event_logger.py        # Risk event logging
│   └── account_tracker.py     # Account state tracking
├── api/
│   ├── __init__.py
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── trades.py          # Trade approval endpoints
│   │   ├── accounts.py        # Account state endpoints
│   │   ├── bots.py            # Bot management endpoints
│   │   └── config.py          # Configuration endpoints
│   └── websocket/
│       ├── __init__.py
│       └── manager.py         # WebSocket connection manager
├── mt5/
│   ├── __init__.py
│   ├── connector.py           # MT5 connection management
│   ├── position_tracker.py    # Real-time position monitoring
│   └── account_info.py        # Account data fetching
├── bot_registry/
│   ├── __init__.py
│   ├── registry.py            # Bot ID registry
│   ├── tags.py                # Tag parsing and management
│   └── assignment.py          # Magic number assignment
└── utils/
    ├── __init__.py
    ├── calculations.py        # Risk calculation helpers
    └── logging.py             # Structured logging
```

### 2.2 Core Data Models

#### AccountMode (Enum)
```python
class AccountMode(str, Enum):
    """Risk management mode."""
    PROP_FIRM = "prop_firm"       # Prop firm challenge rules
    PERSONAL = "personal"         # Personal account rules
    PAPER_TRADING = "paper"       # Paper trading backtest mode
```

#### RiskProfile (Dataclass)
```python
@dataclass
class RiskProfile:
    """Risk parameters for an account."""
    mode: AccountMode
    account_size: float
    daily_loss_limit_pct: float
    total_drawdown_limit_pct: float
    max_risk_per_trade_pct: float
    max_daily_risk_pct: float
    active_capital_ratio: float = 0.5
```

#### BotIdentity (Dataclass)
```python
@dataclass
class BotIdentity:
    """Bot identification and classification."""
    bot_id: str                          # e.g., "analyst-bot-007"
    magic_number: int                    # MT5 magic number
    guild: GuildType                     # ANALYST, QUANTCODE, MANUAL, PAPER
    tags: List[str]                      # e.g., ["@perfect", "@scalping"]
    strategy_type: str                   # e.g., "RSI_SCALPER"
    created_at: datetime
    status: BotStatus                    # ACTIVE, PAUSED, QUARANTINE
```

#### TradeRequest (Pydantic Model)
```python
class TradeRequest(BaseModel):
    """Request to open a trade - sent by EA."""
    bot_id: str
    symbol: str
    direction: Literal["BUY", "SELL"]
    entry_price: float
    stop_loss: float
    take_profit: Optional[float]
    requested_lots: float
    session_token: str                  # For authentication
    metadata: Dict[str, Any] = {}
```

#### TradeResponse (Pydantic Model)
```python
class TradeResponse(BaseModel):
    """Response to trade request."""
    approved: bool
    adjusted_lots: float               # Risk-adjusted position size
    rejection_reason: Optional[str]
    risk_amount: float                  # Actual risk in currency
    risk_pct: float                     # Risk as % of account
    warnings: List[str] = []
```

---

## 3. Risk Management Logic

### 3.1 Dynamic Risk Calculation

The component adjusts risk per trade based on account size:

```python
def calculate_risk_parameters(account_size: float) -> RiskProfile:
    """
    Calculate risk parameters based on account size.

    Account Size Risk Matrix:
    +------------------+-------------------+------------------+
    | Account Size     | Risk Per Trade    | Max Daily Risk   |
    +------------------+-------------------+------------------+
    | > $100,000       | 0.25% - 0.5%      | 2.5%             |
    | $10,000 - $100k  | 0.5% - 1.0%       | 3.0%             |
    | < $10,000        | 1.0%              | 5.0%             |
    +------------------+-------------------+------------------+
    """
    if account_size >= 100000:
        risk_per_trade = 0.005  # 0.5%
        daily_risk = 0.025      # 2.5%
    elif account_size >= 10000:
        risk_per_trade = 0.01   # 1.0%
        daily_risk = 0.03       # 3.0%
    else:
        risk_per_trade = 0.01   # 1.0%
        daily_risk = 0.05       # 5.0%

    return RiskProfile(
        mode=AccountMode.PERSONAL,  # Will be set based on context
        account_size=account_size,
        daily_loss_limit_pct=daily_risk * 100,
        total_drawdown_limit_pct=5.0,  # Hard 5% limit
        max_risk_per_trade_pct=risk_per_trade * 100,
        max_daily_risk_pct=daily_risk * 100,
    )
```

### 3.2 Position Sizing Formula

```python
def calculate_position_size(
    account_balance: float,
    risk_amount: float,           # In account currency
    stop_loss_points: float,      # Distance in points
    tick_value: float,            # Per lot
    point_value: float,           # Value of one point
) -> float:
    """
    Calculate position size based on risk amount.

    Formula:
    ---------
    Risk = Volume × StopLoss_Points × Tick_Value × Point_Value
    Volume = Risk / (StopLoss_Points × Tick_Value × Point_Value)

    Example:
    --------
    Account: $10,000
    Risk: 1% = $100
    Stop Loss: 200 points
    Tick Value: $1 per lot per point
    Point Value: 0.0001 (for EURUSD)

    Volume = 100 / (200 × 1 × 0.0001)
           = 100 / 0.02
           = 5.0 lots (capped at 1.0 lot for safety)
    """
    if stop_loss_points <= 0:
        raise ValueError("Stop loss must be positive")

    # Calculate raw position size
    position_value = stop_loss_points * tick_value * point_value
    raw_volume = risk_amount / position_value

    # Apply safety cap: Never risk more than 1 lot per $10k
    max_safe_lots = (account_balance / 10000) * 1.0
    adjusted_volume = min(raw_volume, max_safe_lots)

    # Round to 2 decimal places
    return round(adjusted_volume, 2)
```

### 3.3 Pre-Trade Risk Checklist

Before approving any trade, the component checks:

```python
async def pre_trade_check(request: TradeRequest) -> TradeResponse:
    """
    Comprehensive pre-trade risk validation.
    """
    checks = []

    # 1. Bot Status Check
    bot = await registry.get_bot(request.bot_id)
    if bot.status != BotStatus.ACTIVE:
        return TradeResponse(
            approved=False,
            rejection_reason=f"Bot {request.bot_id} is {bot.status.value}",
            adjusted_lots=0,
        )
    checks.append("bot_active")

    # 2. Account Mode Check
    account = await mt5.get_account_info()
    if account.mode == AccountMode.PROP_FIRM:
        # Prop firm has stricter rules
        checks.extend(await _check_prop_firm_limits(account))

    # 3. Daily Loss Limit Check
    daily_loss_pct = await _get_daily_loss_pct(account.id)
    if daily_loss_pct >= account.risk_profile.daily_loss_limit_pct:
        return TradeResponse(
            approved=False,
            rejection_reason=f"Daily loss limit exceeded: {daily_loss_pct:.2f}%",
            adjusted_lots=0,
        )
    checks.append("daily_limit_ok")

    # 4. Total Drawdown Check
    total_dd_pct = await _get_total_drawdown_pct(account.id)
    if total_dd_pct >= account.risk_profile.total_drawdown_limit_pct:
        return TradeResponse(
            approved=False,
            rejection_reason=f"Total drawdown exceeded: {total_dd_pct:.2f}%",
            adjusted_lots=0,
        )
    checks.append("drawdown_ok")

    # 5. Calculate Risk Amount
    sl_distance = abs(request.entry_price - request.stop_loss)
    risk_pct = account.risk_profile.max_risk_per_trade_pct
    risk_amount = account.balance * (risk_pct / 100)

    # 6. Adjust Position Size
    tick_value = await mt5.get_tick_value(request.symbol)
    point_value = await mt5.get_point_value(request.symbol)
    sl_points = sl_distance / point_value

    adjusted_lots = calculate_position_size(
        account_balance=account.balance,
        risk_amount=risk_amount,
        stop_loss_points=sl_points,
        tick_value=tick_value,
        point_value=point_value,
    )

    # 7. Spread Check (for scalping)
    if any("@scalping" in tag for tag in bot.tags):
        current_spread = await mt5.get_spread(request.symbol)
        max_allowed_spread = sl_points * 0.5  # Spread must be < 50% of SL
        if current_spread > max_allowed_spread:
            return TradeResponse(
                approved=False,
                rejection_reason=f"Spread too wide: {current_spread} points",
                adjusted_lots=0,
            )
    checks.append("spread_ok")

    # 8. Return Approval
    return TradeResponse(
        approved=True,
        adjusted_lots=adjusted_lots,
        risk_amount=risk_amount,
        risk_pct=risk_pct,
        warnings=[],
    )
```

### 3.4 Post-Trade Slippage Check

```python
async def post_trade_check(position: Position) -> bool:
    """
    Check if slippage caused unacceptable risk.

    Called after position is opened to verify actual risk.
    """
    # Re-calculate actual risk with real entry price
    actual_sl_distance = abs(position.entry_price - position.stop_loss)
    actual_risk = actual_sl_distance * position.volume * position.tick_value

    # Get planned risk from trade request
    planned_risk = await _get_planned_risk(position.ticket)

    # If actual risk > 2x planned risk, close position
    if actual_risk > planned_risk * 2.0:
        await mt5.close_position(position.ticket)
        await event_logger.log(
            event_type=RiskEventType.SLIPPAGE_EXCEEDED,
            level=RiskLevel.CRITICAL,
            message=f"Closed position {position.ticket} due to slippage: "
                    f"${actual_risk:.2f} vs ${planned_risk:.2f} planned",
        )
        return False

    return True
```

---

## 4. Bot Identification & Tagging

### 4.1 Hybrid Identification System

Combines **Central Registry** + **Comment Field Tags**:

```python
# Central Registry Entry
{
    "bot_id": "analyst-bot-007",
    "magic_number": 10007,
    "guild": "ANALYST",
    "tags": ["@perfect", "@scalping", "@london-session"],
    "strategy_type": "RSI_OVERSOLD_SCALPER",
    "created_at": "2026-01-27T10:00:00Z",
    "status": "ACTIVE",
    "assigned_account_id": "account_001"  # Which account it trades
}

# Comment Field Tag (set by EA in MT5)
// Position comment: "@analyst-bot-007 @scalping TP:1.5% SL:0.8%"
```

### 4.2 Magic Number Ranges

| Guild | Range | Example |
|-------|-------|---------|
| Analyst Agent | 10000 - 19999 | 10007 |
| QuantCode | 20000 - 29999 | 20042 |
| Manual Trading | 30000 - 39999 | 30100 |
| Paper Trading | 90000 - 99999 | 90001 |

### 4.3 Bot Registry API Endpoints

```
POST   /api/v1/bots/register          # Register new bot
GET    /api/v1/bots/{bot_id}          # Get bot details
PUT    /api/v1/bots/{bot_id}/status   # Update bot status
GET    /api/v1/bots                   # List all bots
DELETE /api/v1/bots/{bot_id}          # Unregister bot
```

---

## 5. API Specification

### 5.1 Trade Approval Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRADE APPROVAL FLOW                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  MT5 Expert Advisor                     RiskMind Service        │
│  ┌──────────────────────┐              ┌──────────────────────┐ │
│  │ 1. OnTick() detected │              │                      │ │
│  │    entry signal      │              │                      │ │
│  └──────────┬───────────┘              │                      │ │
│             │                          │                      │ │
│             ▼                          │                      │ │
│  ┌──────────────────────┐              │                      │ │
│  │ 2. Call API:         │─────────────│► 3. Validate risk   │ │
│  │    POST /trades/approve            │    - Bot status      │ │
│  │    {                 │              │    - Daily limits    │ │
│  │      bot_id,         │              │    - Drawdown       │ │
│  │      symbol,         │              │    - Position size  │ │
│  │      entry,          │              │                      │ │
│  │      sl, tp          │              │                      │ │
│  │    }                 │              │                      │ │
│  └──────────┬───────────┘              │                      │ │
│             │                          │                      │ │
│             │◄─────────────────────────│ 4. Response:        │ │
│             │                          │    {                 │ │
│             │                          │      approved,       │ │
│             │                          │      adjusted_lots,  │ │
│             │                          │      risk_amount     │ │
│             │                          │    }                 │ │
│             ▼                          │                      │ │
│  ┌──────────────────────┐              │                      │ │
│  │ 5a. IF approved:     │              │                      │ │
│  │     Open trade with  │              │                      │ │
│  │     adjusted_lots    │              │                      │ │
│  │ 5b. ELSE:            │              │                      │ │
│  │     Skip trade       │              │                      │ │
│  └──────────────────────┘              └──────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 Core API Endpoints

#### Trade Approval

```http
POST /api/v1/trades/approve
Content-Type: application/json
Authorization: Bearer {session_token}

{
    "bot_id": "analyst-bot-007",
    "symbol": "EURUSD",
    "direction": "BUY",
    "entry_price": 1.08567,
    "stop_loss": 1.08423,
    "take_profit": 1.08732,
    "requested_lots": 0.1,
    "metadata": {
        "strategy": "RSI_oversold_28",
        "confidence": 0.85
    }
}
```

**Response:**
```http
HTTP/1.1 200 OK
Content-Type: application/json

{
    "approved": true,
    "adjusted_lots": 0.05,
    "risk_amount": 50.00,
    "risk_pct": 0.5,
    "warnings": [],
    "request_id": "req_abc123"
}
```

#### Account State

```http
GET /api/v1/accounts/{account_id}/state
```

**Response:**
```http
HTTP/1.1 200 OK

{
    "account_id": "account_001",
    "mode": "prop_firm",
    "balance": 100000.00,
    "equity": 99750.00,
    "daily_pnl": -250.00,
    "daily_loss_pct": 0.25,
    "daily_loss_limit_pct": 5.0,
    "total_drawdown_pct": 0.25,
    "total_drawdown_limit_pct": 5.0,
    "open_positions": 3,
    "trading_allowed": true,
    "risk_level": "SAFE"
}
```

#### Bot Performance

```http
GET /api/v1/bots/{bot_id}/performance?period=today
```

**Response:**
```http
HTTP/1.1 200 OK

{
    "bot_id": "analyst-bot-007",
    "period": "today",
    "trades": 7,
    "wins": 5,
    "losses": 2,
    "win_rate": 0.714,
    "total_pnl": 87.50,
    "total_risked": 150.00,
    "risk_reward_ratio": 0.58,
    "max_drawdown": 45.00,
    "current_positions": [
        {
            "ticket": 123456,
            "symbol": "EURUSD",
            "volume": 0.05,
            "unrealized_pnl": 12.30
        }
    ]
}
```

---

## 6. WebSocket Protocol

### 6.1 Connection

```javascript
// QuantMind IDE connects to:
ws://riskmind-server:8000/ws/dashboard?token={auth_token}

// Subscribe to updates
{
    "action": "subscribe",
    "channels": [
        "account_updates",     // Account state changes
        "bot_updates",         // Bot performance
        "risk_events",         // Risk alerts
        "trade_updates"        // Trade executions
    ]
}
```

### 6.2 Real-Time Account Updates

```json
{
    "channel": "account_updates",
    "data": {
        "account_id": "account_001",
        "timestamp": "2026-01-27T14:30:22Z",
        "balance": 100000.00,
        "equity": 99750.00,
        "daily_pnl": -250.00,
        "daily_loss_pct": 0.25,
        "daily_limit_pct": 5.0,
        "total_drawdown_pct": 0.25,
        "total_limit_pct": 5.0,
        "risk_level": "SAFE",
        "open_positions": 3,
        "trading_allowed": true
    }
}
```

### 6.3 Bot Performance Update

```json
{
    "channel": "bot_updates",
    "data": {
        "bot_id": "analyst-bot-007",
        "timestamp": "2026-01-27T14:30:22Z",
        "today_trades": 7,
        "today_wins": 5,
        "today_pnl": 87.50,
        "today_risked": 150.00,
        "current_positions": 1,
        "unrealized_pnl": 12.30,
        "status": "ACTIVE",
        "tags": ["@perfect", "@scalping"]
    }
}
```

### 6.4 Risk Event Alert

```json
{
    "channel": "risk_events",
    "data": {
        "severity": "WARNING",
        "event_type": "daily_limit_approaching",
        "message": "Daily loss limit at 75% (3.75% / 5.0%)",
        "account_id": "account_001",
        "timestamp": "2026-01-27T14:30:22Z",
        "metrics": {
            "current_loss_pct": 3.75,
            "limit_pct": 5.0,
            "remaining_pct": 1.25
        }
    }
}
```

---

## 7. Integration Points

### 7.1 MT5 Expert Advisor Integration

```mql5
// RiskMind.mqh - Include file for MT5 EAs
#include <Trade/Trade.mqh>
#include <Trade/PositionInfo.mqh>

class CRiskMindClient
{
private:
    string m_server_url;
    string m_session_token;
    string m_bot_id;
    int m_magic_number;

    // HTTP request helper
    string SendPostRequest(string endpoint, string json_data);

public:
    // Initialize with bot credentials
    void Init(string bot_id, int magic, string server = "http://localhost:8000");

    // Request trade approval
    bool RequestTradeApproval(
        string symbol,
        ENUM_POSITION_TYPE direction,
        double entry_price,
        double sl,
        double tp,
        double requested_lots,
        double &approved_lots,
        string &error_msg
    );

    // Report trade execution
    void ReportTradeOpened(ulong ticket, double actual_lots, double actual_entry);
};

// Usage in EA
void OnTick()
{
    // ... signal detection logic ...

    if(signal_found)
    {
        double approved_lots = 0;
        string error = "";

        if(risk_client.RequestTradeApproval(
            _Symbol, POSITION_TYPE_BUY, Ask, sl, tp, 0.1,
            approved_lots, error))
        {
            trade.Buy(approved_lots, _Symbol, Ask, sl, tp);
        }
    }
}
```

### 7.2 MCP Server Integration

The RiskMind service will expose an MCP server for integration with other QuantMindX components:

```python
# mcp_server_riskmind.py
from mcp.server import Server
import httpx

app = Server("riskmind")

@app.tool("check_trade_risk")
async def check_trade_risk(
    bot_id: str,
    symbol: str,
    entry: float,
    sl: float
) -> dict:
    """Check if a trade meets risk criteria."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/api/v1/trades/approve",
            json={"bot_id": bot_id, "symbol": symbol, ...}
        )
        return response.json()

@app.tool("get_bot_performance")
async def get_bot_performance(bot_id: str) -> dict:
    """Get performance metrics for a bot."""
    # ... implementation ...

@app.tool("list_active_bots")
async def list_active_bots() -> list:
    """List all active trading bots."""
    # ... implementation ...
```

### 7.3 QuantMind IDE Dashboard Integration

```typescript
// RiskDashboard.tsx
import { useWebSocket } from '@hooks/useWebSocket';

export function RiskDashboard() {
    const { data, isConnected } = useWebSocket('ws://localhost:8000/ws/dashboard', {
        onMessage: (event) => {
            const update = JSON.parse(event.data);
            switch(update.channel) {
                case 'account_updates':
                    updateAccountState(update.data);
                    break;
                case 'bot_updates':
                    updateBotMetrics(update.data);
                    break;
                case 'risk_events':
                    showRiskAlert(update.data);
                    break;
            }
        }
    });

    return (
        <DashboardLayout>
            <TrafficLightWidget level={data?.risk_level} />
            <AccountSummary account={data?.account} />
            <BotPerformanceGrid bots={data?.bots} />
            <RiskEventLog events={data?.events} />
        </DashboardLayout>
    );
}
```

---

## 8. Database Schema

### 8.1 PostgreSQL Tables

```sql
-- Accounts table
CREATE TABLE accounts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    mt5_login INTEGER NOT NULL UNIQUE,
    mode VARCHAR(20) NOT NULL,  -- 'prop_firm', 'personal', 'paper'
    initial_balance DECIMAL(15,2) NOT NULL,
    daily_start_balance DECIMAL(15,2),
    daily_start_equity DECIMAL(15,2),
    max_equity_peak DECIMAL(15,2),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Bots registry
CREATE TABLE bots (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    bot_id VARCHAR(100) UNIQUE NOT NULL,
    magic_number INTEGER NOT NULL UNIQUE,
    guild VARCHAR(50) NOT NULL,  -- 'ANALYST', 'QUANTCODE', 'MANUAL', 'PAPER'
    tags JSONB DEFAULT '[]',
    strategy_type VARCHAR(100),
    status VARCHAR(20) NOT NULL DEFAULT 'ACTIVE',
    assigned_account_id UUID REFERENCES accounts(id),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Trades journal
CREATE TABLE trades (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    ticket INTEGER NOT NULL,
    bot_id VARCHAR(100) NOT NULL,
    account_id UUID REFERENCES accounts(id),
    symbol VARCHAR(10) NOT NULL,
    direction VARCHAR(4) NOT NULL,
    entry_price DECIMAL(15,5),
    stop_loss DECIMAL(15,5),
    take_profit DECIMAL(15,5),
    volume DECIMAL(10,2),
    planned_risk DECIMAL(15,2),
    actual_risk DECIMAL(15,2),
    exit_price DECIMAL(15,5),
    pnl DECIMAL(15,2),
    opened_at TIMESTAMPTZ,
    closed_at TIMESTAMPTZ
);

-- Risk events log
CREATE TABLE risk_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    account_id UUID REFERENCES accounts(id),
    bot_id VARCHAR(100),
    event_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    message TEXT,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Daily snapshots
CREATE TABLE daily_snapshots (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    account_id UUID REFERENCES accounts(id),
    snapshot_date DATE NOT NULL,
    start_balance DECIMAL(15,2),
    end_balance DECIMAL(15,2),
    start_equity DECIMAL(15,2),
    end_equity DECIMAL(15,2),
    trades_count INTEGER,
    wins INTEGER,
    losses INTEGER,
    total_pnl DECIMAL(15,2),
    max_drawdown DECIMAL(10,2),
    UNIQUE(account_id, snapshot_date)
);

-- Indexes for performance
CREATE INDEX idx_trades_bot_id ON trades(bot_id);
CREATE INDEX idx_trades_account_id ON trades(account_id);
CREATE INDEX idx_trades_opened_at ON trades(opened_at);
CREATE INDEX idx_risk_events_account_id ON risk_events(account_id);
CREATE INDEX idx_risk_events_created_at ON risk_events(created_at);
CREATE INDEX idx_bots_status ON bots(status);
```

### 8.2 Redis Cache Structure

```
# Account state (TTL: 1 second)
account:{account_id}:state -> JSON {
    balance, equity, daily_pnl, daily_loss_pct, ...
}

# Bot performance (TTL: 5 seconds)
bot:{bot_id}:performance -> JSON {
    trades, wins, pnl, current_positions, ...
}

# Active positions (TTL: 1 second)
positions:{account_id}:active -> JSON [
    {ticket, symbol, volume, pnl, ...},
    ...
]

# Risk limits (TTL: 60 seconds)
limits:{account_id} -> JSON {
    daily_loss_limit, total_dd_limit, risk_per_trade, ...
}

# Daily reset flags (TTL: 24 hours)
reset:{account_id}:{date} -> "completed"
```

---

## 9. Configuration

### 9.1 Environment Variables

```bash
# Server Configuration
RISKMIND_HOST=0.0.0.0
RISKMIND_PORT=8000
RISKMIND_WORKERS=4

# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/riskmind
REDIS_URL=redis://localhost:6379/0

# MT5 Connection
MT5_HOST=127.0.0.1
MT5_PORT=443

# Security
API_SECRET_KEY=your-secret-key-here
SESSION_TTL=3600

# Risk Limits (defaults)
DEFAULT_DAILY_LOSS_LIMIT=5.0
DEFAULT_TOTAL_DRAWDOWN_LIMIT=5.0
DEFAULT_RISK_PER_TRADE=1.0

# Logging
LOG_LEVEL=INFO
LOG_FILE=/var/log/riskmind/riskmind.log

# WebSocket
WS_HEARTBEAT_INTERVAL=30
```

### 9.2 Account-Specific Configuration

```yaml
# config/accounts.yaml
accounts:
  - id: "account_001"
    mt5_login: 12345678
    mode: "prop_firm"
    initial_balance: 100000
    daily_loss_limit_pct: 5.0
    total_drawdown_limit_pct: 10.0
    max_risk_per_trade_pct: 0.5

  - id: "account_002"
    mt5_login: 87654321
    mode: "personal"
    initial_balance: 50000
    daily_loss_limit_pct: 3.0
    total_drawdown_limit_pct: 10.0
    max_risk_per_trade_pct: 1.0
```

---

## 10. Implementation Phases

### Phase 1: Core Risk Engine (Week 1-2)

**Deliverables:**
- [x] RiskMind service scaffold
- [x] Database models and migrations
- [x] Core risk calculation functions
- [x] Position sizing calculator
- [x] Account state tracker

**Testing:**
- Unit tests for risk calculations
- Position sizing formula validation

### Phase 2: API & MT5 Integration (Week 3-4)

**Deliverables:**
- [x] FastAPI endpoints
- [x] MT5 connector service
- [x] Trade approval flow
- [x] Post-trade slippage check
- [x] MT5 include file (.mqh)

**Testing:**
- API integration tests
- MT5 connection tests
- End-to-end trade approval test

### Phase 3: Bot Registry & Tagging (Week 5)

**Deliverables:**
- [x] Bot registry service
- [x] Magic number assignment
- [x] Tag parsing system
- [x] Bot management endpoints

**Testing:**
- Bot registration flow
- Tag parsing validation
- Multi-bot coordination test

### Phase 4: WebSocket & Dashboard (Week 6-7)

**Deliverables:**
- [x] WebSocket manager
- [x] Real-time data streaming
- [x] QuantMind IDE dashboard components
- [x] Traffic light widget
- [x] Per-bot performance cards

**Testing:**
- WebSocket connection tests
- Real-time update latency tests
- Dashboard rendering tests

### Phase 5: MCP Server & External API (Week 8)

**Deliverables:**
- [x] MCP server implementation
- [x] External API endpoints
- [x] Rate limiting and authentication
- [x] API documentation

**Testing:**
- MCP tool execution tests
- API security tests
- Load testing (200 concurrent bots)

### Phase 6: Paper Trading Integration (Week 9)

**Deliverables:**
- [x] Paper trading account mode
- [x] Paper trading bot assignments
- [x] Performance comparison (paper vs live)

**Testing:**
- Paper trading workflow
- Risk limit enforcement on paper accounts

---

## Verification & Testing

### End-to-End Test Scenario

1. **Register a bot** as "analyst-bot-007" with magic 10007
2. **Open 3 trades** that respect risk limits → All approved
3. **Open 4th trade** that would exceed daily limit → Rejected with warning
4. **Verify dashboard** shows traffic light change from GREEN to YELLOW
5. **Wait for daily reset** (server time 00:00) → Limits reset
6. **Open trade with excessive slippage** → Position auto-closed
7. **Verify risk event logged** to database and dashboard

### Load Testing

- Simulate 200 concurrent bots
- Each bot sends 10 trade requests/minute
- Target: <100ms response time, 99.9% uptime

---

## References

1. [Risk Manager for Algorithmic Trading](../../data/scraped_articles/trading/risk_manager_for_algorithmic_trading.md) - Slippage control, spread monitoring
2. [Multi-Currency EA Part 12: Prop Trading Risk Manager](../../data/scraped_articles/trading/developing_a_multi-currency_expert_advisor__part_12___developing_prop_trading_level_risk_manager.md)
3. [Dynamic Position Sizing](../../data/scraped_articles/trading_systems/build_self_optimizing_expert_advisors_in_mql5__part_4___dynamic_position_sizing.md)
4. [MT5 Python Integration](../../data/scraped_articles/trading/metatrader_5_and_python_integration__receiving_and_sending_data.md)

---

**Status**: Ready for Implementation
**Estimated Duration**: 9 weeks
**Priority**: HIGH (Critical for prop firm challenges)
