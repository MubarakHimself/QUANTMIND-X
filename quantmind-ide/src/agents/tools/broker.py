"""
Broker Tools for QuantMind agents.

These tools provide MT5 broker connectivity:
- connect_broker: Connect to MT5 broker
- get_account_info: Get account information
- sync_mt5_data: Sync MT5 data to database
- get_market_data: Get current market data
- get_positions: Get open positions
- get_orders: Get pending orders
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from .base import (
    QuantMindTool,
    ToolCategory,
    ToolError,
    ToolPriority,
    ToolResult,
    register_tool,
)
from .registry import AgentType


logger = logging.getLogger(__name__)


class ConnectionStatus(str, Enum):
    """Broker connection status."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


class AccountType(str, Enum):
    """MT5 account types."""
    DEMO = "demo"
    REAL = "real"
    CONTEST = "contest"


@dataclass
class AccountInfo:
    """MT5 account information."""
    login: int
    name: str
    server: str
    currency: str
    balance: float
    equity: float
    margin: float
    free_margin: float
    margin_level: float
    profit: float
    account_type: AccountType
    leverage: int


@dataclass
class Position:
    """Open position details."""
    ticket: int
    symbol: str
    type: str  # BUY, SELL
    volume: float
    open_price: float
    current_price: float
    sl: float
    tp: float
    profit: float
    swap: float
    commission: float
    open_time: datetime
    comment: str
    magic: int


@dataclass
class MarketData:
    """Market data for a symbol."""
    symbol: str
    bid: float
    ask: float
    spread: float
    last: float
    volume: int
    time: datetime
    high: float
    low: float
    daily_change: float


class ConnectBrokerInput(BaseModel):
    """Input schema for connect_broker tool."""
    login: Optional[int] = Field(
        default=None,
        description="MT5 account login"
    )
    password: Optional[str] = Field(
        default=None,
        description="MT5 account password"
    )
    server: Optional[str] = Field(
        default=None,
        description="MT5 server name"
    )
    terminal_id: Optional[str] = Field(
        default=None,
        description="MT5 terminal ID (uses default if not specified)"
    )
    timeout: int = Field(
        default=30000,
        description="Connection timeout in milliseconds"
    )


class GetAccountInfoInput(BaseModel):
    """Input schema for get_account_info tool."""
    terminal_id: Optional[str] = Field(
        default=None,
        description="MT5 terminal ID (uses default if not specified)"
    )
    include_history: bool = Field(
        default=False,
        description="Include trading history summary"
    )


class SyncMT5DataInput(BaseModel):
    """Input schema for sync_mt5_data tool."""
    data_types: List[str] = Field(
        default=["positions", "orders", "history"],
        description="Types of data to sync"
    )
    terminal_id: Optional[str] = Field(
        default=None,
        description="MT5 terminal ID (uses default if not specified)"
    )
    full_sync: bool = Field(
        default=False,
        description="Perform full sync instead of incremental"
    )


class GetMarketDataInput(BaseModel):
    """Input schema for get_market_data tool."""
    symbols: List[str] = Field(
        description="List of symbols to get data for"
    )
    terminal_id: Optional[str] = Field(
        default=None,
        description="MT5 terminal ID (uses default if not specified)"
    )


class GetPositionsInput(BaseModel):
    """Input schema for get_positions tool."""
    symbol: Optional[str] = Field(
        default=None,
        description="Filter by symbol (all if not specified)"
    )
    terminal_id: Optional[str] = Field(
        default=None,
        description="MT5 terminal ID (uses default if not specified)"
    )


class GetOrdersInput(BaseModel):
    """Input schema for get_orders tool."""
    symbol: Optional[str] = Field(
        default=None,
        description="Filter by symbol (all if not specified)"
    )
    terminal_id: Optional[str] = Field(
        default=None,
        description="MT5 terminal ID (uses default if not specified)"
    )


@register_tool(
    agent_types=[AgentType.COPILOT],
    tags=["broker", "mt5", "connection"],
)
class ConnectBrokerTool(QuantMindTool):
    """Connect to MT5 broker."""

    name: str = "connect_broker"
    description: str = """Connect to a MetaTrader 5 broker terminal.
    Requires login credentials or uses saved connection.
    Returns connection status and account information."""

    args_schema: type[BaseModel] = ConnectBrokerInput
    category: ToolCategory = ToolCategory.BROKER
    priority: ToolPriority = ToolPriority.CRITICAL

    def execute(
        self,
        login: Optional[int] = None,
        password: Optional[str] = None,
        server: Optional[str] = None,
        terminal_id: Optional[str] = None,
        timeout: int = 30000,
        **kwargs
    ) -> ToolResult:
        """Execute broker connection."""
        logger.info("Connecting to MT5 broker")

        # In production, would call MCP MT5 server
        # Simulate connection
        connection_result = self._connect_via_mcp(
            login=login,
            password=password,
            server=server,
            terminal_id=terminal_id,
            timeout=timeout
        )

        return ToolResult.ok(
            data={
                "status": connection_result["status"],
                "terminal_id": connection_result.get("terminal_id", "default"),
                "server": connection_result.get("server", "Unknown"),
                "login": connection_result.get("login"),
            },
            metadata={
                "connected_at": datetime.now().isoformat(),
                "timeout": timeout,
            }
        )

    def _connect_via_mcp(
        self,
        login: Optional[int],
        password: Optional[str],
        server: Optional[str],
        terminal_id: Optional[str],
        timeout: int
    ) -> Dict[str, Any]:
        """Connect via MCP server."""
        # Simulated connection
        return {
            "status": ConnectionStatus.CONNECTED.value,
            "terminal_id": terminal_id or "default",
            "server": server or "MetaQuotes-Demo",
            "login": login or 12345678,
        }


@register_tool(
    agent_types=[AgentType.COPILOT, AgentType.ANALYST],
    tags=["broker", "mt5", "account"],
)
class GetAccountInfoTool(QuantMindTool):
    """Get MT5 account information."""

    name: str = "get_account_info"
    description: str = """Get detailed account information from MetaTrader 5.
    Returns balance, equity, margin, and other account metrics.
    Optionally includes trading history summary."""

    args_schema: type[BaseModel] = GetAccountInfoInput
    category: ToolCategory = ToolCategory.BROKER
    priority: ToolPriority = ToolPriority.NORMAL

    def execute(
        self,
        terminal_id: Optional[str] = None,
        include_history: bool = False,
        **kwargs
    ) -> ToolResult:
        """Execute account info retrieval."""
        logger.info("Getting MT5 account info")

        # In production, would call MCP MT5 server
        account_info = self._get_account_info_via_mcp(terminal_id)

        result_data = {
            "login": account_info.login,
            "name": account_info.name,
            "server": account_info.server,
            "currency": account_info.currency,
            "balance": account_info.balance,
            "equity": account_info.equity,
            "margin": account_info.margin,
            "free_margin": account_info.free_margin,
            "margin_level": account_info.margin_level,
            "profit": account_info.profit,
            "account_type": account_info.account_type.value,
            "leverage": account_info.leverage,
        }

        if include_history:
            result_data["history_summary"] = {
                "total_trades": 156,
                "winning_trades": 94,
                "losing_trades": 62,
                "total_profit": 1250.50,
                "total_loss": 750.25,
                "net_profit": 500.25,
            }

        return ToolResult.ok(
            data=result_data,
            metadata={
                "retrieved_at": datetime.now().isoformat(),
                "terminal_id": terminal_id or "default",
            }
        )

    def _get_account_info_via_mcp(self, terminal_id: Optional[str]) -> AccountInfo:
        """Get account info via MCP server."""
        # Simulated account info
        return AccountInfo(
            login=12345678,
            name="Trading Account",
            server="MetaQuotes-Demo",
            currency="USD",
            balance=10500.00,
            equity=10650.50,
            margin=500.00,
            free_margin=10150.50,
            margin_level=213.01,
            profit=150.50,
            account_type=AccountType.DEMO,
            leverage=100
        )


@register_tool(
    agent_types=[AgentType.COPILOT],
    tags=["broker", "mt5", "sync", "database"],
)
class SyncMT5DataTool(QuantMindTool):
    """Sync MT5 data to database."""

    name: str = "sync_mt5_data"
    description: str = """Synchronize MT5 data to the QuantMind database.
    Syncs positions, orders, and trading history.
    Supports incremental and full synchronization."""

    args_schema: type[BaseModel] = SyncMT5DataInput
    category: ToolCategory = ToolCategory.BROKER
    priority: ToolPriority = ToolPriority.NORMAL

    def execute(
        self,
        data_types: List[str] = None,
        terminal_id: Optional[str] = None,
        full_sync: bool = False,
        **kwargs
    ) -> ToolResult:
        """Execute MT5 data sync."""
        data_types = data_types or ["positions", "orders", "history"]
        logger.info(f"Syncing MT5 data: {data_types}")

        sync_results = {}

        for data_type in data_types:
            if data_type == "positions":
                sync_results["positions"] = self._sync_positions(terminal_id, full_sync)
            elif data_type == "orders":
                sync_results["orders"] = self._sync_orders(terminal_id, full_sync)
            elif data_type == "history":
                sync_results["history"] = self._sync_history(terminal_id, full_sync)

        return ToolResult.ok(
            data={
                "sync_results": sync_results,
                "data_types": data_types,
                "full_sync": full_sync,
            },
            metadata={
                "synced_at": datetime.now().isoformat(),
                "terminal_id": terminal_id or "default",
            }
        )

    def _sync_positions(self, terminal_id: Optional[str], full_sync: bool) -> Dict:
        """Sync positions."""
        # In production, would fetch from MT5 and store in database
        return {
            "synced": 3,
            "added": 0,
            "updated": 3,
            "removed": 0,
        }

    def _sync_orders(self, terminal_id: Optional[str], full_sync: bool) -> Dict:
        """Sync pending orders."""
        return {
            "synced": 2,
            "added": 0,
            "updated": 2,
            "removed": 0,
        }

    def _sync_history(self, terminal_id: Optional[str], full_sync: bool) -> Dict:
        """Sync trading history."""
        return {
            "synced": 150,
            "added": 10 if not full_sync else 150,
            "updated": 0,
            "removed": 0,
        }


@register_tool(
    agent_types=[AgentType.COPILOT, AgentType.ANALYST],
    tags=["broker", "mt5", "market", "data"],
)
class GetMarketDataTool(QuantMindTool):
    """Get current market data from MT5."""

    name: str = "get_market_data"
    description: str = """Get current market data for specified symbols.
    Returns bid, ask, spread, volume, and daily statistics.
    Useful for real-time analysis and decision making."""

    args_schema: type[BaseModel] = GetMarketDataInput
    category: ToolCategory = ToolCategory.BROKER
    priority: ToolPriority = ToolPriority.HIGH

    def execute(
        self,
        symbols: List[str],
        terminal_id: Optional[str] = None,
        **kwargs
    ) -> ToolResult:
        """Execute market data retrieval."""
        logger.info(f"Getting market data for: {symbols}")

        market_data = []
        for symbol in symbols:
            data = self._get_symbol_data(symbol, terminal_id)
            market_data.append({
                "symbol": data.symbol,
                "bid": data.bid,
                "ask": data.ask,
                "spread": data.spread,
                "last": data.last,
                "volume": data.volume,
                "time": data.time.isoformat(),
                "high": data.high,
                "low": data.low,
                "daily_change": data.daily_change,
            })

        return ToolResult.ok(
            data={
                "market_data": market_data,
                "symbols_count": len(market_data),
            },
            metadata={
                "retrieved_at": datetime.now().isoformat(),
                "terminal_id": terminal_id or "default",
            }
        )

    def _get_symbol_data(self, symbol: str, terminal_id: Optional[str]) -> MarketData:
        """Get data for a single symbol."""
        # Simulated market data
        import random
        base_price = {"EURUSD": 1.0850, "GBPUSD": 1.2650, "USDJPY": 149.50}.get(symbol, 1.0)
        spread = random.uniform(0.0001, 0.0003)

        return MarketData(
            symbol=symbol,
            bid=base_price - spread / 2,
            ask=base_price + spread / 2,
            spread=spread,
            last=base_price,
            volume=random.randint(1000, 10000),
            time=datetime.now(),
            high=base_price * 1.005,
            low=base_price * 0.995,
            daily_change=random.uniform(-0.5, 0.5)
        )


@register_tool(
    agent_types=[AgentType.COPILOT, AgentType.ANALYST],
    tags=["broker", "mt5", "positions"],
)
class GetPositionsTool(QuantMindTool):
    """Get open positions from MT5."""

    name: str = "get_positions"
    description: str = """Get all open positions or filter by symbol.
    Returns detailed position information including profit, SL, TP.
    Useful for portfolio monitoring and risk management."""

    args_schema: type[BaseModel] = GetPositionsInput
    category: ToolCategory = ToolCategory.BROKER
    priority: ToolPriority = ToolPriority.NORMAL

    def execute(
        self,
        symbol: Optional[str] = None,
        terminal_id: Optional[str] = None,
        **kwargs
    ) -> ToolResult:
        """Execute positions retrieval."""
        logger.info(f"Getting MT5 positions (symbol={symbol})")

        positions = self._get_positions_via_mcp(symbol, terminal_id)

        # Calculate summary
        total_profit = sum(p.profit for p in positions)
        total_swap = sum(p.swap for p in positions)

        return ToolResult.ok(
            data={
                "positions": [
                    {
                        "ticket": p.ticket,
                        "symbol": p.symbol,
                        "type": p.type,
                        "volume": p.volume,
                        "open_price": p.open_price,
                        "current_price": p.current_price,
                        "sl": p.sl,
                        "tp": p.tp,
                        "profit": p.profit,
                        "swap": p.swap,
                        "commission": p.commission,
                        "open_time": p.open_time.isoformat(),
                        "comment": p.comment,
                        "magic": p.magic,
                    }
                    for p in positions
                ],
                "summary": {
                    "total_positions": len(positions),
                    "total_profit": total_profit,
                    "total_swap": total_swap,
                    "net_profit": total_profit + total_swap,
                },
            },
            metadata={
                "retrieved_at": datetime.now().isoformat(),
                "terminal_id": terminal_id or "default",
                "filter_symbol": symbol,
            }
        )

    def _get_positions_via_mcp(
        self,
        symbol: Optional[str],
        terminal_id: Optional[str]
    ) -> List[Position]:
        """Get positions via MCP server."""
        # Simulated positions
        positions = [
            Position(
                ticket=1001,
                symbol="EURUSD",
                type="BUY",
                volume=0.10,
                open_price=1.0845,
                current_price=1.0855,
                sl=1.0800,
                tp=1.0900,
                profit=10.00,
                swap=-0.50,
                commission=0.0,
                open_time=datetime.now(),
                comment="QuantMind EA",
                magic=123456
            ),
            Position(
                ticket=1002,
                symbol="GBPUSD",
                type="SELL",
                volume=0.05,
                open_price=1.2670,
                current_price=1.2650,
                sl=1.2700,
                tp=1.2600,
                profit=10.00,
                swap=-0.25,
                commission=0.0,
                open_time=datetime.now(),
                comment="QuantMind EA",
                magic=123456
            ),
        ]

        if symbol:
            positions = [p for p in positions if p.symbol == symbol]

        return positions


@register_tool(
    agent_types=[AgentType.COPILOT, AgentType.ANALYST],
    tags=["broker", "mt5", "orders"],
)
class GetOrdersTool(QuantMindTool):
    """Get pending orders from MT5."""

    name: str = "get_orders"
    description: str = """Get all pending orders or filter by symbol.
    Returns order details including type, price, volume, and expiration.
    Useful for order management and strategy monitoring."""

    args_schema: type[BaseModel] = GetOrdersInput
    category: ToolCategory = ToolCategory.BROKER
    priority: ToolPriority = ToolPriority.NORMAL

    def execute(
        self,
        symbol: Optional[str] = None,
        terminal_id: Optional[str] = None,
        **kwargs
    ) -> ToolResult:
        """Execute orders retrieval."""
        logger.info(f"Getting MT5 orders (symbol={symbol})")

        orders = self._get_orders_via_mcp(symbol, terminal_id)

        return ToolResult.ok(
            data={
                "orders": orders,
                "total_orders": len(orders),
            },
            metadata={
                "retrieved_at": datetime.now().isoformat(),
                "terminal_id": terminal_id or "default",
                "filter_symbol": symbol,
            }
        )

    def _get_orders_via_mcp(
        self,
        symbol: Optional[str],
        terminal_id: Optional[str]
    ) -> List[Dict]:
        """Get orders via MCP server."""
        # Simulated orders
        orders = [
            {
                "ticket": 2001,
                "symbol": "EURUSD",
                "type": "BUY LIMIT",
                "volume": 0.10,
                "price": 1.0800,
                "sl": 1.0750,
                "tp": 1.0900,
                "expiration": None,
                "comment": "QuantMind EA",
                "magic": 123456,
            },
            {
                "ticket": 2002,
                "symbol": "GBPUSD",
                "type": "SELL STOP",
                "volume": 0.05,
                "price": 1.2700,
                "sl": 1.2750,
                "tp": 1.2600,
                "expiration": None,
                "comment": "QuantMind EA",
                "magic": 123456,
            },
        ]

        if symbol:
            orders = [o for o in orders if o["symbol"] == symbol]

        return orders


# Export all tools
__all__ = [
    "ConnectBrokerTool",
    "GetAccountInfoTool",
    "SyncMT5DataTool",
    "GetMarketDataTool",
    "GetPositionsTool",
    "GetOrdersTool",
    "ConnectionStatus",
    "AccountType",
    "AccountInfo",
    "Position",
    "MarketData",
]
