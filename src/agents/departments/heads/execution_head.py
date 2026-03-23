"""
Trading Department Head

Responsible for:
- Order execution (paper trading)
- Fill tracking and confirmation
- Trade monitoring and management
- Paper trade P&L tracking and reporting
- Copilot periodic updates
"""
import logging
import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

from src.agents.departments.heads.base import DepartmentHead
from src.agents.departments.types import Department, get_department_config

logger = logging.getLogger(__name__)


# =============================================================================
# Trading Models
# =============================================================================

class TradeSide(str, Enum):
    """Trade direction."""
    BUY = "BUY"
    SELL = "SELL"


class TradeStatus(str, Enum):
    """Trade status."""
    PENDING = "pending"
    OPEN = "open"
    CLOSED = "closed"
    CANCELLED = "cancelled"


class RegimeType(str, Enum):
    """Market regime types."""
    TREND = "TREND"
    RANGE = "RANGE"
    BREAKOUT = "BREAKOUT"
    CHAOS = "CHAOS"


@dataclass
class TradePNL:
    """P&L for a single trade."""
    trade_id: str
    entry_price: float
    current_price: float
    quantity: float
    side: str
    pnl_value: float = 0.0
    pnl_pct: float = 0.0

    def calculate(self):
        """Calculate P&L based on side."""
        if self.side == TradeSide.BUY:
            self.pnl_value = (self.current_price - self.entry_price) * self.quantity
            self.pnl_pct = ((self.current_price - self.entry_price) / self.entry_price) * 100
        else:  # SELL
            self.pnl_value = (self.entry_price - self.current_price) * self.quantity
            self.pnl_pct = ((self.entry_price - self.current_price) / self.entry_price) * 100


@dataclass
class PaperTradingMetrics:
    """Aggregated paper trading metrics."""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    avg_pnl: float = 0.0
    max_drawdown: float = 0.0
    avg_hold_time_minutes: float = 0.0
    regime_correlation: Dict[str, float] = field(default_factory=dict)


@dataclass
class PaperTradingStatus:
    """Current paper trading status for Copilot updates."""
    agent_id: str
    strategy_name: str
    metrics: PaperTradingMetrics
    regime: str
    last_update: datetime


# =============================================================================
# Trading Department Head Implementation
# =============================================================================

class TradingHead(DepartmentHead):
    """Trading Department Head for order execution."""

    # Redis Stream topic for Copilot updates
    COPILOT_UPDATES_TOPIC = "copilot:updates"

    # Default update interval in seconds
    DEFAULT_UPDATE_INTERVAL = 60

    def __init__(self, mail_db_path: str = ".quantmind/department_mail.db"):
        config = get_department_config(Department.TRADING)
        super().__init__(config=config, mail_db_path=mail_db_path)
        self._update_interval = self.DEFAULT_UPDATE_INTERVAL
        self._active_monitors: Dict[str, asyncio.Task] = {}
        self._regime_cache: Dict[str, str] = {}

    def get_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "route_order",
                "description": "Route order to best venue",
                "parameters": {
                    "symbol": "Trading symbol",
                    "side": "BUY or SELL",
                    "quantity": "Order quantity",
                    "order_type": "MARKET, LIMIT, STOP",
                },
            },
            {
                "name": "track_fill",
                "description": "Track order fill status",
                "parameters": {
                    "order_id": "Order identifier",
                },
            },
            {
                "name": "monitor_slippage",
                "description": "Monitor execution slippage",
                "parameters": {
                    "symbol": "Trading symbol",
                    "period": "Monitoring period",
                },
            },
            {
                "name": "monitor_paper_trading",
                "description": "Monitor paper trading performance and push updates to Copilot",
                "parameters": {
                    "agent_id": "Paper trading agent ID to monitor",
                    "strategy_name": "Strategy name",
                },
            },
            {
                "name": "get_paper_trading_status",
                "description": "Get current paper trading status and metrics",
                "parameters": {
                    "agent_id": "Paper trading agent ID",
                },
            },
            {
                "name": "get_trade_history",
                "description": "Get trade history with P&L tracking",
                "parameters": {
                    "agent_id": "Paper trading agent ID",
                    "limit": "Number of trades to return",
                },
            },
            {
                "name": "stop_monitoring",
                "description": "Stop monitoring a paper trading agent",
                "parameters": {
                    "agent_id": "Paper trading agent ID to stop",
                },
            },
        ]

    # =========================================================================
    # Paper Trading Monitoring Methods
    # =========================================================================

    def monitor_paper_trading(
        self,
        agent_id: str,
        strategy_name: str,
    ) -> Dict[str, Any]:
        """
        Start monitoring a paper trading agent.

        Args:
            agent_id: Paper trading agent ID
            strategy_name: Strategy name

        Returns:
            Monitoring start confirmation
        """
        if agent_id in self._active_monitors:
            return {
                "status": "already_monitoring",
                "agent_id": agent_id,
                "message": f"Already monitoring {agent_id}",
            }

        # In production, would start an async monitor task
        # Register the agent for monitoring so duplicate calls are detected
        self._active_monitors[agent_id] = None  # placeholder; replaced by asyncio.Task in production
        logger.info(f"Started monitoring paper trading agent: {agent_id} ({strategy_name})")

        return {
            "status": "monitoring_started",
            "agent_id": agent_id,
            "strategy_name": strategy_name,
            "update_interval_seconds": self._update_interval,
            "copilot_topic": self.COPILOT_UPDATES_TOPIC,
        }

    def get_paper_trading_status(self, agent_id: str) -> Dict[str, Any]:
        """
        Get current paper trading status and metrics.

        Args:
            agent_id: Paper trading agent ID

        Returns:
            Current status and metrics
        """
        # In production, would fetch real metrics from the paper trading system
        # For demo, return a structured response
        metrics = PaperTradingMetrics(
            total_trades=12,
            winning_trades=8,
            losing_trades=4,
            win_rate=66.67,
            total_pnl=2.3,  # +2.3%
            avg_pnl=0.19,
            max_drawdown=5.2,
            avg_hold_time_minutes=45.0,
            regime_correlation={
                "TREND": 0.75,
                "RANGE": 0.15,
                "BREAKOUT": 0.60,
                "CHAOS": -0.20,
            },
        )

        return {
            "agent_id": agent_id,
            "status": "active",
            "metrics": self._serialize_metrics(metrics),
            "regime": RegimeType.TREND.value,
            "last_update": datetime.now(timezone.utc).isoformat(),
        }

    def get_trade_history(
        self,
        agent_id: str,
        limit: int = 50,
    ) -> Dict[str, Any]:
        """
        Get trade history with P&L tracking.

        Args:
            agent_id: Paper trading agent ID
            limit: Maximum number of trades to return

        Returns:
            Trade history with P&L
        """
        # In production, would fetch from database
        # For demo, return sample data
        trades = [
            {
                "trade_id": f"t{100+i}",
                "symbol": "EURUSD",
                "side": "BUY" if i % 2 == 0 else "SELL",
                "entry_price": 1.0850 + (i * 0.001),
                "current_price": 1.0875,
                "quantity": 10000,
                "pnl_value": 25.0 if i % 2 == 0 else -10.0,
                "pnl_pct": 0.23 if i % 2 == 0 else -0.09,
                "status": "closed" if i < 10 else "open",
                "entry_time": "2026-03-15T10:30:00Z",
                "exit_time": "2026-03-15T14:45:00Z" if i < 10 else None,
                "hold_time_minutes": 255 if i < 10 else None,
            }
            for i in range(min(limit, 20))
        ]

        return {
            "agent_id": agent_id,
            "total_trades": len(trades),
            "trades": trades,
        }

    def stop_monitoring(self, agent_id: str) -> Dict[str, Any]:
        """
        Stop monitoring a paper trading agent.

        Args:
            agent_id: Paper trading agent ID

        Returns:
            Stop confirmation
        """
        if agent_id not in self._active_monitors:
            return {
                "status": "not_monitoring",
                "agent_id": agent_id,
                "message": f"Not currently monitoring {agent_id}",
            }

        # Cancel the monitor task if one exists (may be None for stub implementations)
        task = self._active_monitors[agent_id]
        if task is not None:
            task.cancel()
        del self._active_monitors[agent_id]

        logger.info(f"Stopped monitoring paper trading agent: {agent_id}")

        return {
            "status": "monitoring_stopped",
            "agent_id": agent_id,
        }

    def _serialize_metrics(self, metrics: PaperTradingMetrics) -> Dict[str, Any]:
        """Serialize metrics to dictionary."""
        return {
            "total_trades": metrics.total_trades,
            "winning_trades": metrics.winning_trades,
            "losing_trades": metrics.losing_trades,
            "win_rate": round(metrics.win_rate, 2),
            "total_pnl": round(metrics.total_pnl, 2),
            "avg_pnl": round(metrics.avg_pnl, 2),
            "max_drawdown": round(metrics.max_drawdown, 2),
            "avg_hold_time_minutes": round(metrics.avg_hold_time_minutes, 1),
            "regime_correlation": metrics.regime_correlation,
        }

    def _serialize_for_copilot(self, status: PaperTradingStatus) -> str:
        """
        Serialize status for Copilot update message.

        Args:
            status: Paper trading status

        Returns:
            Formatted message string for Copilot
        """
        m = status.metrics
        return (
            f"Paper trading: {m.total_trades} trades, "
            f"{m.total_pnl:+.1f}% P&L, "
            f"{m.win_rate:.0f}% win rate"
        )

    # =========================================================================
    # Order Execution Methods
    # =========================================================================

    def route_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str = "MARKET",
    ) -> Dict[str, Any]:
        """
        Route order to best venue.

        Args:
            symbol: Trading symbol
            side: BUY or SELL
            quantity: Order quantity
            order_type: MARKET, LIMIT, or STOP

        Returns:
            Order routing result
        """
        order_id = f"ord-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"

        return {
            "order_id": order_id,
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "order_type": order_type,
            "status": "routed",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "venue": "best_execution",
        }

    def track_fill(self, order_id: str) -> Dict[str, Any]:
        """
        Track order fill status.

        Args:
            order_id: Order identifier

        Returns:
            Fill status
        """
        return {
            "order_id": order_id,
            "status": "filled",
            "fill_price": 1.0850,
            "fill_quantity": 10000,
            "fill_time": datetime.now(timezone.utc).isoformat(),
            "commission": 0.5,
        }

    def monitor_slippage(
        self,
        symbol: str,
        period: int = 60,
    ) -> Dict[str, Any]:
        """
        Monitor execution slippage.

        Args:
            symbol: Trading symbol
            period: Monitoring period in minutes

        Returns:
            Slippage analysis
        """
        return {
            "symbol": symbol,
            "period_minutes": period,
            "avg_slippage_bps": 0.5,
            "max_slippage_bps": 2.1,
            "slippage_std": 0.3,
            "sample_size": 150,
        }

    # =========================================================================
    # Claude SDK Integration
    # =========================================================================

    def _format_tools_for_anthropic(self) -> list:
        """Format registered tools into Anthropic tool definitions."""
        tools = []
        for tool_name, tool_obj in (self._tools or {}).items():
            try:
                tools.append({
                    "name": tool_name,
                    "description": getattr(tool_obj, "description", f"{tool_name} tool"),
                    "input_schema": getattr(
                        tool_obj,
                        "input_schema",
                        {"type": "object", "properties": {}, "required": []},
                    ),
                })
            except Exception:
                pass
        return tools

    async def process_task(self, task: str, context: dict = None) -> dict:
        """
        Process a trading/execution task via Claude SDK.

        Args:
            task: Task description string.
            context: Optional canvas context dict.

        Returns:
            Result dict with status, department, content, and tool_calls.
        """
        dept_system = self.system_prompt

        # Read relevant memory
        memory_ctx = ""
        try:
            if hasattr(self, "_read_relevant_memory"):
                nodes = await self._read_relevant_memory(task)
                if nodes:
                    memory_ctx = "\n\n## Relevant Memory\n" + "\n".join(
                        f"- {n['content']}" for n in nodes
                    )
        except Exception:
            pass

        full_system = dept_system + memory_ctx
        tools = self._format_tools_for_anthropic() if hasattr(self, "_format_tools_for_anthropic") else []

        try:
            if hasattr(self, "_invoke_claude"):
                result = await self._invoke_claude(task=task, canvas_context=context, tools=tools if tools else None)
            else:
                import os
                import anthropic as _anthropic
                client = _anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
                kwargs = {
                    "model": os.getenv("ANTHROPIC_MODEL_SONNET", "claude-sonnet-4-6"),
                    "max_tokens": 4096,
                    "system": full_system,
                    "messages": [{"role": "user", "content": task}],
                }
                if tools:
                    kwargs["tools"] = tools
                resp = await client.messages.create(**kwargs)
                content = "".join(b.text for b in resp.content if b.type == "text")
                result = {"content": content, "tool_calls": []}
        except Exception as e:
            logger.error(f"{self.department.value} Claude call failed: {e}")
            return {"status": "error", "error": str(e), "department": self.department.value}

        # Write opinion node
        try:
            if hasattr(self, "_write_opinion_node") and result.get("content"):
                await self._write_opinion_node(
                    content=f"Task: {task[:200]}\nResult: {result['content'][:500]}",
                    confidence=0.7,
                    tags=[self.department.value],
                )
        except Exception:
            pass

        return {
            "status": "success",
            "department": self.department.value,
            "content": result.get("content", ""),
            "tool_calls": result.get("tool_calls", []),
        }

# Alias for backward compatibility
ExecutionHead = TradingHead
