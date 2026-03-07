"""
Trading Tools for Agent SDK

Provides custom trading tools: execute_trade, analyze_market, calculate_risk, optimize_portfolio.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging
import random

logger = logging.getLogger(__name__)


class OrderType(Enum):
    BUY = "buy"
    SELL = "sell"
    BUY_LIMIT = "buy_limit"
    SELL_LIMIT = "sell_limit"
    BUY_STOP = "buy_stop"
    SELL_STOP = "sell_stop"


class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class MarketRegime(Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE = "volatile"
    CALM = "calm"


@dataclass
class TradingTools:
    """Trading tools for order execution, market analysis, risk calculation, and portfolio optimization."""

    broker_type: str = "mt5"
    _orders: Dict[str, Dict] = field(default_factory=dict, repr=False)

    def execute_trade(
        self,
        symbol: str,
        order_type: str,
        volume: float,
        price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        comment: str = "",
    ) -> Dict[str, Any]:
        """Execute a trade order (simulation mode)."""
        try:
            order_type_enum = OrderType(order_type.lower())
        except ValueError:
            return {"success": False, "error": f"Invalid order type: {order_type}"}

        if volume <= 0:
            return {"success": False, "error": "Volume must be positive"}

        order_id = f"{symbol}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        execution_price = price if price else 1.0850

        result = {
            "success": True,
            "order_id": order_id,
            "status": OrderStatus.FILLED.value,
            "symbol": symbol,
            "order_type": order_type,
            "volume": volume,
            "filled_volume": volume,
            "price": execution_price,
            "slippage": 0.0,
            "message": f"Order {order_id} executed successfully",
            "timestamp": datetime.now().isoformat(),
        }

        self._orders[order_id] = result
        logger.info(f"Trade executed: {order_type} {volume} {symbol} at {execution_price}")
        return result

    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """Get the status of an order."""
        order = self._orders.get(order_id)
        if not order:
            return {"success": False, "error": f"Order not found: {order_id}"}
        return {"success": True, "order_id": order["order_id"], "status": order["status"], "filled_volume": order["filled_volume"], "filled_price": order["price"]}

    def analyze_market(self, symbol: str, price_data: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """Analyze market conditions for a symbol."""
        if not price_data:
            price_data = self._generate_sample_price_data()

        closes = [p["close"] for p in price_data]
        highs = [p["high"] for p in price_data]
        lows = [p["low"] for p in price_data]

        current_price = closes[-1]
        sma_short = sum(closes[-5:]) / 5
        sma_long = sum(closes[-20:]) / 20

        if sma_short > sma_long * 1.02:
            regime = MarketRegime.TRENDING_UP
            trend_strength = min((sma_short / sma_long - 1) * 100, 10)
        elif sma_short < sma_long * 0.98:
            regime = MarketRegime.TRENDING_DOWN
            trend_strength = min((1 - sma_short / sma_long) * 100, 10)
        else:
            regime = MarketRegime.RANGING
            trend_strength = 0.0

        returns = [(closes[i] - closes[i-1]) / closes[i-1] for i in range(1, len(closes))]
        volatility = (max(returns) - min(returns)) * 100 if returns else 0

        market_regime = MarketRegime.VOLATILE if volatility > 5 else (MarketRegime.CALM if volatility < 1 else regime)

        recommendation = "Consider long positions with tight stops" if regime == MarketRegime.TRENDING_UP else ("Consider short positions or stay flat" if regime == MarketRegime.TRENDING_DOWN else "Range-bound conditions - consider sell high buy low")

        return {
            "success": True,
            "symbol": symbol,
            "regime": market_regime.value,
            "volatility": round(volatility, 2),
            "trend_strength": round(trend_strength, 2),
            "current_price": current_price,
            "support_level": round(min(lows[-20:]), 5),
            "resistance_level": round(max(highs[-20:]), 5),
            "recommendation": recommendation,
            "timestamp": datetime.now().isoformat(),
        }

    def calculate_risk(
        self,
        symbol: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        volume: float,
        account_balance: float,
        current_positions: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """Calculate risk metrics for a potential trade."""
        if entry_price <= 0 or stop_loss <= 0 or take_profit <= 0:
            return {"success": False, "error": "Prices must be positive"}
        if volume <= 0:
            return {"success": False, "error": "Volume must be positive"}

        risk_per_point = volume * 100000
        stop_distance = abs(entry_price - stop_loss)
        position_risk_amount = stop_distance * risk_per_point
        position_risk_percent = (position_risk_amount / account_balance * 100) if account_balance > 0 else 0

        reward_distance = abs(take_profit - entry_price)
        risk_reward_ratio = reward_distance / stop_distance if stop_distance > 0 else 0

        total_exposure = volume * entry_price * risk_per_point / entry_price
        portfolio_risk = (total_exposure / account_balance * 100) if account_balance > 0 else 0

        var_95 = position_risk_amount * 1.65
        var_99 = position_risk_amount * 2.33
        max_drawdown = position_risk_percent * 2

        risk_score = 1
        if position_risk_percent > 5:
            risk_score += 2
        if position_risk_percent > 2:
            risk_score += 1
        if risk_reward_ratio < 1:
            risk_score += 3
        elif risk_reward_ratio < 2:
            risk_score += 1
        if portfolio_risk > 20:
            risk_score += 2
        risk_score = min(risk_score, 10)

        warnings = []
        if position_risk_percent > 5:
            warnings.append(f"High position risk: {position_risk_percent:.1f}% of account")
        if risk_reward_ratio < 1:
            warnings.append("Poor risk/reward ratio below 1:1")
        if portfolio_risk > 30:
            warnings.append("High portfolio exposure")

        return {
            "success": True,
            "symbol": symbol,
            "position_risk_percent": round(position_risk_percent, 2),
            "portfolio_risk_percent": round(portfolio_risk, 2),
            "risk_reward_ratio": round(risk_reward_ratio, 2),
            "var_95": round(var_95, 2),
            "var_99": round(var_99, 2),
            "max_drawdown": round(max_drawdown, 2),
            "sharpe_ratio": round((reward_distance / stop_distance - 1) * 10 if stop_distance > 0 else 0, 2),
            "risk_score": risk_score,
            "warnings": warnings,
            "recommendation": "APPROVE" if risk_score <= 5 else "REVIEW_REQUIRED",
        }

    def optimize_portfolio(
        self,
        positions: List[Dict],
        account_balance: float,
        target_risk: float = 2.0,
        risk_free_rate: float = 0.04,
    ) -> Dict[str, Any]:
        """Optimize portfolio allocation."""
        if not positions:
            return {"success": False, "error": "No positions provided"}
        if account_balance <= 0:
            return {"success": False, "error": "Invalid account balance"}

        allocations = []
        total_weight = 0.0

        for pos in positions:
            symbol = pos.get("symbol", "UNKNOWN")
            current_weight = (pos.get("exposure", 0) / account_balance * 100) if account_balance > 0 else 0
            volatility = pos.get("volatility", 10.0)
            target_weight = min(100 / volatility, 30) if volatility > 0 else current_weight

            target_exposure = account_balance * (target_weight / 100)
            current_exposure = pos.get("exposure", 0)

            allocations.append({
                "symbol": symbol,
                "current_weight": round(current_weight, 2),
                "target_weight": round(target_weight, 2),
                "suggested_volume": round((target_exposure - current_exposure) / 100000, 2),
                "current_exposure": round(current_exposure, 2),
                "risk_contribution": round((current_weight / 100) * volatility, 2),
            })
            total_weight += current_weight

        portfolio_volatility = sum(a["risk_contribution"] for a in allocations)
        risk_score = min(int(portfolio_volatility / 5), 10)
        rebalancing_required = any(abs(a["current_weight"] - a["target_weight"]) > 5 for a in allocations)

        suggestions = []
        if total_weight > 80:
            suggestions.append("Portfolio is heavily concentrated - consider reducing exposure")
        if rebalancing_required:
            suggestions.append("Rebalancing recommended to align with target weights")

        return {
            "success": True,
            "allocations": allocations,
            "expected_return": round(sum(a["current_weight"] * positions[i].get("return", 5) / 100 for i, a in enumerate(allocations)), 2),
            "expected_volatility": round(portfolio_volatility, 2),
            "sharpe_ratio": round(portfolio_volatility / 10 if portfolio_volatility > 0 else 0, 2),
            "risk_score": risk_score,
            "rebalancing_required": rebalancing_required,
            "suggestions": suggestions,
        }

    def _generate_sample_price_data(self) -> List[Dict]:
        """Generate sample price data for demonstration."""
        base_price = 1.0850
        data = []
        for _ in range(50):
            change = random.uniform(-0.005, 0.005)
            close = base_price + change
            data.append({
                "open": base_price,
                "high": round(close + random.uniform(0, 0.003), 5),
                "low": round(close - random.uniform(0, 0.003), 5),
                "close": round(close, 5),
                "volume": random.randint(1000, 10000),
            })
            base_price = close
        return data


TRADING_TOOL_SCHEMAS = {
    "execute_trade": {
        "name": "execute_trade",
        "description": "Execute a trading order (simulation mode for backtesting and paper trading)",
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Trading symbol (e.g., EURUSD, BTCUSD)"},
                "order_type": {"type": "string", "description": "Order type", "enum": ["buy", "sell", "buy_limit", "sell_limit", "buy_stop", "sell_stop"]},
                "volume": {"type": "number", "description": "Lot size"},
                "price": {"type": "number", "description": "Limit price"},
                "stop_loss": {"type": "number", "description": "Stop loss price"},
                "take_profit": {"type": "number", "description": "Take profit price"},
                "comment": {"type": "string", "description": "Order comment"},
            },
            "required": ["symbol", "order_type", "volume"],
        },
    },
    "analyze_market": {
        "name": "analyze_market",
        "description": "Analyze market conditions including trend, volatility, support/resistance levels",
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Trading symbol to analyze"},
                "price_data": {"type": "array", "description": "Optional OHLCV data", "items": {"type": "object", "properties": {"open": {"type": "number"}, "high": {"type": "number"}, "low": {"type": "number"}, "close": {"type": "number"}, "volume": {"type": "number"}}}},
            },
            "required": ["symbol"],
        },
    },
    "calculate_risk": {
        "name": "calculate_risk",
        "description": "Calculate risk metrics including VaR, Sharpe ratio, and risk score",
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Trading symbol"},
                "entry_price": {"type": "number", "description": "Planned entry price"},
                "stop_loss": {"type": "number", "description": "Stop loss price"},
                "take_profit": {"type": "number", "description": "Take profit price"},
                "volume": {"type": "number", "description": "Position volume in lots"},
                "account_balance": {"type": "number", "description": "Account balance"},
                "current_positions": {"type": "array", "description": "Current open positions"},
            },
            "required": ["symbol", "entry_price", "stop_loss", "take_profit", "volume", "account_balance"],
        },
    },
    "optimize_portfolio": {
        "name": "optimize_portfolio",
        "description": "Optimize portfolio allocation using inverse volatility weighting",
        "input_schema": {
            "type": "object",
            "properties": {
                "positions": {"type": "array", "description": "Current positions", "items": {"type": "object", "properties": {"symbol": {"type": "string"}, "exposure": {"type": "number"}, "volatility": {"type": "number"}, "return": {"type": "number"}}}},
                "account_balance": {"type": "number", "description": "Total account balance"},
                "target_risk": {"type": "number", "description": "Target risk per position", "default": 2.0},
                "risk_free_rate": {"type": "number", "description": "Risk-free rate", "default": 0.04},
            },
            "required": ["positions", "account_balance"],
        },
    },
}


def get_trading_tool_schemas() -> List[Dict]:
    """Get trading tool schemas for Agent SDK."""
    return list(TRADING_TOOL_SCHEMAS.values())
