"""
Tests for Trading Tools

Tests for execute_trade, analyze_market, calculate_risk, optimize_portfolio tools.
"""

import pytest
from datetime import datetime
from src.agents.tools.trading_tools import (
    TradingTools,
    OrderType,
    OrderStatus,
    MarketRegime,
    TRADING_TOOL_SCHEMAS,
    get_trading_tool_schemas,
)


class TestTradingTools:
    """Test suite for TradingTools class."""

    @pytest.fixture
    def tools(self):
        """Create TradingTools instance."""
        return TradingTools(broker_type="mt5")

    # =========================================================================
    # execute_trade tests
    # =========================================================================

    def test_execute_trade_buy_market(self, tools):
        """Test executing a market buy order."""
        result = tools.execute_trade(
            symbol="EURUSD",
            order_type="buy",
            volume=0.1,
            stop_loss=1.0800,
            take_profit=1.0950,
        )

        assert result["success"] is True
        assert result["symbol"] == "EURUSD"
        assert result["order_type"] == "buy"
        assert result["volume"] == 0.1
        assert result["status"] == "filled"
        assert "order_id" in result

    def test_execute_trade_sell_limit(self, tools):
        """Test executing a limit sell order."""
        result = tools.execute_trade(
            symbol="GBPUSD",
            order_type="sell_limit",
            volume=0.2,
            price=1.2700,
            stop_loss=1.2750,
            take_profit=1.2600,
        )

        assert result["success"] is True
        assert result["symbol"] == "GBPUSD"
        assert result["order_type"] == "sell_limit"
        assert result["volume"] == 0.2

    def test_execute_trade_invalid_type(self, tools):
        """Test executing trade with invalid order type."""
        result = tools.execute_trade(
            symbol="EURUSD",
            order_type="invalid_type",
            volume=0.1,
        )

        assert result["success"] is False
        assert "Invalid order type" in result["error"]

    def test_execute_trade_invalid_volume(self, tools):
        """Test executing trade with invalid volume."""
        result = tools.execute_trade(
            symbol="EURUSD",
            order_type="buy",
            volume=-0.1,
        )

        assert result["success"] is False
        assert "Volume must be positive" in result["error"]

    def test_execute_trade_zero_volume(self, tools):
        """Test executing trade with zero volume."""
        result = tools.execute_trade(
            symbol="EURUSD",
            order_type="buy",
            volume=0,
        )

        assert result["success"] is False

    def test_get_order_status(self, tools):
        """Test getting order status."""
        # First execute a trade
        execute_result = tools.execute_trade(
            symbol="EURUSD",
            order_type="buy",
            volume=0.1,
        )
        order_id = execute_result["order_id"]

        # Then check status
        status = tools.get_order_status(order_id)

        assert status["success"] is True
        assert status["order_id"] == order_id
        assert status["status"] == "filled"

    def test_get_order_status_not_found(self, tools):
        """Test getting status of non-existent order."""
        status = tools.get_order_status("non_existent_order")

        assert status["success"] is False
        assert "Order not found" in status["error"]

    # =========================================================================
    # analyze_market tests
    # =========================================================================

    def test_analyze_market_default_data(self, tools):
        """Test market analysis with default simulated data."""
        result = tools.analyze_market(symbol="EURUSD")

        assert result["success"] is True
        assert result["symbol"] == "EURUSD"
        assert "regime" in result
        assert "volatility" in result
        assert "trend_strength" in result
        assert "support_level" in result
        assert "resistance_level" in result
        assert "recommendation" in result
        assert "timestamp" in result

    def test_analyze_market_with_price_data(self, tools):
        """Test market analysis with provided price data."""
        price_data = [
            {"open": 1.080, "high": 1.085, "low": 1.078, "close": 1.083, "volume": 5000},
            {"open": 1.083, "high": 1.088, "low": 1.081, "close": 1.086, "volume": 6000},
            {"open": 1.086, "high": 1.090, "low": 1.084, "close": 1.088, "volume": 7000},
        ]

        result = tools.analyze_market(symbol="EURUSD", price_data=price_data)

        assert result["success"] is True
        assert result["symbol"] == "EURUSD"

    def test_analyze_market_regimes(self, tools):
        """Test different market regimes are detected."""
        # Trending up data
        trending_up_data = [
            {"open": 1.050 + i * 0.002, "high": 1.055 + i * 0.002, "low": 1.048 + i * 0.002, "close": 1.053 + i * 0.002, "volume": 5000}
            for i in range(30)
        ]

        result = tools.analyze_market(symbol="EURUSD", price_data=trending_up_data)

        assert result["success"] is True
        # In strong uptrend, should detect trending_up or volatile

    # =========================================================================
    # calculate_risk tests
    # =========================================================================

    def test_calculate_risk_valid_inputs(self, tools):
        """Test risk calculation with valid inputs."""
        result = tools.calculate_risk(
            symbol="EURUSD",
            entry_price=1.0850,
            stop_loss=1.0800,
            take_profit=1.0950,
            volume=0.1,
            account_balance=10000.0,
        )

        assert result["success"] is True
        assert "position_risk_percent" in result
        assert "portfolio_risk_percent" in result
        assert "risk_reward_ratio" in result
        assert "var_95" in result
        assert "var_99" in result
        assert "sharpe_ratio" in result
        assert "risk_score" in result
        assert "recommendation" in result

    def test_calculate_risk_invalid_price(self, tools):
        """Test risk calculation with invalid price."""
        result = tools.calculate_risk(
            symbol="EURUSD",
            entry_price=-1.0850,
            stop_loss=1.0800,
            take_profit=1.0950,
            volume=0.1,
            account_balance=10000.0,
        )

        assert result["success"] is False
        assert "Prices must be positive" in result["error"]

    def test_calculate_risk_invalid_volume(self, tools):
        """Test risk calculation with invalid volume."""
        result = tools.calculate_risk(
            symbol="EURUSD",
            entry_price=1.0850,
            stop_loss=1.0800,
            take_profit=1.0950,
            volume=0,
            account_balance=10000.0,
        )

        assert result["success"] is False
        assert "Volume must be positive" in result["error"]

    def test_calculate_risk_high_position_risk(self, tools):
        """Test risk calculation with high position risk."""
        result = tools.calculate_risk(
            symbol="EURUSD",
            entry_price=1.0850,
            stop_loss=1.0600,  # Wide stop
            take_profit=1.0900,  # Tight target
            volume=1.0,  # Large volume
            account_balance=1000.0,  # Small account
        )

        assert result["success"] is True
        # Should have warnings for high risk
        assert len(result["warnings"]) > 0
        assert any("High position risk" in w for w in result["warnings"])

    def test_calculate_risk_poor_risk_reward(self, tools):
        """Test risk calculation with poor risk/reward ratio."""
        result = tools.calculate_risk(
            symbol="EURUSD",
            entry_price=1.0850,
            stop_loss=1.0830,  # Tight stop
            take_profit=1.0855,  # Very tight target
            volume=0.1,
            account_balance=10000.0,
        )

        assert result["success"] is True
        # Should have warning for poor R/R
        assert "risk_reward_ratio" in result
        assert result["risk_reward_ratio"] < 1

    # =========================================================================
    # optimize_portfolio tests
    # =========================================================================

    def test_optimize_portfolio_valid_positions(self, tools):
        """Test portfolio optimization with valid positions."""
        positions = [
            {"symbol": "EURUSD", "exposure": 5000, "volatility": 8.0, "return": 0.05},
            {"symbol": "GBPUSD", "exposure": 3000, "volatility": 10.0, "return": 0.04},
            {"symbol": "USDJPY", "exposure": 2000, "volatility": 6.0, "return": 0.03},
        ]

        result = tools.optimize_portfolio(
            positions=positions,
            account_balance=10000.0,
            target_risk=2.0,
        )

        assert result["success"] is True
        assert "allocations" in result
        assert len(result["allocations"]) == 3
        assert "expected_return" in result
        assert "expected_volatility" in result
        assert "sharpe_ratio" in result
        assert "risk_score" in result
        assert "rebalancing_required" in result

    def test_optimize_portfolio_empty_positions(self, tools):
        """Test portfolio optimization with no positions."""
        result = tools.optimize_portfolio(
            positions=[],
            account_balance=10000.0,
        )

        assert result["success"] is False
        assert "No positions provided" in result["error"]

    def test_optimize_portfolio_invalid_balance(self, tools):
        """Test portfolio optimization with invalid balance."""
        positions = [
            {"symbol": "EURUSD", "exposure": 5000, "volatility": 8.0},
        ]

        result = tools.optimize_portfolio(
            positions=positions,
            account_balance=0,
        )

        assert result["success"] is False
        assert "Invalid account balance" in result["error"]

    def test_optimize_portfolio_concentration(self, tools):
        """Test portfolio optimization detects concentration."""
        # Highly concentrated portfolio
        positions = [
            {"symbol": "EURUSD", "exposure": 9000, "volatility": 15.0, "return": 0.08},
            {"symbol": "GBPUSD", "exposure": 500, "volatility": 10.0, "return": 0.04},
        ]

        result = tools.optimize_portfolio(
            positions=positions,
            account_balance=10000.0,
        )

        assert result["success"] is True
        # Should have suggestion about concentration
        assert any("concentrated" in s.lower() for s in result["suggestions"])

    # =========================================================================
    # Tool schema tests
    # =========================================================================

    def test_trading_tool_schemas_exist(self):
        """Test all required tool schemas are defined."""
        required_tools = ["execute_trade", "analyze_market", "calculate_risk", "optimize_portfolio"]

        for tool_name in required_tools:
            assert tool_name in TRADING_TOOL_SCHEMAS
            schema = TRADING_TOOL_SCHEMAS[tool_name]
            assert "name" in schema
            assert "description" in schema
            assert "input_schema" in schema

    def test_get_trading_tool_schemas(self):
        """Test getting all trading tool schemas."""
        schemas = get_trading_tool_schemas()

        assert len(schemas) == 4
        tool_names = [s["name"] for s in schemas]
        assert "execute_trade" in tool_names
        assert "analyze_market" in tool_names
        assert "calculate_risk" in tool_names
        assert "optimize_portfolio" in tool_names

    def test_execute_trade_schema(self):
        """Test execute_trade tool schema structure."""
        schema = TRADING_TOOL_SCHEMAS["execute_trade"]

        assert schema["name"] == "execute_trade"
        assert "description" in schema

        input_schema = schema["input_schema"]
        assert input_schema["type"] == "object"
        assert "properties" in input_schema

        props = input_schema["properties"]
        assert "symbol" in props
        assert "order_type" in props
        assert "volume" in props
        assert "required" in input_schema

    def test_analyze_market_schema(self):
        """Test analyze_market tool schema structure."""
        schema = TRADING_TOOL_SCHEMAS["analyze_market"]

        props = schema["input_schema"]["properties"]
        assert "symbol" in props
        assert "price_data" in props

    def test_calculate_risk_schema(self):
        """Test calculate_risk tool schema structure."""
        schema = TRADING_TOOL_SCHEMAS["calculate_risk"]

        props = schema["input_schema"]["properties"]
        assert "symbol" in props
        assert "entry_price" in props
        assert "stop_loss" in props
        assert "take_profit" in props
        assert "volume" in props
        assert "account_balance" in props

    def test_optimize_portfolio_schema(self):
        """Test optimize_portfolio tool schema structure."""
        schema = TRADING_TOOL_SCHEMAS["optimize_portfolio"]

        props = schema["input_schema"]["properties"]
        assert "positions" in props
        assert "account_balance" in props


class TestOrderTypes:
    """Test suite for OrderType enum."""

    def test_order_type_values(self):
        """Test OrderType enum values."""
        assert OrderType.BUY.value == "buy"
        assert OrderType.SELL.value == "sell"
        assert OrderType.BUY_LIMIT.value == "buy_limit"
        assert OrderType.SELL_LIMIT.value == "sell_limit"
        assert OrderType.BUY_STOP.value == "buy_stop"
        assert OrderType.SELL_STOP.value == "sell_stop"


class TestMarketRegime:
    """Test suite for MarketRegime enum."""

    def test_market_regime_values(self):
        """Test MarketRegime enum values."""
        assert MarketRegime.TRENDING_UP.value == "trending_up"
        assert MarketRegime.TRENDING_DOWN.value == "trending_down"
        assert MarketRegime.RANGING.value == "ranging"
        assert MarketRegime.VOLATILE.value == "volatile"
        assert MarketRegime.CALM.value == "calm"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
