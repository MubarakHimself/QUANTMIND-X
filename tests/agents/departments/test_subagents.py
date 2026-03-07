# tests/agents/departments/test_subagents.py
"""
Tests for Department Subagents

Tests all department subagents for correct initialization,
task execution, and functionality.
"""
import pytest
from typing import Dict, Any

from src.agents.departments.subagents import (
    ResearchSubAgent,
    ResearchTask,
    TradingSubAgent,
    TradingTask,
    RiskSubAgent,
    RiskTask,
    RiskLimits,
    PortfolioSubAgent,
    PortfolioTask,
    DevelopmentSubAgent,
    DevelopmentTask,
)
from src.agents.departments.types import (
    Department,
    SubAgentType,
    SUBAGENT_DEPARTMENT_MAP,
    get_subagent_class,
)


class TestResearchSubAgent:
    """Tests for Research SubAgent."""

    def test_research_subagent_initializes(self):
        """Research subagent should initialize correctly."""
        task = ResearchTask(
            task_type="strategy_development",
            symbols=["EURUSD", "GBPUSD"],
        )
        agent = ResearchSubAgent(
            agent_id="test_research_1",
            task=task,
            available_tools=["market_data", "technical_analysis"],
        )

        assert agent.agent_id == "test_research_1"
        assert agent.agent_type == "research"
        assert agent.model_tier == "haiku"
        assert len(agent.available_tools) == 2

    def test_research_subagent_gets_market_data(self):
        """Research subagent should get market data."""
        agent = ResearchSubAgent(
            agent_id="test_research_2",
            available_tools=["market_data"],
        )

        result = agent.get_market_data("EURUSD", "1h", 100)

        assert result["symbol"] == "EURUSD"
        assert result["timeframe"] == "1h"
        assert result["candles"] == 100

    def test_research_subagent_performs_technical_analysis(self):
        """Research subagent should perform technical analysis."""
        agent = ResearchSubAgent(
            agent_id="test_research_3",
            available_tools=["technical_analysis"],
        )

        result = agent.perform_technical_analysis("EURUSD", ["RSI", "MACD"])

        assert result["symbol"] == "EURUSD"
        assert "RSI" in result["indicators"]
        assert "results" in result

    def test_research_subagent_generates_signals(self):
        """Research subagent should generate trading signals."""
        agent = ResearchSubAgent(
            agent_id="test_research_4",
            available_tools=["signal_generation"],
        )

        result = agent.generate_signals(["EURUSD", "GBPUSD"])

        assert "signals" in result
        assert result["count"] == 2

    def test_research_subagent_execute(self):
        """Research subagent should execute tasks."""
        task = ResearchTask(
            task_type="market_analysis",
            symbols=["EURUSD"],
        )
        agent = ResearchSubAgent(
            agent_id="test_research_5",
            task=task,
        )

        result = agent.execute()

        assert result["status"] == "completed"
        assert result["task_type"] == "market_analysis"


class TestTradingSubAgent:
    """Tests for Trading SubAgent."""

    def test_trading_subagent_initializes(self):
        """Trading subagent should initialize correctly."""
        task = TradingTask(
            task_type="order_execution",
            orders=[],
        )
        agent = TradingSubAgent(
            agent_id="test_trading_1",
            task=task,
            available_tools=["execute_order", "get_positions"],
        )

        assert agent.agent_id == "test_trading_1"
        assert agent.agent_type == "trading"
        assert agent.model_tier == "haiku"

    def test_trading_subagent_executes_order(self):
        """Trading subagent should execute orders."""
        agent = TradingSubAgent(
            agent_id="test_trading_2",
            available_tools=["execute_order"],
        )

        result = agent.execute_order(
            symbol="EURUSD",
            order_type="market",
            side="buy",
            volume=0.1,
            price=1.0850,
        )

        assert result["status"] == "success"
        assert "order" in result
        assert result["order"]["symbol"] == "EURUSD"

    def test_trading_subagent_gets_positions(self):
        """Trading subagent should get positions."""
        agent = TradingSubAgent(agent_id="test_trading_3")

        # Execute an order to create a position
        agent.execute_order(
            symbol="EURUSD",
            order_type="market",
            side="buy",
            volume=0.1,
        )

        result = agent.get_positions()

        assert "positions" in result
        assert result["count"] == 1

    def test_trading_subagent_closes_position(self):
        """Trading subagent should close positions."""
        agent = TradingSubAgent(agent_id="test_trading_4")

        # Create a position
        exec_result = agent.execute_order(
            symbol="EURUSD",
            order_type="market",
            side="buy",
            volume=0.1,
        )

        order_id = list(agent._open_positions.keys())[0]

        # Close the position
        close_result = agent.close_position(order_id)

        assert close_result["status"] == "success"
        assert len(agent._open_positions) == 0


class TestRiskSubAgent:
    """Tests for Risk SubAgent."""

    def test_risk_subagent_initializes(self):
        """Risk subagent should initialize correctly."""
        task = RiskTask(
            task_type="risk_validation",
        )
        agent = RiskSubAgent(
            agent_id="test_risk_1",
            task=task,
            available_tools=["validate_trade", "calculate_position_size"],
        )

        assert agent.agent_id == "test_risk_1"
        assert agent.agent_type == "risk"
        assert agent.model_tier == "haiku"

    def test_risk_subagent_calculates_position_size(self):
        """Risk subagent should calculate position size."""
        agent = RiskSubAgent(
            agent_id="test_risk_2",
            available_tools=["calculate_position_size"],
        )

        result = agent.calculate_position_size(
            account_balance=10000,
            entry_price=1.0850,
            stop_loss=1.0800,
            risk_percent=0.02,
        )

        assert "position_size" in result
        assert result["status"] == "calculated"
        assert result["position_size"] > 0

    def test_risk_subagent_validates_trade(self):
        """Risk subagent should validate trades."""
        agent = RiskSubAgent(
            agent_id="test_risk_3",
            available_tools=["validate_trade"],
        )

        result = agent.validate_trade(
            symbol="EURUSD",
            side="buy",
            volume=0.1,
            price=1.0850,
            account_balance=10000,
            current_exposure=0,
        )

        assert "is_valid" in result
        assert result["is_valid"] == True

    def test_risk_subagent_checks_drawdown(self):
        """Risk subagent should check drawdown."""
        agent = RiskSubAgent(agent_id="test_risk_4")

        result = agent.check_drawdown(
            account_balance=9500,
            peak_balance=10000,
        )

        assert result["drawdown_percent"] == 5.0
        assert result["within_limits"] == True


class TestPortfolioSubAgent:
    """Tests for Portfolio SubAgent."""

    def test_portfolio_subagent_initializes(self):
        """Portfolio subagent should initialize correctly."""
        task = PortfolioTask(
            task_type="allocation",
        )
        agent = PortfolioSubAgent(
            agent_id="test_portfolio_1",
            task=task,
            available_tools=["get_allocation", "track_performance"],
        )

        assert agent.agent_id == "test_portfolio_1"
        assert agent.agent_type == "portfolio"
        assert agent.model_tier == "haiku"

    def test_portfolio_subagent_gets_allocation(self):
        """Portfolio subagent should get allocation."""
        agent = PortfolioSubAgent(
            agent_id="test_portfolio_2",
            available_tools=["get_allocation"],
        )

        positions = [
            {"symbol": "EURUSD", "volume": 1.0, "entry_price": 1.0850, "current_price": 1.0900},
        ]

        result = agent.get_allocation(positions, total_value=10000)

        assert "allocation" in result
        assert result["total_value"] == 10000

    def test_portfolio_subagent_calculates_rebalance(self):
        """Portfolio subagent should calculate rebalancing."""
        agent = PortfolioSubAgent(
            agent_id="test_portfolio_3",
            available_tools=["calculate_rebalance"],
        )

        current = {"EURUSD": 0.6, "GBPUSD": 0.4}
        target = {"EURUSD": 0.5, "GBPUSD": 0.5}

        result = agent.calculate_rebalance(current, target, total_value=10000)

        assert "rebalance_trades" in result

    def test_portfolio_subagent_tracks_performance(self):
        """Portfolio subagent should track performance."""
        agent = PortfolioSubAgent(
            agent_id="test_portfolio_4",
            available_tools=["track_performance"],
        )

        positions = [
            {"symbol": "EURUSD", "volume": 1.0, "entry_price": 1.0850, "current_price": 1.0900},
        ]

        result = agent.track_performance(
            positions=positions,
            period_start_value=10000,
            period_end_value=10500,
        )

        assert "period_return" in result
        assert result["period_return"] == 5.0


class TestDevelopmentSubAgent:
    """Tests for Development SubAgent."""

    def test_development_subagent_initializes(self):
        """Development subagent should initialize correctly."""
        task = DevelopmentTask(
            task_type="python_dev",
        )
        agent = DevelopmentSubAgent(
            agent_id="test_dev_1",
            task=task,
            available_tools=["generate_python_ea", "validate_ea_code"],
        )

        assert agent.agent_id == "test_dev_1"
        assert agent.agent_type == "development"
        assert agent.model_tier == "haiku"

    def test_development_subagent_generates_python_ea(self):
        """Development subagent should generate Python EA code."""
        agent = DevelopmentSubAgent(
            agent_id="test_dev_2",
            available_tools=["generate_python_ea"],
        )

        result = agent.generate_python_ea(
            strategy_name="TestStrategy",
            strategy_params={"period": 14},
        )

        assert "code" in result
        assert result["language"] == "python"
        assert "TestStrategy" in result["code"]

    def test_development_subagent_generates_pinescript(self):
        """Development subagent should generate PineScript code."""
        agent = DevelopmentSubAgent(
            agent_id="test_dev_3",
            available_tools=["generate_pinescript"],
        )

        result = agent.generate_pinescript(
            strategy_name="TestPineStrategy",
            strategy_params={"period": 14},
        )

        assert "code" in result
        assert result["language"] == "pinescript"

    def test_development_subagent_generates_mql5(self):
        """Development subagent should generate MQL5 code."""
        agent = DevelopmentSubAgent(
            agent_id="test_dev_4",
            available_tools=["generate_mql5"],
        )

        result = agent.generate_mql5(
            strategy_name="TestEA",
            strategy_params={},
        )

        assert "code" in result
        assert result["language"] == "mql5"

    def test_development_subagent_validates_code(self):
        """Development subagent should validate code."""
        agent = DevelopmentSubAgent(
            agent_id="test_dev_5",
            available_tools=["validate_ea_code"],
        )

        code = '''
def on_bar(bar):
    return "BUY"
'''
        result = agent.validate_ea_code(code, "python")

        assert result["valid"] == True


class TestSubAgentTypes:
    """Tests for sub-agent type mappings."""

    def test_subagent_department_mapping(self):
        """Sub-agent types should map to correct departments."""
        assert SUBAGENT_DEPARTMENT_MAP["strategy_researcher"] == Department.RESEARCH
        assert SUBAGENT_DEPARTMENT_MAP["python_dev"] == Department.DEVELOPMENT
        assert SUBAGENT_DEPARTMENT_MAP["order_executor"] == Department.TRADING
        assert SUBAGENT_DEPARTMENT_MAP["position_sizer"] == Department.RISK
        assert SUBAGENT_DEPARTMENT_MAP["allocation_manager"] == Department.PORTFOLIO

    def test_get_subagent_class(self):
        """Should get correct sub-agent class for type."""
        research_class = get_subagent_class("strategy_researcher")
        assert research_class == ResearchSubAgent

        trading_class = get_subagent_class("order_executor")
        assert trading_class == TradingSubAgent

        risk_class = get_subagent_class("position_sizer")
        assert risk_class == RiskSubAgent

        portfolio_class = get_subagent_class("allocation_manager")
        assert portfolio_class == PortfolioSubAgent

        dev_class = get_subagent_class("python_dev")
        assert dev_class == DevelopmentSubAgent

    def test_get_subagent_class_unknown_type(self):
        """Should return None for unknown sub-agent type."""
        result = get_subagent_class("unknown_type")
        assert result is None
