# src/agents/departments/schemas/test_schemas.py
"""
Tests for department schemas.
"""
import pytest
from datetime import datetime, timedelta

from pydantic import ValidationError

from src.agents.departments.schemas.common import (
    DepartmentOutput,
    TaskResult,
    ErrorResponse,
    AgentMessage,
    ValidationResult,
    Department,
    TaskStatus,
)
from src.agents.departments.schemas.research import (
    StrategyOutput,
    BacktestResult,
    AlphaFactor,
    StrategyType,
    TimeFrame,
    StrategyStatus,
)
from src.agents.departments.schemas.trading import (
    OrderRequest,
    OrderResponse,
    FillInfo,
    OrderSide,
    OrderType,
    OrderStatus,
    TimeInForce,
)
from src.agents.departments.schemas.risk import (
    PositionSizeRequest,
    PositionSizeResponse,
    DrawdownInfo,
    VaRResult,
    RiskLevel,
    RiskLimits,
)
from src.agents.departments.schemas.portfolio import (
    AllocationRequest,
    AllocationResult,
    RebalancePlan,
    PerformanceMetrics,
    OptimizationObjective,
)
from src.agents.departments.schemas.development import (
    EACreationRequest,
    EACreationResponse,
    EALanguage,
    EAStatus,
    TestResult,
    DeploymentConfig,
)


class TestCommonSchemas:
    """Tests for common schemas."""

    def test_department_output_creation(self):
        """Test DepartmentOutput creation."""
        output = DepartmentOutput(
            department=Department.RESEARCH,
            task_id="task_001",
            status=TaskStatus.COMPLETED,
            message="Task completed successfully",
            data={"result": "value"}
        )
        assert output.department == Department.RESEARCH
        assert output.task_id == "task_001"
        assert output.status == TaskStatus.COMPLETED

    def test_task_result_success(self):
        """Test TaskResult with success."""
        result = TaskResult(
            success=True,
            task_id="task_001",
            result={"output": "test"},
            execution_time_ms=100
        )
        assert result.success is True
        assert result.result["output"] == "test"
        assert result.execution_time_ms == 100

    def test_task_result_failure(self):
        """Test TaskResult with failure."""
        result = TaskResult(
            success=False,
            task_id="task_001",
            error="Something went wrong"
        )
        assert result.success is False
        assert result.error == "Something went wrong"

    def test_error_response(self):
        """Test ErrorResponse creation."""
        error = ErrorResponse(
            error="ValidationError",
            message="Invalid input",
            details={"field": "symbol"}
        )
        assert error.error == "ValidationError"
        assert error.details["field"] == "symbol"

    def test_agent_message(self):
        """Test AgentMessage creation."""
        msg = AgentMessage(
            sender="research_head",
            recipient="development_head",
            subject="New strategy",
            body="Strategy is ready",
            priority="high"
        )
        assert msg.sender == "research_head"
        assert msg.priority == "high"

    def test_validation_result(self):
        """Test ValidationResult."""
        result = ValidationResult(
            valid=True,
            warnings=["Consider lower risk"]
        )
        assert result.valid is True
        assert len(result.warnings) == 1


class TestResearchSchemas:
    """Tests for research department schemas."""

    def test_strategy_output_creation(self):
        """Test StrategyOutput creation."""
        strategy = StrategyOutput(
            strategy_id="STRAT_001",
            name="Trend Follower",
            strategy_type=StrategyType.TREND,
            description="SMA crossover strategy",
            symbols=["EURUSD", "GBPUSD"],
            timeframes=[TimeFrame.H1, TimeFrame.H4],
            entry_rules="Enter long on SMA crossover",
            exit_rules="Exit on reverse crossover"
        )
        assert strategy.strategy_id == "STRAT_001"
        assert strategy.strategy_type == StrategyType.TREND

    def test_strategy_output_with_risk_parameters(self):
        """Test StrategyOutput with risk parameters."""
        strategy = StrategyOutput(
            strategy_id="STRAT_002",
            name="Breakout Strategy",
            strategy_type=StrategyType.BREAKOUT,
            description="Breakout strategy",
            symbols=["EURUSD"],
            entry_rules="Enter on breakout",
            exit_rules="Exit on reversal",
            risk_parameters={"stop_loss_pips": 30, "take_profit_pips": 60}
        )
        assert strategy.risk_parameters["stop_loss_pips"] == 30

    def test_backtest_result_creation(self):
        """Test BacktestResult creation."""
        result = BacktestResult(
            backtest_id="BT_001",
            strategy_id="STRAT_001",
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 12, 31),
            initial_balance=10000.0,
            final_balance=12000.0,
            total_return=20.0,
            max_drawdown=-15.5,
            sharpe_ratio=1.2,
            win_rate=0.55,
            total_trades=100,
            profitable_trades=55,
            losing_trades=45,
            avg_profit=100.0,
            avg_loss=-50.0,
            profit_factor=2.0,
            running_time_ms=5000
        )
        assert result.backtest_id == "BT_001"
        assert result.total_return == 20.0

    def test_backtest_result_drawdown_negative(self):
        """Test that drawdown is stored as negative."""
        result = BacktestResult(
            backtest_id="BT_002",
            strategy_id="STRAT_001",
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 12, 31),
            initial_balance=10000.0,
            final_balance=11000.0,
            total_return=10.0,
            max_drawdown=10.0,  # Should be converted to negative
            win_rate=0.5,
            total_trades=50,
            profitable_trades=25,
            losing_trades=25,
            avg_profit=50.0,
            avg_loss=-25.0,
            profit_factor=2.0,
            running_time_ms=3000
        )
        assert result.max_drawdown < 0

    def test_alpha_factor(self):
        """Test AlphaFactor creation."""
        alpha = AlphaFactor(
            factor_id="ALPHA_001",
            name="RSI Momentum",
            category="momentum",
            description="RSI-based factor",
            universe=["EURUSD", "GBPUSD"],
            calculation_method="RSI(14) with z-score",
            historical_data_range={"start": "2023-01-01", "end": "2024-01-01"},
            performance_metrics={"ic": 0.05, "ir": 1.2}
        )
        assert alpha.factor_id == "ALPHA_001"
        assert alpha.performance_metrics["ic"] == 0.05


class TestTradingSchemas:
    """Tests for trading department schemas."""

    def test_order_request_market_order(self):
        """Test market order request."""
        order = OrderRequest(
            symbol="EURUSD",
            side=OrderSide.BUY,
            quantity=1.0,
            order_type=OrderType.MARKET,
            time_in_force=TimeInForce.GTC
        )
        assert order.symbol == "EURUSD"
        assert order.side == OrderSide.BUY
        assert order.order_type == OrderType.MARKET

    def test_order_request_limit_order(self):
        """Test limit order request."""
        order = OrderRequest(
            symbol="GBPUSD",
            side=OrderSide.SELL,
            quantity=2.0,
            order_type=OrderType.LIMIT,
            price=1.2650,
            time_in_force=TimeInForce.GTC
        )
        assert order.price == 1.2650
        assert order.order_type == OrderType.LIMIT

    def test_fill_info(self):
        """Test FillInfo creation."""
        fill = FillInfo(
            fill_id="FILL_001",
            order_id="ORD_001",
            fill_price=1.0850,
            fill_quantity=1.0,
            commission=-1.0,
            slippage=0.5
        )
        assert fill.fill_id == "FILL_001"
        assert fill.commission == -1.0
        assert fill.slippage == 0.5

    def test_order_response(self):
        """Test OrderResponse creation."""
        response = OrderResponse(
            order_id="ORD_001",
            symbol="EURUSD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1.0,
            status=OrderStatus.FILLED,
            avg_fill_price=1.0850
        )
        assert response.order_id == "ORD_001"
        assert response.status == OrderStatus.FILLED


class TestRiskSchemas:
    """Tests for risk department schemas."""

    def test_position_size_request(self):
        """Test PositionSizeRequest creation."""
        request = PositionSizeRequest(
            symbol="EURUSD",
            account_balance=10000.0,
            risk_percent=2.0,
            entry_price=1.0850,
            stop_loss=1.0800
        )
        assert request.symbol == "EURUSD"
        assert request.account_balance == 10000.0

    def test_position_size_response(self):
        """Test PositionSizeResponse creation."""
        response = PositionSizeResponse(
            symbol="EURUSD",
            quantity=2.0,
            risk_amount=200.0,
            risk_percent=2.0,
            pip_risk=50,
            potential_loss=200.0,
            reward_risk_ratio=2.0,
            risk_level=RiskLevel.LOW
        )
        assert response.quantity == 2.0
        assert response.risk_level == RiskLevel.LOW

    def test_drawdown_info(self):
        """Test DrawdownInfo creation."""
        info = DrawdownInfo(
            account_id="ACC_001",
            current_drawdown=-5.0,
            peak_balance=10000.0,
            current_balance=9500.0,
            max_drawdown=-12.5,
            risk_level=RiskLevel.MEDIUM
        )
        assert info.current_drawdown == -5.0
        assert info.risk_level == RiskLevel.MEDIUM

    def test_var_result(self):
        """Test VaRResult creation."""
        var = VaRResult(
            portfolio={"EURUSD": 10000, "GBPUSD": 5000},
            confidence_level=0.95,
            timeframe_days=1,
            var_absolute=250.0,
            var_percentage=1.67,
            cvar=375.0,
            method="historical",
            risk_level=RiskLevel.MEDIUM
        )
        assert var.confidence_level == 0.95
        assert var.risk_level == RiskLevel.MEDIUM

    def test_risk_limits(self):
        """Test RiskLimits creation."""
        limits = RiskLimits(
            account_id="ACC_001",
            max_daily_loss_percent=5.0,
            max_position_size=10.0,
            max_exposure_per_symbol_percent=20.0
        )
        assert limits.account_id == "ACC_001"
        assert limits.max_daily_loss_percent == 5.0


class TestPortfolioSchemas:
    """Tests for portfolio department schemas."""

    def test_allocation_request(self):
        """Test AllocationRequest creation."""
        request = AllocationRequest(
            assets=["EURUSD", "GBPUSD", "USDJPY"],
            target_return=10.0,
            max_risk=15.0,
            optimization_objective=OptimizationObjective.MAXIMIZE_SHARPE
        )
        assert len(request.assets) == 3
        assert request.optimization_objective == OptimizationObjective.MAXIMIZE_SHARPE

    def test_allocation_result(self):
        """Test AllocationResult creation."""
        from src.agents.departments.schemas.portfolio import AssetAllocation

        result = AllocationResult(
            allocation_id="ALLOC_001",
            allocations=[
                AssetAllocation(asset="EURUSD", weight=0.5, expected_return=8.0, volatility=10.0),
                AssetAllocation(asset="GBPUSD", weight=0.3, expected_return=6.0, volatility=12.0),
                AssetAllocation(asset="USDJPY", weight=0.2, expected_return=4.0, volatility=8.0)
            ],
            expected_return=6.8,
            portfolio_volatility=9.5,
            sharpe_ratio=0.72,
            optimization_objective=OptimizationObjective.MAXIMIZE_SHARPE,
            optimization_status="optimal",
            execution_time_ms=500
        )
        assert result.allocation_id == "ALLOC_001"
        assert len(result.allocations) == 3

    def test_performance_metrics(self):
        """Test PerformanceMetrics creation."""
        metrics = PerformanceMetrics(
            period="YTD",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 6, 1),
            start_value=10000.0,
            end_value=11500.0,
            total_return=15.0,
            annualized_return=30.0,
            volatility=12.0,
            sharpe_ratio=2.5,
            max_drawdown=-8.0
        )
        assert metrics.total_return == 15.0
        assert metrics.sharpe_ratio == 2.5


class TestDevelopmentSchemas:
    """Tests for development department schemas."""

    def test_ea_creation_request(self):
        """Test EACreationRequest creation."""
        request = EACreationRequest(
            name="TrendFollower",
            strategy_type="trend",
            language=EALanguage.MQL5,
            description="SMA crossover",
            symbols=["EURUSD"]
        )
        assert request.name == "TrendFollower"
        assert request.language == EALanguage.MQL5

    def test_ea_creation_response(self):
        """Test EACreationResponse creation."""
        response = EACreationResponse(
            ea_id="EA_001",
            name="TrendFollower",
            language=EALanguage.MQL5,
            status=EAStatus.CODE_GENERATED,
            file_path="/experts/TrendFollower.mq5"
        )
        assert response.ea_id == "EA_001"
        assert response.status == EAStatus.CODE_GENERATED

    def test_test_result(self):
        """Test TestResult creation."""
        result = TestResult(
            test_id="TEST_001",
            test_name="Backtest",
            ea_id="EA_001",
            symbol="EURUSD",
            timeframe="H1",
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 12, 31),
            status="passed",
            execution_time_ms=5000
        )
        assert result.test_id == "TEST_001"
        assert result.status == "passed"

    def test_deployment_config(self):
        """Test DeploymentConfig creation."""
        config = DeploymentConfig(
            ea_id="EA_001",
            symbol="EURUSD",
            parameters={"lot_size": 1.0},
            risk_mode="conservative",
            max_spread=3.0
        )
        assert config.ea_id == "EA_001"
        assert config.risk_mode == "conservative"


class TestValidation:
    """Tests for schema validation."""

    def test_order_quantity_must_be_positive(self):
        """Test that order quantity must be positive."""
        with pytest.raises(ValidationError):
            OrderRequest(
                symbol="EURUSD",
                side=OrderSide.BUY,
                quantity=-1.0,
                order_type=OrderType.MARKET
            )

    def test_position_size_account_balance_must_be_positive(self):
        """Test that account balance must be positive."""
        with pytest.raises(ValidationError):
            PositionSizeRequest(
                symbol="EURUSD",
                account_balance=-1000.0,
                risk_percent=2.0,
                entry_price=1.0850,
                stop_loss=1.0800
            )

    def test_var_confidence_level_bounds(self):
        """Test VaR confidence level bounds."""
        with pytest.raises(ValidationError):
            VaRResult(
                portfolio={"EURUSD": 10000},
                confidence_level=0.3,  # Too low
                timeframe_days=1,
                var_absolute=100.0,
                var_percentage=1.0,
                cvar=150.0,
                method="historical",
                risk_level=RiskLevel.LOW
            )

    def test_allocation_weights_sum(self):
        """Test allocation weights validation."""
        from src.agents.departments.schemas.portfolio import AssetAllocation

        # This should work (total = 1.0)
        result = AllocationResult(
            allocation_id="ALLOC_001",
            allocations=[
                AssetAllocation(asset="EURUSD", weight=0.5),
                AssetAllocation(asset="GBPUSD", weight=0.5)
            ],
            expected_return=5.0,
            portfolio_volatility=10.0,
            sharpe_ratio=0.5,
            optimization_objective=OptimizationObjective.MAXIMIZE_SHARPE,
            optimization_status="optimal",
            execution_time_ms=100
        )
        assert len(result.allocations) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
