"""
Tests for EA Lifecycle Tools
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from src.agents.tools.ea_lifecycle import EALifecycleTools


@pytest.fixture
def temp_workspace():
    """Create temporary workspace for testing."""
    temp_dir = tempfile.mkdtemp()
    strategies_path = Path(temp_dir) / "strategies-yt"
    strategies_path.mkdir(parents=True)

    # Create sample strategy file
    strategy_file = strategies_path / "test_strategy.md"
    strategy_file.write_text("# Test Strategy\n\nA simple trading strategy.")

    yield temp_dir

    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def ea_tools(temp_workspace):
    """Create EALifecycleTools instance with temp workspace."""
    return EALifecycleTools(base_path=temp_workspace)


class TestEALifecycleTools:
    """Test EA lifecycle operations."""

    def test_create_ea(self, ea_tools):
        """Test EA creation from strategy."""
        result = ea_tools.create_ea(
            strategy_name="test_strategy",
            ea_name="TestEA",
            parameters={"risk_percent": 2.0, "lot_size": 0.1}
        )

        assert result["success"] is True
        assert result["ea_name"] == "TestEA"
        assert "file_path" in result
        assert result["strategy"] == "test_strategy"

        # Verify file was created
        ea_file = Path(result["file_path"])
        assert ea_file.exists()
        assert "TestEA" in ea_file.read_text()

    def test_create_ea_default_name(self, ea_tools):
        """Test EA creation with default name."""
        result = ea_tools.create_ea(strategy_name="test_strategy")

        assert result["success"] is True
        assert result["ea_name"] == "test_strategy_EA"

    def test_create_ea_missing_strategy(self, ea_tools):
        """Test EA creation with non-existent strategy."""
        result = ea_tools.create_ea(strategy_name="nonexistent")

        assert result["success"] is False
        assert "not found" in result["error"]

    def test_validate_ea_success(self, ea_tools):
        """Test EA validation with valid code."""
        # First create an EA
        ea_tools.create_ea(strategy_name="test_strategy", ea_name="ValidEA")

        result = ea_tools.validate_ea("ValidEA")

        assert result["success"] is True
        assert result["valid"] is True
        assert len(result["errors"]) == 0

    def test_validate_ea_missing_file(self, ea_tools):
        """Test validation with non-existent EA."""
        result = ea_tools.validate_ea("NonExistent")

        assert result["success"] is False
        assert "not found" in result["error"]

    def test_backtest_ea(self, ea_tools):
        """Test EA backtest queuing."""
        ea_tools.create_ea(strategy_name="test_strategy", ea_name="BacktestEA")

        result = ea_tools.backtest_ea(
            ea_name="BacktestEA",
            symbol="GBPUSD",
            timeframe="M15",
            deposit=5000
        )

        assert result["success"] is True
        assert result["ea_name"] == "BacktestEA"
        assert result["symbol"] == "GBPUSD"
        assert result["status"] == "queued"

    def test_backtest_ea_missing(self, ea_tools):
        """Test backtest with non-existent EA."""
        result = ea_tools.backtest_ea(ea_name="MissingEA")

        assert result["success"] is False
        assert "not found" in result["error"]

    def test_deploy_paper(self, ea_tools):
        """Test paper trading deployment."""
        ea_tools.create_ea(strategy_name="test_strategy", ea_name="PaperEA")

        result = ea_tools.deploy_paper(
            ea_name="PaperEA",
            account_id="paper_123"
        )

        assert result["success"] is True
        assert result["ea_name"] == "PaperEA"
        assert result["environment"] == "paper"
        assert result["status"] == "deployed"
        assert result["account_id"] == "paper_123"

    def test_deploy_paper_validation_failure(self, ea_tools):
        """Test deployment with invalid EA."""
        # Try to deploy non-existent EA
        result = ea_tools.deploy_paper(ea_name="InvalidEA")

        assert result["success"] is False

    def test_stop_ea(self, ea_tools):
        """Test stopping a running EA."""
        result = ea_tools.stop_ea(ea_name="RunningEA", environment="paper")

        assert result["success"] is True
        assert result["ea_name"] == "RunningEA"
        assert result["environment"] == "paper"
        assert result["status"] == "stopped"

    def test_full_lifecycle(self, ea_tools):
        """Test complete EA lifecycle."""
        # Create
        create_result = ea_tools.create_ea(
            strategy_name="test_strategy",
            ea_name="LifecycleEA"
        )
        assert create_result["success"] is True

        # Validate
        validate_result = ea_tools.validate_ea("LifecycleEA")
        assert validate_result["valid"] is True

        # Backtest
        backtest_result = ea_tools.backtest_ea(ea_name="LifecycleEA")
        assert backtest_result["success"] is True

        # Deploy to paper
        deploy_result = ea_tools.deploy_paper(ea_name="LifecycleEA")
        assert deploy_result["success"] is True

        # Stop
        stop_result = ea_tools.stop_ea(ea_name="LifecycleEA")
        assert stop_result["success"] is True
