"""
Tests for Backtest Results API Endpoints

Story: 4-4-backtest-results-api

Tests:
- Task 4.1 - Unit tests for each endpoint response schema
- Task 4.2 - Integration tests verifying backtest engine wiring
"""

import pytest
from pydantic import ValidationError
from datetime import datetime, timezone

from src.api.backtest_endpoints import (
    BacktestSummary,
    BacktestDetail,
    RunningBacktest,
    BacktestMode,
    ReportType,
    _completed_backtests,
    _running_backtests,
    refresh_backtests_from_storage,
)


class TestBacktestSummarySchema:
    """Unit tests for BacktestSummary response model."""

    def test_valid_summary_creation(self):
        """Test creating a valid BacktestSummary."""
        data = {
            "id": "test-001",
            "ea_name": "TestStrategy",
            "mode": "VANILLA",
            "run_at_utc": "2026-03-01T12:00:00Z",
            "net_pnl": 10.5,
            "sharpe": 1.5,
            "max_drawdown": 5.0,
            "win_rate": 60.0,
        }
        summary = BacktestSummary(**data)
        assert summary.id == "test-001"
        assert summary.ea_name == "TestStrategy"
        assert summary.mode == BacktestMode.VANILLA
        assert summary.net_pnl == 10.5

    def test_summary_all_modes(self):
        """Test all 6 confirmed working modes."""
        for mode in BacktestMode:
            data = {
                "id": f"test-{mode.value}",
                "ea_name": "TestStrategy",
                "mode": mode.value,
                "run_at_utc": "2026-03-01T12:00:00Z",
                "net_pnl": 10.5,
                "sharpe": 1.5,
                "max_drawdown": 5.0,
                "win_rate": 60.0,
            }
            summary = BacktestSummary(**data)
            assert summary.mode == mode

    def test_summary_required_fields(self):
        """Test that all required fields are enforced."""
        with pytest.raises(ValidationError):
            BacktestSummary(
                id="test-001",
                # Missing required fields
            )


class TestBacktestDetailSchema:
    """Unit tests for BacktestDetail response model."""

    def test_valid_detail_creation(self):
        """Test creating a valid BacktestDetail with full data."""
        data = {
            "id": "test-001",
            "ea_name": "TestStrategy",
            "mode": "VANILLA",
            "run_at_utc": "2026-03-01T12:00:00Z",
            "net_pnl": 10.5,
            "sharpe": 1.5,
            "max_drawdown": 5.0,
            "win_rate": 60.0,
            "equity_curve": [
                {"timestamp": "2026-01-01T00:00:00Z", "equity": 10000},
                {"timestamp": "2026-02-01T00:00:00Z", "equity": 11000},
            ],
            "trade_distribution": [
                {"bin": "0-10", "count": 5},
                {"bin": "10-20", "count": 3},
            ],
            "mode_params": {"lookback_period": 20, "entry_threshold": 0.02},
        }
        detail = BacktestDetail(**data)
        assert detail.id == "test-001"
        assert len(detail.equity_curve) == 2
        assert len(detail.trade_distribution) == 2
        assert detail.mode_params["lookback_period"] == 20

    def test_detail_default_values(self):
        """Test that default values are applied."""
        data = {
            "id": "test-002",
            "ea_name": "TestStrategy",
            "mode": "SPICED",
            "run_at_utc": "2026-03-01T12:00:00Z",
            "net_pnl": 10.5,
            "sharpe": 1.5,
            "max_drawdown": 5.0,
            "win_rate": 60.0,
        }
        detail = BacktestDetail(**data)
        assert detail.equity_curve == []
        assert detail.trade_distribution == []
        assert detail.mode_params == {}
        assert detail.total_trades == 0
        assert detail.profit_factor == 0.0


class TestRunningBacktestSchema:
    """Unit tests for RunningBacktest response model."""

    def test_valid_running_backtest(self):
        """Test creating a valid RunningBacktest."""
        data = {
            "id": "running-001",
            "ea_name": "TestStrategy",
            "mode": "SPICED_FULL",
            "progress_pct": 50.0,
            "started_at_utc": "2026-03-01T12:00:00Z",
            "partial_metrics": {"net_pnl": 5.0, "drawdown": 2.0, "sharpe": 1.2},
        }
        running = RunningBacktest(**data)
        assert running.id == "running-001"
        assert running.progress_pct == 50.0
        assert running.partial_metrics["net_pnl"] == 5.0

    def test_progress_bounds(self):
        """Test that progress_pct is validated (0-100)."""
        # Valid bounds should work
        data = {
            "id": "running-002",
            "ea_name": "TestStrategy",
            "mode": "VANILLA",
            "progress_pct": 0.0,
            "started_at_utc": "2026-03-01T12:00:00Z",
            "partial_metrics": {},
        }
        running = RunningBacktest(**data)
        assert running.progress_pct == 0.0

        data["progress_pct"] = 100.0
        running = RunningBacktest(**data)
        assert running.progress_pct == 100.0

    def test_progress_invalid_rejected(self):
        """Test that invalid progress values are rejected."""
        with pytest.raises(ValidationError):
            RunningBacktest(
                id="running-003",
                ea_name="TestStrategy",
                mode="VANILLA",
                progress_pct=150.0,  # Invalid: > 100
                started_at_utc="2026-03-01T12:00:00Z",
                partial_metrics={},
            )


class TestBacktestModes:
    """Test BacktestMode enum values."""

    def test_all_six_modes(self):
        """Verify all 6 confirmed working modes are present."""
        expected_modes = {
            "VANILLA",
            "SPICED",
            "VANILLA_FULL",
            "SPICED_FULL",
            "MODE_B",
            "MODE_C",
        }
        actual_modes = {mode.value for mode in BacktestMode}
        assert expected_modes == actual_modes


class TestReportTypes:
    """Test ReportType enum values."""

    def test_all_report_types(self):
        """Verify all report types are present."""
        expected = {"basic", "monte_carlo", "walk_forward", "pbo"}
        actual = {rt.value for rt in ReportType}
        assert expected == actual


class TestStorageBacktests:
    """Backtests now come from persisted artifacts, never seeded demo rows."""

    def test_refresh_storage_empty_state(self, monkeypatch, tmp_path):
        monkeypatch.setattr("src.api.backtest_endpoints.iter_strategy_roots", lambda: iter([]))
        refresh_backtests_from_storage()
        assert _completed_backtests == {}
        assert _running_backtests == {}

    def test_refresh_storage_reads_completed_and_running(self, monkeypatch, tmp_path):
        strategy_root = tmp_path / "shared_assets" / "strategies" / "scalping" / "single-videos" / "london_scalper"
        backtests_dir = strategy_root / "backtests"
        reports_dir = strategy_root / "reports"
        backtests_dir.mkdir(parents=True, exist_ok=True)
        reports_dir.mkdir(parents=True, exist_ok=True)

        (backtests_dir / "bt-001.json").write_text(
            """
            {
              "id": "bt-001",
              "ea_name": "LondonScalper_v1",
              "mode": "VANILLA",
              "run_at_utc": "2026-03-01T12:00:00Z",
              "net_pnl": 12.5,
              "sharpe": 1.8,
              "max_drawdown": 4.2,
              "win_rate": 61.0,
              "equity_curve": [{"timestamp":"2026-03-01T12:00:00Z","equity":10000}],
              "trade_distribution": [],
              "mode_params": {"symbol":"EURUSD"},
              "report_type": "basic",
              "total_trades": 25,
              "profit_factor": 1.7,
              "avg_trade_pnl": 0.5
            }
            """,
            encoding="utf-8",
        )
        (reports_dir / "bt-running-001.json").write_text(
            """
            {
              "id": "bt-running-001",
              "ea_name": "LondonScalper_v2",
              "mode": "SPICED",
              "status": "running",
              "progress_pct": 45.0,
              "started_at_utc": "2026-03-02T08:00:00Z",
              "partial_metrics": {"net_pnl": 3.0}
            }
            """,
            encoding="utf-8",
        )

        monkeypatch.setattr("src.api.backtest_endpoints.iter_strategy_roots", lambda: iter([strategy_root]))
        refresh_backtests_from_storage()

        assert "bt-001" in _completed_backtests
        assert _completed_backtests["bt-001"].mode == BacktestMode.VANILLA
        assert _completed_backtests["bt-001"].net_pnl == 12.5
        assert "bt-running-001" in _running_backtests
        assert _running_backtests["bt-running-001"].progress_pct == 45.0


class TestEdgeCases:
    """Test edge cases for validation."""

    def test_negative_pnl_allowed(self):
        """Negative PnL should be allowed (losses are valid)."""
        data = {
            "id": "loss-001",
            "ea_name": "BadStrategy",
            "mode": "VANILLA",
            "run_at_utc": "2026-03-01T12:00:00Z",
            "net_pnl": -15.5,
            "sharpe": -0.5,
            "max_drawdown": 25.0,
            "win_rate": 30.0,
        }
        summary = BacktestSummary(**data)
        assert summary.net_pnl == -15.5

    def test_zero_win_rate(self):
        """Zero win rate should be valid."""
        data = {
            "id": "zero-win-001",
            "ea_name": "WorstStrategy",
            "mode": "VANILLA",
            "run_at_utc": "2026-03-01T12:00:00Z",
            "net_pnl": -10.0,
            "sharpe": -1.0,
            "max_drawdown": 10.0,
            "win_rate": 0.0,
        }
        summary = BacktestSummary(**data)
        assert summary.win_rate == 0.0

    # Note: Empty EA name is not currently validated by Pydantic
    # This is acceptable as the field is required (non-None) but not length-validated
