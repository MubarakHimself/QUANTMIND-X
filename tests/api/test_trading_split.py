"""
Tests for Trading API modular structure.

Tests that the trading module can be imported and has the expected structure.
"""

import pytest


class TestTradingModuleStructure:
    """Test that the trading module has expected structure."""

    def test_trading_module_importable(self):
        """Test that trading module can be imported."""
        from src.api.trading import router
        assert router is not None

    def test_trading_has_models(self):
        """Test that models module exists."""
        from src.api.trading import models
        assert models is not None

    def test_trading_has_backtest_routes(self):
        """Test that backtest routes exist."""
        from src.api.trading import backtest
        assert backtest is not None

    def test_trading_has_data_routes(self):
        """Test that data routes exist."""
        from src.api.trading import data
        assert data is not None

    def test_trading_has_control_routes(self):
        """Test that control routes exist."""
        from src.api.trading import control
        assert control is not None


class TestTradingModels:
    """Test trading models are available."""

    def test_backtest_run_request_model(self):
        """Test BacktestRunRequest model exists."""
        from src.api.trading.models import BacktestRunRequest
        assert BacktestRunRequest is not None

    def test_emergency_stop_request_model(self):
        """Test EmergencyStopRequest model exists."""
        from src.api.trading.models import EmergencyStopRequest
        assert EmergencyStopRequest is not None

    def test_timeframe_enum(self):
        """Test Timeframe enum exists."""
        from src.api.trading.models import Timeframe
        assert Timeframe is not None


class TestTradingBackwardCompatibility:
    """Test backward compatibility with original module."""

    def test_original_module_still_works(self):
        """Test original trading_endpoints still works."""
        from src.api import trading_endpoints
        assert trading_endpoints is not None
        assert hasattr(trading_endpoints, 'create_fastapi_app')

    def test_original_models_still_available(self):
        """Test original models still accessible."""
        from src.api.trading_endpoints import (
            BacktestRunRequest,
            EmergencyStopRequest,
            Timeframe
        )
        assert BacktestRunRequest is not None
        assert EmergencyStopRequest is not None
        assert Timeframe is not None

    def test_original_handlers_still_available(self):
        """Test original handlers still accessible."""
        from src.api.trading_endpoints import (
            BacktestAPIHandler,
            TradingControlAPIHandler
        )
        assert BacktestAPIHandler is not None
        assert TradingControlAPIHandler is not None
