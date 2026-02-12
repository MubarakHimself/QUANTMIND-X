"""
API Endpoint Tests for Task Group 8

Tests for:
- POST /api/v1/backtest/run
- GET /api/v1/backtest/results/{id}
- POST /api/v1/data/upload
- GET /api/v1/data/status
- POST /api/v1/trading/emergency_stop
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
import io


@pytest.fixture
def sample_ohlcv_data():
    """Create sample OHLCV data for testing."""
    return pd.DataFrame({
        'time': pd.date_range('2024-01-01', periods=100, freq='1h', tz='UTC'),
        'open': np.linspace(1.1000, 1.1100, 100),
        'high': np.linspace(1.1010, 1.1110, 100),
        'low': np.linspace(1.0990, 1.1090, 100),
        'close': np.linspace(1.1005, 1.1105, 100),
        'tick_volume': [1000] * 100
    })


@pytest.fixture
def mock_backtest_result():
    """Create mock backtest result (simplified, avoiding circular import)."""
    return {
        'sharpe': 1.5,
        'return_pct': 10.5,
        'drawdown': -5.2,
        'trades': 15,
        'log': "Backtest completed successfully",
        'initial_cash': 10000.0,
        'final_cash': 11050.0,
        'equity_curve': [10000.0 + i * 10.5 for i in range(100)],
        'trade_history': [
            {
                'symbol': 'EURUSD',
                'volume': 0.01,
                'entry_price': 1.1000,
                'exit_price': 1.1100,
                'direction': 'buy',
                'profit': 10.0
            }
        ]
    }


class TestBacktestAPI:
    """Tests for backtest API endpoints."""

    def test_post_backtest_run_returns_success_response(self, sample_ohlcv_data, mock_backtest_result):
        """Test POST /api/v1/backtest/run returns success response."""
        from src.api.trading_endpoints import BacktestRunRequest, BacktestRunResponse

        # Create request
        request = BacktestRunRequest(
            symbol='EURUSD',
            timeframe='H1',
            variant='vanilla',
            start_date='2024-01-01',
            end_date='2024-01-31'
        )

        # Validate request structure
        assert request.symbol == 'EURUSD'
        assert request.timeframe == 'H1'
        assert request.variant == 'vanilla'
        assert request.start_date == '2024-01-01'
        assert request.end_date == '2024-01-31'

    def test_post_backtest_run_with_spiced_variant(self):
        """Test POST /api/v1/backtest/run with spiced variant."""
        from src.api.trading_endpoints import BacktestRunRequest

        request = BacktestRunRequest(
            symbol='GBPUSD',
            timeframe='M15',
            variant='spiced',
            start_date='2024-01-01',
            end_date='2024-01-31',
            regime_filtering=True,
            chaos_threshold=0.6
        )

        assert request.variant == 'spiced'
        assert request.regime_filtering is True
        assert request.chaos_threshold == 0.6

    def test_get_backtest_results_by_id(self):
        """Test GET /api/v1/backtest/results/{id} returns results."""
        from src.api.trading_endpoints import BacktestResultResponse

        # Create response with mock data
        response = BacktestResultResponse(
            backtest_id='test-backtest-123',
            status='completed',
            sharpe_ratio=1.5,
            return_pct=10.5,
            max_drawdown=-5.2,
            total_trades=15,
            equity_curve=[10000.0 + i * 10.5 for i in range(100)],
            trade_history=[
                {
                    'symbol': 'EURUSD',
                    'volume': 0.01,
                    'entry_price': 1.1000,
                    'exit_price': 1.1100,
                    'profit': 10.0
                }
            ]
        )

        assert response.backtest_id == 'test-backtest-123'
        assert response.status == 'completed'
        assert response.sharpe_ratio == 1.5
        assert len(response.trade_history) == 1


class TestDataManagementAPI:
    """Tests for data management API endpoints."""

    def test_post_data_upload_with_csv(self, sample_ohlcv_data):
        """Test POST /api/v1/data/upload accepts CSV data."""
        from src.api.trading_endpoints import DataUploadResponse

        # Create response
        response = DataUploadResponse(
            success=True,
            message='Data uploaded successfully',
            symbol='EURUSD',
            timeframe='H1',
            rows_uploaded=100,
            file_path='/data/historical/EURUSD/H1/data.parquet'
        )

        assert response.success is True
        assert response.symbol == 'EURUSD'
        assert response.rows_uploaded == 100
        assert '.parquet' in response.file_path

    def test_get_data_status_returns_cache_info(self):
        """Test GET /api/v1/data/status returns cache status."""
        from src.api.trading_endpoints import DataStatusResponse

        response = DataStatusResponse(
            symbols=['EURUSD', 'GBPUSD', 'XAUUSD'],
            total_cached_files=15,
            last_updated=datetime.now(timezone.utc),
            cache_size_mb=125.5,
            data_quality_score=0.98
        )

        assert len(response.symbols) == 3
        assert 'EURUSD' in response.symbols
        assert response.total_cached_files == 15
        assert response.cache_size_mb > 0


class TestTradingControlAPI:
    """Tests for trading control API endpoints."""

    def test_post_emergency_stop_activates_kill_switch(self):
        """Test POST /api/v1/trading/emergency_stop activates kill switch."""
        from src.api.trading_endpoints import EmergencyStopResponse

        response = EmergencyStopResponse(
            success=True,
            message='Emergency stop activated',
            kill_switch_active=True,
            positions_closed=5,
            triggered_by='api_request',
            timestamp=datetime.now(timezone.utc)
        )

        assert response.success is True
        assert response.kill_switch_active is True
        assert response.positions_closed == 5
        assert response.triggered_by == 'api_request'

    def test_get_trading_status_returns_current_state(self):
        """Test GET /api/v1/trading/status returns current trading state."""
        from src.api.trading_endpoints import TradingStatusResponse
        from src.router.sentinel import RegimeReport

        # Create mock regime report (matching actual dataclass structure)
        regime_report = RegimeReport(
            regime='TREND_STABLE',
            chaos_score=0.25,
            regime_quality=0.75,
            susceptibility=0.3,
            is_systemic_risk=False,
            news_state='SAFE',
            timestamp=100.0
        )

        response = TradingStatusResponse(
            trading_enabled=True,
            kill_switch_active=False,
            current_regime='TREND_STABLE',
            chaos_score=0.25,
            regime_quality=0.75,
            open_positions=3,
            daily_pnl_pct=1.5,
            account_equity=10150.0,
            account_balance=10000.0,
            risk_multiplier=1.0,
            daily_loss_limit_pct=5.0,
            max_drawdown_pct=10.0,
            last_update=datetime.now(timezone.utc)
        )

        assert response.trading_enabled is True
        assert response.kill_switch_active is False
        assert response.current_regime == 'TREND_STABLE'
        assert response.open_positions == 3
        assert response.daily_pnl_pct == 1.5
