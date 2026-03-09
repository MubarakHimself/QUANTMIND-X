"""
Tests for Risk Mode Implementation.

Tests the risk_mode selection feature for position sizing:
- fixed: Always use default risk percentage
- dynamic: Use Kelly calculation with fraction (normal behavior)
- conservative: Cap at 50% of default risk
"""

import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone

from src.router.enhanced_governor import EnhancedGovernor
from src.router.sentinel import RegimeReport


class TestRiskMode:
    """Test suite for risk mode selection in EnhancedGovernor."""

    @pytest.fixture
    def regime_report(self):
        """Create a standard regime report for testing."""
        return RegimeReport(
            regime='TREND_STABLE',
            chaos_score=0.2,
            regime_quality=1.0,
            susceptibility=0.1,
            is_systemic_risk=False,
            news_state='NONE',
            timestamp=datetime.now(timezone.utc).timestamp()
        )

    @pytest.fixture
    def trade_proposal(self):
        """Create a standard trade proposal for testing."""
        return {
            'symbol': 'EURUSD',
            'broker_id': 'mt5_default',
            'stop_loss_pips': 20.0,
            'win_rate': 0.55,
            'avg_win': 400.0,
            'avg_loss': 200.0,
            'current_atr': 0.0012,
            'average_atr': 0.0010,
            'current_balance': 100000.0,
        }

    def test_conservative_mode_limits_risk(self):
        """Conservative mode should use lower risk % regardless of Kelly."""
        governor = EnhancedGovernor()
        governor.risk_mode = "conservative"

        # Even with high Kelly, conservative should cap lower
        kelly_fraction = governor._calculate_kelly_fraction(win_rate=0.6, payoff_ratio=2.0)

        # Should be capped at conservative limit (e.g., 50% of default risk)
        # With default risk of 2%, conservative should cap at 1%
        assert kelly_fraction <= 0.01  # 1% max in conservative

    def test_fixed_mode_ignores_kelly(self):
        """Fixed mode should always use default risk percentage."""
        governor = EnhancedGovernor()
        governor.risk_mode = "fixed"

        # Should always return default risk regardless of Kelly
        kelly_fraction = governor._calculate_kelly_fraction(win_rate=0.6, payoff_ratio=2.0)

        # Should be the default risk percentage (2%)
        assert kelly_fraction == 0.02

    def test_dynamic_mode_uses_kelly(self):
        """Dynamic mode should use Kelly calculation with fraction."""
        governor = EnhancedGovernor()
        governor.risk_mode = "dynamic"

        # Should use Kelly calculation
        kelly_fraction = governor._calculate_kelly_fraction(win_rate=0.55, payoff_ratio=2.0)

        # Dynamic mode uses Kelly calculation, which should be between 0 and default risk
        assert kelly_fraction > 0
        assert kelly_fraction <= 0.02  # Should not exceed default risk

    def test_risk_mode_defaults_to_dynamic(self):
        """Risk mode should default to dynamic."""
        governor = EnhancedGovernor()

        # Default should be dynamic
        assert hasattr(governor, 'risk_mode')
        assert governor.risk_mode == "dynamic"

    def test_conservative_with_high_kelly_still_capped(self):
        """Conservative mode should cap even with very high Kelly."""
        governor = EnhancedGovernor()
        governor.risk_mode = "conservative"

        # Very high Kelly would suggest 10%+ risk
        kelly_fraction = governor._calculate_kelly_fraction(win_rate=0.8, payoff_ratio=3.0)

        # Should still be capped at 50% of default (1%)
        assert kelly_fraction <= 0.01
