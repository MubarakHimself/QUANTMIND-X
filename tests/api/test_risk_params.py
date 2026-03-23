# tests/api/test_risk_params.py
"""
Tests for Risk Parameters API Endpoints (Story 4.2)

Tests for:
- GET /api/risk/params/{account_tag}
- PUT /api/risk/params/{account_tag}
- Validation (Kelly fraction > 1.0 → 422)
- Audit logging
"""

import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI
from pydantic import ValidationError


class TestValidation:
    """Test validation for risk parameters (AC #5)."""

    def test_kelly_fraction_greater_than_one_rejected(self):
        """Should reject Kelly fraction > 1.0 with 422."""
        from src.api.risk_endpoints import RiskParamsUpdateRequest

        with pytest.raises(ValidationError) as exc_info:
            RiskParamsUpdateRequest(kelly_fraction=1.5)

        assert "kelly_fraction" in str(exc_info.value)

    def test_kelly_fraction_equal_to_one_accepted(self):
        """Should accept Kelly fraction = 1.0."""
        from src.api.risk_endpoints import RiskParamsUpdateRequest

        # Should not raise - 1.0 is valid
        request = RiskParamsUpdateRequest(kelly_fraction=1.0)
        assert request.kelly_fraction == 1.0

    def test_kelly_fraction_zero_rejected(self):
        """Should reject Kelly fraction = 0."""
        from src.api.risk_endpoints import RiskParamsUpdateRequest

        with pytest.raises(ValidationError):
            RiskParamsUpdateRequest(kelly_fraction=0.0)

    def test_daily_loss_cap_validates_upper_range(self):
        """Should validate daily_loss_cap_pct is within range (0-100)."""
        from src.api.risk_endpoints import RiskParamsUpdateRequest

        # > 100 should fail
        with pytest.raises(ValidationError):
            RiskParamsUpdateRequest(daily_loss_cap_pct=150.0)

    def test_daily_loss_cap_validates_lower_range(self):
        """Should validate daily_loss_cap_pct is > 0."""
        from src.api.risk_endpoints import RiskParamsUpdateRequest

        # <= 0 should fail
        with pytest.raises(ValidationError):
            RiskParamsUpdateRequest(daily_loss_cap_pct=0)

    def test_max_trades_per_day_validates_positive(self):
        """Should validate max_trades_per_day is >= 1."""
        from src.api.risk_endpoints import RiskParamsUpdateRequest

        # < 1 should fail
        with pytest.raises(ValidationError):
            RiskParamsUpdateRequest(max_trades_per_day=0)

    def test_position_multiplier_range(self):
        """Should validate position_multiplier range (0.1-10.0)."""
        from src.api.risk_endpoints import RiskParamsUpdateRequest

        # Too low
        with pytest.raises(ValidationError):
            RiskParamsUpdateRequest(position_multiplier=0.01)

        # Too high
        with pytest.raises(ValidationError):
            RiskParamsUpdateRequest(position_multiplier=20.0)

    def test_lyapunov_threshold_range(self):
        """Should validate lyapunov_threshold range (0-1)."""
        from src.api.risk_endpoints import RiskParamsUpdateRequest

        # > 1 should fail
        with pytest.raises(ValidationError):
            RiskParamsUpdateRequest(lyapunov_threshold=1.5)

    def test_hmm_retrain_trigger_range(self):
        """Should validate hmm_retrain_trigger range (0-1)."""
        from src.api.risk_endpoints import RiskParamsUpdateRequest

        # > 1 should fail
        with pytest.raises(ValidationError):
            RiskParamsUpdateRequest(hmm_retrain_trigger=2.0)

    def test_partial_update_allows_optional_fields(self):
        """Should allow partial updates - only required fields need to be provided."""
        from src.api.risk_endpoints import RiskParamsUpdateRequest

        # Just kelly_fraction
        request = RiskParamsUpdateRequest(kelly_fraction=0.25)
        assert request.kelly_fraction == 0.25
        assert request.daily_loss_cap_pct is None

        # Multiple fields
        request = RiskParamsUpdateRequest(
            kelly_fraction=0.3,
            max_trades_per_day=5
        )
        assert request.kelly_fraction == 0.3
        assert request.max_trades_per_day == 5


class TestPropFirmModels:
    """Test Prop Firm request/response models."""

    def test_prop_firm_create_accepts_valid_risk_modes(self):
        """Should accept valid risk_mode values: growth, scaling, guardian."""
        from src.api.risk_endpoints import PropFirmCreateRequest

        # Valid modes - should not raise
        for mode in ['growth', 'scaling', 'guardian']:
            request = PropFirmCreateRequest(
                firm_name="Test Firm",
                account_id="12345",
                daily_loss_limit_pct=5.0,
                target_profit_pct=10.0,
                risk_mode=mode
            )
            assert request.risk_mode == mode

    def test_prop_firm_create_validates_daily_loss_limit(self):
        """Should validate daily_loss_limit_pct range."""
        from src.api.risk_endpoints import PropFirmCreateRequest

        # Too high
        with pytest.raises(ValidationError):
            PropFirmCreateRequest(
                firm_name="Test Firm",
                account_id="12345",
                daily_loss_limit_pct=150.0,
                target_profit_pct=10.0,
                risk_mode="growth"
            )

    def test_prop_firm_update_allows_partial(self):
        """PropFirmUpdateRequest should allow partial updates."""
        from src.api.risk_endpoints import PropFirmUpdateRequest

        request = PropFirmUpdateRequest(firm_name="New Name")
        assert request.firm_name == "New Name"
        assert request.daily_loss_limit_pct is None


class TestCalendarRuleBug:
    """Test for calendar rule bug fix (Issue #1 from code review)."""

    def test_blacklist_enabled_assignment_not_overwritten(self):
        """Verify blacklist_enabled is not overwritten by blackout_minutes."""
        from src.api.risk_endpoints import CalendarRuleUpdateRequest

        # Create update request with blacklist_enabled
        request = CalendarRuleUpdateRequest(
            blacklist_enabled=False,
            blackout_minutes=60
        )

        # Verify both values are preserved correctly
        assert request.blacklist_enabled == False
        assert request.blackout_minutes == 60
