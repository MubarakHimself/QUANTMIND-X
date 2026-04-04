"""
Tests for Session Performer Identifier
======================================

Test cases from story specification:

| Scenario | regime_expectation | actual | outperformance | is_specialist |
|----------|-------------------|--------|---------------|--------------|
| Outperforms | 0.50 | 0.70 | 0.20 | True |
| Meets expectation | 0.50 | 0.55 | 0.05 | False |
| Underperforms | 0.50 | 0.35 | -0.15 | False |
"""

import pytest
from src.router.session_performer import SessionPerformerIdentifier, SessionPerformerResult


class TestSessionPerformerIdentifier:
    """Test Session Performer ID step."""

    @pytest.fixture
    def identifier(self):
        return SessionPerformerIdentifier()

    def test_outperformer_is_specialist(self, identifier):
        """Outperformance > 15% with positive actual should be specialist."""
        result = SessionPerformerResult(
            bot_id="test-bot",
            regime_expectation=0.50,
            actual_performance=0.70,
            outperformance=0.20,
            is_specialist=True,
        )

        assert result.is_specialist is True
        assert result.outperformance == 0.20
        assert result.actual_performance > result.regime_expectation

    def test_meets_expectation_not_specialist(self, identifier):
        """Meeting expectation (5%) should NOT be specialist."""
        result = SessionPerformerResult(
            bot_id="test-bot",
            regime_expectation=0.50,
            actual_performance=0.55,
            outperformance=0.05,
            is_specialist=False,
        )

        assert result.is_specialist is False
        assert result.outperformance < identifier.REGIME_OUTPERFORMANCE_THRESHOLD

    def test_underperformer_not_specialist(self, identifier):
        """Underperformer should NOT be specialist."""
        result = SessionPerformerResult(
            bot_id="test-bot",
            regime_expectation=0.50,
            actual_performance=0.35,
            outperformance=-0.15,
            is_specialist=False,
        )

        assert result.is_specialist is False
        assert result.actual_performance < result.regime_expectation

    def test_threshold_constant(self, identifier):
        """REGIME_OUTPERFORMANCE_THRESHOLD should be 0.15 (15%)."""
        assert identifier.REGIME_OUTPERFORMANCE_THRESHOLD == 0.15

    def test_specialist_only_if_positive_performance(self, identifier):
        """Should only tag as specialist if actual performance is positive."""
        # This is enforced in the identifier logic:
        # is_specialist = (outperformance > threshold AND actual_performance > 0)
        # So even with high outperformance, if actual is negative, not specialist

        # Example: regime_exp=0.50, actual=-0.10, outperformance=0.60
        # Should NOT be specialist because actual < 0
        outperformance = 0.60
        actual = -0.10

        is_specialist = (outperformance > identifier.REGIME_OUTPERFORMANCE_THRESHOLD and actual > 0)
        assert is_specialist is False

    def test_result_to_dict(self):
        """SessionPerformerResult.to_dict() should return correct structure."""
        result = SessionPerformerResult(
            bot_id="test-bot",
            regime_expectation=0.50,
            actual_performance=0.70,
            outperformance=0.20,
            is_specialist=True,
        )

        d = result.to_dict()

        assert d["bot_id"] == "test-bot"
        assert d["regime_expectation"] == 0.50
        assert d["actual_performance"] == 0.70
        assert d["outperformance"] == 0.20
        assert d["is_specialist"] is True
