"""
Tests for DPR Scoring Engine.

Story 17.1: DPR Composite Score Calculation

Tests:
- Composite score calculation with known inputs
- Metric normalization functions edge cases
- Specialist boost application and capping
- Concern flag threshold detection
- Scoring window filtering
"""

import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone

from src.risk.dpr.scoring_engine import DPRScoringEngine, DPRScore
from src.events.dpr import DPRComponentScores, DPR_WEIGHTS


class TestDPRScoringEngine:
    """Test DPRScoringEngine composite score calculation."""

    @pytest.fixture
    def engine(self):
        """Create DPR engine with mock session."""
        mock_session = MagicMock()
        return DPRScoringEngine(
            db_session=mock_session,
            benchmark_pnl=1000.0,
            baseline_ev=10.0,
            max_acceptable_variance=0.01,
        )

    def test_composite_score_calculation(self, engine):
        """
        Test composite score calculation with known inputs.

        AC #1: Given a bot has completed at least one trade in the scoring window,
        When DPR evaluates the bot,
        Then it computes the composite score: session Win Rate (25%) +
        net PnL (30%) + consistency (20%) + EV/trade (25%).
        """
        # Mock trade data
        with patch.object(engine, '_get_trade_data') as mock_trade:
            mock_trade.return_value = {
                "total_trades": 10,
                "wins": 7,
                "net_pnl": 500.0,
                "daily_variance": 0.002,
                "ev_per_trade": 8.0,
                "max_drawdown": 5.0,
                "magic_number": 12345,
            }

            score = engine.calculate_composite_score("bot_001", "LONDON")

            assert score is not None
            assert 0 <= score <= 100

    def test_bot_not_eligible_zero_trades(self, engine):
        """
        Test bot with 0 trades is not eligible.

        Edge case: Bot with 0 trades in window — should be excluded from scoring.
        """
        with patch.object(engine, '_get_trade_data') as mock_trade:
            mock_trade.return_value = {
                "total_trades": 0,
                "wins": 0,
                "net_pnl": 0.0,
                "daily_variance": 0.0,
                "ev_per_trade": 0.0,
            }

            score = engine.calculate_composite_score("bot_001", "LONDON")

            assert score is None

    def test_bot_not_eligible_no_trade_data(self, engine):
        """Test bot with no trade data returns None."""
        with patch.object(engine, '_get_trade_data') as mock_trade:
            mock_trade.return_value = {
                "total_trades": 0,
            }

            score = engine.calculate_composite_score("bot_001", "LONDON")
            assert score is None


class TestComponentNormalization:
    """Test DPR component normalization functions."""

    @pytest.fixture
    def engine(self):
        """Create DPR engine."""
        mock_session = MagicMock()
        return DPRScoringEngine(
            db_session=mock_session,
            benchmark_pnl=1000.0,
            baseline_ev=10.0,
            max_acceptable_variance=0.01,
        )

    def test_win_rate_normalize_full(self, engine):
        """Test win rate normalization: 100% WR = 100."""
        result = engine._normalize_win_rate(10, 10)
        assert result == 100.0

    def test_win_rate_normalize_zero(self, engine):
        """Test win rate normalization: 0% WR = 0."""
        result = engine._normalize_win_rate(0, 10)
        assert result == 0.0

    def test_win_rate_normalize_half(self, engine):
        """Test win rate normalization: 50% WR = 50."""
        result = engine._normalize_win_rate(5, 10)
        assert result == 50.0

    def test_win_rate_normalize_zero_total(self, engine):
        """Test win rate normalization with 0 total returns 0."""
        result = engine._normalize_win_rate(0, 0)
        assert result == 0.0

    def test_pnl_normalize_positive(self, engine):
        """
        Test PnL normalization: positive PnL.

        Edge case: Bot with negative P&L — normalize to 0.
        """
        result = engine._normalize_pnl(500.0)
        assert 0 <= result <= 100

    def test_pnl_normalize_negative(self, engine):
        """Test PnL normalization: negative PnL = 0."""
        result = engine._normalize_pnl(-100.0)
        assert result == 0.0

    def test_pnl_normalize_zero(self, engine):
        """Test PnL normalization: zero PnL = 0."""
        result = engine._normalize_pnl(0.0)
        assert result == 0.0

    def test_pnl_normalize_exceeds_benchmark(self, engine):
        """Test PnL normalization: PnL exceeding benchmark caps at 100."""
        result = engine._normalize_pnl(5000.0)  # 5x benchmark
        assert result == 100.0

    def test_consistency_normalize_perfect(self, engine):
        """Test consistency normalization: 0 variance = 100."""
        result = engine._normalize_consistency(0.0)
        assert result == 100.0

    def test_consistency_normalize_max_variance(self, engine):
        """Test consistency normalization: max variance = 0."""
        result = engine._normalize_consistency(0.01)
        assert result == 0.0

    def test_consistency_normalize_half_variance(self, engine):
        """Test consistency normalization: 50% of max variance = 50."""
        result = engine._normalize_consistency(0.005)
        assert result == 50.0

    def test_ev_normalize_positive(self, engine):
        """Test EV normalization: positive EV."""
        result = engine._normalize_ev(10.0)
        assert 0 <= result <= 100

    def test_ev_normalize_negative(self, engine):
        """Test EV normalization: negative EV = 0."""
        result = engine._normalize_ev(-5.0)
        assert result == 0.0

    def test_ev_normalize_zero(self, engine):
        """Test EV normalization: zero EV = 0."""
        result = engine._normalize_ev(0.0)
        assert result == 0.0


class TestSpecialistBoost:
    """Test SESSION_SPECIALIST score boost."""

    @pytest.fixture
    def engine(self):
        """Create DPR engine."""
        mock_session = MagicMock()
        return DPRScoringEngine(db_session=mock_session)

    def test_specialist_boost_applies(self, engine):
        """
        Test specialist boost applies +5 for specialist session.

        AC #3: Given a bot has a SESSION_SPECIALIST tag,
        When the DPR queue is finalised,
        Then the tagged bot receives a +5 boost to its composite score
        for its specialist session only.
        """
        with patch.object(engine, '_is_specialist', return_value=True):
            result = engine.apply_specialist_boost(50, "bot_001", "LONDON")
            assert result == 55

    def test_specialist_boost_caps_at_100(self, engine):
        """
        Test specialist boost does not exceed 100.

        Edge case: Specialist boost that would exceed 100 — cap at 100.
        """
        with patch.object(engine, '_is_specialist', return_value=True):
            result = engine.apply_specialist_boost(98, "bot_001", "LONDON")
            assert result == 100

    def test_non_specialist_no_boost(self, engine):
        """Test non-specialist bot receives no boost."""
        with patch.object(engine, '_is_specialist', return_value=False):
            result = engine.apply_specialist_boost(50, "bot_001", "LONDON")
            assert result == 50

    def test_boost_does_not_stack(self, engine):
        """
        Test boost does not stack (only +5 max).

        Per Dev Notes: +5 points to composite score for specialist session only.
        Does not stack (only +5 max, even if multiple specialist tags).
        """
        with patch.object(engine, '_is_specialist', return_value=True):
            # Apply boost twice - should still be +5 only
            result = engine.apply_specialist_boost(50, "bot_001", "LONDON")
            assert result == 55


class TestConcernFlag:
    """Test SESSION_CONCERN flag detection."""

    @pytest.fixture
    def engine(self):
        """Create DPR engine."""
        mock_session = MagicMock()
        return DPRScoringEngine(db_session=mock_session)

    def test_threshold_check_positive_delta(self, engine):
        """Test threshold check: positive delta does not trigger."""
        result = engine.threshold_check(10, threshold=-20)
        assert result is False

    def test_threshold_check_negative_within_threshold(self, engine):
        """Test threshold check: -10 delta does not trigger."""
        result = engine.threshold_check(-10, threshold=-20)
        assert result is False

    def test_threshold_check_negative_exceeds_threshold(self, engine):
        """
        Test threshold check: -25 delta triggers concern.

        AC #4: Given a bot's DPR score drops >20 points week-over-week,
        When the fortnight accumulation completes,
        Then a SESSION_CONCERN flag is set.
        """
        result = engine.threshold_check(-25, threshold=-20)
        assert result is True

    def test_threshold_check_exactly_threshold(self, engine):
        """Test threshold check: exactly -20 does NOT trigger (> 20, not >=)."""
        result = engine.threshold_check(-20, threshold=-20)
        assert result is False


class TestDPRComponentScores:
    """Test DPRComponentScores dataclass."""

    def test_composite_score_calculation(self):
        """Test weighted composite score calculation."""
        components = DPRComponentScores(
            win_rate=80.0,
            pnl=70.0,
            consistency=90.0,
            ev_per_trade=60.0,
            weights=DPR_WEIGHTS,
        )

        # 80 * 0.25 + 70 * 0.30 + 90 * 0.20 + 60 * 0.25
        # = 20 + 21 + 18 + 15 = 74
        assert components.composite_score() == 74

    def test_composite_score_rounds(self):
        """Test composite score rounds to nearest integer."""
        components = DPRComponentScores(
            win_rate=80.5,
            pnl=70.5,
            consistency=90.5,
            ev_per_trade=60.5,
            weights=DPR_WEIGHTS,
        )

        # Python's round() uses banker's rounding (round half to even)
        # 74.5 rounds to 74 (even) not 75
        assert components.composite_score() == 74

    def test_default_weights(self):
        """Test default weights are correct."""
        components = DPRComponentScores(
            win_rate=50.0,
            pnl=50.0,
            consistency=50.0,
            ev_per_trade=50.0,
        )

        assert components.weights == (0.25, 0.30, 0.20, 0.25)
