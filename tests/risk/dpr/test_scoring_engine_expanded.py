"""
Expanded Tests for DPR Scoring Engine.

Story 17.1: DPR Composite Score Calculation

Tests additional coverage:
- get_dpr_score method
- week_over_week_score_delta
- get_session_specialists
- specialist_session_check
- _get_trade_data edge cases
- Scoring window filtering
- Multiple bots scoring
"""

import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone, timedelta

from src.risk.dpr.scoring_engine import DPRScoringEngine, DPRScore
from src.events.dpr import DPRComponentScores, DPR_WEIGHTS


class TestGetDPRScore:
    """Test DPRScoringEngine.get_dpr_score method."""

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

    def test_get_dpr_score_returns_dpr_score_object(self, engine):
        """Test get_dpr_score returns DPRScore object."""
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

            result = engine.get_dpr_score("bot_001", "LONDON")

            assert result is not None
            assert isinstance(result, DPRScore)
            assert result.bot_id == "bot_001"
            assert result.session_id == "LONDON"
            assert 0 <= result.composite_score <= 100

    def test_get_dpr_score_includes_component_scores(self, engine):
        """Test get_dpr_score includes component scores."""
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

            result = engine.get_dpr_score("bot_001", "LONDON")

            assert isinstance(result.component_scores, DPRComponentScores)
            assert result.trade_count == 10
            assert result.session_win_rate == 0.7

    def test_get_dpr_score_not_eligible(self, engine):
        """Test get_dpr_score returns None when not eligible."""
        with patch.object(engine, '_get_trade_data') as mock_trade:
            mock_trade.return_value = {
                "total_trades": 0,
                "wins": 0,
                "net_pnl": 0.0,
                "daily_variance": 0.0,
                "ev_per_trade": 0.0,
            }

            result = engine.get_dpr_score("bot_001", "LONDON")

            assert result is None

    def test_get_dpr_score_specialist_boost_applied_flag(self, engine):
        """Test specialist_boost_applied flag is set correctly."""
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
            with patch.object(engine, '_is_specialist', return_value=True):
                result = engine.get_dpr_score("bot_001", "LONDON")

                assert result.specialist_boost_applied is True

    def test_get_dpr_score_specialist_boost_not_applied(self, engine):
        """Test specialist_boost_applied flag is False for non-specialist."""
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
            with patch.object(engine, '_is_specialist', return_value=False):
                result = engine.get_dpr_score("bot_001", "LONDON")

                assert result.specialist_boost_applied is False


class TestWeekOverWeekScoreDelta:
    """Test DPRScoringEngine.week_over_week_score_delta method."""

    @pytest.fixture
    def engine(self):
        """Create DPR engine with mock session."""
        mock_session = MagicMock()
        return DPRScoringEngine(db_session=mock_session)

    def test_delta_calculation_positive(self, engine):
        """Test positive score delta."""
        mock_history_record_current = MagicMock()
        mock_history_record_current.composite_score = 80

        mock_history_record_previous = MagicMock()
        mock_history_record_previous.composite_score = 60

        mock_history = MagicMock()
        mock_history.get_bot_scores.return_value = [
            mock_history_record_current,
            mock_history_record_previous,
        ]
        with patch('src.risk.dpr.history.DPRScoreHistory', return_value=mock_history):
            delta = engine.week_over_week_score_delta("bot_001")

            assert delta == 20  # 80 - 60

    def test_delta_calculation_negative(self, engine):
        """Test negative score delta (score dropped)."""
        mock_history_record_current = MagicMock()
        mock_history_record_current.composite_score = 55

        mock_history_record_previous = MagicMock()
        mock_history_record_previous.composite_score = 80

        mock_history = MagicMock()
        mock_history.get_bot_scores.return_value = [
            mock_history_record_current,
            mock_history_record_previous,
        ]
        with patch('src.risk.dpr.history.DPRScoreHistory', return_value=mock_history):
            delta = engine.week_over_week_score_delta("bot_001")

            assert delta == -25  # 55 - 80

    def test_delta_insufficient_history(self, engine):
        """Test delta returns 0 with insufficient history."""
        mock_history = MagicMock()
        mock_history.get_bot_scores.return_value = []  # No history
        with patch('src.risk.dpr.history.DPRScoreHistory', return_value=mock_history):
            delta = engine.week_over_week_score_delta("bot_001")

            assert delta == 0

    def test_delta_single_score(self, engine):
        """Test delta returns 0 with only one score."""
        mock_history_record = MagicMock()
        mock_history_record.composite_score = 75

        mock_history = MagicMock()
        mock_history.get_bot_scores.return_value = [mock_history_record]
        with patch('src.risk.dpr.history.DPRScoreHistory', return_value=mock_history):
            delta = engine.week_over_week_score_delta("bot_001")

            assert delta == 0


class TestGetSessionSpecialists:
    """Test DPRScoringEngine.get_session_specialists method."""

    @pytest.fixture
    def engine(self):
        """Create DPR engine with mock session."""
        mock_session = MagicMock()
        return DPRScoringEngine(db_session=mock_session)

    def test_get_specialists_returns_list(self, engine):
        """Test get_session_specialists returns list of bot IDs."""
        with patch('src.router.bot_manifest.BotRegistry') as mock_registry:
            mock_bot1 = MagicMock()
            mock_bot1.bot_id = "specialist_bot_1"
            mock_bot2 = MagicMock()
            mock_bot2.bot_id = "specialist_bot_2"

            mock_registry_instance = MagicMock()
            mock_registry_instance.list_by_tag.return_value = [mock_bot1, mock_bot2]
            mock_registry.return_value = mock_registry_instance

            result = engine.get_session_specialists("LONDON")

            assert isinstance(result, list)
            assert "specialist_bot_1" in result
            assert "specialist_bot_2" in result

    def test_get_specialists_empty_when_no_specialists(self, engine):
        """Test get_session_specialists returns empty list when none exist."""
        with patch('src.router.bot_manifest.BotRegistry') as mock_registry:
            mock_registry_instance = MagicMock()
            mock_registry_instance.list_by_tag.return_value = []
            mock_registry.return_value = mock_registry_instance

            result = engine.get_session_specialists("LONDON")

            assert result == []

    def test_get_specialists_exception_handling(self, engine):
        """Test get_session_specialists handles exceptions gracefully."""
        with patch('src.router.bot_manifest.BotRegistry', side_effect=Exception("DB error")):
            result = engine.get_session_specialists("LONDON")

            assert result == []


class TestSpecialistSessionCheck:
    """Test DPRScoringEngine.specialist_session_check method."""

    @pytest.fixture
    def engine(self):
        """Create DPR engine with mock session."""
        mock_session = MagicMock()
        return DPRScoringEngine(db_session=mock_session)

    def test_specialist_for_session(self, engine):
        """Test specialist_session_check returns True for specialist."""
        with patch.object(engine, 'get_session_specialists', return_value=["bot_001", "bot_002"]):
            result = engine.specialist_session_check("bot_001", "LONDON")

            assert result is True

    def test_not_specialist_for_session(self, engine):
        """Test specialist_session_check returns False for non-specialist."""
        with patch.object(engine, 'get_session_specialists', return_value=["bot_003"]):
            result = engine.specialist_session_check("bot_001", "LONDON")

            assert result is False


class TestIsSpecialist:
    """Test DPRScoringEngine._is_specialist method."""

    @pytest.fixture
    def engine(self):
        """Create DPR engine with mock session."""
        mock_session = MagicMock()
        return DPRScoringEngine(db_session=mock_session)

    def test_is_specialist_delegates(self, engine):
        """Test _is_specialist delegates to specialist_session_check."""
        with patch.object(engine, 'specialist_session_check', return_value=True) as mock_check:
            result = engine._is_specialist("bot_001", "LONDON")

            mock_check.assert_called_once_with("bot_001", "LONDON")
            assert result is True


class TestGetTradeData:
    """Test DPRScoringEngine._get_trade_data method."""

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

    def test_trade_data_empty_result(self, engine):
        """Test _get_trade_data returns empty metrics when no trades."""
        engine.db_session.query.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = []

        result = engine._get_trade_data("bot_001", "LONDON", "session")

        assert result["total_trades"] == 0
        assert result["wins"] == 0
        assert result["net_pnl"] == 0.0

    def test_trade_data_exception_handling(self, engine):
        """Test _get_trade_data handles exceptions gracefully."""
        engine.db_session.query.side_effect = Exception("DB error")

        result = engine._get_trade_data("bot_001", "LONDON", "session")

        assert result["total_trades"] == 0
        assert result["wins"] == 0


class TestNormalizationEdgeCases:
    """Test normalization functions edge cases."""

    @pytest.fixture
    def engine(self):
        """Create DPR engine."""
        mock_session = MagicMock()
        return DPRScoringEngine(
            db_session=mock_session,
            benchmark_pnl=0.0,  # Test with no benchmark
            baseline_ev=0.0,  # Test with no baseline
            max_acceptable_variance=0.01,
        )

    def test_pnl_normalize_zero_benchmark(self, engine):
        """Test PnL normalization with zero benchmark uses fixed reference."""
        engine.benchmark_pnl = 0
        result = engine._normalize_pnl(500.0)
        # 500 / 1000 * 50 = 25
        assert result == 25.0

    def test_pnl_normalize_very_large_pnl(self, engine):
        """Test PnL normalization caps at 100."""
        engine.benchmark_pnl = 1000.0
        result = engine._normalize_pnl(100000.0)  # 100x benchmark
        assert result == 100.0

    def test_ev_normalize_zero_baseline(self, engine):
        """Test EV normalization with zero baseline uses fixed reference."""
        engine.baseline_ev = 0
        result = engine._normalize_ev(5.0)
        # 5 / 10 * 50 = 25
        assert result == 25.0

    def test_ev_normalize_very_large_ev(self, engine):
        """Test EV normalization caps at 100."""
        engine.baseline_ev = 10.0
        result = engine._normalize_ev(100.0)  # 10x baseline
        assert result == 100.0

    def test_consistency_normalize_zero_variance(self, engine):
        """Test consistency normalization with zero variance."""
        result = engine._normalize_consistency(0.0)
        assert result == 100.0

    def test_consistency_normalize_exceeds_max(self, engine):
        """Test consistency normalization with variance exceeding max."""
        result = engine._normalize_consistency(0.02)  # > max_acceptable_variance of 0.01
        assert result == 0.0


class TestThresholdCheckEdgeCases:
    """Test threshold_check edge cases."""

    @pytest.fixture
    def engine(self):
        """Create DPR engine."""
        mock_session = MagicMock()
        return DPRScoringEngine(db_session=mock_session)

    def test_threshold_check_exactly_at_boundary(self, engine):
        """Test threshold_check at exact boundary (should not trigger)."""
        # -20 exactly should NOT trigger (> 20, not >=)
        result = engine.threshold_check(-20, threshold=-20)
        assert result is False

    def test_threshold_check_one_below_boundary(self, engine):
        """Test threshold_check one below boundary triggers."""
        result = engine.threshold_check(-21, threshold=-20)
        assert result is True

    def test_threshold_check_custom_threshold(self, engine):
        """Test threshold_check with custom threshold."""
        result = engine.threshold_check(-15, threshold=-15)
        assert result is False

        result = engine.threshold_check(-16, threshold=-15)
        assert result is True


class TestCalculateCompositeScoreEdgeCases:
    """Test calculate_composite_score edge cases."""

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

    def test_score_with_single_trade(self, engine):
        """Test composite score with single trade (minimum eligible)."""
        with patch.object(engine, '_get_trade_data') as mock_trade:
            mock_trade.return_value = {
                "total_trades": 1,
                "wins": 1,
                "net_pnl": 100.0,
                "daily_variance": 0.0,
                "ev_per_trade": 100.0,
                "max_drawdown": 0.0,
                "magic_number": 12345,
            }

            score = engine.calculate_composite_score("bot_001", "LONDON")

            assert score is not None
            assert 0 <= score <= 100

    def test_score_with_perfect_trading(self, engine):
        """Test composite score with perfect trading (all wins)."""
        with patch.object(engine, '_get_trade_data') as mock_trade:
            mock_trade.return_value = {
                "total_trades": 20,
                "wins": 20,
                "net_pnl": 5000.0,
                "daily_variance": 0.0,
                "ev_per_trade": 250.0,
                "max_drawdown": 0.0,
                "magic_number": 12345,
            }

            score = engine.calculate_composite_score("bot_001", "LONDON")

            assert score is not None
            assert score == 100  # Perfect score capped at 100

    def test_score_with_all_losing_trades(self, engine):
        """Test composite score with all losing trades."""
        with patch.object(engine, '_get_trade_data') as mock_trade:
            mock_trade.return_value = {
                "total_trades": 10,
                "wins": 0,
                "net_pnl": -500.0,
                "daily_variance": 0.005,
                "ev_per_trade": -50.0,
                "max_drawdown": 10.0,
                "magic_number": 12345,
            }

            score = engine.calculate_composite_score("bot_001", "LONDON")

            assert score is not None
            # WR=0 gives 0, PnL negative gives 0, consistency should be reasonable, EV negative gives 0
            assert 0 <= score <= 100


class TestMultipleBotsScoring:
    """Test scoring multiple bots."""

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

    def test_scores_different_bots_different_results(self, engine):
        """Test that different bots get different scores based on performance."""
        def mock_trade_data(bot_id, session_id, window):
            if bot_id == "bot_best":
                return {
                    "total_trades": 10,
                    "wins": 9,
                    "net_pnl": 1000.0,
                    "daily_variance": 0.001,
                    "ev_per_trade": 90.0,
                    "max_drawdown": 2.0,
                    "magic_number": 100,
                }
            else:  # bot_worst
                return {
                    "total_trades": 10,
                    "wins": 3,
                    "net_pnl": -200.0,
                    "daily_variance": 0.008,
                    "ev_per_trade": -20.0,
                    "max_drawdown": 15.0,
                    "magic_number": 200,
                }

        with patch.object(engine, '_get_trade_data', side_effect=mock_trade_data):
            best_score = engine.calculate_composite_score("bot_best", "LONDON")
            worst_score = engine.calculate_composite_score("bot_worst", "LONDON")

            assert best_score > worst_score
