"""
Tests for DPR Tie-Break Cascade.

Story 17.1: DPR Composite Score Calculation

Tests:
- 4-level tie-break cascade at all levels
- GG-2 resolution (Magic Number as final arbiter)
- Perfect tie on all cascade levels
"""

import pytest
from unittest.mock import MagicMock

from src.risk.dpr.scoring_engine import DPRScoringEngine, DPRScore
from src.events.dpr import DPRComponentScores


class TestTieBreakCascade:
    """Test 4-level tie-break cascade."""

    @pytest.fixture
    def engine(self):
        """Create DPR engine."""
        mock_session = MagicMock()
        return DPRScoringEngine(db_session=mock_session)

    def _make_dpr_score(
        self,
        bot_id: str,
        composite: int,
        win_rate: float,
        max_drawdown: float,
        trade_count: int,
        magic_number: int,
    ) -> DPRScore:
        """Helper to create DPRScore."""
        return DPRScore(
            bot_id=bot_id,
            session_id="LONDON",
            composite_score=composite,
            component_scores=DPRComponentScores(
                win_rate=win_rate,
                pnl=50.0,
                consistency=50.0,
                ev_per_trade=50.0,
            ),
            trade_count=trade_count,
            session_win_rate=win_rate,
            max_drawdown=max_drawdown,
            magic_number=magic_number,
        )

    def test_level_1_higher_win_rate_wins(self, engine):
        """
        Level 1: Higher session-specific win rate wins.

        AC #2: Given two bots tie on the composite DPR score,
        When the tie-break cascade evaluates,
        Then it applies in order:
        (1) higher session-specific win rate wins.
        """
        bot_a = self._make_dpr_score(
            bot_id="bot_A",
            composite=75,
            win_rate=0.70,
            max_drawdown=5.0,
            trade_count=10,
            magic_number=100,
        )
        bot_b = self._make_dpr_score(
            bot_id="bot_B",
            composite=75,
            win_rate=0.65,
            max_drawdown=5.0,
            trade_count=10,
            magic_number=100,
        )

        winner = engine.tie_break_cascade(bot_a, bot_b)
        assert winner == "bot_A"

    def test_level_2_lower_drawdown_wins(self, engine):
        """
        Level 2: Lower max drawdown wins.

        (2) lower max drawdown wins.
        """
        bot_a = self._make_dpr_score(
            bot_id="bot_A",
            composite=75,
            win_rate=0.65,
            max_drawdown=4.0,
            trade_count=10,
            magic_number=100,
        )
        bot_b = self._make_dpr_score(
            bot_id="bot_B",
            composite=75,
            win_rate=0.65,
            max_drawdown=6.0,
            trade_count=10,
            magic_number=100,
        )

        winner = engine.tie_break_cascade(bot_a, bot_b)
        assert winner == "bot_A"

    def test_level_3_higher_trade_count_wins(self, engine):
        """
        Level 3: Higher trade count wins.

        (3) higher trade count wins.
        """
        bot_a = self._make_dpr_score(
            bot_id="bot_A",
            composite=75,
            win_rate=0.65,
            max_drawdown=5.0,
            trade_count=15,
            magic_number=100,
        )
        bot_b = self._make_dpr_score(
            bot_id="bot_B",
            composite=75,
            win_rate=0.65,
            max_drawdown=5.0,
            trade_count=10,
            magic_number=100,
        )

        winner = engine.tie_break_cascade(bot_a, bot_b)
        assert winner == "bot_A"

    def test_level_4_lower_magic_number_wins(self, engine):
        """
        Level 4: Lower Magic Number wins (GG-2 resolution).

        (4) lower Magic Number wins (GG-2 resolution).
        """
        bot_a = self._make_dpr_score(
            bot_id="bot_A",
            composite=75,
            win_rate=0.65,
            max_drawdown=5.0,
            trade_count=10,
            magic_number=100,
        )
        bot_b = self._make_dpr_score(
            bot_id="bot_B",
            composite=75,
            win_rate=0.65,
            max_drawdown=5.0,
            trade_count=10,
            magic_number=200,
        )

        winner = engine.tie_break_cascade(bot_a, bot_b)
        assert winner == "bot_A"

    def test_bot_b_wins_level_1(self, engine):
        """Test bot B wins at level 1 with higher win rate."""
        bot_a = self._make_dpr_score(
            bot_id="bot_A",
            composite=75,
            win_rate=0.60,
            max_drawdown=5.0,
            trade_count=10,
            magic_number=100,
        )
        bot_b = self._make_dpr_score(
            bot_id="bot_B",
            composite=75,
            win_rate=0.70,
            max_drawdown=5.0,
            trade_count=10,
            magic_number=100,
        )

        winner = engine.tie_break_cascade(bot_a, bot_b)
        assert winner == "bot_B"

    def test_perfect_tie_all_levels_bot_a_wins(self, engine):
        """
        Test perfect tie on all 4 cascade levels.

        Edge case: Perfect tie on all 4 cascade levels — use Magic Number as final arbiter.
        """
        bot_a = self._make_dpr_score(
            bot_id="bot_A",
            composite=75,
            win_rate=0.65,
            max_drawdown=5.0,
            trade_count=10,
            magic_number=100,
        )
        bot_b = self._make_dpr_score(
            bot_id="bot_B",
            composite=75,
            win_rate=0.65,
            max_drawdown=5.0,
            trade_count=10,
            magic_number=200,
        )

        winner = engine.tie_break_cascade(bot_a, bot_b)
        # Lower Magic Number (100) wins
        assert winner == "bot_A"

    def test_perfect_tie_all_levels_bot_b_wins(self, engine):
        """Test perfect tie where bot B wins (lower Magic Number)."""
        bot_a = self._make_dpr_score(
            bot_id="bot_A",
            composite=75,
            win_rate=0.65,
            max_drawdown=5.0,
            trade_count=10,
            magic_number=300,
        )
        bot_b = self._make_dpr_score(
            bot_id="bot_B",
            composite=75,
            win_rate=0.65,
            max_drawdown=5.0,
            trade_count=10,
            magic_number=100,
        )

        winner = engine.tie_break_cascade(bot_a, bot_b)
        # Lower Magic Number (100) wins
        assert winner == "bot_B"
