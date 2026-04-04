"""
Tests for Queue Reranker + SESSION_CONCERN Flags
=================================================

Test cases from story specification:

| Scenario | consecutive_negative_ev | session_concern |
|----------|----------------------|-----------------|
| 2 consecutive | 2 | False |
| 3 consecutive | 3 | True |
| 5 consecutive | 5 | True |
"""

import pytest
import asyncio
from src.router.queue_reranker import QueueReRanker, QueueReRankResult
from src.router.dpr_scoring_engine import DprScore, DprComponents


class TestQueueReRanker:
    """Test Queue Re-rank step and SESSION_CONCERN flags."""

    @pytest.fixture
    def reranker(self):
        return QueueReRanker()

    def _create_score(self, bot_id: str, consecutive_negative_ev: int, composite: float = 70.0) -> DprScore:
        """Helper to create a DprScore."""
        return DprScore(
            bot_id=bot_id,
            composite_score=composite,
            components=DprComponents(
                session_win_rate=0.5,
                net_pnl=0,
                consistency=0.5,
                ev_per_trade=0.5,
            ),
            consecutive_negative_ev=consecutive_negative_ev,
            session_concern=False,
        )

    @pytest.mark.asyncio
    async def test_2_consecutive_not_flagged(self, reranker):
        """2 consecutive negative EV sessions should NOT set session_concern."""
        scores = [self._create_score("bot-1", consecutive_negative_ev=2)]

        result = await reranker.run(scores)

        # Bot should not be in concerns list
        assert "bot-1" not in result.concerns
        # The score's session_concern should still be False
        assert scores[0].session_concern is False

    @pytest.mark.asyncio
    async def test_3_consecutive_flagged(self, reranker):
        """3 consecutive negative EV sessions should set session_concern."""
        scores = [self._create_score("bot-1", consecutive_negative_ev=3)]

        result = await reranker.run(scores)

        # Bot should be in concerns list
        assert "bot-1" in result.concerns
        assert scores[0].session_concern is True

    @pytest.mark.asyncio
    async def test_5_consecutive_flagged(self, reranker):
        """5 consecutive negative EV sessions should set session_concern."""
        scores = [self._create_score("bot-1", consecutive_negative_ev=5)]

        result = await reranker.run(scores)

        assert "bot-1" in result.concerns
        assert scores[0].session_concern is True

    @pytest.mark.asyncio
    async def test_mixed_consecutive(self, reranker):
        """Mixed consecutive counts should flag correctly."""
        scores = [
            self._create_score("bot-safe", consecutive_negative_ev=0),
            self._create_score("bot-warn1", consecutive_negative_ev=2),
            self._create_score("bot-flag1", consecutive_negative_ev=3),
            self._create_score("bot-flag2", consecutive_negative_ev=5),
        ]

        result = await reranker.run(scores)

        assert "bot-safe" not in result.concerns
        assert "bot-warn1" not in result.concerns
        assert "bot-flag1" in result.concerns
        assert "bot-flag2" in result.concerns
        assert len(result.concerns) == 2

    @pytest.mark.asyncio
    async def test_queue_sorted_by_composite_score(self, reranker):
        """Queue should be sorted by composite score descending."""
        scores = [
            self._create_score("bot-low", composite=30.0, consecutive_negative_ev=0),
            self._create_score("bot-high", composite=90.0, consecutive_negative_ev=0),
            self._create_score("bot-mid", composite=60.0, consecutive_negative_ev=0),
        ]

        result = await reranker.run(scores)

        # Queue should be sorted by score descending
        assert result.queue == ["bot-high", "bot-mid", "bot-low"]

    def test_threshold_constant(self, reranker):
        """CONSECUTIVE_NEGATIVE_EV_THRESHOLD should be 3."""
        assert reranker.CONSECUTIVE_NEGATIVE_EV_THRESHOLD == 3

    @pytest.mark.asyncio
    async def test_result_to_dict(self, reranker):
        """QueueReRankResult.to_dict() should return correct structure."""
        scores = [
            self._create_score("bot-1", consecutive_negative_ev=3),
            self._create_score("bot-2", consecutive_negative_ev=0),
        ]

        result = await reranker.run(scores)
        d = result.to_dict()

        assert "queue" in d
        assert "concerns" in d
        assert d["concerns_count"] == 1
