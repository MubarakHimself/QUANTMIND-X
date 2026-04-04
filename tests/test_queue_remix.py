"""
Tests for Queue Tier Remix
==========================

Test cases from story specification:

| Scenario | T1 count | T2 count | T3 count | Expected interleaving |
|----------|----------|----------|----------|----------------------|
| Equal tiers | 5 | 5 | 5 | T1[0], T3[0], T2[0], T1[1], T3[1], T2[1]... |
| Empty T3 | 5 | 5 | 0 | T1[0], T2[0], T1[1], T2[1]... |
"""

import pytest
from src.router.queue_remix import QueueRemix
from src.router.dpr_scoring_engine import DprScore, DprComponents


class TestQueueRemix:
    """Test Queue Tier Remix computation."""

    @pytest.fixture
    def remix(self):
        return QueueRemix()

    def _create_score(self, bot_id: str, composite: float, tier: str) -> DprScore:
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
            tier=tier,
        )

    def test_equal_tiers_interleaving(self, remix):
        """Equal tiers should interleave as T1[0], T3[0], T2[0], T1[1], T3[1], T2[1]..."""
        scores = []
        # Create 5 bots in each tier
        for i in range(5):
            scores.append(self._create_score(f"T1-{i}", 90 - i, "T1"))
            scores.append(self._create_score(f"T2-{i}", 70 - i, "T2"))
            scores.append(self._create_score(f"T3-{i}", 50 - i, "T3"))

        result = remix.compute_remix(scores)

        # Check tier lists
        assert len(result["T1"]) == 5
        assert len(result["T2"]) == 5
        assert len(result["T3"]) == 5

        # T1 should be sorted descending by score
        t1_scores = [s.composite_score for s in scores if s.tier == "T1"]
        assert t1_scores == sorted(t1_scores, reverse=True)

        # Check interleaved order
        interleaved = result["interleaved_order"]
        assert interleaved[0] == "T1-0"  # T1[0] first
        assert interleaved[1] == "T3-0"  # T3[0] second
        assert interleaved[2] == "T2-0"  # T2[0] third
        assert interleaved[3] == "T1-1"  # T1[1] fourth

    def test_empty_t3_interleaving(self, remix):
        """Empty T3 should skip T3 positions in interleaving."""
        scores = []
        # Create 5 bots in T1 and T2, none in T3
        for i in range(5):
            scores.append(self._create_score(f"T1-{i}", 90 - i, "T1"))
            scores.append(self._create_score(f"T2-{i}", 70 - i, "T2"))

        result = remix.compute_remix(scores)

        assert len(result["T1"]) == 5
        assert len(result["T2"]) == 5
        assert len(result["T3"]) == 0

        # Interleaved should be T1[0], T2[0], T1[1], T2[1], ...
        interleaved = result["interleaved_order"]
        assert interleaved[0] == "T1-0"
        assert interleaved[1] == "T2-0"
        assert interleaved[2] == "T1-1"
        assert interleaved[3] == "T2-1"

    def test_empty_t1_t2(self, remix):
        """Only T3 bots should still work."""
        scores = [self._create_score(f"T3-{i}", 50 - i * 5, "T3") for i in range(3)]

        result = remix.compute_remix(scores)

        assert len(result["T1"]) == 0
        assert len(result["T2"]) == 0
        assert len(result["T3"]) == 3
        assert result["interleaved_order"] == ["T3-0", "T3-1", "T3-2"]

    def test_single_bot_per_tier(self, remix):
        """Single bot per tier should interleave correctly."""
        scores = [
            self._create_score("T1-0", 85, "T1"),
            self._create_score("T2-0", 55, "T2"),
            self._create_score("T3-0", 35, "T3"),
        ]

        result = remix.compute_remix(scores)

        assert result["interleaved_order"] == ["T1-0", "T3-0", "T2-0"]

    def test_tier_counts_in_result(self, remix):
        """Result should include tier counts."""
        scores = [
            self._create_score("T1-0", 85, "T1"),
            self._create_score("T1-1", 80, "T1"),
            self._create_score("T2-0", 55, "T2"),
            self._create_score("T3-0", 35, "T3"),
            self._create_score("T3-1", 30, "T3"),
        ]

        result = remix.compute_remix(scores)

        assert result["tier_counts"]["T1"] == 2
        assert result["tier_counts"]["T2"] == 1
        assert result["tier_counts"]["T3"] == 2
