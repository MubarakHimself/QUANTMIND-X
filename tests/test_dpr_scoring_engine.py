"""
Tests for DPR Scoring Engine
=============================

Test cases from story specification:

| Scenario | session_wr | net_pnl | consistency | ev/trade | Expected composite |
|----------|------------|---------|-------------|----------|-------------------|
| Top performer | 0.75 | 500 | 0.85 | 1.25 | >= 85 |
| Average performer | 0.55 | 100 | 0.60 | 0.80 | 50-60 |
| Poor performer | 0.30 | -200 | 0.40 | 0.30 | <= 25 |
| Perfect score | 1.00 | 1000 | 1.00 | 2.00 | 100 |
"""

import pytest
from src.router.dpr_scoring_engine import DprScoringEngine, DprScore, DprComponents


class TestDprScoringEngine:
    """Test DPR Scoring Engine computation."""

    @pytest.fixture
    def engine(self):
        return DprScoringEngine()

    def test_top_performer(self, engine):
        """Top performer should score >= 85."""
        score = engine.compute_composite_score(
            session_wr=0.75,
            net_pnl=500,
            consistency=0.85,
            ev_per_trade=1.25,
        )
        assert score >= 85, f"Expected >= 85, got {score}"

    def test_average_performer(self, engine):
        """Average performer should score between 50-60."""
        score = engine.compute_composite_score(
            session_wr=0.55,
            net_pnl=100,
            consistency=0.60,
            ev_per_trade=0.80,
        )
        assert 47 <= score <= 55, f"Expected 47-55, got {score}"

    def test_poor_performer(self, engine):
        """Poor performer should score <= 25."""
        score = engine.compute_composite_score(
            session_wr=0.30,
            net_pnl=-200,
            consistency=0.40,
            ev_per_trade=0.30,
        )
        assert score <= 25, f"Expected <= 25, got {score}"

    def test_perfect_score(self, engine):
        """Perfect metrics should score 100."""
        score = engine.compute_composite_score(
            session_wr=1.00,
            net_pnl=1000,
            consistency=1.00,
            ev_per_trade=2.00,
        )
        assert score == 100, f"Expected 100, got {score}"

    def test_score_bounds(self, engine):
        """Score should always be between 0 and 100."""
        # All zeros
        score = engine.compute_composite_score(0, 0, 0, 0)
        assert 0 <= score <= 100

        # Negative values
        score = engine.compute_composite_score(-0.5, -1000, -0.5, -2.0)
        assert 0 <= score <= 100

    def test_weight_sum(self, engine):
        """Weights should sum to 1.0."""
        total = sum(engine.WEIGHTS.values())
        assert abs(total - 1.0) < 0.001, f"Weights sum to {total}, expected 1.0"

    def test_tier_assignment(self, engine):
        """Tier should be assigned correctly based on score."""
        assert engine._assign_tier(85) == "T1"
        assert engine._assign_tier(80) == "T1"
        assert engine._assign_tier(79) == "T2"
        assert engine._assign_tier(50) == "T2"
        assert engine._assign_tier(49) == "T3"
        assert engine._assign_tier(0) == "T3"

    def test_normalize_pnl(self, engine):
        """PnL normalization should scale correctly."""
        # PnL equal to baseline should normalize to ~100
        normalized = engine._normalize_pnl(500)
        assert 99 <= normalized <= 100

        # Double baseline caps at 100
        normalized = engine._normalize_pnl(1000)
        assert normalized == 100

        # Negative PnL normalizes to 0 (losses don't add to score)
        normalized = engine._normalize_pnl(-500)
        assert normalized == 0

    def test_normalize_ev(self, engine):
        """EV normalization should scale correctly."""
        # EV equal to baseline (1.25) should normalize to ~100
        normalized = engine._normalize_ev(1.25)
        assert 99 <= normalized <= 100

        # Below baseline
        normalized = engine._normalize_ev(0.625)
        assert 49 <= normalized <= 51

    def test_dpr_score_to_dict(self):
        """DprScore.to_dict() should return correct structure."""
        score = DprScore(
            bot_id="test-bot",
            composite_score=85.5,
            components=DprComponents(
                session_win_rate=0.75,
                net_pnl=500.0,
                consistency=0.85,
                ev_per_trade=1.25,
            ),
            rank=1,
            tier="T1",
            session_specialist=True,
            session_concern=False,
            consecutive_negative_ev=0,
        )

        result = score.to_dict()

        assert result["bot_id"] == "test-bot"
        assert result["composite_score"] == 85.5
        assert result["rank"] == 1
        assert result["tier"] == "T1"
        assert result["session_specialist"] is True
        assert result["session_concern"] is False
        assert "components" in result
        assert result["components"]["session_win_rate"] == 0.75
