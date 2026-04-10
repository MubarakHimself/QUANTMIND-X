"""Tests for QuantMindLib V1 — DPR tier to BotTier mapping."""

import pytest

from src.library.core.bridges.dpr_redis_bridge import dpr_tier_to_bot_tier
from src.library.core.types.enums import BotTier


class TestTierMapping:
    def test_elite_to_elite(self):
        """DPR ELITE tier maps to BotTier.ELITE."""
        result = dpr_tier_to_bot_tier("ELITE")
        assert result == BotTier.ELITE

    def test_performing_to_standard(self):
        """DPR PERFORMING tier maps to BotTier.STANDARD (no PERFORMING in BotTier)."""
        result = dpr_tier_to_bot_tier("PERFORMING")
        assert result == BotTier.STANDARD

    def test_standard_to_standard(self):
        """DPR STANDARD tier maps to BotTier.STANDARD."""
        result = dpr_tier_to_bot_tier("STANDARD")
        assert result == BotTier.STANDARD

    def test_at_risk_to_at_risk(self):
        """DPR AT_RISK tier maps to BotTier.AT_RISK."""
        result = dpr_tier_to_bot_tier("AT_RISK")
        assert result == BotTier.AT_RISK

    def test_circuit_broken_to_circuit_broken(self):
        """DPR CIRCUIT_BROKEN tier maps to BotTier.CIRCUIT_BROKEN."""
        result = dpr_tier_to_bot_tier("CIRCUIT_BROKEN")
        assert result == BotTier.CIRCUIT_BROKEN

    def test_unknown_tier_defaults_to_standard(self):
        """Unknown DPR tier maps to BotTier.STANDARD as fallback."""
        result = dpr_tier_to_bot_tier("UNKNOWN_TIER")
        assert result == BotTier.STANDARD

    def test_empty_string_defaults_to_standard(self):
        """Empty DPR tier string maps to BotTier.STANDARD as fallback."""
        result = dpr_tier_to_bot_tier("")
        assert result == BotTier.STANDARD
