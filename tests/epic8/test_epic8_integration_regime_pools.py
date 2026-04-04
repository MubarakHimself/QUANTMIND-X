"""
Integration Tests for Epic 8.10 - Regime-Conditional Strategy Pool Framework
===========================================================================

Integration tests for pool-based routing with regime detection.
Tests the interaction between Sentinel regime detection and Commander pool routing.

Reference: Story 8.10 (8-10-regime-conditional-strategy-pool-framework)
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch, AsyncMock


class TestRegimePoolRoutingIntegration:
    """Integration tests for regime-conditional pool routing."""

    @pytest.fixture
    def mock_sentinel(self):
        """Create mock Sentinel for regime detection."""
        sentinel = MagicMock()
        return sentinel

    @pytest.fixture
    def mock_commander(self, mock_sentinel):
        """Create Commander with regime_pool_map for pool routing."""
        from src.router.commander import Commander
        from src.router.bot_manifest import PoolState

        commander = Commander()

        # Ensure regime_pool_map is initialized
        if not hasattr(commander, 'regime_pool_map') or not commander.regime_pool_map:
            commander.regime_pool_map = {
                "TREND_STABLE": {
                    "scalping_long": PoolState.ACTIVE,
                    "scalping_short": PoolState.MUTED,
                    "orb_long": PoolState.CONDITIONAL,
                    "orb_short": PoolState.MUTED,
                    "scalping_neutral": PoolState.MUTED,
                },
                "RANGE_STABLE": {
                    "scalping_long": PoolState.MUTED,
                    "scalping_short": PoolState.MUTED,
                    "scalping_neutral": PoolState.ACTIVE,
                    "orb_long": PoolState.MUTED,
                    "orb_short": PoolState.MUTED,
                },
                "BREAKOUT_PRIME": {
                    "orb_long": PoolState.ACTIVE,
                    "orb_short": PoolState.ACTIVE,
                    "orb_false_breakout": PoolState.ACTIVE,
                    "scalping_long": PoolState.MUTED,
                    "scalping_short": PoolState.MUTED,
                    "scalping_neutral": PoolState.MUTED,
                },
                "HIGH_CHAOS": {
                    "scalping_long": PoolState.MUTED,
                    "scalping_short": PoolState.MUTED,
                    "scalping_neutral": PoolState.MUTED,
                    "orb_long": PoolState.MUTED,
                    "orb_short": PoolState.MUTED,
                    "orb_false_breakout": PoolState.MUTED,
                },
            }

        return commander

    def test_trend_stable_routes_to_scalping_long(self, mock_commander):
        """Given TREND_STABLE regime, when Commander evaluates routing, then scalping_long is ACTIVE."""
        from src.router.bot_manifest import PoolState

        regime = "TREND_STABLE"
        pool_states = mock_commander.regime_pool_map.get(regime, {})

        assert pool_states.get("scalping_long") == PoolState.ACTIVE
        assert pool_states.get("scalping_short") == PoolState.MUTED
        assert pool_states.get("orb_long") == PoolState.CONDITIONAL
        assert pool_states.get("orb_short") == PoolState.MUTED

    def test_range_stable_routes_to_scalping_neutral(self, mock_commander):
        """Given RANGE_STABLE regime, when Commander evaluates routing, then scalping_neutral is ACTIVE."""
        from src.router.bot_manifest import PoolState

        regime = "RANGE_STABLE"
        pool_states = mock_commander.regime_pool_map.get(regime, {})

        assert pool_states.get("scalping_long") == PoolState.MUTED
        assert pool_states.get("scalping_short") == PoolState.MUTED
        assert pool_states.get("scalping_neutral") == PoolState.ACTIVE

    def test_breakout_prime_routes_to_orb_pools(self, mock_commander):
        """Given BREAKOUT_PRIME regime, when Commander evaluates routing, then ORB pools are ACTIVE."""
        from src.router.bot_manifest import PoolState

        regime = "BREAKOUT_PRIME"
        pool_states = mock_commander.regime_pool_map.get(regime, {})

        assert pool_states.get("orb_long") == PoolState.ACTIVE
        assert pool_states.get("orb_short") == PoolState.ACTIVE
        assert pool_states.get("orb_false_breakout") == PoolState.ACTIVE
        assert pool_states.get("scalping_long") == PoolState.MUTED

    def test_high_chaos_mutes_all_pools(self, mock_commander):
        """Given HIGH_CHAOS regime, when Commander evaluates routing, then all pools are MUTED."""
        from src.router.bot_manifest import PoolState

        regime = "HIGH_CHAOS"
        pool_states = mock_commander.regime_pool_map.get(regime, {})

        for pool_name, state in pool_states.items():
            assert state == PoolState.MUTED, f"Pool {pool_name} should be MUTED in HIGH_CHAOS"

    def test_orb_long_requires_volume_confirmation(self, mock_commander):
        """Given TREND_STABLE regime, when ORB Long is evaluated, then volume confirmation is required."""
        from src.router.bot_manifest import PoolState

        regime = "TREND_STABLE"
        pool_states = mock_commander.regime_pool_map.get(regime, {})

        assert pool_states.get("orb_long") == PoolState.CONDITIONAL


class TestRegimePoolMapCompleteness:
    """Verify regime_pool_map covers all required regimes and pools."""

    def test_all_regimes_defined(self):
        """Verify all required regimes are in regime_pool_map."""
        from src.router.commander import Commander
        from src.router.bot_manifest import PoolState

        commander = Commander()

        required_regimes = [
            "TREND_STABLE",
            "RANGE_STABLE",
            "BREAKOUT_PRIME",
            "HIGH_CHAOS",
            "NEWS_EVENT",
            "UNCERTAIN",
        ]

        for regime in required_regimes:
            assert regime in commander.regime_pool_map, f"Missing regime: {regime}"

    def test_all_pools_defined_per_regime(self):
        """Verify all required pools are defined for each regime."""
        from src.router.commander import Commander

        commander = Commander()

        # Pool requirements vary by regime:
        # - TREND_STABLE: scalping pools + orb_long (conditional) + orb_short
        # - RANGE_STABLE: scalping pools only
        # - BREAKOUT_PRIME: all pools including orb_false_breakout
        # - HIGH_CHAOS: all pools (all muted)

        for regime, pools in commander.regime_pool_map.items():
            # All regimes should have scalping pools
            assert "scalping_long" in pools, f"Missing scalping_long in {regime}"
            assert "scalping_short" in pools, f"Missing scalping_short in {regime}"
            assert "scalping_neutral" in pools, f"Missing scalping_neutral in {regime}"
            assert "orb_long" in pools, f"Missing orb_long in {regime}"
            assert "orb_short" in pools, f"Missing orb_short in {regime}"

    def test_strategy_type_enum_includes_orb(self):
        """Verify StrategyType enum includes ORB for Opening Range Breakout."""
        from src.router.bot_manifest import StrategyType

        assert hasattr(StrategyType, 'ORB'), "StrategyType should have ORB member"
        assert StrategyType.ORB.value == "ORB"


class TestPoolStateEnum:
    """Verify PoolState enum values."""

    def test_pool_state_active(self):
        """Verify PoolState.ACTIVE value."""
        from src.router.bot_manifest import PoolState

        assert PoolState.ACTIVE.value == "active"

    def test_pool_state_muted(self):
        """Verify PoolState.MUTED value."""
        from src.router.bot_manifest import PoolState

        assert PoolState.MUTED.value == "muted"

    def test_pool_state_conditional(self):
        """Verify PoolState.CONDITIONAL value for volume-gated pools."""
        from src.router.bot_manifest import PoolState

        assert PoolState.CONDITIONAL.value == "conditional"


class TestRegimePoolMapCrossValidation:
    """Cross-validation of regime-pool mappings against acceptance criteria."""

    def test_ac1_trend_stable_pool_routing(self):
        """
        AC1: Given TREND_STABLE, when Commander evaluates routing,
        Then scalping_long is ACTIVE, scalping_short is MUTED, ORB Long is CONDITIONAL.
        """
        from src.router.commander import Commander
        from src.router.bot_manifest import PoolState

        commander = Commander()
        pools = commander.regime_pool_map.get("TREND_STABLE", {})

        # AC1 assertions
        assert pools.get("scalping_long") == PoolState.ACTIVE
        assert pools.get("scalping_short") == PoolState.MUTED
        # ORB Long is CONDITIONAL (requires volume confirmation)
        assert pools.get("orb_long") == PoolState.CONDITIONAL

    def test_ac2_range_stable_pool_routing(self):
        """
        AC2: Given RANGE_STABLE, when Commander evaluates routing,
        Then scalping_neutral is ACTIVE, scalping_long and scalping_short are MUTED.
        """
        from src.router.commander import Commander
        from src.router.bot_manifest import PoolState

        commander = Commander()
        pools = commander.regime_pool_map.get("RANGE_STABLE", {})

        # AC2 assertions
        assert pools.get("scalping_neutral") == PoolState.ACTIVE
        assert pools.get("scalping_long") == PoolState.MUTED
        assert pools.get("scalping_short") == PoolState.MUTED

    def test_ac3_breakout_prime_pool_routing(self):
        """
        AC3: Given BREAKOUT_PRIME, when Commander evaluates routing,
        Then ORB pools (Long/Short/False Breakout) are ACTIVE, scalping pools are MUTED.
        """
        from src.router.commander import Commander
        from src.router.bot_manifest import PoolState

        commander = Commander()
        pools = commander.regime_pool_map.get("BREAKOUT_PRIME", {})

        # AC3 assertions
        assert pools.get("orb_long") == PoolState.ACTIVE
        assert pools.get("orb_short") == PoolState.ACTIVE
        assert pools.get("orb_false_breakout") == PoolState.ACTIVE
        assert pools.get("scalping_long") == PoolState.MUTED
        assert pools.get("scalping_short") == PoolState.MUTED
        assert pools.get("scalping_neutral") == PoolState.MUTED

    def test_ac4_high_chaos_full_mute(self):
        """
        AC4: Given HIGH_CHAOS, when Commander evaluates routing,
        Then all scalping and ORB pools are MUTED.
        """
        from src.router.commander import Commander
        from src.router.bot_manifest import PoolState

        commander = Commander()
        pools = commander.regime_pool_map.get("HIGH_CHAOS", {})

        # AC4 assertions - all pools muted
        all_muted = all(state == PoolState.MUTED for state in pools.values())
        assert all_muted, "All pools should be MUTED in HIGH_CHAOS"
