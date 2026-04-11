"""
QuantMindLib V1 — DPR Dual Engine Router Tests

Phase 10 Packet 10C: DPRDualEngineRouter tests.
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from src.library.core.bridges.dpr_dual_engine import (
    DPRDualEngineRouter,
)
from src.library.core.bridges.sentinel_dpr_bridges import DPRScore, DPRBridge
from src.library.core.bridges.dpr_redis_bridge import DPRRedisPublisher
from src.library.core.bridges.registry_journal_bridges import RegistryBridge
from src.library.core.domain.registry_record import RegistryRecord
from src.library.core.types.enums import BotTier, RegistryStatus


def _make_dpr_score(
    bot_id: str = "bot-001",
    score: float = 0.85,
    tier: str = "ELITE",
    rank: int = 1,
) -> DPRScore:
    return DPRScore(
        bot_id=bot_id,
        dpr_score=score,
        sharpe_today=score,
        win_rate_today=score,
        daily_pnl=score,
        rank=rank,
        tier=tier,
        computed_at_ms=int(time.time() * 1000),
    )


def _make_mock_redis_publisher() -> MagicMock:
    """Create a fully-mocked DPRRedisPublisher."""
    pub = MagicMock(spec=DPRRedisPublisher)
    pub.publish_single = MagicMock(return_value=True)
    return pub


# ---------------------------------------------------------------------------
# TestDualEngineInit
# ---------------------------------------------------------------------------

class TestDualEngineInit:
    def test_init_default_components(self):
        """Initializes with DPRBridge, DPRRedisPublisher, and RegistryBridge."""
        router = DPRDualEngineRouter()
        assert router._dpr_bridge is not None
        assert isinstance(router._dpr_bridge, DPRBridge)
        assert router._redis_publisher is not None
        assert isinstance(router._redis_publisher, DPRRedisPublisher)
        assert router._registry_bridge is not None
        assert isinstance(router._registry_bridge, RegistryBridge)
        assert router._router_cache == {}
        assert router._risk_modifiers == {}

    def test_init_custom_components(self):
        """Accepts custom DPRBridge, DPRRedisPublisher, and RegistryBridge."""
        dpr = DPRBridge()
        pub = _make_mock_redis_publisher()
        registry = RegistryBridge()
        router = DPRDualEngineRouter(
            dpr_bridge=dpr,
            redis_publisher=pub,
            registry_bridge=registry,
        )
        assert router._dpr_bridge is dpr
        assert router._redis_publisher is pub
        assert router._registry_bridge is registry


# ---------------------------------------------------------------------------
# TestRouterLayerUpdate
# ---------------------------------------------------------------------------

class TestRouterLayerUpdate:
    def test_standard_tier_eligible(self):
        """STANDARD tier with rank <= 10 and ACTIVE status is routing-eligible."""
        dpr = DPRBridge()
        registry = RegistryBridge()
        registry.register(
            bot_id="bot-std",
            bot_spec_id="spec-1",
            owner="test",
            tier=BotTier.STANDARD,
        )
        dpr.scores = {
            "bot-std": _make_dpr_score(bot_id="bot-std", score=0.60, tier="STANDARD", rank=5),
        }

        router = DPRDualEngineRouter(dpr_bridge=dpr, registry_bridge=registry)
        result = router.update_router_layer()

        assert result["bot-std"] is True
        assert router._router_cache["bot-std"] is True

    def test_circuit_broken_ineligible(self):
        """CIRCUIT_BROKEN tier is not routing-eligible."""
        dpr = DPRBridge()
        registry = RegistryBridge()
        registry.register(
            bot_id="bot-broken",
            bot_spec_id="spec-1",
            owner="test",
            tier=BotTier.CIRCUIT_BROKEN,
        )
        dpr.scores = {
            "bot-broken": _make_dpr_score(bot_id="bot-broken", score=0.10, tier="CIRCUIT_BROKEN", rank=2),
        }

        router = DPRDualEngineRouter(dpr_bridge=dpr, registry_bridge=registry)
        result = router.update_router_layer()

        assert result["bot-broken"] is False
        assert router._router_cache["bot-broken"] is False

    def test_at_risk_ineligible(self):
        """AT_RISK tier is not routing-eligible."""
        dpr = DPRBridge()
        registry = RegistryBridge()
        registry.register(
            bot_id="bot-risk",
            bot_spec_id="spec-1",
            owner="test",
            tier=BotTier.AT_RISK,
        )
        dpr.scores = {
            "bot-risk": _make_dpr_score(bot_id="bot-risk", score=0.35, tier="AT_RISK", rank=3),
        }

        router = DPRDualEngineRouter(dpr_bridge=dpr, registry_bridge=registry)
        result = router.update_router_layer()

        assert result["bot-risk"] is False

    def test_rank_over_10_ineligible(self):
        """Rank > 10 makes bot ineligible even with good tier."""
        dpr = DPRBridge()
        registry = RegistryBridge()
        registry.register(
            bot_id="bot-lowrank",
            bot_spec_id="spec-1",
            owner="test",
            tier=BotTier.STANDARD,
        )
        dpr.scores = {
            "bot-lowrank": _make_dpr_score(bot_id="bot-lowrank", score=0.65, tier="STANDARD", rank=15),
        }

        router = DPRDualEngineRouter(dpr_bridge=dpr, registry_bridge=registry)
        result = router.update_router_layer()

        assert result["bot-lowrank"] is False


# ---------------------------------------------------------------------------
# TestRoutingDecision
# ---------------------------------------------------------------------------

class TestRoutingDecision:
    def test_eligible_returns_true(self):
        """get_routing_decision returns True for eligible bot."""
        dpr = DPRBridge()
        registry = RegistryBridge()
        registry.register(
            bot_id="bot-ok",
            bot_spec_id="spec-1",
            owner="test",
            tier=BotTier.ELITE,
        )
        dpr.scores = {
            "bot-ok": _make_dpr_score(bot_id="bot-ok", score=0.90, tier="ELITE", rank=1),
        }

        router = DPRDualEngineRouter(dpr_bridge=dpr, registry_bridge=registry)
        router.update_router_layer()
        result = router.get_routing_decision("bot-ok")
        assert result is True

    def test_ineligible_returns_false(self):
        """get_routing_decision returns False for ineligible bot."""
        dpr = DPRBridge()
        registry = RegistryBridge()
        dpr.scores = {
            "bot-bad": _make_dpr_score(bot_id="bot-bad", score=0.10, tier="CIRCUIT_BROKEN", rank=1),
        }

        router = DPRDualEngineRouter(dpr_bridge=dpr, registry_bridge=registry)
        router.update_router_layer()
        result = router.get_routing_decision("bot-bad")
        assert result is False

    def test_unknown_returns_false(self):
        """get_routing_decision returns False for unknown bot not in DPR."""
        dpr = DPRBridge()
        registry = RegistryBridge()
        router = DPRDualEngineRouter(dpr_bridge=dpr, registry_bridge=registry)
        result = router.get_routing_decision("unknown-bot")
        assert result is False


# ---------------------------------------------------------------------------
# TestRiskModifier
# ---------------------------------------------------------------------------

class TestRiskModifier:
    def test_elite_score_full_kelly(self):
        """Score >= 0.85 returns modifier 1.0."""
        dpr = DPRBridge()
        dpr.scores = {
            "bot-e1": _make_dpr_score(bot_id="bot-e1", score=0.92, tier="ELITE", rank=1),
        }
        router = DPRDualEngineRouter(dpr_bridge=dpr)
        modifier = router.compute_risk_modifier("bot-e1")
        assert modifier == 1.0

    def test_score_0_85_returns_1_0(self):
        """Score exactly 0.85 returns modifier 1.0."""
        dpr = DPRBridge()
        dpr.scores = {
            "bot-e2": _make_dpr_score(bot_id="bot-e2", score=0.85, tier="ELITE", rank=1),
        }
        router = DPRDualEngineRouter(dpr_bridge=dpr)
        modifier = router.compute_risk_modifier("bot-e2")
        assert modifier == 1.0

    def test_score_0_75_returns_0_8(self):
        """Score in [0.70, 0.85) returns modifier 0.8."""
        dpr = DPRBridge()
        dpr.scores = {
            "bot-e3": _make_dpr_score(bot_id="bot-e3", score=0.75, tier="PERFORMING", rank=2),
        }
        router = DPRDualEngineRouter(dpr_bridge=dpr)
        modifier = router.compute_risk_modifier("bot-e3")
        assert modifier == 0.8

    def test_score_0_40_returns_0_3(self):
        """Score in [0.30, 0.50) returns modifier 0.3."""
        dpr = DPRBridge()
        dpr.scores = {
            "bot-e4": _make_dpr_score(bot_id="bot-e4", score=0.40, tier="AT_RISK", rank=8),
        }
        router = DPRDualEngineRouter(dpr_bridge=dpr)
        modifier = router.compute_risk_modifier("bot-e4")
        assert modifier == 0.3

    def test_score_0_20_returns_0_0(self):
        """Score < 0.30 returns modifier 0.0 (no position)."""
        dpr = DPRBridge()
        dpr.scores = {
            "bot-e5": _make_dpr_score(bot_id="bot-e5", score=0.20, tier="CIRCUIT_BROKEN", rank=1),
        }
        router = DPRDualEngineRouter(dpr_bridge=dpr)
        modifier = router.compute_risk_modifier("bot-e5")
        assert modifier == 0.0


# ---------------------------------------------------------------------------
# TestConflictResolution
# ---------------------------------------------------------------------------

class TestConflictResolution:
    def test_routing_no_modifier_gt_0_sets_modifier_0(self):
        """Routing NO + modifier > 0: risk wins, sets modifier to 0."""
        router = DPRDualEngineRouter()
        routing, modifier = router.resolve_conflict(
            bot_id="bot-conflict-1",
            routing_decision=False,
            risk_modifier=0.8,
        )
        assert routing is False
        assert modifier == 0.0

    def test_routing_yes_modifier_0_sets_routing_false(self):
        """Routing YES + modifier = 0: routing wins but set to NO."""
        router = DPRDualEngineRouter()
        routing, modifier = router.resolve_conflict(
            bot_id="bot-conflict-2",
            routing_decision=True,
            risk_modifier=0.0,
        )
        assert routing is False
        assert modifier == 0.0

    def test_no_conflict_returns_both(self):
        """No conflict: returns both values as-is."""
        router = DPRDualEngineRouter()
        routing, modifier = router.resolve_conflict(
            bot_id="bot-ok-3",
            routing_decision=True,
            risk_modifier=0.8,
        )
        assert routing is True
        assert modifier == 0.8


# ---------------------------------------------------------------------------
# TestRedisSync
# ---------------------------------------------------------------------------

class TestRedisSync:
    def test_syncs_to_redis(self):
        """sync_to_redis publishes scores to Redis publisher."""
        dpr = DPRBridge()
        pub = _make_mock_redis_publisher()
        dpr.scores = {
            "bot-sync-1": _make_dpr_score(bot_id="bot-sync-1", score=0.85, tier="ELITE", rank=1),
            "bot-sync-2": _make_dpr_score(bot_id="bot-sync-2", score=0.75, tier="PERFORMING", rank=2),
        }

        router = DPRDualEngineRouter(dpr_bridge=dpr, redis_publisher=pub)
        router.update_router_layer()
        count = router.sync_to_redis()

        assert count == 2
        assert pub.publish_single.call_count == 2

    def test_returns_count(self):
        """sync_to_redis returns count of bots synced."""
        dpr = DPRBridge()
        pub = _make_mock_redis_publisher()
        dpr.scores = {
            "bot-rc": _make_dpr_score(bot_id="bot-rc", score=0.90, tier="ELITE", rank=1),
        }

        router = DPRDualEngineRouter(dpr_bridge=dpr, redis_publisher=pub)
        router.update_router_layer()
        count = router.sync_to_redis()
        assert count == 1
