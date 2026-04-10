"""
QuantMindLib V1 — DPR Dual Engine Router

Phase 10 Packet 10C: DPRDualEngineRouter
Handles DPR dual-engine routing (router layer + risk layer).
"""
from __future__ import annotations

from typing import Dict, Optional

from src.library.core.bridges.sentinel_dpr_bridges import DPRBridge
from src.library.core.bridges.dpr_redis_bridge import DPRRedisPublisher
from src.library.core.bridges.registry_journal_bridges import RegistryBridge


# Tier order for eligibility checks
_ROUTING_TIER_ORDER: Dict[str, int] = {
    "ELITE": 0,
    "PERFORMING": 1,
    "STANDARD": 2,
    "AT_RISK": 3,
    "CIRCUIT_BROKEN": 4,
}

# Tiers that are routing-eligible (STANDARD or better)
_ROUTING_ELIGIBLE_MAX_ORDER = 2  # STANDARD is order 2


class DPRDualEngineRouter:
    """
    Handles DPR dual-engine routing (router layer + risk layer).

    The DPR system has two engines:
    1. Router layer: fast selection decision (which bots can trade)
       - Uses DPR rank + tier thresholds
       - Feed into Governor's activation decisions
    2. Risk layer: pre-trade position sizing (how much to risk)
       - Uses DPR score for Kelly sizing modifiers
       - Feed into RiskEnvelope.position_size

    This bridge coordinates both layers and resolves conflicts.
    """

    def __init__(
        self,
        dpr_bridge: Optional[DPRBridge] = None,
        redis_publisher: Optional[DPRRedisPublisher] = None,
        registry_bridge: Optional[RegistryBridge] = None,
    ) -> None:
        self._dpr_bridge = dpr_bridge or DPRBridge()
        self._redis_publisher = redis_publisher
        self._registry_bridge = registry_bridge or RegistryBridge()
        # bot_id -> is_routing_eligible
        self._router_cache: Dict[str, bool] = {}
        # bot_id -> Kelly modifier
        self._risk_modifiers: Dict[str, float] = {}

    def update_router_layer(self) -> Dict[str, bool]:
        """
        Update router layer: determine which bots are routing-eligible.

        Routing eligibility:
        - Bot tier must be STANDARD or better (not AT_RISK, not CIRCUIT_BROKEN)
        - DPR rank must be <= 10
        - Bot must be ACTIVE in registry

        Returns dict of bot_id -> is_eligible.
        Updates _router_cache.
        """
        scores = self._dpr_bridge.scores
        active_records = self._registry_bridge.get_active_bots()
        active_bot_ids = {r.bot_id for r in active_records}

        result: Dict[str, bool] = {}

        for bot_id, score in scores.items():
            tier_order = _ROUTING_TIER_ORDER.get(score.tier, 99)
            tier_eligible = tier_order <= _ROUTING_ELIGIBLE_MAX_ORDER
            rank_eligible = score.rank <= 10
            registry_active = bot_id in active_bot_ids

            is_eligible = tier_eligible and rank_eligible and registry_active
            result[bot_id] = is_eligible
            self._router_cache[bot_id] = is_eligible

            # Also update risk modifier in sync
            self._risk_modifiers[bot_id] = self.compute_risk_modifier(bot_id)

        return result

    def get_routing_decision(self, bot_id: str) -> bool:
        """
        Fast routing decision for Governor.
        Returns True if bot is routing-eligible.
        """
        # Try cache first
        if bot_id in self._router_cache:
            return self._router_cache[bot_id]
        # If not in cache, compute from current DPR scores
        score = self._dpr_bridge.get_score(bot_id)
        if score is None:
            return False
        tier_order = _ROUTING_TIER_ORDER.get(score.tier, 99)
        tier_eligible = tier_order <= _ROUTING_ELIGIBLE_MAX_ORDER
        rank_eligible = score.rank <= 10
        registry_active = self._registry_bridge.get_record(bot_id) is not None
        return tier_eligible and rank_eligible and registry_active

    def compute_risk_modifier(self, bot_id: str) -> float:
        """
        Compute Kelly position sizing modifier from DPR score.

        Modifier formula:
        - score >= 0.85: modifier = 1.0 (full Kelly)
        - 0.70 <= score < 0.85: modifier = 0.8
        - 0.50 <= score < 0.70: modifier = 0.6
        - 0.30 <= score < 0.50: modifier = 0.3
        - score < 0.30: modifier = 0.0 (no position)
        """
        score = self._dpr_bridge.get_score(bot_id)
        if score is None:
            return 0.0

        dpr_score = score.dpr_score

        if dpr_score >= 0.85:
            return 1.0
        elif dpr_score >= 0.70:
            return 0.8
        elif dpr_score >= 0.50:
            return 0.6
        elif dpr_score >= 0.30:
            return 0.3
        else:
            return 0.0

    def sync_to_redis(self) -> int:
        """
        Sync both engine states to Redis via DPRRedisPublisher.
        Returns count of bots synced.
        """
        if self._redis_publisher is None:
            return 0

        count = 0
        for bot_id, is_eligible in self._router_cache.items():
            score = self._dpr_bridge.get_score(bot_id)
            if score is not None:
                if self._redis_publisher.publish_single(bot_id, score):
                    count += 1
        return count

    def resolve_conflict(
        self,
        bot_id: str,
        routing_decision: bool,
        risk_modifier: float,
    ) -> tuple[bool, float]:
        """
        Resolve conflicts between router and risk layers.

        Conflict resolution:
        - If routing says NO but risk modifier > 0: risk wins, set modifier=0
        - If routing says YES but risk modifier = 0: routing wins, keep routing=NO
        - Otherwise: use both as-is
        """
        if not routing_decision and risk_modifier > 0:
            # Router says no but risk says position allowed -> risk wins, modifier=0
            return (False, 0.0)
        elif routing_decision and risk_modifier == 0.0:
            # Router says yes but risk modifier is 0 -> routing wins, routing=NO
            return (False, 0.0)
        else:
            # No conflict or both agree
            return (routing_decision, risk_modifier)


__all__ = ["DPRDualEngineRouter"]
