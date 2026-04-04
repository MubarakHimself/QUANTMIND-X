"""
Queue Tier Remix
================

Queue tier remix computation at DPR Update step (17:00 GMT).
T1/T3/T2 interleaving based on updated DPR scores.
"""

import logging
from typing import Dict, List, Any

from src.router.dpr_scoring_engine import DprScore

logger = logging.getLogger(__name__)


class QueueRemix:
    """
    Queue tier remix computation at DPR Update step (17:00 GMT).

    T1/T3/T2 interleaving based on updated DPR scores.
    """

    def __init__(self):
        logger.info("QueueRemix initialized")

    def compute_remix(self, dpr_scores: List[DprScore]) -> Dict[str, Any]:
        """
        Compute queue tier remix.

        Args:
            dpr_scores: List of DprScore objects

        Returns:
            Dict with tier names and ordered bot_id lists, plus interleaved order
        """
        t1_bots = [s for s in dpr_scores if s.tier == "T1"]
        t2_bots = [s for s in dpr_scores if s.tier == "T2"]
        t3_bots = [s for s in dpr_scores if s.tier == "T3"]

        # Sort each tier by composite score descending
        t1_bots.sort(key=lambda s: s.composite_score, reverse=True)
        t2_bots.sort(key=lambda s: s.composite_score, reverse=True)
        t3_bots.sort(key=lambda s: s.composite_score, reverse=True)

        # Interleave: highest T1, highest T3, highest T2, repeat
        interleaved = []
        max_len = max(len(t1_bots), len(t2_bots), len(t3_bots))
        for i in range(max_len):
            if i < len(t1_bots):
                interleaved.append(t1_bots[i].bot_id)
            if i < len(t3_bots):
                interleaved.append(t3_bots[i].bot_id)
            if i < len(t2_bots):
                interleaved.append(t2_bots[i].bot_id)

        result = {
            "T1": [b.bot_id for b in t1_bots],
            "T2": [b.bot_id for b in t2_bots],
            "T3": [b.bot_id for b in t3_bots],
            "interleaved_order": interleaved,
            "tier_counts": {
                "T1": len(t1_bots),
                "T2": len(t2_bots),
                "T3": len(t3_bots),
            },
        }

        logger.info(
            f"Queue remix computed: T1={len(t1_bots)}, T2={len(t2_bots)}, "
            f"T3={len(t3_bots)}, interleaved={len(interleaved)}"
        )

        return result

    async def update_routing_matrix(self, remix_result: Dict[str, Any]) -> None:
        """
        Update the routing matrix with new tier assignments.

        Args:
            remix_result: Result from compute_remix
        """
        try:
            from src.router.routing_matrix import get_routing_matrix

            matrix = get_routing_matrix()

            # Update tier assignments
            for tier_name, bot_ids in [("T1", remix_result["T1"]),
                                         ("T2", remix_result["T2"]),
                                         ("T3", remix_result["T3"])]:
                for rank, bot_id in enumerate(bot_ids):
                    matrix.set_bot_tier(bot_id, tier_name, rank)

            logger.info("Routing matrix updated with new tier assignments")

        except Exception as e:
            logger.warning(f"Could not update routing matrix: {e}")


# ============= Singleton Factory =============
_queue_remix: QueueRemix | None = None


def get_queue_remix() -> QueueRemix:
    """Get singleton instance of QueueRemix."""
    global _queue_remix
    if _queue_remix is None:
        _queue_remix = QueueRemix()
    return _queue_remix
