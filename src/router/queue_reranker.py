"""
Queue Re-rank + SESSION_CONCERN Flags
======================================

Queue Re-rank step (17:30 GMT).
Updates ranked queue for next session (NY open) and applies
SESSION_CONCERN flags to bots with 3 consecutive negative EV sessions.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Any

from src.router.dpr_scoring_engine import DprScore

logger = logging.getLogger(__name__)


@dataclass
class QueueReRankResult:
    """Result of Queue Re-rank step."""
    queue: List[str]  # Ordered bot IDs
    concerns: List[str]  # Bot IDs with SESSION_CONCERN flag

    def to_dict(self) -> Dict[str, Any]:
        return {
            "queue": self.queue,
            "concerns": self.concerns,
            "concerns_count": len(self.concerns),
        }


class QueueReRanker:
    """
    Queue Re-rank step (17:30 GMT).

    Updates ranked queue for next session (NY open) and applies
    SESSION_CONCERN flags to bots with 3 consecutive negative EV sessions.
    """

    CONSECUTIVE_NEGATIVE_EV_THRESHOLD = 3

    def __init__(self):
        logger.info("QueueReRanker initialized")

    async def run(self, dpr_scores: List[DprScore]) -> QueueReRankResult:
        """
        Run Queue Re-rank and apply SESSION_CONCERN flags.

        Args:
            dpr_scores: List of DprScore objects from DPR Update step

        Returns:
            QueueReRankResult with ordered queue and concern flags
        """
        # Apply SESSION_CONCERN flags
        concerns = []
        for score in dpr_scores:
            if score.consecutive_negative_ev >= self.CONSECUTIVE_NEGATIVE_EV_THRESHOLD:
                score.session_concern = True
                await self._flag_session_concern(score.bot_id)
                concerns.append(score.bot_id)
                logger.info(
                    f"Bot {score.bot_id} flagged with SESSION_CONCERN: "
                    f"{score.consecutive_negative_ev} consecutive negative EV sessions"
                )

                # Dispatch to Research Head for hypothesis review
                await self._dispatch_session_concern_to_research(score)

        # Sort by composite score for final queue order
        ranked_queue = sorted(dpr_scores, key=lambda s: s.composite_score, reverse=True)
        queue = [s.bot_id for s in ranked_queue]

        # Update routing matrix/commander queue
        await self._update_queue(ranked_queue)

        result = QueueReRankResult(
            queue=queue,
            concerns=concerns,
        )

        logger.info(
            f"Queue Re-rank complete: {len(queue)} bots in queue, "
            f"{len(concerns)} with SESSION_CONCERN"
        )

        return result

    async def _flag_session_concern(self, bot_id: str) -> None:
        """
        Apply SESSION_CONCERN flag to a bot.

        Args:
            bot_id: Bot ID to flag
        """
        try:
            from src.router.bot_manifest import BotRegistry

            registry = BotRegistry()
            bot = registry.get(bot_id)

            if bot:
                tags = list(bot.tags) if bot.tags else []
                if 'SESSION_CONCERN' not in tags:
                    tags.append('SESSION_CONCERN')
                    bot.tags = tags
                    logger.info(f"Applied SESSION_CONCERN flag to bot {bot_id}")

        except Exception as e:
            logger.warning(f"Could not apply SESSION_CONCERN flag to {bot_id}: {e}")

    async def _update_queue(self, ranked_queue: List[DprScore]) -> None:
        """
        Update routing matrix/commander queue with new rankings.

        Args:
            ranked_queue: List of DprScore objects sorted by composite score
        """
        try:
            from src.router.commander import get_commander

            commander = get_commander()

            # Build new queue in rank order
            new_queue = [s.bot_id for s in ranked_queue]

            # Update commander queue
            if hasattr(commander, 'update_queue'):
                commander.update_queue(new_queue)
                logger.info(f"Commander queue updated with {len(new_queue)} bots")

        except Exception as e:
            logger.warning(f"Could not update commander queue: {e}")

    async def _dispatch_session_concern_to_research(self, score: DprScore) -> None:
        """
        Dispatch SESSION_CONCERN alert to Research department for hypothesis review.

        Args:
            score: DprScore object with bot's scoring data
        """
        import json

        try:
            from src.agents.departments.department_mail import (
                DepartmentMailService,
                RedisDepartmentMailService,
                get_redis_mail_service,
                MessageType,
                Priority,
            )
            from src.router.bot_manifest import BotRegistry

            # Get additional bot metadata from registry
            registry = BotRegistry()
            bot = registry.get(score.bot_id)
            symbols = bot.symbols if bot else []
            strategy_type = bot.strategy_type.value if bot else "UNKNOWN"

            # Use Redis mail service if available, fall back to SQLite
            try:
                mail = get_redis_mail_service()
            except Exception:
                logger.warning("Redis mail service unavailable, falling back to SQLite")
                mail = DepartmentMailService()
            mail.send(
                from_dept="risk",
                to_dept="research",
                type=MessageType.DISPATCH,
                subject=f"SESSION_CONCERN: Strategy {score.bot_id} hypothesis review needed",
                body=json.dumps({
                    "task_type": "CONTRADICTED_HYPOTHESIS_REVIEW",
                    "bot_id": score.bot_id,
                    "strategy_type": strategy_type,
                    "symbols": symbols,
                    "consecutive_negative_ev": score.consecutive_negative_ev,
                    "composite_score": round(score.composite_score, 2),
                    "tier": score.tier,
                    "priority": "HIGH"
                }),
                priority=Priority.HIGH,
            )
            logger.info(f"Dispatched SESSION_CONCERN mail to research for bot {score.bot_id}")

        except Exception as e:
            logger.warning(f"Could not dispatch SESSION_CONCERN mail: {e}")


# ============= Singleton Factory =============
_queue_reranker: QueueReRanker | None = None


def get_queue_reranker() -> QueueReRanker:
    """Get singleton instance of QueueReRanker."""
    global _queue_reranker
    if _queue_reranker is None:
        _queue_reranker = QueueReRanker()
    return _queue_reranker
