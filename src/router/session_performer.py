"""
Session Performer Identifier
============================

Session Performer ID step — scores bots on session-specific metrics
and applies SESSION_SPECIALIST tags.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class SessionPerformerResult:
    """Result of Session Performer ID for a single bot."""
    bot_id: str
    regime_expectation: float
    actual_performance: float
    outperformance: float
    is_specialist: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bot_id": self.bot_id,
            "regime_expectation": round(self.regime_expectation, 4),
            "actual_performance": round(self.actual_performance, 4),
            "outperformance": round(self.outperformance, 4),
            "is_specialist": self.is_specialist,
        }


@dataclass
class SessionPerformerOutput:
    """Output of the Session Performer ID step."""
    results: List[SessionPerformerResult]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "results": [r.to_dict() for r in self.results],
            "total_bots": len(self.results),
            "specialists_count": sum(1 for r in self.results if r.is_specialist),
        }


class SessionPerformerIdentifier:
    """
    Session Performer ID step — scores bots on session-specific metrics
    and applies SESSION_SPECIALIST tags.

    A bot is tagged as SESSION_SPECIALIST if it outperforms its regime
    expectation by more than REGIME_OUTPERFORMANCE_THRESHOLD (15%).
    """

    REGIME_OUTPERFORMANCE_THRESHOLD = 0.15  # 15% above regime expectation

    def __init__(self):
        logger.info("SessionPerformerIdentifier initialized")

    async def run(self) -> SessionPerformerOutput:
        """
        Run Session Performer ID for all active bots.

        Returns:
            SessionPerformerOutput with results for each bot
        """
        bots = await self._get_active_bots()
        results = []

        logger.info(f"Running Session Performer ID for {len(bots)} active bots")

        for bot in bots:
            regime_expectation = await self._get_regime_expectation(bot)
            actual_performance = await self._get_session_performance(bot)
            outperformance = actual_performance - regime_expectation

            is_specialist = (
                outperformance > self.REGIME_OUTPERFORMANCE_THRESHOLD and
                actual_performance > 0
            )

            if is_specialist:
                await self._apply_session_specialist_tag(bot)

            results.append(SessionPerformerResult(
                bot_id=bot,
                regime_expectation=regime_expectation,
                actual_performance=actual_performance,
                outperformance=outperformance,
                is_specialist=is_specialist,
            ))

        specialist_count = sum(1 for r in results if r.is_specialist)
        logger.info(
            f"Session Performer ID complete: {len(bots)} bots scored, "
            f"{specialist_count} tagged as SESSION_SPECIALIST"
        )

        return SessionPerformerOutput(results=results)

    async def _get_active_bots(self) -> List[str]:
        """Get list of active bot IDs."""
        try:
            from src.router.bot_manifest import BotRegistry

            registry = BotRegistry.get_instance()
            active_bots = registry.get_active_bots()
            return [bot.bot_id for bot in active_bots]

        except Exception as e:
            logger.warning(f"Could not get active bots: {e}")
            return []

    async def _get_regime_expectation(self, bot_id: str) -> float:
        """
        Get regime expectation for a bot.

        The regime expectation is the expected win rate for the current regime,
        based on historical performance in similar regimes.
        """
        try:
            from src.router.hmm_deployment import get_deployment_manager

            deployment = get_deployment_manager()
            current_state = deployment.get_current_state()

            if current_state:
                # Map deployment mode to regime-based expectations
                # HMMDeploymentManager uses DeploymentMode, not regime strings
                mode = current_state.mode
                # Use mode-based expectations as proxy for regime
                mode_expectations = {
                    "TREND": 0.55,
                    "RANGE": 0.52,
                    "BREAKOUT": 0.50,
                    "CHAOS": 0.45,
                }
                # Default to RANGE if mode not recognized
                return mode_expectations.get(str(mode), 0.50)

        except Exception as e:
            logger.warning(f"Could not get regime expectation for {bot_id}: {e}")

        return 0.50  # Default expectation

    async def _get_session_performance(self, bot_id: str) -> float:
        """
        Get session performance for a bot.

        Returns the actual win rate achieved during the session.
        """
        try:
            from src.router.bot_manifest import BotRegistry

            registry = BotRegistry()
            bot = registry.get(bot_id)

            if bot:
                # Try to get session metrics from live_stats first, then paper_stats
                stats = getattr(bot, 'live_stats', None) or getattr(bot, 'paper_stats', None)
                if stats and hasattr(stats, 'win_rate'):
                    return stats.win_rate
                return getattr(bot, 'win_rate', 0.5)

        except Exception as e:
            logger.warning(f"Could not get session performance for {bot_id}: {e}")

        return 0.5  # Default to 50% win rate

    async def _apply_session_specialist_tag(self, bot_id: str) -> None:
        """
        Apply SESSION_SPECIALIST tag to a bot.

        This would update the bot manifest/state with the tag.
        """
        try:
            from src.router.bot_manifest import BotRegistry

            registry = BotRegistry()
            bot = registry.get(bot_id)

            if bot:
                # Apply the tag
                tags = list(bot.tags) if bot.tags else []
                if 'SESSION_SPECIALIST' not in tags:
                    tags.append('SESSION_SPECIALIST')
                    bot.tags = tags
                    logger.info(f"Applied SESSION_SPECIALIST tag to bot {bot_id}")

        except Exception as e:
            logger.warning(f"Could not apply SESSION_SPECIALIST tag to {bot_id}: {e}")


# ============= Singleton Factory =============
_session_performer_identifier: Optional[SessionPerformerIdentifier] = None


def get_session_performer_identifier() -> SessionPerformerIdentifier:
    """Get singleton instance of SessionPerformerIdentifier."""
    global _session_performer_identifier
    if _session_performer_identifier is None:
        _session_performer_identifier = SessionPerformerIdentifier()
    return _session_performer_identifier
