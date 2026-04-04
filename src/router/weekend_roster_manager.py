"""
Weekend Roster Manager
=====================

Fresh roster preparation + deployment to Commander's SessionDetector.
RoutingMatrix is NOT modified — only SessionDetector bot roster changes.

Reference: Story 8.13 (8-13-workflow-4-weekend-update-cycle) AC4
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class RosterDeploymentResult:
    """Result of roster deployment."""
    status: str  # "deployed", "failed", "no_change"
    roster: Optional[Dict[str, Any]] = None
    bots_deployed: int = 0
    deployed_at: Optional[datetime] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "roster": self.roster,
            "bots_deployed": self.bots_deployed,
            "deployed_at": self.deployed_at.isoformat() if self.deployed_at else None,
            "error": self.error,
        }


class WeekendRosterManager:
    """
    Weekend Roster Manager.

    Manages fresh roster preparation and deployment to Commander's SessionDetector.
    RoutingMatrix is NOT modified — only SessionDetector bot roster changes.
    """

    def __init__(self):
        self._current_roster: Optional[Dict[str, Any]] = None
        logger.info("WeekendRosterManager initialized")

    async def prepare_roster(self) -> Optional[Dict[str, Any]]:
        """
        Prepare fresh roster file for Monday deployment.

        Returns:
            Roster data or None if preparation failed
        """
        logger.info("Preparing fresh roster for Monday")

        try:
            # 1. Get current active bots from SessionDetector
            active_bots = await self._get_active_bots()

            if not active_bots:
                logger.warning("No active bots found for roster")
                return None

            # 2. Apply any refinements from Saturday
            refined_bots = await self._apply_refinements(active_bots)

            # 3. Sort by DPR score for Monday queue order
            ranked_bots = await self._rank_bots_for_monday(refined_bots)

            # 4. Create roster
            self._current_roster = {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "bots": ranked_bots,
                "session_configs": await self._get_session_configs(ranked_bots),
                "metadata": {
                    "week_number": datetime.now().isocalendar()[1],
                    "year": datetime.now().year,
                    "refinement_count": len(refined_bots),
                },
            }

            # 5. Store roster file
            await self._store_roster()

            logger.info(f"Roster prepared: {len(ranked_bots)} bots")
            return self._current_roster

        except Exception as e:
            logger.error(f"Error preparing roster: {e}", exc_info=True)
            return None

    async def deploy_roster(self) -> RosterDeploymentResult:
        """
        Deploy fresh roster to Commander's SessionDetector.

        Returns:
            RosterDeploymentResult with deployment outcome
        """
        logger.info("Deploying fresh roster to SessionDetector")

        try:
            # Load roster if not already prepared
            if self._current_roster is None:
                roster = await self._load_latest_roster()
                if roster is None:
                    return RosterDeploymentResult(
                        status="failed",
                        error="No roster available for deployment",
                    )
                self._current_roster = roster

            # Deploy to SessionDetector
            success = await self._deploy_to_session_detector(self._current_roster)

            if success:
                # Trigger SQS Monday warmup
                from src.router.sqs_monday_warmup import get_sqs_monday_warmup
                warmup = get_sqs_monday_warmup()
                await warmup.execute_warmup()

                return RosterDeploymentResult(
                    status="deployed",
                    roster=self._current_roster,
                    bots_deployed=len(self._current_roster.get("bots", [])),
                    deployed_at=datetime.now(timezone.utc),
                )
            else:
                return RosterDeploymentResult(
                    status="failed",
                    error="Deployment to SessionDetector failed",
                )

        except Exception as e:
            logger.error(f"Error deploying roster: {e}", exc_info=True)
            return RosterDeploymentResult(
                status="failed",
                error=str(e),
            )

    async def _get_active_bots(self) -> List[str]:
        """Get current active bot IDs."""
        try:
            from src.router.bot_manifest import BotRegistry

            registry = BotRegistry()
            active_bots = registry.list_live_trading()
            return [bot.bot_id for bot in active_bots]

        except Exception as e:
            logger.error(f"Error getting active bots: {e}")
            return []

    async def _apply_refinements(self, bot_ids: List[str]) -> List[str]:
        """
        Apply Saturday refinements to bot list.

        Placeholder - would apply parameter changes from Saturday refinement.
        """
        # Placeholder - would filter/modify bots based on Saturday refinement results
        return bot_ids

    async def _rank_bots_for_monday(self, bot_ids: List[str]) -> List[str]:
        """
        Rank bots for Monday queue order using DPR scores.

        Placeholder - would integrate with DPR scoring engine.
        """
        try:
            from src.router.dpr_scoring_engine import get_dpr_scoring_engine

            scoring_engine = get_dpr_scoring_engine()
            scores = await scoring_engine.score_all_bots(bot_ids)

            # Sort by composite score descending
            scores.sort(key=lambda s: s.composite_score, reverse=True)

            return [s.bot_id for s in scores]

        except Exception as e:
            logger.error(f"Error ranking bots: {e}")
            return bot_ids  # Return unsorted

    async def _get_session_configs(self, bot_ids: List[str]) -> Dict[str, Any]:
        """
        Get session configurations for bots.

        Placeholder - would get session configs from SessionDetector.
        """
        # Placeholder - would return session configs for each bot
        return {bot_id: {"session_enabled": True} for bot_id in bot_ids}

    async def _store_roster(self) -> None:
        """Store roster file to cold storage."""
        if self._current_roster is None:
            return

        try:
            from src.router.cold_storage_writer import get_cold_storage_writer

            writer = get_cold_storage_writer()
            date_str = datetime.now().strftime("%Y-%m-%d")
            file_key = f"weekend_roster/{date_str}_roster.json"

            await writer.write(file_key, self._current_roster)
            logger.info(f"Roster stored to cold storage: {file_key}")

        except Exception as e:
            logger.error(f"Error storing roster: {e}", exc_info=True)

    async def _load_latest_roster(self) -> Optional[Dict[str, Any]]:
        """Load latest roster from cold storage."""
        try:
            from src.router.cold_storage_writer import get_cold_storage_writer

            writer = get_cold_storage_writer()

            # Try to find most recent roster
            # In production, would list files and find latest
            # For now, return None
            return None

        except Exception as e:
            logger.error(f"Error loading roster: {e}")
            return None

    async def _deploy_to_session_detector(self, roster: Dict[str, Any]) -> bool:
        """
        Deploy roster to Commander's SessionDetector.

        This updates the SessionDetector's bot roster but does NOT modify RoutingMatrix.

        Args:
            roster: Roster data to deploy

        Returns:
            True if deployment succeeded
        """
        try:
            from src.router.commander import get_commander

            commander = get_commander()

            # Get SessionDetector from commander
            session_detector = getattr(commander, 'session_detector', None)
            if not session_detector:
                logger.warning("SessionDetector not found on Commander, attempting direct roster update")
                return await self._deploy_direct(roster)

            # Deploy roster to SessionDetector
            bot_ids = roster.get('bots', [])
            session_configs = roster.get('session_configs', {})

            success = await session_detector.update_bot_roster(
                bot_ids=bot_ids,
                session_configs=session_configs,
                source="weekend_update_cycle",
            )

            if success:
                logger.info(f"Deployed {len(bot_ids)} bots to SessionDetector")
            else:
                logger.error("SessionDetector rejected roster update")

            return success

        except AttributeError:
            logger.warning("Commander or SessionDetector not fully implemented, using direct deployment")
            return await self._deploy_direct(roster)
        except Exception as e:
            logger.error(f"Error deploying to SessionDetector: {e}", exc_info=True)
            return False

    async def _deploy_direct(self, roster: Dict[str, Any]) -> bool:
        """
        Direct roster deployment when SessionDetector is not available.

        Writes roster to a shared location that SessionDetector reads on startup.

        Args:
            roster: Roster data to deploy

        Returns:
            True if deployment succeeded
        """
        try:
            from src.router.cold_storage_writer import get_cold_storage_writer

            writer = get_cold_storage_writer()

            # Write roster to deployment location
            deployment_key = "weekend_roster/deployed_roster.json"
            await writer.write(deployment_key, roster)

            # Also write a marker file for SessionDetector to pick up
            marker_key = "weekend_roster/.deployment_marker"
            await writer.write(marker_key, {
                "deployed_at": datetime.now(timezone.utc).isoformat(),
                "bot_count": len(roster.get('bots', [])),
                "ready": True,
            })

            logger.info(
                f"Direct roster deployment: {len(roster.get('bots', []))} bots written to cold storage"
            )
            return True

        except Exception as e:
            logger.error(f"Error in direct roster deployment: {e}", exc_info=True)
            return False


# ============= Singleton Factory =============
_manager_instance: Optional[WeekendRosterManager] = None


def get_weekend_roster_manager() -> WeekendRosterManager:
    """Get singleton instance of WeekendRosterManager."""
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = WeekendRosterManager()
    return _manager_instance
