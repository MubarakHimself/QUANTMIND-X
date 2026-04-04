"""
Saturday Refinement Service
==========================

Saturday workflow step — Workflow 2 refinement + Walk-Forward Analysis.
Runs Saturday 06:00-18:00 GMT.

Reference: Story 8.13 (8-13-workflow-4-weekend-update-cycle) AC2
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

from src.router.bot_manifest import BotRegistry

logger = logging.getLogger(__name__)


@dataclass
class RefinementChange:
    """Represents a parameter or regime filter change."""
    parameter_name: str
    old_value: any
    new_value: any
    reason: str


@dataclass
class RefinementResult:
    """Result of refining a single bot."""
    bot_id: str
    param_changes: List[RefinementChange]
    regime_filter_updates: Dict[str, Any]
    wfa_result: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bot_id": self.bot_id,
            "param_changes": [
                {
                    "parameter_name": c.parameter_name,
                    "old_value": c.old_value,
                    "new_value": c.new_value,
                    "reason": c.reason,
                }
                for c in self.param_changes
            ],
            "regime_filter_updates": self.regime_filter_updates,
            "wfa_result": self.wfa_result,
        }


@dataclass
class SaturdayRefinementResults:
    """Results of Saturday refinement for all bots."""
    results: List[RefinementResult]
    total_refined: int
    timestamp: datetime

    def to_dict(self) -> Dict[str, Any]:
        return {
            "results": [r.to_dict() for r in self.results],
            "total_refined": self.total_refined,
            "timestamp": self.timestamp.isoformat(),
        }


class SaturdayRefinementService:
    """
    Saturday workflow step — Workflow 2 refinement + Walk-Forward Analysis.

    Runs: Saturday 06:00-18:00 GMT
    - Workflow 2 refinement on selected bots
    - Parameter adjustments based on Friday analysis
    - Regime filter updates
    - Walk-Forward Analysis on validated candidates
    """

    WFA_WINDOW_DAYS = 14  # 14-day rolling data
    WFA_MAX_CANDIDATES = 5

    def __init__(self):
        logger.info("SaturdayRefinementService initialized")

    async def run(self, candidate_bot_ids: List[str]) -> SaturdayRefinementResults:
        """
        Execute Saturday refinement workflow.

        Args:
            candidate_bot_ids: List of bot IDs selected from Friday analysis

        Returns:
            SaturdayRefinementResults with all refinement outcomes
        """
        logger.info(f"Starting Saturday refinement for {len(candidate_bot_ids)} candidates")

        results = []
        candidates = candidate_bot_ids[:self.WFA_MAX_CANDIDATES]

        for bot_id in candidates:
            try:
                refinement = await self._run_workflow2_refinement(bot_id)

                # Apply parameter changes (would call parameter guard here)
                if refinement.param_changes:
                    await self._apply_parameter_changes(bot_id, refinement.param_changes)

                # Update regime filters
                if refinement.regime_filter_updates:
                    await self._update_regime_filters(bot_id, refinement.regime_filter_updates)

                # Run Walk-Forward Analysis
                from src.router.walk_forward_analyzer import get_walk_forward_analyzer
                analyzer = get_walk_forward_analyzer()
                wfa_result = await analyzer.run(bot_id)

                refinement.wfa_result = wfa_result.to_dict() if hasattr(wfa_result, 'to_dict') else wfa_result

                results.append(refinement)

            except Exception as e:
                logger.error(f"Error refining bot {bot_id}: {e}", exc_info=True)
                # Continue with other bots
                results.append(RefinementResult(
                    bot_id=bot_id,
                    param_changes=[],
                    regime_filter_updates={},
                    wfa_result={"error": str(e)},
                ))

        final_results = SaturdayRefinementResults(
            results=results,
            total_refined=len(results),
            timestamp=datetime.now(timezone.utc),
        )

        # Store results
        await self._store_refinement_results(final_results)

        logger.info(f"Saturday refinement complete: {len(results)} bots processed")
        return final_results

    async def _run_workflow2_refinement(self, bot_id: str) -> RefinementResult:
        """
        Run Workflow 2 refinement for a single bot.

        Integrates with the DPR scoring engine to get current bot performance
        and proposes parameter adjustments based on regime behaviour analysis.
        """
        param_changes = []
        regime_filter_updates = {}

        try:
            # Get bot's current performance from DPR scoring engine
            from src.router.dpr_scoring_engine import get_dpr_scoring_engine
            from src.router.bot_manifest import BotRegistry

            scoring_engine = get_dpr_scoring_engine()
            registry = BotRegistry()

            # Get bot manifest
            bot = registry.get_bot(bot_id)
            if not bot:
                logger.warning(f"Bot {bot_id} not found in registry")
                return RefinementResult(
                    bot_id=bot_id,
                    param_changes=param_changes,
                    regime_filter_updates=regime_filter_updates,
                )

            # Get DPR score to assess if refinement is needed
            scores = await scoring_engine.score_all_bots([bot_id])
            if scores:
                dpr_score = scores[0]

                # If DPR score is below threshold, propose parameter adjustments
                if dpr_score.composite_score < 0.5:
                    # Get current parameters from bot manifest
                    current_params = getattr(bot, 'parameters', {})

                    # Analyze regime mismatch from HMM sensor data
                    from src.risk.physics.correlation_sensor import CorrelationSensor
                    sensor = CorrelationSensor()
                    regime_mismatch = sensor.get_bot_regime_mismatch(bot_id)

                    if regime_mismatch:
                        # Propose regime filter update
                        regime_filter_updates = {
                            "regime_filter_enabled": True,
                            "preferred_regimes": regime_mismatch.get("preferred_regimes", []),
                            "avoid_regimes": regime_mismatch.get("avoid_regimes", []),
                            "confidence_threshold": regime_mismatch.get("confidence", 0.7),
                        }

                        # Propose parameter adjustments based on regime
                        # Example: increase stop loss in volatile regimes
                        param_adjustments = self._calculate_regime_based_adjustments(
                            current_params, regime_mismatch
                        )

                        for param_name, (old_val, new_val) in param_adjustments.items():
                            param_changes.append(RefinementChange(
                                parameter_name=param_name,
                                old_value=old_val,
                                new_value=new_val,
                                reason=f"Regime-based adjustment for {regime_mismatch.get('current_regime', 'UNKNOWN')}",
                            ))

            logger.info(
                f"Workflow 2 refinement for {bot_id}: "
                f"{len(param_changes)} param changes, {len(regime_filter_updates)} regime updates"
            )

        except Exception as e:
            logger.error(f"Error in Workflow 2 refinement for {bot_id}: {e}", exc_info=True)
            # Continue with empty changes rather than failing the whole workflow

        return RefinementResult(
            bot_id=bot_id,
            param_changes=param_changes,
            regime_filter_updates=regime_filter_updates,
        )

    def _calculate_regime_based_adjustments(
        self,
        current_params: Dict[str, Any],
        regime_mismatch: Dict[str, Any]
    ) -> Dict[str, tuple]:
        """
        Calculate parameter adjustments based on regime analysis.

        Args:
            current_params: Current bot parameters
            regime_mismatch: Regime mismatch analysis result

        Returns:
            Dict of parameter_name -> (old_value, new_value)
        """
        adjustments = {}
        current_regime = regime_mismatch.get("current_regime", "UNKNOWN")

        # Adjust risk parameters based on regime
        if current_regime == "VOLATILE":
            # In volatile regimes, reduce position size and widen stops
            if "risk_per_trade" in current_params:
                adjustments["risk_per_trade"] = (
                    current_params["risk_per_trade"],
                    current_params["risk_per_trade"] * 0.8,  # Reduce by 20%
                )
            if "stop_loss_pips" in current_params:
                adjustments["stop_loss_pips"] = (
                    current_params["stop_loss_pips"],
                    current_params["stop_loss_pips"] * 1.2,  # Widen by 20%
                )
        elif current_regime == "TREND":
            # In trend regimes, slightly increase position size
            if "risk_per_trade" in current_params:
                adjustments["risk_per_trade"] = (
                    current_params["risk_per_trade"],
                    current_params["risk_per_trade"] * 1.1,  # Increase by 10%
                )

        return adjustments

    async def _apply_parameter_changes(
        self,
        bot_id: str,
        changes: List[RefinementChange]
    ) -> None:
        """
        Apply parameter changes to a bot.

        Args:
            bot_id: Bot ID
            changes: List of parameter changes to apply
        """
        try:
            # Import guard to check weekday blocking
            from src.router.weekday_parameter_guard import get_weekday_parameter_guard

            guard = get_weekday_parameter_guard()

            # Check if changes are allowed (should always be allowed on Saturday)
            if not guard.is_change_allowed():
                logger.warning(f"Weekday guard blocking parameter changes on {bot_id}")

            # Apply changes via Commander
            from src.router.commander import get_commander

            commander = get_commander()

            for change in changes:
                try:
                    # This would call the actual parameter update
                    # await commander.set_bot_parameter(bot_id, change.parameter_name, change.new_value)
                    logger.info(
                        f"Would apply param change: bot={bot_id}, "
                        f"param={change.parameter_name}, value={change.new_value}"
                    )
                except Exception as e:
                    logger.error(f"Error applying param change {change.parameter_name}: {e}")

        except Exception as e:
            logger.error(f"Error applying parameter changes for {bot_id}: {e}", exc_info=True)

    async def _update_regime_filters(
        self,
        bot_id: str,
        updates: Dict[str, Any]
    ) -> None:
        """
        Update regime filters for a bot.

        Args:
            bot_id: Bot ID
            updates: Dictionary of regime filter updates
        """
        logger.info(f"Updating regime filters for {bot_id}: {updates}")
        # Placeholder - would integrate with actual regime filter system

    async def _store_refinement_results(self, results: SaturdayRefinementResults) -> None:
        """Store Saturday refinement results to cold storage."""
        try:
            from src.router.cold_storage_writer import get_cold_storage_writer

            writer = get_cold_storage_writer()
            date_str = datetime.now().strftime("%Y-%m-%d")
            file_key = f"saturday_refinement/{date_str}_saturday_refinement.json"

            await writer.write(file_key, results.to_dict())
            logger.info(f"Saturday refinement stored to cold storage: {file_key}")

        except Exception as e:
            logger.error(f"Error storing Saturday refinement results: {e}", exc_info=True)


# ============= Singleton Factory =============
_service_instance: Optional[SaturdayRefinementService] = None


def get_saturday_refinement_service() -> SaturdayRefinementService:
    """Get singleton instance of SaturdayRefinementService."""
    global _service_instance
    if _service_instance is None:
        _service_instance = SaturdayRefinementService()
    return _service_instance
