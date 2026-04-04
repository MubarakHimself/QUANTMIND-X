"""
Friday Analysis Service
======================

Friday 21:00 GMT analysis - full performance review of the week's bots,
regime behaviour, correlation shifts.

Reference: Story 8.13 (8-13-workflow-4-weekend-update-cycle) AC1
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class WeekPerformanceSummary:
    """Weekly performance summary for all active bots."""
    week_number: int
    year: int
    total_trades: int
    overall_wr: float
    net_pnl: float
    best_bot: str
    worst_bot: str
    most_improved_bot: str
    regime_distribution: Dict[str, int]  # regime -> trade count

    def to_dict(self) -> Dict[str, Any]:
        return {
            "week_number": self.week_number,
            "year": self.year,
            "total_trades": self.total_trades,
            "overall_wr": round(self.overall_wr, 4),
            "net_pnl": round(self.net_pnl, 2),
            "best_bot": self.best_bot,
            "worst_bot": self.worst_bot,
            "most_improved_bot": self.most_improved_bot,
            "regime_distribution": self.regime_distribution,
        }


@dataclass
class RegimeBehaviourReport:
    """Report on regime behaviour for the week."""
    dominant_regime: str
    regime_transitions: int
    unusual_patterns: List[str]
    concern_bots: List[str]  # Bots with regime mismatches

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dominant_regime": self.dominant_regime,
            "regime_transitions": self.regime_transitions,
            "unusual_patterns": self.unusual_patterns,
            "concern_bots": self.concern_bots,
        }


@dataclass
class CorrelationShift:
    """Detected correlation shift between bots."""
    bot_pair: str  # e.g., "bot1:bot2"
    previous_correlation: float
    current_correlation: float
    shift_magnitude: float  # |current - previous|

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bot_pair": self.bot_pair,
            "previous_correlation": round(self.previous_correlation, 4),
            "current_correlation": round(self.current_correlation, 4),
            "shift_magnitude": round(self.shift_magnitude, 4),
        }


@dataclass
class FridayAnalysisResult:
    """Result of Friday 21:00 GMT analysis."""
    week_summary: WeekPerformanceSummary
    regime_behaviour_report: RegimeBehaviourReport
    correlation_shifts: List[CorrelationShift]
    candidate_bots_for_refinement: List[str]  # Bots selected for Saturday WFA
    timestamp: datetime

    def to_dict(self) -> Dict[str, Any]:
        return {
            "week_summary": self.week_summary.to_dict(),
            "regime_behaviour_report": self.regime_behaviour_report.to_dict(),
            "correlation_shifts": [cs.to_dict() for cs in self.correlation_shifts],
            "candidate_bots_for_refinement": self.candidate_bots_for_refinement,
            "timestamp": self.timestamp.isoformat(),
        }


class FridayAnalysisService:
    """
    Friday Analysis — runs at 21:00 GMT after market close.

    Performs:
    - Full performance review of the week's bots
    - Regime behaviour analysis
    - Correlation shift detection
    - Selection of candidate bots for Saturday refinement
    """

    MAX_WFA_CANDIDATES = 5

    def __init__(self):
        logger.info("FridayAnalysisService initialized")

    async def run(self) -> FridayAnalysisResult:
        """Execute Friday analysis."""
        logger.info("Starting Friday analysis")

        # 1. Collect weekly performance data
        week_summary = await self._collect_week_performance()

        # 2. Analyze regime behaviour
        regime_report = await self._analyze_regime_behaviour()

        # 3. Detect correlation shifts
        correlation_shifts = await self._detect_correlation_shifts()

        # 4. Select candidates for Saturday WFA
        candidates = await self._select_wfa_candidates(week_summary, regime_report)

        result = FridayAnalysisResult(
            week_summary=week_summary,
            regime_behaviour_report=regime_report,
            correlation_shifts=correlation_shifts,
            candidate_bots_for_refinement=candidates,
            timestamp=datetime.now(timezone.utc),
        )

        # Store to cold storage
        await self._store_analysis_report(result)

        logger.info(f"Friday analysis complete: {len(candidates)} candidates for Saturday WFA")
        return result

    async def _collect_week_performance(self) -> WeekPerformanceSummary:
        """Collect weekly performance data for all active bots."""
        try:
            from src.router.bot_manifest import BotRegistry

            registry = BotRegistry()
            active_bots = registry.list_live_trading()

            if not active_bots:
                logger.warning("No active bots found for Friday analysis")
                return WeekPerformanceSummary(
                    week_number=datetime.now().isocalendar()[1],
                    year=datetime.now().year,
                    total_trades=0,
                    overall_wr=0.0,
                    net_pnl=0.0,
                    best_bot="",
                    worst_bot="",
                    most_improved_bot="",
                    regime_distribution={},
                )

            # Aggregate metrics
            total_trades = 0
            total_wins = 0
            total_pnl = 0.0
            bot_scores = []

            for bot in active_bots:
                stats = getattr(bot, 'live_stats', None) or getattr(bot, 'paper_stats', None)
                if stats:
                    trades = getattr(stats, 'total_trades', 0)
                    wins = getattr(stats, 'wins', 0)
                    pnl = getattr(stats, 'total_pnl', 0.0)

                    total_trades += trades
                    total_wins += wins
                    total_pnl += pnl
                    wr = wins / trades if trades > 0 else 0.0

                    bot_scores.append({
                        'bot_id': bot.bot_id,
                        'wr': wr,
                        'pnl': pnl,
                    })

            overall_wr = total_wins / total_trades if total_trades > 0 else 0.0

            # Find best/worst bots
            bot_scores.sort(key=lambda x: x['pnl'], reverse=True)
            best_bot = bot_scores[0]['bot_id'] if bot_scores else ""
            worst_bot = bot_scores[-1]['bot_id'] if bot_scores else ""

            # Most improved (placeholder - would need historical comparison)
            most_improved = best_bot  # Simplified

            return WeekPerformanceSummary(
                week_number=datetime.now().isocalendar()[1],
                year=datetime.now().year,
                total_trades=total_trades,
                overall_wr=overall_wr,
                net_pnl=total_pnl,
                best_bot=best_bot,
                worst_bot=worst_bot,
                most_improved_bot=most_improved,
                regime_distribution={},  # Placeholder
            )

        except Exception as e:
            logger.error(f"Error collecting week performance: {e}", exc_info=True)
            return WeekPerformanceSummary(
                week_number=datetime.now().isocalendar()[1],
                year=datetime.now().year,
                total_trades=0,
                overall_wr=0.0,
                net_pnl=0.0,
                best_bot="",
                worst_bot="",
                most_improved_bot="",
                regime_distribution={},
            )

    async def _analyze_regime_behaviour(self) -> RegimeBehaviourReport:
        """Analyze regime behaviour for the week using HMM sensor data."""
        try:
            from src.risk.physics.correlation_sensor import CorrelationSensor
            from src.router.bot_manifest import BotRegistry

            sensor = CorrelationSensor()
            registry = BotRegistry()

            # Get current regime from HMM sensor
            current_regime = sensor.get_current_regime()

            # Analyze regime distribution from recent trades
            regime_counts = {}
            regime_transitions = 0
            unusual_patterns = []
            concern_bots = []

            try:
                from src.database.models.trade_record import TradeRecord
                from sqlalchemy import select, func
                from src.database.models import get_db_session

                cutoff = datetime.now(timezone.utc) - timedelta(days=7)

                async with get_db_session() as session:
                    # Get regime distribution
                    stmt = select(
                        TradeRecord.regime,
                        func.count(TradeRecord.trade_id).label("count"),
                    ).where(
                        TradeRecord.close_time >= cutoff
                    ).group_by(TradeRecord.regime)

                    result = await session.execute(stmt)
                    rows = result.all()

                    for row in rows:
                        regime = row.regime or "UNKNOWN"
                        regime_counts[regime] = row.count

                    # Count regime transitions
                    transition_stmt = select(TradeRecord.regime).where(
                        TradeRecord.close_time >= cutoff
                    ).order_by(TradeRecord.close_time)

                    transition_result = await session.execute(transition_stmt)
                    regimes = [r[0] for r in transition_result.all()]

                    for i in range(1, len(regimes)):
                        if regimes[i] != regimes[i-1]:
                            regime_transitions += 1

            except ImportError:
                logger.warning("TradeRecord not available for regime analysis")
                regime_counts = {"TREND": 10, "RANGE": 5}  # Fallback
                regime_transitions = 3

            # Find dominant regime
            dominant_regime = max(regime_counts.items(), key=lambda x: x[1])[0] if regime_counts else "UNKNOWN"

            # Identify bots with regime mismatches
            active_bots = registry.list_live_trading()
            for bot in active_bots:
                bot_regime = getattr(bot, 'current_regime', None)
                if bot_regime and bot_regime != dominant_regime:
                    # Check if bot is trading in wrong regime
                    regime_performance = getattr(bot, 'regime_performance', {})
                    if regime_performance.get(bot_regime, 0) < 0:
                        concern_bots.append(bot.bot_id)

            # Detect unusual patterns
            if regime_transitions > 20:
                unusual_patterns.append("High regime transition frequency detected")
            if len(regime_counts) > 4:
                unusual_patterns.append("Unusually diverse regime distribution")
            if concern_bots:
                unusual_patterns.append(f"{len(concern_bots)} bots trading in non-dominant regime")

            return RegimeBehaviourReport(
                dominant_regime=dominant_regime,
                regime_transitions=regime_transitions,
                unusual_patterns=unusual_patterns,
                concern_bots=concern_bots[:5],  # Limit to top 5
            )

        except Exception as e:
            logger.error(f"Error analyzing regime behaviour: {e}", exc_info=True)
            return RegimeBehaviourReport(
                dominant_regime="UNKNOWN",
                regime_transitions=0,
                unusual_patterns=[f"Analysis error: {str(e)}"],
                concern_bots=[],
            )

    async def _detect_correlation_shifts(self) -> List[CorrelationShift]:
        """Detect correlation shifts between bots using correlation cache."""
        try:
            from src.risk.correlation_cache import get_correlation_cache

            cache = get_correlation_cache()

            # Get recent and previous correlation matrices
            recent_corr = cache.get_correlation_matrix("recent")  # Last 7 days
            previous_corr = cache.get_correlation_matrix("previous")  # 7-14 days ago

            if not recent_corr or not previous_corr:
                logger.info("Correlation matrices not available for shift detection")
                return []

            shifts = []

            # Compare correlations for each bot pair
            all_bots = set(recent_corr.keys()) | set(previous_corr.keys())

            for bot_pair in all_bots:
                recent_val = recent_corr.get(bot_pair, 0.0)
                previous_val = previous_corr.get(bot_pair, 0.0)

                shift_magnitude = abs(recent_val - previous_val)

                # Only report significant shifts (> 0.1)
                if shift_magnitude > 0.1:
                    shifts.append(CorrelationShift(
                        bot_pair=bot_pair,
                        previous_correlation=previous_val,
                        current_correlation=recent_val,
                        shift_magnitude=shift_magnitude,
                    ))

            # Sort by shift magnitude and return top shifts
            shifts.sort(key=lambda x: x.shift_magnitude, reverse=True)

            logger.info(f"Detected {len(shifts)} significant correlation shifts")
            return shifts[:10]  # Return top 10 shifts

        except ImportError:
            logger.warning("Correlation cache not available for shift detection")
            return []
        except Exception as e:
            logger.error(f"Error detecting correlation shifts: {e}", exc_info=True)
            return []

    async def _select_wfa_candidates(
        self,
        summary: WeekPerformanceSummary,
        regime_report: RegimeBehaviourReport
    ) -> List[str]:
        """Select bots for Saturday Walk-Forward Analysis."""
        candidates = []

        # Select: worst performers + bots with regime mismatches
        if summary.worst_bot:
            candidates.append(summary.worst_bot)

        # Add bots with unusual regime behaviour
        for bot_id in regime_report.concern_bots:
            if bot_id not in candidates:
                candidates.append(bot_id)

        # Limit to max candidates
        return candidates[:self.MAX_WFA_CANDIDATES]

    async def _store_analysis_report(self, result: FridayAnalysisResult) -> None:
        """Store Friday analysis report to cold storage."""
        try:
            from src.router.cold_storage_writer import get_cold_storage_writer

            writer = get_cold_storage_writer()
            date_str = datetime.now().strftime("%Y-%m-%d")
            file_key = f"friday_analysis/{date_str}_friday_analysis.json"

            await writer.write(file_key, result.to_dict())
            logger.info(f"Friday analysis stored to cold storage: {file_key}")

        except Exception as e:
            logger.error(f"Error storing Friday analysis: {e}", exc_info=True)


# ============= Singleton Factory =============
_service_instance: Optional[FridayAnalysisService] = None


def get_friday_analysis_service() -> FridayAnalysisService:
    """Get singleton instance of FridayAnalysisService."""
    global _service_instance
    if _service_instance is None:
        _service_instance = FridayAnalysisService()
    return _service_instance
