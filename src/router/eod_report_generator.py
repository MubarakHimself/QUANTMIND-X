"""
EOD Report Generator
====================

Generates End-of-Day Report for the Dead Zone workflow.
Captures all trading day outcomes, regime states, anomaly events, and P&L attribution.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class TradingDayOutcome:
    """Outcome of a single trading day."""
    date: str
    total_trades: int
    winning_trades: int
    losing_trades: int
    net_pnl: float
    win_rate: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float


@dataclass
class RegimeState:
    """HMM regime state for the trading day."""
    regime: str  # TREND, RANGE, BREAKOUT, CHAOS
    confidence: float
    dominant_features: List[str]


@dataclass
class AnomalyEvent:
    """Anomaly event detected during the trading day."""
    timestamp: str
    event_type: str  # VOLATILITY_SPIKE, REGIME_CHANGE, CORRELATION_BREAKDOWN, etc.
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    description: str
    affected_bots: List[str]


@dataclass
class PnLAttribution:
    """P&L attribution by strategy/bot."""
    bot_id: str
    strategy_name: str
    pnl: float
    trades: int
    win_rate: float
    contribution_percent: float


@dataclass
class EODReport:
    """End-of-Day Report for the Dead Zone."""
    report_id: str
    generated_at: datetime
    trading_date: str
    day_outcome: TradingDayOutcome
    regime_states: List[RegimeState]
    anomaly_events: List[AnomalyEvent]
    pnl_attribution: List[PnLAttribution]
    total_pnl: float
    total_trades: int
    regime_at_close: str
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "generated_at": self.generated_at.isoformat(),
            "trading_date": self.trading_date,
            "day_outcome": {
                "date": self.day_outcome.date,
                "total_trades": self.day_outcome.total_trades,
                "winning_trades": self.day_outcome.winning_trades,
                "losing_trades": self.day_outcome.losing_trades,
                "net_pnl": self.day_outcome.net_pnl,
                "win_rate": self.day_outcome.win_rate,
                "avg_win": self.day_outcome.avg_win,
                "avg_loss": self.day_outcome.avg_loss,
                "largest_win": self.day_outcome.largest_win,
                "largest_loss": self.day_outcome.largest_loss,
            },
            "regime_states": [
                {
                    "regime": r.regime,
                    "confidence": r.confidence,
                    "dominant_features": r.dominant_features,
                }
                for r in self.regime_states
            ],
            "anomaly_events": [
                {
                    "timestamp": e.timestamp,
                    "event_type": e.event_type,
                    "severity": e.severity,
                    "description": e.description,
                    "affected_bots": e.affected_bots,
                }
                for e in self.anomaly_events
            ],
            "pnl_attribution": [
                {
                    "bot_id": a.bot_id,
                    "strategy_name": a.strategy_name,
                    "pnl": a.pnl,
                    "trades": a.trades,
                    "win_rate": a.win_rate,
                    "contribution_percent": a.contribution_percent,
                }
                for a in self.pnl_attribution
            ],
            "total_pnl": self.total_pnl,
            "total_trades": self.total_trades,
            "regime_at_close": self.regime_at_close,
            "notes": self.notes,
        }


class EODReportGenerator:
    """
    Generates End-of-Day Report for the Dead Zone workflow.

    Captures:
    - Trading day outcomes (trades, P&L, win rates)
    - Regime states (HMM regime at various timeframes)
    - Anomaly events (volatility spikes, regime changes, etc.)
    - P&L attribution by bot/strategy
    """

    def __init__(self):
        logger.info("EODReportGenerator initialized")

    async def generate(self) -> EODReport:
        """
        Generate the EOD Report for the current trading day.

        Returns:
            EODReport with all trading day data
        """
        import uuid

        now = datetime.now(timezone.utc)
        trading_date = now.strftime("%Y-%m-%d")

        logger.info(f"Generating EOD Report for {trading_date}")

        # Gather data from various sources
        day_outcome = await self._get_trading_day_outcome(trading_date)
        regime_states = await self._get_regime_states(trading_date)
        anomaly_events = await self._get_anomaly_events(trading_date)
        pnl_attribution = await self._get_pnl_attribution(trading_date)
        regime_at_close = await self._get_regime_at_close()

        total_pnl = sum(a.pnl for a in pnl_attribution) if pnl_attribution else 0.0
        total_trades = day_outcome.total_trades if day_outcome else 0

        report = EODReport(
            report_id=str(uuid.uuid4()),
            generated_at=now,
            trading_date=trading_date,
            day_outcome=day_outcome,
            regime_states=regime_states,
            anomaly_events=anomaly_events,
            pnl_attribution=pnl_attribution,
            total_pnl=total_pnl,
            total_trades=total_trades,
            regime_at_close=regime_at_close,
        )

        logger.info(
            f"EOD Report generated: {total_trades} trades, "
            f"P&L: {total_pnl:.2f}, regime: {regime_at_close}"
        )

        return report

    async def _get_trading_day_outcome(self, trading_date: str) -> TradingDayOutcome:
        """Get aggregated trading day outcomes from bot manifest/sessions."""
        try:
            from src.router.bot_manifest import BotRegistry

            registry = BotRegistry()

            # Get session outcomes for the trading date
            # Placeholder implementation - would integrate with actual session data
            total_trades = 0
            winning_trades = 0
            losing_trades = 0
            net_pnl = 0.0
            largest_win = 0.0
            largest_loss = 0.0
            total_wins = 0.0
            total_losses = 0.0

            # Aggregate from active bots (live trading bots)
            active_bots = registry.list_live_trading()
            for bot in active_bots:
                # Placeholder - would fetch actual session data from trading session
                # Use ModePerformanceStats from live_stats if available
                stats = getattr(bot, 'live_stats', None)
                if stats:
                    total_trades += stats.total_trades
                    winning_trades += stats.winning_trades
                    losing_trades += stats.losing_trades
                    net_pnl += stats.total_pnl

            # Calculate aggregated metrics
            if total_trades > 0:
                win_rate = winning_trades / total_trades
                avg_win = total_wins / winning_trades if winning_trades > 0 else 0.0
                avg_loss = abs(total_losses) / losing_trades if losing_trades > 0 else 0.0
            else:
                win_rate = 0.0
                avg_win = 0.0
                avg_loss = 0.0

            return TradingDayOutcome(
                date=trading_date,
                total_trades=total_trades,
                winning_trades=winning_trades,
                losing_trades=losing_trades,
                net_pnl=net_pnl,
                win_rate=win_rate,
                avg_win=avg_win,
                avg_loss=avg_loss,
                largest_win=largest_win,
                largest_loss=largest_loss,
            )

        except Exception as e:
            logger.warning(f"Could not get trading day outcome: {e}")
            return TradingDayOutcome(
                date=trading_date,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                net_pnl=0.0,
                win_rate=0.0,
                avg_win=0.0,
                avg_loss=0.0,
                largest_win=0.0,
                largest_loss=0.0,
            )

    async def _get_regime_states(self, trading_date: str) -> List[RegimeState]:
        """Get HMM regime states for the trading day."""
        try:
            from src.router.hmm_deployment import get_deployment_manager

            deployment = get_deployment_manager()
            current_state = deployment.get_current_state()

            if current_state:
                # Map DeploymentMode to regime string
                mode = current_state.mode
                regime_map = {
                    "ISING_ONLY": "TREND",
                    "HMM_ONLY": "RANGE",
                    "HMM_HYBRID_50": "BREAKOUT",
                    "HMM_HYBRID_80": "CHAOS",
                }
                regime = regime_map.get(str(mode), "UNKNOWN")
                return [
                    RegimeState(
                        regime=regime,
                        confidence=0.75,  # Placeholder - no confidence in DeploymentState
                        dominant_features=[],  # Placeholder
                    )
                ]
            return []

        except Exception as e:
            logger.warning(f"Could not get regime states: {e}")
            return []

    async def _get_anomaly_events(self, trading_date: str) -> List[AnomalyEvent]:
        """Get anomaly events detected during the trading day."""
        # Placeholder - would integrate with Sentinel/HMM anomaly detection
        # Returns empty list for now, can be extended with actual anomaly detection
        try:
            # Check multi_timeframe_sentinel for anomalies
            from src.router.multi_timeframe_sentinel import MultiTimeframeSentinel
            sentinel = MultiTimeframeSentinel()
            if hasattr(sentinel, 'get_recent_anomalies'):
                anomalies = sentinel.get_recent_anomalies()
                return anomalies

        except Exception as e:
            logger.warning(f"Could not get anomaly events: {e}")
        return []

    async def _get_pnl_attribution(self, trading_date: str) -> List[PnLAttribution]:
        """Get P&L attribution by bot/strategy."""
        try:
            from src.router.bot_manifest import BotRegistry

            registry = BotRegistry()
            active_bots = registry.list_live_trading()

            attribution = []
            total_pnl = 0.0

            # First pass: calculate total P&L from ModePerformanceStats
            bot_pnls = []
            for bot in active_bots:
                # Get P&L from live_stats if available
                stats = getattr(bot, 'live_stats', None)
                if stats:
                    pnl = stats.total_pnl
                else:
                    pnl = 0.0
                total_pnl += pnl
                strategy_name = bot.strategy_type.value if hasattr(bot.strategy_type, 'value') else str(bot.strategy_type)
                bot_pnls.append((bot.bot_id, strategy_name, pnl, stats))

            # Second pass: calculate attribution percentages
            for bot_id, strategy_name, pnl, stats in bot_pnls:
                contribution = (pnl / total_pnl * 100) if total_pnl != 0 else 0.0
                trades = stats.total_trades if stats else 0
                win_rate = stats.win_rate if stats else 0.0
                attribution.append(PnLAttribution(
                    bot_id=bot_id,
                    strategy_name=strategy_name,
                    pnl=pnl,
                    trades=trades,
                    win_rate=win_rate,
                    contribution_percent=contribution,
                ))

            # Sort by P&L descending
            attribution.sort(key=lambda x: x.pnl, reverse=True)
            return attribution

        except Exception as e:
            logger.warning(f"Could not get P&L attribution: {e}")
            return []

    async def _get_regime_at_close(self) -> str:
        """Get the HMM regime at market close."""
        try:
            from src.router.hmm_deployment import get_deployment_manager

            deployment = get_deployment_manager()
            current_state = deployment.get_current_state()

            if current_state:
                # Map DeploymentMode to regime string
                mode = current_state.mode
                regime_map = {
                    "ISING_ONLY": "TREND",
                    "HMM_ONLY": "RANGE",
                    "HMM_HYBRID_50": "BREAKOUT",
                    "HMM_HYBRID_80": "CHAOS",
                }
                return regime_map.get(str(mode), "UNKNOWN")
            return "UNKNOWN"

        except Exception as e:
            logger.warning(f"Could not get regime at close: {e}")
            return "UNKNOWN"

    async def save_to_cold_storage(self, report: EODReport) -> str:
        """Save EOD Report to cold storage."""
        try:
            from src.router.cold_storage_writer import get_cold_storage_writer

            writer = get_cold_storage_writer()
            file_key = f"eod_reports/{report.trading_date}_eod_report.json"
            await writer.write(file_key, report.to_dict())

            logger.info(f"EOD Report saved to cold storage: {file_key}")
            return file_key

        except Exception as e:
            logger.error(f"Failed to save EOD Report to cold storage: {e}")
            raise


# ============= Singleton Factory =============
_eod_report_generator: Optional[EODReportGenerator] = None


def get_eod_report_generator() -> EODReportGenerator:
    """Get singleton instance of EODReportGenerator."""
    global _eod_report_generator
    if _eod_report_generator is None:
        _eod_report_generator = EODReportGenerator()
    return _eod_report_generator
