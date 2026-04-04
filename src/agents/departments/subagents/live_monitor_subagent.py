"""
LiveMonitorSubAgent — Monitors live trading vs backtest/paper reports.

Continuously compares live performance against Stage 2 (backtest) and Stage 3 (paper)
reports to detect anomalies. Flags bots that deviate significantly from expected
performance and triggers Workflow 2 improvement loop when needed.

FAIL-W03: If anomaly persists for 50+ trades, mark as PAPER_FAIL and return to
Workflow 2 improvement loop with diagnosis.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List

from src.router.bot_manifest import BotRegistry, TradingMode
from src.router.promotion_manager import PromotionManager, PerformanceTracker
from src.router.dpr_scoring_engine import get_dpr_scoring_engine
from src.agents.departments.department_mail import (
    get_mail_service,
    MessageType,
    Priority,
)
from src.database.models.performance import PaperTradingPerformance, StrategyPerformance
from src.config import get_database_url
from sqlalchemy import select
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession

logger = logging.getLogger(__name__)

# Anomaly detection threshold: Live WR >5% below paper WR
ANOMALY_WR_THRESHOLD = 0.05
# Fail threshold: anomaly persists for 50+ trades
PAPER_FAIL_TRADE_THRESHOLD = 50


@dataclass
class LiveMetrics:
    """Live trading metrics for a bot."""
    win_rate: float
    rr_ratio: float
    expected_value: float
    total_trades: int
    losing_trades: int
    winning_trades: int
    total_pnl: float


@dataclass
class ComparisonMetrics:
    """Paper or backtest metrics for comparison."""
    win_rate: float
    rr_ratio: float
    expected_value: float


@dataclass
class DeltaReport:
    """Delta report comparing live vs expected metrics."""
    bot_id: str
    date: str
    live_metrics: LiveMetrics
    paper_metrics: Optional[ComparisonMetrics]
    backtest_metrics: Optional[ComparisonMetrics]
    delta: Dict[str, float]
    status: str  # "ANOMALY" or "NORMAL"
    anomaly_threshold_exceeded: bool
    anomaly_trade_count: int  # Number of trades since anomaly started
    paper_fail_triggered: bool


class LiveMonitorSubAgent:
    """
    Continuously monitors live trading vs backtest/paper reports.

    Responsibilities:
    - Read TradeLogger for live trade history
    - Compare rolling WR/R:R/EV against Stage 2 (backtest) and Stage 3 (paper) reports
    - Flag anomalies (>5% WR drop from paper)
    - Write daily delta reports
    - Send recommendations via mail system
    """

    def __init__(self):
        self._bot_registry: Optional[BotRegistry] = None
        self._promotion_manager: Optional[PromotionManager] = None
        self._performance_tracker: Optional[PerformanceTracker] = None
        self._dpr_engine = get_dpr_scoring_engine()
        # Track anomaly state per bot (anomaly_trade_count)
        self._anomaly_state: Dict[str, int] = {}

    @property
    def bot_registry(self) -> BotRegistry:
        """Lazy load BotRegistry."""
        if self._bot_registry is None:
            self._bot_registry = BotRegistry()
        return self._bot_registry

    @property
    def promotion_manager(self) -> PromotionManager:
        """Lazy load PromotionManager."""
        if self._promotion_manager is None:
            self._promotion_manager = PromotionManager()
        return self._promotion_manager

    @property
    def performance_tracker(self) -> PerformanceTracker:
        """Lazy load PerformanceTracker."""
        if self._performance_tracker is None:
            self._performance_tracker = PerformanceTracker()
        return self._performance_tracker

    async def _get_live_metrics(self, bot_id: str, days: int = 7) -> Optional[LiveMetrics]:
        """
        Get rolling live metrics for a bot from the last N days.

        Args:
            bot_id: Bot identifier
            days: Number of days to look back

        Returns:
            LiveMetrics or None if insufficient data
        """
        try:
            trades = self.performance_tracker.get_trade_history(bot_id, mode="live")
            if not trades:
                return None

            # Filter to last N days
            cutoff = datetime.now(timezone.utc) - timedelta(days=days)
            recent_trades = [
                t for t in trades
                if datetime.fromisoformat(t.get("timestamp", "2000-01-01")) > cutoff
            ]

            if len(recent_trades) < 5:
                return None

            total_trades = len(recent_trades)
            winning_trades = sum(1 for t in recent_trades if (t.get("pnl", 0)) > 0)
            losing_trades = sum(1 for t in recent_trades if (t.get("pnl", 0)) < 0)
            win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
            total_pnl = sum(t.get("pnl", 0) for t in recent_trades)

            # Calculate R:R (simplified - would need actual RR data from trades)
            # Using profit factor as proxy for RR estimation
            gross_profit = sum(t.get("pnl", 0) for t in recent_trades if t.get("pnl", 0) > 0)
            gross_loss = abs(sum(t.get("pnl", 0) for t in recent_trades if t.get("pnl", 0) < 0))
            rr_ratio = gross_profit / gross_loss if gross_loss > 0 else 0.0

            # Calculate EV per trade
            ev_per_trade = total_pnl / total_trades if total_trades > 0 else 0.0

            return LiveMetrics(
                win_rate=win_rate,
                rr_ratio=rr_ratio,
                expected_value=ev_per_trade,
                total_trades=total_trades,
                losing_trades=losing_trades,
                winning_trades=winning_trades,
                total_pnl=total_pnl,
            )
        except Exception as e:
            logger.warning(f"Failed to get live metrics for {bot_id}: {e}")
            return None

    async def _get_paper_metrics(self, bot_id: str) -> Optional[ComparisonMetrics]:
        """
        Get paper trading metrics for a bot.

        Args:
            bot_id: Bot identifier

        Returns:
            ComparisonMetrics or None if not found
        """
        try:
            db_url = get_database_url()
            async_db_url = db_url.replace("sqlite:///", "sqlite+aiosqlite:///")
            engine = create_async_engine(async_db_url, echo=False)

            async with AsyncSession(engine) as session:
                stmt = select(PaperTradingPerformance).where(
                    PaperTradingPerformance.agent_id == bot_id
                ).order_by(PaperTradingPerformance.timestamp.desc()).limit(1)

                result = await session.execute(stmt)
                paper = result.scalar_one_or_none()

                if paper:
                    # Get rr_ratio from extra_data if available
                    rr_ratio = 0.0
                    extra = paper.extra_data or {}
                    if isinstance(extra, dict):
                        rr_ratio = extra.get("rr_ratio", 0.0) or extra.get("profit_factor", 0.0)

                    return ComparisonMetrics(
                        win_rate=paper.win_rate,
                        rr_ratio=rr_ratio,
                        expected_value=paper.average_pnl,
                    )
        except Exception as e:
            logger.warning(f"Failed to get paper metrics for {bot_id}: {e}")

        return None

    async def _get_backtest_metrics(self, bot_id: str) -> Optional[ComparisonMetrics]:
        """
        Get backtest metrics for a bot.

        Args:
            bot_id: Bot identifier

        Returns:
            ComparisonMetrics or None if not found
        """
        try:
            db_url = get_database_url()
            async_db_url = db_url.replace("sqlite:///", "sqlite+aiosqlite:///")
            engine = create_async_engine(async_db_url, echo=False)

            async with AsyncSession(engine) as session:
                stmt = select(StrategyPerformance).where(
                    StrategyPerformance.strategy_name == bot_id
                ).order_by(StrategyPerformance.created_at.desc()).limit(1)

                result = await session.execute(stmt)
                backtest = result.scalar_one_or_none()

                if backtest:
                    return ComparisonMetrics(
                        win_rate=backtest.win_rate or 0.0,
                        rr_ratio=backtest.profit_factor or 0.0,
                        expected_value=backtest.kelly_score or 0.0,
                    )
        except Exception as e:
            logger.warning(f"Failed to get backtest metrics for {bot_id}: {e}")

        return None

    def _detect_anomaly(
        self,
        live_metrics: LiveMetrics,
        paper_metrics: Optional[ComparisonMetrics],
    ) -> tuple[bool, float, Dict[str, float]]:
        """
        Detect if live metrics indicate an anomaly vs paper.

        Anomaly threshold: Live WR >5% below paper WR

        Args:
            live_metrics: Live trading metrics
            paper_metrics: Paper trading metrics

        Returns:
            Tuple of (is_anomaly, wr_drop, delta_dict)
        """
        if paper_metrics is None:
            return False, 0.0, {}

        wr_drop = paper_metrics.win_rate - live_metrics.win_rate
        rr_drop = paper_metrics.rr_ratio - live_metrics.rr_ratio
        ev_drop = paper_metrics.expected_value - live_metrics.expected_value

        delta = {
            "wr_drop": wr_drop,
            "rr_drop": rr_drop,
            "ev_drop": ev_drop,
        }

        is_anomaly = wr_drop > ANOMALY_WR_THRESHOLD

        return is_anomaly, wr_drop, delta

    async def monitor_bot(self, bot_id: str) -> Optional[Dict[str, Any]]:
        """
        Monitor a single bot — returns anomaly report or None.

        Args:
            bot_id: Bot identifier

        Returns:
            Delta report dict or None if no anomaly detected
        """
        # Get live metrics
        live_metrics = await self._get_live_metrics(bot_id, days=7)
        if live_metrics is None:
            return None

        # Get paper and backtest metrics for comparison
        paper_metrics = await self._get_paper_metrics(bot_id)
        backtest_metrics = await self._get_backtest_metrics(bot_id)

        # Detect anomaly
        is_anomaly, wr_drop, delta = self._detect_anomaly(live_metrics, paper_metrics)

        # Track anomaly trade count
        current_anomaly_count = self._anomaly_state.get(bot_id, 0)
        if is_anomaly:
            self._anomaly_state[bot_id] = current_anomaly_count + live_metrics.total_trades
        else:
            # Reset anomaly state if back to normal
            if bot_id in self._anomaly_state:
                del self._anomaly_state[bot_id]
            current_anomaly_count = 0

        anomaly_trade_count = self._anomaly_state.get(bot_id, 0)

        # Check for PAPER_FAIL threshold
        paper_fail_triggered = anomaly_trade_count >= PAPER_FAIL_TRADE_THRESHOLD

        # Determine status
        status = "ANOMALY" if is_anomaly else "NORMAL"
        if paper_fail_triggered:
            status = "PAPER_FAIL"

        # Build delta report
        report = DeltaReport(
            bot_id=bot_id,
            date=datetime.now(timezone.utc).date().isoformat(),
            live_metrics=live_metrics,
            paper_metrics=paper_metrics,
            backtest_metrics=backtest_metrics,
            delta=delta,
            status=status,
            anomaly_threshold_exceeded=is_anomaly,
            anomaly_trade_count=anomaly_trade_count,
            paper_fail_triggered=paper_fail_triggered,
        )

        # If anomaly detected, notify DPR scoring engine to reduce score
        if is_anomaly:
            await self._notify_dpr_engine(bot_id, wr_drop)

        # If PAPER_FAIL triggered, send to Workflow 2 improvement loop
        if paper_fail_triggered:
            await self._trigger_workflow_2(report)

        # Convert to dict for return
        return self._delta_report_to_dict(report)

    def _delta_report_to_dict(self, report: DeltaReport) -> Dict[str, Any]:
        """Convert DeltaReport to dictionary format."""
        return {
            "bot_id": report.bot_id,
            "date": report.date,
            "live_metrics": {
                "win_rate": round(report.live_metrics.win_rate, 4),
                "rr_ratio": round(report.live_metrics.rr_ratio, 2),
                "expected_value": round(report.live_metrics.expected_value, 2),
                "total_trades": report.live_metrics.total_trades,
            },
            "paper_metrics": {
                "win_rate": round(report.paper_metrics.win_rate, 4),
                "rr_ratio": round(report.paper_metrics.rr_ratio, 2),
                "expected_value": round(report.paper_metrics.expected_value, 2),
            } if report.paper_metrics else None,
            "backtest_metrics": {
                "win_rate": round(report.backtest_metrics.win_rate, 4),
                "rr_ratio": round(report.backtest_metrics.rr_ratio, 2),
                "expected_value": round(report.backtest_metrics.expected_value, 2),
            } if report.backtest_metrics else None,
            "delta": {k: round(v, 4) for k, v in report.delta.items()},
            "status": report.status,
            "anomaly_threshold_exceeded": report.anomaly_threshold_exceeded,
            "anomaly_trade_count": report.anomaly_trade_count,
            "paper_fail_triggered": report.paper_fail_triggered,
        }

    async def _notify_dpr_engine(self, bot_id: str, wr_drop: float) -> None:
        """
        Notify DPR scoring engine to reduce score for anomalous bot.

        Args:
            bot_id: Bot identifier
            wr_drop: Win rate drop from paper (used to adjust score)
        """
        try:
            # The DPR engine will reduce the score based on the wr_drop
            # Higher drop = bigger score reduction
            logger.info(
                f"LiveMonitor: Notifying DPR engine for {bot_id} "
                f"(WR drop: {wr_drop:.2%})"
            )
            # DPR engine integration point - the engine itself handles score reduction
            # based on the anomaly flag being set
            dpr = get_dpr_scoring_engine()
            # Note: In a full implementation, we'd update the bot's DPR score here
            # For now, we just log the anomaly
        except Exception as e:
            logger.warning(f"Failed to notify DPR engine for {bot_id}: {e}")

    async def _trigger_workflow_2(self, report: DeltaReport) -> None:
        """
        Trigger Workflow 2 (improvement loop) for PAPER_FAIL bots.

        Sends a message to the Development department via the mail system
        with the diagnosis for the improvement workflow.

        Args:
            report: The delta report triggering the failure
        """
        try:
            mail_service = get_mail_service()

            # Build diagnosis message
            diagnosis = self._build_paper_fail_diagnosis(report)

            # Send to Development department
            mail_service.send(
                from_dept="trading",
                to_dept="development",
                type=MessageType.DISPATCH,
                subject=f"PAPER_FAIL: Bot {report.bot_id} needs improvement",
                body=diagnosis,
                priority=Priority.HIGH,
                workflow_id="improvement_loop_flow",
            )

            logger.info(
                f"LiveMonitor: PAPER_FAIL triggered for {report.bot_id} "
                f"(anomaly persisted for {report.anomaly_trade_count} trades)"
            )

        except Exception as e:
            logger.warning(f"Failed to trigger Workflow 2 for {report.bot_id}: {e}")

    def _build_paper_fail_diagnosis(self, report: DeltaReport) -> str:
        """Build diagnosis message for PAPER_FAIL."""
        live_wr = report.live_metrics.win_rate
        paper_wr = report.paper_metrics.win_rate if report.paper_metrics else 0.0
        wr_drop = paper_wr - live_wr

        diagnosis = f"""
# PAPER_FAIL Diagnosis for {report.bot_id}

## Failure Summary
- Status: PAPER_FAIL (anomaly persisted for {report.anomaly_trade_count}+ trades)
- Live Win Rate: {live_wr:.2%}
- Paper Win Rate: {paper_wr:.2%}
- WR Drop: {wr_drop:.2%} (threshold: {ANOMALY_WR_THRESHOLD:.2%})

## Live Metrics
- Total Trades: {report.live_metrics.total_trades}
- Win Rate: {report.live_metrics.win_rate:.2%}
- R:R Ratio: {report.live_metrics.rr_ratio:.2f}
- EV/Trade: ${report.live_metrics.expected_value:.2f}
- Total PnL: ${report.live_metrics.total_pnl:.2f}

## Expected Metrics (Paper)
- Win Rate: {paper_wr:.2%}
- R:R Ratio: {report.paper_metrics.rr_ratio:.2f if report.paper_metrics else 'N/A'}
- EV/Trade: ${report.paper_metrics.expected_value:.2f if report.paper_metrics else 'N/A'}

## Recommended Action
Return to Workflow 2 (improvement_loop_flow) for parameter review and EA adjustment.
Bot has underperformed paper trading by more than 5% WR for 50+ trades.
"""
        return diagnosis.strip()

    async def monitor_all_bots(self) -> List[Dict[str, Any]]:
        """
        Scan all active bots, return list of anomaly reports.

        Returns:
            List of anomaly report dicts for bots with detected anomalies
        """
        # Get all live trading bots
        live_bots = self.bot_registry.list_live_trading()

        anomalies = []
        for bot in live_bots:
            try:
                report = await self.monitor_bot(bot.bot_id)
                if report and report.get("anomaly_threshold_exceeded"):
                    anomalies.append(report)
            except Exception as e:
                logger.warning(f"Failed to monitor bot {bot.bot_id}: {e}")

        logger.info(
            f"LiveMonitor: Scanned {len(live_bots)} live bots, "
            f"found {len(anomalies)} anomalies"
        )

        return anomalies

    async def generate_daily_delta_report(self, bot_id: str) -> Dict[str, Any]:
        """
        Generate daily delta report comparing live vs expected.

        Args:
            bot_id: Bot identifier

        Returns:
            Delta report dictionary
        """
        report = await self.monitor_bot(bot_id)

        if report is None:
            # No live data yet - return placeholder
            return {
                "bot_id": bot_id,
                "date": datetime.now(timezone.utc).date().isoformat(),
                "live_metrics": None,
                "paper_metrics": None,
                "backtest_metrics": None,
                "delta": {},
                "status": "NO_DATA",
                "anomaly_threshold_exceeded": False,
                "anomaly_trade_count": 0,
                "paper_fail_triggered": False,
            }

        return report
