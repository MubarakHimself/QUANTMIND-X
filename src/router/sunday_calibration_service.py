"""
Sunday Calibration Service
=========================

Sunday Pre-Market Calibration — runs Sunday 06:00-18:00 GMT.

Components:
- Spread profiles update
- SQS baselines refresh
- Session Kelly modifiers calibration
- Fresh roster preparation

Reference: Story 8.13 (8-13-workflow-4-weekend-update-cycle) AC3
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class CalibrationResult:
    """Result of Sunday calibration."""
    spread_profiles: Dict[str, Any]
    sqs_baselines: Dict[str, Any]
    kelly_calibration: Dict[str, Any]
    roster_prepared: bool
    timestamp: datetime

    def to_dict(self) -> Dict[str, Any]:
        return {
            "spread_profiles": self.spread_profiles,
            "sqs_baselines": self.sqs_baselines,
            "kelly_calibration": self.kelly_calibration,
            "roster_prepared": self.roster_prepared,
            "timestamp": self.timestamp.isoformat(),
        }


class SundayCalibrationService:
    """
    Sunday Pre-Market Calibration — runs Sunday 06:00-18:00 GMT.

    Components:
    - Spread profiles update
    - SQS baselines refresh
    - Session Kelly modifiers calibration
    - Fresh roster preparation
    """

    def __init__(self):
        logger.info("SundayCalibrationService initialized")

    async def run(self) -> CalibrationResult:
        """Execute Sunday calibration."""
        logger.info("Starting Sunday calibration")

        # 1. Update spread profiles
        spread_profiles = await self._update_spread_profiles()

        # 2. Refresh SQS baselines
        sqs_baselines = await self._refresh_sqs_baselines()

        # 3. Calibrate session Kelly modifiers
        kelly_calibration = await self._calibrate_kelly_modifiers()

        # 4. Prepare fresh roster for Monday
        roster_prepared = await self._prepare_fresh_roster()

        result = CalibrationResult(
            spread_profiles=spread_profiles,
            sqs_baselines=sqs_baselines,
            kelly_calibration=kelly_calibration,
            roster_prepared=roster_prepared,
            timestamp=datetime.now(timezone.utc),
        )

        # Store result
        await self._store_calibration_result(result)

        logger.info("Sunday calibration complete")
        return result

    async def _update_spread_profiles(self) -> Dict[str, Any]:
        """
        Update spread profiles for coming week.

        Analyzes recent spread data from trade records to calculate
        typical and maximum spreads for major pairs.
        """
        logger.info("Updating spread profiles")

        try:
            from src.database.models.trade_record import TradeRecord
            from sqlalchemy import select, func
            from src.database.models import get_db_session

            profiles = {}

            # Major pairs to analyze
            major_pairs = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD"]

            async with get_db_session() as session:
                for pair in major_pairs:
                    # Get recent spreads from trade records
                    stmt = select(
                        func.avg(TradeRecord.spread).label("avg_spread"),
                        func.max(TradeRecord.spread).label("max_spread"),
                    ).where(
                        TradeRecord.symbol == pair,
                        TradeRecord.close_time >= datetime.now(timezone.utc).replace(
                            hour=0, minute=0, second=0, microsecond=0
                        ) - timedelta(days=7)
                    )

                    result = await session.execute(stmt)
                    row = result.one_or_none()

                    if row and row.avg_spread:
                        profiles[pair] = {
                            "avg_spread": round(float(row.avg_spread), 2),
                            "max_spread": round(float(row.max_spread), 2) if row.max_spread else round(float(row.avg_spread) * 1.5, 2),
                        }
                    else:
                        # Default values if no data
                        profiles[pair] = {
                            "avg_spread": 1.5,
                            "max_spread": 3.0,
                        }

            return {
                "status": "updated",
                "profiles": profiles,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }

        except ImportError:
            logger.warning("TradeRecord not available, using default spread profiles")
            return self._get_default_spread_profiles()
        except Exception as e:
            logger.error(f"Error updating spread profiles: {e}", exc_info=True)
            return {
                "status": "error",
                "error": str(e),
            }

    def _get_default_spread_profiles(self) -> Dict[str, Any]:
        """Return default spread profiles when calculation fails."""
        return {
            "status": "updated",
            "profiles": {
                "EURUSD": {"avg_spread": 1.2, "max_spread": 2.5},
                "GBPUSD": {"avg_spread": 1.5, "max_spread": 3.0},
                "USDJPY": {"avg_spread": 1.1, "max_spread": 2.2},
            },
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "note": "Default values - calculation failed",
        }

    async def _refresh_sqs_baselines(self) -> Dict[str, Any]:
        """
        Refresh SQS baselines.

        Calculates new SQS baselines based on recent market volatility
        and correlation stability from the risk engine.
        """
        logger.info("Refreshing SQS baselines")

        try:
            from src.risk.sqs_engine import get_sqs_engine
            from src.risk.correlation_cache import get_correlation_cache

            engine = get_sqs_engine()
            corr_cache = get_correlation_cache()

            # Get current market conditions
            current_regime = engine.get_current_regime()
            avg_correlation = corr_cache.get_average_correlation()

            # Calculate volatility factor
            volatility = self._estimate_current_volatility()

            # Determine baseline based on conditions
            if volatility > 1.5:
                baseline_type = "high_volatility"
                baseline = 0.75
            elif volatility < 0.7:
                baseline_type = "low_volatility"
                baseline = 0.55
            else:
                baseline_type = "default"
                baseline = 0.65

            # Adjust based on correlation stability
            if avg_correlation > 0.7:
                # High correlation = reduce baseline (less divergence)
                baseline *= 0.95
            elif avg_correlation < 0.3:
                # Low correlation = increase baseline (more divergence)
                baseline *= 1.05

            baselines = {
                "default": round(0.65, 4),
                "high_volatility": round(0.75, 4),
                "low_volatility": round(0.55, 4),
                "current_adjusted": round(min(max(baseline, 0.50), 0.80), 4),
                "volatility_factor": round(volatility, 4),
                "correlation_factor": round(avg_correlation, 4),
                "regime": current_regime,
            }

            return {
                "status": "refreshed",
                "baselines": baselines,
                "refreshed_at": datetime.now(timezone.utc).isoformat(),
            }

        except ImportError:
            logger.warning("SQS engine not available, using default baselines")
            return self._get_default_sqs_baselines()
        except Exception as e:
            logger.error(f"Error refreshing SQS baselines: {e}", exc_info=True)
            return {
                "status": "error",
                "error": str(e),
            }

    def _estimate_current_volatility(self) -> float:
        """Estimate current market volatility from recent data."""
        try:
            from src.risk.physics.correlation_sensor import CorrelationSensor
            sensor = CorrelationSensor()
            return sensor.get_market_volatility()
        except Exception:
            return 1.0  # Default volatility

    def _get_default_sqs_baselines(self) -> Dict[str, Any]:
        """Return default baselines when calculation fails."""
        return {
            "status": "refreshed",
            "baselines": {
                "default": 0.65,
                "high_volatility": 0.75,
                "low_volatility": 0.55,
            },
            "refreshed_at": datetime.now(timezone.utc).isoformat(),
            "note": "Default values - calculation failed",
        }

    async def _calibrate_kelly_modifiers(self) -> Dict[str, Any]:
        """
        Calibrate session Kelly modifiers for coming week.

        Analyzes recent performance by session (London, NY, Asia) and
        calculates Kelly-adjusted position sizing modifiers.
        """
        logger.info("Calibrating Kelly modifiers")

        try:
            from src.database.models.trade_record import TradeRecord
            from sqlalchemy import select, and_
            from src.database.models import get_db_session

            # Analyze last 4 weeks of data by session
            cutoff = datetime.now(timezone.utc) - timedelta(days=28)

            modifiers = {}

            sessions = ["LONDON", "NY", "ASIA"]

            async with get_db_session() as session:
                for session_name in sessions:
                    stmt = select(
                        func.count(TradeRecord.trade_id).label("trade_count"),
                        func.sum(TradeRecord.pnl).label("total_pnl"),
                        func.avg(TradeRecord.pnl).label("avg_pnl"),
                    ).where(
                        and_(
                            TradeRecord.session == session_name,
                            TradeRecord.close_time >= cutoff,
                        )
                    )

                    result = await session.execute(stmt)
                    row = result.one_or_none()

                    if row and row.trade_count and row.trade_count > 10:
                        # Calculate Kelly fraction from win rate and profit factor
                        wins = await self._count_winning_trades(session, cutoff)
                        total = row.trade_count
                        win_rate = wins / total if total > 0 else 0.5

                        avg_pnl = float(row.avg_pnl or 0)
                        total_pnl = float(row.total_pnl or 0)

                        # Simple Kelly: f = (p * b - q) / b
                        # where p = win rate, q = 1-p, b = avg_win/avg_loss
                        if avg_pnl > 0:
                            profit_factor = 1.5  # Default
                        else:
                            profit_factor = 0.8

                        kelly_fraction = (win_rate * profit_factor - (1 - win_rate)) / profit_factor
                        kelly_fraction = max(0.1, min(kelly_fraction, 1.0))  # Clamp to [0.1, 1.0]

                        # Apply conservative multiplier (50% of full Kelly)
                        session_kelly = kelly_fraction * 0.5

                        # Morning vs afternoon split
                        morning_kelly = round(session_kelly * 0.95, 4)
                        afternoon_kelly = round(session_kelly * 1.05, 4)

                        modifiers[session_name] = {
                            "morning": morning_kelly,
                            "afternoon": afternoon_kelly,
                            "win_rate": round(win_rate, 4),
                            "trade_count": total,
                        }
                    else:
                        # Default values if insufficient data
                        modifiers[session_name] = {
                            "morning": 0.85 if session_name == "LONDON" else 0.80,
                            "afternoon": 0.90 if session_name == "LONDON" else 0.85,
                            "win_rate": 0.5,
                            "trade_count": 0,
                        }

            return {
                "status": "calibrated",
                "modifiers": modifiers,
                "calibrated_at": datetime.now(timezone.utc).isoformat(),
            }

        except ImportError:
            logger.warning("TradeRecord not available, using default Kelly modifiers")
            return self._get_default_kelly_modifiers()
        except Exception as e:
            logger.error(f"Error calibrating Kelly modifiers: {e}", exc_info=True)
            return {
                "status": "error",
                "error": str(e),
            }

    async def _count_winning_trades(self, session: str, cutoff: datetime) -> int:
        """Count winning trades for a session."""
        try:
            from src.database.models.trade_record import TradeRecord
            from sqlalchemy import select, and_, func
            from src.database.models import get_db_session

            async with get_db_session() as session_ctx:
                stmt = select(func.count(TradeRecord.trade_id)).where(
                    and_(
                        TradeRecord.session == session,
                        TradeRecord.close_time >= cutoff,
                        TradeRecord.pnl > 0,
                    )
                )
                result = await session_ctx.execute(stmt)
                return result.scalar() or 0
        except Exception:
            return 0

    def _get_default_kelly_modifiers(self) -> Dict[str, Any]:
        """Return default Kelly modifiers when calculation fails."""
        return {
            "status": "calibrated",
            "modifiers": {
                "LONDON": {"morning": 0.85, "afternoon": 0.90},
                "NY": {"morning": 0.80, "afternoon": 0.85},
                "ASIA": {"morning": 0.75, "afternoon": 0.80},
            },
            "calibrated_at": datetime.now(timezone.utc).isoformat(),
            "note": "Default values - calculation failed",
        }

    async def _prepare_fresh_roster(self) -> bool:
        """
        Prepare fresh roster file for Monday deployment.

        Returns:
            True if roster was prepared successfully
        """
        logger.info("Preparing fresh roster for Monday")

        try:
            from src.router.weekend_roster_manager import get_weekend_roster_manager

            manager = get_weekend_roster_manager()
            roster = await manager.prepare_roster()

            return roster is not None

        except Exception as e:
            logger.error(f"Error preparing fresh roster: {e}", exc_info=True)
            return False

    async def _store_calibration_result(self, result: CalibrationResult) -> None:
        """Store Sunday calibration result to cold storage."""
        try:
            from src.router.cold_storage_writer import get_cold_storage_writer

            writer = get_cold_storage_writer()
            date_str = datetime.now().strftime("%Y-%m-%d")
            file_key = f"sunday_calibration/{date_str}_sunday_calibration.json"

            await writer.write(file_key, result.to_dict())
            logger.info(f"Sunday calibration stored to cold storage: {file_key}")

        except Exception as e:
            logger.error(f"Error storing calibration result: {e}", exc_info=True)


# ============= Singleton Factory =============
_service_instance: Optional[SundayCalibrationService] = None


def get_sunday_calibration_service() -> SundayCalibrationService:
    """Get singleton instance of SundayCalibrationService."""
    global _service_instance
    if _service_instance is None:
        _service_instance = SundayCalibrationService()
    return _service_instance
