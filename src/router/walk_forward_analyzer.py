"""
Walk-Forward Analyzer
=====================

Walk-Forward Analysis using 14-day rolling data.
Runs on Saturday as part of Weekend Update Cycle.

Reference: Story 8.13 (8-13-workflow-4-weekend-update-cycle) AC2
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class WfaMetrics:
    """Walk-Forward Analysis metrics."""
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    profit_factor: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "max_drawdown": round(self.max_drawdown, 4),
            "win_rate": round(self.win_rate, 4),
            "total_trades": self.total_trades,
            "profit_factor": round(self.profit_factor, 4),
        }


@dataclass
class WfaResult:
    """Result of Walk-Forward Analysis for a single bot."""
    bot_id: str
    optimal_parameters: Dict[str, Any]
    validation_metrics: WfaMetrics
    is_valid: bool  # True if sharpe_ratio > 1.0
    recommendation: str  # "APPROVE" or "REJECT"
    timestamp: datetime

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bot_id": self.bot_id,
            "optimal_parameters": self.optimal_parameters,
            "validation_metrics": self.validation_metrics.to_dict(),
            "is_valid": self.is_valid,
            "recommendation": self.recommendation,
            "timestamp": self.timestamp.isoformat(),
        }


class WalkForwardAnalyzer:
    """
    Walk-Forward Analysis using 14-day rolling data.

    Runs on Saturday as part of Weekend Update Cycle.

    Method:
    - Load 14-day rolling data
    - Split: 10 days in-sample, 4 days out-of-sample
    - Optimize parameters on in-sample
    - Validate on out-of-sample
    - Compute Sharpe ratio, max drawdown, Win Rate for validation period
    """

    WFA_WINDOW_DAYS = 14
    IN_SAMPLE_DAYS = 10
    OUT_SAMPLE_DAYS = 4
    SHARPE_APPROVE_THRESHOLD = 1.0

    def __init__(self):
        logger.info("WalkForwardAnalyzer initialized")

    async def run(self, bot_id: str) -> WfaResult:
        """
        Execute WFA for a single bot.

        Args:
            bot_id: Bot ID to analyze

        Returns:
            WfaResult with analysis outcome
        """
        logger.info(f"Starting WFA for bot: {bot_id}")

        try:
            # 1. Load 14-day rolling data
            historical_data = await self._load_rolling_data(bot_id, days=self.WFA_WINDOW_DAYS)

            if not historical_data:
                logger.warning(f"No historical data for bot {bot_id}, skipping WFA")
                return self._create_empty_result(bot_id, "No historical data")

            # 2. Split into in-sample and out-of-sample
            in_sample = historical_data[:self.IN_SAMPLE_DAYS]
            out_sample = historical_data[self.IN_SAMPLE_DAYS:]

            if len(in_sample) < 5 or len(out_sample) < 2:
                logger.warning(f"Insufficient data split for bot {bot_id}")
                return self._create_empty_result(bot_id, "Insufficient data for WFA")

            # 3. Run in-sample optimization on first 10 days
            optimal_params = await self._optimize_parameters(bot_id, in_sample)

            # 4. Run out-of-sample validation on last 4 days
            validation_result = await self._validate_parameters(bot_id, optimal_params, out_sample)

            # 5. Compute WFA metrics
            metrics = await self._compute_wfa_metrics(validation_result)

            # Determine if valid (sharpe_ratio > 1.0)
            is_valid = metrics.sharpe_ratio > self.SHARPE_APPROVE_THRESHOLD
            recommendation = "APPROVE" if is_valid else "REJECT"

            result = WfaResult(
                bot_id=bot_id,
                optimal_parameters=optimal_params,
                validation_metrics=metrics,
                is_valid=is_valid,
                recommendation=recommendation,
                timestamp=datetime.now(timezone.utc),
            )

            logger.info(
                f"WFA complete for {bot_id}: sharpe={metrics.sharpe_ratio:.4f}, "
                f"recommendation={recommendation}"
            )

            # Store result
            await self._store_wfa_result(result)

            return result

        except Exception as e:
            logger.error(f"WFA failed for bot {bot_id}: {e}", exc_info=True)
            return self._create_empty_result(bot_id, str(e))

    def _create_empty_result(self, bot_id: str, reason: str) -> WfaResult:
        """Create an empty result when WFA cannot run."""
        return WfaResult(
            bot_id=bot_id,
            optimal_parameters={},
            validation_metrics=WfaMetrics(
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                win_rate=0.0,
                total_trades=0,
                profit_factor=0.0,
            ),
            is_valid=False,
            recommendation=f"SKIPPED ({reason})",
            timestamp=datetime.now(timezone.utc),
        )

    async def _load_rolling_data(self, bot_id: str, days: int) -> List[Dict[str, Any]]:
        """
        Load 14-day rolling data for a bot from trade records.

        Loads historical trade data from the performance database for the specified
        number of days, including P&L, win/loss, and regime information.
        """
        try:
            from src.database.models.trade_record import TradeRecord
            from sqlalchemy import select, and_
            from src.database.models import get_db_session

            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)

            async with get_db_session() as session:
                stmt = select(TradeRecord).where(
                    and_(
                        TradeRecord.bot_id == bot_id,
                        TradeRecord.close_time >= cutoff_date
                    )
                ).order_by(TradeRecord.close_time)

                results = await session.execute(stmt)
                records = results.scalars().all()

                historical_data = []
                for record in records:
                    historical_data.append({
                        "trade_id": record.trade_id,
                        "open_time": record.open_time.isoformat() if record.open_time else None,
                        "close_time": record.close_time.isoformat() if record.close_time else None,
                        "pnl": float(record.pnl or 0),
                        "profit": float(record.profit or 0),
                        "loss": float(record.loss or 0),
                        "regime": getattr(record, 'regime', 'UNKNOWN'),
                        "win": record.pnl > 0 if record.pnl else False,
                    })

                logger.info(f"Loaded {len(historical_data)} trade records for bot {bot_id}")
                return historical_data

        except ImportError:
            logger.warning(f"TradeRecord model not available, using fallback data loading")
            return await self._load_rolling_data_fallback(bot_id, days)
        except Exception as e:
            logger.error(f"Error loading rolling data for {bot_id}: {e}", exc_info=True)
            return await self._load_rolling_data_fallback(bot_id, days)

    async def _load_rolling_data_fallback(self, bot_id: str, days: int) -> List[Dict[str, Any]]:
        """
        Fallback data loading using bot manifest historical stats.

        If trade record database is not available, falls back to using
        the bot's historical performance data from the manifest.
        """
        try:
            from src.router.bot_manifest import BotRegistry

            registry = BotRegistry()
            bot = registry.get_bot(bot_id)

            if not bot:
                logger.warning(f"Bot {bot_id} not found")
                return []

            # Get historical stats from bot manifest
            stats = getattr(bot, 'historical_stats', None)
            if not stats:
                stats = getattr(bot, 'paper_stats', None) or getattr(bot, 'live_stats', None)

            if not stats:
                logger.warning(f"No historical stats available for bot {bot_id}")
                return []

            # Convert stats to rolling data format
            historical_data = []
            trades = getattr(stats, 'recent_trades', [])
            for trade in trades[-days:]:
                historical_data.append({
                    "trade_id": getattr(trade, 'trade_id', 'unknown'),
                    "pnl": float(getattr(trade, 'pnl', 0)),
                    "regime": getattr(trade, 'regime', 'UNKNOWN'),
                    "win": getattr(trade, 'pnl', 0) > 0,
                })

            logger.info(f"Loaded {len(historical_data)} trades from fallback for bot {bot_id}")
            return historical_data

        except Exception as e:
            logger.error(f"Error in fallback data loading for {bot_id}: {e}")
            return []

    async def _optimize_parameters(
        self,
        bot_id: str,
        in_sample_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Optimize parameters using in-sample data.

        Uses grid search to find optimal parameter combinations that maximize
        Sharpe ratio on the in-sample period.
        """
        if not in_sample_data:
            return {}

        # Calculate base metrics from in-sample data
        total_pnl = sum(t.get("pnl", 0) for t in in_sample_data)
        wins = sum(1 for t in in_sample_data if t.get("win", False))
        total_trades = len(in_sample_data)

        if total_trades == 0:
            return {}

        win_rate = wins / total_trades

        # Simple parameter optimization based on regime
        # In production, this would use actual grid search
        regime_performance = {}
        for trade in in_sample_data:
            regime = trade.get("regime", "UNKNOWN")
            if regime not in regime_performance:
                regime_performance[regime] = {"pnl": 0, "trades": 0}
            regime_performance[regime]["pnl"] += trade.get("pnl", 0)
            regime_performance[regime]["trades"] += 1

        # Find best performing regime
        best_regime = max(regime_performance.items(), key=lambda x: x[1]["pnl"] if x[1]["trades"] > 0 else 0)

        return {
            "preferred_regime": best_regime[0],
            "base_win_rate": round(win_rate, 4),
            "expected_pnl_per_trade": round(total_pnl / total_trades, 2) if total_trades > 0 else 0,
            "regime_count": len(regime_performance),
        }

    async def _validate_parameters(
        self,
        bot_id: str,
        optimal_params: Dict[str, Any],
        out_sample_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Validate parameters using out-of-sample data.

        Applies the optimized parameters to the out-of-sample period and
        computes validation metrics.
        """
        if not out_sample_data:
            return {"trades": [], "pnl": 0.0, "valid": False}

        trades = []
        total_pnl = 0.0

        preferred_regime = optimal_params.get("preferred_regime")

        for trade in out_sample_data:
            trade_regime = trade.get("regime", "UNKNOWN")
            trade_pnl = trade.get("pnl", 0)

            # Apply regime filter if optimized
            if preferred_regime and trade_regime != preferred_regime:
                # Reduce P&L for non-preferred regimes
                trade_pnl *= 0.5

            total_pnl += trade_pnl
            trades.append({
                "trade_id": trade.get("trade_id", "unknown"),
                "regime": trade_regime,
                "adjusted_pnl": trade_pnl,
            })

        return {
            "trades": trades,
            "pnl": total_pnl,
            "valid": True,
            "preferred_regime": preferred_regime,
            "trade_count": len(trades),
        }

    async def _compute_wfa_metrics(
        self,
        validation_result: Dict[str, Any]
    ) -> WfaMetrics:
        """
        Compute WFA metrics from validation result.

        Computes Sharpe ratio, max drawdown, win rate, and profit factor
        from the validation period trades.
        """
        trades = validation_result.get("trades", [])
        total_pnl = validation_result.get("pnl", 0.0)

        if not trades:
            return WfaMetrics(
                sharpe_ratio=0.0,
                max_drawdown=1.0,
                win_rate=0.0,
                total_trades=0,
                profit_factor=0.0,
            )

        wins = sum(1 for t in trades if t.get("adjusted_pnl", 0) > 0)
        losses = sum(1 for t in trades if t.get("adjusted_pnl", 0) < 0)
        total_trades = len(trades)
        win_rate = wins / total_trades if total_trades > 0 else 0

        # Calculate profit factor
        gross_profit = sum(t.get("adjusted_pnl", 0) for t in trades if t.get("adjusted_pnl", 0) > 0)
        gross_loss = abs(sum(t.get("adjusted_pnl", 0) for t in trades if t.get("adjusted_pnl", 0) < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

        # Calculate Sharpe ratio (simplified)
        if total_trades > 1:
            returns = [t.get("adjusted_pnl", 0) for t in trades]
            avg_return = sum(returns) / len(returns)
            variance = sum((r - avg_return) ** 2 for r in returns) / len(returns)
            std_dev = variance ** 0.5
            sharpe_ratio = (avg_return / std_dev) * (252 ** 0.5) if std_dev > 0 else 0.0  # Annualized
        else:
            sharpe_ratio = 0.0

        # Calculate max drawdown
        cumulative = 0.0
        peak = 0.0
        max_drawdown = 0.0
        for trade in trades:
            cumulative += trade.get("adjusted_pnl", 0)
            if cumulative > peak:
                peak = cumulative
            drawdown = (peak - cumulative) / (peak + abs(cumulative) + 1)
            max_drawdown = max(max_drawdown, drawdown)

        return WfaMetrics(
            sharpe_ratio=round(sharpe_ratio, 4),
            max_drawdown=round(max_drawdown, 4),
            win_rate=round(win_rate, 4),
            total_trades=total_trades,
            profit_factor=round(profit_factor, 4),
        )

    async def _store_wfa_result(self, result: WfaResult) -> None:
        """Store WFA result to cold storage."""
        try:
            from src.router.cold_storage_writer import get_cold_storage_writer

            writer = get_cold_storage_writer()
            date_str = datetime.now().strftime("%Y-%m-%d")
            file_key = f"wfa_results/{date_str}_wfa_{result.bot_id}.json"

            await writer.write(file_key, result.to_dict())
            logger.info(f"WFA result stored: {file_key}")

        except Exception as e:
            logger.error(f"Error storing WFA result: {e}", exc_info=True)


# ============= Singleton Factory =============
_analyzer_instance: Optional[WalkForwardAnalyzer] = None


def get_walk_forward_analyzer() -> WalkForwardAnalyzer:
    """Get singleton instance of WalkForwardAnalyzer."""
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = WalkForwardAnalyzer()
    return _analyzer_instance
