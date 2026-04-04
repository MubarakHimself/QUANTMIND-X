"""
DPR Scoring Engine
==================

Daily Performance Ranking - composite 0-100 score for bots.
Weights: session WR 25%, net PnL 30%, consistency 20%, EV/trade 25%
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class DprComponents:
    """Components of DPR composite score."""
    session_win_rate: float  # 25% weight
    net_pnl: float  # 30% weight
    consistency: float  # 20% weight
    ev_per_trade: float  # 25% weight


@dataclass
class DprScore:
    """Daily Performance Ranking score for a bot."""
    bot_id: str
    composite_score: float  # 0-100
    components: DprComponents
    rank: int = 0
    tier: str = "T3"  # T1, T2, T3
    session_specialist: bool = False
    session_concern: bool = False
    consecutive_negative_ev: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bot_id": self.bot_id,
            "composite_score": round(self.composite_score, 2),
            "components": {
                "session_win_rate": round(self.components.session_win_rate, 4),
                "net_pnl": round(self.components.net_pnl, 2),
                "consistency": round(self.components.consistency, 4),
                "ev_per_trade": round(self.components.ev_per_trade, 4),
            },
            "rank": self.rank,
            "tier": self.tier,
            "session_specialist": self.session_specialist,
            "session_concern": self.session_concern,
            "consecutive_negative_ev": self.consecutive_negative_ev,
        }


@dataclass
class BotSessionMetrics:
    """Session metrics for a bot."""
    bot_id: str
    session_wr: float  # Session win rate
    net_pnl: float  # Net PnL for session
    consistency: float  # Consistency score (0-1)
    ev_per_trade: float  # Expected value per trade
    consecutive_negative_ev: int = 0


class DprScoringEngine:
    """
    DPR Scoring Engine — composite 0-100 score.

    Weights: session WR 25%, net PnL 30%, consistency 20%, EV/trade 25%
    """

    WEIGHTS = {
        "session_win_rate": 0.25,
        "net_pnl": 0.30,
        "consistency": 0.20,
        "ev_per_trade": 0.25,
    }

    # Tier thresholds
    T1_THRESHOLD = 80.0
    T2_THRESHOLD = 50.0

    def __init__(self):
        # Baselines for normalization - calibrated for DPR scoring
        # PnL baseline = 500: so 500 PnL → 100, 1000 PnL → 200 (allows high performers to score well)
        self._pnl_baseline = 500.0
        # EV baseline = 1.25: so 1.25 EV → 100 (matches max EV expected in trading)
        self._ev_baseline = 1.25
        logger.info("DprScoringEngine initialized")

    def compute_composite_score(
        self,
        session_wr: float,
        net_pnl: float,
        consistency: float,
        ev_per_trade: float
    ) -> float:
        """
        Compute weighted composite DPR score 0-100.

        Args:
            session_wr: Session win rate (0-1)
            net_pnl: Net PnL for the session
            consistency: Consistency score (0-1)
            ev_per_trade: Expected value per trade

        Returns:
            Composite score (0-100)
        """
        score = (
            (session_wr * 100) * self.WEIGHTS["session_win_rate"] +
            self._normalize_pnl(net_pnl) * self.WEIGHTS["net_pnl"] +
            (consistency * 100) * self.WEIGHTS["consistency"] +
            self._normalize_ev(ev_per_trade) * self.WEIGHTS["ev_per_trade"]
        )
        return min(100.0, max(0.0, score))

    def _normalize_pnl(self, pnl: float) -> float:
        """
        Normalize PnL to 0-100 scale using historical distribution.

        Higher PnL = higher normalized score.
        PnL baseline = 500 means 500 PnL → 100, 1000 PnL → 200.
        """
        return min(100.0, max(0.0, (pnl / self._pnl_baseline) * 100))

    def _normalize_ev(self, ev: float) -> float:
        """
        Normalize EV/trade to 0-100 scale.

        Higher EV = higher score.
        EV baseline = 1.25 means 1.25 EV → 100.
        """
        return min(100.0, max(0.0, (ev / self._ev_baseline) * 100))

    def _assign_tier(self, composite_score: float) -> str:
        """Assign tier based on composite score."""
        if composite_score >= self.T1_THRESHOLD:
            return "T1"
        elif composite_score >= self.T2_THRESHOLD:
            return "T2"
        return "T3"

    async def score_all_bots(self, bot_ids: List[str]) -> List[DprScore]:
        """
        Score all active bots for DPR ranking.

        Args:
            bot_ids: List of bot IDs to score

        Returns:
            List of DprScore objects sorted by composite score descending
        """
        scores = []

        for bot_id in bot_ids:
            metrics = await self._get_bot_session_metrics(bot_id)
            composite = self.compute_composite_score(
                session_wr=metrics.session_wr,
                net_pnl=metrics.net_pnl,
                consistency=metrics.consistency,
                ev_per_trade=metrics.ev_per_trade
            )

            scores.append(DprScore(
                bot_id=bot_id,
                composite_score=composite,
                components=DprComponents(
                    session_win_rate=metrics.session_wr,
                    net_pnl=metrics.net_pnl,
                    consistency=metrics.consistency,
                    ev_per_trade=metrics.ev_per_trade,
                ),
                rank=0,  # Set after sorting
                tier=self._assign_tier(composite),
                session_specialist=False,  # Set by Session Performer step
                session_concern=False,  # Set by Queue Re-rank step
                consecutive_negative_ev=metrics.consecutive_negative_ev,
            ))

        # Sort by composite score descending and assign ranks
        scores.sort(key=lambda s: s.composite_score, reverse=True)
        for i, score in enumerate(scores):
            score.rank = i + 1

        logger.info(f"Scored {len(scores)} bots for DPR")
        return scores

    async def _get_bot_session_metrics(self, bot_id: str) -> BotSessionMetrics:
        """
        Get session metrics for a bot.

        Computes actual consistency and ev_per_trade from bot's ModePerformanceStats
        if available; otherwise uses default placeholders.
        """
        try:
            from src.router.bot_manifest import BotRegistry

            registry = BotRegistry()
            bot = registry.get(bot_id)

            if bot:
                # Get metrics from live_stats or paper_stats
                stats = getattr(bot, 'live_stats', None) or getattr(bot, 'paper_stats', None)
                if stats:
                    session_wr = stats.win_rate
                    net_pnl = stats.total_pnl

                    # Compute consistency: reliability proxy = min(1.0, total_trades / 100)
                    if stats.total_trades > 0:
                        consistency = min(1.0, stats.total_trades / 100.0)
                    else:
                        consistency = 0.0

                    # Compute ev_per_trade from profit_factor when available
                    # ev_per_trade = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
                    # Using profit_factor to derive avg_win and avg_loss:
                    #   PF = (wr * avg_win) / ((1-wr) * avg_loss)
                    #   net_pnl = (wr * total_trades * avg_win) - ((1-wr) * total_trades * avg_loss)
                    ev_per_trade = 0.0
                    if stats.profit_factor > 1.0 and stats.winning_trades > 0 and stats.losing_trades > 0:
                        wr = stats.win_rate
                        PF = stats.profit_factor
                        # gross_losses = total_pnl / (PF - 1)
                        gross_losses = abs(net_pnl) / (PF - 1) if (PF - 1) != 0 else 0
                        gross_wins = PF * gross_losses
                        avg_win = gross_wins / stats.winning_trades if stats.winning_trades > 0 else 0
                        avg_loss = gross_losses / stats.losing_trades if stats.losing_trades > 0 else 0
                        ev_per_trade = (wr * avg_win) - ((1 - wr) * avg_loss)
                    elif stats.total_trades > 0:
                        # Fallback: use net PnL per trade as EV proxy
                        ev_per_trade = net_pnl / stats.total_trades
                else:
                    session_wr = getattr(bot, 'win_rate', 0.5)
                    net_pnl = 0.0
                    consistency = 0.5
                    ev_per_trade = 0.0

                return BotSessionMetrics(
                    bot_id=bot_id,
                    session_wr=session_wr,
                    net_pnl=net_pnl,
                    consistency=consistency,
                    ev_per_trade=ev_per_trade,
                    consecutive_negative_ev=0,  # Tracked externally by Session Performer
                )

        except Exception as e:
            logger.warning(f"Could not get metrics for bot {bot_id}: {e}")

        # Return default metrics if bot not found
        return BotSessionMetrics(
            bot_id=bot_id,
            session_wr=0.5,
            net_pnl=0.0,
            consistency=0.5,
            ev_per_trade=0.0,
            consecutive_negative_ev=0,
        )


# ============= Singleton Factory =============
_dpr_scoring_engine: Optional[DprScoringEngine] = None


def get_dpr_scoring_engine() -> DprScoringEngine:
    """Get singleton instance of DprScoringEngine."""
    global _dpr_scoring_engine
    if _dpr_scoring_engine is None:
        _dpr_scoring_engine = DprScoringEngine()
    return _dpr_scoring_engine
