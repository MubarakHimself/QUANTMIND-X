"""
DPR Scoring Engine — Daily Performance Ranking Composite Score Calculation.

Story 17.1: DPR Composite Score Calculation

Provides DPRScoringEngine class for calculating bot composite scores on 0-100 scale.

Composite Score Formula:
    session Win Rate (25%) + net PnL (30%) + consistency (20%) + EV/trade (25%)

Per NFR-M2: DPR is a synchronous scoring engine — NO LLM calls in scoring path.
Per NFR-D1: All DPR score calculations logged before any system acknowledgment.
"""

from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Tuple, Any
from dataclasses import dataclass
import logging

import redis

from sqlalchemy.orm import Session
from sqlalchemy import select, func, and_

from src.database.models import BotCircuitBreaker, SessionLocal, TradeJournal


logger = logging.getLogger(__name__)
from src.database.models.base import TradingMode
from src.events.dpr import (
    DPRComponentScores,
    DPRScoreEvent,
    DPRConcernEvent,
    DPR_WEIGHTS,
)


@dataclass
class DPRScore:
    """
    DPR score data for a single bot.

    Attributes:
        bot_id: Bot identifier
        session_id: Session identifier
        composite_score: Final composite score (0-100)
        component_scores: Individual component scores
        trade_count: Number of trades in scoring window
        session_win_rate: Session-specific win rate (for tie-breaking)
        max_drawdown: Max drawdown percentage (for tie-breaking)
        magic_number: Magic number (for final tie-breaking)
        specialist_boost_applied: Whether SESSION_SPECIALIST boost was applied
        consecutive_negative_ev: Consecutive sessions with negative EV per trade
    """
    bot_id: str
    session_id: str
    composite_score: int
    component_scores: DPRComponentScores
    trade_count: int
    session_win_rate: float
    max_drawdown: float
    magic_number: int
    specialist_boost_applied: bool = False
    consecutive_negative_ev: int = 0


class DPRScoringEngine:
    """
    DPR Scoring Engine for calculating bot composite scores.

    Implements the Daily Performance Ranking system for scoring active bots
    on a composite 0-100 scale using four weighted metrics.

    Composite Score Formula:
        session Win Rate (25%) + net PnL (30%) + consistency (20%) + EV/trade (25%)

    Attributes:
        db_session: SQLAlchemy session for database access
        benchmark_pnl: Portfolio benchmark PnL for normalization (default 0)
        baseline_ev: Baseline EV per trade for normalization (default 0)
        max_acceptable_variance: Max daily return variance for consistency normalization
    """

    def __init__(
        self,
        db_session: Optional[Session] = None,
        benchmark_pnl: float = 0.0,
        baseline_ev: float = 0.0,
        max_acceptable_variance: float = 0.01,
        redis_host: str = "localhost",
        redis_port: int = 6379,
    ):
        """
        Initialize DPR Scoring Engine.

        Args:
            db_session: SQLAlchemy session (creates new if None)
            benchmark_pnl: Portfolio benchmark PnL for PnL normalization
            baseline_ev: Baseline expected value per trade for EV normalization
            max_acceptable_variance: Max daily variance for consistency scoring
            redis_host: Redis host for counter persistence
            redis_port: Redis port for counter persistence
        """
        self._db_session = db_session
        self.benchmark_pnl = benchmark_pnl
        self.baseline_ev = baseline_ev
        self.max_acceptable_variance = max_acceptable_variance
        self._redis_host = redis_host
        self._redis_port = redis_port
        self._redis_client = None

    @property
    def db_session(self) -> Session:
        """Get or create database session."""
        if self._db_session is None:
            self._db_session = SessionLocal()
        return self._db_session

    @property
    def redis_client(self):
        """Get or create async Redis client (lazy initialization)."""
        if self._redis_client is None:
            self._redis_client = redis.asyncio.Redis(
                host=self._redis_host,
                port=self._redis_port,
                decode_responses=True,
            )
        return self._redis_client

    async def _get_consecutive_negative_ev_counter(self, magic_number: int) -> int:
        """
        Get consecutive negative EV counter for a bot from Redis.

        Args:
            magic_number: MT5 magic number

        Returns:
            Current counter value (0 if not set or Redis unavailable)
        """
        try:
            key = f"session_concern:{magic_number}"
            value = await self.redis_client.get(key)
            if value is not None:
                return int(value)
            return 0
        except Exception as e:
            logger.warning(f"Failed to get consecutive negative EV counter for magic {magic_number}: {e}")
            return 0

    async def _increment_consecutive_negative_ev(self, magic_number: int) -> int:
        """
        Atomically increment consecutive negative EV counter.

        Args:
            magic_number: MT5 magic number

        Returns:
            New counter value after increment
        """
        try:
            key = f"session_concern:{magic_number}"
            new_count = await self.redis_client.incr(key)
            await self.redis_client.expire(key, 604800)  # 7-day TTL
            return new_count
        except Exception as e:
            logger.warning(f"Failed to increment consecutive negative EV counter for magic {magic_number}: {e}")
            return 0

    async def _reset_consecutive_negative_ev(self, magic_number: int) -> int:
        """
        Atomically reset consecutive negative EV counter to 0.

        Args:
            magic_number: MT5 magic number

        Returns:
            0 (reset complete)
        """
        try:
            key = f"session_concern:{magic_number}"
            await self.redis_client.delete(key)
            return 0
        except Exception as e:
            logger.warning(f"Failed to reset consecutive negative EV counter for magic {magic_number}: {e}")
            return 0

    def calculate_composite_score(
        self,
        bot_id: str,
        session_id: str,
        scoring_window: str = "session",
    ) -> Optional[int]:
        """
        Calculate composite DPR score for a bot.

        AC #1: Given a bot has completed at least one trade in the scoring window,
        When DPR evaluates the bot,
        Then it computes the composite score...

        Args:
            bot_id: Bot identifier
            session_id: Session identifier
            scoring_window: Time window for scoring ("session" or "fortnight")

        Returns:
            Composite score (0-100) or None if bot not eligible (< 1 trade)
        """
        # Get trade data for the bot in scoring window
        trade_data = self._get_trade_data(bot_id, session_id, scoring_window)

        # Check eligibility: minimum 1 trade
        if trade_data["total_trades"] < 1:
            return None

        # Calculate component scores
        win_rate_score = self._normalize_win_rate(
            trade_data["wins"],
            trade_data["total_trades"]
        )
        pnl_score = self._normalize_pnl(trade_data["net_pnl"])
        consistency_score = self._normalize_consistency(trade_data["daily_variance"])
        ev_score = self._normalize_ev(trade_data["ev_per_trade"])

        # Build component scores object
        component_scores = DPRComponentScores(
            win_rate=win_rate_score,
            pnl=pnl_score,
            consistency=consistency_score,
            ev_per_trade=ev_score,
            weights=DPR_WEIGHTS,
        )

        # Calculate composite score
        composite = component_scores.composite_score()

        # Apply specialist boost
        if self._is_specialist(bot_id, session_id):
            composite = min(100, composite + 5)

        return composite

    async def get_dpr_score(self, bot_id: str, session_id: str) -> Optional[DPRScore]:
        """
        Get full DPR score data for a bot.

        Args:
            bot_id: Bot identifier
            session_id: Session identifier

        Returns:
            DPRScore object or None if not eligible
        """
        trade_data = self._get_trade_data(bot_id, session_id, "session")

        if trade_data["total_trades"] < 1:
            return None

        # Calculate component scores
        win_rate_score = self._normalize_win_rate(
            trade_data["wins"],
            trade_data["total_trades"]
        )
        pnl_score = self._normalize_pnl(trade_data["net_pnl"])
        consistency_score = self._normalize_consistency(trade_data["daily_variance"])
        ev_score = self._normalize_ev(trade_data["ev_per_trade"])

        component_scores = DPRComponentScores(
            win_rate=win_rate_score,
            pnl=pnl_score,
            consistency=consistency_score,
            ev_per_trade=ev_score,
            weights=DPR_WEIGHTS,
        )

        composite = component_scores.composite_score()
        specialist_boost_applied = False

        # Apply specialist boost
        if self._is_specialist(bot_id, session_id):
            composite = min(100, composite + 5)
            specialist_boost_applied = True

        # Track consecutive negative EV sessions (atomic operations)
        ev_per_trade = trade_data["ev_per_trade"]
        magic_number = trade_data.get("magic_number", 0)
        if ev_per_trade < 0:
            consecutive_negative_ev = await self._increment_consecutive_negative_ev(magic_number)
        else:
            await self._reset_consecutive_negative_ev(magic_number)
            consecutive_negative_ev = 0

        return DPRScore(
            bot_id=bot_id,
            session_id=session_id,
            composite_score=composite,
            component_scores=component_scores,
            trade_count=trade_data["total_trades"],
            session_win_rate=trade_data["wins"] / trade_data["total_trades"] if trade_data["total_trades"] > 0 else 0,
            max_drawdown=trade_data.get("max_drawdown", 0.0),
            magic_number=magic_number,
            specialist_boost_applied=specialist_boost_applied,
            consecutive_negative_ev=consecutive_negative_ev,
        )

    def tie_break_cascade(
        self,
        bot_a_scores: DPRScore,
        bot_b_scores: DPRScore,
    ) -> str:
        """
        Apply 4-level tie-break cascade when two bots have equal composite scores.

        AC #2: Given two bots tie on the composite DPR score,
        When the tie-break cascade evaluates,
        Then it applies in order:
            (1) higher session-specific win rate wins
            (2) lower max drawdown wins
            (3) higher trade count wins
            (4) lower Magic Number wins (GG-2 resolution)

        Args:
            bot_a_scores: DPRScore for bot A
            bot_b_scores: DPRScore for bot B

        Returns:
            Winner bot_id
        """
        # Level 1: Higher session-specific win rate wins
        if bot_a_scores.session_win_rate != bot_b_scores.session_win_rate:
            return bot_a_scores.bot_id if bot_a_scores.session_win_rate > bot_b_scores.session_win_rate else bot_b_scores.bot_id

        # Level 2: Lower max drawdown wins
        if bot_a_scores.max_drawdown != bot_b_scores.max_drawdown:
            return bot_a_scores.bot_id if bot_a_scores.max_drawdown < bot_b_scores.max_drawdown else bot_b_scores.bot_id

        # Level 3: Higher trade count wins
        if bot_a_scores.trade_count != bot_b_scores.trade_count:
            return bot_a_scores.bot_id if bot_a_scores.trade_count > bot_b_scores.trade_count else bot_b_scores.bot_id

        # Level 4: Lower Magic Number wins (GG-2 resolution)
        return bot_a_scores.bot_id if bot_a_scores.magic_number < bot_b_scores.magic_number else bot_b_scores.bot_id

    def apply_specialist_boost(
        self,
        score: int,
        bot_id: str,
        session: str,
    ) -> int:
        """
        Apply SESSION_SPECIALIST +5 boost for specialist session only.

        AC #3: Given a bot has a SESSION_SPECIALIST tag,
        When the DPR queue is finalised,
        Then the tagged bot receives a +5 boost to its composite score
        for its specialist session only.

        Boost does not stack (only +5 max) and caps at 100.

        Args:
            score: Current composite score
            bot_id: Bot identifier
            session: Current session identifier

        Returns:
            Score with boost applied (capped at 100)
        """
        if self._is_specialist(bot_id, session):
            return min(100, score + 5)
        return score

    def check_concern_flag(self, bot_id: str) -> bool:
        """
        Check if bot's DPR score dropped >20 points week-over-week.

        AC #4: Given a bot's DPR score drops >20 points week-over-week,
        When the fortnight accumulation completes,
        Then a SESSION_CONCERN flag is set on that bot.

        Args:
            bot_id: Bot identifier

        Returns:
            True if concern flag should be set
        """
        delta = self.week_over_week_score_delta(bot_id)
        return self.threshold_check(delta, threshold=-20)

    def week_over_week_score_delta(self, bot_id: str) -> int:
        """
        Calculate week-over-week DPR score delta.

        Args:
            bot_id: Bot identifier

        Returns:
            Score delta (current - previous) or 0 if insufficient history
        """
        from src.risk.dpr.history import DPRScoreHistory

        history = DPRScoreHistory(db_session=self.db_session)
        scores = history.get_bot_scores(bot_id, limit=2)

        if len(scores) < 2:
            return 0

        # Most recent is current, previous is second most recent
        current_score = scores[0].composite_score
        previous_score = scores[1].composite_score

        return current_score - previous_score

    def threshold_check(self, delta: int, threshold: int = -20) -> bool:
        """
        Check if score delta exceeds threshold.

        Args:
            delta: Score change (negative = dropped)
            threshold: Threshold to check against (default -20)

        Returns:
            True if threshold exceeded
        """
        return delta < threshold

    def get_session_specialists(self, session_id: str) -> List[str]:
        """
        Get list of bot IDs with SESSION_SPECIALIST tag for a session.

        Args:
            session_id: Session identifier

        Returns:
            List of bot IDs with specialist tag for this session
        """
        try:
            from src.router.bot_manifest import BotRegistry

            registry = BotRegistry()
            specialists = registry.list_by_tag('SESSION_SPECIALIST')
            return [bot.bot_id for bot in specialists]
        except Exception:
            return []

    def specialist_session_check(self, bot_id: str, current_session: str) -> bool:
        """
        Check if bot is a specialist for the current session.

        Args:
            bot_id: Bot identifier
            current_session: Current session identifier

        Returns:
            True if bot is specialist for this session
        """
        specialists = self.get_session_specialists(current_session)
        return bot_id in specialists

    # ==================== Normalization Functions ====================

    def _normalize_win_rate(self, wins: int, total: int) -> float:
        """
        Normalize win rate to 0-100 scale.

        Args:
            wins: Number of winning trades
            total: Total number of trades

        Returns:
            Normalized score (0-100)
        """
        if total == 0:
            return 0.0
        # 0% WR = 0, 100% WR = 100 (linear)
        return (wins / total) * 100.0

    def _normalize_pnl(self, net_pnl: float) -> float:
        """
        Normalize net PnL to 0-100 scale.

        Uses benchmark PnL for relative scoring. Negative PnL normalizes to 0.

        Args:
            net_pnl: Net PnL value

        Returns:
            Normalized score (0-100)
        """
        if net_pnl <= 0:
            return 0.0

        # If no benchmark, use absolute PnL scaled
        if self.benchmark_pnl == 0:
            # Use a fixed reference — e.g., 1000 points = score of 50
            normalized = (net_pnl / 1000.0) * 50.0
        else:
            # Relative to benchmark
            normalized = (net_pnl / self.benchmark_pnl) * 50.0

        return min(100.0, max(0.0, normalized))

    def _normalize_consistency(self, daily_variance: float) -> float:
        """
        Normalize consistency (inverse of daily return variance) to 0-100.

        Lower variance = higher score. Max acceptable variance yields score of 50.

        Args:
            daily_variance: Daily return variance

        Returns:
            Normalized score (0-100)
        """
        if daily_variance <= 0:
            return 100.0  # Perfect consistency

        if daily_variance >= self.max_acceptable_variance:
            return 0.0  # Max variance

        # Inverse scaling: 0 variance = 100, max_variance = 0
        normalized = (1.0 - (daily_variance / self.max_acceptable_variance)) * 100.0
        return max(0.0, min(100.0, normalized))

    def _normalize_ev(self, ev_per_trade: float) -> float:
        """
        Normalize expected value per trade to 0-100 scale.

        Uses baseline EV for relative scoring. Negative EV normalizes to 0.

        Args:
            ev_per_trade: Expected value per trade

        Returns:
            Normalized score (0-100)
        """
        if ev_per_trade <= 0:
            return 0.0

        if self.baseline_ev == 0:
            # Use fixed reference — e.g., 10 points EV = score of 50
            normalized = (ev_per_trade / 10.0) * 50.0
        else:
            # Relative to baseline
            normalized = (ev_per_trade / self.baseline_ev) * 50.0

        return min(100.0, max(0.0, normalized))

    # ==================== Data Access ====================

    def _get_trade_data(
        self,
        bot_id: str,
        session_id: str,
        scoring_window: str,
    ) -> Dict[str, Any]:
        """
        Get trade data for bot in scoring window.

        Args:
            bot_id: Bot identifier
            session_id: Session identifier
            scoring_window: "session" or "fortnight"

        Returns:
            Dictionary with trade metrics
        """
        # Determine time window
        now = datetime.now(timezone.utc)
        if scoring_window == "fortnight":
            start_time = now - timedelta(days=14)
        else:
            # Session window: last 24 hours
            start_time = now - timedelta(hours=24)

        try:
            # Query TradeJournal for bot's trades in the scoring window
            trades = self.db_session.query(TradeJournal).filter(
                and_(
                    TradeJournal.bot_id == bot_id,
                    TradeJournal.timestamp >= start_time,
                    TradeJournal.timestamp <= now,
                    TradeJournal.pnl.isnot(None)  # Only closed trades
                )
            ).order_by(TradeJournal.timestamp.desc()).all()

            if not trades:
                return {
                    "total_trades": 0,
                    "wins": 0,
                    "net_pnl": 0.0,
                    "daily_variance": 0.0,
                    "ev_per_trade": 0.0,
                    "max_drawdown": 0.0,
                    "magic_number": 0,
                }

            # Calculate metrics
            total_trades = len(trades)
            wins = sum(1 for t in trades if t.pnl > 0)
            net_pnl = sum(t.pnl for t in trades if t.pnl is not None)

            # Calculate daily returns for variance
            daily_returns = {}
            for trade in trades:
                if trade.timestamp:
                    day_key = trade.timestamp.date().isoformat()
                    if day_key not in daily_returns:
                        daily_returns[day_key] = []
                    if trade.pnl is not None:
                        daily_returns[day_key].append(trade.pnl)

            # Calculate variance of daily returns
            if daily_returns:
                daily_pnls = [sum(pnls) for pnls in daily_returns.values()]
                if len(daily_pnls) > 1:
                    mean_daily = sum(daily_pnls) / len(daily_pnls)
                    variance = sum((p - mean_daily) ** 2 for p in daily_pnls) / len(daily_pnls)
                else:
                    variance = 0.0
            else:
                variance = 0.0

            # Calculate EV per trade
            avg_win = sum(t.pnl for t in trades if t.pnl > 0) / wins if wins > 0 else 0
            avg_loss = abs(sum(t.pnl for t in trades if t.pnl < 0) / (total_trades - wins)) if total_trades > wins else 0
            win_rate = wins / total_trades if total_trades > 0 else 0
            ev_per_trade = (win_rate * avg_win) - ((1 - win_rate) * avg_loss) if total_trades > 0 else 0

            # Calculate max drawdown
            running_pnl = 0.0
            max_drawdown = 0.0
            peak_pnl = 0.0
            for t in sorted(trades, key=lambda x: x.timestamp):
                if t.pnl is not None:
                    running_pnl += t.pnl
                    if running_pnl > peak_pnl:
                        peak_pnl = running_pnl
                    drawdown = peak_pnl - running_pnl
                    if drawdown > max_drawdown:
                        max_drawdown = drawdown

            # Get magic number from first trade's metadata or default to 0
            magic_number = 0
            if trades and hasattr(trades[0], 'strategy_folder_id') and trades[0].strategy_folder_id:
                magic_number = trades[0].strategy_folder_id

            return {
                "total_trades": total_trades,
                "wins": wins,
                "net_pnl": net_pnl,
                "daily_variance": variance,
                "ev_per_trade": ev_per_trade,
                "max_drawdown": max_drawdown,
                "magic_number": magic_number,
            }

        except Exception:
            return {
                "total_trades": 0,
                "wins": 0,
                "net_pnl": 0.0,
                "daily_variance": 0.0,
                "ev_per_trade": 0.0,
                "max_drawdown": 0.0,
                "magic_number": 0,
            }

    def _is_specialist(self, bot_id: str, session_id: str) -> bool:
        """
        Check if bot is specialist for session.

        Args:
            bot_id: Bot identifier
            session_id: Session identifier

        Returns:
            True if bot has SESSION_SPECIALIST tag for session
        """
        return self.specialist_session_check(bot_id, session_id)

    def recalculate_paper_only_score(
        self,
        bot_id: str,
        session_id: str,
    ) -> Optional[int]:
        """
        Recalculate DPR score with paper-only inputs.

        When a bot moves to paper-only:
        - Win Rate = 0 (paper trades do not contribute to live P&L)
        - PnL = 0
        - EV/trade = 0
        - Consistency = calculated from session performance

        Args:
            bot_id: Bot identifier
            session_id: Session identifier

        Returns:
            Paper-only composite score (0-100) or None if not eligible
        """
        trade_data = self._get_trade_data(bot_id, session_id, "session")

        # Check eligibility: minimum 1 trade
        if trade_data["total_trades"] < 1:
            return None

        # Paper-only scoring: WR=0, PnL=0, EV=0, Consistency=calculated
        win_rate_score = 0.0  # Paper trades don't contribute to live P&L
        pnl_score = 0.0  # Paper P&L doesn't count
        ev_score = 0.0  # Paper EV doesn't count
        consistency_score = self._normalize_consistency(trade_data["daily_variance"])

        # Build component scores object
        component_scores = DPRComponentScores(
            win_rate=win_rate_score,
            pnl=pnl_score,
            consistency=consistency_score,
            ev_per_trade=ev_score,
            weights=DPR_WEIGHTS,
        )

        # Calculate composite score
        composite = component_scores.composite_score()

        # No specialist boost for paper-only bots
        return composite

    async def fetch_trade_results_from_node_trading(
        self,
        session_date: Optional[str] = None,
        timeout: float = 30.0,
    ) -> Dict[str, Any]:
        """
        Fetch trade result summaries from node_trading for DPR scoring.

        Called by T2 (DPR scoring engine) to pull trade data from T1 (node_trading)
        at the start of each scoring cycle.

        Args:
            session_date: ISO date string (YYYY-MM-DD) to fetch results for.
                          If None, fetches the last 24 hours.
            timeout: Request timeout in seconds.

        Returns:
            Dict with 'session_date' and 'summaries' list of trade results.

        Raises:
            RuntimeError: If the HTTP call fails or returns an error.
        """
        import os
        import httpx

        base_url = os.environ.get("NODE_TRADING_URL", "http://localhost:8001")
        token = os.environ.get("NODE_INTERNAL_TOKEN", "")

        params = {}
        if session_date:
            params["session_date"] = session_date
        if token:
            params["token"] = token

        url = f"{base_url}/api/trading/trade-results/summary"
        logger.info(f"Fetching trade results from node_trading: {url}")

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                logger.info(
                    f"Fetched {len(data.get('summaries', []))} trade summaries "
                    f"for session {session_date or 'last-24h'}"
                )
                return data
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error fetching trade results: {e.response.status_code} — {e}")
            raise RuntimeError(f"node_trading returned {e.response.status_code}") from e
        except httpx.RequestError as e:
            logger.error(f"Request error fetching trade results from node_trading: {e}")
            raise RuntimeError(f"Failed to reach node_trading at {base_url}") from e
        except Exception as e:
            logger.error(f"Unexpected error fetching trade results: {e}")
            raise RuntimeError(str(e)) from e

    def close(self):
        """Close database session and Redis client if we created them."""
        if self._db_session is not None:
            self._db_session.close()
            self._db_session = None
        if self._redis_client is not None:
            self._redis_client.close()
            self._redis_client = None
