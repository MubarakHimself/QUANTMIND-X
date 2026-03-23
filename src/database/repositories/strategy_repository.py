"""
Strategy Repository

Provides data access methods for StrategyPerformance model.
"""

from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session

from ..models import StrategyPerformance
from ..engine import Session as SessionFactory


class StrategyRepository:
    """
    Repository for StrategyPerformance data access.

    Handles all database operations related to strategy performance tracking.
    """

    def __init__(self, session: Optional[Session] = None):
        """Initialize repository with optional session."""
        self._session = session

    @property
    def session(self) -> Session:
        """Get the session (creates new if not provided)."""
        if self._session is not None:
            return self._session
        return SessionFactory()

    def create(
        self,
        strategy_name: str,
        backtest_results: Dict[str, Any],
        kelly_score: float,
        sharpe_ratio: float,
        max_drawdown: float,
        win_rate: Optional[float] = None,
        profit_factor: Optional[float] = None,
        total_trades: Optional[int] = None
    ) -> StrategyPerformance:
        """
        Create a new strategy performance record.

        Args:
            strategy_name: Name of the strategy
            backtest_results: Dictionary containing backtest metrics
            kelly_score: Kelly criterion score
            sharpe_ratio: Sharpe ratio
            max_drawdown: Maximum drawdown percentage
            win_rate: Win rate percentage (optional)
            profit_factor: Profit factor (optional)
            total_trades: Total number of trades (optional)

        Returns:
            Created StrategyPerformance object
        """
        performance = StrategyPerformance(
            strategy_name=strategy_name,
            backtest_results=backtest_results,
            kelly_score=kelly_score,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=total_trades
        )
        self.session.add(performance)
        self.session.flush()
        self.session.refresh(performance)
        self.session.expunge(performance)
        return performance

    def get(
        self,
        strategy_name: Optional[str] = None,
        min_kelly_score: Optional[float] = None,
        min_sharpe_ratio: Optional[float] = None,
        limit: int = 100
    ) -> List[StrategyPerformance]:
        """
        Retrieve strategy performance records with optional filtering.

        Args:
            strategy_name: Filter by strategy name (optional)
            min_kelly_score: Minimum Kelly score filter (optional)
            min_sharpe_ratio: Minimum Sharpe ratio filter (optional)
            limit: Maximum number of records to return

        Returns:
            List of StrategyPerformance objects
        """
        query = self.session.query(StrategyPerformance)

        if strategy_name is not None:
            query = query.filter(StrategyPerformance.strategy_name == strategy_name)

        if min_kelly_score is not None:
            query = query.filter(StrategyPerformance.kelly_score >= min_kelly_score)

        if min_sharpe_ratio is not None:
            query = query.filter(StrategyPerformance.sharpe_ratio >= min_sharpe_ratio)

        query = query.order_by(StrategyPerformance.created_at.desc()).limit(limit)

        performances = query.all()
        # Expunge all performances to avoid detached instance errors
        for perf in performances:
            self.session.expunge(perf)
        return performances

    def get_best(
        self,
        limit: int = 10,
        order_by: str = 'kelly_score'
    ) -> List[StrategyPerformance]:
        """
        Get the best performing strategies.

        Args:
            limit: Maximum number of strategies to return
            order_by: Metric to order by ('kelly_score' or 'sharpe_ratio')

        Returns:
            List of top StrategyPerformance objects
        """
        query = self.session.query(StrategyPerformance)

        if order_by == 'sharpe_ratio':
            query = query.order_by(StrategyPerformance.sharpe_ratio.desc())
        else:
            query = query.order_by(StrategyPerformance.kelly_score.desc())

        query = query.limit(limit)

        strategies = query.all()
        # Expunge all strategies to avoid detached instance errors
        for strategy in strategies:
            self.session.expunge(strategy)
        return strategies
