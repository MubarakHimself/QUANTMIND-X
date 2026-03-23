"""
Unified Database Manager

Provides a unified interface to SQLite (via SQLAlchemy).
Implements singleton pattern for application-wide database access with
automatic retry logic and connection management.

This module delegates to repository classes for modularity while maintaining
backward compatibility with the existing API.
"""

import os
from datetime import datetime, date
from typing import Optional, List, Dict, Any
from contextlib import contextmanager

from .engine import engine, Session, init_database as init_sqlite_db, close_session
from .models import PropFirmAccount, DailySnapshot, TradeProposal, AgentTasks, StrategyPerformance, BrokerRegistry, HouseMoneyState, BotCircuitBreaker, Base
from .retry import DatabaseConnectionManager, create_connection_manager, with_retry
from .repositories import (
    AccountRepository,
    SnapshotRepository,
    ProposalRepository,
    TaskRepository,
    StrategyRepository,
)


class DatabaseManager:
    """
    Unified database manager for QuantMind Hybrid Core.

    Provides methods for SQLite operations (accounts, snapshots, proposals).

    Usage:
        db = DatabaseManager()
        account = db.get_prop_account("12345")
        db.save_daily_snapshot("12345", 105000.0, 100000.0)

    Context manager support:
        with db.get_session() as session:
            # automatic commit/rollback
            pass

        # Or use DatabaseManager directly as context manager
        with DatabaseManager() as db:
            account = db.get_prop_account("12345")
    """

    _instance: Optional['DatabaseManager'] = None

    def __new__(cls):
        """Implement singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize the database manager."""
        if self._initialized:
            return

        # Initialize SQLite database
        init_sqlite_db()

        # Initialize connection manager with retry logic
        self.connection_manager = create_connection_manager(engine)

        # Verify database connection
        try:
            self.connection_manager.ensure_connection()
        except Exception as e:
            # Log but don't fail initialization
            import logging
            logging.getLogger(__name__).warning(f"Initial database connection check failed: {e}")

        # Initialize repository instances
        self.accounts = AccountRepository()
        self.snapshots = SnapshotRepository()
        self.proposals = ProposalRepository()
        self.tasks = TaskRepository()
        self.strategies = StrategyRepository()

        self._initialized = True

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup sessions."""
        try:
            self.close_all_sessions()
        except Exception:
            # Don't raise exceptions during cleanup
            pass
        return False

    @contextmanager
    def get_session(self):
        """
        Context manager for database sessions.

        Automatically handles commit/rollback and session cleanup.

        Yields:
            SQLAlchemy session object

        Example:
            with db.get_session() as session:
                account = session.query(PropFirmAccount).first()
        """
        session = Session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            Session.remove()

    # ========================================================================
    # SQLite Methods - Prop Firm Accounts (delegated to AccountRepository)
    # ========================================================================

    def get_prop_account(self, account_id: str) -> Optional[PropFirmAccount]:
        """
        Retrieve a prop firm account by account ID.

        Args:
            account_id: MT5 account number

        Returns:
            PropFirmAccount object or None if not found
        """
        return self.accounts.get(account_id)

    def create_prop_account(
        self,
        account_id: str,
        firm_name: str,
        daily_loss_limit_pct: float = 5.0,
        hard_stop_buffer_pct: float = 1.0,
        target_profit_pct: float = 8.0,
        min_trading_days: int = 5
    ) -> PropFirmAccount:
        """
        Create a new prop firm account.

        Args:
            account_id: MT5 account number
            firm_name: Name of prop firm
            daily_loss_limit_pct: Maximum daily loss percentage
            hard_stop_buffer_pct: Safety buffer percentage
            target_profit_pct: Profit target percentage
            min_trading_days: Minimum trading days required

        Returns:
            Created PropFirmAccount object
        """
        return self.accounts.create(
            account_id=account_id,
            firm_name=firm_name,
            daily_loss_limit_pct=daily_loss_limit_pct,
            hard_stop_buffer_pct=hard_stop_buffer_pct,
            target_profit_pct=target_profit_pct,
            min_trading_days=min_trading_days
        )

    # ========================================================================
    # SQLite Methods - Daily Snapshots (delegated to SnapshotRepository)
    # ========================================================================

    def save_daily_snapshot(
        self,
        account_id: str,
        equity: float,
        balance: float,
        snapshot_date: Optional[str] = None
    ) -> DailySnapshot:
        """
        Save or update a daily snapshot for an account.

        Implements upsert behavior: creates new snapshot if none exists
        for the date, otherwise updates the existing one.

        Args:
            account_id: MT5 account number or string ID
            equity: Current equity value
            balance: Current balance value
            snapshot_date: Date string (YYYY-MM-DD), defaults to today

        Returns:
            Created or updated DailySnapshot object
        """
        return self.snapshots.save(
            account_id=account_id,
            equity=equity,
            balance=balance,
            snapshot_date=snapshot_date
        )

    def get_daily_snapshot(
        self,
        account_id: str,
        snapshot_date: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve a daily snapshot for an account.

        Args:
            account_id: MT5 account number
            snapshot_date: Date string (YYYY-MM-DD), defaults to today

        Returns:
            Dictionary with snapshot data or None if not found
        """
        return self.snapshots.get(account_id, snapshot_date)

    def get_latest_snapshot(self, account_id: str) -> Optional[DailySnapshot]:
        """
        Retrieve the most recent daily snapshot for an account.

        Args:
            account_id: MT5 account number or string ID

        Returns:
            Latest DailySnapshot object or None if not found
        """
        return self.snapshots.get_latest(account_id)

    def get_daily_drawdown(self, account_id: str) -> float:
        """
        Calculate current daily drawdown percentage.

        Args:
            account_id: MT5 account number

        Returns:
            Daily drawdown percentage (e.g., 2.5 for 2.5% drawdown)
            Returns 0.0 if account or snapshot not found
        """
        return self.snapshots.get_drawdown(account_id)

    def get_daily_start_balance(self, account_id: str) -> float:
        """
        Get the daily start balance for an account.

        Args:
            account_id: MT5 account number

        Returns:
            Daily start balance, or 0.0 if not found
        """
        return self.snapshots.get_start_balance(account_id)

    # ========================================================================
    # SQLite Methods - Trade Proposals (delegated to ProposalRepository)
    # ========================================================================

    def create_trade_proposal(
        self,
        bot_id: str,
        symbol: str,
        kelly_score: float,
        regime: str,
        proposed_lot_size: float
    ) -> TradeProposal:
        """
        Create a new trade proposal.

        Args:
            bot_id: Bot/strategy identifier
            symbol: Trading symbol
            kelly_score: Kelly criterion score
            regime: Market regime
            proposed_lot_size: Suggested position size

        Returns:
            Created TradeProposal object
        """
        return self.proposals.create(
            bot_id=bot_id,
            symbol=symbol,
            kelly_score=kelly_score,
            regime=regime,
            proposed_lot_size=proposed_lot_size
        )

    def update_trade_proposal(
        self,
        proposal_id: int,
        status: str
    ) -> Optional[TradeProposal]:
        """
        Update the status of a trade proposal.

        Args:
            proposal_id: Proposal ID
            status: New status (pending/approved/rejected)

        Returns:
            Updated TradeProposal object or None if not found
        """
        return self.proposals.update(proposal_id, status)

    # ========================================================================
    # SQLite Methods - Agent Tasks (delegated to TaskRepository)
    # ========================================================================

    def create_agent_task(
        self,
        agent_type: str,
        task_type: str,
        task_data: Dict[str, Any],
        status: str = 'pending'
    ) -> AgentTasks:
        """
        Create a new agent task record.

        Args:
            agent_type: Type of agent (analyst/quant/copilot)
            task_type: Type of task being performed
            task_data: Dictionary containing task details
            status: Initial task status (default: pending)

        Returns:
            Created AgentTasks object
        """
        return self.tasks.create(
            agent_type=agent_type,
            task_type=task_type,
            task_data=task_data,
            status=status
        )

    def update_agent_task(
        self,
        task_id: int,
        status: str,
        completed_at: Optional[datetime] = None
    ) -> Optional[AgentTasks]:
        """
        Update the status of an agent task.

        Args:
            task_id: Task ID
            status: New status (pending/in_progress/completed/failed)
            completed_at: Completion timestamp (defaults to now if status is completed)

        Returns:
            Updated AgentTasks object or None if not found
        """
        return self.tasks.update(task_id, status, completed_at)

    def get_agent_tasks(
        self,
        agent_type: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100
    ) -> List[AgentTasks]:
        """
        Retrieve agent tasks with optional filtering.

        Args:
            agent_type: Filter by agent type (optional)
            status: Filter by status (optional)
            limit: Maximum number of tasks to return

        Returns:
            List of AgentTasks objects
        """
        return self.tasks.get_all(
            agent_type=agent_type,
            status=status,
            limit=limit
        )

    # ========================================================================
    # SQLite Methods - Strategy Performance (delegated to StrategyRepository)
    # ========================================================================

    def create_strategy_performance(
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
        return self.strategies.create(
            strategy_name=strategy_name,
            backtest_results=backtest_results,
            kelly_score=kelly_score,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=total_trades
        )

    def get_strategy_performance(
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
        return self.strategies.get(
            strategy_name=strategy_name,
            min_kelly_score=min_kelly_score,
            min_sharpe_ratio=min_sharpe_ratio,
            limit=limit
        )

    def get_best_strategies(
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
        return self.strategies.get_best(limit=limit, order_by=order_by)

    def close_all_sessions(self):
        """Close all database sessions."""
        close_session()
