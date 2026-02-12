"""
Unified Database Manager

Provides a unified interface to both SQLite (via SQLAlchemy) and ChromaDB.
Implements singleton pattern for application-wide database access with
automatic retry logic and connection management.
"""

import os
from datetime import datetime, date
from typing import Optional, List, Dict, Any
from contextlib import contextmanager

from .engine import engine, Session, init_database as init_sqlite_db
from .models import PropFirmAccount, DailySnapshot, TradeProposal, AgentTasks, StrategyPerformance, BrokerRegistry, HouseMoneyState, BotCircuitBreaker, Base
from .retry import DatabaseConnectionManager, create_connection_manager, with_retry


class DatabaseManager:
    """
    Unified database manager for QuantMind Hybrid Core.

    Provides methods for:
    - SQLite operations (accounts, snapshots, proposals)
    - ChromaDB operations (strategies, knowledge, patterns)

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

        # Initialize ChromaDB client
        try:
            from .chroma_client import ChromaDBClient
            self.chroma = ChromaDBClient()
        except ImportError as e:
            self.chroma = None

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
    # SQLite Methods - Prop Firm Accounts
    # ========================================================================

    def get_prop_account(self, account_id: str) -> Optional[PropFirmAccount]:
        """
        Retrieve a prop firm account by account ID.

        Args:
            account_id: MT5 account number

        Returns:
            PropFirmAccount object or None if not found
        """
        with self.get_session() as session:
            account = session.query(PropFirmAccount).filter(
                PropFirmAccount.account_id == account_id
            ).first()
            if account is not None:
                # Load all attributes to avoid detached instance errors
                session.expunge(account)
            return account

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
        with self.get_session() as session:
            account = PropFirmAccount(
                firm_name=firm_name,
                account_id=account_id,
                daily_loss_limit_pct=daily_loss_limit_pct,
                hard_stop_buffer_pct=hard_stop_buffer_pct,
                target_profit_pct=target_profit_pct,
                min_trading_days=min_trading_days
            )
            session.add(account)
            session.flush()
            session.refresh(account)
            session.expunge(account)
            return account

    # ========================================================================
    # SQLite Methods - Daily Snapshots
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
        if snapshot_date is None:
            snapshot_date = date.today().isoformat()

        with self.get_session() as session:
            # Get account by account_id string
            account = session.query(PropFirmAccount).filter(
                PropFirmAccount.account_id == str(account_id)
            ).first()

            # Auto-create account if it doesn't exist
            if account is None:
                account = PropFirmAccount(
                    firm_name="Unknown",
                    account_id=str(account_id)
                )
                session.add(account)
                session.flush()

            # Check if snapshot exists for this date
            snapshot = session.query(DailySnapshot).filter(
                DailySnapshot.account_id == account.id,
                DailySnapshot.date == snapshot_date
            ).first()

            if snapshot is None:
                # Create new snapshot
                snapshot = DailySnapshot(
                    account_id=account.id,
                    date=snapshot_date,
                    daily_start_balance=balance,
                    high_water_mark=max(equity, balance),
                    current_equity=equity,
                    daily_drawdown_pct=0.0,
                    is_breached=False
                )
                session.add(snapshot)
            else:
                # Update existing snapshot
                snapshot.current_equity = equity
                snapshot.high_water_mark = max(snapshot.high_water_mark, equity)
                snapshot.snapshot_timestamp = datetime.utcnow()

                # Recalculate drawdown
                if snapshot.daily_start_balance > 0:
                    snapshot.daily_drawdown_pct = (
                        (snapshot.daily_start_balance - equity) /
                        snapshot.daily_start_balance * 100
                    )

            session.flush()
            session.refresh(snapshot)
            session.expunge(snapshot)
            return snapshot

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
        if snapshot_date is None:
            snapshot_date = date.today().isoformat()

        with self.get_session() as session:
            account = session.query(PropFirmAccount).filter(
                PropFirmAccount.account_id == str(account_id)
            ).first()

            if account is None:
                return None

            snapshot = session.query(DailySnapshot).filter(
                DailySnapshot.account_id == account.id,
                DailySnapshot.date == snapshot_date
            ).first()

            if snapshot is None:
                return None

            # Return dictionary to avoid DetachedInstanceError
            return {
                'id': snapshot.id,
                'account_id': snapshot.account_id,
                'date': snapshot.date,
                'daily_start_balance': snapshot.daily_start_balance,
                'high_water_mark': snapshot.high_water_mark,
                'current_equity': snapshot.current_equity,
                'daily_drawdown_pct': snapshot.daily_drawdown_pct,
                'is_breached': snapshot.is_breached,
                'snapshot_timestamp': snapshot.snapshot_timestamp
            }

    def get_latest_snapshot(self, account_id: str) -> Optional[DailySnapshot]:
        """
        Retrieve the most recent daily snapshot for an account.

        Args:
            account_id: MT5 account number or string ID

        Returns:
            Latest DailySnapshot object or None if not found
        """
        with self.get_session() as session:
            account = session.query(PropFirmAccount).filter(
                PropFirmAccount.account_id == str(account_id)
            ).first()

            if account is None:
                return None

            snapshot = session.query(DailySnapshot).filter(
                DailySnapshot.account_id == account.id
            ).order_by(DailySnapshot.date.desc()).first()

            if snapshot is None:
                return None

            # Detach from session to avoid DetachedInstanceError
            session.expunge(snapshot)
            return snapshot

    def get_daily_drawdown(self, account_id: str) -> float:
        """
        Calculate current daily drawdown percentage.

        Args:
            account_id: MT5 account number

        Returns:
            Daily drawdown percentage (e.g., 2.5 for 2.5% drawdown)
            Returns 0.0 if account or snapshot not found
        """
        snapshot = self.get_daily_snapshot(account_id)
        if snapshot is None:
            return 0.0

        return snapshot['daily_drawdown_pct']

    def get_daily_start_balance(self, account_id: str) -> float:
        """
        Get the daily start balance for an account.

        Args:
            account_id: MT5 account number

        Returns:
            Daily start balance, or 0.0 if not found
        """
        snapshot = self.get_daily_snapshot(account_id)
        if snapshot is None:
            return 0.0

        return snapshot['daily_start_balance']

    # ========================================================================
    # SQLite Methods - Trade Proposals
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
        with self.get_session() as session:
            proposal = TradeProposal(
                bot_id=bot_id,
                symbol=symbol,
                kelly_score=kelly_score,
                regime=regime,
                proposed_lot_size=proposed_lot_size,
                status='pending'
            )
            session.add(proposal)
            session.flush()
            session.refresh(proposal)
            session.expunge(proposal)
            return proposal

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
        with self.get_session() as session:
            proposal = session.query(TradeProposal).filter(
                TradeProposal.id == proposal_id
            ).first()

            if proposal is None:
                return None

            proposal.status = status
            proposal.reviewed_at = datetime.utcnow()

            session.flush()
            session.refresh(proposal)
            session.expunge(proposal)
            return proposal

    # ========================================================================
    # SQLite Methods - Agent Tasks
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
        with self.get_session() as session:
            task = AgentTasks(
                agent_type=agent_type,
                task_type=task_type,
                task_data=task_data,
                status=status
            )
            session.add(task)
            session.flush()
            session.refresh(task)
            session.expunge(task)
            return task

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
        with self.get_session() as session:
            task = session.query(AgentTasks).filter(
                AgentTasks.id == task_id
            ).first()

            if task is None:
                return None

            task.status = status
            if status == 'completed' and completed_at is None:
                task.completed_at = datetime.utcnow()
            elif completed_at is not None:
                task.completed_at = completed_at

            session.flush()
            session.refresh(task)
            session.expunge(task)
            return task

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
        with self.get_session() as session:
            query = session.query(AgentTasks)

            if agent_type is not None:
                query = query.filter(AgentTasks.agent_type == agent_type)

            if status is not None:
                query = query.filter(AgentTasks.status == status)

            query = query.order_by(AgentTasks.created_at.desc()).limit(limit)

            tasks = query.all()
            # Expunge all tasks to avoid detached instance errors
            for task in tasks:
                session.expunge(task)
            return tasks

    # ========================================================================
    # SQLite Methods - Strategy Performance
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
        with self.get_session() as session:
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
            session.add(performance)
            session.flush()
            session.refresh(performance)
            session.expunge(performance)
            return performance

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
        with self.get_session() as session:
            query = session.query(StrategyPerformance)

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
                session.expunge(perf)
            return performances

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
        with self.get_session() as session:
            query = session.query(StrategyPerformance)

            if order_by == 'sharpe_ratio':
                query = query.order_by(StrategyPerformance.sharpe_ratio.desc())
            else:
                query = query.order_by(StrategyPerformance.kelly_score.desc())

            query = query.limit(limit)

            strategies = query.all()
            # Expunge all strategies to avoid detached instance errors
            for strategy in strategies:
                session.expunge(strategy)
            return strategies

    # ========================================================================
    # ChromaDB Methods
    # ========================================================================

    def search_strategies(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search trading strategies by semantic similarity.

        Args:
            query: Search query text
            limit: Maximum results to return

        Returns:
            List of matching strategies with metadata
        """
        if self.chroma is None:
            return []
        return self.chroma.search_strategies(query, limit)

    def add_strategy(
        self,
        strategy_id: str,
        code: str,
        strategy_name: str,
        code_hash: str,
        performance_metrics: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add a trading strategy to the vector store.

        Args:
            strategy_id: Unique identifier
            code: Strategy code
            strategy_name: Human-readable name
            code_hash: Hash of the code
            performance_metrics: Optional performance data
        """
        if self.chroma is None:
            raise RuntimeError("ChromaDB not initialized")
        self.chroma.add_strategy(strategy_id, code, strategy_name, code_hash, performance_metrics)

    def search_knowledge(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search knowledge base by semantic similarity.

        Args:
            query: Search query text
            limit: Maximum results to return

        Returns:
            List of matching knowledge articles
        """
        if self.chroma is None:
            return []
        return self.chroma.search_knowledge(query, limit)

    def add_knowledge(
        self,
        article_id: str,
        content: str,
        title: str,
        url: str,
        categories: str,
        relevance_score: float = 0.5
    ) -> None:
        """
        Add knowledge article to the vector store.

        Args:
            article_id: Unique identifier
            content: Article content
            title: Article title
            url: Article URL
            categories: Comma-separated categories
            relevance_score: Relevance score (0-1)
        """
        if self.chroma is None:
            raise RuntimeError("ChromaDB not initialized")
        self.chroma.add_knowledge(article_id, content, title, url, categories, relevance_score)

    def search_patterns(
        self,
        query: str,
        limit: int = 10,
        where: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search market patterns by semantic similarity.

        Args:
            query: Search query text
            limit: Maximum results to return
            where: Optional metadata filter

        Returns:
            List of matching market patterns
        """
        if self.chroma is None:
            return []
        return self.chroma.search_patterns(query, limit, where)

    def add_pattern(
        self,
        pattern_id: str,
        description: str,
        pattern_type: str,
        volatility_level: str,
        timestamp: Optional[str] = None
    ) -> None:
        """
        Add market pattern to the vector store.

        Args:
            pattern_id: Unique identifier
            description: Pattern description
            pattern_type: Type of pattern
            volatility_level: Market volatility (low, medium, high)
            timestamp: ISO timestamp
        """
        if self.chroma is None:
            raise RuntimeError("ChromaDB not initialized")
        self.chroma.add_market_pattern(pattern_id, description, pattern_type, volatility_level, timestamp)

    def add_agent_memory(
        self,
        memory_id: str,
        content: str,
        agent_type: str,
        memory_type: str,
        context: str,
        importance: float = 0.5
    ) -> None:
        """
        Add agent memory to the vector store.

        Args:
            memory_id: Unique identifier
            content: Memory content
            agent_type: Type of agent (analyst/quant/executor)
            memory_type: Type of memory (semantic/episodic/procedural)
            context: Context in which memory was created
            importance: Importance score (0-1)
        """
        if self.chroma is None:
            raise RuntimeError("ChromaDB not initialized")
        self.chroma.add_agent_memory(memory_id, content, agent_type, memory_type, context, importance)

    def search_agent_memory(
        self,
        query: str,
        agent_type: Optional[str] = None,
        memory_type: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search agent memories by semantic similarity.

        Args:
            query: Search query text
            agent_type: Filter by agent type (optional)
            memory_type: Filter by memory type (optional)
            limit: Maximum results to return

        Returns:
            List of matching agent memories
        """
        if self.chroma is None:
            return []

        where = {}
        if agent_type is not None:
            where['agent_type'] = agent_type
        if memory_type is not None:
            where['memory_type'] = memory_type

        return self.chroma.search_patterns(query, limit, where if where else None)

    def close_all_sessions(self):
        """Close all database sessions."""
        close_session()
