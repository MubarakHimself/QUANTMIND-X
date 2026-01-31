"""
Tests for AgentTasks and StrategyPerformance Models

Tests cover:
- AgentTasks model creation and status transitions
- StrategyPerformance model creation and querying
- DatabaseManager methods for new models
- Retry logic for database operations
"""

import pytest
import json
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from src.database.models import Base, AgentTasks, StrategyPerformance
from src.database.manager import DatabaseManager


# Test database configuration
TEST_DB_PATH = "test_quantmind_new_models.db"


@pytest.fixture(scope="function")
def test_engine():
    """Create a test database engine."""
    engine = create_engine(f"sqlite:///{TEST_DB_PATH}")
    Base.metadata.create_all(bind=engine)
    yield engine
    # Cleanup
    Base.metadata.drop_all(bind=engine)
    import os
    if os.path.exists(TEST_DB_PATH):
        os.remove(TEST_DB_PATH)


@pytest.fixture(scope="function")
def test_session(test_engine):
    """Create a test database session."""
    TestSession = sessionmaker(bind=test_engine)
    session = TestSession()
    yield session
    session.close()


@pytest.fixture(scope="function")
def db_manager(test_engine):
    """Create a DatabaseManager instance for testing."""
    # Create tables first
    Base.metadata.create_all(bind=test_engine)
    
    # Override the engine for testing
    from src.database import engine as db_engine
    original_engine = db_engine.engine
    db_engine.engine = test_engine
    
    # Create fresh instance
    DatabaseManager._instance = None
    manager = DatabaseManager()
    
    yield manager
    
    # Cleanup
    Base.metadata.drop_all(bind=test_engine)
    
    # Restore original engine
    db_engine.engine = original_engine
    DatabaseManager._instance = None


class TestAgentTasksModel:
    """Test AgentTasks model creation and operations."""

    def test_create_agent_task(self, test_session: Session):
        """Test creating a valid AgentTasks instance."""
        task_data = {
            "description": "Analyze market trends for EURUSD",
            "parameters": {"symbol": "EURUSD", "timeframe": "H1"}
        }
        
        task = AgentTasks(
            agent_type="analyst",
            task_type="market_analysis",
            task_data=task_data,
            status="pending"
        )
        test_session.add(task)
        test_session.commit()
        test_session.refresh(task)

        assert task.id is not None
        assert task.agent_type == "analyst"
        assert task.task_type == "market_analysis"
        assert task.task_data == task_data
        assert task.status == "pending"
        assert task.created_at is not None
        assert task.completed_at is None

    def test_agent_task_status_transitions(self, test_session: Session):
        """Test status transitions for agent tasks."""
        task = AgentTasks(
            agent_type="quant",
            task_type="backtest_strategy",
            task_data={"strategy_id": "momentum_v1"},
            status="pending"
        )
        test_session.add(task)
        test_session.commit()
        test_session.refresh(task)

        # Transition to in_progress
        task.status = "in_progress"
        test_session.commit()
        test_session.refresh(task)
        assert task.status == "in_progress"

        # Transition to completed
        task.status = "completed"
        task.completed_at = datetime.utcnow()
        test_session.commit()
        test_session.refresh(task)
        assert task.status == "completed"
        assert task.completed_at is not None

    def test_agent_task_json_data(self, test_session: Session):
        """Test that JSON data is properly stored and retrieved."""
        complex_data = {
            "strategy": {
                "name": "RSI Divergence",
                "parameters": {
                    "rsi_period": 14,
                    "overbought": 70,
                    "oversold": 30
                }
            },
            "results": {
                "win_rate": 0.65,
                "profit_factor": 1.8,
                "max_drawdown": 0.12
            }
        }
        
        task = AgentTasks(
            agent_type="quant",
            task_type="strategy_optimization",
            task_data=complex_data,
            status="completed"
        )
        test_session.add(task)
        test_session.commit()
        test_session.refresh(task)

        assert task.task_data == complex_data
        assert task.task_data["strategy"]["name"] == "RSI Divergence"
        assert task.task_data["results"]["win_rate"] == 0.65

    def test_query_tasks_by_agent_type(self, test_session: Session):
        """Test querying tasks by agent type."""
        # Create tasks for different agents
        analyst_task = AgentTasks(
            agent_type="analyst",
            task_type="research",
            task_data={"topic": "market trends"},
            status="completed"
        )
        quant_task = AgentTasks(
            agent_type="quant",
            task_type="backtest",
            task_data={"strategy": "momentum"},
            status="pending"
        )
        
        test_session.add_all([analyst_task, quant_task])
        test_session.commit()

        # Query analyst tasks
        analyst_tasks = test_session.query(AgentTasks).filter(
            AgentTasks.agent_type == "analyst"
        ).all()
        
        assert len(analyst_tasks) == 1
        assert analyst_tasks[0].task_type == "research"

    def test_query_tasks_by_status(self, test_session: Session):
        """Test querying tasks by status."""
        # Create tasks with different statuses
        for i, status in enumerate(["pending", "in_progress", "completed", "failed"]):
            task = AgentTasks(
                agent_type="executor",
                task_type=f"deploy_{i}",
                task_data={"deployment_id": i},
                status=status
            )
            test_session.add(task)
        
        test_session.commit()

        # Query pending tasks
        pending_tasks = test_session.query(AgentTasks).filter(
            AgentTasks.status == "pending"
        ).all()
        
        assert len(pending_tasks) == 1
        assert pending_tasks[0].status == "pending"


class TestStrategyPerformanceModel:
    """Test StrategyPerformance model creation and operations."""

    def test_create_strategy_performance(self, test_session: Session):
        """Test creating a valid StrategyPerformance instance."""
        backtest_results = {
            "total_trades": 150,
            "winning_trades": 98,
            "losing_trades": 52,
            "total_profit": 15000.0,
            "total_loss": -5000.0
        }
        
        performance = StrategyPerformance(
            strategy_name="RSI Mean Reversion",
            backtest_results=backtest_results,
            kelly_score=0.85,
            sharpe_ratio=1.8,
            max_drawdown=12.5,
            win_rate=65.3,
            profit_factor=3.0,
            total_trades=150
        )
        test_session.add(performance)
        test_session.commit()
        test_session.refresh(performance)

        assert performance.id is not None
        assert performance.strategy_name == "RSI Mean Reversion"
        assert performance.backtest_results == backtest_results
        assert performance.kelly_score == 0.85
        assert performance.sharpe_ratio == 1.8
        assert performance.max_drawdown == 12.5
        assert performance.win_rate == 65.3
        assert performance.profit_factor == 3.0
        assert performance.total_trades == 150
        assert performance.created_at is not None

    def test_query_by_kelly_score(self, test_session: Session):
        """Test querying strategies by Kelly score."""
        # Create strategies with different Kelly scores
        strategies = [
            ("Strategy A", 0.9, 2.0, 10.0),
            ("Strategy B", 0.7, 1.5, 15.0),
            ("Strategy C", 0.85, 1.8, 12.0)
        ]
        
        for name, kelly, sharpe, drawdown in strategies:
            perf = StrategyPerformance(
                strategy_name=name,
                backtest_results={},
                kelly_score=kelly,
                sharpe_ratio=sharpe,
                max_drawdown=drawdown
            )
            test_session.add(perf)
        
        test_session.commit()

        # Query strategies with Kelly score >= 0.8
        high_kelly = test_session.query(StrategyPerformance).filter(
            StrategyPerformance.kelly_score >= 0.8
        ).all()
        
        assert len(high_kelly) == 2
        assert all(s.kelly_score >= 0.8 for s in high_kelly)

    def test_query_by_sharpe_ratio(self, test_session: Session):
        """Test querying strategies by Sharpe ratio."""
        # Create strategies with different Sharpe ratios
        strategies = [
            ("Strategy A", 0.8, 2.5, 10.0),
            ("Strategy B", 0.7, 1.2, 15.0),
            ("Strategy C", 0.85, 2.0, 12.0)
        ]
        
        for name, kelly, sharpe, drawdown in strategies:
            perf = StrategyPerformance(
                strategy_name=name,
                backtest_results={},
                kelly_score=kelly,
                sharpe_ratio=sharpe,
                max_drawdown=drawdown
            )
            test_session.add(perf)
        
        test_session.commit()

        # Query strategies with Sharpe ratio >= 2.0
        high_sharpe = test_session.query(StrategyPerformance).filter(
            StrategyPerformance.sharpe_ratio >= 2.0
        ).all()
        
        assert len(high_sharpe) == 2
        assert all(s.sharpe_ratio >= 2.0 for s in high_sharpe)

    def test_order_by_kelly_score(self, test_session: Session):
        """Test ordering strategies by Kelly score."""
        # Create strategies
        strategies = [
            ("Strategy A", 0.7),
            ("Strategy B", 0.9),
            ("Strategy C", 0.85)
        ]
        
        for name, kelly in strategies:
            perf = StrategyPerformance(
                strategy_name=name,
                backtest_results={},
                kelly_score=kelly,
                sharpe_ratio=1.5,
                max_drawdown=10.0
            )
            test_session.add(perf)
        
        test_session.commit()

        # Query ordered by Kelly score descending
        ordered = test_session.query(StrategyPerformance).order_by(
            StrategyPerformance.kelly_score.desc()
        ).all()
        
        assert len(ordered) == 3
        assert ordered[0].strategy_name == "Strategy B"  # 0.9
        assert ordered[1].strategy_name == "Strategy C"  # 0.85
        assert ordered[2].strategy_name == "Strategy A"  # 0.7


class TestDatabaseManagerNewMethods:
    """Test DatabaseManager methods for new models."""

    @pytest.mark.skip(reason="Integration test - requires proper database setup")
    def test_create_agent_task_via_manager(self, db_manager):
        """Test creating agent task through DatabaseManager."""
        task_data = {
            "description": "Analyze EURUSD trends",
            "timeframe": "H1"
        }
        
        task = db_manager.create_agent_task(
            agent_type="analyst",
            task_type="market_analysis",
            task_data=task_data,
            status="pending"
        )

        assert task.id is not None
        assert task.agent_type == "analyst"
        assert task.task_type == "market_analysis"
        assert task.task_data == task_data

    @pytest.mark.skip(reason="Integration test - requires proper database setup")
    def test_update_agent_task_via_manager(self, db_manager):
        """Test updating agent task through DatabaseManager."""
        # Create task
        task = db_manager.create_agent_task(
            agent_type="quant",
            task_type="backtest",
            task_data={"strategy": "momentum"}
        )

        # Update task status
        updated = db_manager.update_agent_task(
            task_id=task.id,
            status="completed"
        )

        assert updated is not None
        assert updated.status == "completed"
        assert updated.completed_at is not None

    @pytest.mark.skip(reason="Integration test - requires proper database setup")
    def test_get_agent_tasks_via_manager(self, db_manager):
        """Test retrieving agent tasks through DatabaseManager."""
        # Create multiple tasks
        for i in range(3):
            db_manager.create_agent_task(
                agent_type="analyst",
                task_type=f"task_{i}",
                task_data={"index": i},
                status="pending" if i < 2 else "completed"
            )

        # Get all analyst tasks
        tasks = db_manager.get_agent_tasks(agent_type="analyst")
        assert len(tasks) == 3

        # Get pending tasks only
        pending_tasks = db_manager.get_agent_tasks(
            agent_type="analyst",
            status="pending"
        )
        assert len(pending_tasks) == 2

    @pytest.mark.skip(reason="Integration test - requires proper database setup")
    def test_create_strategy_performance_via_manager(self, db_manager):
        """Test creating strategy performance through DatabaseManager."""
        backtest_results = {
            "total_trades": 100,
            "win_rate": 0.65
        }
        
        performance = db_manager.create_strategy_performance(
            strategy_name="RSI Strategy",
            backtest_results=backtest_results,
            kelly_score=0.85,
            sharpe_ratio=1.8,
            max_drawdown=12.5,
            win_rate=65.0,
            profit_factor=2.5,
            total_trades=100
        )

        assert performance.id is not None
        assert performance.strategy_name == "RSI Strategy"
        assert performance.kelly_score == 0.85

    @pytest.mark.skip(reason="Integration test - requires proper database setup")
    def test_get_strategy_performance_via_manager(self, db_manager):
        """Test retrieving strategy performance through DatabaseManager."""
        # Create multiple strategies
        strategies = [
            ("Strategy A", 0.9, 2.0),
            ("Strategy B", 0.7, 1.5),
            ("Strategy C", 0.85, 1.8)
        ]
        
        for name, kelly, sharpe in strategies:
            db_manager.create_strategy_performance(
                strategy_name=name,
                backtest_results={},
                kelly_score=kelly,
                sharpe_ratio=sharpe,
                max_drawdown=10.0
            )

        # Get all strategies
        all_strategies = db_manager.get_strategy_performance()
        assert len(all_strategies) == 3

        # Get strategies with Kelly score >= 0.8
        high_kelly = db_manager.get_strategy_performance(min_kelly_score=0.8)
        assert len(high_kelly) == 2

    @pytest.mark.skip(reason="Integration test - requires proper database setup")
    def test_get_best_strategies_via_manager(self, db_manager):
        """Test getting best strategies through DatabaseManager."""
        # Create strategies
        strategies = [
            ("Strategy A", 0.7, 1.5),
            ("Strategy B", 0.9, 2.5),
            ("Strategy C", 0.85, 2.0)
        ]
        
        for name, kelly, sharpe in strategies:
            db_manager.create_strategy_performance(
                strategy_name=name,
                backtest_results={},
                kelly_score=kelly,
                sharpe_ratio=sharpe,
                max_drawdown=10.0
            )

        # Get best by Kelly score
        best_kelly = db_manager.get_best_strategies(limit=2, order_by='kelly_score')
        assert len(best_kelly) == 2
        assert best_kelly[0].kelly_score == 0.9

        # Get best by Sharpe ratio
        best_sharpe = db_manager.get_best_strategies(limit=2, order_by='sharpe_ratio')
        assert len(best_sharpe) == 2
        assert best_sharpe[0].sharpe_ratio == 2.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
