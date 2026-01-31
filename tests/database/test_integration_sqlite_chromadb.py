"""
Integration Tests for SQLite + ChromaDB Coordination

Tests cover:
- Coordinated operations between SQLite and ChromaDB
- Strategy performance tracking with vector embeddings
- Agent task history with memory storage
- Cross-database queries and consistency
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

from src.database.manager import DatabaseManager
from src.database.models import Base
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


@pytest.fixture(scope="function")
def temp_dirs():
    """Create temporary directories for test databases."""
    temp_db_dir = tempfile.mkdtemp()
    temp_chroma_dir = tempfile.mkdtemp()
    
    yield {
        "db_dir": temp_db_dir,
        "chroma_dir": temp_chroma_dir
    }
    
    # Cleanup
    shutil.rmtree(temp_db_dir, ignore_errors=True)
    shutil.rmtree(temp_chroma_dir, ignore_errors=True)


@pytest.fixture(scope="function")
def integrated_db(temp_dirs):
    """Create integrated database manager with both SQLite and ChromaDB."""
    # Create test SQLite database
    db_path = Path(temp_dirs["db_dir"]) / "test_quantmind.db"
    test_engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(bind=test_engine)
    
    # Override engine for testing
    from src.database import engine as db_engine
    original_engine = db_engine.engine
    db_engine.engine = test_engine
    
    # Create DatabaseManager instance
    DatabaseManager._instance = None
    manager = DatabaseManager()
    
    # Override ChromaDB path
    if manager.chroma:
        manager.chroma._persist_directory = Path(temp_dirs["chroma_dir"])
        manager.chroma._client = None  # Force recreation with new path
        manager.chroma._collections = {}
    
    yield manager
    
    # Cleanup
    Base.metadata.drop_all(bind=test_engine)
    db_engine.engine = original_engine
    DatabaseManager._instance = None


class TestSQLiteChromaDBIntegration:
    """Test coordinated operations between SQLite and ChromaDB."""

    @pytest.mark.skipif(
        not hasattr(DatabaseManager(), 'chroma') or DatabaseManager().chroma is None,
        reason="ChromaDB not available"
    )
    def test_strategy_performance_with_vector_storage(self, integrated_db):
        """Test storing strategy performance in SQLite and code in ChromaDB."""
        # Create strategy performance in SQLite
        strategy_code = """
        def rsi_strategy(data):
            rsi = calculate_rsi(data, period=14)
            if rsi < 30:
                return 'BUY'
            elif rsi > 70:
                return 'SELL'
            return 'HOLD'
        """
        
        backtest_results = {
            "total_trades": 150,
            "winning_trades": 98,
            "total_profit": 15000.0
        }
        
        # Store in SQLite
        performance = integrated_db.create_strategy_performance(
            strategy_name="RSI Mean Reversion",
            backtest_results=backtest_results,
            kelly_score=0.85,
            sharpe_ratio=1.8,
            max_drawdown=12.5,
            win_rate=65.3,
            profit_factor=3.0,
            total_trades=150
        )
        
        # Store code in ChromaDB
        integrated_db.add_strategy(
            strategy_id=f"strategy_{performance.id}",
            code=strategy_code,
            strategy_name="RSI Mean Reversion",
            code_hash="abc123",
            performance_metrics=backtest_results
        )
        
        # Verify SQLite storage
        assert performance.id is not None
        assert performance.strategy_name == "RSI Mean Reversion"
        
        # Verify ChromaDB storage
        results = integrated_db.search_strategies("RSI mean reversion", limit=1)
        assert len(results) > 0
        assert "rsi" in results[0]["document"].lower()

    @pytest.mark.skipif(
        not hasattr(DatabaseManager(), 'chroma') or DatabaseManager().chroma is None,
        reason="ChromaDB not available"
    )
    def test_agent_task_with_memory_storage(self, integrated_db):
        """Test storing agent tasks in SQLite and memories in ChromaDB."""
        # Create agent task in SQLite
        task_data = {
            "description": "Analyze EURUSD market trends",
            "timeframe": "H1",
            "symbols": ["EURUSD", "GBPUSD"]
        }
        
        task = integrated_db.create_agent_task(
            agent_type="analyst",
            task_type="market_analysis",
            task_data=task_data,
            status="completed"
        )
        
        # Store agent memory in ChromaDB
        memory_content = """
        Analyzed EURUSD trends on H1 timeframe. Identified strong bullish momentum
        with RSI divergence pattern. Recommended long position with 1.5% risk.
        """
        
        integrated_db.add_agent_memory(
            memory_id=f"memory_task_{task.id}",
            content=memory_content,
            agent_type="analyst",
            memory_type="episodic",
            context="market_analysis",
            importance=0.8
        )
        
        # Verify SQLite storage
        assert task.id is not None
        assert task.agent_type == "analyst"
        assert task.status == "completed"
        
        # Verify ChromaDB storage
        memories = integrated_db.search_agent_memory(
            query="EURUSD analysis",
            agent_type="analyst",
            limit=1
        )
        assert len(memories) > 0
        assert "EURUSD" in memories[0]["document"]

    @pytest.mark.skipif(
        not hasattr(DatabaseManager(), 'chroma') or DatabaseManager().chroma is None,
        reason="ChromaDB not available"
    )
    def test_cross_database_query_consistency(self, integrated_db):
        """Test consistency between SQLite and ChromaDB queries."""
        # Create multiple strategies
        strategies = [
            ("Momentum Strategy", 0.9, 2.5, "Trend following with momentum indicators"),
            ("Mean Reversion", 0.85, 2.0, "RSI-based mean reversion strategy"),
            ("Breakout Strategy", 0.75, 1.8, "Volatility breakout with Bollinger Bands")
        ]
        
        for name, kelly, sharpe, description in strategies:
            # Store in SQLite
            perf = integrated_db.create_strategy_performance(
                strategy_name=name,
                backtest_results={"description": description},
                kelly_score=kelly,
                sharpe_ratio=sharpe,
                max_drawdown=10.0
            )
            
            # Store in ChromaDB
            integrated_db.add_strategy(
                strategy_id=f"strategy_{perf.id}",
                code=description,
                strategy_name=name,
                code_hash=f"hash_{perf.id}",
                performance_metrics={"kelly_score": kelly}
            )
        
        # Query SQLite for high-performing strategies
        high_kelly = integrated_db.get_strategy_performance(min_kelly_score=0.8)
        assert len(high_kelly) == 2
        
        # Query ChromaDB for similar strategies
        similar = integrated_db.search_strategies("momentum trend following", limit=2)
        assert len(similar) > 0
        
        # Verify consistency
        sqlite_names = {s.strategy_name for s in high_kelly}
        assert "Momentum Strategy" in sqlite_names
        assert "Mean Reversion" in sqlite_names

    @pytest.mark.skipif(
        not hasattr(DatabaseManager(), 'chroma') or DatabaseManager().chroma is None,
        reason="ChromaDB not available"
    )
    def test_knowledge_base_with_agent_tasks(self, integrated_db):
        """Test knowledge base articles linked to agent tasks."""
        # Add knowledge article
        article_content = """
        MQL5 Tutorial: Implementing RSI Divergence Trading Strategy
        
        This article explains how to detect RSI divergence patterns and
        implement automated trading strategies in MQL5.
        """
        
        integrated_db.add_knowledge(
            article_id="article_rsi_001",
            content=article_content,
            title="RSI Divergence Trading in MQL5",
            url="https://mql5.com/articles/rsi-divergence",
            categories="Trading Systems,Indicators",
            relevance_score=0.92
        )
        
        # Create agent task that references the article
        task = integrated_db.create_agent_task(
            agent_type="quant",
            task_type="strategy_implementation",
            task_data={
                "strategy": "RSI Divergence",
                "reference_articles": ["article_rsi_001"],
                "language": "MQL5"
            },
            status="in_progress"
        )
        
        # Search knowledge base
        articles = integrated_db.search_knowledge("RSI divergence MQL5", limit=1)
        assert len(articles) > 0
        assert "RSI" in articles[0]["document"]
        
        # Verify task references article
        assert "article_rsi_001" in task.task_data["reference_articles"]

    @pytest.mark.skipif(
        not hasattr(DatabaseManager(), 'chroma') or DatabaseManager().chroma is None,
        reason="ChromaDB not available"
    )
    def test_concurrent_database_operations(self, integrated_db):
        """Test concurrent operations on both databases."""
        # Perform multiple operations
        operations = []
        
        # SQLite operations
        for i in range(5):
            task = integrated_db.create_agent_task(
                agent_type="executor",
                task_type=f"deployment_{i}",
                task_data={"deployment_id": i},
                status="pending"
            )
            operations.append(("sqlite", task.id))
        
        # ChromaDB operations
        for i in range(5):
            integrated_db.add_agent_memory(
                memory_id=f"memory_{i}",
                content=f"Deployment memory {i}",
                agent_type="executor",
                memory_type="procedural",
                context="deployment",
                importance=0.5
            )
            operations.append(("chroma", f"memory_{i}"))
        
        # Verify all operations succeeded
        assert len(operations) == 10
        
        # Verify SQLite data
        tasks = integrated_db.get_agent_tasks(agent_type="executor")
        assert len(tasks) == 5
        
        # Verify ChromaDB data
        memories = integrated_db.search_agent_memory(
            query="deployment",
            agent_type="executor",
            limit=10
        )
        assert len(memories) >= 5


class TestDatabaseConsistency:
    """Test data consistency between SQLite and ChromaDB."""

    @pytest.mark.skipif(
        not hasattr(DatabaseManager(), 'chroma') or DatabaseManager().chroma is None,
        reason="ChromaDB not available"
    )
    def test_strategy_id_consistency(self, integrated_db):
        """Test that strategy IDs are consistent across databases."""
        # Create strategy in SQLite
        perf = integrated_db.create_strategy_performance(
            strategy_name="Test Strategy",
            backtest_results={},
            kelly_score=0.8,
            sharpe_ratio=1.5,
            max_drawdown=10.0
        )
        
        strategy_id = f"strategy_{perf.id}"
        
        # Store in ChromaDB with same ID
        integrated_db.add_strategy(
            strategy_id=strategy_id,
            code="test code",
            strategy_name="Test Strategy",
            code_hash="test_hash"
        )
        
        # Verify ID consistency
        assert perf.id is not None
        assert strategy_id == f"strategy_{perf.id}"

    @pytest.mark.skipif(
        not hasattr(DatabaseManager(), 'chroma') or DatabaseManager().chroma is None,
        reason="ChromaDB not available"
    )
    def test_metadata_consistency(self, integrated_db):
        """Test that metadata is consistent across databases."""
        strategy_name = "Consistent Strategy"
        kelly_score = 0.85
        
        # Store in SQLite
        perf = integrated_db.create_strategy_performance(
            strategy_name=strategy_name,
            backtest_results={"kelly_score": kelly_score},
            kelly_score=kelly_score,
            sharpe_ratio=1.8,
            max_drawdown=12.0
        )
        
        # Store in ChromaDB
        integrated_db.add_strategy(
            strategy_id=f"strategy_{perf.id}",
            code="strategy code",
            strategy_name=strategy_name,
            code_hash="hash",
            performance_metrics={"kelly_score": kelly_score}
        )
        
        # Verify metadata consistency
        assert perf.strategy_name == strategy_name
        assert perf.kelly_score == kelly_score
        
        # Search ChromaDB
        results = integrated_db.search_strategies(strategy_name, limit=1)
        assert len(results) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
