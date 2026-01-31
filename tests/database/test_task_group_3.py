#!/usr/bin/env python3
"""
Focused tests for Task Groups 1, 2, and 3 - QuantMind Hybrid Core v7.

Tests cover:
- Task Group 1: SQLite Models (PropFirmAccount, DailySnapshot, TradeProposal)
- Task Group 2: ChromaDB Collections and Embeddings
- Task Group 3: DatabaseManager unified interface

Run only these tests:
    pytest tests/database/test_task_group_3.py -v
"""

import pytest
import sys
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
import uuid

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# Task Group 1: SQLite Model Tests
# =============================================================================

class TestPropFirmAccountModel:
    """Test PropFirmAccount model creation and validation."""

    def test_create_prop_firm_account(self):
        """Test creating a PropFirmAccount with required fields."""
        from src.database.models import PropFirmAccount

        account = PropFirmAccount(
            firm_name="MyForexFunds",
            account_id="12345",
            daily_loss_limit_pct=5.0,
            hard_stop_buffer_pct=1.0,
            target_profit_pct=8.0,
            min_trading_days=5
        )

        assert account.firm_name == "MyForexFunds"
        assert account.account_id == "12345"
        assert account.daily_loss_limit_pct == 5.0
        assert account.hard_stop_buffer_pct == 1.0
        assert account.target_profit_pct == 8.0
        assert account.min_trading_days == 5

    def test_prop_firm_account_relationships(self):
        """Test PropFirmAccount relationships to snapshots and proposals."""
        from src.database.models import PropFirmAccount

        account = PropFirmAccount(
            firm_name="TopTierTrader",
            account_id="67890"
        )

        # Test relationships are initialized
        assert hasattr(account, 'daily_snapshots')
        assert hasattr(account, 'trade_proposals')


class TestDailySnapshotModel:
    """Test DailySnapshot model with unique constraint."""

    def test_daily_snapshot_creation(self):
        """Test creating a DailySnapshot with all required fields."""
        from src.database.models import DailySnapshot

        snapshot = DailySnapshot(
            account_id=1,
            date="2024-01-30",
            daily_start_balance=100000.0,
            high_water_mark=101000.0,
            current_equity=100500.0,
            daily_drawdown_pct=0.5,
            is_breached=False
        )

        assert snapshot.account_id == 1
        assert snapshot.date == "2024-01-30"
        assert snapshot.daily_start_balance == 100000.0
        assert snapshot.high_water_mark == 101000.0
        assert snapshot.current_equity == 100500.0
        assert snapshot.daily_drawdown_pct == 0.5
        assert snapshot.is_breached is False

    def test_daily_snapshot_breach_flag(self):
        """Test is_breached flag for limit breach tracking."""
        from src.database.models import DailySnapshot

        # Breached snapshot
        breached_snapshot = DailySnapshot(
            account_id=1,
            date="2024-01-30",
            daily_start_balance=100000.0,
            high_water_mark=100000.0,
            current_equity=95000.0,
            daily_drawdown_pct=5.0,
            is_breached=True
        )

        assert breached_snapshot.is_breached is True


class TestTradeProposalModel:
    """Test TradeProposal model status transitions."""

    def test_trade_proposal_creation(self):
        """Test creating a TradeProposal with pending status."""
        from src.database.models import TradeProposal

        proposal = TradeProposal(
            bot_id="rsi_bot",
            symbol="EURUSD",
            kelly_score=0.85,
            regime="trend_following",
            proposed_lot_size=0.1,
            status="pending"
        )

        assert proposal.bot_id == "rsi_bot"
        assert proposal.symbol == "EURUSD"
        assert proposal.kelly_score == 0.85
        assert proposal.status == "pending"
        assert proposal.reviewed_at is None

    def test_trade_proposal_status_transitions(self):
        """Test proposal status transitions: pending -> approved/rejected."""
        from src.database.models import TradeProposal

        proposal = TradeProposal(
            bot_id="ma_crossover",
            symbol="GBPUSD",
            kelly_score=0.75,
            regime="mean_reversion",
            proposed_lot_size=0.2,
            status="pending"
        )

        # Test approve
        proposal.status = "approved"
        proposal.reviewed_at = datetime.utcnow()
        assert proposal.status == "approved"
        assert proposal.reviewed_at is not None

        # Test reject
        proposal.status = "rejected"
        assert proposal.status == "rejected"


# =============================================================================
# Task Group 2: ChromaDB Tests
# =============================================================================

class TestChromaDBCollections:
    """Test ChromaDB collection creation and initialization."""

    @pytest.fixture
    def temp_chroma_path(self):
        """Create temporary directory for test ChromaDB data."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def chroma_client(self, temp_chroma_path):
        """Create ChromaDB client with temporary storage."""
        try:
            from src.database.chroma_client import ChromaDBClient
        except ImportError:
            pytest.skip("ChromaDB not installed")

        client = ChromaDBClient(persist_directory=Path(temp_chroma_path))
        return client

    def test_collection_creation(self, chroma_client):
        """Test all three required collections are created."""
        # Access collections to trigger creation
        _ = chroma_client.strategies_collection
        _ = chroma_client.knowledge_collection
        _ = chroma_client.patterns_collection

        collections = chroma_client.list_collections()

        assert "quantmind_strategies" in collections
        assert "quantmind_knowledge" in collections
        assert "market_patterns" in collections

    def test_strategies_collection_metadata(self, chroma_client):
        """Test strategies collection accepts strategy metadata."""
        chroma_client.add_strategy(
            strategy_id="test_strategy_001",
            code="RSI strategy with entry and exit logic",
            strategy_name="RSI Basic",
            code_hash="abc123",
            performance_metrics={"win_rate": 0.6, "profit_factor": 1.5}
        )

        results = chroma_client.search_strategies("RSI strategy", limit=5)
        assert len(results) > 0
        assert results[0]["id"] == "test_strategy_001"

    def test_knowledge_collection_metadata(self, chroma_client):
        """Test knowledge collection accepts article metadata."""
        chroma_client.add_knowledge(
            article_id="article_001",
            content="Article about RSI indicator usage",
            title="RSI Trading Strategies",
            url="https://example.com/rsi",
            categories="indicators,trading",
            relevance_score=0.9
        )

        results = chroma_client.search_knowledge("RSI indicator", limit=5)
        assert len(results) > 0
        assert results[0]["id"] == "article_001"

    def test_patterns_collection_metadata(self, chroma_client):
        """Test patterns collection accepts pattern metadata."""
        chroma_client.add_market_pattern(
            pattern_id="pattern_001",
            description="Bullish trend with low volatility",
            pattern_type="trend_following",
            volatility_level="low",
            timestamp=datetime.utcnow().isoformat() + "Z"
        )

        results = chroma_client.search_patterns("trend pattern", limit=5)
        assert len(results) > 0
        assert results[0]["id"] == "pattern_001"

    def test_collection_stats(self, chroma_client):
        """Test get_collection_stats returns correct information."""
        stats = chroma_client.get_collection_stats()

        assert "persist_directory" in stats
        assert "embedding_model" in stats
        assert stats["embedding_dimension"] == 384
        assert stats["similarity"] == "cosine"
        assert "collections" in stats


# =============================================================================
# Task Group 3: DatabaseManager Tests
# =============================================================================

class TestDatabaseManagerSQLite:
    """Test DatabaseManager SQLite wrapper methods."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset singleton before and after each test."""
        import src.database.manager
        src.database.manager.DatabaseManager._instance = None
        yield
        src.database.manager.DatabaseManager._instance = None

    @pytest.fixture
    def temp_db_path(self):
        """Create temporary directory for test database."""
        temp_dir = tempfile.mkdtemp()
        data_dir = Path(temp_dir) / "data" / "db"
        data_dir.mkdir(parents=True)
        yield data_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def db_manager(self, temp_db_path, monkeypatch):
        """Create DatabaseManager with temporary database path."""
        # Use unique DB file per test
        unique_id = str(uuid.uuid4())[:8]
        temp_db_file = temp_db_path / f"quantmind_{unique_id}.db"

        import src.database.engine
        import src.database.manager

        # Patch DB path
        monkeypatch.setattr(src.database.engine, "DB_ABSOLUTE_PATH", str(temp_db_file))

        # Recreate engine with new path
        from sqlalchemy import create_engine
        from sqlalchemy.pool import StaticPool
        from sqlalchemy.orm import scoped_session, sessionmaker
        from sqlalchemy import event

        temp_engine = create_engine(
            f'sqlite:///{temp_db_file}',
            connect_args={'check_same_thread': False},
            poolclass=StaticPool,
            echo=False,
        )

        @event.listens_for(temp_engine, "connect")
        def set_sqlite_pragma(dbapi_conn, _):
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

        # Monkey patch the engine and Session
        monkeypatch.setattr(src.database.engine, "engine", temp_engine)
        temp_session_factory = sessionmaker(bind=temp_engine)
        temp_session = scoped_session(temp_session_factory)
        monkeypatch.setattr(src.database.engine, "Session", temp_session)

        # Import models and create tables with temp engine
        from src.database import models
        models.Base.metadata.create_all(bind=temp_engine)

        # Return fresh DatabaseManager instance
        src.database.manager.DatabaseManager._instance = None
        return src.database.manager.DatabaseManager()

    def test_get_prop_account(self, db_manager):
        """Test get_prop_account retrieves account by account_id."""
        # Create test account with unique ID
        account_id = f"test_{uuid.uuid4().hex[:8]}"
        account = db_manager.create_prop_account(
            account_id=account_id,
            firm_name="MyForexFunds",
            daily_loss_limit_pct=5.0
        )

        # Retrieve account
        retrieved = db_manager.get_prop_account(account_id)
        assert retrieved is not None
        assert retrieved.account_id == account_id
        assert retrieved.firm_name == "MyForexFunds"
        assert retrieved.daily_loss_limit_pct == 5.0

    def test_save_daily_snapshot_upsert(self, db_manager):
        """Test save_daily_snapshot implements upsert behavior."""
        # Create account with unique ID
        account_id = f"test_{uuid.uuid4().hex[:8]}_upsert"
        db_manager.create_prop_account(
            account_id=account_id,
            firm_name="TestFirm"
        )

        # First call - should create new snapshot
        snapshot1 = db_manager.save_daily_snapshot(
            account_id=account_id,
            equity=100500.0,
            balance=100000.0
        )
        assert snapshot1.daily_start_balance == 100000.0
        assert snapshot1.current_equity == 100500.0

        # Second call - should update existing snapshot
        snapshot2 = db_manager.save_daily_snapshot(
            account_id=account_id,
            equity=100800.0,
            balance=100000.0
        )
        assert snapshot2.id == snapshot1.id  # Same record
        assert snapshot2.current_equity == 100800.0

    def test_get_daily_drawdown(self, db_manager):
        """Test get_daily_drawdown calculates drawdown correctly."""
        # Create account with unique ID
        account_id = f"test_{uuid.uuid4().hex[:8]}_drawdown"
        db_manager.create_prop_account(
            account_id=account_id,
            firm_name="TestFirm"
        )

        # Save snapshot with loss
        db_manager.save_daily_snapshot(
            account_id=account_id,
            equity=99500.0,  # 0.5% loss
            balance=100000.0
        )

        drawdown = db_manager.get_daily_drawdown(account_id)
        assert drawdown == 0.5  # (100000 - 99500) / 100000 * 100

    def test_create_trade_proposal(self, db_manager):
        """Test create_trade_proposal inserts new proposal."""
        bot_id = f"rsi_bot_{uuid.uuid4().hex[:8]}"
        proposal = db_manager.create_trade_proposal(
            bot_id=bot_id,
            symbol="EURUSD",
            kelly_score=0.85,
            regime="trend_following",
            proposed_lot_size=0.1
        )

        assert proposal.id is not None
        assert proposal.bot_id == bot_id
        assert proposal.symbol == "EURUSD"
        assert proposal.kelly_score == 0.85
        assert proposal.status == "pending"

    def test_update_trade_proposal(self, db_manager):
        """Test update_trade_proposal changes status."""
        bot_id = f"rsi_bot_{uuid.uuid4().hex[:8]}_update"
        proposal = db_manager.create_trade_proposal(
            bot_id=bot_id,
            symbol="EURUSD",
            kelly_score=0.85,
            regime="trend_following",
            proposed_lot_size=0.1
        )

        updated = db_manager.update_trade_proposal(proposal.id, "approved")

        assert updated is not None
        assert updated.status == "approved"
        assert updated.reviewed_at is not None

    def test_context_manager_support(self, db_manager):
        """Test DatabaseManager supports context manager protocol."""
        # Test __enter__ and __exit__
        with db_manager:
            assert db_manager is not None
            db_manager.create_prop_account(
                account_id=f"context_test_{uuid.uuid4().hex[:8]}",
                firm_name="ContextTest"
            )


class TestDatabaseManagerChromaDB:
    """Test DatabaseManager ChromaDB wrapper methods."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset singleton before and after each test."""
        import src.database.manager
        src.database.manager.DatabaseManager._instance = None
        yield
        src.database.manager.DatabaseManager._instance = None

    @pytest.fixture
    def temp_chroma_path(self):
        """Create temporary directory for test ChromaDB data."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def db_manager(self, temp_chroma_path):
        """Create DatabaseManager with temporary ChromaDB path."""
        from src.database.chroma_client import ChromaDBClient
        from src.database.manager import DatabaseManager

        # Create a test instance with temp ChromaDB
        db = DatabaseManager()

        # Replace chroma client with temp one
        db.chroma = ChromaDBClient(persist_directory=Path(temp_chroma_path))
        db._initialized = True

        return db

    def test_search_strategies(self, db_manager):
        """Test search_strategies wraps ChromaDB client method."""
        strategy_id = f"test_{uuid.uuid4().hex[:8]}"
        db_manager.add_strategy(
            strategy_id=strategy_id,
            code="RSI strategy code",
            strategy_name="RSI Strategy",
            code_hash="hash123"
        )

        results = db_manager.search_strategies("RSI trading", limit=5)
        assert len(results) > 0
        assert results[0]["id"] == strategy_id

    def test_add_strategy(self, db_manager):
        """Test add_strategy wraps ChromaDB client method."""
        strategy_id = f"test_{uuid.uuid4().hex[:8]}"
        db_manager.add_strategy(
            strategy_id=strategy_id,
            code="Moving average strategy",
            strategy_name="MA Strategy",
            code_hash="hash456",
            performance_metrics={"win_rate": 0.7}
        )

        results = db_manager.search_strategies("Moving average", limit=5)
        assert len(results) > 0

    def test_search_knowledge(self, db_manager):
        """Test search_knowledge wraps ChromaDB client method."""
        article_id = f"article_{uuid.uuid4().hex[:8]}"
        db_manager.add_knowledge(
            article_id=article_id,
            content="Article content about indicators",
            title="Trading Indicators Guide",
            url="https://example.com",
            categories="education"
        )

        results = db_manager.search_knowledge("trading indicators", limit=5)
        assert len(results) > 0
        assert results[0]["id"] == article_id

    def test_add_knowledge(self, db_manager):
        """Test add_knowledge wraps ChromaDB client method."""
        article_id = f"article_{uuid.uuid4().hex[:8]}"
        db_manager.add_knowledge(
            article_id=article_id,
            content="Risk management best practices",
            title="Risk Management",
            url="https://example.com/risk",
            categories="risk",
            relevance_score=0.95
        )

        results = db_manager.search_knowledge("risk management", limit=5)
        assert len(results) > 0
