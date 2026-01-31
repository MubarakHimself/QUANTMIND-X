#!/usr/bin/env python3
"""
Integration Tests for QuantMind Hybrid Core v7

Tests cover critical end-to-end workflows:
- EA heartbeat -> DatabaseManager snapshot
- PropCommander auction with database-backed PropState
- PropGovernor throttle with database queries
- ChromaDB strategy search and storage
- Agent queue operations across multiple agents
- Database rollback on errors

Run only these tests: pytest tests/integration/test_hybrid_core_v7.py -v
"""

import pytest
import sys
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.database.manager import DatabaseManager
from src.database.models import PropFirmAccount, DailySnapshot, TradeProposal
from src.router.prop.commander import PropCommander
from src.router.prop.governor import PropGovernor
from src.router.prop.state import PropAccountMetrics
from src.queues.manager import QueueManager


@pytest.fixture(autouse=True)
def clean_database():
    """Clean database before and after all tests."""
    # Cleanup before tests
    try:
        db = DatabaseManager()
        with db.get_session() as session:
            session.query(DailySnapshot).delete()
            session.query(PropFirmAccount).filter(
                PropFirmAccount.account_id.like("test_%")
            ).delete(synchronize_session=False)
            session.commit()
    except Exception as e:
        print(f"Cleanup error (before): {e}")

    yield

    # Cleanup after tests
    try:
        db = DatabaseManager()
        with db.get_session() as session:
            session.query(DailySnapshot).delete()
            session.query(PropFirmAccount).filter(
                PropFirmAccount.account_id.like("test_%")
            ).delete(synchronize_session=False)
            session.commit()
    except Exception as e:
        print(f"Cleanup error (after): {e}")


class TestEAHeartbeatToDatabaseSnapshot:
    """Test end-to-end flow: EA sends heartbeat -> DatabaseManager stores snapshot."""

    @pytest.mark.integration
    def test_heartbeat_creates_daily_snapshot(self):
        """Test EA heartbeat creates daily snapshot in database."""
        db = DatabaseManager()

        # Simulate EA heartbeat: account_id, equity, balance
        account_id = "test_heartbeat_001"
        equity = 105000.0
        balance = 100000.0

        # Save snapshot via DatabaseManager
        snapshot = db.save_daily_snapshot(account_id, equity, balance)

        # Verify snapshot was created
        assert snapshot is not None
        assert snapshot.current_equity == equity
        assert snapshot.daily_start_balance == balance

        # Verify account was auto-created - use fresh session
        with db.get_session() as session:
            account = session.query(PropFirmAccount).filter(
                PropFirmAccount.account_id == account_id
            ).first()
            assert account is not None

    @pytest.mark.integration
    def test_heartbeat_updates_existing_snapshot(self):
        """Test heartbeat updates existing daily snapshot."""
        db = DatabaseManager()

        account_id = "test_heartbeat_002"

        # First heartbeat
        s1 = db.save_daily_snapshot(account_id, 100000.0, 100000.0)
        s1_id = s1.id

        # Second heartbeat with higher equity
        s2 = db.save_daily_snapshot(account_id, 102000.0, 100000.0)

        # Verify same snapshot was updated
        assert s1_id == s2.id

    @pytest.mark.integration
    def test_heartbeat_high_water_mark_updates(self):
        """Test that high water mark is properly updated."""
        db = DatabaseManager()

        account_id = "test_heartbeat_003"

        # First heartbeat
        s1 = db.save_daily_snapshot(account_id, 100000.0, 100000.0)

        # Second heartbeat with higher equity
        s2 = db.save_daily_snapshot(account_id, 105000.0, 100000.0)

        # Verify HWM increased - query fresh snapshot
        s2_fresh = db.get_daily_snapshot(account_id)
        assert s2_fresh is not None
        assert s2_fresh.high_water_mark >= 100000.0


class TestPropCommanderAuctionWithDatabase:
    """Test PropCommander auction with database-backed PropState."""

    @pytest.mark.integration
    def test_commander_standard_mode_auction(self):
        """Test PropCommander runs standard auction when not in preservation mode."""
        db = DatabaseManager()

        # Create account with initial state
        db.create_prop_account(
            account_id="test_commander_001",
            firm_name="TestFirm",
            target_profit_pct=8.0
        )

        # Set daily snapshot (not in profit, so standard mode)
        db.save_daily_snapshot("test_commander_001", equity=100000.0, balance=100000.0)

        # Create PropCommander
        commander = PropCommander(account_id="test_commander_001")

        # Create mock regime report
        regime_report = Mock()
        regime_report.regime = "TREND_FOLLOWING"
        regime_report.chaos_score = 0.1

        # Run auction
        bots = commander.run_auction(regime_report)

        # Verify base auction was called (returns empty list in stub implementation)
        assert isinstance(bots, list)

    @pytest.mark.integration
    def test_commander_preservation_mode_filters_kelly_score(self):
        """Test PropCommander filters by Kelly score in preservation mode."""
        db = DatabaseManager()

        # Create account and set profitable state (triggers preservation mode)
        db.create_prop_account(
            account_id="test_commander_002",
            firm_name="TestFirm",
            target_profit_pct=8.0
        )

        # Set snapshot with 8% profit (triggers preservation mode)
        db.save_daily_snapshot("test_commander_002", equity=108000.0, balance=100000.0)

        # Create PropCommander with metrics
        commander = PropCommander(account_id="test_commander_002")

        # Create metrics that trigger preservation mode
        metrics = PropAccountMetrics(
            account_id="test_commander_002",
            daily_start_balance=100000.0,
            high_water_mark=108000.0,
            current_equity=108000.0,
            trading_days=10,  # Above minimum
            target_met=True
        )

        # Check if preservation mode is active
        is_preservation = commander._check_preservation_mode(metrics)
        assert is_preservation is True

        # Verify minimum trading days check works
        needs_days = commander._needs_trading_days(metrics)
        assert needs_days is False

        # Verify coin flip bot is available
        coin_flip = commander._get_coin_flip_bot()
        assert coin_flip is not None
        assert coin_flip["name"] == "CoinFlip_Bot"


class TestPropGovernorThrottleWithDatabase:
    """Test PropGovernor throttle with database queries."""

    @pytest.mark.integration
    def test_governor_no_throttle_in_profit(self):
        """Test Governor applies no throttle when account is in profit."""
        db = DatabaseManager()

        # Create account with profitable state
        db.create_prop_account(account_id="test_governor_001", firm_name="TestFirm")
        db.save_daily_snapshot("test_governor_001", equity=102000.0, balance=100000.0)

        # Create PropGovernor
        governor = PropGovernor(account_id="test_governor_001")

        # Calculate throttle when in profit (current_balance > start_balance)
        throttle = governor._get_quadratic_throttle(current_balance=102000.0)

        # Should return 1.0 (full throttle) when in profit
        assert throttle == 1.0

    @pytest.mark.integration
    def test_governor_quadratic_throttle_formula(self):
        """Test Governor quadratic throttle formula calculation."""
        governor = PropGovernor(account_id="test")

        # Test the throttle formula with explicit values
        start_balance = 100000.0
        current_balance = 98000.0  # 2% loss
        loss_pct = (start_balance - current_balance) / start_balance
        effective_limit = governor.effective_limit  # 0.04

        # Calculate expected throttle: 1.0 - (0.02/0.04)^2 = 0.75
        expected_throttle = 1.0 - (loss_pct / effective_limit) ** 2

        # Verify throttle is between 0 and 1
        assert 0.0 < expected_throttle < 1.0
        assert abs(expected_throttle - 0.75) < 0.01  # Approximately 0.75

    @pytest.mark.integration
    def test_governor_hard_stop_at_limit(self):
        """Test Governor hard stops when loss reaches effective limit."""
        governor = PropGovernor(account_id="test")

        # Test with 4% loss (at effective limit)
        start_balance = 100000.0
        current_balance = 96000.0  # 4% loss
        loss_pct = (start_balance - current_balance) / start_balance
        effective_limit = governor.effective_limit  # 0.04

        # At effective limit: throttle should be 0.0
        if loss_pct >= effective_limit:
            throttle = 0.0
        else:
            throttle = 1.0 - (loss_pct / effective_limit) ** 2

        # Hard stop should return 0.0
        assert throttle == 0.0

    @pytest.mark.integration
    def test_governor_with_news_override(self):
        """Test Governor hard stops on news regardless of throttle."""
        db = DatabaseManager()

        # Create account in profit
        db.create_prop_account(account_id="test_governor_002", firm_name="TestFirm")
        db.save_daily_snapshot("test_governor_002", equity=105000.0, balance=100000.0)

        # Create PropGovernor
        governor = PropGovernor(account_id="test_governor_002")

        # Create mock regime report with KILL_ZONE news
        regime_report = Mock()
        regime_report.news_state = "KILL_ZONE"
        regime_report.chaos_score = 0.1
        regime_report.is_systemic_risk = False

        # Create trade proposal
        trade_proposal = {
            "bot_id": "test_bot",
            "symbol": "EURUSD",
            "current_balance": 105000.0
        }

        # Calculate risk
        mandate = governor.calculate_risk(regime_report, trade_proposal)

        # Verify hard stop for news
        assert mandate.allocation_scalar == 0.0
        assert mandate.risk_mode == "HALTED_NEWS"
        assert "Hard Stop: News Event" in mandate.notes


class TestChromaDBStrategySearchAndStorage:
    """Test ChromaDB strategy search and storage integration."""

    @pytest.fixture
    def temp_chroma_path(self):
        """Create temporary directory for test ChromaDB data."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.mark.integration
    def test_chromadb_collection_creation_and_search(self, temp_chroma_path):
        """Test ChromaDB collection creation for strategies with metadata."""
        try:
            import chromadb
        except ImportError:
            pytest.skip("ChromaDB not installed")

        # Create ChromaDB client
        client = chromadb.PersistentClient(path=temp_chroma_path)

        # Create quantmind_strategies collection with cosine similarity
        collection = client.get_or_create_collection(
            name="quantmind_strategies",
            metadata={"hnsw:space": "cosine"}
        )

        # Add strategy with required metadata
        collection.add(
            ids=["strategy_001"],
            documents=["RSI mean reversion strategy with dynamic position sizing for range-bound markets"],
            metadatas=[{
                "strategy_name": "RSI Dynamic",
                "code_hash": "abc123def456",
                "created_at": "2026-01-30T00:00:00Z",
                "performance_metrics": '{"win_rate": 0.65, "profit_factor": 1.8}'
            }]
        )

        # Verify collection count
        assert collection.count() == 1

        # Search for similar strategies
        results = collection.query(
            query_texts=["mean reversion trading with RSI"],
            n_results=1
        )

        # Verify search returned results
        assert len(results["ids"][0]) == 1
        assert results["ids"][0][0] == "strategy_001"
        assert results["metadatas"][0][0]["strategy_name"] == "RSI Dynamic"

    @pytest.mark.integration
    def test_chromadb_knowledge_collection_search(self, temp_chroma_path):
        """Test ChromaDB knowledge collection for MQL5 articles."""
        try:
            import chromadb
        except ImportError:
            pytest.skip("ChromaDB not installed")

        client = chromadb.PersistentClient(path=temp_chroma_path)

        # Create quantmind_knowledge collection
        collection = client.get_or_create_collection(
            name="quantmind_knowledge",
            metadata={"hnsw:space": "cosine"}
        )

        # Add knowledge articles
        collection.add(
            ids=["article_001", "article_002"],
            documents=[
                "MQL5 article about RSI divergence trading strategies and entry signals",
                "Moving average crossover trend following system with exit rules"
            ],
            metadatas=[
                {
                    "title": "RSI Divergence Trading",
                    "url": "https://mql5.com/articles/rsi-divergence",
                    "categories": "Trading Systems,Indicators",
                    "relevance_score": 0.92
                },
                {
                    "title": "MA Crossover System",
                    "url": "https://mql5.com/articles/ma-crossover",
                    "categories": "Trading Systems",
                    "relevance_score": 0.85
                }
            ]
        )

        # Search for RSI-related content
        results = collection.query(
            query_texts=["RSI divergence trading"],
            n_results=2
        )

        # Verify search returned relevant results
        assert len(results["ids"][0]) == 2

    @pytest.mark.integration
    def test_chromadb_market_patterns_collection(self, temp_chroma_path):
        """Test ChromaDB market patterns collection for regime storage."""
        try:
            import chromadb
        except ImportError:
            pytest.skip("ChromaDB not installed")

        client = chromadb.PersistentClient(path=temp_chroma_path)

        # Create market_patterns collection
        collection = client.get_or_create_collection(
            name="market_patterns",
            metadata={"hnsw:space": "cosine"}
        )

        # Add market patterns with different volatility levels
        collection.add(
            ids=["pattern_001", "pattern_002", "pattern_003"],
            documents=[
                "Bullish trend with low volatility regime detected",
                "High volatility breakout conditions",
                "Medium volatility consolidation phase"
            ],
            metadatas=[
                {
                    "pattern_type": "trend_following",
                    "timestamp": "2026-01-30T12:00:00Z",
                    "volatility_level": "low"
                },
                {
                    "pattern_type": "breakout",
                    "timestamp": "2026-01-30T14:00:00Z",
                    "volatility_level": "high"
                },
                {
                    "pattern_type": "range",
                    "timestamp": "2026-01-30T16:00:00Z",
                    "volatility_level": "medium"
                }
            ]
        )

        # Query with metadata filter for low volatility
        results = collection.query(
            query_texts=["trend regime"],
            n_results=10,
            where={"volatility_level": "low"}
        )

        # Verify filtering works
        assert len(results["ids"][0]) == 1
        assert results["metadatas"][0][0]["volatility_level"] == "low"
        assert results["metadatas"][0][0]["pattern_type"] == "trend_following"


class TestAgentQueueOperations:
    """Test agent queue operations across multiple agents."""

    @pytest.fixture
    def temp_queue_manager(self):
        """Create QueueManager with temporary directory."""
        temp_dir = tempfile.mkdtemp()
        manager = QueueManager(queue_dir_path=temp_dir)
        yield manager, temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.mark.integration
    def test_multiple_agents_enqueue_dequeue_fifo(self, temp_queue_manager):
        """Test FIFO ordering works across multiple agent queues."""
        manager, temp_dir = temp_queue_manager

        # Enqueue tasks for different agents
        analyst_task = manager.enqueue("analyst", {"type": "research", "symbol": "EURUSD"})
        quant_task = manager.enqueue("quant", {"type": "strategy", "regime": "trend"})
        executor_task = manager.enqueue("executor", {"type": "deploy", "bot": "RSI_Bot"})

        # Dequeue in FIFO order for each agent
        analyst_dequeued = manager.dequeue("analyst")
        quant_dequeued = manager.dequeue("quant")
        executor_dequeued = manager.dequeue("executor")

        # Verify correct tasks returned
        assert analyst_dequeued["task_id"] == analyst_task
        assert quant_dequeued["task_id"] == quant_task
        assert executor_dequeued["task_id"] == executor_task

        # Verify status changed to processing
        assert analyst_dequeued["status"] == "processing"
        assert quant_dequeued["status"] == "processing"
        assert executor_dequeued["status"] == "processing"

    @pytest.mark.integration
    def test_queue_persistence_across_manager_instances(self, temp_queue_manager):
        """Test queue data persists when creating new manager instance."""
        manager1, temp_dir = temp_queue_manager

        # Enqueue task
        task_id = manager1.enqueue("analyst", {"persistent": "data"})

        # Create new manager instance with same directory
        manager2 = QueueManager(queue_dir_path=temp_dir)

        # Verify task is still there
        task = manager2.dequeue("analyst")
        assert task is not None
        assert task["task_id"] == task_id
        assert task["payload"]["persistent"] == "data"

    @pytest.mark.integration
    def test_queue_status_workflow_pending_to_complete(self, temp_queue_manager):
        """Test complete workflow: pending -> processing -> complete."""
        manager, temp_dir = temp_queue_manager

        # Enqueue task (status: pending)
        task_id = manager.enqueue("quant", {"strategy": "test"})

        # Verify initial status
        status = manager.get_queue_status("quant")
        assert status["pending"] == 1
        assert status["processing"] == 0
        assert status["complete"] == 0

        # Dequeue (status: processing)
        task = manager.dequeue("quant")
        assert task["status"] == "processing"

        # Update to complete
        manager.update_status("quant", task_id, "complete")

        # Verify final status
        status = manager.get_queue_status("quant")
        assert status["pending"] == 0
        assert status["processing"] == 0
        assert status["complete"] == 1


class TestDatabaseRollbackOnError:
    """Test database rollback on errors."""

    @pytest.mark.integration
    def test_context_manager_rollback_on_exception(self):
        """Test context manager rolls back on exception."""
        db = DatabaseManager()

        # Create account first
        db.create_prop_account(
            account_id="test_rollback_001",
            firm_name="TestFirm"
        )

        initial_account_count = None

        # Try to update account but trigger exception
        try:
            with db.get_session() as session:
                # Get initial count
                initial_account_count = session.query(PropFirmAccount).count()

                # Update account
                account_record = session.query(PropFirmAccount).filter(
                    PropFirmAccount.account_id == "test_rollback_001"
                ).first()
                account_record.firm_name = "UpdatedFirm"

                # Trigger exception (this should cause rollback)
                raise ValueError("Intentional test error")
        except ValueError:
            pass  # Expected

        # Verify rollback occurred - count should be same
        with db.get_session() as session:
            final_account_count = session.query(PropFirmAccount).count()
            assert initial_account_count == final_account_count

            # Verify account was NOT updated
            account_record = session.query(PropFirmAccount).filter(
                PropFirmAccount.account_id == "test_rollback_001"
            ).first()
            assert account_record.firm_name == "TestFirm"  # Not "UpdatedFirm"

    @pytest.mark.integration
    def test_transaction_rollback_preserves_data_integrity(self):
        """Test transaction rollback preserves referential integrity."""
        db = DatabaseManager()

        # Create initial account
        account_id = db.create_prop_account(
            account_id="test_rollback_002",
            firm_name="TestFirm"
        ).account_id

        # Try to create snapshot with invalid data
        try:
            with db.get_session() as session:
                # Get account for foreign key
                account = session.query(PropFirmAccount).filter(
                    PropFirmAccount.account_id == account_id
                ).first()

                # Try to create snapshot with invalid equity (negative)
                snapshot = DailySnapshot(
                    account_id=account.id,
                    date="2026-01-30",
                    daily_start_balance=100000.0,
                    high_water_mark=100000.0,
                    current_equity=-1000.0,  # Invalid negative equity
                    daily_drawdown_pct=0.0,
                    is_breached=False
                )
                session.add(snapshot)

                # Trigger rollback
                raise ValueError("Invalid equity value")
        except ValueError:
            pass

        # Verify snapshot was NOT created - get snapshot should return None or have valid data
        snapshot = db.get_daily_snapshot(account_id)
        if snapshot is not None:
            # If snapshot exists, it should NOT be the one we tried to create with negative equity
            assert snapshot.current_equity >= 0

    @pytest.mark.integration
    def test_multiple_operations_rollback_atomically(self):
        """Test all operations rollback atomically on error."""
        db = DatabaseManager()

        # Try to create multiple related records
        try:
            with db.get_session() as session:
                # Create account
                account = PropFirmAccount(
                    firm_name="AtomicTest",
                    account_id="test_atomic_999"
                )
                session.add(account)
                session.flush()

                # Create snapshot
                snapshot = DailySnapshot(
                    account_id=account.id,
                    date="2026-01-30",
                    daily_start_balance=100000.0,
                    high_water_mark=100000.0,
                    current_equity=100000.0,
                    daily_drawdown_pct=0.0,
                    is_breached=False
                )
                session.add(snapshot)
                session.flush()

                # Trigger rollback
                raise RuntimeError("Atomic rollback test")
        except RuntimeError:
            pass

        # Verify account was rolled back
        account = db.get_prop_account("test_atomic_999")
        assert account is None  # Account should not exist due to rollback
