"""
Tests for SQLite Models (Task Group 1)

Focused tests for SQLAlchemy models covering:
- PropFirmAccount model validation and creation
- DailySnapshot model with unique constraint on (account_id, date)
- TradeProposal model status transitions
- Foreign key relationships between models
- Database session management

Reference: agent-os/specs/2026-01-30-quantmind-hybrid-core/spec.md
"""

import pytest
from datetime import datetime, date
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import IntegrityError

from src.database.models import Base, PropFirmAccount, DailySnapshot, TradeProposal


# Test database configuration
TEST_DB_PATH = "test_quantmind.db"


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


class TestPropFirmAccount:
    """Test PropFirmAccount model validation and creation."""

    def test_create_prop_firm_account(self, test_session: Session):
        """Test creating a valid PropFirmAccount instance."""
        account = PropFirmAccount(
            firm_name="MyForexFunds",
            account_id="12345",
            daily_loss_limit_pct=5.0,
            hard_stop_buffer_pct=1.0,
            target_profit_pct=8.0,
            min_trading_days=5
        )
        test_session.add(account)
        test_session.commit()
        test_session.refresh(account)

        assert account.id is not None
        assert account.firm_name == "MyForexFunds"
        assert account.account_id == "12345"
        assert account.daily_loss_limit_pct == 5.0
        assert account.hard_stop_buffer_pct == 1.0
        assert account.target_profit_pct == 8.0
        assert account.min_trading_days == 5
        assert account.created_at is not None
        assert account.updated_at is not None

    def test_account_id_unique_constraint(self, test_session: Session):
        """Test that account_id must be unique."""
        account1 = PropFirmAccount(
            firm_name="MyForexFunds",
            account_id="12345"
        )
        test_session.add(account1)
        test_session.commit()

        # Try to create another account with same account_id
        account2 = PropFirmAccount(
            firm_name="OtherFirm",
            account_id="12345"  # Duplicate
        )
        test_session.add(account2)

        with pytest.raises(IntegrityError):
            test_session.commit()

    def test_default_values(self, test_session: Session):
        """Test that default values are applied correctly."""
        account = PropFirmAccount(
            firm_name="TestFirm",
            account_id="99999"
        )
        test_session.add(account)
        test_session.commit()
        test_session.refresh(account)

        assert account.daily_loss_limit_pct == 5.0  # Default
        assert account.hard_stop_buffer_pct == 1.0  # Default
        assert account.target_profit_pct == 8.0  # Default
        assert account.min_trading_days == 5  # Default


class TestDailySnapshot:
    """Test DailySnapshot model with unique constraint."""

    def test_create_daily_snapshot(self, test_session: Session):
        """Test creating a valid DailySnapshot instance."""
        account = PropFirmAccount(
            firm_name="MyForexFunds",
            account_id="12345"
        )
        test_session.add(account)
        test_session.commit()
        test_session.refresh(account)

        snapshot = DailySnapshot(
            account_id=account.id,
            date="2026-01-30",
            daily_start_balance=100000.0,
            high_water_mark=102000.0,
            current_equity=101000.0,
            daily_drawdown_pct=0.0,
            is_breached=False
        )
        test_session.add(snapshot)
        test_session.commit()
        test_session.refresh(snapshot)

        assert snapshot.id is not None
        assert snapshot.account_id == account.id
        assert snapshot.date == "2026-01-30"
        assert snapshot.daily_start_balance == 100000.0
        assert snapshot.high_water_mark == 102000.0
        assert snapshot.current_equity == 101000.0
        assert snapshot.is_breached is False

    def test_unique_constraint_account_date(self, test_session: Session):
        """Test unique constraint on (account_id, date)."""
        account = PropFirmAccount(
            firm_name="MyForexFunds",
            account_id="12345"
        )
        test_session.add(account)
        test_session.commit()
        test_session.refresh(account)

        # Create first snapshot
        snapshot1 = DailySnapshot(
            account_id=account.id,
            date="2026-01-30",
            daily_start_balance=100000.0,
            high_water_mark=102000.0,
            current_equity=101000.0
        )
        test_session.add(snapshot1)
        test_session.commit()

        # Try to create duplicate snapshot for same account and date
        snapshot2 = DailySnapshot(
            account_id=account.id,
            date="2026-01-30",  # Same date
            daily_start_balance=100000.0,
            high_water_mark=102000.0,
            current_equity=101000.0
        )
        test_session.add(snapshot2)

        with pytest.raises(IntegrityError):
            test_session.commit()

    def test_foreign_key_relationship(self, test_session: Session):
        """Test foreign key relationship between DailySnapshot and PropFirmAccount."""
        account = PropFirmAccount(
            firm_name="MyForexFunds",
            account_id="12345"
        )
        test_session.add(account)
        test_session.commit()
        test_session.refresh(account)

        snapshot = DailySnapshot(
            account_id=account.id,
            date="2026-01-30",
            daily_start_balance=100000.0,
            high_water_mark=102000.0,
            current_equity=101000.0
        )
        test_session.add(snapshot)
        test_session.commit()
        test_session.refresh(snapshot)

        # Test relationship
        assert snapshot.account.firm_name == "MyForexFunds"
        assert len(account.daily_snapshots) == 1
        assert account.daily_snapshots[0].date == "2026-01-30"


class TestTradeProposal:
    """Test TradeProposal model status transitions."""

    def test_create_trade_proposal(self, test_session: Session):
        """Test creating a valid TradeProposal instance."""
        proposal = TradeProposal(
            bot_id="momentum_bot_v1",
            symbol="EURUSD",
            kelly_score=0.85,
            regime="trending",
            proposed_lot_size=0.1,
            status="pending"
        )
        test_session.add(proposal)
        test_session.commit()
        test_session.refresh(proposal)

        assert proposal.id is not None
        assert proposal.bot_id == "momentum_bot_v1"
        assert proposal.symbol == "EURUSD"
        assert proposal.kelly_score == 0.85
        assert proposal.regime == "trending"
        assert proposal.proposed_lot_size == 0.1
        assert proposal.status == "pending"
        assert proposal.created_at is not None
        assert proposal.reviewed_at is None

    def test_status_transitions(self, test_session: Session):
        """Test status transitions from pending to approved/rejected."""
        proposal = TradeProposal(
            bot_id="momentum_bot_v1",
            symbol="EURUSD",
            kelly_score=0.85,
            regime="trending",
            proposed_lot_size=0.1,
            status="pending"
        )
        test_session.add(proposal)
        test_session.commit()
        test_session.refresh(proposal)

        # Transition to approved
        proposal.status = "approved"
        proposal.reviewed_at = datetime.utcnow()
        test_session.commit()
        test_session.refresh(proposal)

        assert proposal.status == "approved"
        assert proposal.reviewed_at is not None

        # Transition to rejected
        proposal.status = "rejected"
        proposal.reviewed_at = datetime.utcnow()
        test_session.commit()
        test_session.refresh(proposal)

        assert proposal.status == "rejected"

    def test_default_status_pending(self, test_session: Session):
        """Test that default status is 'pending'."""
        proposal = TradeProposal(
            bot_id="momentum_bot_v1",
            symbol="EURUSD",
            kelly_score=0.85,
            regime="trending",
            proposed_lot_size=0.1
        )
        test_session.add(proposal)
        test_session.commit()
        test_session.refresh(proposal)

        assert proposal.status == "pending"


class TestDatabaseSessionManagement:
    """Test database session management."""

    def test_session_commit_and_rollback(self, test_engine):
        """Test session commit and rollback behavior."""
        TestSession = sessionmaker(bind=test_engine)

        # Test successful commit
        session = TestSession()
        account = PropFirmAccount(
            firm_name="TestFirm",
            account_id="11111"
        )
        session.add(account)
        session.commit()
        session.close()

        # Verify data was committed
        session = TestSession()
        retrieved = session.query(PropFirmAccount).filter_by(account_id="11111").first()
        assert retrieved is not None
        assert retrieved.firm_name == "TestFirm"
        session.close()

        # Test rollback on error
        session = TestSession()
        account2 = PropFirmAccount(
            firm_name="TestFirm2",
            account_id="11111"  # Duplicate account_id
        )
        session.add(account2)
        with pytest.raises(IntegrityError):
            session.commit()
        session.rollback()
        session.close()

        # Verify no duplicate was created
        session = TestSession()
        count = session.query(PropFirmAccount).filter_by(account_id="11111").count()
        assert count == 1
        session.close()

    def test_cascade_delete_relationship(self, test_session: Session):
        """Test cascade delete when parent account is deleted."""
        account = PropFirmAccount(
            firm_name="MyForexFunds",
            account_id="12345"
        )
        test_session.add(account)
        test_session.commit()
        test_session.refresh(account)

        snapshot = DailySnapshot(
            account_id=account.id,
            date="2026-01-30",
            daily_start_balance=100000.0,
            high_water_mark=102000.0,
            current_equity=101000.0
        )
        test_session.add(snapshot)
        test_session.commit()

        # Delete account (should cascade delete snapshots)
        test_session.delete(account)
        test_session.commit()

        # Verify snapshot was deleted
        remaining_snapshots = test_session.query(DailySnapshot).filter_by(
            account_id=account.id
        ).count()
        assert remaining_snapshots == 0
