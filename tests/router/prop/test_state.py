"""
Tests for PropState Database Integration

Tests the PropState class integration with DatabaseManager for:
- State persistence across sessions
- Daily snapshot management
- Daily loss calculation from database
- Quadratic throttle calculation using database data
"""

import pytest
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from src.router.prop.state import PropState
from src.database.manager import DatabaseManager


@pytest.fixture
def test_db_path():
    """Create a temporary database for testing."""
    fd, path = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    yield path
    # Cleanup
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def mock_db_manager(test_db_path):
    """Create a mock DatabaseManager with test database."""
    # Reset singleton
    DatabaseManager._instance = None

    # Patch the database path before importing the module
    with patch('src.database.engine.DB_ABSOLUTE_PATH', test_db_path):
        # Clear the engine module cache to use new database path
        if 'src.database.engine' in sys.modules:
            del sys.modules['src.database.engine']

        db = DatabaseManager.__new__(DatabaseManager)
        db._initialized = False
        db.__init__()
        yield db
        # Cleanup singleton
        DatabaseManager._instance = None
        # Clean up module cache
        if 'src.database.engine' in sys.modules:
            del sys.modules['src.database.engine']


@pytest.fixture
def test_account_id():
    """Return a unique test account ID."""
    import uuid
    return f"TEST{uuid.uuid4().hex[:8]}"


class TestPropStateInitialization:
    """Test PropState initialization with DatabaseManager."""

    def test_init_with_database_manager(self, mock_db_manager, test_account_id):
        """Test PropState initialization accepts DatabaseManager."""
        prop_state = PropState(
            account_id=test_account_id,
            db_manager=mock_db_manager
        )

        assert prop_state.account_id == test_account_id
        assert prop_state.db_manager == mock_db_manager

    def test_init_creates_database_manager_if_not_provided(self, test_account_id):
        """Test PropState auto-creates DatabaseManager if not provided."""
        # Reset singleton before this test
        DatabaseManager._instance = None

        prop_state = PropState(account_id=test_account_id)

        assert prop_state.account_id == test_account_id
        assert prop_state.db_manager is not None
        assert isinstance(prop_state.db_manager, DatabaseManager)


class TestPropStateUpdateSnapshot:
    """Test update_snapshot() method with database persistence."""

    def test_update_snapshot_saves_to_database(self, mock_db_manager, test_account_id):
        """Test update_snapshot() saves DailySnapshot to database."""
        prop_state = PropState(
            account_id=test_account_id,
            db_manager=mock_db_manager
        )

        # Update snapshot with equity and balance
        prop_state.update_snapshot(equity=105000.0, balance=100000.0)

        # Verify snapshot was saved
        snapshot = mock_db_manager.get_daily_snapshot(test_account_id)
        assert snapshot is not None
        assert snapshot['current_equity'] == 105000.0
        assert snapshot['daily_start_balance'] == 100000.0

    def test_update_snapshot_updates_high_water_mark(self, mock_db_manager, test_account_id):
        """Test update_snapshot() updates high water mark correctly."""
        prop_state = PropState(
            account_id=test_account_id,
            db_manager=mock_db_manager
        )

        # First snapshot
        prop_state.update_snapshot(equity=102000.0, balance=100000.0)
        snapshot1 = mock_db_manager.get_daily_snapshot(test_account_id)
        assert snapshot1['high_water_mark'] == 102000.0

        # Second snapshot with higher equity
        prop_state.update_snapshot(equity=105000.0, balance=100000.0)
        snapshot2 = mock_db_manager.get_daily_snapshot(test_account_id)
        assert snapshot2['high_water_mark'] == 105000.0

    def test_update_snapshot_upserts_existing_record(self, mock_db_manager, test_account_id):
        """Test update_snapshot() upserts - updates if exists, creates if not."""
        prop_state = PropState(
            account_id=test_account_id,
            db_manager=mock_db_manager
        )

        # First update (create)
        prop_state.update_snapshot(equity=101000.0, balance=100000.0)
        snapshot1 = mock_db_manager.get_daily_snapshot(test_account_id)
        assert snapshot1['current_equity'] == 101000.0

        # Second update (same day - should update)
        prop_state.update_snapshot(equity=102000.0, balance=100000.0)
        snapshot2 = mock_db_manager.get_daily_snapshot(test_account_id)

        # Should be same snapshot record (same ID)
        assert snapshot1['id'] == snapshot2['id']
        assert snapshot2['current_equity'] == 102000.0


class TestPropStateCheckDailyLoss:
    """Test check_daily_loss() method with database queries."""

    def test_check_daily_loss_calculates_from_database(self, mock_db_manager, test_account_id):
        """Test check_daily_loss() calculates drawdown from database snapshot."""
        prop_state = PropState(
            account_id=test_account_id,
            db_manager=mock_db_manager
        )

        # Save snapshot: started with 100k, now at 98k (2% loss)
        mock_db_manager.save_daily_snapshot(
            account_id=test_account_id,
            equity=98000.0,
            balance=100000.0
        )

        # Check daily loss
        loss_pct = prop_state.check_daily_loss(current_equity=98000.0)

        # Should return 2.0% drawdown
        assert abs(loss_pct - 2.0) < 0.01

    def test_check_daily_loss_returns_zero_when_no_snapshot(self, mock_db_manager, test_account_id):
        """Test check_daily_loss() returns 0.0 when no snapshot exists."""
        prop_state = PropState(
            account_id=test_account_id,
            db_manager=mock_db_manager
        )

        # No snapshot saved
        loss_pct = prop_state.check_daily_loss(current_equity=100000.0)

        assert loss_pct == 0.0

    def test_check_daily_loss_with_profit(self, mock_db_manager, test_account_id):
        """Test check_daily_loss() returns 0.0 when in profit."""
        prop_state = PropState(
            account_id=test_account_id,
            db_manager=mock_db_manager
        )

        # Save snapshot: started with 100k, now at 105k (in profit)
        mock_db_manager.save_daily_snapshot(
            account_id=test_account_id,
            equity=105000.0,
            balance=100000.0
        )

        # Check daily loss (should be 0 since we're in profit)
        loss_pct = prop_state.check_daily_loss(current_equity=105000.0)

        assert loss_pct == 0.0


class TestPropStateGetQuadraticThrottle:
    """Test get_quadratic_throttle() method using database daily_start_balance."""

    def test_get_quadratic_throttle_uses_database_start_balance(self, mock_db_manager, test_account_id):
        """Test get_quadratic_throttle() fetches daily_start_balance from database."""
        prop_state = PropState(
            account_id=test_account_id,
            db_manager=mock_db_manager
        )

        # Save snapshot with daily start balance
        mock_db_manager.save_daily_snapshot(
            account_id=test_account_id,
            equity=98000.0,
            balance=100000.0
        )

        # Get throttle with current balance at 99k (1% loss)
        throttle = prop_state.get_quadratic_throttle(
            current_balance=99000.0,
            limit_pct=0.05
        )

        # Formula: throttle = 1.0 - (loss_pct / effective_limit)^2
        # loss_pct = (100000 - 99000) / 100000 = 0.01 (1%)
        # effective_limit = 0.05 - 0.01 = 0.04 (4%)
        # throttle = 1.0 - (0.01 / 0.04)^2 = 1.0 - 0.0625 = 0.9375
        expected_throttle = 1.0 - (0.01 / 0.04) ** 2
        assert abs(throttle - expected_throttle) < 0.01

    def test_get_quadratic_throttle_returns_one_when_in_profit(self, mock_db_manager, test_account_id):
        """Test get_quadratic_throttle() returns 1.0 when in profit."""
        prop_state = PropState(
            account_id=test_account_id,
            db_manager=mock_db_manager
        )

        # Save snapshot
        mock_db_manager.save_daily_snapshot(
            account_id=test_account_id,
            equity=100000.0,
            balance=100000.0
        )

        # Current balance higher than start (in profit)
        throttle = prop_state.get_quadratic_throttle(
            current_balance=105000.0,
            limit_pct=0.05
        )

        assert throttle == 1.0

    def test_get_quadratic_throttle_hard_stop_at_limit(self, mock_db_manager, test_account_id):
        """Test get_quadratic_throttle() returns 0.0 at hard stop limit."""
        prop_state = PropState(
            account_id=test_account_id,
            db_manager=mock_db_manager
        )

        # Save snapshot
        mock_db_manager.save_daily_snapshot(
            account_id=test_account_id,
            equity=100000.0,
            balance=100000.0
        )

        # Current balance at 96k (4% loss = effective limit = hard stop)
        throttle = prop_state.get_quadratic_throttle(
            current_balance=96000.0,
            limit_pct=0.05
        )

        # Should return 0.0 (hard stop)
        assert throttle == 0.0

    def test_get_quadratic_throttle_returns_one_when_no_snapshot(self, mock_db_manager, test_account_id):
        """Test get_quadratic_throttle() returns 1.0 when no snapshot exists."""
        prop_state = PropState(
            account_id=test_account_id,
            db_manager=mock_db_manager
        )

        # No snapshot saved
        throttle = prop_state.get_quadratic_throttle(
            current_balance=100000.0,
            limit_pct=0.05
        )

        assert throttle == 1.0


class TestPropStatePersistence:
    """Test state persists across Python process restarts."""

    def test_state_persists_across_instances(self, mock_db_manager, test_account_id):
        """Test state survives recreating PropState instance."""
        # First instance - save snapshot
        prop_state1 = PropState(
            account_id=test_account_id,
            db_manager=mock_db_manager
        )
        prop_state1.update_snapshot(equity=103000.0, balance=100000.0)

        # Second instance - should retrieve saved state
        PropState(
            account_id=test_account_id,
            db_manager=mock_db_manager
        )

        # Verify state persisted
        snapshot = mock_db_manager.get_daily_snapshot(test_account_id)
        assert snapshot is not None
        assert snapshot['current_equity'] == 103000.0
        assert snapshot['daily_start_balance'] == 100000.0
