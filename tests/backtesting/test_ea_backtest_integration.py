"""
Tests for EA Backtest Integration

Task Group 4: Connect EA variants to all backtest modes

This test module validates:
- EA BACKTEST_REQUEST message type in department mail system
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from agents.departments.department_mail import MessageType, DepartmentMailService, Priority


@pytest.mark.unit
class TestBACKTESTREQUESTMessageType:
    """Test BACKTEST_REQUEST message type in department mail (Task 5)."""

    def test_backtest_request_message_type_exists(self):
        """Test BACKTEST_REQUEST message type is defined."""
        assert hasattr(MessageType, 'BACKTEST_REQUEST')
        assert MessageType.BACKTEST_REQUEST.value == "backtest_request"

    def test_message_type_values(self):
        """Test all message type values are correct."""
        assert MessageType.STATUS.value == "status"
        assert MessageType.QUESTION.value == "question"
        assert MessageType.RESULT.value == "result"
        assert MessageType.ERROR.value == "error"
        assert MessageType.DISPATCH.value == "dispatch"
        assert MessageType.STRATEGY_DISPATCH.value == "strategy_dispatch"
        assert MessageType.BACKTEST_REQUEST.value == "backtest_request"

    def test_can_create_message_with_backtest_request_type(self):
        """Test creating a message with BACKTEST_REQUEST type."""
        from datetime import datetime, timezone
        from agents.departments.department_mail import DepartmentMessage

        message = DepartmentMessage(
            id="test-123",
            from_dept="trading",
            to_dept="backtesting",
            type=MessageType.BACKTEST_REQUEST,
            subject="Run EURUSD backtest",
            body='{"ea_variant": "vanilla", "backtest_type": "normal"}',
            priority=Priority.HIGH,
            timestamp=datetime.now(timezone.utc),
            read=False
        )

        assert message.type == MessageType.BACKTEST_REQUEST
        assert message.from_dept == "trading"
        assert message.to_dept == "backtesting"


@pytest.mark.unit
class TestDepartmentMailWithBacktestRequest:
    """Test department mail service with BACKTEST_REQUEST messages."""

    def test_send_backtest_request_message(self, tmp_path):
        """Test sending a BACKTEST_REQUEST message."""
        db_path = tmp_path / "test_mail.db"
        mail_service = DepartmentMailService(db_path=str(db_path))

        try:
            message = mail_service.send(
                from_dept="trading",
                to_dept="backtesting",
                type=MessageType.BACKTEST_REQUEST,
                subject="Run EURUSD backtest",
                body='{"ea_variant": "spiced", "backtest_type": "monte_carlo", "symbol": "EURUSD"}',
                priority=Priority.HIGH
            )

            assert message.type == MessageType.BACKTEST_REQUEST
            assert message.subject == "Run EURUSD backtest"
            assert "spiced" in message.body
        finally:
            mail_service.close()

    def test_check_inbox_for_backtest_requests(self, tmp_path):
        """Test checking inbox for BACKTEST_REQUEST messages."""
        db_path = tmp_path / "test_mail.db"
        mail_service = DepartmentMailService(db_path=str(db_path))

        try:
            # Send backtest request
            mail_service.send(
                from_dept="trading",
                to_dept="backtesting",
                type=MessageType.BACKTEST_REQUEST,
                subject="Run EURUSD backtest",
                body='{"ea_variant": "vanilla", "backtest_type": "normal"}',
                priority=Priority.NORMAL
            )

            # Check inbox
            inbox = mail_service.check_inbox(
                dept="backtesting",
                unread_only=True,
                limit=10
            )

            assert len(inbox) == 1
            assert inbox[0].type == MessageType.BACKTEST_REQUEST
        finally:
            mail_service.close()


# --- Fixtures ---

@pytest.fixture
def sample_backtest_data():
    """Create sample backtest data for testing."""
    return pd.DataFrame({
        'time': pd.date_range(start='2024-01-01', periods=100, freq='1h'),
        'open': np.linspace(1.1000, 1.1100, 100),
        'high': np.linspace(1.1050, 1.1150, 100),
        'low': np.linspace(1.0950, 1.1050, 100),
        'close': np.linspace(1.1000, 1.1100, 100),
        'tick_volume': np.random.randint(1000, 3000, 100)
    })
