import pytest
from datetime import date
from unittest.mock import MagicMock, patch
from src.router.fee_monitor import FeeMonitor, FeeReport

@pytest.fixture
def mock_db():
    db = MagicMock()
    return db

def test_record_trade_fee(mock_db):
    monitor = FeeMonitor('test_account', mock_db, 1000.0)
    monitor.record_trade_fee('bot1', 5.0)
    # Verify DB calls

def test_should_halt_trading(mock_db):
    monitor = FeeMonitor('test_account', mock_db, 1000.0)
    with patch.object(monitor, 'get_daily_report', return_value=FeeReport('2026-02-13', 150.0, 10, 15.0, 1000.0, 15.0, 'ACTIVE')):
        should_halt, reason = monitor.should_halt_trading()
        assert should_halt  # 15% > 10%

def test_calculate_scalping_fee_burn():
    monitor = FeeMonitor('test', account_balance=1000.0)
    burn = monitor.calculate_scalping_fee_burn(20, 24, 5.0)
    assert burn == 240.0  # 20*24*5 /1000 *100 = 240%
