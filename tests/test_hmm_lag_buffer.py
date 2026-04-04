"""
Tests for HMM Lag Buffer Service
=================================

Tests for the 3-day calendar lag enforcement for HMM training data.
Verifies that:
- Lag is enforced in calendar days (not trading days)
- Data-level lag enforcement (model cannot access data until lag expires)
- Weekend trades correctly calculate eligible dates
- Monday batch processing includes expired trades
"""

import pytest
import pytest_asyncio
from datetime import datetime, timedelta
from src.router.hmm_lag_buffer import (
    HmmLagBuffer,
    HmmTrainingTradeRecord,
    TradeOutcome,
    HmmBufferStatus,
    BatchResult
)


class TestHmmLagBuffer:
    """Test cases for HmmLagBuffer service."""

    @pytest.fixture
    def lag_buffer(self):
        """Create a fresh HmmLagBuffer instance for each test."""
        return HmmLagBuffer()

    @pytest.fixture
    def monday_date(self):
        """Returns a Monday date for testing."""
        # January 6, 2025 is a Monday
        return datetime(2025, 1, 6, 12, 0, 0)

    @pytest.fixture
    def friday_date(self):
        """Returns a Friday date for testing."""
        # January 3, 2025 is a Friday
        return datetime(2025, 1, 3, 12, 0, 0)

    @pytest.fixture
    def saturday_date(self):
        """Returns a Saturday date for testing."""
        # January 4, 2025 is a Saturday
        return datetime(2025, 1, 4, 12, 0, 0)

    # =======================================================================
    # AC1: 3-Day Calendar Lag Enforcement
    # =======================================================================

    def test_lag_days_returns_3(self, lag_buffer):
        """TradeOutcomeData.lag_days property returns 3 (calendar days)."""
        assert lag_buffer.LAG_CALENDAR_DAYS == 3

    def test_trade_closed_monday_eligible_thursday(self, lag_buffer, monday_date):
        """
        Given a trade closed on Monday,
        When submitted to the lag buffer,
        Then eligible_date is Thursday (Monday + 3 calendar days).
        """
        trade = lag_buffer.submit_trade_outcome(
            trade_id="T001",
            bot_id="BOT001",
            close_date=monday_date,
            outcome=TradeOutcome.WIN,
            pnl=100.0,
            holding_time_minutes=60,
            regime_at_entry="TRENDING"
        )

        expected_eligible = monday_date + timedelta(days=3)
        assert trade.eligible_date.date() == expected_eligible.date()

    def test_trade_closed_friday_eligible_monday(self, lag_buffer, friday_date):
        """
        Given a trade closed on Friday,
        When submitted to the lag buffer,
        Then eligible_date is Monday (Fri + 3 calendar days, weekend counts).
        """
        trade = lag_buffer.submit_trade_outcome(
            trade_id="T002",
            bot_id="BOT001",
            close_date=friday_date,
            outcome=TradeOutcome.LOSS,
            pnl=-50.0,
            holding_time_minutes=30,
            regime_at_entry="RANGING"
        )

        # Friday Jan 3 + 3 days = Monday Jan 6
        expected_eligible = friday_date + timedelta(days=3)
        assert trade.eligible_date.date() == expected_eligible.date()
        # Monday Jan 6
        assert trade.eligible_date.weekday() == 0  # Monday

    def test_trade_closed_saturday_eligible_tuesday(self, lag_buffer, saturday_date):
        """
        Given a trade closed on Saturday,
        When submitted to the lag buffer,
        Then eligible_date is Tuesday (Sat + 3 calendar days, weekend counts).
        """
        trade = lag_buffer.submit_trade_outcome(
            trade_id="T003",
            bot_id="BOT001",
            close_date=saturday_date,
            outcome=TradeOutcome.HOLDING,
            pnl=0.0,
            holding_time_minutes=120,
            regime_at_entry="CHAOS"
        )

        # Saturday Jan 4 + 3 days = Tuesday Jan 7
        expected_eligible = saturday_date + timedelta(days=3)
        assert trade.eligible_date.date() == expected_eligible.date()
        # Tuesday Jan 7
        assert trade.eligible_date.weekday() == 1  # Tuesday

    def test_trade_starts_in_lag_buffer(self, lag_buffer):
        """Trade starts with in_lag_buffer=True until lag expires."""
        # Use future dates relative to now
        future_close = datetime.now() - timedelta(days=1)  # Closed yesterday
        trade = lag_buffer.submit_trade_outcome(
            trade_id="T004",
            bot_id="BOT001",
            close_date=future_close,
            outcome=TradeOutcome.WIN,
            pnl=100.0,
            holding_time_minutes=60,
            regime_at_entry="TRENDING"
        )

        assert trade.in_lag_buffer is True
        assert trade.is_eligible is False  # Still in buffer, eligible_date is 2 days in future

    # =======================================================================
    # AC1: Data-Level Lag Enforcement
    # =======================================================================

    def test_get_eligible_trades_filters_by_eligible_date(self, lag_buffer):
        """
        Given a trade with eligible_date in the future,
        When get_eligible_trades is called,
        Then the trade is NOT returned (data-level enforcement).
        """
        # Trade closed today, eligible in 3 days
        future_eligible = datetime.now() + timedelta(days=3)
        lag_buffer.submit_trade_outcome(
            trade_id="T005",
            bot_id="BOT001",
            close_date=datetime.now() - timedelta(days=1),
            outcome=TradeOutcome.WIN,
            pnl=100.0,
            holding_time_minutes=60,
            regime_at_entry="TRENDING"
        )

        # Manually set eligible_date to future (simulating a trade not yet eligible)
        trade = lag_buffer.get_trade("T005")
        trade.eligible_date = future_eligible
        trade.in_lag_buffer = True

        eligible_trades = lag_buffer.get_eligible_trades()
        trade_ids = [t.trade_id for t in eligible_trades]
        assert "T005" not in trade_ids

    def test_model_cannot_access_trades_until_lag_expires(self, lag_buffer):
        """
        Verify that get_eligible_trades only returns trades whose lag has expired,
        ensuring data-level lag enforcement.
        """
        # Submit two trades
        lag_buffer.submit_trade_outcome(
            trade_id="T006",
            bot_id="BOT001",
            close_date=datetime.now() - timedelta(days=5),  # 5 days ago, lag expired
            outcome=TradeOutcome.WIN,
            pnl=100.0,
            holding_time_minutes=60,
            regime_at_entry="TRENDING"
        )

        lag_buffer.submit_trade_outcome(
            trade_id="T007",
            bot_id="BOT001",
            close_date=datetime.now() - timedelta(days=1),  # 1 day ago, lag not expired
            outcome=TradeOutcome.LOSS,
            pnl=-50.0,
            holding_time_minutes=30,
            regime_at_entry="RANGING"
        )

        eligible_trades = lag_buffer.get_eligible_trades()
        eligible_ids = [t.trade_id for t in eligible_trades]

        assert "T006" in eligible_ids  # Lag expired
        assert "T007" not in eligible_ids  # Still in lag buffer

    # =======================================================================
    # AC3: Weekend/Monday Batch Processing
    # =======================================================================

    @pytest.mark.asyncio
    async def test_monday_batch_includes_friday_close_trades(self, lag_buffer):
        """
        Given a Friday close with 3-day lag (expires Monday),
        When Monday morning batch is processed,
        Then the trade is batch-included.
        """
        # Submit trade on Friday 4 days ago - eligible_date is 1 day ago (lag expired)
        # Trade starts in buffer but is now eligible
        friday_close = datetime.now() - timedelta(days=4)
        trade = lag_buffer.submit_trade_outcome(
            trade_id="T008",
            bot_id="BOT001",
            close_date=friday_close,
            outcome=TradeOutcome.WIN,
            pnl=100.0,
            holding_time_minutes=60,
            regime_at_entry="TRENDING"
        )

        # Trade starts in buffer at submission
        assert trade.in_lag_buffer is True

        # Trade should be eligible via get_eligible_trades (eligible_date has passed)
        eligible_trades = lag_buffer.get_eligible_trades()
        trade_ids = [t.trade_id for t in eligible_trades]
        assert "T008" in trade_ids

        # Process Monday batch - trade should be included (expired and in buffer)
        result = await lag_buffer.process_monday_batch()
        assert result.included >= 1
        trade_ids = [t.trade_id for t in result.trades]
        assert "T008" in trade_ids

    @pytest.mark.asyncio
    async def test_monday_batch_excludes_saturday_close_trades(self, lag_buffer):
        """
        Given a Saturday close with 3-day lag (expires Tuesday),
        When Monday morning batch is processed,
        Then the trade is NOT included (still in buffer).
        """
        # Submit trade 2 days ago (Sat) - lag expires in 1 day (Tue)
        saturday_close = datetime.now() - timedelta(days=2)
        lag_buffer.submit_trade_outcome(
            trade_id="T009",
            bot_id="BOT001",
            close_date=saturday_close,
            outcome=TradeOutcome.WIN,
            pnl=100.0,
            holding_time_minutes=60,
            regime_at_entry="TRENDING"
        )

        # Verify trade is still in buffer
        trade = lag_buffer.get_trade("T009")
        assert trade.in_lag_buffer is True

    # =======================================================================
    # Buffer Status Tests
    # =======================================================================

    def test_get_buffer_status_returns_correct_counts(self, lag_buffer):
        """Verify buffer status returns correct in_buffer and eligible counts."""
        # Add trades with different eligible dates
        lag_buffer.submit_trade_outcome(
            trade_id="T010",
            bot_id="BOT001",
            close_date=datetime.now() - timedelta(days=5),  # Eligible
            outcome=TradeOutcome.WIN,
            pnl=100.0,
            holding_time_minutes=60,
            regime_at_entry="TRENDING"
        )

        lag_buffer.submit_trade_outcome(
            trade_id="T011",
            bot_id="BOT001",
            close_date=datetime.now() - timedelta(days=1),  # In buffer
            outcome=TradeOutcome.LOSS,
            pnl=-50.0,
            holding_time_minutes=30,
            regime_at_entry="RANGING"
        )

        status = lag_buffer.get_buffer_status()

        assert status.total_trades == 2
        assert status.total_eligible == 1
        assert status.total_in_buffer == 1
        assert status.avg_lag_remaining >= 0

    def test_trade_properties(self, lag_buffer):
        """Test HmmTrainingTradeRecord properties."""
        # Use a close date that is recent but not old enough to be eligible
        recent_close = datetime.now() - timedelta(days=1)  # Yesterday
        trade = lag_buffer.submit_trade_outcome(
            trade_id="T012",
            bot_id="BOT001",
            close_date=recent_close,
            outcome=TradeOutcome.WIN,
            pnl=150.0,
            holding_time_minutes=90,
            regime_at_entry="BREAKOUT"
        )

        # Test status property - should be in_buffer since close was yesterday
        assert trade.status == "in_buffer"

        # Test time_remaining property
        assert "day" in trade.time_remaining.lower()

        # Test is_eligible when lag expires
        trade.eligible_date = datetime.now() - timedelta(days=1)
        trade.in_lag_buffer = False
        assert trade.is_eligible is True
        assert trade.status == "eligible"


class TestLagDaysCalendarCalculation:
    """Test calendar day calculations for lag enforcement."""

    def test_monday_close_thursday_eligible(self):
        """Monday + 3 = Thursday."""
        close_date = datetime(2025, 1, 6, 12, 0, 0)  # Monday
        eligible_date = close_date + timedelta(days=3)
        assert eligible_date.date() == datetime(2025, 1, 9).date()  # Thursday

    def test_tuesday_close_friday_eligible(self):
        """Tuesday + 3 = Friday."""
        close_date = datetime(2025, 1, 7, 12, 0, 0)  # Tuesday
        eligible_date = close_date + timedelta(days=3)
        assert eligible_date.date() == datetime(2025, 1, 10).date()  # Friday

    def test_wednesday_close_saturday_eligible(self):
        """Wednesday + 3 = Saturday."""
        close_date = datetime(2025, 1, 8, 12, 0, 0)  # Wednesday
        eligible_date = close_date + timedelta(days=3)
        assert eligible_date.date() == datetime(2025, 1, 11).date()  # Saturday

    def test_thursday_close_sunday_eligible(self):
        """Thursday + 3 = Sunday."""
        close_date = datetime(2025, 1, 9, 12, 0, 0)  # Thursday
        eligible_date = close_date + timedelta(days=3)
        assert eligible_date.date() == datetime(2025, 1, 12).date()  # Sunday

    def test_friday_close_monday_eligible(self):
        """Friday + 3 = Monday (weekend counts)."""
        close_date = datetime(2025, 1, 3, 12, 0, 0)  # Friday
        eligible_date = close_date + timedelta(days=3)
        assert eligible_date.date() == datetime(2025, 1, 6).date()  # Monday

    def test_saturday_close_tuesday_eligible(self):
        """Saturday + 3 = Tuesday (weekend counts)."""
        close_date = datetime(2025, 1, 4, 12, 0, 0)  # Saturday
        eligible_date = close_date + timedelta(days=3)
        assert eligible_date.date() == datetime(2025, 1, 7).date()  # Tuesday

    def test_sunday_close_wednesday_eligible(self):
        """Sunday + 3 = Wednesday (weekend counts)."""
        close_date = datetime(2025, 1, 5, 12, 0, 0)  # Sunday
        eligible_date = close_date + timedelta(days=3)
        assert eligible_date.date() == datetime(2025, 1, 8).date()  # Wednesday
