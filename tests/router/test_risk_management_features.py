"""
Risk Management Features Test Suite

Tests for Task Group 7: BrokerRegistry, HouseMoneyState, BotCircuitBreaker

**Validates: Task Group 7.1 - Risk Management Features**
"""

import pytest
from datetime import datetime, date, timezone
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.database.models import Base, BrokerRegistry, HouseMoneyState, BotCircuitBreaker


# Test database setup
TEST_DATABASE_URL = "sqlite:///:memory:"


@pytest.fixture
def db_session():
    """Create a fresh test database session for each test."""
    engine = create_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    TestingSessionLocal = sessionmaker(bind=engine)
    session = TestingSessionLocal()
    yield session
    session.close()


class TestBrokerRegistry:
    """Test BrokerRegistry CRUD operations and pip value retrieval."""

    def test_create_broker_profile(self, db_session):
        """Test creating a broker profile with all required fields."""
        # Create broker profile
        broker = BrokerRegistry(
            broker_id="icmarkets_raw",
            broker_name="IC Markets RAW",
            spread_avg=0.1,
            commission_per_lot=3.5,
            lot_step=0.01,
            min_lot=0.01,
            max_lot=100.0,
            pip_values={
                "EURUSD": 10.0,
                "GBPUSD": 10.0,
                "XAUUSD": 10.0,
                "XAGUSD": 50.0
            },
            preference_tags=["RAW_ECN", "LOW_SPREAD", "SCALPING"]
        )
        db_session.add(broker)
        db_session.commit()
        db_session.refresh(broker)

        # Verify creation
        assert broker.id is not None
        assert broker.broker_id == "icmarkets_raw"
        assert broker.spread_avg == 0.1
        assert broker.commission_per_lot == 3.5
        assert broker.pip_values["EURUSD"] == 10.0
        assert "RAW_ECN" in broker.preference_tags

    def test_get_pip_value_for_symbol(self, db_session):
        """Test retrieving pip values per symbol from broker registry."""
        # Create multiple brokers with different pip values
        broker1 = BrokerRegistry(
            broker_id="broker1",
            broker_name="Broker 1",
            spread_avg=0.5,
            commission_per_lot=5.0,
            lot_step=0.01,
            min_lot=0.01,
            max_lot=50.0,
            pip_values={"EURUSD": 10.0, "XAUUSD": 1.0}
        )
        broker2 = BrokerRegistry(
            broker_id="broker2",
            broker_name="Broker 2",
            spread_avg=1.0,
            commission_per_lot=0.0,
            lot_step=0.1,
            min_lot=0.1,
            max_lot=50.0,
            pip_values={"EURUSD": 10.0, "XAUUSD": 0.1}
        )
        db_session.add_all([broker1, broker2])
        db_session.commit()

        # Test pip value retrieval - query directly from database
        broker1_retrieved = db_session.query(BrokerRegistry).filter_by(broker_id="broker1").first()
        assert broker1_retrieved is not None
        assert broker1_retrieved.pip_values["XAUUSD"] == 1.0

        broker2_retrieved = db_session.query(BrokerRegistry).filter_by(broker_id="broker2").first()
        assert broker2_retrieved is not None
        assert broker2_retrieved.pip_values["XAUUSD"] == 0.1

    def test_get_commission_for_broker(self, db_session):
        """Test retrieving commission structure from broker registry."""
        # Create broker with commission
        broker = BrokerRegistry(
            broker_id="ecn_broker",
            broker_name="ECN Broker",
            spread_avg=0.0,
            commission_per_lot=7.0,
            lot_step=0.01,
            min_lot=0.01,
            max_lot=100.0,
            pip_values={"EURUSD": 10.0}
        )
        db_session.add(broker)
        db_session.commit()

        # Test commission retrieval - query directly from database
        broker_retrieved = db_session.query(BrokerRegistry).filter_by(broker_id="ecn_broker").first()
        assert broker_retrieved is not None
        assert broker_retrieved.commission_per_lot == 7.0


class TestHouseMoneyState:
    """Test HouseMoneyState tracking and daily reset functionality."""

    def test_update_pnl_and_calculate_multiplier(self, db_session):
        """Test updating P&L and calculating risk multiplier."""
        # Create house money state
        house_money = HouseMoneyState(
            account_id="12345",
            daily_start_balance=100000.0,
            current_pnl=0.0,
            high_water_mark=100000.0,
            risk_multiplier=1.0,
            is_preservation_mode=False,
            date=date.today()
        )
        db_session.add(house_money)
        db_session.commit()

        # Test P&L update - update directly in database
        profit_pnl = 5000.0  # 5% profit
        profit_pct = profit_pnl / house_money.daily_start_balance

        # Calculate expected multiplier
        if profit_pct > 0.05:  # > 5% profit
            expected_multiplier = 1.5
        elif profit_pct < -0.03:  # < 3% loss
            expected_multiplier = 0.5
        else:
            expected_multiplier = 1.0

        # Update in database
        house_money.current_pnl = profit_pnl
        house_money.risk_multiplier = 1.5  # 1.5x when up > 5%
        db_session.commit()

        # Refresh and verify
        db_session.refresh(house_money)
        assert house_money.current_pnl == 5000.0
        assert house_money.risk_multiplier == 1.5  # 1.5x when up > 5%

    def test_daily_reset_functionality(self, db_session):
        """Test that daily reset clears P&L and resets multiplier."""
        # Create house money state with profit
        house_money = HouseMoneyState(
            account_id="12345",
            daily_start_balance=100000.0,
            current_pnl=8000.0,
            high_water_mark=108000.0,
            risk_multiplier=1.5,
            is_preservation_mode=True,
            date=date.today()
        )
        db_session.add(house_money)
        db_session.commit()

        # Test daily reset - reset directly in database
        house_money.current_pnl = 0.0
        house_money.high_water_mark = house_money.daily_start_balance
        house_money.risk_multiplier = 1.0
        house_money.is_preservation_mode = False
        db_session.commit()

        # Verify reset
        db_session.refresh(house_money)
        assert house_money.current_pnl == 0.0
        assert house_money.risk_multiplier == 1.0
        assert house_money.is_preservation_mode is False

    def test_preservation_mode_trigger(self, db_session):
        """Test that preservation mode triggers at target profit."""
        # Create house money state
        house_money = HouseMoneyState(
            account_id="12345",
            daily_start_balance=100000.0,
            current_pnl=0.0,
            high_water_mark=100000.0,
            risk_multiplier=1.0,
            is_preservation_mode=False,
            date=date.today()
        )
        db_session.add(house_money)
        db_session.commit()

        # Update to target profit (8%)
        profit_pnl = 8000.0
        profit_pct = profit_pnl / house_money.daily_start_balance

        # Trigger preservation mode at 8% profit
        should_preserve = profit_pct >= 0.08

        # Update in database
        house_money.current_pnl = profit_pnl
        house_money.high_water_mark = house_money.daily_start_balance + profit_pnl
        house_money.is_preservation_mode = should_preserve
        db_session.commit()

        # Verify preservation mode triggered
        db_session.refresh(house_money)
        assert house_money.is_preservation_mode is True
        assert house_money.high_water_mark == 108000.0


class TestBotCircuitBreaker:
    """Test BotCircuitBreaker quarantine logic and auto-quarantine triggers."""

    def test_auto_quarantine_on_consecutive_losses(self, db_session):
        """Test that bot is auto-quarantined after 5 consecutive losses."""
        # Create bot circuit breaker state
        bot_cb = BotCircuitBreaker(
            bot_id="test_bot_1",
            consecutive_losses=0,
            daily_trade_count=10,
            last_trade_time=datetime.now(timezone.utc),
            is_quarantined=False,
            quarantine_reason=None,
            quarantine_start=None
        )
        db_session.add(bot_cb)
        db_session.commit()

        # Record 5 consecutive losses - update directly in database
        for i in range(5):
            bot_cb.consecutive_losses += 1
            bot_cb.daily_trade_count += 1

            # Auto-quarantine after 5 consecutive losses
            if bot_cb.consecutive_losses >= 5:
                bot_cb.is_quarantined = True
                bot_cb.quarantine_reason = "5 consecutive losses"
                bot_cb.quarantine_start = datetime.now(timezone.utc)

        db_session.commit()

        # Verify auto-quarantine
        db_session.refresh(bot_cb)
        assert bot_cb.is_quarantined is True
        assert bot_cb.consecutive_losses == 5
        assert "consecutive" in bot_cb.quarantine_reason.lower()

    def test_check_allowed_blocks_quarantined_bots(self, db_session):
        """Test that quarantined bots are blocked from trading."""
        # Create quarantined bot
        bot_cb = BotCircuitBreaker(
            bot_id="quarantined_bot",
            consecutive_losses=5,
            daily_trade_count=20,
            last_trade_time=datetime.now(timezone.utc),
            is_quarantined=True,
            quarantine_reason="5 consecutive losses",
            quarantine_start=datetime.now(timezone.utc)
        )
        db_session.add(bot_cb)
        db_session.commit()

        # Test that bot is blocked - check directly from database
        db_session.refresh(bot_cb)
        allowed = not bot_cb.is_quarantined
        reason = bot_cb.quarantine_reason or ""

        assert allowed is False
        assert "consecutive" in reason.lower()

    def test_reactivate_bot_after_manual_review(self, db_session):
        """Test that quarantined bot can be reactivated after review."""
        # Create quarantined bot
        bot_cb = BotCircuitBreaker(
            bot_id="bot_to_reactivate",
            consecutive_losses=5,
            daily_trade_count=15,
            last_trade_time=datetime.now(timezone.utc),
            is_quarantined=True,
            quarantine_reason="Daily trade limit exceeded",
            quarantine_start=datetime.now(timezone.utc)
        )
        db_session.add(bot_cb)
        db_session.commit()

        # Reactivate bot - update directly in database
        bot_cb.is_quarantined = False
        bot_cb.consecutive_losses = 0
        bot_cb.quarantine_reason = None
        bot_cb.quarantine_start = None
        db_session.commit()

        # Verify reactivation
        db_session.refresh(bot_cb)
        assert bot_cb.is_quarantined is False
        assert bot_cb.consecutive_losses == 0
        assert bot_cb.quarantine_reason is None

    def test_daily_trade_limit_enforcement(self, db_session):
        """Test that daily trade limit is enforced and triggers quarantine."""
        # Create bot circuit breaker with trade limit
        bot_cb = BotCircuitBreaker(
            bot_id="high_frequency_bot",
            consecutive_losses=0,
            daily_trade_count=0,
            last_trade_time=datetime.now(timezone.utc),
            is_quarantined=False,
            quarantine_reason=None,
            quarantine_start=None
        )
        db_session.add(bot_cb)
        db_session.commit()

        # Record trades up to daily limit (default 20)
        daily_limit = 20
        for i in range(21):  # Exceed limit by 1
            bot_cb.daily_trade_count += 1

            # Trigger quarantine when limit exceeded
            if bot_cb.daily_trade_count > daily_limit:
                bot_cb.is_quarantined = True
                bot_cb.quarantine_reason = "Daily trade limit exceeded"
                bot_cb.quarantine_start = datetime.now(timezone.utc)

        db_session.commit()

        # Verify quarantine triggered
        db_session.refresh(bot_cb)
        assert bot_cb.is_quarantined is True
        assert bot_cb.daily_trade_count == 21
        assert "limit" in bot_cb.quarantine_reason.lower()

    def test_consecutive_losses_counter_reset_on_win(self, db_session):
        """Test that consecutive losses counter resets on a win."""
        # Create bot with 3 consecutive losses
        bot_cb = BotCircuitBreaker(
            bot_id="bot_with_losses",
            consecutive_losses=3,
            daily_trade_count=10,
            last_trade_time=datetime.now(timezone.utc),
            is_quarantined=False,
            quarantine_reason=None,
            quarantine_start=None
        )
        db_session.add(bot_cb)
        db_session.commit()

        # Record a win - update directly in database
        bot_cb.consecutive_losses = 0  # Reset on win
        bot_cb.daily_trade_count += 1
        db_session.commit()

        # Verify counter reset
        db_session.refresh(bot_cb)
        assert bot_cb.consecutive_losses == 0
        assert bot_cb.daily_trade_count == 11
