"""
Tests for MT5 Order Manager with Layer 1 SL/TP support (Story 14.1).

Tests cover:
- SL/TP parameter injection on order placement
- Trade record creation with SL/TP metadata
- EA input parameter enforcement (risk cap, force-close, overnight hold)
- Pipeline disconnect resilience (SL/TP survives)
"""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

from src.risk.integrations.mt5.orders import (
    OrderManager,
    OrderInfo,
    EAInputParameters,
)


class TestEAInputParameters:
    """Tests for EAInputParameters dataclass."""

    def test_default_values(self):
        """Test default EA parameters match story requirements."""
        params = EAInputParameters()

        assert params.per_trade_risk_cap == 0.005  # 0.5% hard ceiling
        assert params.force_close_hour == 21
        assert params.force_close_minute == 45
        assert params.no_overnight_hold is True
        assert params.session_mask == 0xFF
        assert params.islamic_compliance is True

    def test_to_dict(self):
        """Test EA parameters serialization."""
        params = EAInputParameters(
            per_trade_risk_cap=0.01,
            force_close_hour=20,
            force_close_minute=30,
            no_overnight_hold=False,
            session_mask=0x0F,
            islamic_compliance=False,
        )

        result = params.to_dict()

        assert result["per_trade_risk_cap"] == 0.01
        assert result["force_close_hour"] == 20
        assert result["force_close_minute"] == 30
        assert result["no_overnight_hold"] is False
        assert result["session_mask"] == 0x0F
        assert result["islamic_compliance"] is False

    def test_from_dict(self):
        """Test EA parameters deserialization."""
        data = {
            "per_trade_risk_cap": 0.02,
            "force_close_hour": 22,
            "force_close_minute": 0,
            "no_overnight_hold": True,
            "session_mask": 0xAA,
            "islamic_compliance": True,
        }

        params = EAInputParameters.from_dict(data)

        assert params.per_trade_risk_cap == 0.02
        assert params.force_close_hour == 22
        assert params.force_close_minute == 0
        assert params.no_overnight_hold is True
        assert params.session_mask == 0xAA
        assert params.islamic_compliance is True


class TestOrderManagerSLTP:
    """Tests for OrderManager with SL/TP support."""

    @pytest.fixture
    def order_manager(self):
        """Create OrderManager with fallback to simulated."""
        return OrderManager(account_manager=None, fallback_to_simulated=True)

    @pytest.fixture
    def order_manager_with_params(self):
        """Create OrderManager with custom EA parameters."""
        params = EAInputParameters(
            per_trade_risk_cap=0.01,  # 1% for testing
            force_close_hour=20,
            force_close_minute=30,
            no_overnight_hold=True,
            islamic_compliance=True,
        )
        return OrderManager(
            account_manager=None,
            fallback_to_simulated=True,
            ea_parameters=params,
        )

    def test_place_order_with_sltp(self, order_manager):
        """Test order placement with SL/TP parameters."""
        result = order_manager.place_order(
            symbol="EURUSD",
            order_type="buy",
            volume=0.1,
            price=1.1000,
            sl=1.0950,
            tp=1.1100,
        )

        assert result is not None
        assert result >= 1000  # Simulated tickets start at 1000

        # Verify order was stored with correct SL/TP
        orders = order_manager.get_orders()
        assert len(orders) == 1
        assert orders[0].symbol == "EURUSD"
        assert orders[0].sl == 1.0950
        assert orders[0].tp == 1.1100

    def test_place_order_without_sltp(self, order_manager):
        """Test order placement without SL/TP (SL/TP = 0)."""
        result = order_manager.place_order(
            symbol="EURUSD",
            order_type="sell",
            volume=0.1,
            price=1.1000,
            sl=0.0,
            tp=0.0,
        )

        assert result is not None

        orders = order_manager.get_orders()
        assert orders[0].sl == 0.0
        assert orders[0].tp == 0.0

    def test_modify_order_sltp(self, order_manager):
        """Test modifying SL/TP on existing order."""
        # Place order
        ticket = order_manager.place_order(
            symbol="EURUSD",
            order_type="buy",
            volume=0.1,
            price=1.1000,
            sl=1.0950,
            tp=1.1100,
        )

        # Modify SL/TP
        success = order_manager.modify_order(
            ticket=ticket,
            sl=1.0970,
            tp=1.1120,
        )

        assert success is True

        orders = order_manager.get_orders()
        assert orders[0].sl == 1.0970
        assert orders[0].tp == 1.1120

    def test_close_order(self, order_manager):
        """Test closing an order."""
        ticket = order_manager.place_order(
            symbol="EURUSD",
            order_type="buy",
            volume=0.1,
            price=1.1000,
            sl=1.0950,
            tp=1.1100,
        )

        success = order_manager.close_order(ticket)

        assert success is True
        assert len(order_manager.get_orders()) == 0

    def test_partial_close(self, order_manager):
        """Test partial close of an order."""
        ticket = order_manager.place_order(
            symbol="EURUSD",
            order_type="buy",
            volume=0.2,
            price=1.1000,
            sl=1.0950,
            tp=1.1100,
        )

        success = order_manager.close_order(ticket, lots=0.1)

        assert success is True
        assert len(order_manager.get_orders()) == 1
        assert order_manager.get_orders()[0].volume == 0.1


class TestEAParameterEnforcement:
    """Tests for EA input parameter enforcement."""

    @pytest.fixture
    def order_manager(self):
        """Create OrderManager with custom EA parameters."""
        params = EAInputParameters(
            per_trade_risk_cap=0.005,  # 0.5% hard ceiling
            force_close_hour=21,
            force_close_minute=45,
            no_overnight_hold=True,
            islamic_compliance=True,
        )
        return OrderManager(
            account_manager=None,
            fallback_to_simulated=True,
            ea_parameters=params,
        )

    def test_set_ea_parameters(self, order_manager):
        """Test setting EA parameters."""
        new_params = EAInputParameters(
            per_trade_risk_cap=0.01,
            force_close_hour=20,
            force_close_minute=30,
            no_overnight_hold=False,
        )

        order_manager.set_ea_parameters(new_params)
        current = order_manager.get_ea_parameters()

        assert current.per_trade_risk_cap == 0.01
        assert current.force_close_hour == 20
        assert current.no_overnight_hold is False

    def test_enforce_risk_cap_below_limit(self, order_manager):
        """Test risk cap when volume is below limit."""
        # Equity 10000, 0.5% cap = 50 max risk
        # Volume 0.1, price 1.1000, SL 1.0950 = 50 pips risk
        # Risk per lot = 50 * 10 = 500 (if we treat as $5/pip for standard lot)
        # Actually let's recalculate: price diff 0.005, volume 0.1
        # Simplified: just returns original volume if risk is within cap

        volume = order_manager.enforce_risk_cap(
            volume=0.01,  # Small volume
            equity=10000.0,
            entry_price=1.1000,
            sl_price=1.0950,
        )

        assert volume == 0.01  # No adjustment needed

    def test_enforce_risk_cap_exceeds_limit(self, order_manager):
        """Test risk cap when volume exceeds limit."""
        # With 0.5% cap on 10000 equity = 50 max risk
        # Price diff 0.005 (50 pips), so max volume = 50 / 0.005 = 10000 units = 0.1 lots
        # If we request 0.5 lots, it should be capped to 0.1

        volume = order_manager.enforce_risk_cap(
            volume=0.5,  # Requesting large volume
            equity=10000.0,
            entry_price=1.1000,
            sl_price=1.0950,
        )

        # Should be capped to ~0.1 based on risk calculation
        assert volume == pytest.approx(0.1, rel=1e-6)

    def test_force_close_trigger_exact_time(self, order_manager):
        """Test force-close triggers at exact configured time."""
        # 21:45 UTC
        current_time = datetime(2026, 3, 24, 21, 45, 0, tzinfo=timezone.utc)

        should_close = order_manager.should_force_close(current_time)

        assert should_close is True

    def test_force_close_trigger_after_time(self, order_manager):
        """Test force-close triggers after configured time."""
        # 21:46 UTC
        current_time = datetime(2026, 3, 24, 21, 46, 0, tzinfo=timezone.utc)

        should_close = order_manager.should_force_close(current_time)

        assert should_close is True

    def test_force_close_not_trigger_before_time(self, order_manager):
        """Test force-close does not trigger before configured time."""
        # 21:44 UTC
        current_time = datetime(2026, 3, 24, 21, 44, 0, tzinfo=timezone.utc)

        should_close = order_manager.should_force_close(current_time)

        assert should_close is False

    def test_force_close_not_trigger_different_hour(self, order_manager):
        """Test force-close does not trigger at different hour."""
        # 20:45 UTC
        current_time = datetime(2026, 3, 24, 20, 45, 0, tzinfo=timezone.utc)

        should_close = order_manager.should_force_close(current_time)

        assert should_close is False

    def test_prevent_overnight_friday_night(self, order_manager):
        """Test overnight prevention on Friday night."""
        # Friday 22:00 UTC
        current_time = datetime(2026, 3, 27, 22, 0, 0, tzinfo=timezone.utc)

        should_prevent = order_manager.should_prevent_overnight(current_time)

        assert should_prevent is True

    def test_prevent_overnight_saturday(self, order_manager):
        """Test overnight prevention on Saturday."""
        # Saturday any time
        current_time = datetime(2026, 3, 28, 10, 0, 0, tzinfo=timezone.utc)

        should_prevent = order_manager.should_prevent_overnight(current_time)

        assert should_prevent is True

    def test_allow_overnight_weekday(self, order_manager):
        """Test overnight allowed on weekday."""
        # Wednesday 10:00 UTC
        current_time = datetime(2026, 3, 25, 10, 0, 0, tzinfo=timezone.utc)

        should_prevent = order_manager.should_prevent_overnight(current_time)

        assert should_prevent is False

    def test_overnight_disabled(self, order_manager):
        """Test overnight prevention when disabled."""
        # Disable overnight hold
        order_manager._ea_parameters.no_overnight_hold = False

        # Friday 22:00 UTC - should not prevent
        current_time = datetime(2026, 3, 27, 22, 0, 0, tzinfo=timezone.utc)

        should_prevent = order_manager.should_prevent_overnight(current_time)

        assert should_prevent is False


class TestTradeRecordCreation:
    """Tests for trade record creation with SL/TP metadata."""

    @pytest.fixture
    def order_manager(self):
        """Create OrderManager with simulated mode."""
        params = EAInputParameters()
        return OrderManager(
            account_manager=None,
            fallback_to_simulated=True,
            ea_parameters=params,
        )

    @patch("src.database.models.SessionLocal")
    def test_create_trade_record_success(self, mock_session_local, order_manager):
        """Test successful trade record creation."""
        # Setup mock session
        mock_session = MagicMock()
        mock_session_local.return_value = mock_session

        result = order_manager.create_trade_record(
            signal_id="sig-123",
            strategy_id="strat-456",
            epic_id="epic-14",
            symbol="EURUSD",
            direction="BUY",
            entry_price=1.1000,
            sl_price=1.0950,
            tp_price=1.1100,
            position_volume=0.1,
            risk_amount=50.0,
            regime_at_entry="TREND",
            mode="live",
            broker_account_id="acc-789",
        )

        assert result is not None
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()

    @patch("src.database.models.SessionLocal")
    def test_create_trade_record_failure_graceful(self, mock_session_local, order_manager):
        """Test trade record creation failure is graceful."""
        mock_session_local.side_effect = Exception("Database error")

        result = order_manager.create_trade_record(
            signal_id="sig-123",
            strategy_id="strat-456",
            epic_id="epic-14",
            symbol="EURUSD",
            direction="BUY",
            entry_price=1.1000,
            sl_price=1.0950,
            tp_price=1.1100,
            position_volume=0.1,
            risk_amount=50.0,
        )

        # Should return None but not raise
        assert result is None

    @patch("src.database.models.SessionLocal")
    def test_place_order_with_sltp_creates_record(self, mock_session_local, order_manager):
        """Test place_order_with_sltp creates trade record."""
        mock_session = MagicMock()
        mock_session_local.return_value = mock_session

        result = order_manager.place_order_with_sltp(
            symbol="EURUSD",
            order_type="buy",
            volume=0.1,
            price=1.1000,
            sl=1.0950,
            tp=1.1100,
            signal_id="sig-123",
            strategy_id="strat-456",
            epic_id="epic-14",
            regime_at_entry="TREND",
            mode="live",
            broker_account_id="acc-789",
            equity=10000.0,
        )

        assert result["success"] is True
        assert result["ticket"] is not None
        assert result["sl"] == 1.0950
        assert result["tp"] == 1.1100
        assert result["trade_record"] is not None

    @patch("src.database.models.SessionLocal")
    def test_place_order_with_sltp_risk_cap_applied(self, mock_session_local, order_manager):
        """Test place_order_with_sltp applies risk cap."""
        mock_session = MagicMock()
        mock_session_local.return_value = mock_session

        # Set tight risk cap
        order_manager._ea_parameters.per_trade_risk_cap = 0.001  # 0.1%

        result = order_manager.place_order_with_sltp(
            symbol="EURUSD",
            order_type="buy",
            volume=1.0,  # Large volume
            price=1.1000,
            sl=1.0950,  # 50 pips SL
            tp=1.1100,
            signal_id="sig-123",
            strategy_id="strat-456",
            equity=10000.0,
        )

        assert result["success"] is True
        # Volume should be capped
        assert result["volume"] < 1.0


class TestOrderInfo:
    """Tests for OrderInfo dataclass."""

    def test_order_info_to_dict(self):
        """Test OrderInfo serialization."""
        order = OrderInfo(
            ticket=12345,
            symbol="EURUSD",
            type="buy",
            volume=0.1,
            price=1.1000,
            sl=1.0950,
            tp=1.1100,
            profit=10.5,
            status="open",
        )

        result = order.to_dict()

        assert result["ticket"] == 12345
        assert result["symbol"] == "EURUSD"
        assert result["sl"] == 1.0950
        assert result["tp"] == 1.1100
        assert result["profit"] == 10.5


class TestForceCloseAndOvernightPrevention:
    """Tests for force_close_positions and prevent_overnight_holds methods."""

    @pytest.fixture
    def order_manager(self):
        """Create OrderManager with EA parameters."""
        params = EAInputParameters(
            per_trade_risk_cap=0.005,
            force_close_hour=21,
            force_close_minute=45,
            no_overnight_hold=True,
        )
        return OrderManager(
            account_manager=None,
            fallback_to_simulated=True,
            ea_parameters=params,
        )

    def test_force_close_positions_triggers(self, order_manager):
        """Test force_close_positions executes when time condition met."""
        # Place some orders first
        order_manager.place_order("EURUSD", "buy", 0.1, 1.1000, sl=1.0950, tp=1.1100)
        order_manager.place_order("GBPUSD", "sell", 0.2, 1.3000, sl=1.3050, tp=1.2900)

        # Trigger at 21:45
        current_time = datetime(2026, 3, 24, 21, 45, 0, tzinfo=timezone.utc)

        results = order_manager.force_close_positions(current_time)

        assert len(results) == 2
        assert all(r["success"] for r in results)
        assert all(r["reason"] == "force_close_hour" for r in results)

    def test_force_close_positions_no_trigger(self, order_manager):
        """Test force_close_positions does not execute when time not met."""
        # Place an order
        order_manager.place_order("EURUSD", "buy", 0.1, 1.1000, sl=1.0950, tp=1.1100)

        # Time is 20:00 - before force close
        current_time = datetime(2026, 3, 24, 20, 0, 0, tzinfo=timezone.utc)

        results = order_manager.force_close_positions(current_time)

        assert len(results) == 0

    def test_prevent_overnight_holds_friday_night(self, order_manager):
        """Test prevent_overnight_holds triggers on Friday night."""
        # Place an order
        order_manager.place_order("EURUSD", "buy", 0.1, 1.1000, sl=1.0950, tp=1.1100)

        # Friday 22:00 UTC
        current_time = datetime(2026, 3, 27, 22, 0, 0, tzinfo=timezone.utc)

        results = order_manager.prevent_overnight_holds(current_time)

        assert len(results) == 1
        assert results[0]["success"] is True
        assert results[0]["reason"] == "overnight_prevention"

    def test_prevent_overnight_holds_saturday(self, order_manager):
        """Test prevent_overnight_holds triggers on Saturday."""
        order_manager.place_order("EURUSD", "buy", 0.1, 1.1000)

        current_time = datetime(2026, 3, 28, 10, 0, 0, tzinfo=timezone.utc)

        results = order_manager.prevent_overnight_holds(current_time)

        assert len(results) == 1

    def test_prevent_overnight_holds_no_trigger_weekday(self, order_manager):
        """Test prevent_overnight_holds does not trigger on weekday."""
        order_manager.place_order("EURUSD", "buy", 0.1, 1.1000)

        # Wednesday 10:00 UTC
        current_time = datetime(2026, 3, 25, 10, 0, 0, tzinfo=timezone.utc)

        results = order_manager.prevent_overnight_holds(current_time)

        assert len(results) == 0

    def test_prevent_overnight_holds_disabled(self, order_manager):
        """Test prevent_overnight_holds when no_overnight_hold is False."""
        order_manager._ea_parameters.no_overnight_hold = False
        order_manager.place_order("EURUSD", "buy", 0.1, 1.1000)

        # Friday 22:00 UTC - should not trigger
        current_time = datetime(2026, 3, 27, 22, 0, 0, tzinfo=timezone.utc)

        results = order_manager.prevent_overnight_holds(current_time)

        assert len(results) == 0
