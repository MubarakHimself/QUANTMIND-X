"""
Tests for Session Mask, Islamic Compliance & Loss Cap APIs (Story 3-3)

Tests:
- Bot params endpoint
- Force close countdown calculation
- Islamic compliance detection
- Loss cap breach events
"""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, patch, MagicMock


# =============================================================================
# Session Detection Tests
# =============================================================================

class TestSessionDetection:
    """Tests for trading session detection."""

    def test_detect_session_asian(self):
        """Test Asian session detection."""
        from src.router.sessions import SessionDetector, TradingSession

        # 00:00 UTC = 09:00 Tokyo (Asian session 00:00-09:00 local)
        # Use a weekday (Monday = 0)
        utc_time = datetime(2026, 3, 16, 0, 0, tzinfo=timezone.utc)  # Monday, midnight UTC
        session = SessionDetector.detect_session(utc_time)

        # At midnight UTC = 09:00 Tokyo, which is at the END of Asian session
        # Let's use 22:00 UTC = 07:00 Tokyo (within Asian session)
        utc_time2 = datetime(2026, 3, 16, 22, 0, tzinfo=timezone.utc)  # 22:00 UTC = 07:00 Tokyo
        session2 = SessionDetector.detect_session(utc_time2)

        assert session2 == TradingSession.ASIAN

    def test_detect_session_london(self):
        """Test London session detection."""
        from src.router.sessions import SessionDetector, TradingSession

        # 09:00 UTC = 09:00 London (London session starts)
        utc_time = datetime(2026, 3, 17, 9, 0, tzinfo=timezone.utc)
        session = SessionDetector.detect_session(utc_time)

        assert session == TradingSession.LONDON

    def test_detect_session_new_york(self):
        """Test New York session detection."""
        from src.router.sessions import SessionDetector, TradingSession

        # 14:00 UTC = 10:00 NY (New York session)
        utc_time = datetime(2026, 3, 17, 14, 0, tzinfo=timezone.utc)
        session = SessionDetector.detect_session(utc_time)

        # During overlap period might be OVERLAP
        assert session in [TradingSession.NEW_YORK, TradingSession.OVERLAP]

    def test_detect_session_closed_weekend(self):
        """Test closed session on weekend."""
        from src.router.sessions import SessionDetector, TradingSession

        # Saturday
        utc_time = datetime(2026, 3, 21, 10, 0, tzinfo=timezone.utc)
        session = SessionDetector.detect_session(utc_time)

        assert session == TradingSession.CLOSED


# =============================================================================
# Islamic Compliance Tests
# =============================================================================

class TestIslamicCompliance:
    """Tests for Islamic compliance functions."""

    def test_is_past_islamic_cutoff_before(self):
        """Test Islamic cutoff - before 21:45 UTC."""
        from src.router.sessions import is_past_islamic_cutoff

        # 20:00 UTC - before cutoff
        utc_time = datetime(2026, 3, 17, 20, 0, tzinfo=timezone.utc)
        result = is_past_islamic_cutoff(utc_time)

        assert result is False

    def test_is_past_islamic_cutoff_after(self):
        """Test Islamic cutoff - after 21:45 UTC."""
        from src.router.sessions import is_past_islamic_cutoff

        # 22:00 UTC - after cutoff
        utc_time = datetime(2026, 3, 17, 22, 0, tzinfo=timezone.utc)
        result = is_past_islamic_cutoff(utc_time)

        assert result is True

    def test_is_past_islamic_cutoff_at_cutoff(self):
        """Test Islamic cutoff - exactly at 21:45 UTC."""
        from src.router.sessions import is_past_islamic_cutoff

        # 21:45 UTC - at cutoff
        utc_time = datetime(2026, 3, 17, 21, 45, tzinfo=timezone.utc)
        result = is_past_islamic_cutoff(utc_time)

        assert result is True

    def test_get_force_close_countdown_within_window(self):
        """Test force close countdown - within 60 min window."""
        from src.router.sessions import get_force_close_countdown_seconds

        # 21:00 UTC - within 60 min of 21:45 cutoff
        utc_time = datetime(2026, 3, 17, 21, 0, tzinfo=timezone.utc)
        result = get_force_close_countdown_seconds(utc_time)

        assert result is not None
        assert 0 <= result <= 2700  # 45 minutes = 2700 seconds

    def test_get_force_close_countdown_outside_window(self):
        """Test force close countdown - outside window."""
        from src.router.sessions import get_force_close_countdown_seconds

        # 19:00 UTC - outside 60 min window
        utc_time = datetime(2026, 3, 17, 19, 0, tzinfo=timezone.utc)
        result = get_force_close_countdown_seconds(utc_time)

        assert result is None

    def test_is_within_countdown_window(self):
        """Test countdown window detection."""
        from src.router.sessions import is_within_countdown_window

        # 21:00 UTC - within window
        utc_time = datetime(2026, 3, 17, 21, 0, tzinfo=timezone.utc)
        assert is_within_countdown_window(utc_time) is True

        # 19:00 UTC - outside window
        utc_time = datetime(2026, 3, 17, 19, 0, tzinfo=timezone.utc)
        assert is_within_countdown_window(utc_time) is False


# =============================================================================
# Loss Cap Breach Tests
# =============================================================================

class TestLossCapBreach:
    """Tests for loss cap breach detection and logging."""

    def test_loss_cap_audit_log_append(self):
        """Test loss cap audit log append."""
        from src.router.sessions import LossCapAuditLog, _get_loss_cap_audit_log

        # Get a fresh audit log for testing
        audit_log = LossCapAuditLog()

        entry = {
            "bot_id": "test_bot",
            "loss_pct": -6.5,
            "daily_loss_cap": 5.0,
            "excess_pct": 1.5
        }

        entry_id = audit_log.append(entry)

        assert entry_id is not None
        assert len(audit_log.get_all()) == 1
        assert audit_log.get_all()[0]["bot_id"] == "test_bot"

    def test_loss_cap_audit_log_get_by_bot(self):
        """Test getting breach events by bot."""
        from src.router.sessions import LossCapAuditLog

        audit_log = LossCapAuditLog()

        # Add entries for two bots
        audit_log.append({"bot_id": "bot1", "loss_pct": -6.0})
        audit_log.append({"bot_id": "bot1", "loss_pct": -7.0})
        audit_log.append({"bot_id": "bot2", "loss_pct": -5.5})

        bot1_breaches = audit_log.get_by_bot("bot1")
        assert len(bot1_breaches) == 2

        bot2_breaches = audit_log.get_by_bot("bot2")
        assert len(bot2_breaches) == 1

    @pytest.mark.asyncio
    async def test_check_loss_cap_breach_breached(self):
        """Test loss cap breach detection when breached."""
        from src.router.sessions import check_and_notify_loss_cap_breach, LossCapAuditLog

        # Create a fresh audit log for this test
        with patch('src.router.sessions._loss_cap_audit_log', LossCapAuditLog()):
            with patch('src.api.websocket_endpoints.manager') as mock_manager:
                mock_manager.broadcast = AsyncMock()

                result = await check_and_notify_loss_cap_breach(
                    bot_id="test_bot",
                    current_loss_pct=-6.0,  # 6% loss
                    daily_loss_cap=5.0,    # 5% cap
                    account_equity=9400.0,
                    account_balance=10000.0
                )

                assert result is not None
                assert result["bot_id"] == "test_bot"
                assert result["loss_pct"] == -6.0
                assert result["excess_pct"] == 1.0
                mock_manager.broadcast.assert_called_once()

    @pytest.mark.asyncio
    async def test_check_loss_cap_breach_not_breached(self):
        """Test loss cap breach detection when not breached."""
        from src.router.sessions import check_and_notify_loss_cap_breach, LossCapAuditLog

        # Create a fresh audit log for this test
        with patch('src.router.sessions._loss_cap_audit_log', LossCapAuditLog()):
            with patch('src.api.websocket_endpoints.manager') as mock_manager:
                mock_manager.broadcast = AsyncMock()

                result = await check_and_notify_loss_cap_breach(
                    bot_id="test_bot",
                    current_loss_pct=-3.0,  # 3% loss
                    daily_loss_cap=5.0,     # 5% cap
                    account_equity=9700.0,
                    account_balance=10000.0
                )

                assert result is None
                mock_manager.broadcast.assert_not_called()


# =============================================================================
# API Endpoint Tests
# =============================================================================

class TestBotParamsEndpoint:
    """Tests for bot params API endpoint."""

    def test_bot_params_response_model(self):
        """Test BotParamsResponse model validation."""
        from src.api.trading.models import BotParamsResponse

        # Valid response
        response = BotParamsResponse(
            bot_id="test_bot",
            session_mask="LONDON",
            force_close_hour=21,
            overnight_hold=True,
            daily_loss_cap=5.0,
            current_loss_pct=-2.5,
            islamic_compliance=True,
            swap_free=True,
            force_close_countdown_seconds=1800
        )

        assert response.bot_id == "test_bot"
        assert response.session_mask == "LONDON"
        assert response.force_close_countdown_seconds == 1800

    def test_bot_params_response_model_no_countdown(self):
        """Test BotParamsResponse without countdown."""
        from src.api.trading.models import BotParamsResponse

        response = BotParamsResponse(
            bot_id="test_bot",
            session_mask="CLOSED",
            force_close_hour=21,
            overnight_hold=False,
            daily_loss_cap=5.0,
            current_loss_pct=1.0,
            islamic_compliance=False,
            swap_free=False,
            force_close_countdown_seconds=None
        )

        assert response.force_close_countdown_seconds is None


class TestLossCapEndpoints:
    """Tests for loss cap breach API endpoints."""

    def test_get_loss_cap_audit_logs(self):
        """Test getting all loss cap audit logs."""
        from src.router.sessions import get_loss_cap_audit_logs, LossCapAuditLog

        with patch('src.router.sessions._loss_cap_audit_log', LossCapAuditLog()):
            logs = get_loss_cap_audit_logs()
            assert isinstance(logs, list)

    def test_get_loss_cap_breach_by_bot(self):
        """Test getting breach events by bot."""
        from src.router.sessions import get_loss_cap_breach_by_bot, LossCapAuditLog

        with patch('src.router.sessions._loss_cap_audit_log', LossCapAuditLog()):
            breaches = get_loss_cap_breach_by_bot("test_bot")
            assert isinstance(breaches, list)
