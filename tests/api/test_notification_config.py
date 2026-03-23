# tests/api/test_notification_config.py
"""
Tests for Notification Config API Endpoints (Story 10-5)

AC1: GET /api/notifications - returns events grouped by category with correct fields
AC2: PUT /api/notifications - toggles event notifications
AC3: High-priority events (kill_switch, loss_cap, system_critical) are always-on
"""
import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI
from unittest.mock import patch, MagicMock


@pytest.fixture
def test_app():
    """Create a test FastAPI app with the notification config router."""
    from src.api.notification_config_endpoints import router
    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture
def client(test_app):
    return TestClient(test_app)


def make_mock_config(event_type="trade_executed", category="trade", is_enabled=True, is_always_on=False):
    """Build a mock NotificationConfig ORM object."""
    m = MagicMock()
    m.event_type = event_type
    m.category = category
    m.is_enabled = is_enabled
    m.is_always_on = is_always_on
    m.description = f"Mock {event_type}"
    return m


class TestGetNotificationSettings:
    """AC1: GET /api/notifications returns all event types with correct fields."""

    @patch('src.api.notification_config_endpoints.get_session')
    def test_returns_200_with_events_and_categories(self, mock_get_session, client):
        """GET /api/notifications returns {events, categories} structure."""
        mock_session = MagicMock()
        mock_get_session.return_value = mock_session
        mock_session.query.return_value.all.return_value = [
            make_mock_config("trade_executed", "trade"),
            make_mock_config("kill_switch_triggered", "system", is_always_on=True),
        ]
        response = client.get("/api/notifications")
        assert response.status_code == 200
        data = response.json()
        assert "events" in data
        assert "categories" in data

    @patch('src.api.notification_config_endpoints.get_session')
    def test_event_has_required_fields(self, mock_get_session, client):
        """Each event object exposes event_type, category, severity, enabled, delivery_channel, is_always_on."""
        mock_session = MagicMock()
        mock_get_session.return_value = mock_session
        mock_session.query.return_value.all.return_value = [
            make_mock_config("trade_executed", "trade"),
        ]
        response = client.get("/api/notifications")
        assert response.status_code == 200
        event = response.json()["events"][0]
        for field in ("event_type", "category", "severity", "enabled", "delivery_channel", "is_always_on"):
            assert field in event, f"Missing field: {field}"

    @patch('src.api.notification_config_endpoints.get_session')
    def test_categories_list_is_sorted(self, mock_get_session, client):
        """Categories list is sorted alphabetically."""
        mock_session = MagicMock()
        mock_get_session.return_value = mock_session
        mock_session.query.return_value.all.return_value = [
            make_mock_config("trade_executed", "trade"),
            make_mock_config("agent_task_complete", "agent"),
            make_mock_config("kill_switch_triggered", "system", is_always_on=True),
        ]
        response = client.get("/api/notifications")
        cats = response.json()["categories"]
        assert cats == sorted(cats)


class TestUpdateNotificationSettings:
    """AC2: PUT /api/notifications toggles enabled state."""

    @patch('src.api.notification_config_endpoints.get_session')
    def test_toggle_off_returns_updated_state(self, mock_get_session, client):
        """PUT disables an enabled event and returns is_enabled=False."""
        mock_session = MagicMock()
        mock_get_session.return_value = mock_session
        mock_config = make_mock_config("trade_executed", "trade", is_enabled=True, is_always_on=False)
        mock_session.query.return_value.filter.return_value.first.return_value = mock_config
        response = client.put(
            "/api/notifications",
            json={"event_type": "trade_executed", "is_enabled": False}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["event_type"] == "trade_executed"
        assert "is_enabled" in data

    @patch('src.api.notification_config_endpoints.get_session')
    def test_toggle_on_not_found_returns_404(self, mock_get_session, client):
        """PUT returns 404 when event_type does not exist."""
        mock_session = MagicMock()
        mock_get_session.return_value = mock_session
        mock_session.query.return_value.filter.return_value.first.return_value = None
        response = client.put(
            "/api/notifications",
            json={"event_type": "nonexistent_event", "is_enabled": False}
        )
        assert response.status_code == 404


class TestAlwaysOnEvents:
    """AC3: Always-on events cannot be disabled via PUT."""

    @patch('src.api.notification_config_endpoints.get_session')
    def test_kill_switch_returns_400(self, mock_get_session, client):
        """Attempting to disable kill_switch_triggered returns 400."""
        mock_session = MagicMock()
        mock_get_session.return_value = mock_session
        mock_config = make_mock_config("kill_switch_triggered", "system", is_always_on=True)
        mock_session.query.return_value.filter.return_value.first.return_value = mock_config
        response = client.put(
            "/api/notifications",
            json={"event_type": "kill_switch_triggered", "is_enabled": False}
        )
        assert response.status_code == 400

    @patch('src.api.notification_config_endpoints.get_session')
    def test_loss_cap_returns_400(self, mock_get_session, client):
        """Attempting to disable loss_cap_triggered_system returns 400."""
        mock_session = MagicMock()
        mock_get_session.return_value = mock_session
        mock_config = make_mock_config("loss_cap_triggered_system", "system", is_always_on=True)
        mock_session.query.return_value.filter.return_value.first.return_value = mock_config
        response = client.put(
            "/api/notifications",
            json={"event_type": "loss_cap_triggered_system", "is_enabled": False}
        )
        assert response.status_code == 400

    @patch('src.api.notification_config_endpoints.get_session')
    def test_system_critical_returns_400(self, mock_get_session, client):
        """Attempting to disable system_critical returns 400."""
        mock_session = MagicMock()
        mock_get_session.return_value = mock_session
        mock_config = make_mock_config("system_critical", "system", is_always_on=True)
        mock_session.query.return_value.filter.return_value.first.return_value = mock_config
        response = client.put(
            "/api/notifications",
            json={"event_type": "system_critical", "is_enabled": False}
        )
        assert response.status_code == 400

    @patch('src.api.notification_config_endpoints.get_session')
    def test_always_on_error_message_is_descriptive(self, mock_get_session, client):
        """400 response contains descriptive detail about why it failed."""
        mock_session = MagicMock()
        mock_get_session.return_value = mock_session
        mock_config = make_mock_config("kill_switch_triggered", "system", is_always_on=True)
        mock_session.query.return_value.filter.return_value.first.return_value = mock_config
        response = client.put(
            "/api/notifications",
            json={"event_type": "kill_switch_triggered", "is_enabled": False}
        )
        assert response.status_code == 400
        assert "always-on" in response.json().get("detail", "").lower()
