# tests/api/test_workshop_copilot.py
"""
Tests for Workshop Copilot Service

Tests for the intent classification and message handling of WorkshopCopilotService.
"""
import pytest
from src.api.services.workshop_copilot_service import (
    WorkshopCopilotService,
    WorkshopCopilotRequest,
    WorkshopCopilotResponse,
    _classify_intent,
    get_workshop_copilot_service,
    TRADING_KEYWORDS,
    WORKFLOW_KEYWORDS,
)


class TestIntentClassification:
    """Tests for intent classification function."""

    def test_classify_simple_query(self):
        """Simple queries should be classified as 'simple'."""
        assert _classify_intent("Hello, how are you?") == "simple"
        assert _classify_intent("What's the status?") == "simple"
        assert _classify_intent("Tell me about the weather") == "simple"

    def test_classify_trading_intent(self):
        """Trading keywords should be classified as 'trading'."""
        assert _classify_intent("Deploy my strategy") == "trading"
        assert _classify_intent("Buy 1 lot EURUSD") == "trading"
        assert _classify_intent("Run backtest on RSI") == "trading"
        assert _classify_intent("Start paper trading bot") == "trading"
        assert _classify_intent("Execute sell order") == "trading"
        assert _classify_intent("Create a trading bot") == "trading"

    def test_classify_workflow_intent(self):
        """Workflow keywords should be classified as 'workflow'."""
        assert _classify_intent("Create a new project") == "workflow"
        assert _classify_intent("Build a website") == "workflow"
        assert _classify_intent("Generate code from video") == "workflow"
        assert _classify_intent("Analyze video content") == "workflow"
        assert _classify_intent("Develop an application") == "workflow"
        assert _classify_intent("Build from YouTube video") == "workflow"

    def test_classify_trading_takes_priority(self):
        """Trading keywords should take priority over workflow keywords."""
        # "deploy" is both trading and workflow, but trading comes first
        assert _classify_intent("Deploy a new strategy") == "trading"

    def test_classify_case_insensitive(self):
        """Classification should be case insensitive."""
        assert _classify_intent("DEPLOY my bot") == "trading"
        assert _classify_intent("Create a PROJECT") == "workflow"
        assert _classify_intent("Hello WORLD") == "simple"


class TestWorkshopCopilotService:
    """Tests for WorkshopCopilotService."""

    @pytest.fixture
    def service(self):
        """Create a WorkshopCopilotService instance."""
        return WorkshopCopilotService()

    @pytest.mark.asyncio
    async def test_handle_greeting(self, service):
        """Should respond to greetings appropriately."""
        request = WorkshopCopilotRequest(message="Hello!")
        response = await service.handle_message(request)

        assert isinstance(response, WorkshopCopilotResponse)
        assert "help" in response.reply.lower() or "workshop" in response.reply.lower()
        assert response.action_taken == "greeting"

    @pytest.mark.asyncio
    async def test_handle_status_query(self, service):
        """Should respond to status queries."""
        request = WorkshopCopilotRequest(message="What's the status?")
        response = await service.handle_message(request)

        assert isinstance(response, WorkshopCopilotResponse)
        assert response.action_taken == "status_check"

    @pytest.mark.asyncio
    async def test_handle_simple_query(self, service):
        """Should handle simple queries."""
        request = WorkshopCopilotRequest(message="What is 2+2?")
        response = await service.handle_message(request)

        assert isinstance(response, WorkshopCopilotResponse)
        assert response.action_taken == "simple_query"

    @pytest.mark.asyncio
    async def test_handle_trading_request(self, service):
        """Should delegate trading requests to floor manager."""
        request = WorkshopCopilotRequest(message="Deploy my strategy")
        response = await service.handle_message(request)

        assert isinstance(response, WorkshopCopilotResponse)
        assert response.delegation == "floor_manager"
        assert response.action_taken == "trading_request"

    @pytest.mark.asyncio
    async def test_handle_workflow_request(self, service):
        """Should handle workflow requests."""
        request = WorkshopCopilotRequest(message="Create a new project")
        response = await service.handle_message(request)

        assert isinstance(response, WorkshopCopilotResponse)
        assert response.delegation == "department_workflow"
        assert response.action_taken == "workflow_request"

    @pytest.mark.asyncio
    async def test_response_to_dict(self, service):
        """Should convert response to dict correctly."""
        request = WorkshopCopilotRequest(message="Hello")
        response = await service.handle_message(request)

        result = response.to_dict()
        assert "reply" in result
        assert "delegation" in result
        assert "action_taken" in result


class TestWorkshopCopilotSingleton:
    """Tests for singleton getter."""

    def test_get_service_returns_singleton(self):
        """Should return the same instance on multiple calls."""
        service1 = get_workshop_copilot_service()
        service2 = get_workshop_copilot_service()

        assert service1 is service2

    def test_singleton_is_workshop_copilot_service(self):
        """Singleton should be instance of WorkshopCopilotService."""
        service = get_workshop_copilot_service()
        assert isinstance(service, WorkshopCopilotService)


class TestWorkshopCopilotServiceConfig:
    """Sync config checks that do not require async pytest plugins."""

    def test_internal_api_base_url_takes_priority(self, monkeypatch):
        monkeypatch.setenv("INTERNAL_API_BASE_URL", "https://internal.quantmindx.local")
        monkeypatch.setenv("API_BASE_URL", "https://public.quantmindx.local")

        service = WorkshopCopilotService()

        assert service.floor_manager_url == "https://internal.quantmindx.local"


class TestWorkshopCopilotRequest:
    """Tests for request model."""

    def test_request_defaults(self):
        """Request should have default values."""
        request = WorkshopCopilotRequest(message="test")

        assert request.message == "test"
        assert request.history == []
        assert request.session_id is None

    def test_request_with_all_params(self):
        """Request should accept all parameters."""
        request = WorkshopCopilotRequest(
            message="test",
            history=[{"role": "user", "content": "hello"}],
            session_id="session-123"
        )

        assert request.message == "test"
        assert len(request.history) == 1
        assert request.session_id == "session-123"
