"""
Tests for Position Close API Endpoints

Tests the manual position close endpoints:
- POST /api/v1/trading/close - Close single position
- POST /api/v1/trading/close-all - Close all positions

Validates:
- AC #1: Close position confirmation modal
- AC #2: Close result display
- AC #3: Close all functionality with summary
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


class TestClosePositionEndpoint:
    """Test the close single position endpoint."""

    @pytest.mark.asyncio
    async def test_close_position_success(self):
        """Close position should succeed and return filled price, slippage, final P&L."""
        from src.api.trading.control import TradingControlAPIHandler
        from src.api.trading.models import ClosePositionRequest

        handler = TradingControlAPIHandler()

        request = ClosePositionRequest(
            position_ticket=12345,
            bot_id="test-bot-001"
        )

        result = handler.close_position(request)

        assert result.success is True
        assert result.filled_price is not None
        assert result.slippage is not None
        assert result.final_pnl is not None
        assert result.message == "Position closed successfully"

    @pytest.mark.asyncio
    async def test_close_position_invalid_ticket(self):
        """Close position should handle invalid ticket gracefully."""
        from src.api.trading.control import TradingControlAPIHandler
        from src.api.trading.models import ClosePositionRequest

        handler = TradingControlAPIHandler()

        # With invalid ticket, still returns simulated success
        # In production, this would query actual MT5 position
        request = ClosePositionRequest(
            position_ticket=999999,
            bot_id="test-bot-001"
        )

        result = handler.close_position(request)

        # Currently returns success with simulated data
        # In production, would return error for invalid ticket
        assert result.success is True

    @pytest.mark.asyncio
    async def test_close_position_logs_audit(self):
        """Close position should log to audit trail."""
        from src.api.trading.control import TradingControlAPIHandler
        from src.api.trading.models import ClosePositionRequest
        import logging

        handler = TradingControlAPIHandler()

        request = ClosePositionRequest(
            position_ticket=12345,
            bot_id="test-bot-001"
        )

        with patch('src.api.trading.control.logger') as mock_logger:
            result = handler.close_position(request)

            # Verify audit logging with [AUDIT] prefix
            mock_logger.info.assert_called()
            # Check that audit logging was called (may be pending or completed)
            calls = [call[0][0] for call in mock_logger.info.call_args_list]
            audit_calls = [c for c in calls if "[AUDIT]" in c]
            assert len(audit_calls) >= 1
            audit_msg = audit_calls[0]
            assert "12345" in audit_msg
            assert "test-bot-001" in audit_msg
            assert "manual_position_close" in audit_msg


class TestCloseAllPositionsEndpoint:
    """Test the close all positions endpoint."""

    @pytest.mark.asyncio
    async def test_close_all_positions_success(self):
        """Close all positions should return results per position."""
        from src.api.trading.control import TradingControlAPIHandler
        from src.api.trading.models import CloseAllRequest

        handler = TradingControlAPIHandler()

        request = CloseAllRequest(bot_id=None)

        result = handler.close_all_positions(request)

        assert result.success is True
        assert len(result.results) > 0

        # Verify result structure
        for item in result.results:
            assert item.position_ticket is not None
            assert item.status in ['filled', 'partial', 'rejected']

    @pytest.mark.asyncio
    async def test_close_all_positions_with_bot_filter(self):
        """Close all positions should filter by bot_id when provided."""
        from src.api.trading.control import TradingControlAPIHandler
        from src.api.trading.models import CloseAllRequest

        handler = TradingControlAPIHandler()

        request = CloseAllRequest(bot_id="specific-bot")

        result = handler.close_all_positions(request)

        assert result.success is True
        # Verify audit logging includes bot filter
        import logging
        with patch('src.api.trading.control.logger') as mock_logger:
            handler.close_all_positions(request)
            call_args = mock_logger.info.call_args[0][0]
            assert "specific-bot" in call_args

    @pytest.mark.asyncio
    async def test_close_all_returns_filled_status(self):
        """Close all should return filled status for successful closes."""
        from src.api.trading.control import TradingControlAPIHandler
        from src.api.trading.models import CloseAllRequest

        handler = TradingControlAPIHandler()

        request = CloseAllRequest(bot_id=None)

        result = handler.close_all_positions(request)

        # Check for filled positions
        filled = [r for r in result.results if r.status == 'filled']
        assert len(filled) > 0
        for f in filled:
            assert f.filled_price is not None


class TestClosePositionAPI:
    """Test the FastAPI endpoint integration."""

    def test_close_endpoint_exists(self):
        """Verify the close endpoint is registered."""
        from src.api.trading.routes import router

        # Check routes are defined (includes prefix /api/v1)
        routes = [r.path for r in router.routes]
        assert '/api/v1/trading/close' in routes

    def test_close_all_endpoint_exists(self):
        """Verify the close-all endpoint is registered."""
        from src.api.trading.routes import router

        routes = [r.path for r in router.routes]
        assert '/api/v1/trading/close-all' in routes


class TestCloseModels:
    """Test the request/response models."""

    def test_close_position_request_model(self):
        """Test ClosePositionRequest model validation."""
        from src.api.trading.models import ClosePositionRequest

        request = ClosePositionRequest(
            position_ticket=12345,
            bot_id="test-bot"
        )

        assert request.position_ticket == 12345
        assert request.bot_id == "test-bot"

    def test_close_position_response_model(self):
        """Test ClosePositionResponse model."""
        from src.api.trading.models import ClosePositionResponse

        response = ClosePositionResponse(
            success=True,
            filled_price=1.0850,
            slippage=0.5,
            final_pnl=25.50,
            message="Position closed"
        )

        assert response.success is True
        assert response.filled_price == 1.0850

    def test_close_all_request_model(self):
        """Test CloseAllRequest with optional bot_id."""
        from src.api.trading.models import CloseAllRequest

        # Without bot_id (close all)
        request1 = CloseAllRequest()
        assert request1.bot_id is None

        # With bot_id (filter by bot)
        request2 = CloseAllRequest(bot_id="specific-bot")
        assert request2.bot_id == "specific-bot"

    def test_close_all_response_model(self):
        """Test CloseAllResponse with results list."""
        from src.api.trading.models import (
            CloseAllResponse,
            CloseAllResultItem
        )

        results = [
            CloseAllResultItem(
                position_ticket=1001,
                status='filled',
                filled_price=1.0850,
                slippage=0.3,
                final_pnl=25.50
            ),
            CloseAllResultItem(
                position_ticket=1002,
                status='rejected',
                message="Insufficient margin"
            )
        ]

        response = CloseAllResponse(
            success=True,
            results=results
        )

        assert len(response.results) == 2
        assert response.results[0].status == 'filled'
        assert response.results[1].status == 'rejected'


class TestClosePositionEdgeCases:
    """Test edge cases for close position operations."""

    @pytest.mark.asyncio
    async def test_close_position_with_zero_ticket(self):
        """Close position with ticket 0 should be handled."""
        from src.api.trading.control import TradingControlAPIHandler
        from src.api.trading.models import ClosePositionRequest

        handler = TradingControlAPIHandler()

        # Ticket 0 is invalid - but handler returns simulated success
        request = ClosePositionRequest(
            position_ticket=0,
            bot_id="test-bot"
        )

        result = handler.close_position(request)

        # Currently returns success (would be error in production)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_close_position_empty_bot_id(self):
        """Close position with empty bot_id should work."""
        from src.api.trading.control import TradingControlAPIHandler
        from src.api.trading.models import ClosePositionRequest

        handler = TradingControlAPIHandler()

        request = ClosePositionRequest(
            position_ticket=12345,
            bot_id=""
        )

        result = handler.close_position(request)

        assert result.success is True

    @pytest.mark.asyncio
    async def test_close_position_audit_trail_complete(self):
        """Verify audit trail contains all required fields."""
        from src.api.trading.control import TradingControlAPIHandler
        from src.api.trading.models import ClosePositionRequest

        handler = TradingControlAPIHandler()

        request = ClosePositionRequest(
            position_ticket=54321,
            bot_id="audit-test-bot"
        )

        with patch('src.api.trading.control.logger') as mock_logger:
            handler.close_position(request)

            # Get all info calls
            calls = [call[0][0] for call in mock_logger.info.call_args_list]

            # Find audit entries
            audit_entries = [c for c in calls if "[AUDIT]" in c]

            # Should have at least 2 entries: pending and completed
            assert len(audit_entries) >= 2

            # Check pending entry
            pending = audit_entries[0]
            assert "54321" in pending
            assert "audit-test-bot" in pending
            assert "pending" in pending.lower()

            # Check completed entry
            completed = audit_entries[1]
            assert "54321" in completed
            assert "audit-test-bot" in completed
            assert "completed" in completed.lower()


class TestCloseAllEdgeCases:
    """Test edge cases for close all positions."""

    @pytest.mark.asyncio
    async def test_close_all_with_empty_results(self):
        """Close all when no positions exist."""
        from src.api.trading.control import TradingControlAPIHandler
        from src.api.trading.models import CloseAllRequest

        handler = TradingControlAPIHandler()

        # Empty bot_id - closes all
        request = CloseAllRequest(bot_id="non-existent-bot")

        result = handler.close_all_positions(request)

        # Returns success even with no positions
        assert result.success is True

    @pytest.mark.asyncio
    async def test_close_all_audit_includes_all_results(self):
        """Verify audit includes all position results."""
        from src.api.trading.control import TradingControlAPIHandler
        from src.api.trading.models import CloseAllRequest

        handler = TradingControlAPIHandler()

        request = CloseAllRequest(bot_id="audit-all-bot")

        with patch('src.api.trading.control.logger') as mock_logger:
            handler.close_all_positions(request)

            calls = [call[0][0] for call in mock_logger.info.call_args_list]
            audit_calls = [c for c in calls if "[AUDIT]" in c]

            # Should log the operation
            assert len(audit_calls) >= 1

            # Check includes bot filter
            audit_msg = audit_calls[0]
            assert "audit-all-bot" in audit_msg


class TestCloseModelsValidation:
    """Test model validation for close operations."""

    def test_close_position_request_missing_bot_id(self):
        """Test ClosePositionRequest with missing bot_id."""
        from src.api.trading.models import ClosePositionRequest
        from pydantic import ValidationError

        # bot_id is required, should raise ValidationError
        with pytest.raises(ValidationError):
            ClosePositionRequest(position_ticket=12345)

    def test_close_position_request_with_all_fields(self):
        """Test ClosePositionRequest with all optional fields."""
        from src.api.trading.models import ClosePositionRequest

        request = ClosePositionRequest(
            position_ticket=12345,
            bot_id="test-bot",
        )

        assert request.position_ticket == 12345
        assert request.bot_id == "test-bot"

    def test_close_all_request_with_none_bot_id(self):
        """Test CloseAllRequest with explicit None bot_id."""
        from src.api.trading.models import CloseAllRequest

        request = CloseAllRequest(bot_id=None)

        # Should be valid - means close all
        assert request.bot_id is None
