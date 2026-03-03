"""
Tests for Paper Trading API modular structure.

Tests that the paper_trading module can be imported and has the expected structure.
"""

import pytest


class TestPaperTradingModuleStructure:
    """Test that the paper_trading module has expected structure."""

    def test_paper_trading_module_importable(self):
        """Test that paper_trading module can be imported."""
        from src.api.paper_trading import router
        assert router is not None

    def test_paper_trading_has_models(self):
        """Test that models module exists."""
        from src.api.paper_trading import models
        assert models is not None

    def test_paper_trading_has_deployment_routes(self):
        """Test that deployment routes exist."""
        from src.api.paper_trading import deployment
        assert deployment is not None

    def test_paper_trading_has_agents_routes(self):
        """Test that agents routes exist."""
        from src.api.paper_trading import agents
        assert agents is not None

    def test_paper_trading_has_promotion_routes(self):
        """Test that promotion routes exist."""
        from src.api.paper_trading import promotion
        assert promotion is not None


class TestPaperTradingModels:
    """Test paper_trading models are available."""

    def test_promotion_request_model(self):
        """Test PromotionRequest model exists."""
        from src.api.paper_trading.models import PromotionRequest
        assert PromotionRequest is not None

    def test_promotion_result_model(self):
        """Test PromotionResult model exists."""
        from src.api.paper_trading.models import PromotionResult
        assert PromotionResult is not None

    def test_agent_performance_response_model(self):
        """Test AgentPerformanceResponse model exists."""
        from src.api.paper_trading.models import AgentPerformanceResponse
        assert AgentPerformanceResponse is not None


class TestPaperTradingBackwardCompatibility:
    """Test backward compatibility with original module."""

    def test_original_module_still_works(self):
        """Test original paper_trading_endpoints still works."""
        from src.api import paper_trading_endpoints
        assert paper_trading_endpoints is not None
        assert hasattr(paper_trading_endpoints, 'router')

    def test_original_models_still_available(self):
        """Test original models still accessible."""
        from src.api.paper_trading_endpoints import (
            PromotionRequest,
            PromotionResult,
            AgentPerformanceResponse
        )
        assert PromotionRequest is not None
        assert PromotionResult is not None
        assert AgentPerformanceResponse is not None
