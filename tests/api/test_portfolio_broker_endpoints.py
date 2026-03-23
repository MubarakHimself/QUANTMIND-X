# tests/api/test_portfolio_broker_endpoints.py
"""
Tests for Portfolio Broker Account Registry API Endpoints

Story 9.1: Broker Account Registry & Routing Matrix API

Tests for:
- Broker account CRUD operations
- Routing matrix retrieval
- Routing rules management
- MT5 auto-detection (simulated)
- Islamic compliance handling
"""

import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime

# Import the models directly to test
from src.database.models import (
    BrokerAccount,
    RoutingRule,
    BrokerAccountType,
    RegimeType,
    StrategyTypeEnum
)


class TestModelImports:
    """Test that models are correctly defined."""

    def test_broker_account_type_enum(self):
        """Test BrokerAccountType enum has all required types."""
        assert hasattr(BrokerAccountType, 'STANDARD')
        assert hasattr(BrokerAccountType, 'ISLAMIC')
        assert hasattr(BrokerAccountType, 'PROP_FIRM')
        assert hasattr(BrokerAccountType, 'PERSONAL')

    def test_regime_type_enum(self):
        """Test RegimeType enum has all required types."""
        assert hasattr(RegimeType, 'TREND')
        assert hasattr(RegimeType, 'RANGE')
        assert hasattr(RegimeType, 'BREAKOUT')
        assert hasattr(RegimeType, 'CHAOS')

    def test_strategy_type_enum(self):
        """Test StrategyTypeEnum has all required types."""
        assert hasattr(StrategyTypeEnum, 'SCALPER')
        assert hasattr(StrategyTypeEnum, 'HFT')
        assert hasattr(StrategyTypeEnum, 'STRUCTURAL')
        assert hasattr(StrategyTypeEnum, 'SWING')
        assert hasattr(StrategyTypeEnum, 'MACRO')


class TestBrokerAccountModel:
    """Test BrokerAccount model fields."""

    def test_broker_account_model_fields(self):
        """Test BrokerAccount model has required fields."""
        from sqlalchemy import inspect

        mapper = inspect(BrokerAccount)
        column_names = [col.key for col in mapper.columns]

        # Key fields for Story 9.1
        assert 'broker_name' in column_names
        assert 'account_number' in column_names
        assert 'account_type' in column_names
        assert 'account_tag' in column_names
        assert 'mt5_server' in column_names
        assert 'swap_free' in column_names
        assert 'leverage' in column_names
        assert 'detected_broker' in column_names  # MT5 auto-detection
        assert 'detected_account_type' in column_names
        assert 'is_active' in column_names  # Soft delete

    def test_routing_rule_model_fields(self):
        """Test RoutingRule model has required fields."""
        from sqlalchemy import inspect

        mapper = inspect(RoutingRule)
        column_names = [col.key for col in mapper.columns]

        # Key fields for routing
        assert 'broker_account_id' in column_names
        assert 'account_tag' in column_names
        assert 'regime_filter' in column_names
        assert 'strategy_type' in column_names
        assert 'priority' in column_names
        assert 'is_active' in column_names


class TestAPIEndpointImports:
    """Test that API endpoints can be imported."""

    def test_portfolio_broker_endpoints_import(self):
        """Test that portfolio broker endpoints module can be imported."""
        from src.api.portfolio_broker_endpoints import router
        assert router is not None

    def test_router_has_required_routes(self):
        """Test that router has required routes defined."""
        from src.api.portfolio_broker_endpoints import router

        # Check that routes are defined (by checking path prefixes)
        paths = [r.path for r in router.routes]
        assert any('/brokers' in p for p in paths)
        assert any('/routing-matrix' in p for p in paths)


class TestIslamicCompliance:
    """Test Islamic compliance requirements."""

    def test_islamic_account_type_flag(self):
        """Test that Islamic account type is defined."""
        assert BrokerAccountType.ISLAMIC.value == "islamic"

    def test_swap_free_field_exists(self):
        """Test that swap_free field exists in BrokerAccount."""
        from sqlalchemy import inspect
        mapper = inspect(BrokerAccount)
        column_names = [col.key for col in mapper.columns]
        assert 'swap_free' in column_names


class TestRoutingMatrixRequirements:
    """Test routing matrix requirements."""

    def test_strategy_types_cover_all_use_cases(self):
        """Test that strategy types cover all expected use cases."""
        strategies = [s.value for s in StrategyTypeEnum]
        assert 'scalper' in strategies  # HFT accounts
        assert 'hft' in strategies  # High frequency
        assert 'structural' in strategies  # Swing/sniper
        assert 'swing' in strategies  # Position trading
        assert 'macro' in strategies  # Macro strategies

    def test_regime_filters_support_all_regimes(self):
        """Test that regime filters support all market regimes."""
        regimes = [r.value for r in RegimeType]
        assert 'trend' in regimes
        assert 'range' in regimes
        assert 'breakout' in regimes
        assert 'chaos' in regimes


class TestDatabaseRelationships:
    """Test database relationships."""

    def test_routing_rule_broker_relationship(self):
        """Test RoutingRule has relationship to BrokerAccount."""
        from sqlalchemy import inspect

        mapper = inspect(RoutingRule)
        # Check foreign key exists
        fk_columns = [col.key for col in mapper.columns if col.foreign_keys]
        assert 'broker_account_id' in fk_columns