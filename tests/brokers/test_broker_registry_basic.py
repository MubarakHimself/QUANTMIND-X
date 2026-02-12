"""
V8 Broker Registry: Basic Integration Test

Quick test to verify broker registry works with mock adapter.

**Validates: Task 25.21 (partial)**
"""

import pytest
import asyncio
from src.data.brokers import BrokerRegistry


class TestBrokerRegistryBasic:
    """Basic integration tests for broker registry."""
    
    def test_registry_initialization(self):
        """Test registry initializes successfully."""
        registry = BrokerRegistry("config/brokers.yaml")
        
        assert registry is not None
        assert len(registry.list_brokers()) > 0
        print(f"✓ Registry initialized with {len(registry.list_brokers())} brokers")
    
    def test_list_brokers(self):
        """Test listing configured brokers."""
        registry = BrokerRegistry("config/brokers.yaml")
        brokers = registry.list_brokers()
        
        assert 'exness_demo_mock' in brokers
        assert 'binance_spot_testnet' in brokers
        print(f"✓ Found brokers: {brokers}")
    
    def test_get_mock_broker(self):
        """Test getting mock MT5 broker."""
        registry = BrokerRegistry("config/brokers.yaml")
        broker = registry.get_broker("exness_demo_mock")
        
        assert broker is not None
        assert broker.validate_connection() is True
        print("✓ Mock MT5 broker retrieved and validated")
    
    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_mock_broker_balance(self):
        """Test getting balance from mock broker."""
        registry = BrokerRegistry("config/brokers.yaml")
        broker = registry.get_broker("exness_demo_mock")
        
        balance = await broker.get_balance()
        
        assert balance > 0
        assert balance == 10000.0  # Initial balance from config
        print(f"✓ Mock broker balance: ${balance}")
    
    @pytest.mark.asyncio
    async def test_mock_broker_place_order(self):
        """Test placing order with mock broker."""
        registry = BrokerRegistry("config/brokers.yaml")
        broker = registry.get_broker("exness_demo_mock")
        
        order = await broker.place_order(
            symbol="EURUSD",
            volume=0.01,
            direction="buy",
            order_type="market"
        )
        
        assert order is not None
        assert order['symbol'] == "EURUSD"
        assert order['volume'] == 0.01
        assert order['direction'] == "buy"
        assert order['status'] == 'filled'
        print(f"✓ Order placed: {order['order_id']}")
    
    @pytest.mark.asyncio
    async def test_mock_broker_get_positions(self):
        """Test getting positions from mock broker."""
        registry = BrokerRegistry("config/brokers.yaml")
        broker = registry.get_broker("exness_demo_mock")
        
        # Place an order first
        await broker.place_order(
            symbol="GBPUSD",
            volume=0.02,
            direction="sell",
            order_type="market"
        )
        
        # Get positions
        positions = await broker.get_positions()
        
        assert len(positions) > 0
        assert positions[0]['symbol'] == "GBPUSD"
        assert positions[0]['volume'] == 0.02
        print(f"✓ Found {len(positions)} position(s)")
    
    def test_validate_all_connections(self):
        """Test validating all broker connections."""
        registry = BrokerRegistry("config/brokers.yaml")
        
        # Register enabled brokers
        registry.register_all()
        
        # Validate connections
        status = registry.validate_all_connections()
        
        assert len(status) > 0
        assert 'exness_demo_mock' in status
        assert status['exness_demo_mock'] is True
        print(f"✓ Validated {len(status)} broker(s)")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
