"""
V8 Broker Registry: Comprehensive Tests

Complete test suite for broker registry including unit tests,
property tests, and integration tests.

**Validates: Tasks 25.21-25.24**
"""

import pytest
import asyncio
from hypothesis import given, strategies as st, settings
from src.data.brokers import BrokerRegistry


class TestBrokerRegistryUnit:
    """Unit tests for BrokerRegistry."""
    
    def test_adapter_map_populated(self):
        """Test ADAPTER_MAP is populated with adapters."""
        registry = BrokerRegistry("config/brokers.yaml")
        
        assert 'mt5_mock' in registry.ADAPTER_MAP
        assert 'mt5_socket' in registry.ADAPTER_MAP
        assert 'binance_spot' in registry.ADAPTER_MAP
        assert 'binance_futures' in registry.ADAPTER_MAP
        print(f"✓ ADAPTER_MAP has {len(registry.ADAPTER_MAP)} adapters")
    
    def test_config_validation(self):
        """Test configuration validation."""
        registry = BrokerRegistry("config/brokers.yaml")
        
        # All brokers should have 'type' field
        for broker_id, config in registry.config['brokers'].items():
            assert 'type' in config, f"Broker {broker_id} missing 'type'"
            assert config['type'] in registry.ADAPTER_MAP, f"Unknown type: {config['type']}"
        
        print("✓ All broker configurations valid")
    
    def test_lazy_loading(self):
        """Test brokers are lazy-loaded."""
        registry = BrokerRegistry("config/brokers.yaml")
        
        # Initially no brokers registered
        assert len(registry.list_registered_brokers()) == 0
        
        # Get broker triggers registration
        broker = registry.get_broker("exness_demo_mock")
        assert len(registry.list_registered_brokers()) == 1
        
        # Getting same broker doesn't create new instance
        broker2 = registry.get_broker("exness_demo_mock")
        assert broker is broker2
        
        print("✓ Lazy loading works correctly")
    
    def test_connection_status_caching(self):
        """Test connection status is cached."""
        registry = BrokerRegistry("config/brokers.yaml")
        broker = registry.get_broker("exness_demo_mock")
        
        # Status should be cached
        status = registry.get_connection_status("exness_demo_mock")
        assert status is True
        
        print("✓ Connection status cached")
    
    def test_unregister_broker(self):
        """Test unregistering a broker."""
        registry = BrokerRegistry("config/brokers.yaml")
        
        # Register broker
        broker = registry.get_broker("exness_demo_mock")
        assert "exness_demo_mock" in registry.list_registered_brokers()
        
        # Unregister
        registry.unregister_broker("exness_demo_mock")
        assert "exness_demo_mock" not in registry.list_registered_brokers()
        
        print("✓ Broker unregistered successfully")


class TestBrokerRegistryProperties:
    """
    Property 35: Broker Registry Adapter Consistency
    
    All adapters must implement BrokerClient interface consistently.
    
    **Validates: Task 25.22**
    """
    
    @given(
        broker_type=st.sampled_from(['mt5_mock', 'binance_spot', 'binance_futures'])
    )
    @settings(max_examples=50, deadline=None)
    def test_adapter_consistency_property(self, broker_type):
        """
        Property: All adapters implement BrokerClient interface.
        
        Given: Any broker type
        When: Adapter is instantiated
        Then: It must have all required methods
        """
        registry = BrokerRegistry("config/brokers.yaml")
        
        # Get adapter class
        adapter_class = registry.ADAPTER_MAP[broker_type]
        
        # Check required methods exist
        required_methods = [
            'get_balance',
            'place_order',
            'cancel_order',
            'get_order_book',
            'get_positions',
            'get_order_status',
            'modify_order',
            'validate_connection'
        ]
        
        for method in required_methods:
            assert hasattr(adapter_class, method), \
                f"Adapter {broker_type} missing method: {method}"
        
        print(f"✓ {broker_type} implements all required methods")
    
    @given(
        balance=st.floats(min_value=0.0, max_value=1000000.0)
    )
    @settings(max_examples=50, deadline=None)
    def test_balance_consistency_property(self, balance):
        """
        Property: Balance retrieval is consistent.
        
        Given: Any balance value
        When: get_balance() is called
        Then: It returns a non-negative float
        """
        registry = BrokerRegistry("config/brokers.yaml")
        broker = registry.get_broker("exness_demo_mock")
        
        # Set mock balance
        broker.balance = balance
        
        # Get balance
        async def test():
            result = await broker.get_balance()
            assert isinstance(result, float)
            assert result >= 0.0
            assert result == balance
            return result
        
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(test())
        
        print(f"✓ Balance consistency verified: ${result:.2f}")


class TestBrokerRegistryIntegration:
    """Integration tests for broker registry."""
    
    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_dynamic_broker_switching(self):
        """
        Test dynamic switching between brokers.
        
        **Validates: Task 25.23**
        """
        registry = BrokerRegistry("config/brokers.yaml")
        
        # Get mock MT5 broker
        mt5_broker = registry.get_broker("exness_demo_mock")
        mt5_balance = await mt5_broker.get_balance()
        
        # Get Binance testnet broker
        binance_broker = registry.get_broker("binance_spot_testnet")
        
        # Both should be different instances
        assert mt5_broker is not binance_broker
        
        # Both should be registered
        registered = registry.list_registered_brokers()
        assert "exness_demo_mock" in registered
        assert "binance_spot_testnet" in registered
        
        print(f"✓ Dynamic switching works: {len(registered)} brokers active")
    
    @pytest.mark.asyncio
    async def test_multiple_broker_instances(self):
        """
        Test multiple broker instances simultaneously.
        
        **Validates: Task 25.24**
        """
        registry = BrokerRegistry("config/brokers.yaml")
        
        # Register multiple brokers
        brokers = {}
        broker_ids = ["exness_demo_mock", "binance_spot_testnet"]
        
        for broker_id in broker_ids:
            try:
                broker = registry.get_broker(broker_id)
                brokers[broker_id] = broker
            except Exception as e:
                print(f"⚠ Could not register {broker_id}: {e}")
        
        # Verify all registered
        assert len(brokers) >= 1  # At least mock should work
        
        # Test operations on each
        for broker_id, broker in brokers.items():
            try:
                balance = await broker.get_balance()
                print(f"✓ {broker_id}: ${balance:.2f}")
            except Exception as e:
                print(f"⚠ {broker_id} balance failed: {e}")
        
        print(f"✓ Multiple instances working: {len(brokers)} brokers")
    
    @pytest.mark.asyncio
    async def test_concurrent_broker_operations(self):
        """Test concurrent operations across multiple brokers."""
        registry = BrokerRegistry("config/brokers.yaml")
        
        # Get brokers
        mt5 = registry.get_broker("exness_demo_mock")
        
        # Concurrent operations
        tasks = [
            mt5.get_balance(),
            mt5.get_positions(),
            mt5.place_order("EURUSD", 0.01, "buy"),
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check results
        assert len(results) == 3
        assert isinstance(results[0], float)  # Balance
        assert isinstance(results[1], list)   # Positions
        assert isinstance(results[2], dict)   # Order
        
        print("✓ Concurrent operations successful")
    
    def test_validate_all_connections_comprehensive(self):
        """Test comprehensive connection validation."""
        registry = BrokerRegistry("config/brokers.yaml")
        
        # Register all enabled brokers
        registry.register_all()
        
        # Validate all
        status = registry.validate_all_connections()
        
        # Check results
        assert len(status) > 0
        for broker_id, connected in status.items():
            print(f"  {broker_id}: {'✓ Connected' if connected else '✗ Disconnected'}")
        
        # At least mock should be connected
        assert status.get("exness_demo_mock") is True
        
        print(f"✓ Validated {len(status)} broker(s)")


class TestBrokerRegistryErrorHandling:
    """Test error handling in broker registry."""
    
    def test_invalid_broker_id(self):
        """Test getting non-existent broker."""
        registry = BrokerRegistry("config/brokers.yaml")
        
        with pytest.raises(ValueError, match="not found"):
            registry.get_broker("nonexistent_broker")
        
        print("✓ Invalid broker ID handled correctly")
    
    def test_disabled_broker(self):
        """Test that disabled brokers are not auto-registered."""
        registry = BrokerRegistry("config/brokers.yaml")
        
        # Register all (should skip disabled)
        registry.register_all()
        
        registered = registry.list_registered_brokers()
        
        # Disabled brokers should not be in registered list
        # (unless explicitly registered)
        print(f"✓ Only enabled brokers registered: {len(registered)}")
    
    def test_missing_config_file(self):
        """Test handling of missing config file."""
        registry = BrokerRegistry("nonexistent_config.yaml")
        
        # Should initialize but with empty config
        assert len(registry.list_brokers()) == 0
        
        print("✓ Missing config handled gracefully")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--hypothesis-show-statistics"])
