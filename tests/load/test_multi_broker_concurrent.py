"""
V8 Broker Registry: Load Tests for Multi-Broker Concurrent Trading

Tests the BrokerRegistry's ability to handle concurrent trading across
multiple brokers with high throughput. Validates connection pooling,
resource management, and adapter consistency under load.

**Validates: Task 26.10, Requirements 19.7, 19.8**
"""

import pytest
import asyncio
import time
import statistics
from typing import Dict, List, Any, Tuple
from unittest.mock import AsyncMock, MagicMock, patch

from src.data.brokers.registry import BrokerRegistry
from src.integrations.crypto.broker_client import BrokerClient


class BrokerLoadMonitor:
    """Monitor broker operations and performance metrics."""
    
    def __init__(self):
        self.operations: Dict[str, List[Dict[str, Any]]] = {}
        self.latencies: Dict[str, List[float]] = {}
        self.errors: Dict[str, List[str]] = {}
        self.total_operations = 0
        self.start_time = time.time()
    
    def record_operation(
        self,
        broker_id: str,
        operation: str,
        latency_ms: float,
        success: bool,
        error: str = None
    ):
        """Record a broker operation."""
        if broker_id not in self.operations:
            self.operations[broker_id] = []
            self.latencies[broker_id] = []
            self.errors[broker_id] = []
        
        self.operations[broker_id].append({
            'operation': operation,
            'latency_ms': latency_ms,
            'success': success,
            'timestamp': time.time()
        })
        
        if success:
            self.latencies[broker_id].append(latency_ms)
        else:
            self.errors[broker_id].append(error or 'Unknown error')
        
        self.total_operations += 1
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        total_duration = time.time() - self.start_time
        
        stats = {
            'total_operations': self.total_operations,
            'duration_seconds': total_duration,
            'throughput_ops_per_sec': self.total_operations / total_duration if total_duration > 0 else 0,
            'brokers': {}
        }
        
        # Per-broker statistics
        for broker_id in self.operations.keys():
            broker_ops = len(self.operations[broker_id])
            broker_latencies = self.latencies[broker_id]
            broker_errors = len(self.errors[broker_id])
            
            if broker_latencies:
                sorted_latencies = sorted(broker_latencies)
                stats['brokers'][broker_id] = {
                    'operations': broker_ops,
                    'successful': len(broker_latencies),
                    'failed': broker_errors,
                    'success_rate': (len(broker_latencies) / broker_ops) * 100 if broker_ops > 0 else 0,
                    'avg_latency_ms': statistics.mean(broker_latencies),
                    'median_latency_ms': statistics.median(broker_latencies),
                    'p95_latency_ms': sorted_latencies[int(len(sorted_latencies) * 0.95)],
                    'p99_latency_ms': sorted_latencies[int(len(sorted_latencies) * 0.99)],
                    'max_latency_ms': max(broker_latencies),
                    'min_latency_ms': min(broker_latencies)
                }
            else:
                stats['brokers'][broker_id] = {
                    'operations': broker_ops,
                    'successful': 0,
                    'failed': broker_errors,
                    'success_rate': 0,
                    'avg_latency_ms': 0,
                    'median_latency_ms': 0,
                    'p95_latency_ms': 0,
                    'p99_latency_ms': 0,
                    'max_latency_ms': 0,
                    'min_latency_ms': 0
                }
        
        # Overall latency statistics
        all_latencies = []
        for latencies in self.latencies.values():
            all_latencies.extend(latencies)
        
        if all_latencies:
            sorted_all = sorted(all_latencies)
            stats['overall_latency'] = {
                'avg_ms': statistics.mean(all_latencies),
                'median_ms': statistics.median(all_latencies),
                'p95_ms': sorted_all[int(len(sorted_all) * 0.95)],
                'p99_ms': sorted_all[int(len(sorted_all) * 0.99)],
                'max_ms': max(all_latencies),
                'min_ms': min(all_latencies)
            }
        
        return stats


class TestMultiBrokerConcurrentTrading:
    """Load tests for multi-broker concurrent trading scenarios."""
    
    @pytest.fixture
    def mock_broker_config(self):
        """Create mock broker configuration."""
        return {
            'brokers': {
                'mt5_broker_1': {
                    'type': 'mt5_mock',
                    'account_id': 'MT5_ACCOUNT_1',
                    'initial_balance': 10000.0,
                    'enabled': True
                },
                'mt5_broker_2': {
                    'type': 'mt5_mock',
                    'account_id': 'MT5_ACCOUNT_2',
                    'initial_balance': 15000.0,
                    'enabled': True
                },
                'mt5_broker_3': {
                    'type': 'mt5_mock',
                    'account_id': 'MT5_ACCOUNT_3',
                    'initial_balance': 20000.0,
                    'enabled': True
                },
                'binance_spot': {
                    'type': 'binance_spot',
                    'api_key': 'test_api_key',
                    'api_secret': 'test_api_secret',
                    'testnet': True,
                    'enabled': True
                },
                'binance_futures': {
                    'type': 'binance_futures',
                    'api_key': 'test_api_key',
                    'api_secret': 'test_api_secret',
                    'testnet': True,
                    'enabled': True
                }
            }
        }
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    @pytest.mark.timeout(30)
    async def test_load_5_brokers_50_trades_per_second(self, mock_broker_config, tmp_path):
        """
        Load Test: 5 brokers with 50 trades/second total throughput.
        
        Tests BrokerRegistry with multiple simultaneous broker connections
        handling high-frequency trading operations.
        
        **Target**: 5 brokers, 50 trades/sec total (10 trades/sec per broker)
        **Validates**: Requirements 19.7, 19.8
        """
        # Create temporary config file
        import yaml
        config_path = tmp_path / "brokers.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(mock_broker_config, f)
        
        # Initialize registry
        registry = BrokerRegistry(str(config_path))
        registry.register_all()
        
        # Verify all brokers registered
        registered = registry.list_registered_brokers()
        assert len(registered) == 5, f"Expected 5 brokers, got {len(registered)}"
        
        # Test parameters
        test_duration_seconds = 10
        target_trades_per_sec = 50
        trades_per_broker_per_sec = target_trades_per_sec / 5
        
        monitor = BrokerLoadMonitor()
        
        async def broker_trading_stream(broker_id: str):
            """Simulate continuous trading on a broker."""
            broker = registry.get_broker(broker_id)
            trade_count = 0
            interval = 1.0 / trades_per_broker_per_sec
            
            start_time = time.time()
            
            while (time.time() - start_time) < test_duration_seconds:
                # Alternate between buy and sell
                direction = 'buy' if trade_count % 2 == 0 else 'sell'
                symbol = 'EURUSD' if 'mt5' in broker_id else 'BTCUSDT'
                
                # Place order
                op_start = time.time()
                try:
                    result = await broker.place_order(
                        symbol=symbol,
                        volume=0.1,
                        direction=direction,
                        order_type='market'
                    )
                    latency_ms = (time.time() - op_start) * 1000
                    
                    monitor.record_operation(
                        broker_id=broker_id,
                        operation='place_order',
                        latency_ms=latency_ms,
                        success=True
                    )
                    
                    trade_count += 1
                    
                except Exception as e:
                    latency_ms = (time.time() - op_start) * 1000
                    monitor.record_operation(
                        broker_id=broker_id,
                        operation='place_order',
                        latency_ms=latency_ms,
                        success=False,
                        error=str(e)
                    )
                
                # Wait for next trade interval
                await asyncio.sleep(interval)
            
            return trade_count
        
        print(f"\n=== Multi-Broker Concurrent Trading Load Test ===")
        print(f"Brokers: {len(registered)}")
        print(f"Target throughput: {target_trades_per_sec} trades/sec")
        print(f"Per broker: {trades_per_broker_per_sec} trades/sec")
        print(f"Duration: {test_duration_seconds}s")
        print(f"Expected total trades: {target_trades_per_sec * test_duration_seconds}")
        print(f"\nStarting concurrent trading streams...")
        
        # Run all broker streams concurrently
        start_time = time.time()
        trade_counts = await asyncio.gather(*[
            broker_trading_stream(broker_id) for broker_id in registered
        ])
        actual_duration = time.time() - start_time
        
        # Get statistics
        stats = monitor.get_statistics()
        
        print(f"\n=== Load Test Results ===")
        print(f"Actual duration: {actual_duration:.2f}s")
        print(f"Total operations: {stats['total_operations']}")
        print(f"Actual throughput: {stats['throughput_ops_per_sec']:.1f} ops/sec")
        print(f"Target throughput: {target_trades_per_sec} ops/sec")
        print(f"Efficiency: {(stats['throughput_ops_per_sec']/target_trades_per_sec)*100:.1f}%")
        
        print(f"\n=== Overall Latency Statistics ===")
        if 'overall_latency' in stats:
            ol = stats['overall_latency']
            print(f"  Average: {ol['avg_ms']:.3f}ms")
            print(f"  Median: {ol['median_ms']:.3f}ms")
            print(f"  P95: {ol['p95_ms']:.3f}ms")
            print(f"  P99: {ol['p99_ms']:.3f}ms")
            print(f"  Max: {ol['max_ms']:.3f}ms")
        
        print(f"\n=== Per-Broker Statistics ===")
        for broker_id, broker_stats in stats['brokers'].items():
            print(f"\n{broker_id}:")
            print(f"  Operations: {broker_stats['operations']}")
            print(f"  Success rate: {broker_stats['success_rate']:.1f}%")
            print(f"  Avg latency: {broker_stats['avg_latency_ms']:.3f}ms")
            print(f"  P95 latency: {broker_stats['p95_latency_ms']:.3f}ms")
            print(f"  Max latency: {broker_stats['max_latency_ms']:.3f}ms")
        
        # Assertions
        assert stats['total_operations'] > 0, "No operations completed"
        
        # Verify throughput (allow 80% of target)
        assert stats['throughput_ops_per_sec'] >= (target_trades_per_sec * 0.8), \
            f"Throughput {stats['throughput_ops_per_sec']:.1f} below 80% of target {target_trades_per_sec}"
        
        # Verify all brokers participated
        assert len(stats['brokers']) == 5, \
            f"Expected 5 brokers in stats, got {len(stats['brokers'])}"
        
        # Verify success rates
        for broker_id, broker_stats in stats['brokers'].items():
            assert broker_stats['success_rate'] >= 95.0, \
                f"Broker {broker_id} success rate {broker_stats['success_rate']:.1f}% below 95%"
        
        # Verify latency targets (Requirement 19.8)
        if 'overall_latency' in stats:
            assert stats['overall_latency']['avg_ms'] < 200.0, \
                f"Average latency {stats['overall_latency']['avg_ms']:.3f}ms exceeds 200ms target"
            
            assert stats['overall_latency']['p95_ms'] < 500.0, \
                f"P95 latency {stats['overall_latency']['p95_ms']:.3f}ms exceeds 500ms acceptable limit"

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_load_connection_pooling_resource_management(self, mock_broker_config, tmp_path):
        """
        Load Test: Connection pooling and resource management.
        
        Tests that BrokerRegistry properly manages connections and resources
        when multiple operations are performed concurrently on the same brokers.
        
        **Target**: Verify connection reuse and resource cleanup
        **Validates**: Requirement 19.7
        """
        import yaml
        config_path = tmp_path / "brokers.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(mock_broker_config, f)
        
        registry = BrokerRegistry(str(config_path))
        registry.register_all()
        
        monitor = BrokerLoadMonitor()
        
        # Test parameters
        num_concurrent_operations = 100
        operations_per_broker = 20
        
        async def concurrent_operations_on_broker(broker_id: str, operation_id: int):
            """Perform multiple operations on same broker."""
            broker = registry.get_broker(broker_id)
            
            operations = [
                ('get_balance', lambda: broker.get_balance()),
                ('get_positions', lambda: broker.get_positions()),
                ('place_order', lambda: broker.place_order('EURUSD', 0.1, 'buy')),
                ('get_balance', lambda: broker.get_balance()),
            ]
            
            for op_name, op_func in operations:
                op_start = time.time()
                try:
                    result = await op_func()
                    latency_ms = (time.time() - op_start) * 1000
                    
                    monitor.record_operation(
                        broker_id=broker_id,
                        operation=op_name,
                        latency_ms=latency_ms,
                        success=True
                    )
                except Exception as e:
                    latency_ms = (time.time() - op_start) * 1000
                    monitor.record_operation(
                        broker_id=broker_id,
                        operation=op_name,
                        latency_ms=latency_ms,
                        success=False,
                        error=str(e)
                    )
        
        print(f"\n=== Connection Pooling & Resource Management Test ===")
        print(f"Brokers: {len(registry.list_registered_brokers())}")
        print(f"Concurrent operations: {num_concurrent_operations}")
        print(f"Operations per broker: {operations_per_broker}")
        
        # Create tasks for concurrent operations
        tasks = []
        for broker_id in registry.list_registered_brokers():
            for op_id in range(operations_per_broker):
                tasks.append(concurrent_operations_on_broker(broker_id, op_id))
        
        # Execute all concurrently
        start_time = time.time()
        await asyncio.gather(*tasks)
        duration = time.time() - start_time
        
        stats = monitor.get_statistics()
        
        print(f"\n=== Resource Management Results ===")
        print(f"Duration: {duration:.2f}s")
        print(f"Total operations: {stats['total_operations']}")
        print(f"Throughput: {stats['throughput_ops_per_sec']:.1f} ops/sec")
        
        print(f"\n=== Per-Broker Resource Usage ===")
        for broker_id, broker_stats in stats['brokers'].items():
            print(f"\n{broker_id}:")
            print(f"  Total operations: {broker_stats['operations']}")
            print(f"  Success rate: {broker_stats['success_rate']:.1f}%")
            print(f"  Avg latency: {broker_stats['avg_latency_ms']:.3f}ms")
        
        # Verify connection reuse (all operations should succeed)
        for broker_id, broker_stats in stats['brokers'].items():
            assert broker_stats['success_rate'] >= 95.0, \
                f"Broker {broker_id} success rate {broker_stats['success_rate']:.1f}% indicates connection issues"
        
        # Verify no resource leaks (latency should remain stable)
        if 'overall_latency' in stats:
            assert stats['overall_latency']['max_ms'] < 1000.0, \
                f"Max latency {stats['overall_latency']['max_ms']:.3f}ms suggests resource exhaustion"
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_load_broker_adapter_consistency_under_load(self, mock_broker_config, tmp_path):
        """
        Load Test: Broker adapter consistency under high load.
        
        Tests that all broker adapters maintain consistent behavior and
        interface compliance when under heavy concurrent load.
        
        **Target**: Verify adapter interface consistency
        **Validates**: Requirement 19.6, Property 35
        """
        import yaml
        config_path = tmp_path / "brokers.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(mock_broker_config, f)
        
        registry = BrokerRegistry(str(config_path))
        registry.register_all()
        
        monitor = BrokerLoadMonitor()
        interface_violations = []
        
        async def test_broker_interface(broker_id: str, iteration: int):
            """Test that broker implements full interface."""
            broker = registry.get_broker(broker_id)
            
            # Test all required interface methods
            interface_tests = [
                ('get_balance', lambda: broker.get_balance()),
                ('place_order', lambda: broker.place_order('EURUSD', 0.1, 'buy')),
                ('get_positions', lambda: broker.get_positions()),
                ('get_order_book', lambda: broker.get_order_book('EURUSD')),
            ]
            
            for method_name, method_call in interface_tests:
                op_start = time.time()
                try:
                    result = await method_call()
                    latency_ms = (time.time() - op_start) * 1000
                    
                    # Verify result type
                    if method_name == 'get_balance':
                        if not isinstance(result, (int, float)):
                            interface_violations.append(
                                f"{broker_id}.{method_name} returned {type(result)}, expected float"
                            )
                    elif method_name == 'place_order':
                        if not isinstance(result, dict):
                            interface_violations.append(
                                f"{broker_id}.{method_name} returned {type(result)}, expected dict"
                            )
                    elif method_name in ['get_positions', 'get_order_book']:
                        if not isinstance(result, (list, dict)):
                            interface_violations.append(
                                f"{broker_id}.{method_name} returned {type(result)}, expected list/dict"
                            )
                    
                    monitor.record_operation(
                        broker_id=broker_id,
                        operation=method_name,
                        latency_ms=latency_ms,
                        success=True
                    )
                    
                except Exception as e:
                    latency_ms = (time.time() - op_start) * 1000
                    monitor.record_operation(
                        broker_id=broker_id,
                        operation=method_name,
                        latency_ms=latency_ms,
                        success=False,
                        error=str(e)
                    )
                    interface_violations.append(
                        f"{broker_id}.{method_name} raised {type(e).__name__}: {e}"
                    )
        
        print(f"\n=== Broker Adapter Consistency Test ===")
        print(f"Brokers: {len(registry.list_registered_brokers())}")
        print(f"Testing interface compliance under load...")
        
        # Run interface tests concurrently
        iterations = 50
        tasks = []
        for broker_id in registry.list_registered_brokers():
            for i in range(iterations):
                tasks.append(test_broker_interface(broker_id, i))
        
        start_time = time.time()
        await asyncio.gather(*tasks)
        duration = time.time() - start_time
        
        stats = monitor.get_statistics()
        
        print(f"\n=== Adapter Consistency Results ===")
        print(f"Duration: {duration:.2f}s")
        print(f"Total operations: {stats['total_operations']}")
        print(f"Interface violations: {len(interface_violations)}")
        
        if interface_violations:
            print(f"\n=== Interface Violations ===")
            for violation in interface_violations[:10]:  # Show first 10
                print(f"  - {violation}")
            if len(interface_violations) > 10:
                print(f"  ... and {len(interface_violations) - 10} more")
        
        print(f"\n=== Per-Broker Consistency ===")
        for broker_id, broker_stats in stats['brokers'].items():
            print(f"\n{broker_id}:")
            print(f"  Operations: {broker_stats['operations']}")
            print(f"  Success rate: {broker_stats['success_rate']:.1f}%")
            print(f"  Avg latency: {broker_stats['avg_latency_ms']:.3f}ms")
        
        # Assertions
        assert len(interface_violations) == 0, \
            f"Found {len(interface_violations)} interface violations"
        
        # All brokers should have high success rates
        for broker_id, broker_stats in stats['brokers'].items():
            assert broker_stats['success_rate'] >= 95.0, \
                f"Broker {broker_id} success rate {broker_stats['success_rate']:.1f}% below 95%"
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_load_mixed_broker_operations_high_throughput(self, mock_broker_config, tmp_path):
        """
        Load Test: Mixed operations across brokers with high throughput.
        
        Tests realistic trading scenario with mixed operations (orders, queries,
        modifications) across multiple brokers simultaneously.
        
        **Target**: Realistic mixed workload at high throughput
        **Validates**: Requirements 19.7, 19.8
        """
        import yaml
        config_path = tmp_path / "brokers.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(mock_broker_config, f)
        
        registry = BrokerRegistry(str(config_path))
        registry.register_all()
        
        monitor = BrokerLoadMonitor()
        
        async def mixed_trading_workload(broker_id: str, duration_seconds: int):
            """Simulate realistic mixed trading workload."""
            broker = registry.get_broker(broker_id)
            start_time = time.time()
            operation_count = 0
            
            while (time.time() - start_time) < duration_seconds:
                # Rotate through different operations
                op_type = operation_count % 5
                op_name = 'unknown'  # Initialize to avoid UnboundLocalError
                
                op_start = time.time()
                try:
                    if op_type == 0:
                        # Place order
                        op_name = 'place_order'
                        result = await broker.place_order('EURUSD', 0.1, 'buy')
                    elif op_type == 1:
                        # Get balance
                        op_name = 'get_balance'
                        result = await broker.get_balance()
                    elif op_type == 2:
                        # Get positions
                        op_name = 'get_positions'
                        result = await broker.get_positions()
                    elif op_type == 3:
                        # Get order book
                        op_name = 'get_order_book'
                        result = await broker.get_order_book('EURUSD')
                    else:
                        # Place sell order
                        op_name = 'place_order'
                        result = await broker.place_order('EURUSD', 0.1, 'sell')
                    
                    latency_ms = (time.time() - op_start) * 1000
                    monitor.record_operation(
                        broker_id=broker_id,
                        operation=op_name,
                        latency_ms=latency_ms,
                        success=True
                    )
                    
                    operation_count += 1
                    
                except Exception as e:
                    latency_ms = (time.time() - op_start) * 1000
                    monitor.record_operation(
                        broker_id=broker_id,
                        operation=op_name,
                        latency_ms=latency_ms,
                        success=False,
                        error=str(e)
                    )
                
                # Small delay to control throughput
                await asyncio.sleep(0.02)  # ~50 ops/sec per broker
            
            return operation_count
        
        print(f"\n=== Mixed Operations High Throughput Test ===")
        print(f"Brokers: {len(registry.list_registered_brokers())}")
        print(f"Duration: 15 seconds")
        print(f"Target: ~50 ops/sec per broker (~250 ops/sec total)")
        
        # Run mixed workload on all brokers
        start_time = time.time()
        operation_counts = await asyncio.gather(*[
            mixed_trading_workload(broker_id, 15)
            for broker_id in registry.list_registered_brokers()
        ])
        duration = time.time() - start_time
        
        stats = monitor.get_statistics()
        
        print(f"\n=== Mixed Operations Results ===")
        print(f"Actual duration: {duration:.2f}s")
        print(f"Total operations: {stats['total_operations']}")
        print(f"Throughput: {stats['throughput_ops_per_sec']:.1f} ops/sec")
        print(f"Operations per broker: {sum(operation_counts) / len(operation_counts):.0f}")
        
        if 'overall_latency' in stats:
            print(f"\n=== Overall Latency ===")
            ol = stats['overall_latency']
            print(f"  Average: {ol['avg_ms']:.3f}ms")
            print(f"  Median: {ol['median_ms']:.3f}ms")
            print(f"  P95: {ol['p95_ms']:.3f}ms")
            print(f"  P99: {ol['p99_ms']:.3f}ms")
        
        print(f"\n=== Per-Broker Performance ===")
        for broker_id, broker_stats in stats['brokers'].items():
            print(f"\n{broker_id}:")
            print(f"  Operations: {broker_stats['operations']}")
            print(f"  Success rate: {broker_stats['success_rate']:.1f}%")
            print(f"  Avg latency: {broker_stats['avg_latency_ms']:.3f}ms")
            print(f"  P95 latency: {broker_stats['p95_latency_ms']:.3f}ms")
        
        # Assertions
        assert stats['total_operations'] >= 1000, \
            f"Expected at least 1000 operations, got {stats['total_operations']}"
        
        assert stats['throughput_ops_per_sec'] >= 50, \
            f"Throughput {stats['throughput_ops_per_sec']:.1f} below 50 ops/sec minimum"
        
        # Verify all brokers performed well
        for broker_id, broker_stats in stats['brokers'].items():
            assert broker_stats['success_rate'] >= 95.0, \
                f"Broker {broker_id} success rate {broker_stats['success_rate']:.1f}% below 95%"
            
            assert broker_stats['avg_latency_ms'] < 200.0, \
                f"Broker {broker_id} avg latency {broker_stats['avg_latency_ms']:.3f}ms exceeds 200ms"


class TestBrokerRegistryStressScenarios:
    """Stress tests to find breaking points of broker registry."""
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_stress_rapid_broker_switching(self, mock_broker_config, tmp_path):
        """
        Stress Test: Rapid switching between brokers.
        
        Tests registry's ability to handle rapid broker access patterns
        without connection issues or resource leaks.
        """
        import yaml
        config_path = tmp_path / "brokers.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(mock_broker_config, f)
        
        registry = BrokerRegistry(str(config_path))
        registry.register_all()
        
        broker_ids = registry.list_registered_brokers()
        monitor = BrokerLoadMonitor()
        
        async def rapid_broker_switching(iteration: int):
            """Rapidly switch between brokers."""
            for _ in range(10):
                # Pick random broker
                broker_id = broker_ids[iteration % len(broker_ids)]
                
                op_start = time.time()
                try:
                    broker = registry.get_broker(broker_id)
                    balance = await broker.get_balance()
                    latency_ms = (time.time() - op_start) * 1000
                    
                    monitor.record_operation(
                        broker_id=broker_id,
                        operation='get_balance',
                        latency_ms=latency_ms,
                        success=True
                    )
                except Exception as e:
                    latency_ms = (time.time() - op_start) * 1000
                    monitor.record_operation(
                        broker_id=broker_id,
                        operation='get_balance',
                        latency_ms=latency_ms,
                        success=False,
                        error=str(e)
                    )
        
        print(f"\n=== Rapid Broker Switching Stress Test ===")
        print(f"Brokers: {len(broker_ids)}")
        print(f"Iterations: 100")
        
        # Run rapid switching
        start_time = time.time()
        await asyncio.gather(*[rapid_broker_switching(i) for i in range(100)])
        duration = time.time() - start_time
        
        stats = monitor.get_statistics()
        
        print(f"\n=== Stress Test Results ===")
        print(f"Duration: {duration:.2f}s")
        print(f"Total operations: {stats['total_operations']}")
        print(f"Throughput: {stats['throughput_ops_per_sec']:.1f} ops/sec")
        
        # Verify no degradation
        for broker_id, broker_stats in stats['brokers'].items():
            assert broker_stats['success_rate'] >= 95.0, \
                f"Broker {broker_id} failed under rapid switching"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "-m", "slow"])
