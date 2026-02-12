"""
Task 26.15: Broker Registry Adapter Instantiation Benchmark

Benchmarks the performance of BrokerRegistry.register_broker() and adapter
connection validation to validate <100ms target for adapter setup.

Requirements: 19.4, 19.8
"""

import pytest
import time
import statistics
import os
from typing import List, Dict, Any
from src.data.brokers.registry import BrokerRegistry


class TestBrokerRegistryBenchmark:
    """
    Benchmark tests for broker registry adapter instantiation performance.
    
    Target: <100ms for adapter instantiation and connection validation
    """
    
    @pytest.fixture
    def test_config_path(self, tmp_path):
        """Create a temporary broker configuration file for testing."""
        config_content = """
brokers:
  mock_mt5_1:
    type: mt5_mock
    account_id: "TEST_ACCOUNT_001"
    server: "MockServer"
    login: "12345678"
    initial_balance: 10000.0
    enabled: true
  
  mock_mt5_2:
    type: mt5_mock
    account_id: "TEST_ACCOUNT_002"
    server: "MockServer"
    login: "87654321"
    initial_balance: 5000.0
    enabled: true
  
  mock_mt5_3:
    type: mt5_mock
    account_id: "TEST_ACCOUNT_003"
    server: "MockServer"
    login: "11111111"
    initial_balance: 15000.0
    enabled: true
"""
        config_file = tmp_path / "brokers_test.yaml"
        config_file.write_text(config_content)
        return str(config_file)

    def _calculate_statistics(self, latencies: List[float]) -> Dict[str, float]:
        """Calculate statistical metrics for latency measurements."""
        return {
            'average': statistics.mean(latencies),
            'median': statistics.median(latencies),
            'stdev': statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
            'min': min(latencies),
            'max': max(latencies),
            'p95': statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else max(latencies),
            'p99': statistics.quantiles(latencies, n=100)[98] if len(latencies) >= 100 else max(latencies)
        }
    
    def _print_statistics(self, name: str, stats: Dict[str, float]):
        """Print formatted statistics."""
        print(f"\n{'='*70}")
        print(f"{name}")
        print(f"{'='*70}")
        print(f"Average:  {stats['average']:.6f}ms")
        print(f"Median:   {stats['median']:.6f}ms")
        print(f"Std Dev:  {stats['stdev']:.6f}ms")
        print(f"Min:      {stats['min']:.6f}ms")
        print(f"Max:      {stats['max']:.6f}ms")
        print(f"P95:      {stats['p95']:.6f}ms")
        print(f"P99:      {stats['p99']:.6f}ms")
        print(f"{'='*70}\n")
    
    def test_benchmark_adapter_instantiation(self, test_config_path):
        """
        Benchmark: Adapter instantiation time
        
        Measures the time to instantiate broker adapters without
        connection validation.
        
        Target: <100ms average latency
        """
        print("\n" + "="*70)
        print("BENCHMARK: Broker Adapter Instantiation")
        print("="*70)
        
        # Test configuration
        num_iterations = 1000
        num_warmup = 10
        
        print(f"Test Configuration:")
        print(f"  - Iterations: {num_iterations:,}")
        print(f"  - Warmup iterations: {num_warmup}")
        print(f"  - Adapter type: MockMT5Adapter")
        
        # Warmup phase
        print("\nWarming up...")
        for _ in range(num_warmup):
            registry = BrokerRegistry(test_config_path)
            registry.register_broker('mock_mt5_1')
        
        # Benchmark phase
        print("Running benchmark...")
        latencies = []
        
        for _ in range(num_iterations):
            # Create fresh registry for each iteration
            registry = BrokerRegistry(test_config_path)
            
            start_time = time.perf_counter()
            registry.register_broker('mock_mt5_1')
            latency_ms = (time.perf_counter() - start_time) * 1000
            latencies.append(latency_ms)
        
        # Calculate statistics
        stats = self._calculate_statistics(latencies)
        
        # Print results
        self._print_statistics("Adapter Instantiation Performance", stats)
        
        # Validate targets
        print("Target Validation:")
        print(f"  Target: <100.0ms average latency")
        print(f"  Actual: {stats['average']:.6f}ms")
        print(f"  Status: {'✅ PASSED' if stats['average'] < 100.0 else '❌ FAILED'}")
        print(f"  Margin: {((100.0 - stats['average']) / 100.0 * 100):.2f}% under target\n")
        
        # Assertions
        assert stats['average'] < 100.0, \
            f"Average instantiation latency {stats['average']:.6f}ms exceeds 100ms target"
        assert stats['median'] < 100.0, \
            f"Median instantiation latency {stats['median']:.6f}ms exceeds 100ms target"
        assert stats['p95'] < 200.0, \
            f"P95 instantiation latency {stats['p95']:.6f}ms exceeds 200ms acceptable limit"

    def test_benchmark_connection_validation(self, test_config_path):
        """
        Benchmark: Connection validation time
        
        Measures the time to validate adapter connections.
        
        Target: <100ms average latency
        """
        print("\n" + "="*70)
        print("BENCHMARK: Adapter Connection Validation")
        print("="*70)
        
        # Test configuration
        num_iterations = 1000
        num_warmup = 10
        
        print(f"Test Configuration:")
        print(f"  - Iterations: {num_iterations:,}")
        print(f"  - Warmup iterations: {num_warmup}")
        print(f"  - Adapter type: MockMT5Adapter")
        
        # Create registry once
        registry = BrokerRegistry(test_config_path)
        registry.register_broker('mock_mt5_1')
        broker = registry.get_broker('mock_mt5_1')
        
        # Warmup phase
        print("\nWarming up...")
        for _ in range(num_warmup):
            broker.validate_connection()
        
        # Benchmark phase
        print("Running benchmark...")
        latencies = []
        
        for _ in range(num_iterations):
            start_time = time.perf_counter()
            is_connected = broker.validate_connection()
            latency_ms = (time.perf_counter() - start_time) * 1000
            latencies.append(latency_ms)
            
            assert is_connected, "Connection validation should return True for mock adapter"
        
        # Calculate statistics
        stats = self._calculate_statistics(latencies)
        
        # Print results
        self._print_statistics("Connection Validation Performance", stats)
        
        # Validate targets
        print("Target Validation:")
        print(f"  Target: <100.0ms average latency")
        print(f"  Actual: {stats['average']:.6f}ms")
        print(f"  Status: {'✅ PASSED' if stats['average'] < 100.0 else '❌ FAILED'}")
        print(f"  Margin: {((100.0 - stats['average']) / 100.0 * 100):.2f}% under target\n")
        
        # Assertions
        assert stats['average'] < 100.0, \
            f"Average validation latency {stats['average']:.6f}ms exceeds 100ms target"
        assert stats['median'] < 100.0, \
            f"Median validation latency {stats['median']:.6f}ms exceeds 100ms target"
        assert stats['p95'] < 200.0, \
            f"P95 validation latency {stats['p95']:.6f}ms exceeds 200ms acceptable limit"

    def test_benchmark_register_broker_complete(self, test_config_path):
        """
        Benchmark: Complete register_broker() operation
        
        Measures the time for the complete register_broker() operation
        including instantiation and connection validation.
        
        Target: <100ms average latency
        """
        print("\n" + "="*70)
        print("BENCHMARK: Complete register_broker() Operation")
        print("="*70)
        
        # Test configuration
        num_iterations = 500
        num_warmup = 10
        
        print(f"Test Configuration:")
        print(f"  - Iterations: {num_iterations:,}")
        print(f"  - Warmup iterations: {num_warmup}")
        print(f"  - Operation: Full register_broker() with validation")
        
        # Warmup phase
        print("\nWarming up...")
        for _ in range(num_warmup):
            registry = BrokerRegistry(test_config_path)
            registry.register_broker('mock_mt5_1')
        
        # Benchmark phase
        print("Running benchmark...")
        latencies = []
        
        for _ in range(num_iterations):
            # Create fresh registry for each iteration
            registry = BrokerRegistry(test_config_path)
            
            start_time = time.perf_counter()
            registry.register_broker('mock_mt5_1')
            latency_ms = (time.perf_counter() - start_time) * 1000
            latencies.append(latency_ms)
            
            # Verify registration succeeded
            assert 'mock_mt5_1' in registry.list_registered_brokers()
            assert registry.get_connection_status('mock_mt5_1') is True
        
        # Calculate statistics
        stats = self._calculate_statistics(latencies)
        
        # Print results
        self._print_statistics("Complete register_broker() Performance", stats)
        
        # Validate targets
        print("Target Validation:")
        print(f"  Target: <100.0ms average latency")
        print(f"  Actual: {stats['average']:.6f}ms")
        print(f"  Status: {'✅ PASSED' if stats['average'] < 100.0 else '❌ FAILED'}")
        print(f"  Margin: {((100.0 - stats['average']) / 100.0 * 100):.2f}% under target\n")
        
        # Assertions
        assert stats['average'] < 100.0, \
            f"Average register_broker() latency {stats['average']:.6f}ms exceeds 100ms target (Requirement 19.4)"
        assert stats['median'] < 100.0, \
            f"Median register_broker() latency {stats['median']:.6f}ms exceeds 100ms target"
        assert stats['p95'] < 200.0, \
            f"P95 register_broker() latency {stats['p95']:.6f}ms exceeds 200ms acceptable limit"

    def test_benchmark_multiple_adapter_registration(self, test_config_path):
        """
        Benchmark: Multiple adapter registration
        
        Measures the time to register multiple broker adapters sequentially.
        
        Target: <100ms per adapter average
        """
        print("\n" + "="*70)
        print("BENCHMARK: Multiple Adapter Registration")
        print("="*70)
        
        # Test configuration
        num_iterations = 200
        num_warmup = 5
        broker_ids = ['mock_mt5_1', 'mock_mt5_2', 'mock_mt5_3']
        
        print(f"Test Configuration:")
        print(f"  - Iterations: {num_iterations:,}")
        print(f"  - Warmup iterations: {num_warmup}")
        print(f"  - Brokers per iteration: {len(broker_ids)}")
        print(f"  - Total registrations: {num_iterations * len(broker_ids):,}")
        
        # Warmup phase
        print("\nWarming up...")
        for _ in range(num_warmup):
            registry = BrokerRegistry(test_config_path)
            for broker_id in broker_ids:
                registry.register_broker(broker_id)
        
        # Benchmark phase
        print("Running benchmark...")
        all_latencies = []
        per_broker_latencies = {broker_id: [] for broker_id in broker_ids}
        
        for _ in range(num_iterations):
            # Create fresh registry for each iteration
            registry = BrokerRegistry(test_config_path)
            
            for broker_id in broker_ids:
                start_time = time.perf_counter()
                registry.register_broker(broker_id)
                latency_ms = (time.perf_counter() - start_time) * 1000
                
                all_latencies.append(latency_ms)
                per_broker_latencies[broker_id].append(latency_ms)
            
            # Verify all registered
            assert len(registry.list_registered_brokers()) == len(broker_ids)
        
        # Calculate statistics
        overall_stats = self._calculate_statistics(all_latencies)
        
        # Print results
        self._print_statistics("Overall Multiple Registration Performance", overall_stats)
        
        # Per-broker statistics
        print("Per-Broker Performance:")
        print("-" * 70)
        for broker_id in broker_ids:
            stats = self._calculate_statistics(per_broker_latencies[broker_id])
            print(f"{broker_id}: avg={stats['average']:.6f}ms, median={stats['median']:.6f}ms, p95={stats['p95']:.6f}ms")
        print()
        
        # Validate targets
        print("Target Validation:")
        print(f"  Target: <100.0ms average per adapter")
        print(f"  Actual: {overall_stats['average']:.6f}ms")
        print(f"  Status: {'✅ PASSED' if overall_stats['average'] < 100.0 else '❌ FAILED'}")
        print(f"  Margin: {((100.0 - overall_stats['average']) / 100.0 * 100):.2f}% under target\n")
        
        # Assertions
        assert overall_stats['average'] < 100.0, \
            f"Average registration latency {overall_stats['average']:.6f}ms exceeds 100ms target"
        assert overall_stats['median'] < 100.0, \
            f"Median registration latency {overall_stats['median']:.6f}ms exceeds 100ms target"
        assert overall_stats['p95'] < 200.0, \
            f"P95 registration latency {overall_stats['p95']:.6f}ms exceeds 200ms acceptable limit"

    def test_benchmark_registry_load_config(self, test_config_path):
        """
        Benchmark: Configuration loading time
        
        Measures the time to load and parse broker configuration from YAML.
        
        Target: <100ms average latency
        """
        print("\n" + "="*70)
        print("BENCHMARK: Configuration Loading")
        print("="*70)
        
        # Test configuration
        num_iterations = 1000
        num_warmup = 10
        
        print(f"Test Configuration:")
        print(f"  - Iterations: {num_iterations:,}")
        print(f"  - Warmup iterations: {num_warmup}")
        print(f"  - Config file: {test_config_path}")
        
        # Warmup phase
        print("\nWarming up...")
        for _ in range(num_warmup):
            registry = BrokerRegistry(test_config_path)
        
        # Benchmark phase
        print("Running benchmark...")
        latencies = []
        
        for _ in range(num_iterations):
            start_time = time.perf_counter()
            registry = BrokerRegistry(test_config_path)
            latency_ms = (time.perf_counter() - start_time) * 1000
            latencies.append(latency_ms)
            
            # Verify config loaded
            assert len(registry.list_brokers()) == 3
        
        # Calculate statistics
        stats = self._calculate_statistics(latencies)
        
        # Print results
        self._print_statistics("Configuration Loading Performance", stats)
        
        # Validate targets
        print("Target Validation:")
        print(f"  Target: <100.0ms average latency")
        print(f"  Actual: {stats['average']:.6f}ms")
        print(f"  Status: {'✅ PASSED' if stats['average'] < 100.0 else '❌ FAILED'}")
        print(f"  Margin: {((100.0 - stats['average']) / 100.0 * 100):.2f}% under target\n")
        
        # Assertions
        assert stats['average'] < 100.0, \
            f"Average config loading latency {stats['average']:.6f}ms exceeds 100ms target"
        assert stats['median'] < 100.0, \
            f"Median config loading latency {stats['median']:.6f}ms exceeds 100ms target"
        assert stats['p95'] < 200.0, \
            f"P95 config loading latency {stats['p95']:.6f}ms exceeds 200ms acceptable limit"

    def test_benchmark_get_broker_lazy_loading(self, test_config_path):
        """
        Benchmark: Lazy broker loading via get_broker()
        
        Measures the time for lazy-loading brokers when first accessed
        via get_broker() method.
        
        Target: <100ms average latency
        """
        print("\n" + "="*70)
        print("BENCHMARK: Lazy Broker Loading (get_broker)")
        print("="*70)
        
        # Test configuration
        num_iterations = 500
        num_warmup = 10
        
        print(f"Test Configuration:")
        print(f"  - Iterations: {num_iterations:,}")
        print(f"  - Warmup iterations: {num_warmup}")
        print(f"  - Operation: get_broker() with lazy loading")
        
        # Warmup phase
        print("\nWarming up...")
        for _ in range(num_warmup):
            registry = BrokerRegistry(test_config_path)
            broker = registry.get_broker('mock_mt5_1')
        
        # Benchmark phase
        print("Running benchmark...")
        latencies = []
        
        for _ in range(num_iterations):
            # Create fresh registry (no brokers registered yet)
            registry = BrokerRegistry(test_config_path)
            
            start_time = time.perf_counter()
            broker = registry.get_broker('mock_mt5_1')  # Triggers lazy loading
            latency_ms = (time.perf_counter() - start_time) * 1000
            latencies.append(latency_ms)
            
            # Verify broker loaded
            assert broker is not None
            assert 'mock_mt5_1' in registry.list_registered_brokers()
        
        # Calculate statistics
        stats = self._calculate_statistics(latencies)
        
        # Print results
        self._print_statistics("Lazy Broker Loading Performance", stats)
        
        # Validate targets
        print("Target Validation:")
        print(f"  Target: <100.0ms average latency")
        print(f"  Actual: {stats['average']:.6f}ms")
        print(f"  Status: {'✅ PASSED' if stats['average'] < 100.0 else '❌ FAILED'}")
        print(f"  Margin: {((100.0 - stats['average']) / 100.0 * 100):.2f}% under target\n")
        
        # Assertions
        assert stats['average'] < 100.0, \
            f"Average lazy loading latency {stats['average']:.6f}ms exceeds 100ms target"
        assert stats['median'] < 100.0, \
            f"Median lazy loading latency {stats['median']:.6f}ms exceeds 100ms target"
        assert stats['p95'] < 200.0, \
            f"P95 lazy loading latency {stats['p95']:.6f}ms exceeds 200ms acceptable limit"

    def test_benchmark_comprehensive_registry_summary(self, test_config_path):
        """
        Benchmark: Comprehensive broker registry performance summary
        
        This is the main benchmark test that validates all requirements
        for broker registry adapter instantiation and connection validation.
        
        **Validates: Requirements 19.4, 19.8**
        """
        print("\n" + "="*70)
        print("COMPREHENSIVE BROKER REGISTRY BENCHMARK")
        print("="*70)
        print("\nTask 26.15: Benchmark broker registry adapter instantiation <100ms")
        print("Requirement 19.4: Adapter instantiation based on broker type")
        print("Requirement 19.8: Validate credentials and cache connection status")
        print("="*70 + "\n")
        
        # Test configuration
        num_iterations = 1000
        num_warmup = 20
        broker_ids = ['mock_mt5_1', 'mock_mt5_2', 'mock_mt5_3']
        
        print("Test Configuration:")
        print(f"  - Iterations: {num_iterations:,}")
        print(f"  - Warmup iterations: {num_warmup}")
        print(f"  - Brokers: {len(broker_ids)}")
        print(f"  - Adapter type: MockMT5Adapter")
        
        # Phase 1: Warmup
        print("\nPhase 1: Warmup")
        for _ in range(num_warmup):
            registry = BrokerRegistry(test_config_path)
            for broker_id in broker_ids:
                registry.register_broker(broker_id)
        
        # Phase 2: Config Loading Benchmark
        print("Phase 2: Benchmarking configuration loading...")
        config_latencies = []
        
        for _ in range(num_iterations):
            start_time = time.perf_counter()
            registry = BrokerRegistry(test_config_path)
            latency_ms = (time.perf_counter() - start_time) * 1000
            config_latencies.append(latency_ms)
        
        config_stats = self._calculate_statistics(config_latencies)
        
        # Phase 3: Adapter Instantiation Benchmark
        print("Phase 3: Benchmarking adapter instantiation...")
        instantiation_latencies = []
        
        for _ in range(num_iterations):
            registry = BrokerRegistry(test_config_path)
            
            start_time = time.perf_counter()
            registry.register_broker('mock_mt5_1')
            latency_ms = (time.perf_counter() - start_time) * 1000
            instantiation_latencies.append(latency_ms)
        
        instantiation_stats = self._calculate_statistics(instantiation_latencies)
        
        # Phase 4: Connection Validation Benchmark
        print("Phase 4: Benchmarking connection validation...")
        validation_latencies = []
        
        registry = BrokerRegistry(test_config_path)
        registry.register_broker('mock_mt5_1')
        broker = registry.get_broker('mock_mt5_1')
        
        for _ in range(num_iterations):
            start_time = time.perf_counter()
            is_connected = broker.validate_connection()
            latency_ms = (time.perf_counter() - start_time) * 1000
            validation_latencies.append(latency_ms)
        
        validation_stats = self._calculate_statistics(validation_latencies)
        
        # Phase 5: Multiple Broker Registration
        print("Phase 5: Benchmarking multiple broker registration...")
        multi_latencies = []
        
        for _ in range(num_iterations // 2):  # Fewer iterations for multi-broker
            registry = BrokerRegistry(test_config_path)
            
            start_time = time.perf_counter()
            for broker_id in broker_ids:
                registry.register_broker(broker_id)
            latency_ms = (time.perf_counter() - start_time) * 1000
            multi_latencies.append(latency_ms)
        
        multi_stats = self._calculate_statistics(multi_latencies)
        
        # Phase 6: Lazy Loading Benchmark
        print("Phase 6: Benchmarking lazy loading...")
        lazy_latencies = []
        
        for _ in range(num_iterations):
            registry = BrokerRegistry(test_config_path)
            
            start_time = time.perf_counter()
            broker = registry.get_broker('mock_mt5_1')
            latency_ms = (time.perf_counter() - start_time) * 1000
            lazy_latencies.append(latency_ms)
        
        lazy_stats = self._calculate_statistics(lazy_latencies)
        
        # Print comprehensive results
        print("\n" + "="*70)
        print("BENCHMARK RESULTS SUMMARY")
        print("="*70)
        
        print("\n1. CONFIGURATION LOADING")
        print("-" * 70)
        print(f"Average:  {config_stats['average']:.6f}ms")
        print(f"Median:   {config_stats['median']:.6f}ms")
        print(f"P95:      {config_stats['p95']:.6f}ms")
        print(f"P99:      {config_stats['p99']:.6f}ms")
        print(f"Max:      {config_stats['max']:.6f}ms")
        
        print("\n2. ADAPTER INSTANTIATION (register_broker)")
        print("-" * 70)
        print(f"Average:  {instantiation_stats['average']:.6f}ms")
        print(f"Median:   {instantiation_stats['median']:.6f}ms")
        print(f"P95:      {instantiation_stats['p95']:.6f}ms")
        print(f"P99:      {instantiation_stats['p99']:.6f}ms")
        print(f"Max:      {instantiation_stats['max']:.6f}ms")
        
        print("\n3. CONNECTION VALIDATION")
        print("-" * 70)
        print(f"Average:  {validation_stats['average']:.6f}ms")
        print(f"Median:   {validation_stats['median']:.6f}ms")
        print(f"P95:      {validation_stats['p95']:.6f}ms")
        print(f"P99:      {validation_stats['p99']:.6f}ms")
        print(f"Max:      {validation_stats['max']:.6f}ms")
        
        print("\n4. MULTIPLE BROKER REGISTRATION (3 brokers)")
        print("-" * 70)
        print(f"Average:  {multi_stats['average']:.6f}ms")
        print(f"Median:   {multi_stats['median']:.6f}ms")
        print(f"P95:      {multi_stats['p95']:.6f}ms")
        print(f"P99:      {multi_stats['p99']:.6f}ms")
        print(f"Max:      {multi_stats['max']:.6f}ms")
        print(f"Per-broker avg: {multi_stats['average'] / len(broker_ids):.6f}ms")
        
        print("\n5. LAZY LOADING (get_broker)")
        print("-" * 70)
        print(f"Average:  {lazy_stats['average']:.6f}ms")
        print(f"Median:   {lazy_stats['median']:.6f}ms")
        print(f"P95:      {lazy_stats['p95']:.6f}ms")
        print(f"P99:      {lazy_stats['p99']:.6f}ms")
        print(f"Max:      {lazy_stats['max']:.6f}ms")
        
        print("\n6. PERFORMANCE BREAKDOWN")
        print("-" * 70)
        total_setup = config_stats['average'] + instantiation_stats['average']
        print(f"Config loading:        {config_stats['average']:.6f}ms")
        print(f"Adapter instantiation: {instantiation_stats['average']:.6f}ms")
        print(f"Connection validation: {validation_stats['average']:.6f}ms")
        print(f"Total setup time:      {total_setup:.6f}ms")
        
        print("\n7. TARGET VALIDATION")
        print("-" * 70)
        print(f"Requirement 19.4: Adapter instantiation <100ms")
        print(f"  Target: <100.0ms average")
        print(f"  Actual: {instantiation_stats['average']:.6f}ms")
        print(f"  Status: {'✅ PASSED' if instantiation_stats['average'] < 100.0 else '❌ FAILED'}")
        print(f"  Margin: {((100.0 - instantiation_stats['average']) / 100.0 * 100):.2f}% under target")
        
        print(f"\nRequirement 19.8: Connection validation <100ms")
        print(f"  Target: <100.0ms average")
        print(f"  Actual: {validation_stats['average']:.6f}ms")
        print(f"  Status: {'✅ PASSED' if validation_stats['average'] < 100.0 else '❌ FAILED'}")
        print(f"  Margin: {((100.0 - validation_stats['average']) / 100.0 * 100):.2f}% under target")
        
        print(f"\nMultiple broker registration: <100ms per broker")
        print(f"  Target: <100.0ms per broker")
        print(f"  Actual: {multi_stats['average'] / len(broker_ids):.6f}ms per broker")
        print(f"  Status: {'✅ PASSED' if (multi_stats['average'] / len(broker_ids)) < 100.0 else '❌ FAILED'}")
        
        print("\n8. PERFORMANCE CHARACTERISTICS")
        print("-" * 70)
        print(f"Iterations tested: {num_iterations:,}")
        print(f"Brokers tested: {len(broker_ids)}")
        print(f"Adapter type: MockMT5Adapter")
        print(f"Consistency (instantiation): {instantiation_stats['stdev']:.6f}ms std dev")
        print(f"Consistency (validation): {validation_stats['stdev']:.6f}ms std dev")
        print(f"Consistency (lazy loading): {lazy_stats['stdev']:.6f}ms std dev")
        
        print("\n" + "="*70)
        print("BENCHMARK COMPLETE")
        print("="*70 + "\n")
        
        # Final assertions
        assert config_stats['average'] < 100.0, \
            f"Config loading average latency {config_stats['average']:.6f}ms exceeds 100ms target"
        
        assert instantiation_stats['average'] < 100.0, \
            f"Adapter instantiation average latency {instantiation_stats['average']:.6f}ms exceeds 100ms target (Requirement 19.4)"
        assert instantiation_stats['median'] < 100.0, \
            f"Adapter instantiation median latency {instantiation_stats['median']:.6f}ms exceeds 100ms target"
        
        assert validation_stats['average'] < 100.0, \
            f"Connection validation average latency {validation_stats['average']:.6f}ms exceeds 100ms target (Requirement 19.8)"
        assert validation_stats['median'] < 100.0, \
            f"Connection validation median latency {validation_stats['median']:.6f}ms exceeds 100ms target"
        
        assert lazy_stats['average'] < 100.0, \
            f"Lazy loading average latency {lazy_stats['average']:.6f}ms exceeds 100ms target"
        
        # Per-broker average for multiple registration
        per_broker_avg = multi_stats['average'] / len(broker_ids)
        assert per_broker_avg < 100.0, \
            f"Multiple broker registration per-broker average {per_broker_avg:.6f}ms exceeds 100ms target"
        
        # P95 should be reasonable (allow some outliers)
        assert instantiation_stats['p95'] < 200.0, \
            f"Adapter instantiation P95 latency {instantiation_stats['p95']:.6f}ms exceeds 200ms acceptable limit"
        assert validation_stats['p95'] < 200.0, \
            f"Connection validation P95 latency {validation_stats['p95']:.6f}ms exceeds 200ms acceptable limit"
        
        # Success message
        print(f"✅ BENCHMARK PASSED: Broker registry adapter instantiation <100ms target achieved")
        print(f"✅ Config loading: {config_stats['average']:.6f}ms")
        print(f"✅ Adapter instantiation: {instantiation_stats['average']:.6f}ms")
        print(f"✅ Connection validation: {validation_stats['average']:.6f}ms")
        print(f"✅ Lazy loading: {lazy_stats['average']:.6f}ms")
        print(f"✅ Task 26.15 requirements validated\n")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
